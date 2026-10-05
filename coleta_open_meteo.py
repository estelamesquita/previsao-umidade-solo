#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de coleta meteorologica (Open-Meteo) com calculo de ETo.

Coleta temperatura, umidade relativa, velocidade do vento e precipitacao na
resolucao horaria, agrega para a escala diaria e calcula a evapotranspiracao
de referencia (ETo) por dois metodos:

  * Hargreaves-Samani (1985)  -> "ETo simplificada": exige apenas Tmax, Tmin
    e a radiacao extraterrestre Ra (calculada a partir da latitude e do dia
    do ano). E o metodo recomendado pela FAO-56 quando faltam dados.
  * Penman-Monteith FAO-56    -> metodo padrao, usado aqui como referencia de
    validacao quando a radiacao de ondas curtas vem na resposta da API.

A API tambem devolve o campo `et0_fao_evapotranspiration`, que o script
mantem como terceira coluna de controle de qualidade.

Uso basico
----------
    python coleta_open_meteo.py --lat -23.5505 --lon -46.6333 --dias 30

    # janela historica explicita (usa a Archive API, dados ERA5)
    python coleta_open_meteo.py --lat -12.97 --lon -38.50 \
        --inicio 2024-01-01 --fim 2024-12-31 --fonte archive \
        --saida ./dados --formato parquet

Dependencias
------------
    pip install requests pandas          # numpy vem junto com pandas
    pip install pyarrow                  # opcional, para --formato parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 >= 2 e < 2 expoem Retry em caminhos diferentes
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

LOG = logging.getLogger("open_meteo")

# ---------------------------------------------------------------------------
# Constantes fisicas (FAO-56, Allen et al., 1998)
# ---------------------------------------------------------------------------

GSC = 0.0820          # constante solar [MJ m-2 min-1]
LATENT_HEAT = 2.45    # calor latente de vaporizacao [MJ kg-1] ~ mm
ALBEDO = 0.23         # albedo da cultura de referencia (grama)
STEFAN_BOLTZMANN = 4.903e-09  # [MJ K-4 m-2 dia-1]
KRS = 0.16            # coef. de radiacao Hargreaves (0.16 interior, 0.19 costa)
HARGREAVES_C = 0.0023

# Variaveis horarias pedidas a API
HOURLY_VARS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
)

# Variaveis diarias pedidas a API (usadas para conferencia)
DAILY_VARS: tuple[str, ...] = (
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
)

ENDPOINTS = {
    "forecast": "https://api.open-meteo.com/v1/forecast",
    "archive": "https://archive-api.open-meteo.com/v1/archive",
}


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Local:
    """Ponto de interesse e seus metadados."""

    latitude: float
    longitude: float
    altitude_m: float = 0.0
    timezone: str = "auto"
    nome: str = "local"

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude fora do intervalo [-90, 90]: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude fora do intervalo [-180, 180]: {self.longitude}")

    @property
    def costeiro(self) -> bool:
        """Heuristica simples para escolher o krs de Hargreaves."""
        return False  # ajuste para True se o ponto estiver na faixa litoranea


@dataclass
class JanelaTemporal:
    inicio: date
    fim: date

    def __post_init__(self) -> None:
        if self.inicio > self.fim:
            raise ValueError(f"inicio ({self.inicio}) posterior ao fim ({self.fim})")

    @property
    def dias(self) -> int:
        return (self.fim - self.inicio).days + 1

    def chunks(self, max_dias: int) -> Iterator["JanelaTemporal"]:
        """Fatia a janela em blocos — evita respostas gigantes e timeouts.

        E o ponto de entrada para paralelizar a coleta (um bloco por worker)
        quando a serie passa de alguns anos.
        """
        if max_dias <= 0:
            raise ValueError("max_dias deve ser positivo")
        cursor = self.inicio
        while cursor <= self.fim:
            fim_bloco = min(cursor + timedelta(days=max_dias - 1), self.fim)
            yield JanelaTemporal(cursor, fim_bloco)
            cursor = fim_bloco + timedelta(days=1)


# ---------------------------------------------------------------------------
# Cliente HTTP
# ---------------------------------------------------------------------------


class OpenMeteoError(RuntimeError):
    """Falha de comunicacao ou payload invalido da API."""


@dataclass
class OpenMeteoClient:
    """Wrapper fino sobre `requests` com retry/backoff e sessao reaproveitada.

    A API gratuita nao exige chave, mas aplica rate limit por IP; o retry em
    429/5xx com backoff exponencial resolve a maior parte dos esbarroes.
    """

    fonte: str = "forecast"
    timeout: float = 30.0
    tentativas: int = 5
    backoff: float = 0.8
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fonte not in ENDPOINTS:
            raise ValueError(f"fonte invalida: {self.fonte!r}; use {set(ENDPOINTS)}")
        retry = Retry(
            total=self.tentativas,
            backoff_factor=self.backoff,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "pipeline-eto/1.0 (+requests)"})
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # -- infra ------------------------------------------------------------

    def __enter__(self) -> "OpenMeteoClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    @property
    def url(self) -> str:
        return ENDPOINTS[self.fonte]

    def _get(self, params: dict[str, object]) -> dict:
        LOG.debug("GET %s %s", self.url, params)
        try:
            resposta = self.session.get(self.url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OpenMeteoError(f"falha de rede ao consultar {self.url}: {exc}") from exc

        if resposta.status_code != 200:
            # A API devolve {"error": true, "reason": "..."} com detalhe util.
            detalhe = ""
            try:
                detalhe = resposta.json().get("reason", "")
            except ValueError:
                detalhe = resposta.text[:200]
            raise OpenMeteoError(f"HTTP {resposta.status_code} em {self.url}: {detalhe}")

        payload = resposta.json()
        if payload.get("error"):
            raise OpenMeteoError(f"API retornou erro: {payload.get('reason')}")
        return payload

    # -- consultas --------------------------------------------------------

    def _params_base(self, local: Local, janela: JanelaTemporal) -> dict[str, object]:
        return {
            "latitude": local.latitude,
            "longitude": local.longitude,
            "start_date": janela.inicio.isoformat(),
            "end_date": janela.fim.isoformat(),
            "timezone": local.timezone,
            "timeformat": "iso8601",
            # Unidades explicitas: m/s e mm, para nao depender do default.
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }

    def hourly(
        self,
        local: Local,
        janela: JanelaTemporal,
        variaveis: Sequence[str] = HOURLY_VARS,
        max_dias_por_chunk: int = 90,
    ) -> pd.DataFrame:
        """Serie horaria concatenada, com `time` como indice tz-aware."""
        blocos: list[pd.DataFrame] = []
        for sub in janela.chunks(max_dias_por_chunk):
            params = self._params_base(local, sub) | {"hourly": ",".join(variaveis)}
            payload = self._get(params)
            blocos.append(_bloco_para_frame(payload, "hourly", local))
        return _concat_series(blocos, "time")

    def daily(
        self,
        local: Local,
        janela: JanelaTemporal,
        variaveis: Sequence[str] = DAILY_VARS,
        max_dias_por_chunk: int = 365,
    ) -> pd.DataFrame:
        """Agregados diarios calculados pelo proprio Open-Meteo."""
        blocos: list[pd.DataFrame] = []
        for sub in janela.chunks(max_dias_por_chunk):
            params = self._params_base(local, sub) | {"daily": ",".join(variaveis)}
            payload = self._get(params)
            blocos.append(_bloco_para_frame(payload, "daily", local))
        return _concat_series(blocos, "time")


def _bloco_para_frame(payload: dict, secao: str, local: Local) -> pd.DataFrame:
    """Converte o dict colunar da API em DataFrame tipado."""
    dados = payload.get(secao)
    if not dados or "time" not in dados:
        raise OpenMeteoError(f"secao {secao!r} ausente ou sem eixo temporal na resposta")

    tz = payload.get("timezone", "UTC")
    df = pd.DataFrame(dados)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(
        tz, ambiguous="NaT", nonexistent="shift_forward"
    )

    numericas = [c for c in df.columns if c != "time"]
    df[numericas] = df[numericas].apply(pd.to_numeric, errors="coerce")

    # Metadados vindos da propria API (a elevacao e usada no Penman-Monteith).
    df.attrs["timezone"] = tz
    df.attrs["elevation"] = float(payload.get("elevation", local.altitude_m))
    df.attrs["latitude"] = float(payload.get("latitude", local.latitude))
    df.attrs["longitude"] = float(payload.get("longitude", local.longitude))
    return df


def _concat_series(blocos: list[pd.DataFrame], coluna_tempo: str) -> pd.DataFrame:
    if not blocos:
        return pd.DataFrame()
    attrs = blocos[0].attrs
    df = (
        pd.concat(blocos, ignore_index=True)
        .drop_duplicates(subset=coluna_tempo, keep="last")  # bordas dos chunks
        .sort_values(coluna_tempo)
        .set_index(coluna_tempo)
    )
    df.attrs = attrs
    return df


# ---------------------------------------------------------------------------
# Agregacao horaria -> diaria (vetorizada)
# ---------------------------------------------------------------------------


def agregar_para_diario(horario: pd.DataFrame, cobertura_minima: float = 0.75) -> pd.DataFrame:
    """Resume a serie horaria nos descritores exigidos pelos modelos de ETo.

    `cobertura_minima` descarta dias com poucas horas validas (tipicamente o
    primeiro e o ultimo dia da janela, truncados pelo fuso).
    """
    if horario.empty:
        return pd.DataFrame()

    agrupado = horario.resample("1D")
    diario = pd.DataFrame(
        {
            "t_max": agrupado["temperature_2m"].max(),
            "t_min": agrupado["temperature_2m"].min(),
            "t_mean": agrupado["temperature_2m"].mean(),
            "rh_max": agrupado["relative_humidity_2m"].max(),
            "rh_min": agrupado["relative_humidity_2m"].min(),
            "rh_mean": agrupado["relative_humidity_2m"].mean(),
            "vento_10m_mean": agrupado["wind_speed_10m"].mean(),
            "vento_10m_max": agrupado["wind_speed_10m"].max(),
            "precipitacao_mm": agrupado["precipitation"].sum(min_count=1),
            "horas_validas": agrupado["temperature_2m"].count(),
        }
    )

    if "shortwave_radiation" in horario:
        # W m-2 medio -> MJ m-2 dia-1 : media * 86400 s / 1e6
        diario["rad_solar_mj"] = agrupado["shortwave_radiation"].mean() * 0.0864
    if "et0_fao_evapotranspiration" in horario:
        diario["eto_api_mm"] = agrupado["et0_fao_evapotranspiration"].sum(min_count=1)

    diario.index.name = "data"
    diario.attrs = horario.attrs

    completos = diario["horas_validas"] >= 24 * cobertura_minima
    descartados = int((~completos).sum())
    if descartados:
        LOG.info("descartando %d dia(s) com cobertura horaria insuficiente", descartados)
    resultado = diario.loc[completos].copy()
    resultado.attrs = diario.attrs
    return resultado


# ---------------------------------------------------------------------------
# Fisica / radiacao
# ---------------------------------------------------------------------------


def radiacao_extraterrestre(dia_do_ano: np.ndarray, latitude_graus: float) -> np.ndarray:
    """Ra em MJ m-2 dia-1 (FAO-56, eq. 21). Vetorizada sobre o dia do ano."""
    j = np.asarray(dia_do_ano, dtype=float)
    phi = np.radians(latitude_graus)

    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)          # eq. 23
    delta = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)      # eq. 24

    # Sol permanente / noite polar: limita o argumento do arco-cosseno.
    arg = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    omega_s = np.arccos(arg)                                     # eq. 25

    return (
        (24.0 * 60.0 / np.pi)
        * GSC
        * dr
        * (
            omega_s * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
        )
    )


def pressao_vapor_saturacao(t_celsius: np.ndarray) -> np.ndarray:
    """es(T) em kPa (FAO-56, eq. 11)."""
    t = np.asarray(t_celsius, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def declividade_curva_vapor(t_celsius: np.ndarray) -> np.ndarray:
    """Delta em kPa C-1 (FAO-56, eq. 13)."""
    t = np.asarray(t_celsius, dtype=float)
    return 4098.0 * pressao_vapor_saturacao(t) / np.power(t + 237.3, 2)


def constante_psicrometrica(altitude_m: float) -> tuple[float, float]:
    """Retorna (pressao atmosferica kPa, gamma kPa C-1) — eqs. 7 e 8."""
    p = 101.3 * ((293.0 - 0.0065 * altitude_m) / 293.0) ** 5.26
    return p, 0.000665 * p


def vento_10m_para_2m(u10: np.ndarray, altura_origem_m: float = 10.0) -> np.ndarray:
    """Perfil logaritmico do vento (FAO-56, eq. 47)."""
    fator = 4.87 / np.log(67.8 * altura_origem_m - 5.42)
    return np.asarray(u10, dtype=float) * fator


# ---------------------------------------------------------------------------
# Modelos de ETo
# ---------------------------------------------------------------------------


def eto_hargreaves_samani(
    t_max: np.ndarray,
    t_min: np.ndarray,
    t_mean: np.ndarray,
    dia_do_ano: np.ndarray,
    latitude_graus: float,
) -> np.ndarray:
    """ETo simplificada em mm dia-1 (Hargreaves & Samani, 1985; FAO-56 eq. 52).

        ETo = 0.0023 * (Tmean + 17.8) * (Tmax - Tmin)^0.5 * Ra_mm

    Precisa apenas de temperatura e latitude — e a escolha correta quando nao
    ha radiacao, umidade ou vento confiaveis. A amplitude termica entra como
    proxy da radiacao solar incidente, portanto `t_max - t_min` negativo (dado
    corrompido) e tratado como ausente.
    """
    t_max = np.asarray(t_max, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    t_mean = np.asarray(t_mean, dtype=float)

    amplitude = t_max - t_min
    amplitude = np.where(amplitude > 0, amplitude, np.nan)

    ra_mm = radiacao_extraterrestre(dia_do_ano, latitude_graus) / LATENT_HEAT
    eto = HARGREAVES_C * (t_mean + 17.8) * np.sqrt(amplitude) * ra_mm
    return np.clip(eto, 0.0, None)


def eto_penman_monteith(
    t_max: np.ndarray,
    t_min: np.ndarray,
    t_mean: np.ndarray,
    rh_max: np.ndarray,
    rh_min: np.ndarray,
    u2: np.ndarray,
    dia_do_ano: np.ndarray,
    latitude_graus: float,
    altitude_m: float,
    rad_solar_mj: np.ndarray | None = None,
    krs: float = KRS,
) -> np.ndarray:
    """ETo FAO-56 Penman-Monteith em mm dia-1 (eq. 6).

    Usa a radiacao medida quando disponivel; caso contrario a estima por
    Hargreaves (Rs = krs * sqrt(Tmax - Tmin) * Ra).
    """
    t_max = np.asarray(t_max, dtype=float)
    t_min = np.asarray(t_min, dtype=float)
    t_mean = np.asarray(t_mean, dtype=float)
    u2 = np.asarray(u2, dtype=float)

    _, gamma = constante_psicrometrica(altitude_m)
    delta = declividade_curva_vapor(t_mean)

    # Deficit de pressao de vapor a partir dos extremos de UR (eqs. 11, 17).
    es = 0.5 * (pressao_vapor_saturacao(t_max) + pressao_vapor_saturacao(t_min))
    ea = 0.5 * (
        pressao_vapor_saturacao(t_min) * np.asarray(rh_max, dtype=float) / 100.0
        + pressao_vapor_saturacao(t_max) * np.asarray(rh_min, dtype=float) / 100.0
    )
    vpd = np.clip(es - ea, 0.0, None)

    ra = radiacao_extraterrestre(dia_do_ano, latitude_graus)
    if rad_solar_mj is None:
        amplitude = np.clip(t_max - t_min, 0.0, None)
        rs = krs * np.sqrt(amplitude) * ra
    else:
        rs = np.asarray(rad_solar_mj, dtype=float)

    rso = (0.75 + 2e-05 * altitude_m) * ra              # ceu limpo, eq. 37
    with np.errstate(divide="ignore", invalid="ignore"):
        razao = np.clip(np.where(rso > 0, rs / rso, np.nan), 0.33, 1.0)

    rns = (1.0 - ALBEDO) * rs                            # eq. 38
    tk_max = t_max + 273.16
    tk_min = t_min + 273.16
    rnl = (                                              # eq. 39
        STEFAN_BOLTZMANN
        * 0.5
        * (np.power(tk_max, 4) + np.power(tk_min, 4))
        * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0.0, None)))
        * (1.35 * razao - 0.35)
    )
    rn = rns - rnl
    g = 0.0  # fluxo de calor no solo desprezivel na escala diaria

    numerador = 0.408 * delta * (rn - g) + gamma * (900.0 / (t_mean + 273.0)) * u2 * vpd
    denominador = delta + gamma * (1.0 + 0.34 * u2)
    return np.clip(numerador / denominador, 0.0, None)


def calcular_eto(diario: pd.DataFrame, local: Local, metodo: str = "ambos") -> pd.DataFrame:
    """Adiciona as colunas de ETo ao DataFrame diario."""
    if diario.empty:
        return diario

    df = diario.copy()
    df.attrs = diario.attrs

    latitude = float(df.attrs.get("latitude", local.latitude))
    altitude = float(df.attrs.get("elevation", local.altitude_m))
    doy = df.index.dayofyear.to_numpy()

    df["ra_mj"] = radiacao_extraterrestre(doy, latitude)

    if metodo in ("hargreaves", "ambos"):
        df["eto_hargreaves_mm"] = eto_hargreaves_samani(
            df["t_max"], df["t_min"], df["t_mean"], doy, latitude
        )

    if metodo in ("penman", "ambos"):
        u2 = vento_10m_para_2m(df["vento_10m_mean"].to_numpy())
        df["vento_2m_mean"] = u2
        df["eto_penman_mm"] = eto_penman_monteith(
            t_max=df["t_max"],
            t_min=df["t_min"],
            t_mean=df["t_mean"],
            rh_max=df["rh_max"],
            rh_min=df["rh_min"],
            u2=u2,
            dia_do_ano=doy,
            latitude_graus=latitude,
            altitude_m=altitude,
            rad_solar_mj=df["rad_solar_mj"].to_numpy() if "rad_solar_mj" in df else None,
            krs=0.19 if local.costeiro else KRS,
        )

    # Balanco hidrico simples: quanto a chuva cobriu da demanda atmosferica.
    referencia = (
        "eto_penman_mm" if "eto_penman_mm" in df else "eto_hargreaves_mm"
    )
    if referencia in df:
        df["balanco_hidrico_mm"] = df["precipitacao_mm"].fillna(0.0) - df[referencia]

    return df


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def salvar(df: pd.DataFrame, destino: Path, formato: str, nome: str) -> Path:
    """Grava o DataFrame em CSV ou Parquet (particionado por ano-mes)."""
    destino.mkdir(parents=True, exist_ok=True)

    if formato == "csv":
        caminho = destino / f"{nome}.csv"
        df.to_csv(caminho, index=True, float_format="%.4f")
        return caminho

    if formato == "parquet":
        caminho = destino / nome
        saida = df.copy()
        if isinstance(saida.index, pd.DatetimeIndex):
            # Particionar reduz drasticamente o I/O em consultas por periodo.
            saida["ano"] = saida.index.year
            saida["mes"] = saida.index.month
            if saida.index.tz is not None:
                saida.index = saida.index.tz_localize(None)
        try:
            saida.to_parquet(caminho, partition_cols=["ano", "mes"], index=True)
        except ImportError as exc:
            raise SystemExit(
                "formato parquet requer pyarrow: pip install pyarrow"
            ) from exc
        return caminho

    raise ValueError(f"formato nao suportado: {formato!r}")


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Estatisticas descritivas das colunas de interesse."""
    colunas = [
        c
        for c in (
            "t_mean",
            "t_max",
            "t_min",
            "rh_mean",
            "vento_10m_mean",
            "precipitacao_mm",
            "eto_hargreaves_mm",
            "eto_penman_mm",
            "eto_api_mm",
        )
        if c in df
    ]
    return df[colunas].describe().T[["count", "mean", "std", "min", "max"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Coleta Open-Meteo e calcula ETo (Hargreaves e Penman-Monteith).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lat",
        type=float,
        default=float(os.getenv("METEO_LAT", "-15.7939")),
        help="latitude em graus decimais (sul negativo)",
    )
    p.add_argument(
        "--lon",
        type=float,
        default=float(os.getenv("METEO_LON", "-47.8828")),
        help="longitude em graus decimais (oeste negativo)",
    )
    p.add_argument("--altitude", type=float, default=None,
                   help="altitude em m; se omitida usa a elevacao devolvida pela API")
    p.add_argument("--nome", default="minha_regiao", help="rotulo do ponto nos arquivos")
    p.add_argument("--timezone", default="auto", help="fuso IANA ou 'auto'")

    p.add_argument("--dias", type=int, default=30,
                   help="janela retroativa em dias (ignorada se --inicio for usado)")
    p.add_argument("--inicio", type=_data, default=None, help="data inicial YYYY-MM-DD")
    p.add_argument("--fim", type=_data, default=None, help="data final YYYY-MM-DD")

    p.add_argument("--fonte", choices=sorted(ENDPOINTS), default="forecast",
                   help="forecast (ultimos ~3 meses + previsao) ou archive (ERA5)")
    p.add_argument("--eto", choices=("hargreaves", "penman", "ambos"), default="ambos",
                   help="metodo de calculo da ETo")
    p.add_argument("--saida", type=Path, default=Path("./dados_meteo"),
                   help="diretorio de saida")
    p.add_argument("--formato", choices=("csv", "parquet"), default="csv")
    p.add_argument("--salvar-horario", action="store_true",
                   help="grava tambem a serie horaria bruta")
    p.add_argument("--chunk-dias", type=int, default=90,
                   help="tamanho do bloco de requisicao para a serie horaria")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _data(valor: str) -> date:
    try:
        return datetime.strptime(valor, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"data invalida: {valor!r} (use YYYY-MM-DD)") from exc


def resolver_janela(args: argparse.Namespace) -> JanelaTemporal:
    hoje = date.today()
    if args.inicio or args.fim:
        inicio = args.inicio or (args.fim - timedelta(days=args.dias - 1))
        fim = args.fim or min(inicio + timedelta(days=args.dias - 1), hoje)
        return JanelaTemporal(inicio, fim)
    # A serie horaria do dia corrente ainda esta incompleta: encerra em D-1.
    fim = hoje - timedelta(days=1)
    return JanelaTemporal(fim - timedelta(days=args.dias - 1), fim)


def main(argv: Sequence[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        local = Local(
            latitude=args.lat,
            longitude=args.lon,
            altitude_m=args.altitude or 0.0,
            timezone=args.timezone,
            nome=args.nome,
        )
        janela = resolver_janela(args)
    except ValueError as exc:
        LOG.error("configuracao invalida: %s", exc)
        return 2

    LOG.info(
        "coletando %s (%.4f, %.4f) de %s a %s via %s",
        local.nome, local.latitude, local.longitude, janela.inicio, janela.fim, args.fonte,
    )

    try:
        with OpenMeteoClient(fonte=args.fonte) as cliente:
            horario = cliente.hourly(local, janela, max_dias_por_chunk=args.chunk_dias)
    except OpenMeteoError as exc:
        LOG.error("coleta interrompida: %s", exc)
        return 1

    if horario.empty:
        LOG.error("a API nao devolveu registros para a janela solicitada")
        return 1

    LOG.info("%d registros horarios | elevacao %.0f m | fuso %s",
             len(horario), horario.attrs["elevation"], horario.attrs["timezone"])

    diario = calcular_eto(agregar_para_diario(horario), local, metodo=args.eto)
    if diario.empty:
        LOG.error("nenhum dia com cobertura horaria suficiente")
        return 1

    caminho = salvar(diario, args.saida, args.formato, f"{local.nome}_diario")
    LOG.info("serie diaria (%d dias) -> %s", len(diario), caminho)

    if args.salvar_horario:
        caminho_h = salvar(horario, args.saida, args.formato, f"{local.nome}_horario")
        LOG.info("serie horaria -> %s", caminho_h)

    pd.set_option("display.width", 140, "display.max_columns", 30)
    print("\n=== Estatisticas do periodo ===")
    print(resumo(diario).round(2).to_string())

    print("\n=== Ultimos 7 dias ===")
    colunas = [
        c
        for c in (
            "t_min", "t_max", "rh_mean", "vento_10m_mean", "precipitacao_mm",
            "eto_hargreaves_mm", "eto_penman_mm", "eto_api_mm", "balanco_hidrico_mm",
        )
        if c in diario
    ]
    print(diario[colunas].tail(7).round(2).to_string())

    if {"eto_hargreaves_mm", "eto_penman_mm"} <= set(diario.columns):
        desvio = (diario["eto_hargreaves_mm"] - diario["eto_penman_mm"]).abs()
        print(
            f"\nDesvio medio Hargreaves vs Penman-Monteith: {desvio.mean():.2f} mm/dia "
            f"(maximo {desvio.max():.2f})"
        )
    print(
        f"Precipitacao acumulada: {diario['precipitacao_mm'].sum():.1f} mm | "
        f"ETo acumulada: "
        f"{diario.get('eto_penman_mm', diario.get('eto_hargreaves_mm')).sum():.1f} mm"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
