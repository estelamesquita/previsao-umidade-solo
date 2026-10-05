#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treinamento de modelo de previsao de umidade do solo / deficit hidrico (+12h, +24h).

Reutiliza o coletor e as funcoes de ETo de `coleta_open_meteo.py`, monta um
dataset supervisionado horario e treina regressores de arvore (Random Forest,
HistGradientBoosting e XGBoost quando disponivel) para prever a agua
armazenada na zona de raizes nas proximas 12 e 24 horas.

Alvo
----
`agua_zona_raiz_mm` = soma, nas camadas dentro da profundidade escolhida, de
(umidade volumetrica [m3/m3] x espessura da camada [mm]). Fica na mesma unidade
da ETo e da precipitacao, entao o deficit hidrico sai direto:

    deficit_hidrico_mm = capacidade_de_campo_mm - agua_zona_raiz_mm

A capacidade de campo e o ponto de murcha sao estimados por percentis da
serie de treino (p95 / p02) e ficam gravados no bundle.

Rigor temporal
--------------
Series meteorologicas violam a hipotese i.i.d., portanto:
  * nenhuma janela movel olha para o futuro (apenas t-k ... t);
  * o alvo e deslocado para t+h e a validacao usa TimeSeriesSplit com
    `gap=h` para que nenhuma amostra de treino compartilhe horizonte com a
    de validacao;
  * o holdout final e o ultimo trecho cronologico, nunca uma amostra
    aleatoria;
  * as metricas sao comparadas contra duas baselines (persistencia e
    sazonal-diaria) via skill score. Umidade do solo e fortemente
    autocorrelacionada: um modelo que nao bate a persistencia nao serve.

Uso
---
    # treino com 3 anos de ERA5 (recomendado para aprender a dinamica)
    python treinar_modelo_solo.py --lat -23.5505 --lon -46.6333 --anos 3

    # modelo operacional: mesmo esquema de camadas do endpoint de previsao
    python treinar_modelo_solo.py --lat -23.5505 --lon -46.6333 --fonte forecast

    # inferencia com um bundle salvo
    python treinar_modelo_solo.py --prever modelos/solo_sao_paulo_h24.joblib

Dependencias
------------
    pip install requests pandas scikit-learn joblib
    pip install xgboost      # opcional
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from coleta_open_meteo import (
    JanelaTemporal,
    Local,
    OpenMeteoClient,
    OpenMeteoError,
    pressao_vapor_saturacao,
)

try:
    from xgboost import XGBRegressor

    XGBOOST_DISPONIVEL = True
except ImportError:  # pragma: no cover
    XGBOOST_DISPONIVEL = False

LOG = logging.getLogger("modelo_solo")

# ---------------------------------------------------------------------------
# Esquema de camadas de solo — DIFERENTE por endpoint
# ---------------------------------------------------------------------------
# A Archive API (ERA5) e a Forecast API (ICON) discretizam o perfil de solo de
# formas distintas. Um modelo treinado em um esquema NAO pode ser aplicado ao
# outro: o bundle registra qual foi usado e a inferencia valida.
# (variavel, espessura_cm, profundidade_base_cm)
CAMADAS_SOLO: dict[str, tuple[tuple[str, float, float], ...]] = {
    "archive": (
        ("soil_moisture_0_to_7cm", 7.0, 7.0),
        ("soil_moisture_7_to_28cm", 21.0, 28.0),
        ("soil_moisture_28_to_100cm", 72.0, 100.0),
        ("soil_moisture_100_to_255cm", 155.0, 255.0),
    ),
    "forecast": (
        ("soil_moisture_0_to_1cm", 1.0, 1.0),
        ("soil_moisture_1_to_3cm", 2.0, 3.0),
        ("soil_moisture_3_to_9cm", 6.0, 9.0),
        ("soil_moisture_9_to_27cm", 18.0, 27.0),
        ("soil_moisture_27_to_81cm", 54.0, 81.0),
    ),
}

VARS_METEO: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
)

PERCENTIL_CAPACIDADE_CAMPO = 95.0
PERCENTIL_PONTO_MURCHA = 2.0


# ---------------------------------------------------------------------------
# Configuracao de features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigFeatures:
    """Define o espaco de features. Vai inteira para o bundle."""

    profundidade_cm: float = 28.0
    # janelas de acumulo (horas) para ETo, chuva e balanco hidrico
    janelas_acumulo: tuple[int, ...] = (3, 6, 12, 24, 72, 168)
    # lags e tendencias da propria umidade do solo
    lags_umidade: tuple[int, ...] = (1, 3, 6, 12, 24, 48)
    # janelas de media para os forcantes atmosfericos
    janelas_media: tuple[int, ...] = (6, 24, 72)

    def camadas_zona_raiz(self, fonte: str) -> tuple[tuple[str, float], ...]:
        """Camadas cuja base esta dentro da profundidade pedida."""
        try:
            camadas = CAMADAS_SOLO[fonte]
        except KeyError as exc:
            raise ValueError(f"fonte sem esquema de solo: {fonte!r}") from exc

        selecionadas = tuple(
            (nome, espessura)
            for nome, espessura, base in camadas
            if base <= self.profundidade_cm + 1e-9
        )
        if not selecionadas:
            disponiveis = [base for _, _, base in camadas]
            raise ValueError(
                f"profundidade {self.profundidade_cm} cm nao cobre nenhuma camada de "
                f"{fonte!r}; bases disponiveis: {disponiveis}"
            )
        return selecionadas

    def profundidade_efetiva_mm(self, fonte: str) -> float:
        return sum(esp for _, esp in self.camadas_zona_raiz(fonte)) * 10.0


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------


def coletar_serie(
    local: Local, janela: JanelaTemporal, fonte: str, chunk_dias: int = 365
) -> pd.DataFrame:
    """Baixa meteorologia + perfil de umidade do solo na resolucao horaria."""
    camadas = tuple(nome for nome, _, _ in CAMADAS_SOLO[fonte])
    variaveis = VARS_METEO + camadas
    with OpenMeteoClient(fonte=fonte) as cliente:
        return cliente.hourly(local, janela, variaveis=variaveis,
                              max_dias_por_chunk=chunk_dias)


# ---------------------------------------------------------------------------
# Preparacao da serie base
# ---------------------------------------------------------------------------


def preparar_serie(horario: pd.DataFrame, fonte: str, cfg: ConfigFeatures) -> pd.DataFrame:
    """Deriva o armazenamento na zona de raizes e as variaveis de forcante."""
    if horario.empty:
        raise ValueError("serie horaria vazia")

    df = horario.copy()
    df.attrs = horario.attrs

    camadas = cfg.camadas_zona_raiz(fonte)
    faltantes = [nome for nome, _ in camadas if nome not in df.columns]
    if faltantes:
        raise ValueError(f"camadas ausentes na resposta da API: {faltantes}")

    # Armazenamento em mm: theta [m3/m3] x espessura [mm]
    df["agua_zona_raiz_mm"] = sum(
        df[nome].astype(float) * espessura * 10.0 for nome, espessura in camadas
    )

    # Deficit de pressao de vapor — forcante direta da demanda atmosferica.
    es = pressao_vapor_saturacao(df["temperature_2m"].to_numpy())
    df["vpd_kpa"] = np.clip(es * (1.0 - df["relative_humidity_2m"].to_numpy() / 100.0), 0.0, None)

    # Balanco hidrico instantaneo (entrada - demanda), em mm/h.
    df["balanco_horario_mm"] = df["precipitation"].fillna(0.0) - df[
        "et0_fao_evapotranspiration"
    ].fillna(0.0)

    essenciais = ["agua_zona_raiz_mm", "temperature_2m", "relative_humidity_2m",
                  "wind_speed_10m", "precipitation", "et0_fao_evapotranspiration"]
    antes = len(df)
    df = df.dropna(subset=essenciais)
    if len(df) < antes:
        LOG.info("removidas %d hora(s) com forcantes ausentes", antes - len(df))

    # Grade horaria estritamente regular: lags em numero de linhas so sao
    # equivalentes a lags em horas se nao houver buraco na serie.
    df = _validar_grade_horaria(df)

    df.attrs = horario.attrs
    return df


def _validar_grade_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexa para grade horaria contigua, marcando buracos como NaN."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("a serie precisa de DatetimeIndex")

    grade = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz=df.index.tz)
    buracos = len(grade) - len(df)
    if buracos > 0:
        LOG.warning(
            "serie com %d hora(s) faltante(s) de %d; reindexando (as features "
            "dessas horas ficam NaN e as amostras sao descartadas)",
            buracos, len(grade),
        )
        attrs = df.attrs
        df = df.reindex(grade)
        df.attrs = attrs
    df.index.name = "time"
    return df


def estimar_limites_hidricos(serie: pd.Series) -> dict[str, float]:
    """Capacidade de campo e ponto de murcha estimados por percentis."""
    cc = float(np.nanpercentile(serie, PERCENTIL_CAPACIDADE_CAMPO))
    pm = float(np.nanpercentile(serie, PERCENTIL_PONTO_MURCHA))
    if not np.isfinite(cc) or not np.isfinite(pm) or cc <= pm:
        raise ValueError(
            f"limites hidricos degenerados (cc={cc}, pm={pm}); serie curta ou constante"
        )
    return {
        "capacidade_campo_mm": cc,
        "ponto_murcha_mm": pm,
        "agua_disponivel_total_mm": cc - pm,
    }


# ---------------------------------------------------------------------------
# Engenharia de features (estritamente causal)
# ---------------------------------------------------------------------------


def construir_features(df: pd.DataFrame, cfg: ConfigFeatures) -> pd.DataFrame:
    """Matriz de features usando apenas informacao disponivel ate o instante t.

    Todas as janelas moveis sao retrospectivas (`rolling` termina em t), e
    nenhuma coluna envolve `shift` negativo — esse e o unico ponto onde
    vazamento temporal poderia entrar sem ser notado.
    """
    X = pd.DataFrame(index=df.index)

    # --- estado atual do solo e sua trajetoria recente ---------------------
    X["agua_atual_mm"] = df["agua_zona_raiz_mm"]
    for lag in cfg.lags_umidade:
        X[f"agua_lag{lag}h_mm"] = df["agua_zona_raiz_mm"].shift(lag)
        # A tendencia e mais informativa que o nivel absoluto defasado.
        X[f"agua_delta{lag}h_mm"] = df["agua_zona_raiz_mm"] - df["agua_zona_raiz_mm"].shift(lag)

    for janela in cfg.janelas_media:
        X[f"agua_media{janela}h_mm"] = (
            df["agua_zona_raiz_mm"].rolling(janela, min_periods=janela).mean()
        )
    X["agua_min24h_mm"] = df["agua_zona_raiz_mm"].rolling(24, min_periods=24).min()
    X["agua_max24h_mm"] = df["agua_zona_raiz_mm"].rolling(24, min_periods=24).max()

    # --- demanda atmosferica acumulada (ETo) e entrada de agua (chuva) -----
    for janela in cfg.janelas_acumulo:
        X[f"eto_acum{janela}h_mm"] = (
            df["et0_fao_evapotranspiration"].rolling(janela, min_periods=janela).sum()
        )
        X[f"chuva_acum{janela}h_mm"] = (
            df["precipitation"].rolling(janela, min_periods=janela).sum()
        )
        X[f"balanco_acum{janela}h_mm"] = (
            df["balanco_horario_mm"].rolling(janela, min_periods=janela).sum()
        )

    # Horas desde a ultima chuva significativa: controla o ramo de secagem.
    # O fillna(0) e necessario para o contador cumulativo e trata hora
    # faltante como hora sem chuva. Nao distorce o treino porque a amostra
    # ja e descartada por `agua_atual_mm` ficar NaN no buraco (ver
    # teste_vazamento.py, bloco 4).
    chuva = df["precipitation"].fillna(0.0)
    molhado = chuva > 0.2
    grupo = molhado.cumsum()
    X["horas_desde_chuva"] = (
        molhado.groupby(grupo).cumcount().where(grupo > 0, other=np.nan).clip(upper=336)
    )
    X["chuva_max1h_24h_mm"] = chuva.rolling(24, min_periods=24).max()

    # --- forcantes atmosfericos -------------------------------------------
    for coluna, apelido in (
        ("temperature_2m", "temp"),
        ("relative_humidity_2m", "ur"),
        ("wind_speed_10m", "vento"),
        ("vpd_kpa", "vpd"),
        ("shortwave_radiation", "rad"),
    ):
        if coluna not in df.columns:
            continue
        X[f"{apelido}_atual"] = df[coluna]
        for janela in cfg.janelas_media:
            X[f"{apelido}_media{janela}h"] = (
                df[coluna].rolling(janela, min_periods=janela).mean()
            )
    if "temperature_2m" in df:
        X["temp_amp24h"] = (
            df["temperature_2m"].rolling(24, min_periods=24).max()
            - df["temperature_2m"].rolling(24, min_periods=24).min()
        )

    # --- tempo como variavel ciclica --------------------------------------
    hora = df.index.hour.to_numpy()
    doy = df.index.dayofyear.to_numpy()
    X["hora_sin"] = np.sin(2 * np.pi * hora / 24.0)
    X["hora_cos"] = np.cos(2 * np.pi * hora / 24.0)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return X


def construir_alvo(
    df: pd.DataFrame, horizonte_h: int, formulacao: str = "delta"
) -> pd.Series:
    """Alvo em t + horizonte. O `shift` negativo aparece SO aqui.

    * `nivel`: y = agua(t+h). Direto, mas a serie e fortemente autocorrelacionada
      e nao estacionaria no nivel — arvores nao extrapolam para faixas de
      umidade ausentes no treino, o que produz vies sistematico no holdout.
    * `delta` (default): y = agua(t+h) - agua(t). O modelo aprende a *dinamica*
      (secagem por ETo, recarga por chuva) em vez de reproduzir o nivel, e a
      persistencia passa a ser o zero explicito do alvo. O nivel e reconstruido
      por `reconstruir_nivel`.
    """
    agua = df["agua_zona_raiz_mm"]
    futuro = agua.shift(-horizonte_h)
    if formulacao == "nivel":
        alvo = futuro
        alvo.name = f"agua_t+{horizonte_h}h_mm"
    elif formulacao == "delta":
        alvo = futuro - agua
        alvo.name = f"delta_agua_{horizonte_h}h_mm"
    else:
        raise ValueError(f"formulacao invalida: {formulacao!r}; use 'delta' ou 'nivel'")
    return alvo


def reconstruir_nivel(
    predicao: np.ndarray, X: pd.DataFrame, formulacao: str, escala: float = 1.0
) -> np.ndarray:
    """Converte a saida do modelo para agua na zona de raizes (mm).

    As metricas sao sempre reportadas nesta escala, para que `delta` e `nivel`
    sejam comparaveis entre si e contra as baselines.

    `escala` encolhe a variacao prevista (ver `ajustar_calibracao`). Com
    escala=1.0 a saida e a do estimador, sem alteracao.
    """
    predicao = np.asarray(predicao, dtype=float)
    atual = X["agua_atual_mm"].to_numpy(dtype=float)
    delta = predicao if formulacao == "delta" else predicao - atual
    return atual + escala * delta


def ajustar_calibracao(
    delta_observado: np.ndarray, delta_previsto: np.ndarray
) -> dict[str, float]:
    """Fator de encolhimento da amplitude prevista que minimiza o MSE.

    Um regressor treinado com perda quadratica devolve a media condicional,
    mas sobre uma serie ruidosa a amplitude da previsao costuma ficar acima
    da que o sinal sustenta — e amplitude excessiva custa MSE mesmo quando o
    sinal (a correlacao) e real.

    Para uma previsao `d` com correlacao `rho` contra o observado, o MSE de
    `a * d` e minimo em a = cov(obs, d) / var(d) = rho * sigma_obs / sigma_d,
    e o skill score resultante e exatamente rho^2. E a mesma correcao de
    "regression dilution": um escalar, estimado FORA da amostra (predicoes
    out-of-fold do treino), nunca no holdout.
    """
    d_obs = np.asarray(delta_observado, dtype=float)
    d_prev = np.asarray(delta_previsto, dtype=float)
    ok = np.isfinite(d_obs) & np.isfinite(d_prev)
    d_obs, d_prev = d_obs[ok], d_prev[ok]

    var = float(np.var(d_prev))
    if var <= 1e-12 or d_obs.size < 50:
        LOG.warning("calibracao indisponivel (variancia %.2e, n=%d); escala=1.0",
                    var, d_obs.size)
        return {"escala": 1.0, "rho": float("nan"), "skill_teorico": float("nan"),
                "amplitude_relativa": float("nan")}

    escala = float(np.cov(d_obs, d_prev, ddof=1)[0, 1] / np.var(d_prev, ddof=1))
    rho = float(np.corrcoef(d_obs, d_prev)[0, 1])
    sigma_obs = float(np.std(d_obs))
    sigma_prev = float(np.std(d_prev))
    return {
        "escala": escala,
        "rho": rho,
        "skill_teorico": rho**2,
        "amplitude_relativa": sigma_prev / sigma_obs if sigma_obs > 0 else float("nan"),
    }


def alvo_para_nivel(y: pd.Series, X: pd.DataFrame, formulacao: str) -> np.ndarray:
    """Mesma conversao aplicada ao alvo observado."""
    if formulacao == "delta":
        return X["agua_atual_mm"].to_numpy(dtype=float) + y.to_numpy(dtype=float)
    return y.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Metricas e baselines
# ---------------------------------------------------------------------------


def metricas(y_true: np.ndarray, y_pred: np.ndarray, profundidade_mm: float) -> dict[str, float]:
    """RMSE/MAE em mm e em m3/m3, mais R2 e vies."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "rmse_mm": rmse,
        "mae_mm": mae,
        "rmse_m3m3": rmse / profundidade_mm,
        "mae_m3m3": mae / profundidade_mm,
        "r2": float(r2_score(y_true, y_pred)),
        "vies_mm": float(np.mean(y_pred - y_true)),
        "n": int(y_true.size),
    }


def predicoes_baseline(
    df: pd.DataFrame, indice: pd.DatetimeIndex, horizonte_h: int
) -> dict[str, np.ndarray]:
    """Baselines observaveis em t.

    * persistencia: y(t+h) = y(t)
    * sazonal diaria: y(t+h) = y(t+h-24), que para h < 24 ja aconteceu

    Em h == 24 a sazonal seria `shift(0)`, ou seja, identica a persistencia;
    nesse caso e omitida em vez de reportada como se fosse uma segunda
    baseline independente.
    """
    agua = df["agua_zona_raiz_mm"]
    base = {"persistencia": agua.reindex(indice).to_numpy()}
    if horizonte_h < 24:
        base["sazonal_24h"] = agua.shift(24 - horizonte_h).reindex(indice).to_numpy()
    return base


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------


def catalogo_modelos(semente: int, n_jobs: int) -> dict[str, object]:
    """Regressores candidatos. Arvores dispensam padronizacao das features."""
    modelos: dict[str, object] = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.4,
            n_jobs=n_jobs,
            random_state=semente,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=semente,
        ),
    }
    if XGBOOST_DISPONIVEL:
        modelos["xgboost"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=n_jobs,
            random_state=semente,
            tree_method="hist",
        )
    return modelos


# ---------------------------------------------------------------------------
# Treinamento de um horizonte
# ---------------------------------------------------------------------------


@dataclass
class ResultadoTreino:
    horizonte_h: int
    formulacao: str
    nome_modelo: str
    modelo: object
    features: list[str]
    metricas_holdout: dict[str, float]
    metricas_baselines: dict[str, dict[str, float]]
    skill_score: dict[str, float]
    calibracao: dict[str, float]
    skill_eventos: dict[str, float]
    metricas_cv: dict[str, dict[str, float]]
    importancias: pd.DataFrame
    predicoes: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def treinar_horizonte(
    df: pd.DataFrame,
    cfg: ConfigFeatures,
    fonte: str,
    horizonte_h: int,
    fracao_holdout: float,
    n_splits: int,
    semente: int,
    n_jobs: int,
    modelos_pedidos: Sequence[str],
    formulacao: str = "delta",
    calibrar: bool = True,
    n_dobras_calibracao: int = 2,
) -> ResultadoTreino:
    profundidade_mm = cfg.profundidade_efetiva_mm(fonte)

    X_completo = construir_features(df, cfg)
    y_completo = construir_alvo(df, horizonte_h, formulacao)

    # Descarta o aquecimento das janelas e a cauda sem alvo, de uma vez.
    valido = X_completo.notna().all(axis=1) & y_completo.notna()
    X = X_completo.loc[valido]
    y = y_completo.loc[valido]
    LOG.info(
        "h=%dh: %d amostras utilizaveis de %d (%d features)",
        horizonte_h, len(X), len(X_completo), X.shape[1],
    )
    minimo = max(200, 10 * X.shape[1])
    if len(X) < minimo:
        raise ValueError(
            f"amostras insuficientes para h={horizonte_h}h: {len(X)} < {minimo}. "
            "Aumente --anos ou reduza as janelas em ConfigFeatures."
        )

    # --- particao cronologica ---------------------------------------------
    corte = int(len(X) * (1.0 - fracao_holdout))
    X_treino, X_teste = X.iloc[:corte], X.iloc[corte:]
    y_treino, y_teste = y.iloc[:corte], y.iloc[corte:]
    LOG.info(
        "h=%dh: treino %s -> %s | holdout %s -> %s",
        horizonte_h,
        X_treino.index[0].date(), X_treino.index[-1].date(),
        X_teste.index[0].date(), X_teste.index[-1].date(),
    )

    # --- selecao por validacao cruzada temporal ---------------------------
    # gap=horizonte_h impede que a janela de treino alcance o alvo da validacao.
    cv = TimeSeriesSplit(n_splits=n_splits, gap=horizonte_h)
    candidatos = catalogo_modelos(semente, n_jobs)
    selecionados = {k: v for k, v in candidatos.items() if k in modelos_pedidos}
    if not selecionados:
        raise ValueError(
            f"nenhum modelo disponivel entre {list(modelos_pedidos)}; "
            f"disponiveis: {list(candidatos)}"
        )

    metricas_cv: dict[str, dict[str, float]] = {}
    for nome, estimador in selecionados.items():
        rmses, maes = [], []
        for treino_idx, val_idx in cv.split(X_treino):
            X_val = X_treino.iloc[val_idx]
            estimador.fit(X_treino.iloc[treino_idx], y_treino.iloc[treino_idx])
            # Metricas na escala de nivel (mm de agua), nao na do alvo nativo:
            # so assim 'delta' e 'nivel' sao comparaveis entre si.
            pred = reconstruir_nivel(estimador.predict(X_val), X_val, formulacao)
            obs = alvo_para_nivel(y_treino.iloc[val_idx], X_val, formulacao)
            m = metricas(obs, pred, profundidade_mm)
            rmses.append(m["rmse_mm"])
            maes.append(m["mae_mm"])
        metricas_cv[nome] = {
            "rmse_mm_medio": float(np.mean(rmses)),
            "rmse_mm_desvio": float(np.std(rmses)),
            "mae_mm_medio": float(np.mean(maes)),
        }
        LOG.info(
            "h=%dh %-24s CV RMSE %.3f +/- %.3f mm",
            horizonte_h, nome, np.mean(rmses), np.std(rmses),
        )

    melhor = min(metricas_cv, key=lambda k: metricas_cv[k]["rmse_mm_medio"])
    LOG.info("h=%dh: modelo escolhido -> %s", horizonte_h, melhor)

    # --- calibracao da amplitude, out-of-fold -----------------------------
    # Estimada nas dobras de validacao do TREINO. Usar o holdout aqui seria
    # ajustar no proprio conjunto de teste e inflar o resultado.
    modelo = selecionados[melhor]
    calibracao = {"escala": 1.0, "rho": float("nan"), "skill_teorico": float("nan"),
                  "amplitude_relativa": float("nan")}
    if calibrar:
        # Só as ULTIMAS dobras. As primeiras treinam com uma fracao pequena da
        # serie e caem em regimes antigos: o rho delas subestima o do modelo
        # final (treinado em tudo) e o fator sai encolhido demais. Medido neste
        # projeto: usando as 5 dobras, rho_oof=0.113 contra rho=0.255 no
        # holdout, e o fator ficou ~4x mais conservador que o otimo.
        dobras = list(cv.split(X_treino))
        usadas = dobras[-max(1, min(n_dobras_calibracao, len(dobras))):]
        d_obs_oof, d_prev_oof = [], []
        for treino_idx, val_idx in usadas:
            X_val = X_treino.iloc[val_idx]
            modelo.fit(X_treino.iloc[treino_idx], y_treino.iloc[treino_idx])
            atual = X_val["agua_atual_mm"].to_numpy(dtype=float)
            d_prev_oof.append(
                reconstruir_nivel(modelo.predict(X_val), X_val, formulacao) - atual
            )
            d_obs_oof.append(alvo_para_nivel(y_treino.iloc[val_idx], X_val, formulacao) - atual)
        calibracao = ajustar_calibracao(
            np.concatenate(d_obs_oof), np.concatenate(d_prev_oof)
        )
        calibracao["dobras_usadas"] = len(usadas)
        LOG.info(
            "h=%dh: calibracao escala=%.3f (rho=%.3f em %d dobra(s) finais, "
            "amplitude %.2fx do observado, skill teorico %.3f)",
            horizonte_h, calibracao["escala"], calibracao["rho"], len(usadas),
            calibracao["amplitude_relativa"], calibracao["skill_teorico"],
        )

    # --- reajuste em todo o treino e avaliacao no holdout -----------------
    modelo.fit(X_treino, y_treino)
    bruto = modelo.predict(X_teste)
    y_pred = reconstruir_nivel(bruto, X_teste, formulacao, calibracao["escala"])
    y_obs = alvo_para_nivel(y_teste, X_teste, formulacao)

    # Ganho atribuivel a calibracao, medido no holdout (so para o relatorio).
    y_pred_sem_cal = reconstruir_nivel(bruto, X_teste, formulacao, 1.0)
    calibracao["rmse_sem_calibracao_mm"] = float(
        np.sqrt(mean_squared_error(y_obs, y_pred_sem_cal))
    )

    m_modelo = metricas(y_obs, y_pred, profundidade_mm)
    bases = predicoes_baseline(df, X_teste.index, horizonte_h)
    m_bases, skill = {}, {}
    for nome, pred in bases.items():
        # A baseline sazonal tem NaN no inicio da serie: compara no subconjunto
        # em que ambas existem, senao o skill score mistura amostragens.
        ok = np.isfinite(pred) & np.isfinite(y_obs)
        m_bases[nome] = metricas(y_obs[ok], pred[ok], profundidade_mm)
        # Skill score = 1 - MSE_modelo / MSE_baseline (1.0 = perfeito, <=0 = inutil)
        mse_base = float(mean_squared_error(y_obs[ok], pred[ok]))
        mse_mod = float(mean_squared_error(y_obs[ok], y_pred[ok]))
        skill[nome] = float(1.0 - mse_mod / mse_base) if mse_base > 0 else float("nan")

    # --- skill condicionado a magnitude do evento -------------------------
    # O skill medio e dominado pelas horas em que quase nada muda, e nessas a
    # persistencia e imbativel por definicao. O que decide irrigacao sao as
    # horas de variacao grande: medir o skill so nelas diz se o modelo tem
    # valor operacional, mesmo com skill medio perto de zero.
    persist = X_teste["agua_atual_mm"].to_numpy(dtype=float)
    variacao_real = np.abs(y_obs - persist)
    skill_eventos: dict[str, float] = {}
    for rotulo, quantil in (("decil_superior", 0.90), ("quartil_superior", 0.75)):
        limiar = float(np.quantile(variacao_real, quantil))
        sel = variacao_real >= limiar
        if sel.sum() < 30:
            continue
        mse_mod = float(mean_squared_error(y_obs[sel], y_pred[sel]))
        mse_per = float(mean_squared_error(y_obs[sel], persist[sel]))
        skill_eventos[rotulo] = float(1.0 - mse_mod / mse_per) if mse_per > 0 else float("nan")
        skill_eventos[f"{rotulo}_limiar_mm"] = limiar
        skill_eventos[f"{rotulo}_n"] = int(sel.sum())

    # --- curva de trade-off do fator de escala (DIAGNOSTICO) --------------
    # Mostra o conflito entre as duas coisas que se quer de uma previsao de
    # umidade: erro medio baixo (escala pequena, porque a maioria das horas e
    # calma) e deteccao de evento (escala grande, para acompanhar a amplitude
    # real). Escolher uma escala desta tabela seria ajustar no teste — ela
    # serve para decidir o OBJETIVO, nao para calibrar.
    d_obs_h = y_obs - persist
    d_bruto_h = y_pred_sem_cal - persist
    calmo = np.abs(d_obs_h) <= np.quantile(np.abs(d_obs_h), 0.50)
    decil_mask = np.abs(d_obs_h) >= np.quantile(np.abs(d_obs_h), 0.90)
    curva: list[dict[str, float]] = []
    for a in (0.0, calibracao["escala"], 0.25, 0.5, 0.75, 1.0):
        erro = d_obs_h - a * d_bruto_h
        def _skill(mask: np.ndarray) -> float:
            base = float(np.mean(d_obs_h[mask] ** 2))
            return float(1.0 - np.mean(erro[mask] ** 2) / base) if base > 0 else float("nan")
        curva.append(
            {
                "escala": float(a),
                "rmse_mm": float(np.sqrt(np.mean(erro**2))),
                "mae_mm": float(np.mean(np.abs(erro))),
                "skill_medio": _skill(np.ones_like(calmo, dtype=bool)),
                "skill_decil": _skill(decil_mask),
                "skill_horas_calmas": _skill(calmo),
            }
        )
    calibracao["curva_escala"] = curva

    # --- importancia por permutacao (no holdout, nao no treino) -----------
    LOG.info("h=%dh: calculando importancia por permutacao", horizonte_h)
    perm = permutation_importance(
        modelo, X_teste, y_teste, n_repeats=5, random_state=semente,
        scoring="neg_root_mean_squared_error", n_jobs=n_jobs,
    )
    importancias = (
        pd.DataFrame(
            {"feature": X.columns, "importancia_mm": perm.importances_mean,
             "desvio": perm.importances_std}
        )
        .sort_values("importancia_mm", ascending=False)
        .reset_index(drop=True)
    )

    predicoes = pd.DataFrame(
        {"observado_mm": y_obs, "previsto_mm": y_pred}, index=X_teste.index
    )
    for nome, pred in bases.items():
        predicoes[f"baseline_{nome}_mm"] = pred

    return ResultadoTreino(
        horizonte_h=horizonte_h,
        formulacao=formulacao,
        nome_modelo=melhor,
        modelo=modelo,
        features=list(X.columns),
        metricas_holdout=m_modelo,
        metricas_baselines=m_bases,
        skill_score=skill,
        calibracao=calibracao,
        skill_eventos=skill_eventos,
        metricas_cv=metricas_cv,
        importancias=importancias,
        predicoes=predicoes,
    )


# ---------------------------------------------------------------------------
# Persistencia do modelo
# ---------------------------------------------------------------------------

VERSAO_BUNDLE = 1


def exportar_bundle(
    resultado: ResultadoTreino,
    local: Local,
    cfg: ConfigFeatures,
    fonte: str,
    limites: dict[str, float],
    janela: JanelaTemporal,
    destino: Path,
) -> Path:
    """Grava modelo + contrato de features + metadados num unico .joblib.

    Salvar so o estimador e o erro classico: na inferencia nao haveria como
    saber a ordem das features, o esquema de camadas nem a capacidade de campo
    usada para converter agua em deficit.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "versao_bundle": VERSAO_BUNDLE,
        "modelo": resultado.modelo,
        "nome_modelo": resultado.nome_modelo,
        "features": resultado.features,
        "alvo": "agua_zona_raiz_mm",
        # A formulacao define como a saida bruta do estimador vira nivel de
        # agua: sem ela o consumidor do bundle somaria (ou nao) o estado atual
        # por conta propria e erraria por ~90 mm.
        "formulacao": resultado.formulacao,
        # Escalar de calibracao da amplitude: aplicado em `reconstruir_nivel`.
        "calibracao": resultado.calibracao,
        "escala_delta": resultado.calibracao["escala"],
        "horizonte_h": resultado.horizonte_h,
        "fonte_dados": fonte,
        "esquema_camadas": CAMADAS_SOLO[fonte],
        "config_features": asdict(cfg),
        "local": asdict(local),
        "limites_hidricos": limites,
        "profundidade_efetiva_mm": cfg.profundidade_efetiva_mm(fonte),
        "periodo_treino": {"inicio": janela.inicio.isoformat(), "fim": janela.fim.isoformat()},
        "metricas_holdout": resultado.metricas_holdout,
        "metricas_baselines": resultado.metricas_baselines,
        "skill_score": resultado.skill_score,
        "skill_eventos": resultado.skill_eventos,
        "metricas_cv": resultado.metricas_cv,
        "top_features": resultado.importancias.head(15).to_dict("records"),
        "treinado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versoes": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(bundle, destino, compress=3)
    return destino


def carregar_bundle(caminho: Path) -> dict:
    bundle = joblib.load(caminho)
    if not isinstance(bundle, dict) or "modelo" not in bundle:
        raise ValueError(f"{caminho} nao parece um bundle deste script")
    versao = bundle.get("versao_bundle")
    if versao != VERSAO_BUNDLE:
        LOG.warning("bundle versao %s; este script escreve versao %s", versao, VERSAO_BUNDLE)
    return bundle


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------


def prever_com_bundle(caminho: Path, dias_contexto: int = 30) -> pd.DataFrame:
    """Coleta o historico recente e preve o horizonte do bundle.

    O contexto precisa cobrir a maior janela de feature (168h por padrao),
    por isso o default de 30 dias.
    """
    bundle = carregar_bundle(caminho)
    cfg = ConfigFeatures(**bundle["config_features"])
    local = Local(**bundle["local"])
    fonte = bundle["fonte_dados"]
    horizonte = bundle["horizonte_h"]

    hoje = date.today()
    janela = JanelaTemporal(hoje - timedelta(days=dias_contexto), hoje)
    LOG.info(
        "inferencia h=+%dh | fonte %s | contexto %s a %s",
        horizonte, fonte, janela.inicio, janela.fim,
    )

    horario = coletar_serie(local, janela, fonte, chunk_dias=dias_contexto + 1)
    df = preparar_serie(horario, fonte, cfg)

    X = construir_features(df, cfg)
    X = X.loc[X.notna().all(axis=1)]
    if X.empty:
        raise ValueError(
            "contexto insuficiente para completar as features; aumente --dias-contexto"
        )

    # Contrato de features: mesma ordem e mesmo conjunto do treino.
    esperadas = bundle["features"]
    faltando = [c for c in esperadas if c not in X.columns]
    if faltando:
        raise ValueError(f"features ausentes em relacao ao treino: {faltando}")
    X = X[esperadas]

    formulacao = bundle.get("formulacao", "nivel")
    escala = float(bundle.get("escala_delta", 1.0))
    pred = reconstruir_nivel(bundle["modelo"].predict(X), X, formulacao, escala)
    cc = bundle["limites_hidricos"]["capacidade_campo_mm"]
    pm = bundle["limites_hidricos"]["ponto_murcha_mm"]
    prof = bundle["profundidade_efetiva_mm"]

    saida = pd.DataFrame(index=X.index)
    saida["agua_atual_mm"] = df["agua_zona_raiz_mm"].reindex(X.index)
    saida["agua_prevista_mm"] = pred
    saida["variacao_prevista_mm"] = pred - saida["agua_atual_mm"]
    saida["deficit_previsto_mm"] = cc - pred
    saida["umidade_prevista_m3m3"] = pred / prof
    saida["fracao_agua_disponivel"] = ((pred - pm) / (cc - pm)).clip(0.0, 1.5)
    saida["valido_para"] = saida.index + pd.Timedelta(hours=horizonte)
    saida.attrs["horizonte_h"] = horizonte
    saida.attrs["bundle"] = str(caminho)
    return saida


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------


def imprimir_relatorio(resultado: ResultadoTreino, limites: dict[str, float]) -> None:
    h = resultado.horizonte_h
    m = resultado.metricas_holdout
    prof_disp = limites["agua_disponivel_total_mm"]

    print(f"\n{'=' * 72}")
    print(f"HORIZONTE +{h}h  |  modelo: {resultado.nome_modelo}"
          f"  |  alvo: {resultado.formulacao}")
    print("=" * 72)
    print(f"  holdout: {m['n']} amostras")
    print(f"  RMSE : {m['rmse_mm']:7.3f} mm   ({m['rmse_m3m3']:.5f} m3/m3)")
    print(f"  MAE  : {m['mae_mm']:7.3f} mm   ({m['mae_m3m3']:.5f} m3/m3)")
    print(f"  R2   : {m['r2']:7.3f}")
    print(f"  vies : {m['vies_mm']:+7.3f} mm")
    print(f"  RMSE como % da agua disponivel total: {100 * m['rmse_mm'] / prof_disp:.1f}%")

    print("\n  Comparacao com baselines (RMSE mm | skill score):")
    for nome, mb in resultado.metricas_baselines.items():
        ss = resultado.skill_score[nome]
        veredito = "ganha" if ss > 0 else "PERDE"
        print(f"    {nome:16s} {mb['rmse_mm']:7.3f}  |  {ss:+.3f}  ({veredito})")

    cal = resultado.calibracao
    if np.isfinite(cal.get("rho", float("nan"))):
        print("\n  Calibracao da amplitude (ajustada out-of-fold no treino):")
        print(f"    correlacao da variacao (rho) : {cal['rho']:+.3f}")
        print(f"    skill teorico maximo (rho^2) : {cal['skill_teorico']:.3f}")
        print(f"    amplitude bruta do modelo    : {cal['amplitude_relativa']:.2f}x "
              f"a observada")
        print(f"    fator aplicado               : {cal['escala']:.3f}")
        bruto = cal.get("rmse_sem_calibracao_mm")
        if bruto:
            print(f"    RMSE sem / com calibracao    : {bruto:.3f} -> "
                  f"{m['rmse_mm']:.3f} mm")

    if cal.get("curva_escala"):
        print("\n  Trade-off do fator de escala (DIAGNOSTICO — nao use para")
        print("  escolher a escala, isso seria ajustar no teste):")
        print(f"    {'escala':>7} {'RMSE':>7} {'MAE':>7} {'skill_med':>10}"
              f" {'skill_decil':>12} {'skill_calmas':>13}")
        for linha in cal["curva_escala"]:
            marca = "*" if abs(linha["escala"] - cal["escala"]) < 1e-9 else " "
            print(f"    {linha['escala']:6.3f}{marca} {linha['rmse_mm']:7.3f}"
                  f" {linha['mae_mm']:7.3f} {linha['skill_medio']:10.4f}"
                  f" {linha['skill_decil']:12.4f} {linha['skill_horas_calmas']:13.4f}")
        print("    (* = fator aplicado;  escala 0 = persistencia pura,"
              " 1 = saida crua do modelo)")

    if resultado.skill_eventos:
        print("\n  Skill nas horas de maior variacao real (vs persistencia):")
        for rotulo in ("quartil_superior", "decil_superior"):
            if rotulo not in resultado.skill_eventos:
                continue
            ss = resultado.skill_eventos[rotulo]
            limiar = resultado.skill_eventos[f"{rotulo}_limiar_mm"]
            n = resultado.skill_eventos[f"{rotulo}_n"]
            print(f"    {rotulo:18s} (|d|>={limiar:.2f} mm, n={n:4d}) {ss:+.3f}")

    ss_pers = resultado.skill_score.get("persistencia", 0.0)
    decil = resultado.skill_eventos.get("decil_superior")

    if ss_pers < -0.02:
        # Caso distinto do empate: aqui o modelo e ativamente pior que nao
        # fazer nada, e nenhum ganho em evento compensa isso no agregado.
        print(
            f"\n  [REPROVADO] skill {ss_pers:+.3f}: o modelo e PIOR que repetir o\n"
            "  valor atual. Nao use. Causa mais comum: prever o nivel absoluto\n"
            "  de uma serie autocorrelacionada e nao estacionaria — tente\n"
            "  --formulacao delta. Veja tambem o vies acima: se for grande, o\n"
            "  holdout caiu numa faixa de umidade ausente do treino e arvores\n"
            "  nao extrapolam (mais anos em --anos)."
        )
        if decil and decil > 0.05:
            print(
                f"  Observacao: o skill no decil de maior variacao e {decil:+.3f},\n"
                "  ou seja ha sinal — ele esta sendo anulado pelo erro nas horas\n"
                "  calmas. Ver a curva de trade-off acima."
            )
    elif ss_pers <= 0.01:
        print(
            f"\n  [ATENCAO] skill {ss_pers:+.3f}: empate tecnico com a persistencia.\n"
            "  Em series fortemente autocorrelacionadas isso e esperado, pois a\n"
            "  media e dominada pelas horas em que quase nada muda."
        )
        if decil and decil > 0.05:
            print(
                f"  O modelo ainda agrega no que importa (skill {decil:+.3f} no\n"
                "  decil de maior variacao). Use-o para antecipar EVENTOS, nao\n"
                "  para refinar o nivel em hora calma."
            )
        else:
            print(
                "  E tambem nao agrega nos eventos: nao coloque em producao.\n"
                "  Caminhos: horizonte maior, mais anos (--anos) ou alvo de\n"
                "  sensor real em vez de reanalise."
            )

    print("\n  Top 10 features (perda de RMSE ao permutar, mm):")
    for _, linha in resultado.importancias.head(10).iterrows():
        print(f"    {linha['feature']:28s} {linha['importancia_mm']:7.4f}"
              f" +/- {linha['desvio']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Treina e exporta modelos de previsao de umidade do solo (+12h/+24h).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prever", type=Path, default=None,
                   help="modo inferencia: caminho de um bundle .joblib")
    p.add_argument("--dias-contexto", type=int, default=30,
                   help="dias de historico baixados na inferencia")

    p.add_argument("--lat", type=float, default=float(os.getenv("METEO_LAT", "-15.7939")))
    p.add_argument("--lon", type=float, default=float(os.getenv("METEO_LON", "-47.8828")))
    p.add_argument("--nome", default="minha_regiao")
    p.add_argument("--timezone", default="auto")

    p.add_argument("--anos", type=float, default=3.0,
                   help="anos de historico para treino (fonte archive)")
    p.add_argument("--fim", type=str, default=None,
                   help="data final do treino YYYY-MM-DD (default: ontem, ou D-6 no archive)")
    p.add_argument("--fonte", choices=("archive", "forecast"), default="archive",
                   help="archive = ERA5 multi-anual; forecast = ~92 dias, esquema operacional")

    p.add_argument("--horizontes", type=int, nargs="+", default=[12, 24],
                   help="horizontes de previsao em horas")
    p.add_argument("--profundidade-cm", type=float, default=28.0,
                   help="profundidade da zona de raizes")
    p.add_argument("--formulacao", choices=("delta", "nivel"), default="delta",
                   help="alvo: variacao sobre o estado atual (delta) ou nivel absoluto")
    p.add_argument("--sem-calibracao", dest="calibrar", action="store_false",
                   help="desliga o encolhimento out-of-fold da amplitude prevista")
    p.add_argument("--dobras-calibracao", type=int, default=2,
                   help="quantas dobras FINAIS da CV alimentam a calibracao "
                        "(as iniciais treinam com pouca serie e enviesam o fator)")
    p.add_argument("--modelos", nargs="+",
                   default=["random_forest", "hist_gradient_boosting", "xgboost"],
                   help="candidatos; o melhor por CV temporal e exportado")
    p.add_argument("--holdout", type=float, default=0.2,
                   help="fracao final da serie reservada para teste")
    p.add_argument("--splits", type=int, default=5, help="dobras do TimeSeriesSplit")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--saida", type=Path, default=Path("./modelos"))
    p.add_argument("--salvar-predicoes", action="store_true",
                   help="grava CSV com observado/previsto/baselines do holdout")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolver_janela_treino(args: argparse.Namespace) -> JanelaTemporal:
    if args.fim:
        fim = datetime.strptime(args.fim, "%Y-%m-%d").date()
    elif args.fonte == "archive":
        # O ERA5 tem ~5 dias de latencia; pedir alem disso devolve NaN.
        fim = date.today() - timedelta(days=6)
    else:
        fim = date.today() - timedelta(days=1)

    if args.fonte == "forecast":
        # A Forecast API guarda no maximo ~92 dias de passado.
        dias = min(int(args.anos * 365.25), 90)
        if dias < int(args.anos * 365.25):
            LOG.warning(
                "fonte 'forecast' limita o historico a ~90 dias; usando %d dias "
                "em vez de %.1f ano(s)", dias, args.anos,
            )
    else:
        dias = int(args.anos * 365.25)
    return JanelaTemporal(fim - timedelta(days=dias - 1), fim)


def main(argv: Sequence[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not XGBOOST_DISPONIVEL and "xgboost" in args.modelos:
        LOG.info("xgboost nao instalado; seguindo com os modelos do scikit-learn")

    # ---------------- modo inferencia ----------------
    if args.prever:
        try:
            saida = prever_com_bundle(args.prever, dias_contexto=args.dias_contexto)
        except (OpenMeteoError, ValueError, FileNotFoundError) as exc:
            LOG.error("inferencia falhou: %s", exc)
            return 1
        ultima = saida.iloc[-1]
        print(f"\nPrevisao a partir de {saida.index[-1]:%Y-%m-%d %H:%M %Z} "
              f"(valida para {ultima['valido_para']:%Y-%m-%d %H:%M})")
        print(f"  agua na zona de raizes agora : {ultima['agua_atual_mm']:.2f} mm")
        print(f"  agua prevista                : {ultima['agua_prevista_mm']:.2f} mm "
              f"({ultima['umidade_prevista_m3m3']:.4f} m3/m3)")
        print(f"  deficit hidrico previsto     : {ultima['deficit_previsto_mm']:.2f} mm")
        print(f"  fracao de agua disponivel    : {ultima['fracao_agua_disponivel']:.1%}")
        print("\nUltimas 12 horas de previsao:")
        colunas = ["agua_atual_mm", "agua_prevista_mm", "deficit_previsto_mm",
                   "fracao_agua_disponivel"]
        print(saida[colunas].tail(12).round(3).to_string())
        return 0

    # ---------------- modo treino ----------------
    try:
        local = Local(latitude=args.lat, longitude=args.lon,
                      timezone=args.timezone, nome=args.nome)
        cfg = ConfigFeatures(profundidade_cm=args.profundidade_cm)
        janela = resolver_janela_treino(args)
        camadas = cfg.camadas_zona_raiz(args.fonte)
    except ValueError as exc:
        LOG.error("configuracao invalida: %s", exc)
        return 2

    LOG.info(
        "zona de raizes 0-%.0f cm via %s: %s (%.0f mm de perfil)",
        args.profundidade_cm, args.fonte,
        ", ".join(n for n, _ in camadas), cfg.profundidade_efetiva_mm(args.fonte),
    )
    LOG.info("coletando %s a %s (%d dias)", janela.inicio, janela.fim, janela.dias)

    try:
        horario = coletar_serie(local, janela, args.fonte)
        df = preparar_serie(horario, args.fonte, cfg)
        limites = estimar_limites_hidricos(df["agua_zona_raiz_mm"])
    except (OpenMeteoError, ValueError) as exc:
        LOG.error("preparacao dos dados falhou: %s", exc)
        return 1

    LOG.info(
        "serie pronta: %d horas | agua %.1f-%.1f mm | cc=%.1f pm=%.1f (disponivel %.1f mm)",
        len(df), df["agua_zona_raiz_mm"].min(), df["agua_zona_raiz_mm"].max(),
        limites["capacidade_campo_mm"], limites["ponto_murcha_mm"],
        limites["agua_disponivel_total_mm"],
    )

    args.saida.mkdir(parents=True, exist_ok=True)
    resumo_geral: list[dict[str, object]] = []

    for h in args.horizontes:
        try:
            resultado = treinar_horizonte(
                df=df, cfg=cfg, fonte=args.fonte, horizonte_h=h,
                fracao_holdout=args.holdout, n_splits=args.splits,
                semente=args.semente, n_jobs=args.n_jobs,
                modelos_pedidos=args.modelos, formulacao=args.formulacao,
                calibrar=args.calibrar, n_dobras_calibracao=args.dobras_calibracao,
            )
        except ValueError as exc:
            LOG.error("horizonte %dh ignorado: %s", h, exc)
            continue

        imprimir_relatorio(resultado, limites)

        caminho = exportar_bundle(
            resultado, local, cfg, args.fonte, limites, janela,
            args.saida / f"solo_{local.nome}_h{h}.joblib",
        )
        LOG.info("bundle salvo -> %s (%.1f KB)", caminho, caminho.stat().st_size / 1024)

        if args.salvar_predicoes:
            csv = args.saida / f"predicoes_{local.nome}_h{h}.csv"
            resultado.predicoes.to_csv(csv, float_format="%.4f")
            LOG.info("predicoes do holdout -> %s", csv)

        resumo_geral.append(
            {
                "horizonte_h": h,
                "modelo": resultado.nome_modelo,
                "formulacao": resultado.formulacao,
                "rmse_mm": round(resultado.metricas_holdout["rmse_mm"], 4),
                "mae_mm": round(resultado.metricas_holdout["mae_mm"], 4),
                "r2": round(resultado.metricas_holdout["r2"], 4),
                "skill_vs_persistencia": round(resultado.skill_score["persistencia"], 4),
                "skill_decil_superior": round(
                    resultado.skill_eventos.get("decil_superior", float("nan")), 4
                ),
                "escala_calibracao": round(resultado.calibracao["escala"], 4),
                "bundle": caminho.name,
            }
        )

    if not resumo_geral:
        LOG.error("nenhum horizonte treinado com sucesso")
        return 1

    print(f"\n{'=' * 72}\nRESUMO\n{'=' * 72}")
    print(pd.DataFrame(resumo_geral).to_string(index=False))

    caminho_resumo = args.saida / f"resumo_{local.nome}.json"
    caminho_resumo.write_text(
        json.dumps(
            {"local": asdict(local), "fonte": args.fonte, "limites_hidricos": limites,
             "periodo": {"inicio": janela.inicio.isoformat(), "fim": janela.fim.isoformat()},
             "resultados": resumo_geral},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LOG.info("resumo -> %s", caminho_resumo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
