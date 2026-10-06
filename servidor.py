#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend FastAPI entre o ESP32 e o controle de irrigacao.

Dois endpoints de operacao:

    POST /api/v1/leitura          o ESP32 envia a leitura do sensor
    GET  /api/v1/comando/{id}     devolve se a bomba liga e por quantos segundos

Mais `GET /health` e `GET /api/v1/status/{id}` para operacao e diagnostico.

Arquitetura: dois lacos de velocidades diferentes
-------------------------------------------------
O modelo de IA precisa de 168 h de historico e de uma chamada a Open-Meteo —
leva segundos e tem horizonte de 12 a 24 h. O ESP32 consulta a cada minuto.
Rodar o modelo a cada requisicao seria lento e inutil: a previsao nao muda
em 60 s.

    laco lento  (a cada ~1 h)  modelo -> SETPOINT, em cache com TTL
    laco rapido (a cada poll)  setpoint + leitura -> PID -> PULSO

Se o modelo falhar, ou a internet cair, o setpoint cai para uma fracao fixa
da agua disponivel e o controle **continua funcionando**. Um sistema de
irrigacao que para porque a IA parou e pior que um temporizador burro.

Decisoes de protocolo pensadas para o ESP32
-------------------------------------------
* O relogio e do SERVIDOR. ESP32 sem RTC perde a hora a cada reset, e
  carimbo de tempo errado corrompe o lockout e a cota diaria.
* O GET e idempotente dentro de uma janela: se o dispositivo repetir a
  requisicao (timeout, reenvio), recebe o MESMO comando, com o mesmo
  `comando_id`, em vez de uma segunda dose de agua.
* A resposta traz `proximo_poll_s`, para o dispositivo dormir o tempo certo
  em vez de consultar em laco e gastar bateria.
* O corpo aceita `umidade_m3m3` (o que o sensor entrega) ou `umidade_mm`.
* O estado critico (cota diaria, ultima irrigacao) e persistido em disco:
  sem isso, um laco de reinicializacao zeraria a cota e encharcaria a
  lavoura.

Execucao
--------
    pip install fastapi "uvicorn[standard]"
    uvicorn servidor:app --host 0.0.0.0 --port 8000

    # com autenticacao (recomendado: o endpoint comanda uma bomba)
    set IRRIGACAO_API_KEY=um-segredo-longo
    uvicorn servidor:app --host 0.0.0.0 --port 8000

Documentacao interativa em http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from controle_irrigacao import (
    Atuador,
    ControladorFuzzy,
    ControladorPID,
    Decisao,
    GanhosPID,
    LimitesSeguranca,
    Motivo,
    setpoint_por_fracao_disponivel,
    vwc_para_mm,
)

LOG = logging.getLogger("irrigacao.servidor")


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------


def _env_float(nome: str, padrao: float) -> float:
    try:
        return float(os.getenv(nome, padrao))
    except (TypeError, ValueError):
        LOG.warning("%s invalido no ambiente; usando %s", nome, padrao)
        return padrao


@dataclass(frozen=True)
class Config:
    api_key: str | None = os.getenv("IRRIGACAO_API_KEY") or None
    bundle: Path | None = (
        Path(os.getenv("IRRIGACAO_BUNDLE")) if os.getenv("IRRIGACAO_BUNDLE") else None
    )
    estado_path: Path = Path(os.getenv("IRRIGACAO_ESTADO", "estado_servidor.json"))

    # Hidraulica do conjunto (vazao total da zona)
    vazao_lph: float = _env_float("IRRIGACAO_VAZAO_LPH", 400.0)
    area_m2: float = _env_float("IRRIGACAO_AREA_M2", 50.0)
    eficiencia: float = _env_float("IRRIGACAO_EFICIENCIA", 0.85)
    pulso_min_s: float = _env_float("IRRIGACAO_PULSO_MIN_S", 30.0)
    pulso_max_s: float = _env_float("IRRIGACAO_PULSO_MAX_S", 1800.0)

    # Solo (sobrescritos pelos limites gravados no bundle, quando houver)
    capacidade_campo_mm: float = _env_float("IRRIGACAO_CC_MM", 122.6)
    ponto_murcha_mm: float = _env_float("IRRIGACAO_PM_MM", 71.8)
    profundidade_mm: float = _env_float("IRRIGACAO_PROFUNDIDADE_MM", 280.0)
    lamina_diaria_max_mm: float = _env_float("IRRIGACAO_LAMINA_DIA_MM", 15.0)

    fracao_alvo: float = _env_float("IRRIGACAO_FRACAO_ALVO", 0.65)
    confianca_modelo: float = _env_float("IRRIGACAO_CONFIANCA", 0.5)
    ttl_setpoint_s: float = _env_float("IRRIGACAO_TTL_SETPOINT_S", 3600.0)

    poll_normal_s: int = int(_env_float("IRRIGACAO_POLL_S", 300.0))
    poll_irrigando_s: int = int(_env_float("IRRIGACAO_POLL_IRRIGANDO_S", 60.0))
    validade_comando_s: float = _env_float("IRRIGACAO_VALIDADE_COMANDO_S", 120.0)
    leitura_velha_s: float = _env_float("IRRIGACAO_LEITURA_VELHA_S", 1800.0)
    historico_max: int = int(_env_float("IRRIGACAO_HISTORICO_MAX", 2880.0))

    controlador: Literal["pid", "fuzzy"] = os.getenv("IRRIGACAO_CONTROLADOR", "pid")  # type: ignore[assignment]


CFG = Config()


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------


class LeituraEntrada(BaseModel):
    """Telemetria enviada pelo ESP32."""

    device_id: str = Field(min_length=1, max_length=64,
                           examples=["esp32-horta-01"])
    umidade_m3m3: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Umidade volumetrica lida pelo sensor (m³/m³).",
    )
    umidade_mm: float | None = Field(
        default=None, ge=0.0, le=1000.0,
        description="Lamina ja convertida (mm). Alternativa a umidade_m3m3.",
    )
    temperatura_c: float | None = Field(default=None, ge=-40.0, le=85.0)
    bateria_v: float | None = Field(default=None, ge=0.0, le=20.0)
    rssi_dbm: int | None = Field(default=None, ge=-120, le=0)
    # Confirmacao do comando anterior: permite auditar bomba que nao ligou.
    comando_executado: str | None = Field(default=None, max_length=64)
    segundos_executados: float | None = Field(default=None, ge=0.0, le=7200.0)

    @field_validator("device_id")
    @classmethod
    def _id_seguro(cls, v: str) -> str:
        if not all(c.isalnum() or c in "-_." for c in v):
            raise ValueError("device_id aceita apenas letras, digitos, '-', '_' e '.'")
        return v

    @model_validator(mode="after")
    def _uma_das_duas(self) -> "LeituraEntrada":
        """Exige pelo menos uma das duas formas de umidade.

        Precisa ser `model_validator`: um `field_validator` em `umidade_mm`
        NAO roda quando o campo vem ausente, e o corpo sem nenhuma umidade
        passava. O `or 0.0` la embaixo entao leria 0 mm — "solo totalmente
        seco" — e o controlador abriria a valvula no maximo.
        """
        if self.umidade_m3m3 is None and self.umidade_mm is None:
            raise ValueError("informe umidade_m3m3 ou umidade_mm")
        return self


class LeituraResposta(BaseModel):
    ok: bool = True
    device_id: str
    umidade_mm: float
    recebido_em: float = Field(description="Carimbo do SERVIDOR (epoch).")
    proximo_poll_s: int


class ComandoResposta(BaseModel):
    """Resposta do GET: o que o ESP32 precisa para agir."""

    acionar: bool
    segundos: float = Field(description="Duracao do pulso da bomba.")
    duty: float = Field(ge=0.0, le=1.0, description="Equivalente PWM (0–1).")
    comando_id: str
    motivo: str
    proximo_poll_s: int
    # Contexto, util no serial do dispositivo e no diagnostico
    umidade_mm: float | None = None
    setpoint_mm: float | None = None
    erro_mm: float | None = None
    origem_setpoint: str = "indisponivel"
    degradado: bool = False


class StatusResposta(BaseModel):
    device_id: str
    online: bool
    modo: Literal["auto", "manual"]
    bomba_ligada: bool
    segundos_restantes: float
    ultima_leitura_s: float | None
    umidade_mm: float | None
    setpoint_mm: float | None
    origem_setpoint: str
    consumo_dia_mm: float
    cota_diaria_mm: float
    agua_em_transito_mm: float
    comando_pendente: str | None
    irrigacoes_hoje: int
    falhas_bomba: int
    capacidade_campo_mm: float
    ponto_murcha_mm: float
    ultimo_motivo: str


class ModoEntrada(BaseModel):
    modo: Literal["auto", "manual"]


class AcionamentoManual(BaseModel):
    """Pulso disparado pelo operador no painel.

    Manual dispensa a DECISAO do controlador, nunca os LIMITES: o pulso
    continua sujeito ao maximo do atuador, a cota diaria e a capacidade de
    campo. 'Manual' nao pode virar um atalho para encharcar a lavoura.
    """

    segundos: float = Field(gt=0.0, le=7200.0)


class PontoHistorico(BaseModel):
    t: float
    umidade_mm: float
    setpoint_mm: float | None = None
    irrigado_mm: float | None = None


class HistoricoResposta(BaseModel):
    device_id: str
    pontos: list[PontoHistorico]
    capacidade_campo_mm: float
    ponto_murcha_mm: float


# ---------------------------------------------------------------------------
# Estado por dispositivo
# ---------------------------------------------------------------------------


@dataclass
class ComandoPendente:
    comando_id: str
    segundos: float
    duty: float
    motivo: str
    emitido_em: float
    confirmado: bool = False


@dataclass
class EstadoDispositivo:
    device_id: str
    controlador: ControladorPID | ControladorFuzzy
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    umidade_mm: float | None = None
    recebido_em: float | None = None
    pendente: ComandoPendente | None = None
    irrigacoes: int = 0
    falhas_bomba: int = 0
    ultima_temperatura_c: float | None = None
    ultima_bateria_v: float | None = None
    # Modo de operacao. Vive no SERVIDOR, nao no painel: e o ESP32 que
    # consulta, entao um modo guardado so na sessao do dashboard deixaria a
    # bomba irrigando sozinha com a tela dizendo "Manual".
    modo: Literal["auto", "manual"] = "auto"
    ultimo_motivo: str = "sem_leitura_do_sensor"
    historico: deque[tuple[float, float, float | None, float | None]] = field(
        default_factory=lambda: deque(maxlen=CFG.historico_max)
    )

    def registrar(self, umidade_mm: float, setpoint_mm: float | None,
                  irrigado_mm: float | None = None) -> None:
        self.historico.append((time.time(), umidade_mm, setpoint_mm, irrigado_mm))

    @property
    def bomba_ligada(self) -> bool:
        """Ligada enquanto o pulso emitido ainda estiver correndo."""
        return self.segundos_restantes > 0.0

    @property
    def segundos_restantes(self) -> float:
        if self.pendente is None:
            return 0.0
        decorrido = time.monotonic() - self.pendente.emitido_em
        return max(0.0, self.pendente.segundos - decorrido)


class Registro:
    """Dispositivos conhecidos. Um controlador por dispositivo, pois o
    controle e estadual (integral, lockout, agua em transito)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._d: dict[str, EstadoDispositivo] = {}
        self._lock = asyncio.Lock()
        self.limites = LimitesSeguranca(
            capacidade_campo_mm=cfg.capacidade_campo_mm,
            ponto_murcha_mm=cfg.ponto_murcha_mm,
            lamina_diaria_max_mm=cfg.lamina_diaria_max_mm,
        )
        self.atuador = Atuador(
            vazao_lph=cfg.vazao_lph, area_m2=cfg.area_m2,
            eficiencia=cfg.eficiencia, pulso_min_s=cfg.pulso_min_s,
            pulso_max_s=cfg.pulso_max_s,
        )

    def _novo_controlador(self):
        if self.cfg.controlador == "fuzzy":
            return ControladorFuzzy(self.atuador, self.limites)
        return ControladorPID(self.atuador, self.limites, GanhosPID())

    async def obter(self, device_id: str) -> EstadoDispositivo:
        async with self._lock:
            est = self._d.get(device_id)
            if est is None:
                est = EstadoDispositivo(device_id, self._novo_controlador())
                self._d[device_id] = est
                LOG.info("dispositivo registrado: %s", device_id)
            return est

    def existente(self, device_id: str) -> EstadoDispositivo | None:
        return self._d.get(device_id)

    def todos(self) -> dict[str, EstadoDispositivo]:
        return dict(self._d)

    # -- persistencia do que nao pode ser perdido -------------------------

    def exportar(self) -> dict[str, Any]:
        """So o estado de SEGURANCA. Perder a cota diaria num reinicio em
        laco levaria a sobre-irrigacao."""
        return {
            "versao": 1,
            "salvo_em": time.time(),
            "monotonic": time.monotonic(),
            "dispositivos": {
                did: {
                    "consumo_dia_mm": e.controlador.consumo_dia_mm,
                    "irrigacoes": e.irrigacoes,
                    "falhas_bomba": e.falhas_bomba,
                    "modo": e.modo,
                }
                for did, e in self._d.items()
            },
        }

    async def importar(self, dados: dict[str, Any]) -> None:
        for did, d in (dados.get("dispositivos") or {}).items():
            est = await self.obter(did)
            est.irrigacoes = int(d.get("irrigacoes", 0))
            est.falhas_bomba = int(d.get("falhas_bomba", 0))
            # Modo sobrevive ao reinicio: quem deixou em manual (manutencao,
            # poda) nao quer o automatico voltando sozinho.
            if d.get("modo") in ("auto", "manual"):
                est.modo = d["modo"]
            consumo = float(d.get("consumo_dia_mm", 0.0))
            # Reaplica a cota ja gasta: conservador por construcao.
            est.controlador._consumo_dia_mm = consumo
            est.controlador._inicio_dia_s = time.monotonic()
            LOG.info("estado restaurado para %s (cota gasta %.2f mm)", did, consumo)


# ---------------------------------------------------------------------------
# Setpoint: o laco lento
# ---------------------------------------------------------------------------


@dataclass
class Setpoint:
    valor_mm: float
    origem: str          # "modelo" | "fallback" | "fallback_apos_erro"
    calculado_em: float
    detalhe: str = ""

    @property
    def degradado(self) -> bool:
        return self.origem != "modelo"


class ServicoSetpoint:
    """Calcula a meta. Usa o modelo quando da, cai para regra fixa quando nao.

    O modelo preve a agua na zona de raizes em t+24 h. Se ele antecipa QUEDA,
    a meta sobe agora, para criar folga antes da demanda chegar — e assim que
    a previsao vira acao. Se antecipa chuva, a meta nao sobe.

    O ajuste e multiplicado por `confianca_modelo` (padrao 0,5): com o skill
    medido no projeto (0,060 a +12 h e 0,089 a +24 h), apostar o acionamento
    inteiro na previsao nao se justifica.
    """

    def __init__(self, cfg: Config, limites: LimitesSeguranca) -> None:
        self.cfg = cfg
        self.limites = limites
        self._cache: Setpoint | None = None
        self._lock = asyncio.Lock()
        self._erros_seguidos = 0

    @property
    def base_mm(self) -> float:
        return setpoint_por_fracao_disponivel(self.limites, self.cfg.fracao_alvo)

    def _fallback(self, motivo: str) -> Setpoint:
        return Setpoint(self.base_mm, motivo, time.monotonic(),
                        f"{self.cfg.fracao_alvo:.0%} da agua disponivel")

    async def obter(self) -> Setpoint:
        agora = time.monotonic()
        if self._cache and (agora - self._cache.calculado_em) < self.cfg.ttl_setpoint_s:
            return self._cache

        async with self._lock:
            # Outra corrotina pode ter atualizado enquanto esperavamos.
            if self._cache and (time.monotonic() - self._cache.calculado_em) < self.cfg.ttl_setpoint_s:
                return self._cache

            if not self.cfg.bundle or not self.cfg.bundle.exists():
                self._cache = self._fallback("fallback")
                return self._cache

            try:
                sp = await run_in_threadpool(self._calcular_com_modelo)
                self._erros_seguidos = 0
            except Exception as exc:  # noqa: BLE001 — degradar, nunca derrubar
                self._erros_seguidos += 1
                LOG.warning("modelo indisponivel (%d seguidas): %s",
                            self._erros_seguidos, exc)
                sp = self._fallback("fallback_apos_erro")
                sp.detalhe = f"{type(exc).__name__}: {exc}"[:200]
            self._cache = sp
            return sp

    def _calcular_com_modelo(self) -> Setpoint:
        """Bloqueante: roda em threadpool. Importa pandas/sklearn aqui para
        que o servidor suba mesmo sem elas instaladas."""
        from treinar_modelo_solo import prever_com_bundle  # import tardio

        saida = prever_com_bundle(self.cfg.bundle, dias_contexto=30)
        if saida.empty:
            raise RuntimeError("previsao vazia")

        ultima = saida.iloc[-1]
        atual = float(ultima["agua_atual_mm"])
        previsto = float(ultima["agua_prevista_mm"])
        variacao = previsto - atual          # negativo = vai secar

        # So a queda prevista levanta a meta; alta prevista nao a reduz
        # abaixo da base, para nao deixar a planta descoberta por engano.
        folga = max(0.0, -variacao) * self.cfg.confianca_modelo
        alvo = min(self.base_mm + folga, self.limites.teto_mm)

        return Setpoint(
            valor_mm=alvo, origem="modelo", calculado_em=time.monotonic(),
            detalhe=(f"base {self.base_mm:.1f} + folga {folga:.1f} "
                     f"(variacao prevista {variacao:+.2f} mm)"),
        )


# ---------------------------------------------------------------------------
# Autenticacao
# ---------------------------------------------------------------------------


async def exigir_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Chave compartilhada. O endpoint comanda uma BOMBA: sem nenhuma
    barreira, qualquer um na rede aciona a irrigacao."""
    if CFG.api_key is None:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, CFG.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key ausente ou invalida",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# Aplicacao
# ---------------------------------------------------------------------------

registro = Registro(CFG)
servico_setpoint = ServicoSetpoint(CFG, registro.limites)


async def _salvar_estado() -> None:
    try:
        tmp = CFG.estado_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(registro.exportar(), indent=2), encoding="utf-8")
        tmp.replace(CFG.estado_path)    # troca atomica
    except OSError as exc:
        LOG.warning("nao foi possivel salvar o estado: %s", exc)


async def _carregar_estado() -> None:
    if not CFG.estado_path.exists():
        return
    try:
        dados = json.loads(CFG.estado_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("estado anterior ilegivel, comecando limpo: %s", exc)
        return
    # Estado de mais de 24 h ja nao diz nada sobre a cota de hoje.
    if time.time() - float(dados.get("salvo_em", 0)) > 86400:
        LOG.info("estado anterior tem mais de 24 h; ignorado")
        return
    await registro.importar(dados)


async def _autosave(intervalo_s: float = 60.0) -> None:
    while True:
        await asyncio.sleep(intervalo_s)
        await _salvar_estado()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if CFG.api_key is None:
        LOG.warning(
            "IRRIGACAO_API_KEY nao definida: a API esta ABERTA. "
            "Qualquer um na rede pode acionar a bomba."
        )
    if CFG.bundle and CFG.bundle.exists():
        LOG.info("modelo configurado: %s", CFG.bundle)
    else:
        LOG.info("sem modelo; setpoint fixo em %.0f%% da agua disponivel",
                 CFG.fracao_alvo * 100)

    await _carregar_estado()
    tarefa = asyncio.create_task(_autosave())
    try:
        yield
    finally:
        tarefa.cancel()
        await _salvar_estado()


app = FastAPI(
    title="Irrigacao — backend ESP32",
    description="Telemetria do sensor e comando da bomba, com setpoint do "
                "modelo preditivo e controle PID/Fuzzy.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def _falha_generica(request: Request, exc: Exception) -> JSONResponse:
    """Nenhuma excecao pode virar 'bomba ligada'. Em duvida, nao irriga."""
    LOG.exception("erro nao tratado em %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"acionar": False, "segundos": 0.0,
                 "motivo": "erro_interno_do_servidor"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/leitura", response_model=LeituraResposta,
          dependencies=[Depends(exigir_api_key)], tags=["dispositivo"])
async def receber_leitura(leitura: LeituraEntrada) -> LeituraResposta:
    """Recebe a leitura do sensor do ESP32.

    Converte m³/m³ para mm quando necessario, carimba com a hora do SERVIDOR
    e, se o corpo trouxer `comando_executado`, fecha o ciclo do comando
    anterior — e assim que uma bomba que nao ligou fica visivel.
    """
    est = await registro.obter(leitura.device_id)
    # Setpoint vem do cache (TTL de ~1 h), entao isto nao custa nada e deixa
    # a serie do painel CONTINUA: registrado so nas irrigacoes, ele viraria
    # um punhado de pontos soltos e nenhuma linha.
    sp = await servico_setpoint.obter()

    # Sem `or 0.0`: ausencia de leitura NAO pode virar "0 mm, solo seco".
    if leitura.umidade_mm is not None:
        umidade_mm = leitura.umidade_mm
    elif leitura.umidade_m3m3 is not None:
        umidade_mm = vwc_para_mm(leitura.umidade_m3m3, CFG.profundidade_mm)
    else:  # pragma: no cover — o model_validator ja barrou
        raise HTTPException(status_code=422, detail="leitura de umidade ausente")

    # Arredonda na borda: 1e-3 mm e muito abaixo da resolucao de qualquer
    # sensor, e evita 81.19999999999999 no JSON que o ESP32 vai parsear.
    umidade_mm = round(umidade_mm, 3)

    async with est.lock:
        est.umidade_mm = umidade_mm
        est.recebido_em = time.monotonic()
        est.ultima_temperatura_c = leitura.temperatura_c
        est.ultima_bateria_v = leitura.bateria_v

        if leitura.comando_executado and est.pendente:
            if leitura.comando_executado == est.pendente.comando_id:
                est.pendente.confirmado = True
                executado = leitura.segundos_executados
                if executado is not None and executado < est.pendente.segundos * 0.5:
                    # A bomba ligou por bem menos tempo que o mandado:
                    # valvula travada, falta de pressao ou queda de energia.
                    est.falhas_bomba += 1
                    LOG.warning(
                        "%s: comando %s pedia %.0f s, executou %.0f s",
                        est.device_id, est.pendente.comando_id,
                        est.pendente.segundos, executado,
                    )
                est.pendente = None

        est.registrar(umidade_mm, sp.valor_mm)

    proximo = (CFG.poll_irrigando_s if est.pendente else CFG.poll_normal_s)
    return LeituraResposta(
        device_id=leitura.device_id, umidade_mm=round(umidade_mm, 3),
        recebido_em=time.time(), proximo_poll_s=proximo,
    )


@app.get("/api/v1/comando/{device_id}", response_model=ComandoResposta,
         dependencies=[Depends(exigir_api_key)], tags=["dispositivo"])
async def obter_comando(device_id: str) -> ComandoResposta:
    """Diz se a bomba liga e por quantos segundos.

    Idempotente dentro de `validade_comando_s`: repetir a requisicao devolve
    o MESMO comando, nao uma segunda dose. Em qualquer duvida — leitura
    ausente, leitura velha, erro interno — a resposta e nao acionar.
    """
    est = registro.existente(device_id)
    if est is None or est.umidade_mm is None or est.recebido_em is None:
        return ComandoResposta(
            acionar=False, segundos=0.0, duty=0.0,
            comando_id="-", motivo="sem_leitura_do_sensor",
            proximo_poll_s=CFG.poll_normal_s,
        )

    sp = await servico_setpoint.obter()

    async with est.lock:
        idade = time.monotonic() - est.recebido_em

        # Leitura velha: o sensor pode ter caido. Irrigar as cegas e o
        # unico erro verdadeiramente caro deste sistema.
        if idade > CFG.leitura_velha_s:
            return ComandoResposta(
                acionar=False, segundos=0.0, duty=0.0, comando_id="-",
                motivo="leitura_desatualizada", umidade_mm=est.umidade_mm,
                setpoint_mm=round(sp.valor_mm, 2),
                origem_setpoint=sp.origem, degradado=True,
                proximo_poll_s=CFG.poll_normal_s,
            )

        # Comando ainda valido e nao confirmado: reenvia o mesmo.
        if est.pendente and not est.pendente.confirmado:
            if (time.monotonic() - est.pendente.emitido_em) < CFG.validade_comando_s:
                p = est.pendente
                return ComandoResposta(
                    acionar=True, segundos=p.segundos, duty=p.duty,
                    comando_id=p.comando_id, motivo=p.motivo,
                    umidade_mm=est.umidade_mm,
                    setpoint_mm=round(sp.valor_mm, 2),
                    origem_setpoint=sp.origem, degradado=sp.degradado,
                    proximo_poll_s=CFG.poll_irrigando_s,
                )
            # Expirou sem confirmacao: o dispositivo provavelmente caiu.
            est.falhas_bomba += 1
            LOG.warning("%s: comando %s expirou sem confirmacao",
                        device_id, est.pendente.comando_id)
            est.pendente = None

        # Em modo manual o controlador NAO decide. Qualquer pulso vem de um
        # acionamento explicito do operador, que ja virou comando pendente e
        # foi tratado acima.
        if est.modo == "manual":
            est.ultimo_motivo = "modo_manual"
            return ComandoResposta(
                acionar=False, segundos=0.0, duty=0.0, comando_id="-",
                motivo="modo_manual", umidade_mm=est.umidade_mm,
                setpoint_mm=round(sp.valor_mm, 2),
                origem_setpoint=sp.origem, degradado=sp.degradado,
                proximo_poll_s=CFG.poll_irrigando_s,
            )

        decisao: Decisao = est.controlador.atualizar(
            umidade_mm=est.umidade_mm,
            setpoint_mm=sp.valor_mm,
            instante_s=time.monotonic(),
        )

        est.ultimo_motivo = decisao.motivo.value
        if not decisao.irrigar:
            return ComandoResposta(
                acionar=False, segundos=0.0, duty=0.0, comando_id="-",
                motivo=decisao.motivo.value, umidade_mm=est.umidade_mm,
                setpoint_mm=round(sp.valor_mm, 2), erro_mm=round(decisao.erro_mm, 3),
                origem_setpoint=sp.origem, degradado=sp.degradado,
                proximo_poll_s=CFG.poll_normal_s,
            )

        cid = uuid.uuid4().hex[:12]
        est.pendente = ComandoPendente(
            comando_id=cid, segundos=round(decisao.pulso_s, 1),
            duty=round(decisao.duty, 3), motivo=decisao.motivo.value,
            emitido_em=time.monotonic(),
        )
        est.irrigacoes += 1
        est.registrar(est.umidade_mm, sp.valor_mm, decisao.lamina_mm)
        LOG.info("%s: acionar %.0f s (%.2f mm), comando %s",
                 device_id, decisao.pulso_s, decisao.lamina_mm, cid)

        return ComandoResposta(
            acionar=True, segundos=round(decisao.pulso_s, 1),
            duty=round(decisao.duty, 3), comando_id=cid,
            motivo=decisao.motivo.value, umidade_mm=est.umidade_mm,
            setpoint_mm=round(sp.valor_mm, 2), erro_mm=round(decisao.erro_mm, 3),
            origem_setpoint=sp.origem, degradado=sp.degradado,
            proximo_poll_s=CFG.poll_irrigando_s,
        )


@app.get("/api/v1/status/{device_id}", response_model=StatusResposta,
         dependencies=[Depends(exigir_api_key)], tags=["operacao"])
async def status_dispositivo(device_id: str) -> StatusResposta:
    """Diagnostico: o que o controlador esta vendo e decidindo."""
    est = registro.existente(device_id)
    if est is None:
        raise HTTPException(status_code=404, detail="dispositivo desconhecido")

    sp = await servico_setpoint.obter()
    idade = (time.monotonic() - est.recebido_em) if est.recebido_em else None
    return StatusResposta(
        device_id=device_id,
        online=idade is not None and idade <= CFG.leitura_velha_s,
        modo=est.modo,
        bomba_ligada=est.bomba_ligada,
        segundos_restantes=round(est.segundos_restantes, 1),
        ultima_leitura_s=round(idade, 1) if idade is not None else None,
        umidade_mm=est.umidade_mm,
        setpoint_mm=round(sp.valor_mm, 2),
        origem_setpoint=sp.origem,
        consumo_dia_mm=round(est.controlador.consumo_dia_mm, 3),
        cota_diaria_mm=registro.limites.lamina_diaria_max_mm,
        agua_em_transito_mm=round(est.controlador.agua_em_transito_mm, 3),
        comando_pendente=est.pendente.comando_id if est.pendente else None,
        irrigacoes_hoje=est.irrigacoes,
        falhas_bomba=est.falhas_bomba,
        capacidade_campo_mm=registro.limites.capacidade_campo_mm,
        ponto_murcha_mm=registro.limites.ponto_murcha_mm,
        ultimo_motivo=est.ultimo_motivo,
    )


@app.post("/api/v1/modo/{device_id}", response_model=StatusResposta,
          dependencies=[Depends(exigir_api_key)], tags=["operacao"])
async def definir_modo(device_id: str, entrada: ModoEntrada) -> StatusResposta:
    """Alterna entre automatico e manual.

    O modo vive no SERVIDOR porque e ele que o ESP32 consulta. Ao entrar em
    manual, qualquer comando pendente e cancelado: quem assumiu o controle
    nao quer uma decisao antiga chegando depois.
    """
    est = await registro.obter(device_id)
    async with est.lock:
        anterior = est.modo
        est.modo = entrada.modo
        if entrada.modo == "manual" and est.pendente is not None:
            LOG.info("%s: modo manual cancela o comando pendente %s",
                     device_id, est.pendente.comando_id)
            est.pendente = None
        est.ultimo_motivo = f"modo_alterado_para_{entrada.modo}"
    LOG.info("%s: modo %s -> %s", device_id, anterior, entrada.modo)
    await _salvar_estado()
    return await status_dispositivo(device_id)


@app.post("/api/v1/acionar/{device_id}", response_model=ComandoResposta,
          dependencies=[Depends(exigir_api_key)], tags=["operacao"])
async def acionar_manual(device_id: str,
                         pedido: AcionamentoManual) -> ComandoResposta:
    """Dispara um pulso manual, so em modo manual.

    Dispensa a DECISAO do controlador, nunca os LIMITES. O pulso e cortado
    pelo maximo do atuador, pela cota diaria restante e pelo espaco ate a
    capacidade de campo — 'manual' nao e atalho para encharcar a lavoura.
    """
    est = registro.existente(device_id)
    if est is None:
        raise HTTPException(status_code=404, detail="dispositivo desconhecido")
    if est.modo != "manual":
        raise HTTPException(
            status_code=409,
            detail="acionamento manual exige o modo manual; alterne primeiro",
        )
    if est.umidade_mm is None:
        raise HTTPException(status_code=409, detail="sem leitura do sensor")

    atuador, limites = registro.atuador, registro.limites
    sp = await servico_setpoint.obter()

    async with est.lock:
        if est.pendente is not None and est.segundos_restantes > 0:
            raise HTTPException(status_code=409,
                                detail="ja existe um pulso em andamento")

        segundos = min(pedido.segundos, atuador.pulso_max_s)
        lamina = atuador.lamina_de(segundos)

        restante = limites.lamina_diaria_max_mm - est.controlador.consumo_dia_mm
        espaco = max(0.0, limites.teto_mm - est.umidade_mm
                     - est.controlador.agua_em_transito_mm)
        limite_mm = min(restante, espaco)
        if limite_mm <= 0:
            motivo = ("cota_diaria_esgotada" if restante <= 0
                      else "limitado_pela_capacidade_de_campo")
            raise HTTPException(status_code=409, detail=motivo)

        if lamina > limite_mm:
            lamina = limite_mm
            segundos = atuador.segundos_para(lamina)
            LOG.info("%s: pulso manual reduzido para %.0f s pelos limites",
                     device_id, segundos)

        if segundos < atuador.pulso_min_s:
            raise HTTPException(
                status_code=409,
                detail=f"pulso resultante ({segundos:.0f} s) abaixo do minimo "
                       f"do atuador ({atuador.pulso_min_s:.0f} s)",
            )

        cid = uuid.uuid4().hex[:12]
        est.pendente = ComandoPendente(
            comando_id=cid, segundos=round(segundos, 1),
            duty=round(atuador.duty(segundos), 3), motivo="manual",
            emitido_em=time.monotonic(),
        )
        est.irrigacoes += 1
        est.ultimo_motivo = "manual"
        # Debita no MESMO contador do automatico: senao o manual furaria a
        # cota diaria por fora e o automatico nao saberia.
        est.controlador._consumo_dia_mm += lamina
        est.controlador._transito_mm += lamina
        est.controlador._t_transito = time.monotonic()
        est.controlador._t_ultima_irrigacao = time.monotonic()
        est.registrar(est.umidade_mm, sp.valor_mm, lamina)

    LOG.info("%s: pulso MANUAL de %.0f s (%.2f mm), comando %s",
             device_id, segundos, lamina, cid)
    return ComandoResposta(
        acionar=True, segundos=round(segundos, 1),
        duty=round(atuador.duty(segundos), 3), comando_id=cid,
        motivo="manual", umidade_mm=est.umidade_mm,
        setpoint_mm=round(sp.valor_mm, 2), origem_setpoint=sp.origem,
        degradado=sp.degradado, proximo_poll_s=CFG.poll_irrigando_s,
    )


@app.get("/api/v1/historico/{device_id}", response_model=HistoricoResposta,
         dependencies=[Depends(exigir_api_key)], tags=["operacao"])
async def historico(device_id: str, horas: float = 24.0) -> HistoricoResposta:
    """Serie de umidade para os graficos do painel."""
    est = registro.existente(device_id)
    if est is None:
        raise HTTPException(status_code=404, detail="dispositivo desconhecido")

    corte = time.time() - max(0.1, horas) * 3600.0
    pontos = [
        PontoHistorico(t=t, umidade_mm=u, setpoint_mm=s, irrigado_mm=i)
        for (t, u, s, i) in est.historico if t >= corte
    ]
    return HistoricoResposta(
        device_id=device_id, pontos=pontos,
        capacidade_campo_mm=registro.limites.capacidade_campo_mm,
        ponto_murcha_mm=registro.limites.ponto_murcha_mm,
    )


@app.get("/health", tags=["operacao"])
async def health() -> dict[str, Any]:
    """Sem autenticacao, para o monitor de disponibilidade."""
    sp = await servico_setpoint.obter()
    return {
        "ok": True,
        "controlador": CFG.controlador,
        "dispositivos": len(registro.todos()),
        "setpoint_mm": round(sp.valor_mm, 2),
        "origem_setpoint": sp.origem,
        "modelo_carregado": bool(CFG.bundle and CFG.bundle.exists()),
        "autenticado": CFG.api_key is not None,
    }
