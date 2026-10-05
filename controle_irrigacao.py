#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controle de acionamento do atuador de irrigacao.

Recebe a umidade lida pelo sensor e o setpoint dinamico (a meta calculada
pelo modelo de `treinar_modelo_solo.py`) e devolve a duracao exata do pulso
de acionamento, em segundos e em duty cycle PWM.

Duas implementacoes do mesmo contrato (`Controlador`):

  * `ControladorPID`   — PID com termos adaptados ao processo (abaixo).
  * `ControladorFuzzy` — Mamdani com 15 regras, para comparacao.

Por que um PID de livro-texto falha aqui
----------------------------------------
1. **O solo ja e um integrador.** A agua aplicada se acumula; a planta nao
   e um processo auto-regulado. Acao integral forte sobre um processo
   integrador produz oscilacao, entao `ki` entra pequeno e so para corrigir
   erro sistematico (emissor entupido, calibracao de vazao errada).

2. **Tempo morto grande.** Infiltracao e redistribuicao levam de minutos a
   horas ate o sensor registrar a agua. Um controlador que ignora isso
   segue acionando enquanto espera e encharca. Tratado por dois mecanismos:
   a trava pos-irrigacao (`lockout_s`) e o termo de agua em transito, que
   desconta do erro o que ja foi aplicado mas ainda nao foi lido — um
   preditor de Smith simplificado.

3. **Atuador unidirecional.** Da para adicionar agua, nao para remover.
   Saida negativa nao existe, a saturacao em zero e permanente enquanto o
   solo estiver acima da meta, e por isso o anti-windup e obrigatorio.

4. **Setpoint movel.** A meta vem do modelo e muda sozinha. A derivada e
   calculada sobre a MEDIDA, nunca sobre o erro, senao cada atualizacao do
   setpoint produz um pico de derivada (derivative kick).

Unidades
--------
Tudo em milimetros de lamina de agua, a mesma unidade da ETo e da chuva no
resto do projeto (1 mm = 1 L/m²). A conversao de lamina para tempo fica
isolada em `Atuador`, que guarda a calibracao hidraulica. Assim o
controlador nao sabe nada de vazao, e trocar o emissor nao mexe no ganho.

Uso
---
    from controle_irrigacao import (
        Atuador, ControladorPID, GanhosPID, LimitesSeguranca, vwc_para_mm,
    )

    atuador = Atuador(vazao_lph=240.0, area_m2=100.0, eficiencia=0.85)
    ctrl = ControladorPID(
        atuador=atuador,
        limites=LimitesSeguranca(capacidade_campo_mm=122.6,
                                 ponto_murcha_mm=71.8,
                                 lamina_diaria_max_mm=12.0),
    )

    d = ctrl.atualizar(umidade_mm=88.0, setpoint_mm=96.0, instante_s=t)
    if d.irrigar:
        acionar_valvula(segundos=d.pulso_s)     # ou PWM: d.duty
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Sequence

__all__ = [
    "Atuador",
    "GanhosPID",
    "LimitesSeguranca",
    "Decisao",
    "Motivo",
    "Controlador",
    "ControladorPID",
    "ControladorFuzzy",
    "vwc_para_mm",
    "mm_para_vwc",
]


# ---------------------------------------------------------------------------
# Conversoes de unidade do sensor
# ---------------------------------------------------------------------------


def vwc_para_mm(theta_m3m3: float, profundidade_mm: float) -> float:
    """Umidade volumetrica (m³/m³) -> lamina armazenada (mm).

    A maioria dos sensores capacitivos/TDR entrega m³/m³. O resto do
    projeto trabalha em mm, entao a conversao acontece na borda.
    """
    if profundidade_mm <= 0:
        raise ValueError("profundidade deve ser positiva")
    return float(theta_m3m3) * float(profundidade_mm)


def mm_para_vwc(lamina_mm: float, profundidade_mm: float) -> float:
    """Inverso de `vwc_para_mm`."""
    if profundidade_mm <= 0:
        raise ValueError("profundidade deve ser positiva")
    return float(lamina_mm) / float(profundidade_mm)


# ---------------------------------------------------------------------------
# Atuador: a unica parte que conhece hidraulica
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atuador:
    """Calibracao hidraulica: converte lamina (mm) em tempo de acionamento.

    `vazao_lph` e a vazao TOTAL de todos os emissores da zona, e `area_m2` a
    area molhada. Como 1 L/m² = 1 mm, a taxa de aplicacao sai direto da
    divisao — nao ha constante magica no caminho.

    `eficiencia` desconta o que nao chega a zona de raizes: deriva de
    aspersao, evaporacao do jato, escoamento superficial. Gotejamento fica
    perto de 0,9; aspersao convencional, de 0,7.
    """

    vazao_lph: float
    area_m2: float
    eficiencia: float = 0.85
    pulso_min_s: float = 20.0       # abaixo disso a valvula nem assenta
    pulso_max_s: float = 1800.0     # teto por acionamento (30 min)
    periodo_pwm_s: float = 60.0     # janela para o duty cycle

    def __post_init__(self) -> None:
        if self.vazao_lph <= 0:
            raise ValueError("vazao_lph deve ser positiva")
        if self.area_m2 <= 0:
            raise ValueError("area_m2 deve ser positiva")
        if not 0.0 < self.eficiencia <= 1.0:
            raise ValueError("eficiencia deve estar em (0, 1]")
        if self.pulso_min_s < 0 or self.pulso_max_s <= 0:
            raise ValueError("limites de pulso devem ser positivos")
        if self.pulso_min_s > self.pulso_max_s:
            raise ValueError(
                f"pulso_min_s ({self.pulso_min_s}) acima de pulso_max_s "
                f"({self.pulso_max_s})"
            )
        if self.periodo_pwm_s <= 0:
            raise ValueError("periodo_pwm_s deve ser positivo")

    @property
    def taxa_bruta_mm_h(self) -> float:
        """Lamina por hora antes das perdas (L/h / m² = mm/h)."""
        return self.vazao_lph / self.area_m2

    @property
    def taxa_efetiva_mm_h(self) -> float:
        """Lamina que de fato chega a zona de raizes, por hora."""
        return self.taxa_bruta_mm_h * self.eficiencia

    def segundos_para(self, lamina_mm: float) -> float:
        """Tempo de valvula aberta para entregar `lamina_mm` na raiz."""
        if lamina_mm <= 0:
            return 0.0
        return 3600.0 * lamina_mm / self.taxa_efetiva_mm_h

    def lamina_de(self, segundos: float) -> float:
        """Inverso de `segundos_para` — util para auditoria e simulacao."""
        if segundos <= 0:
            return 0.0
        return self.taxa_efetiva_mm_h * segundos / 3600.0

    @property
    def lamina_min_mm(self) -> float:
        """Menor lamina que o atuador consegue entregar (pulso minimo)."""
        return self.lamina_de(self.pulso_min_s)

    @property
    def lamina_max_mm(self) -> float:
        """Maior lamina por acionamento."""
        return self.lamina_de(self.pulso_max_s)

    def duty(self, pulso_s: float) -> float:
        """Duty cycle equivalente (0–1) na janela `periodo_pwm_s`.

        Para valvula solenoide simples use `pulso_s` direto; o duty serve
        para valvula proporcional ou para bomba com acionamento PWM.
        """
        return min(1.0, max(0.0, pulso_s / self.periodo_pwm_s))


# ---------------------------------------------------------------------------
# Ganhos e limites
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GanhosPID:
    """Ganhos do PID. O erro entra em mm e a saida sai em mm.

    * `kp` e ADIMENSIONAL: fracao do deficit atendida em um acionamento.
      0,7 significa "reponha 70% do que falta agora" — o resto vem nos
      ciclos seguintes, depois que o sensor confirmar o efeito. Deixar em
      1,0 ignora o tempo morto e tende a encharcar.
    * `ki` em 1/s: so existe para zerar erro sistematico (vazao real abaixo
      da nominal, emissor entupido). Como o solo ja integra, valor alto
      produz oscilacao de periodo longo.
    * `kd` em s: amortece quando a umidade sobe rapido — tipicamente por
      chuva que o controlador ainda nao viu. Calculado sobre a MEDIDA.
    """

    kp: float = 0.7
    ki: float = 1.0e-6
    kd: float = 120.0

    def __post_init__(self) -> None:
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("ganhos nao podem ser negativos")
        if self.kp == 0 and self.ki == 0:
            raise ValueError("kp e ki nulos: o controlador nunca irrigaria")


@dataclass(frozen=True)
class LimitesSeguranca:
    """Restricoes agronomicas e de equipamento.

    O teto em capacidade de campo e o mais importante: agua acima dela nao
    fica retida, percola e leva nutriente junto. Irrigar alem disso gasta
    agua e empobrece o solo.
    """

    capacidade_campo_mm: float
    ponto_murcha_mm: float
    lamina_diaria_max_mm: float = 15.0
    folga_capacidade_mm: float = 2.0    # margem para nao encostar na cc
    banda_morta_mm: float = 1.5         # evita liga-desliga perto da meta
    lockout_s: float = 1800.0           # espera pos-irrigacao (30 min)
    tau_transito_s: float = 2700.0      # constante de tempo da infiltracao

    def __post_init__(self) -> None:
        if self.capacidade_campo_mm <= self.ponto_murcha_mm:
            raise ValueError(
                f"capacidade de campo ({self.capacidade_campo_mm}) deve ser "
                f"maior que o ponto de murcha ({self.ponto_murcha_mm})"
            )
        for nome in ("lamina_diaria_max_mm", "folga_capacidade_mm",
                     "banda_morta_mm", "lockout_s", "tau_transito_s"):
            if getattr(self, nome) < 0:
                raise ValueError(f"{nome} nao pode ser negativo")
        if self.tau_transito_s == 0:
            raise ValueError("tau_transito_s deve ser positivo")

    @property
    def agua_disponivel_mm(self) -> float:
        """Agua util total entre murcha e capacidade de campo."""
        return self.capacidade_campo_mm - self.ponto_murcha_mm

    @property
    def teto_mm(self) -> float:
        """Nivel maximo que o controlador pode mirar."""
        return self.capacidade_campo_mm - self.folga_capacidade_mm


# ---------------------------------------------------------------------------
# Saida
# ---------------------------------------------------------------------------


class Motivo(str, Enum):
    """Por que a decisao foi essa. Controlador sem observabilidade nao se
    depura em campo."""

    IRRIGAR = "irrigar"
    DENTRO_DA_BANDA = "dentro_da_banda_morta"
    ACIMA_DO_SETPOINT = "acima_do_setpoint"
    EM_LOCKOUT = "em_lockout_pos_irrigacao"
    AGUA_EM_TRANSITO = "agua_em_transito_cobre_o_deficit"
    COTA_DIARIA = "cota_diaria_esgotada"
    PULSO_ABAIXO_DO_MINIMO = "pulso_abaixo_do_minimo_do_atuador"
    TETO_CAPACIDADE = "limitado_pela_capacidade_de_campo"
    PRIMEIRA_LEITURA = "primeira_leitura_sem_dt"
    SENSOR_INVALIDO = "leitura_de_sensor_invalida"


@dataclass(frozen=True)
class Decisao:
    """Resultado de um ciclo de controle."""

    irrigar: bool
    pulso_s: float
    duty: float
    lamina_mm: float
    erro_mm: float
    motivo: Motivo
    instante_s: float
    # Componentes, para diagnostico e sintonia
    p_mm: float = 0.0
    i_mm: float = 0.0
    d_mm: float = 0.0
    ff_mm: float = 0.0
    saturou: bool = False
    transito_mm: float = 0.0
    consumo_dia_mm: float = 0.0

    def __str__(self) -> str:
        if not self.irrigar:
            return f"nao irrigar ({self.motivo.value}); erro {self.erro_mm:+.2f} mm"
        return (
            f"irrigar {self.pulso_s:.0f} s (duty {self.duty:.2f}) = "
            f"{self.lamina_mm:.2f} mm; erro {self.erro_mm:+.2f} mm "
            f"[P {self.p_mm:+.2f} I {self.i_mm:+.2f} D {self.d_mm:+.2f} "
            f"FF {self.ff_mm:+.2f}]"
        )


# ---------------------------------------------------------------------------
# Contrato comum
# ---------------------------------------------------------------------------


class Controlador(ABC):
    """Interface comum ao PID e ao Fuzzy.

    A logica de seguranca (banda morta, lockout, transito, cota diaria,
    teto de capacidade, pulso minimo) vive aqui e vale para os dois: so o
    calculo da lamina bruta muda entre eles.
    """

    def __init__(
        self,
        atuador: Atuador,
        limites: LimitesSeguranca,
    ) -> None:
        self.atuador = atuador
        self.limites = limites
        self.reiniciar()

    # -- estado ----------------------------------------------------------

    def reiniciar(self) -> None:
        """Zera o estado interno. Chame ao trocar de zona ou apos falha."""
        self._t_anterior: float | None = None
        self._medida_anterior: float | None = None
        self._integral_mm_s: float = 0.0
        self._t_ultima_irrigacao: float | None = None
        self._transito_mm: float = 0.0
        self._t_transito: float | None = None
        self._consumo_dia_mm: float = 0.0
        self._inicio_dia_s: float | None = None

    @property
    def consumo_dia_mm(self) -> float:
        return self._consumo_dia_mm

    @property
    def agua_em_transito_mm(self) -> float:
        return self._transito_mm

    # -- a conta especifica de cada controlador --------------------------

    @abstractmethod
    def _lamina_bruta(
        self, erro_mm: float, medida_mm: float, dt_s: float,
        deriva_medida_mm_s: float, feedforward_mm: float,
    ) -> tuple[float, dict[str, float]]:
        """Lamina desejada (mm) e os componentes, para diagnostico."""

    # -- ciclo de controle -----------------------------------------------

    def atualizar(
        self,
        umidade_mm: float,
        setpoint_mm: float,
        instante_s: float | None = None,
        feedforward_mm: float = 0.0,
    ) -> Decisao:
        """Executa um ciclo e devolve o acionamento.

        `umidade_mm`     leitura do sensor, ja convertida para mm
        `setpoint_mm`    meta vinda do modelo (sera limitada ao teto)
        `instante_s`     relogio monotonico; `time.monotonic()` por padrao
        `feedforward_mm` demanda prevista (ETo das proximas horas menos a
                         chuva prevista). Antecipa a perda em vez de
                         esperar o sensor acusar — e o que transforma o
                         laco de reativo em preditivo.
        """
        agora = time.monotonic() if instante_s is None else float(instante_s)

        # --- validacao da leitura ---------------------------------------
        if not math.isfinite(umidade_mm) or not math.isfinite(setpoint_mm):
            return self._negar(Motivo.SENSOR_INVALIDO, float("nan"), agora)
        if umidade_mm < 0:
            return self._negar(Motivo.SENSOR_INVALIDO, float("nan"), agora)

        self._rolar_dia(agora)
        self._decair_transito(agora)

        # O setpoint vem de um modelo e pode pedir mais do que o solo
        # retem: limitar aqui evita percolacao e lixiviacao.
        setpoint_efetivo = min(setpoint_mm, self.limites.teto_mm)
        limitado_pelo_teto = setpoint_efetivo < setpoint_mm

        # Erro visto pelo controlador: desconta a agua ja aplicada que o
        # sensor ainda nao registrou.
        erro_bruto = setpoint_efetivo - umidade_mm
        erro_mm = erro_bruto - self._transito_mm

        # --- dt e derivada da medida ------------------------------------
        if self._t_anterior is None:
            self._t_anterior = agora
            self._medida_anterior = umidade_mm
            return self._negar(Motivo.PRIMEIRA_LEITURA, erro_mm, agora)

        dt_s = agora - self._t_anterior
        if dt_s <= 0:
            # Relogio parado ou leitura repetida: nao integra nem deriva.
            return self._negar(Motivo.PRIMEIRA_LEITURA, erro_mm, agora)

        deriva = (umidade_mm - (self._medida_anterior or umidade_mm)) / dt_s
        self._t_anterior = agora
        self._medida_anterior = umidade_mm

        # --- bloqueios que dispensam o calculo --------------------------
        if erro_bruto <= 0:
            self._integral_mm_s = 0.0      # solo acima da meta: zera o I
            return self._negar(Motivo.ACIMA_DO_SETPOINT, erro_mm, agora)

        if self._em_lockout(agora):
            return self._negar(Motivo.EM_LOCKOUT, erro_mm, agora)

        if erro_mm <= 0:
            # O deficit ja esta coberto pelo que esta infiltrando.
            return self._negar(Motivo.AGUA_EM_TRANSITO, erro_mm, agora)

        if erro_mm < self.limites.banda_morta_mm:
            return self._negar(Motivo.DENTRO_DA_BANDA, erro_mm, agora)

        restante_dia = self.limites.lamina_diaria_max_mm - self._consumo_dia_mm
        if restante_dia <= 0:
            return self._negar(Motivo.COTA_DIARIA, erro_mm, agora)

        # --- a conta do controlador -------------------------------------
        lamina, comp = self._lamina_bruta(
            erro_mm, umidade_mm, dt_s, deriva, feedforward_mm
        )

        # --- limites, do mais restritivo ao menos -----------------------
        lamina = max(0.0, lamina)
        # Nunca passar do teto de capacidade de campo.
        espaco_mm = max(0.0, self.limites.teto_mm - umidade_mm - self._transito_mm)
        if lamina > espaco_mm:
            lamina = espaco_mm
            limitado_pelo_teto = True
        lamina = min(lamina, restante_dia, self.atuador.lamina_max_mm)

        pulso_s = self.atuador.segundos_para(lamina)
        saturou = pulso_s >= self.atuador.pulso_max_s - 1e-9

        # Pulso curto demais: a valvula nao assenta e o acionamento so
        # desgasta o equipamento. Melhor esperar o deficit crescer.
        if pulso_s < self.atuador.pulso_min_s:
            motivo = (Motivo.TETO_CAPACIDADE if limitado_pelo_teto and lamina <= 0
                      else Motivo.PULSO_ABAIXO_DO_MINIMO)
            return self._negar(motivo, erro_mm, agora, comp=comp)

        pulso_s = min(pulso_s, self.atuador.pulso_max_s)
        lamina = self.atuador.lamina_de(pulso_s)

        # --- registra o que foi aplicado --------------------------------
        self._t_ultima_irrigacao = agora
        self._consumo_dia_mm += lamina
        self._transito_mm += lamina
        self._t_transito = agora

        # Anti-windup: so acumula integral quando NAO saturou. Com atuador
        # unidirecional e tempo morto longo, integrar sob saturacao e o
        # caminho mais rapido para o encharcamento.
        if not saturou:
            self._integral_mm_s += erro_mm * dt_s

        return Decisao(
            irrigar=True,
            pulso_s=pulso_s,
            duty=self.atuador.duty(pulso_s),
            lamina_mm=lamina,
            erro_mm=erro_mm,
            motivo=Motivo.IRRIGAR,
            instante_s=agora,
            p_mm=comp.get("p", 0.0),
            i_mm=comp.get("i", 0.0),
            d_mm=comp.get("d", 0.0),
            ff_mm=comp.get("ff", 0.0),
            saturou=saturou,
            transito_mm=self._transito_mm,
            consumo_dia_mm=self._consumo_dia_mm,
        )

    # -- auxiliares ------------------------------------------------------

    def _negar(
        self, motivo: Motivo, erro_mm: float, agora: float,
        comp: dict[str, float] | None = None,
    ) -> Decisao:
        comp = comp or {}
        return Decisao(
            irrigar=False, pulso_s=0.0, duty=0.0, lamina_mm=0.0,
            erro_mm=erro_mm, motivo=motivo, instante_s=agora,
            p_mm=comp.get("p", 0.0), i_mm=comp.get("i", 0.0),
            d_mm=comp.get("d", 0.0), ff_mm=comp.get("ff", 0.0),
            transito_mm=self._transito_mm, consumo_dia_mm=self._consumo_dia_mm,
        )

    def _em_lockout(self, agora: float) -> bool:
        if self._t_ultima_irrigacao is None:
            return False
        return (agora - self._t_ultima_irrigacao) < self.limites.lockout_s

    def _decair_transito(self, agora: float) -> None:
        """A agua em transito some conforme o sensor passa a enxerga-la.

        Decaimento exponencial com constante `tau_transito_s`: depois de
        ~3 tau o efeito ja foi integralmente lido, e o termo zera.
        """
        if self._t_transito is None or self._transito_mm <= 0:
            return
        dt = agora - self._t_transito
        if dt <= 0:
            return
        self._transito_mm *= math.exp(-dt / self.limites.tau_transito_s)
        self._t_transito = agora
        if self._transito_mm < 1e-4:
            self._transito_mm = 0.0
            self._t_transito = None

    def _rolar_dia(self, agora: float) -> None:
        """Zera a cota a cada 24 h de operacao."""
        if self._inicio_dia_s is None:
            self._inicio_dia_s = agora
            return
        if agora - self._inicio_dia_s >= 86400.0:
            self._inicio_dia_s = agora
            self._consumo_dia_mm = 0.0


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------


class ControladorPID(Controlador):
    """PID com derivada sobre a medida, anti-windup e feedforward de ETo."""

    def __init__(
        self,
        atuador: Atuador,
        limites: LimitesSeguranca,
        ganhos: GanhosPID | None = None,
    ) -> None:
        self.ganhos = ganhos or GanhosPID()
        super().__init__(atuador, limites)

    def _lamina_bruta(
        self, erro_mm: float, medida_mm: float, dt_s: float,
        deriva_medida_mm_s: float, feedforward_mm: float,
    ) -> tuple[float, dict[str, float]]:
        g = self.ganhos

        p = g.kp * erro_mm

        # O integral acumulado entra aqui; o acumulo em si so acontece
        # depois, se nao houver saturacao (ver `atualizar`).
        i = g.ki * self._integral_mm_s
        # Teto no integral: mesmo sem saturar, ele nao pode dominar.
        teto_i = 0.5 * self.limites.agua_disponivel_mm
        i = max(-teto_i, min(teto_i, i))

        # Derivada sobre a MEDIDA e com sinal negativo: se a umidade esta
        # subindo (chuva, ou agua infiltrando), freia o acionamento. Usar
        # a derivada do erro faria cada mudanca de setpoint virar um pico.
        d = -g.kd * deriva_medida_mm_s

        ff = max(0.0, float(feedforward_mm))

        return p + i + d + ff, {"p": p, "i": i, "d": d, "ff": ff}


# ---------------------------------------------------------------------------
# Fuzzy (Mamdani)
# ---------------------------------------------------------------------------


def _triangular(x: float, a: float, b: float, c: float) -> float:
    """Pertinencia triangular. Ombros em a e c, pico em b."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def _ombro_esq(x: float, b: float, c: float) -> float:
    """Pertinencia saturada a esquerda (1 ate b, cai ate c)."""
    if x <= b:
        return 1.0
    if x >= c:
        return 0.0
    return (c - x) / (c - b)


def _ombro_dir(x: float, a: float, b: float) -> float:
    """Pertinencia saturada a direita (0 ate a, sobe e fica 1 apos b)."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return (x - a) / (b - a)


class ControladorFuzzy(Controlador):
    """Mamdani com 15 regras, centroide discreto na defuzzificacao.

    Entradas normalizadas, para que as regras nao dependam do solo:
      * `e`  = erro / agua_disponivel        (deficit relativo)
      * `dm` = deriva da medida, em mm/h     (tendencia da umidade)
    Saida: fracao da lamina maxima por acionamento.

    A vantagem sobre o PID aqui e expressar diretamente o conhecimento
    agronomico ("deficit alto e solo secando rapido -> pulso longo") sem
    precisar sintonizar ganhos. A desvantagem e nao ter garantia de erro
    nulo em regime: por isso o feedforward de ETo entra somado no fim.
    """

    # (termo_erro, termo_tendencia) -> termo_saida
    # tendencia: secando / estavel / molhando
    REGRAS: dict[tuple[str, str], str] = {
        ("nulo", "secando"):   "curto",
        ("nulo", "estavel"):   "nada",
        ("nulo", "molhando"):  "nada",
        ("baixo", "secando"):  "medio",
        ("baixo", "estavel"):  "curto",
        ("baixo", "molhando"): "nada",
        ("medio", "secando"):  "longo",
        ("medio", "estavel"):  "medio",
        ("medio", "molhando"): "curto",
        ("alto", "secando"):   "maximo",
        ("alto", "estavel"):   "longo",
        ("alto", "molhando"):  "medio",
        ("critico", "secando"):  "maximo",
        ("critico", "estavel"):  "maximo",
        ("critico", "molhando"): "longo",
    }

    # Picos das funcoes de saida (fracao da lamina maxima)
    SAIDA: dict[str, float] = {
        "nada": 0.0, "curto": 0.2, "medio": 0.45, "longo": 0.7, "maximo": 1.0,
    }

    def __init__(
        self,
        atuador: Atuador,
        limites: LimitesSeguranca,
        lamina_ref_mm: float | None = None,
    ) -> None:
        # Lamina de referencia = o que a saida "maximo" entrega.
        self.lamina_ref_mm = lamina_ref_mm
        super().__init__(atuador, limites)

    def _ref(self) -> float:
        if self.lamina_ref_mm is not None:
            return self.lamina_ref_mm
        # Por padrao, metade da agua util — um acionamento nunca deve
        # tentar repor o perfil inteiro de uma vez.
        return 0.5 * self.limites.agua_disponivel_mm

    @staticmethod
    def _pertinencias_erro(e: float) -> dict[str, float]:
        """`e` = deficit relativo a agua disponivel (0 a 1)."""
        return {
            "nulo":    _ombro_esq(e, 0.02, 0.10),
            "baixo":   _triangular(e, 0.04, 0.14, 0.26),
            "medio":   _triangular(e, 0.16, 0.30, 0.46),
            "alto":    _triangular(e, 0.34, 0.50, 0.68),
            "critico": _ombro_dir(e, 0.55, 0.75),
        }

    @staticmethod
    def _pertinencias_tendencia(dm_mm_h: float) -> dict[str, float]:
        """Tendencia da umidade medida, em mm/h."""
        return {
            "secando":  _ombro_esq(dm_mm_h, -0.30, -0.05),
            "estavel":  _triangular(dm_mm_h, -0.20, 0.0, 0.20),
            "molhando": _ombro_dir(dm_mm_h, 0.05, 0.30),
        }

    def _lamina_bruta(
        self, erro_mm: float, medida_mm: float, dt_s: float,
        deriva_medida_mm_s: float, feedforward_mm: float,
    ) -> tuple[float, dict[str, float]]:
        disponivel = self.limites.agua_disponivel_mm
        e = erro_mm / disponivel if disponivel > 0 else 0.0
        dm_h = deriva_medida_mm_s * 3600.0

        mu_e = self._pertinencias_erro(e)
        mu_d = self._pertinencias_tendencia(dm_h)

        # Inferencia Mamdani: AND = min, agregacao por max sobre o termo.
        ativacao: dict[str, float] = {}
        for (te, td), ts in self.REGRAS.items():
            forca = min(mu_e.get(te, 0.0), mu_d.get(td, 0.0))
            if forca > 0.0:
                ativacao[ts] = max(ativacao.get(ts, 0.0), forca)

        # Defuzzificacao pelo centroide dos singletons ativados.
        num = sum(f * self.SAIDA[t] for t, f in ativacao.items())
        den = sum(ativacao.values())
        fracao = num / den if den > 0 else 0.0

        base = fracao * self._ref()
        ff = max(0.0, float(feedforward_mm))

        return base + ff, {"p": base, "i": 0.0, "d": 0.0, "ff": ff}


# ---------------------------------------------------------------------------
# Integracao com o modelo preditivo
# ---------------------------------------------------------------------------


def setpoint_por_fracao_disponivel(
    limites: LimitesSeguranca, fracao_alvo: float = 0.65
) -> float:
    """Converte uma meta em fracao de agua disponivel para lamina (mm).

    `fracao_alvo` = 0,65 mantem o solo em 65% da agua util, faixa usual
    para evitar estresse hidrico sem desperdicar.
    """
    if not 0.0 <= fracao_alvo <= 1.0:
        raise ValueError("fracao_alvo deve estar em [0, 1]")
    return limites.ponto_murcha_mm + fracao_alvo * limites.agua_disponivel_mm


def feedforward_de_eto(
    eto_prevista_mm: float, chuva_prevista_mm: float = 0.0,
    confianca: float = 0.5,
) -> float:
    """Demanda liquida prevista, descontada por confianca no modelo.

    `confianca` em 0,5 aplica metade da demanda prevista: o feedforward
    adianta a reposicao sem apostar todo o acionamento numa previsao. Com
    o skill medido neste projeto (0,06 a 0,09), apostar 100% na previsao
    nao se justifica.
    """
    if not 0.0 <= confianca <= 1.0:
        raise ValueError("confianca deve estar em [0, 1]")
    return max(0.0, (float(eto_prevista_mm) - float(chuva_prevista_mm)) * confianca)
