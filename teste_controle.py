#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes do controlador de irrigacao — rodam offline, sem rede e sem sklearn.

Um controlador so se verifica contra um modelo da planta, entao o arquivo
traz um simulador de balanco hidrico com as duas propriedades que tornam
este processo dificil: tempo morto na infiltracao e drenagem irreversivel
acima da capacidade de campo.

    python teste_controle.py
"""

from __future__ import annotations

import math
import sys

from controle_irrigacao import (
    Atuador,
    ControladorFuzzy,
    ControladorPID,
    Decisao,
    GanhosPID,
    LimitesSeguranca,
    Motivo,
    feedforward_de_eto,
    mm_para_vwc,
    setpoint_por_fracao_disponivel,
    vwc_para_mm,
)

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]    {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


# ---------------------------------------------------------------------------
# Planta simulada
# ---------------------------------------------------------------------------


class SoloSimulado:
    """Balanco hidrico com tempo morto e drenagem.

    A agua aplicada NAO aparece no sensor na hora: entra num reservatorio
    de transito que se esvazia com constante `tau_infiltracao_s`. E o que
    faz um controlador ingenuo encharcar.

    Acima da capacidade de campo a agua percola e e perdida — por isso
    ultrapassar a cc nao e so ineficiente, e irreversivel.
    """

    def __init__(
        self,
        agua_inicial_mm: float,
        cc_mm: float,
        pm_mm: float,
        tau_infiltracao_s: float = 2400.0,
        tau_drenagem_s: float = 7200.0,
    ) -> None:
        self.agua = float(agua_inicial_mm)
        self.cc = float(cc_mm)
        self.pm = float(pm_mm)
        self.tau_inf = float(tau_infiltracao_s)
        self.tau_dren = float(tau_drenagem_s)
        self.transito = 0.0
        self.agua_maxima = self.agua
        self.agua_minima = self.agua
        self.aplicado_total = 0.0
        self.percolado_total = 0.0

    def aplicar(self, lamina_mm: float) -> None:
        if lamina_mm > 0:
            self.transito += lamina_mm
            self.aplicado_total += lamina_mm

    def passo(self, dt_s: float, eto_mm_dia: float, chuva_mm: float = 0.0) -> None:
        # Infiltracao: parte do transito chega a zona de raizes.
        liberado = self.transito * (1.0 - math.exp(-dt_s / self.tau_inf))
        self.transito -= liberado
        self.agua += liberado + chuva_mm

        # Demanda atmosferica, continua.
        self.agua -= eto_mm_dia * dt_s / 86400.0

        # Drenagem do que passou da capacidade de campo: perda definitiva.
        if self.agua > self.cc:
            excesso = self.agua - self.cc
            perdido = excesso * (1.0 - math.exp(-dt_s / self.tau_dren))
            self.agua -= perdido
            self.percolado_total += perdido

        self.agua = max(self.agua, self.pm * 0.5)
        self.agua_maxima = max(self.agua_maxima, self.agua)
        self.agua_minima = min(self.agua_minima, self.agua)


def montar(ganhos: GanhosPID | None = None, fuzzy: bool = False):
    """Cenario padrao: gotejamento em 50 m², solo de 0–28 cm."""
    atuador = Atuador(
        vazao_lph=400.0, area_m2=50.0, eficiencia=0.85,
        pulso_min_s=30.0, pulso_max_s=1800.0, periodo_pwm_s=60.0,
    )
    limites = LimitesSeguranca(
        capacidade_campo_mm=122.6, ponto_murcha_mm=71.8,
        lamina_diaria_max_mm=15.0, folga_capacidade_mm=2.0,
        banda_morta_mm=1.5, lockout_s=1800.0, tau_transito_s=2700.0,
    )
    ctrl = (ControladorFuzzy(atuador, limites) if fuzzy
            else ControladorPID(atuador, limites, ganhos))
    return atuador, limites, ctrl


def simular(
    ctrl, limites, setpoint_mm: float, horas: float = 72.0,
    agua_inicial_mm: float = 82.0, eto_mm_dia: float = 5.0,
    dt_s: float = 60.0, chuvas: dict[float, float] | None = None,
    feedforward_mm: float = 0.0, tau_infiltracao_s: float = 2400.0,
) -> tuple[SoloSimulado, list[Decisao]]:
    """Roda o laco fechado. `chuvas` mapeia hora -> mm."""
    solo = SoloSimulado(agua_inicial_mm, limites.capacidade_campo_mm,
                        limites.ponto_murcha_mm,
                        tau_infiltracao_s=tau_infiltracao_s)
    chuvas = chuvas or {}
    decisoes: list[Decisao] = []
    passos = int(horas * 3600 / dt_s)

    for n in range(passos):
        t = n * dt_s
        chuva = chuvas.get(round(t / 3600.0, 3), 0.0)
        d = ctrl.atualizar(solo.agua, setpoint_mm, instante_s=t,
                           feedforward_mm=feedforward_mm)
        decisoes.append(d)
        if d.irrigar:
            solo.aplicar(d.lamina_mm)
        solo.passo(dt_s, eto_mm_dia, chuva)

    return solo, decisoes


# ---------------------------------------------------------------------------


def teste_atuador() -> None:
    print("\n[1] Atuador — conversao lamina <-> tempo")
    a = Atuador(vazao_lph=400.0, area_m2=50.0, eficiencia=0.85)

    checar(math.isclose(a.taxa_bruta_mm_h, 8.0),
           "400 L/h em 50 m² = 8,0 mm/h bruto (1 L/m² = 1 mm)")
    checar(math.isclose(a.taxa_efetiva_mm_h, 6.8),
           "com 85% de eficiencia = 6,8 mm/h efetivo")
    checar(math.isclose(a.segundos_para(6.8), 3600.0),
           "6,8 mm exigem exatamente 1 h de valvula aberta")
    checar(math.isclose(a.lamina_de(a.segundos_para(3.3)), 3.3, rel_tol=1e-9),
           "segundos_para e lamina_de sao inversas")
    checar(a.segundos_para(0.0) == 0.0 and a.segundos_para(-5.0) == 0.0,
           "lamina nula ou negativa nao gera acionamento")

    checar(math.isclose(a.duty(30.0), 0.5),
           "pulso de 30 s em janela de 60 s = duty 0,50")
    checar(a.duty(120.0) == 1.0 and a.duty(-5.0) == 0.0,
           "duty satura em [0, 1]")

    for kwargs, rotulo in (
        ({"vazao_lph": 0.0, "area_m2": 50.0}, "vazao nula"),
        ({"vazao_lph": 400.0, "area_m2": 0.0}, "area nula"),
        ({"vazao_lph": 400.0, "area_m2": 50.0, "eficiencia": 1.4}, "eficiencia > 1"),
        ({"vazao_lph": 400.0, "area_m2": 50.0, "pulso_min_s": 900.0,
          "pulso_max_s": 60.0}, "pulso minimo acima do maximo"),
    ):
        try:
            Atuador(**kwargs)
            checar(False, f"{rotulo} e rejeitada")
        except ValueError:
            checar(True, f"{rotulo} e rejeitada com ValueError")


def teste_unidades() -> None:
    print("\n[2] Conversao do sensor (m³/m³ <-> mm)")
    checar(math.isclose(vwc_para_mm(0.32, 280.0), 89.6),
           "0,32 m³/m³ em perfil de 280 mm = 89,6 mm")
    checar(math.isclose(mm_para_vwc(89.6, 280.0), 0.32),
           "a conversao inversa devolve o valor original")
    try:
        vwc_para_mm(0.3, 0.0)
        checar(False, "profundidade nula e rejeitada")
    except ValueError:
        checar(True, "profundidade nula e rejeitada com ValueError")


def teste_bloqueios() -> None:
    print("\n[3] Bloqueios de seguranca")
    _, limites, ctrl = montar()

    d0 = ctrl.atualizar(82.0, 96.0, instante_s=0.0)
    checar(d0.motivo is Motivo.PRIMEIRA_LEITURA and not d0.irrigar,
           "a primeira leitura nao aciona (sem dt nao ha derivada)")

    d = ctrl.atualizar(82.0, 96.0, instante_s=60.0)
    checar(d.irrigar, "com deficit de 14 mm, o segundo ciclo aciona")
    checar(d.pulso_s >= ctrl.atuador.pulso_min_s, "o pulso respeita o minimo")

    d = ctrl.atualizar(82.0, 96.0, instante_s=120.0)
    checar(d.motivo is Motivo.EM_LOCKOUT and not d.irrigar,
           "durante o lockout nao ha novo acionamento")

    _, _, c2 = montar()
    c2.atualizar(100.0, 96.0, instante_s=0.0)
    d = c2.atualizar(100.0, 96.0, instante_s=60.0)
    checar(d.motivo is Motivo.ACIMA_DO_SETPOINT and not d.irrigar,
           "solo acima do setpoint nao irriga")

    _, _, c3 = montar()
    c3.atualizar(95.0, 96.0, instante_s=0.0)
    d = c3.atualizar(95.0, 96.0, instante_s=60.0)
    checar(d.motivo is Motivo.DENTRO_DA_BANDA and not d.irrigar,
           "deficit de 1 mm cai na banda morta e nao aciona")

    _, _, c4 = montar()
    c4.atualizar(82.0, 96.0, instante_s=0.0)
    for v in (float("nan"), float("inf"), -3.0):
        d = c4.atualizar(v, 96.0, instante_s=120.0)
        checar(d.motivo is Motivo.SENSOR_INVALIDO and not d.irrigar,
               f"leitura invalida ({v}) e recusada sem acionar")


def teste_setpoint_acima_da_capacidade() -> None:
    print("\n[4] Setpoint impossivel vindo do modelo")
    _, limites, ctrl = montar()
    # O modelo pede 200 mm, bem acima da capacidade de campo (122,6).
    ctrl.atualizar(82.0, 200.0, instante_s=0.0)
    d = ctrl.atualizar(82.0, 200.0, instante_s=60.0)

    teto = limites.teto_mm
    checar(math.isclose(teto, 120.6),
           "o teto efetivo e cc menos a folga (122,6 - 2,0)")
    checar(d.erro_mm <= teto - 82.0 + 1e-9,
           "o erro usa o setpoint limitado ao teto, nao o valor pedido")
    checar(d.lamina_mm <= teto - 82.0 + 1e-9,
           "a lamina nunca mira acima da capacidade de campo")


def teste_cota_diaria() -> None:
    print("\n[5] Cota diaria")
    _, limites, ctrl = montar()
    # Deficit enorme e ETo nula: so a cota pode frear.
    solo, decisoes = simular(ctrl, limites, setpoint_mm=limites.teto_mm,
                             horas=23.0, agua_inicial_mm=75.0, eto_mm_dia=0.0)
    aplicado = sum(d.lamina_mm for d in decisoes if d.irrigar)
    checar(aplicado <= limites.lamina_diaria_max_mm + 1e-6,
           f"total aplicado em 23 h ({aplicado:.2f} mm) respeita a cota de "
           f"{limites.lamina_diaria_max_mm:.0f} mm")
    checar(any(d.motivo is Motivo.COTA_DIARIA for d in decisoes),
           "a cota esgotada aparece como motivo da recusa")


def teste_convergencia_pid() -> None:
    print("\n[6] PID em malha fechada")
    _, limites, ctrl = montar()
    setpoint = 96.0
    solo, decisoes = simular(ctrl, limites, setpoint, horas=72.0,
                             agua_inicial_mm=82.0, eto_mm_dia=5.0)

    erro_final = abs(solo.agua - setpoint)
    checar(erro_final < limites.banda_morta_mm * 2.0,
           f"converge ao setpoint: erro final {erro_final:.2f} mm")
    checar(solo.agua_maxima <= limites.capacidade_campo_mm + 1e-6,
           f"nunca ultrapassa a capacidade de campo "
           f"(maximo {solo.agua_maxima:.2f} de {limites.capacidade_campo_mm})")
    checar(solo.percolado_total < 0.5,
           f"quase nada percola ({solo.percolado_total:.3f} mm perdidos)")

    acionamentos = [d for d in decisoes if d.irrigar]
    checar(len(acionamentos) > 0, "houve acionamento")
    checar(all(d.pulso_s >= ctrl.atuador.pulso_min_s for d in acionamentos),
           "todo pulso emitido respeita o minimo do atuador")
    checar(all(d.pulso_s <= ctrl.atuador.pulso_max_s for d in acionamentos),
           "todo pulso emitido respeita o maximo do atuador")
    checar(all(0.0 <= d.duty <= 1.0 for d in decisoes),
           "o duty fica sempre em [0, 1]")

    # Intervalo minimo entre acionamentos = lockout.
    instantes = [d.instante_s for d in acionamentos]
    intervalos = [b - a for a, b in zip(instantes, instantes[1:])]
    checar(all(i >= limites.lockout_s - 1e-6 for i in intervalos),
           "o intervalo entre acionamentos nunca fica abaixo do lockout")


def teste_tempo_morto_e_transito() -> None:
    print("\n[7] O termo de agua em transito e o que evita o encharcamento")
    setpoint = 96.0

    # Com o transito desligado (tau minusculo), o controlador esquece o
    # que ja aplicou e continua acionando enquanto o sensor nao reage.
    atuador = Atuador(vazao_lph=400.0, area_m2=50.0, eficiencia=0.85,
                      pulso_min_s=30.0, pulso_max_s=1800.0)
    sem = LimitesSeguranca(capacidade_campo_mm=122.6, ponto_murcha_mm=71.8,
                           lamina_diaria_max_mm=40.0, lockout_s=60.0,
                           tau_transito_s=1e-6)
    com = LimitesSeguranca(capacidade_campo_mm=122.6, ponto_murcha_mm=71.8,
                           lamina_diaria_max_mm=40.0, lockout_s=60.0,
                           tau_transito_s=2700.0)

    solo_sem, _ = simular(ControladorPID(atuador, sem), sem, setpoint,
                          horas=24.0, agua_inicial_mm=82.0, eto_mm_dia=5.0)
    solo_com, _ = simular(ControladorPID(atuador, com), com, setpoint,
                          horas=24.0, agua_inicial_mm=82.0, eto_mm_dia=5.0)

    excesso_sem = solo_sem.agua_maxima - setpoint
    excesso_com = solo_com.agua_maxima - setpoint
    print(f"          sem transito: pico {solo_sem.agua_maxima:.1f} mm, "
          f"{solo_sem.aplicado_total:.1f} mm aplicados, "
          f"{solo_sem.percolado_total:.1f} percolados")
    print(f"          com transito: pico {solo_com.agua_maxima:.1f} mm, "
          f"{solo_com.aplicado_total:.1f} mm aplicados, "
          f"{solo_com.percolado_total:.1f} percolados")

    checar(excesso_sem > excesso_com,
           "sem o termo de transito o pico de umidade e maior (overshoot)")
    checar(solo_com.aplicado_total < solo_sem.aplicado_total,
           "com o termo de transito gasta-se menos agua para a mesma meta")
    checar(solo_com.percolado_total <= solo_sem.percolado_total,
           "e perde-se menos por percolacao")


def teste_antiwindup() -> None:
    print("\n[8] Anti-windup")
    teto_i = 0.5 * (122.6 - 71.8)

    # (a) O limitador do termo integral, testado diretamente. Um cenario de
    #     malha fechada nao serve aqui: o proprio termo de transito impede
    #     que o integral cresca, entao o teste passaria sem exercitar nada.
    _, limites, ctrl = montar(GanhosPID(kp=0.7, ki=1e-3, kd=0.0))
    ctrl.atualizar(82.0, 96.0, instante_s=0.0)
    ctrl._integral_mm_s = 1.0e9          # valor absurdo, de proposito
    d = ctrl.atualizar(82.0, 96.0, instante_s=60.0)
    checar(math.isclose(d.i_mm, teto_i),
           f"integral absurdo e limitado ao teto ({d.i_mm:.2f} = {teto_i:.2f} mm)")
    checar(d.lamina_mm <= limites.teto_mm - 82.0 + 1e-9,
           "mesmo com o integral no teto, a lamina respeita a capacidade")

    ctrl._integral_mm_s = -1.0e9
    d = ctrl.atualizar(82.0, 96.0, instante_s=4000.0)
    checar(math.isclose(d.i_mm, -teto_i),
           "o limite vale tambem para o lado negativo")

    # (b) Sob saturacao o integral NAO pode acumular — e o que impede o
    #     encharcamento quando o atuador nao da conta do deficit.
    fraco = Atuador(vazao_lph=60.0, area_m2=50.0, eficiencia=0.85,
                    pulso_min_s=30.0, pulso_max_s=600.0)
    lim_b = LimitesSeguranca(capacidade_campo_mm=122.6, ponto_murcha_mm=71.8,
                             lamina_diaria_max_mm=40.0, lockout_s=600.0)
    c_sat = ControladorPID(fraco, lim_b, GanhosPID(kp=0.7, ki=1e-3, kd=0.0))
    solo, decisoes = simular(c_sat, lim_b, setpoint_mm=lim_b.teto_mm,
                             horas=36.0, agua_inicial_mm=74.0, eto_mm_dia=4.0)
    saturados = [d for d in decisoes if d.irrigar and d.saturou]
    checar(len(saturados) > 10,
           f"o cenario de fato satura o atuador ({len(saturados)} pulsos no maximo)")
    checar(c_sat._integral_mm_s == 0.0,
           "sob saturacao continua o integral permanece zerado")
    checar(solo.agua_maxima <= lim_b.capacidade_campo_mm + 1e-6,
           "e o solo nunca ultrapassa a capacidade de campo")

    # (c) O integral acumula quando NAO ha saturacao.
    forte = Atuador(vazao_lph=400.0, area_m2=50.0, eficiencia=0.85,
                    pulso_min_s=30.0, pulso_max_s=5400.0)
    lim_c = LimitesSeguranca(capacidade_campo_mm=122.6, ponto_murcha_mm=71.8,
                             lamina_diaria_max_mm=60.0, lockout_s=1800.0)
    c_ok = ControladorPID(forte, lim_c, GanhosPID(kp=0.5, ki=1e-4, kd=0.0))
    simular(c_ok, lim_c, setpoint_mm=110.0, horas=72.0,
            agua_inicial_mm=80.0, eto_mm_dia=9.0)
    checar(c_ok._integral_mm_s > 0.0,
           "sem saturacao o integral acumula (o mecanismo existe e e exercitado)")

    # (d) E zera assim que o solo passa da meta.
    _, _, c2 = montar(GanhosPID(kp=0.7, ki=1e-3, kd=0.0))
    c2.atualizar(80.0, 96.0, instante_s=0.0)
    c2.atualizar(80.0, 96.0, instante_s=60.0)
    c2.atualizar(120.0, 96.0, instante_s=4000.0)   # chuva forte
    d = c2.atualizar(120.0, 96.0, instante_s=4100.0)
    checar(d.i_mm == 0.0 and c2._integral_mm_s == 0.0,
           "o integral zera assim que o solo passa da meta")


def teste_derivada_sobre_medida() -> None:
    print("\n[9] Derivada sobre a medida (sem chute de setpoint)")
    _, _, ctrl = montar(GanhosPID(kp=0.7, ki=0.0, kd=600.0))
    ctrl.atualizar(85.0, 96.0, instante_s=0.0)
    d_ref = ctrl.atualizar(85.0, 96.0, instante_s=60.0)

    # Mesma medida, setpoint saltando: a derivada NAO pode reagir.
    _, _, c2 = montar(GanhosPID(kp=0.7, ki=0.0, kd=600.0))
    c2.atualizar(85.0, 96.0, instante_s=0.0)
    d_salto = c2.atualizar(85.0, 110.0, instante_s=60.0)

    checar(math.isclose(d_ref.d_mm, d_salto.d_mm, abs_tol=1e-9),
           "mudar o setpoint nao altera o termo derivativo")
    checar(d_salto.p_mm > d_ref.p_mm,
           "mas o termo proporcional reage ao novo setpoint, como deve")

    # Umidade subindo (chuva) tem de FREAR o acionamento.
    _, _, c3 = montar(GanhosPID(kp=0.7, ki=0.0, kd=600.0))
    c3.atualizar(85.0, 96.0, instante_s=0.0)
    d_subindo = c3.atualizar(86.0, 96.0, instante_s=60.0)
    checar(d_subindo.d_mm < 0.0,
           "umidade em alta produz termo derivativo negativo (freia)")


def teste_chuva() -> None:
    print("\n[10] Chuva durante a operacao")
    _, limites, ctrl = montar()
    solo, decisoes = simular(ctrl, limites, setpoint_mm=96.0, horas=48.0,
                             agua_inicial_mm=84.0, eto_mm_dia=5.0,
                             chuvas={6.0: 12.0, 6.1: 8.0})

    checar(solo.agua_maxima <= limites.capacidade_campo_mm + 1e-6,
           f"chuva de 20 mm nao leva o solo acima da cc "
           f"(pico {solo.agua_maxima:.1f} mm)")

    pos_chuva = [d for d in decisoes
                 if d.irrigar and 6.0 * 3600 <= d.instante_s <= 12.0 * 3600]
    checar(len(pos_chuva) == 0,
           "nao ha acionamento nas 6 h seguintes a chuva")


def teste_feedforward() -> None:
    print("\n[11] Feedforward de ETo")
    ff = feedforward_de_eto(eto_prevista_mm=6.0, chuva_prevista_mm=1.0,
                            confianca=0.5)
    checar(math.isclose(ff, 2.5), "(6 - 1) x 0,5 = 2,5 mm de antecipacao")
    checar(feedforward_de_eto(2.0, 10.0) == 0.0,
           "chuva prevista maior que a demanda zera o feedforward")
    try:
        feedforward_de_eto(5.0, confianca=1.5)
        checar(False, "confianca fora de [0,1] e rejeitada")
    except ValueError:
        checar(True, "confianca fora de [0,1] e rejeitada com ValueError")

    # Com feedforward o solo deve ficar mais perto da meta sob demanda alta.
    _, lim_a, c_sem = montar()
    solo_sem, _ = simular(c_sem, lim_a, 96.0, horas=48.0,
                          agua_inicial_mm=96.0, eto_mm_dia=8.0)
    _, lim_b, c_com = montar()
    solo_com, _ = simular(c_com, lim_b, 96.0, horas=48.0,
                          agua_inicial_mm=96.0, eto_mm_dia=8.0,
                          feedforward_mm=1.5)
    desvio_sem = abs(solo_sem.agua - 96.0)
    desvio_com = abs(solo_com.agua - 96.0)
    print(f"          sem FF: {solo_sem.agua:.2f} mm (desvio {desvio_sem:.2f}) | "
          f"com FF: {solo_com.agua:.2f} mm (desvio {desvio_com:.2f})")
    checar(desvio_com < desvio_sem,
           "o feedforward aproxima o solo da meta sob demanda alta "
           "(antecipa a perda em vez de esperar o sensor acusar)")
    checar(solo_com.agua_maxima <= 122.6 + 1e-6,
           "e nao leva a ultrapassar a capacidade de campo")


def teste_fuzzy() -> None:
    print("\n[12] Controlador Fuzzy")
    _, limites, ctrl = montar(fuzzy=True)

    mu = ControladorFuzzy._pertinencias_erro(0.30)
    checar(math.isclose(sum(mu.values()), 1.0, abs_tol=0.35),
           "as pertinencias de erro cobrem o universo sem buracos grandes")
    checar(ControladorFuzzy._pertinencias_erro(0.0)["nulo"] == 1.0,
           "deficit zero pertence totalmente ao termo 'nulo'")
    checar(ControladorFuzzy._pertinencias_erro(0.9)["critico"] == 1.0,
           "deficit de 90% da agua util pertence totalmente a 'critico'")

    mt = ControladorFuzzy._pertinencias_tendencia(0.0)
    checar(mt["estavel"] == 1.0 and mt["secando"] == 0.0 and mt["molhando"] == 0.0,
           "tendencia nula e totalmente 'estavel'")

    checar(len(ControladorFuzzy.REGRAS) == 15, "a base tem 15 regras")
    termos_saida = set(ControladorFuzzy.REGRAS.values())
    checar(termos_saida <= set(ControladorFuzzy.SAIDA),
           "toda regra aponta para um termo de saida definido")

    solo, decisoes = simular(ctrl, limites, setpoint_mm=96.0, horas=72.0,
                             agua_inicial_mm=82.0, eto_mm_dia=5.0)
    erro_final = abs(solo.agua - 96.0)
    print(f"          fuzzy: erro final {erro_final:.2f} mm, "
          f"{solo.aplicado_total:.1f} mm aplicados, "
          f"{solo.percolado_total:.2f} percolados")
    checar(erro_final < limites.banda_morta_mm * 3.0,
           f"o fuzzy tambem converge (erro final {erro_final:.2f} mm)")
    checar(solo.agua_maxima <= limites.capacidade_campo_mm + 1e-6,
           "e tambem respeita a capacidade de campo")

    acionamentos = [d for d in decisoes if d.irrigar]
    checar(all(d.pulso_s >= ctrl.atuador.pulso_min_s for d in acionamentos),
           "os pulsos do fuzzy respeitam o minimo do atuador")


def teste_setpoint_helper() -> None:
    print("\n[13] Setpoint por fracao de agua disponivel")
    limites = LimitesSeguranca(capacidade_campo_mm=122.6, ponto_murcha_mm=71.8)
    sp = setpoint_por_fracao_disponivel(limites, 0.65)
    esperado = 71.8 + 0.65 * 50.8
    checar(math.isclose(sp, esperado),
           f"65% da agua util = {esperado:.2f} mm")
    checar(setpoint_por_fracao_disponivel(limites, 0.0) == limites.ponto_murcha_mm,
           "fracao 0 devolve o ponto de murcha")
    checar(setpoint_por_fracao_disponivel(limites, 1.0) == limites.capacidade_campo_mm,
           "fracao 1 devolve a capacidade de campo")
    try:
        setpoint_por_fracao_disponivel(limites, 1.5)
        checar(False, "fracao fora de [0,1] e rejeitada")
    except ValueError:
        checar(True, "fracao fora de [0,1] e rejeitada com ValueError")

    try:
        LimitesSeguranca(capacidade_campo_mm=70.0, ponto_murcha_mm=80.0)
        checar(False, "cc abaixo do ponto de murcha e rejeitada")
    except ValueError:
        checar(True, "cc abaixo do ponto de murcha e rejeitada com ValueError")


def main() -> int:
    print("=" * 72)
    print("TESTES — controlador de irrigacao")
    print("=" * 72)

    teste_atuador()
    teste_unidades()
    teste_bloqueios()
    teste_setpoint_acima_da_capacidade()
    teste_cota_diaria()
    teste_convergencia_pid()
    teste_tempo_morto_e_transito()
    teste_antiwindup()
    teste_derivada_sobre_medida()
    teste_chuva()
    teste_feedforward()
    teste_fuzzy()
    teste_setpoint_helper()

    print("\n" + "=" * 72)
    if FALHAS:
        print(f"{len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"  - {f}")
        return 1
    print("Todos os testes passaram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
