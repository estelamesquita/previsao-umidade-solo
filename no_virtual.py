#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No virtual: um ESP32 de mentira, para rodar o sistema inteiro sem hardware.

Faz exatamente o que o firmware faz — le o sensor, envia a leitura, pergunta
se deve irrigar, aciona a bomba pelo tempo recebido e confirma a execucao no
ciclo seguinte — so que o sensor, a bomba e o solo sao simulados.

O servidor NAO sabe que e simulacao: ele recebe o mesmo JSON, roda o mesmo
controlador PID e devolve os mesmos comandos. O que esta sendo exercitado e o
sistema de verdade.

    # 1) servidor com o tempo acelerado (ver --aceleracao abaixo)
    uvicorn servidor:app --port 8000

    # 2) o no virtual
    python no_virtual.py --aceleracao 120

    # 3) o painel, para assistir
    streamlit run dashboard.py

Por que acelerar, e o que a aceleracao distorce
----------------------------------------------
Em campo o sistema espera 30 min entre acionamentos e a agua leva ~45 min
para infiltrar. Para ver um dia inteiro em minutos, o SOLO anda mais rapido
aqui (--aceleracao) e o SERVIDOR precisa das constantes equivalentes:

    IRRIGACAO_LOCKOUT_S=20 IRRIGACAO_TAU_TRANSITO_S=40 \
    IRRIGACAO_PULSO_MAX_S=600 uvicorn servidor:app --port 8000

RESSALVA HONESTA: o controlador no servidor mede dt pelo relogio REAL,
enquanto o solo aqui anda acelerado. O termo derivativo, que e uma taxa por
segundo, enxerga o solo mudando mais rapido do que o proprio relogio dele —
entao a aceleracao serve para VER o laco funcionando, nao para avaliar a
qualidade do controle. Para isso existe `teste_controle.py`, que simula em
tempo exato e sem rede.

Use --aceleracao 1 quando quiser fidelidade em vez de velocidade.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time

import requests

# O modelo de solo e o mesmo usado nos testes do controlador: duplicar a
# fisica aqui so criaria duas versoes para divergirem.
from teste_controle import SoloSimulado


# ---------------------------------------------------------------------------


def eto_do_momento(hora_do_dia: float, eto_diaria_mm: float) -> float:
    """Distribui a ETo diaria ao longo do dia, com pico ao meio-dia.

    Meia senoide entre 6 h e 18 h: a demanda atmosferica acompanha a
    radiacao solar, e nao e constante como uma torneira pingando.
    """
    if hora_do_dia < 6.0 or hora_do_dia > 18.0:
        return 0.0
    fase = (hora_do_dia - 6.0) / 12.0
    # Integral da meia senoide = 1, entao o total do dia bate com eto_diaria.
    return eto_diaria_mm * (math.pi / 2.0) * math.sin(math.pi * fase) / 12.0


class NoVirtual:
    """Imita o firmware: mesmo protocolo, mesma maquina de estados."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.base = args.api.rstrip("/")
        self.device = args.device
        self.headers = {"X-API-Key": args.chave} if args.chave else {}
        self.acel = args.aceleracao
        self.eto_diaria = args.eto
        self.chance_chuva = args.chuva

        self.solo = SoloSimulado(
            agua_inicial_mm=args.umidade_inicial,
            cc_mm=args.cc, pm_mm=args.pm,
            tau_infiltracao_s=args.tau_infiltracao,
        )
        self.profundidade_mm = args.profundidade

        # Estado do "rele", igual ao do firmware
        self.pulso_ativo = False
        self.pulso_fim_sim = 0.0
        self.pulso_duracao = 0.0
        self.pulso_id = ""
        self.ultimo_comando = ""
        self.ultimo_segundos = -1.0

        self.t_sim = args.hora_inicial * 3600.0
        self.irrigacoes = 0
        self.aplicado_total = 0.0
        self.lockouts_seguidos = 0

    # -- o "sensor" -------------------------------------------------------

    def ler_sensor(self) -> float:
        """Umidade volumetrica com o ruido que um capacitivo real tem."""
        vwc = self.solo.agua / self.profundidade_mm
        vwc += random.gauss(0.0, 0.002)       # ~0,2% de ruido
        return max(0.0, min(1.0, vwc))

    # -- a "rede" ---------------------------------------------------------

    def enviar_leitura(self, vwc: float) -> bool:
        corpo = {"device_id": self.device, "umidade_m3m3": round(vwc, 4),
                 "rssi_dbm": -55}
        if self.ultimo_comando and self.ultimo_segundos >= 0:
            corpo["comando_executado"] = self.ultimo_comando
            corpo["segundos_executados"] = round(self.ultimo_segundos, 1)
        try:
            r = requests.post(f"{self.base}/api/v1/leitura", json=corpo,
                              headers=self.headers, timeout=6)
        except requests.RequestException as exc:
            print(f"  [REDE] POST falhou: {exc.__class__.__name__}")
            return False
        if r.status_code != 200:
            print(f"  [REDE] POST {r.status_code}: {r.text[:120]}")
            return False
        self.ultimo_comando, self.ultimo_segundos = "", -1.0
        return True

    def buscar_comando(self) -> dict | None:
        try:
            r = requests.get(f"{self.base}/api/v1/comando/{self.device}",
                             headers=self.headers, timeout=6)
        except requests.RequestException as exc:
            print(f"  [REDE] GET falhou: {exc.__class__.__name__}")
            return None
        if r.status_code != 200:
            print(f"  [REDE] GET {r.status_code}: {r.text[:120]}")
            return None
        return r.json()

    # -- a "bomba" --------------------------------------------------------

    def iniciar_pulso(self, segundos: float, comando_id: str,
                      taxa_mm_h: float) -> None:
        if self.pulso_ativo:
            return
        self.pulso_ativo = True
        self.pulso_id = comando_id
        self.pulso_duracao = segundos
        self.pulso_fim_sim = self.t_sim + segundos
        self.irrigacoes += 1
        lamina = taxa_mm_h * segundos / 3600.0
        print(f"  [BOMBA] LIGADA por {segundos:.0f} s "
              f"({lamina:.2f} mm) — comando {comando_id}")

    def atualizar_pulso(self, dt_sim: float, taxa_mm_h: float) -> None:
        if not self.pulso_ativo:
            return
        # Agua entra no solo enquanto a bomba corre, nao de uma vez no fim.
        corrido = min(dt_sim, max(0.0, self.pulso_fim_sim - (self.t_sim - dt_sim)))
        self.solo.aplicar(taxa_mm_h * corrido / 3600.0)
        self.aplicado_total += taxa_mm_h * corrido / 3600.0

        if self.t_sim >= self.pulso_fim_sim:
            self.pulso_ativo = False
            self.ultimo_comando = self.pulso_id
            self.ultimo_segundos = self.pulso_duracao
            print(f"  [BOMBA] desligada ({self.ultimo_segundos:.0f} s)")

    # -- o laco -----------------------------------------------------------

    def rodar(self, passos: int, passo_real_s: float, taxa_mm_h: float) -> int:
        print(f"no virtual '{self.device}' -> {self.base}")
        print(f"aceleracao {self.acel}x | ETo {self.eto_diaria} mm/dia | "
              f"solo {self.solo.agua:.1f} mm (cc {self.solo.cc} / pm {self.solo.pm})")
        print("-" * 72)

        avisou_lento = False

        for n in range(passos):
            dt_sim = passo_real_s * self.acel
            self.t_sim += dt_sim

            hora = (self.t_sim / 3600.0) % 24.0
            eto_s = eto_do_momento(hora, self.eto_diaria) / 3600.0

            chuva = 0.0
            if self.chance_chuva > 0 and random.random() < self.chance_chuva:
                chuva = random.uniform(2.0, 12.0)
                print(f"  [CHUVA] {chuva:.1f} mm")

            self.atualizar_pulso(dt_sim, taxa_mm_h)
            self.solo.passo(dt_sim, eto_s * 86400.0, chuva)

            vwc = self.ler_sensor()
            dia = int(self.t_sim // 86400) + 1
            print(f"[dia {dia} {int(hora):02d}:{int((hora % 1) * 60):02d}] "
                  f"solo {self.solo.agua:6.2f} mm  ({vwc:.3f} m3/m3)", end="")

            if not self.enviar_leitura(vwc):
                print()
                time.sleep(passo_real_s)
                continue

            cmd = self.buscar_comando()
            if cmd is None:
                print()
                time.sleep(passo_real_s)
                continue

            motivo = str(cmd.get("motivo", "?"))

            if cmd.get("acionar"):
                self.lockouts_seguidos = 0
                if self.pulso_ativo and str(cmd["comando_id"]) == self.pulso_id:
                    # O servidor reenvia o MESMO comando dentro da janela de
                    # idempotencia: nao e uma segunda dose.
                    print("  -> mesmo comando, pulso ja em andamento")
                else:
                    print(f"  -> IRRIGAR {cmd['segundos']:.0f} s")
                    self.iniciar_pulso(float(cmd["segundos"]),
                                       str(cmd["comando_id"]), taxa_mm_h)
            else:
                sp = cmd.get("setpoint_mm")
                alvo = f" alvo {sp:.1f}" if sp else ""
                print(f"  -> {motivo}{alvo}")

            # So avisa se o lockout REALMENTE domina a simulacao. Lockout
            # aparecendo e comportamento correto, nao sintoma.
            if motivo == "em_lockout_pos_irrigacao":
                self.lockouts_seguidos += 1
            else:
                self.lockouts_seguidos = 0

            if not avisou_lento and self.lockouts_seguidos >= 30:
                avisou_lento = True
                print("\n  [DICA] o lockout esta consumindo a simulacao. Se o "
                      "servidor subiu com os tempos de campo, reinicie com:\n"
                      "    IRRIGACAO_LOCKOUT_S=20 IRRIGACAO_TAU_TRANSITO_S=40 "
                      "IRRIGACAO_PULSO_MAX_S=600 uvicorn servidor:app --port 8000\n")

            time.sleep(passo_real_s)

        print("-" * 72)
        print(f"fim: {self.irrigacoes} irrigacoes, "
              f"{self.aplicado_total:.1f} mm aplicados, "
              f"{self.solo.percolado_total:.2f} mm percolados")
        print(f"solo: min {self.solo.agua_minima:.1f} / "
              f"max {self.solo.agua_maxima:.1f} / final {self.solo.agua:.1f} mm")
        if self.solo.agua_maxima > self.solo.cc + 0.01:
            print("ATENCAO: o solo passou da capacidade de campo em algum momento")
        return 0


def principal(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="ESP32 virtual: roda o sistema completo sem hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--device", default="esp32-horta-01")
    p.add_argument("--chave", default=None, help="X-API-Key, se o servidor exigir")

    p.add_argument("--aceleracao", type=float, default=120.0,
                   help="quantos segundos simulados por segundo real")
    p.add_argument("--passos", type=int, default=200, help="ciclos a simular")
    p.add_argument("--passo-real", type=float, default=1.0,
                   help="segundos reais entre ciclos")

    p.add_argument("--umidade-inicial", type=float, default=84.0,
                   help="agua inicial na zona de raizes (mm)")
    p.add_argument("--hora-inicial", type=float, default=8.0,
                   help="hora do dia em que a simulacao comeca (0 a 24). "
                        "O padrao e de manha: a ETo e nula de madrugada e "
                        "nada acontece se comecar a meia-noite")
    p.add_argument("--cc", type=float, default=122.6, help="capacidade de campo (mm)")
    p.add_argument("--pm", type=float, default=71.8, help="ponto de murcha (mm)")
    p.add_argument("--profundidade", type=float, default=280.0,
                   help="profundidade do perfil (mm), para converter em m3/m3")
    p.add_argument("--eto", type=float, default=5.0, help="ETo diaria (mm)")
    p.add_argument("--chuva", type=float, default=0.0,
                   help="probabilidade de chuva por ciclo (0 a 1)")
    p.add_argument("--tau-infiltracao", type=float, default=2400.0,
                   help="constante de tempo da infiltracao (s simulados)")
    p.add_argument("--vazao-lph", type=float, default=400.0)
    p.add_argument("--area-m2", type=float, default=50.0)
    p.add_argument("--eficiencia", type=float, default=0.85)
    p.add_argument("--semente", type=int, default=None)

    args = p.parse_args(argv)
    if args.semente is not None:
        random.seed(args.semente)

    # Mesma conta do Atuador: 1 L/m2 = 1 mm.
    taxa_mm_h = args.vazao_lph / args.area_m2 * args.eficiencia

    no = NoVirtual(args)
    try:
        return no.rodar(args.passos, args.passo_real, taxa_mm_h)
    except KeyboardInterrupt:
        print("\ninterrompido")
        return 0


if __name__ == "__main__":
    sys.exit(principal())
