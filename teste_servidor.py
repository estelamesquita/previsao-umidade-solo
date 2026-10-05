#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes do backend FastAPI — rodam offline, sem rede e sem modelo treinado.

O foco nao e "o endpoint responde 200": e que o servidor **nunca ligue a
bomba por engano**. Sensor mudo, leitura velha, modelo fora do ar, retry do
dispositivo, corpo malformado — em todos esses casos a resposta correta e
nao acionar.

    python teste_servidor.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]    {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def montar_app(**env: str):
    """Recarrega o modulo com variaveis de ambiente novas.

    A configuracao e lida na importacao, entao cada cenario precisa de uma
    importacao limpa — e de um arquivo de estado proprio, para um teste nao
    herdar a cota diaria do outro.
    """
    from fastapi.testclient import TestClient

    base = {
        "IRRIGACAO_ESTADO": str(Path(tempfile.mkdtemp()) / "estado.json"),
        "IRRIGACAO_CC_MM": "122.6",
        "IRRIGACAO_PM_MM": "71.8",
        "IRRIGACAO_PROFUNDIDADE_MM": "280.0",
        "IRRIGACAO_VAZAO_LPH": "400.0",
        "IRRIGACAO_AREA_M2": "50.0",
    }
    base.update(env)
    for k in list(os.environ):
        if k.startswith("IRRIGACAO_"):
            del os.environ[k]
    os.environ.update(base)

    import servidor
    importlib.reload(servidor)
    return servidor, TestClient(servidor.app)


# ---------------------------------------------------------------------------


def teste_health() -> None:
    print("\n[1] Health check")
    _, cliente = montar_app()
    with cliente:
        r = cliente.get("/health")
        checar(r.status_code == 200, "GET /health responde 200")
        j = r.json()
        checar(j["ok"] is True, "o corpo indica ok")
        checar(j["modelo_carregado"] is False,
               "sem bundle configurado, reporta modelo ausente")
        checar(j["origem_setpoint"] == "fallback",
               "e o setpoint vem do fallback, nao do modelo")
        checar(j["autenticado"] is False,
               "sem IRRIGACAO_API_KEY, reporta API aberta")


def teste_sem_leitura() -> None:
    print("\n[2] GET antes de qualquer leitura")
    _, cliente = montar_app()
    with cliente:
        r = cliente.get("/api/v1/comando/esp32-novo")
        checar(r.status_code == 200, "responde 200 mesmo sem conhecer o device")
        j = r.json()
        checar(j["acionar"] is False, "NAO aciona sem leitura de sensor")
        checar(j["segundos"] == 0.0, "e a duracao vem zerada")
        checar(j["motivo"] == "sem_leitura_do_sensor", "o motivo e explicito")


def teste_ciclo_basico() -> None:
    print("\n[3] Ciclo POST -> GET")
    _, cliente = montar_app()
    with cliente:
        # 0,30 m³/m³ x 280 mm = 84,0 mm; setpoint base = 71,8 + 0,65x50,8 = 104,8
        r = cliente.post("/api/v1/leitura",
                         json={"device_id": "esp32-01", "umidade_m3m3": 0.30})
        checar(r.status_code == 200, "POST /leitura aceita a telemetria")
        j = r.json()
        checar(abs(j["umidade_mm"] - 84.0) < 1e-6,
               "0,30 m³/m³ em 280 mm e convertido para 84,0 mm")
        checar(j["proximo_poll_s"] > 0, "a resposta diz quando voltar a consultar")
        checar(j["recebido_em"] > 1.7e9,
               "o carimbo e do servidor (nao se confia no relogio do ESP32)")

        # Primeira chamada do controlador nao aciona (sem dt).
        cliente.get("/api/v1/comando/esp32-01")
        r = cliente.get("/api/v1/comando/esp32-01")
        j = r.json()
        checar(j["acionar"] is True, "com deficit de ~20 mm, o comando aciona")
        checar(j["segundos"] > 0, f"com duracao definida ({j['segundos']} s)")
        checar(0.0 <= j["duty"] <= 1.0, "o duty fica em [0, 1]")
        checar(len(j["comando_id"]) >= 8, "o comando traz um identificador")
        checar(j["setpoint_mm"] > 100.0, "o setpoint e ~65% da agua disponivel")


def teste_idempotencia() -> None:
    print("\n[4] Retry do dispositivo nao dobra a dose")
    _, cliente = montar_app()
    with cliente:
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-02", "umidade_m3m3": 0.28})
        cliente.get("/api/v1/comando/esp32-02")
        r1 = cliente.get("/api/v1/comando/esp32-02").json()
        checar(r1["acionar"] is True, "o primeiro GET emite o comando")

        r2 = cliente.get("/api/v1/comando/esp32-02").json()
        r3 = cliente.get("/api/v1/comando/esp32-02").json()
        checar(r2["comando_id"] == r1["comando_id"],
               "repetir o GET devolve o MESMO comando_id")
        checar(r3["comando_id"] == r1["comando_id"],
               "e continua o mesmo na terceira tentativa")
        checar(r2["segundos"] == r1["segundos"],
               "com a mesma duracao — nao e uma segunda dose")

        st = cliente.get("/api/v1/status/esp32-02").json()
        checar(st["irrigacoes_hoje"] == 1,
               "o servidor contabiliza UMA irrigacao, nao tres")


def teste_confirmacao() -> None:
    print("\n[5] Confirmacao de execucao pelo ESP32")
    _, cliente = montar_app()
    with cliente:
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-03", "umidade_m3m3": 0.28})
        cliente.get("/api/v1/comando/esp32-03")
        cmd = cliente.get("/api/v1/comando/esp32-03").json()

        st = cliente.get("/api/v1/status/esp32-03").json()
        checar(st["comando_pendente"] == cmd["comando_id"],
               "o comando fica pendente ate o dispositivo confirmar")

        cliente.post("/api/v1/leitura", json={
            "device_id": "esp32-03", "umidade_m3m3": 0.29,
            "comando_executado": cmd["comando_id"],
            "segundos_executados": cmd["segundos"],
        })
        st = cliente.get("/api/v1/status/esp32-03").json()
        checar(st["comando_pendente"] is None,
               "a confirmacao limpa o comando pendente")
        checar(st["falhas_bomba"] == 0,
               "execucao completa nao conta como falha")


def teste_bomba_nao_ligou() -> None:
    print("\n[6] Bomba que nao ligou fica visivel")
    _, cliente = montar_app()
    with cliente:
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-04", "umidade_m3m3": 0.28})
        cliente.get("/api/v1/comando/esp32-04")
        cmd = cliente.get("/api/v1/comando/esp32-04").json()

        # Mandou 600 s, rodou 5 s: valvula travada ou queda de energia.
        cliente.post("/api/v1/leitura", json={
            "device_id": "esp32-04", "umidade_m3m3": 0.28,
            "comando_executado": cmd["comando_id"],
            "segundos_executados": 5.0,
        })
        st = cliente.get("/api/v1/status/esp32-04").json()
        checar(st["falhas_bomba"] == 1,
               "executar muito menos que o mandado conta como falha de bomba")
        checar(st["comando_pendente"] is None,
               "e mesmo assim o comando sai de pendente")


def teste_leitura_velha() -> None:
    print("\n[7] Leitura velha nao autoriza irrigacao")
    servidor, cliente = montar_app(IRRIGACAO_LEITURA_VELHA_S="1")
    with cliente:
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-05", "umidade_m3m3": 0.25})
        time.sleep(1.2)
        j = cliente.get("/api/v1/comando/esp32-05").json()
        checar(j["acionar"] is False,
               "com a leitura vencida, NAO aciona (irrigar as cegas e o erro caro)")
        checar(j["motivo"] == "leitura_desatualizada", "o motivo e explicito")
        checar(j["degradado"] is True, "e o estado degradado e sinalizado")


def teste_validacao() -> None:
    print("\n[8] Validacao do corpo")
    _, cliente = montar_app()
    with cliente:
        casos = [
            ({"device_id": "x"}, "sem nenhuma umidade"),
            ({"device_id": "x", "umidade_m3m3": 1.8}, "umidade acima de 1 m³/m³"),
            ({"device_id": "x", "umidade_m3m3": -0.2}, "umidade negativa"),
            ({"device_id": "", "umidade_m3m3": 0.3}, "device_id vazio"),
            ({"device_id": "a/../b", "umidade_m3m3": 0.3},
             "device_id com caractere de caminho"),
            ({"device_id": "x", "umidade_m3m3": 0.3, "bateria_v": 99.0},
             "bateria fora da faixa"),
        ]
        for corpo, rotulo in casos:
            r = cliente.post("/api/v1/leitura", json=corpo)
            checar(r.status_code == 422, f"{rotulo} e recusada com 422")

        r = cliente.post("/api/v1/leitura",
                         json={"device_id": "ok-1", "umidade_mm": 90.0})
        checar(r.status_code == 200, "umidade_mm direto tambem e aceita")


def teste_autenticacao() -> None:
    print("\n[9] Autenticacao por chave")
    _, cliente = montar_app(IRRIGACAO_API_KEY="segredo-de-teste")
    with cliente:
        r = cliente.post("/api/v1/leitura",
                         json={"device_id": "esp32-06", "umidade_m3m3": 0.3})
        checar(r.status_code == 401, "POST sem chave e recusado com 401")

        r = cliente.get("/api/v1/comando/esp32-06")
        checar(r.status_code == 401, "GET sem chave e recusado com 401")

        r = cliente.get("/api/v1/comando/esp32-06",
                        headers={"X-API-Key": "chave-errada"})
        checar(r.status_code == 401, "chave errada e recusada com 401")

        h = {"X-API-Key": "segredo-de-teste"}
        r = cliente.post("/api/v1/leitura", headers=h,
                         json={"device_id": "esp32-06", "umidade_m3m3": 0.3})
        checar(r.status_code == 200, "com a chave correta, passa")

        r = cliente.get("/health")
        checar(r.status_code == 200,
               "/health segue aberto, para o monitor de disponibilidade")


def teste_modelo_fora_do_ar() -> None:
    print("\n[10] Degradacao quando o modelo falha")
    _, cliente = montar_app(IRRIGACAO_BUNDLE="modelo_que_nao_existe.joblib")
    with cliente:
        j = cliente.get("/health").json()
        checar(j["ok"] is True, "o servidor sobe mesmo com bundle inexistente")
        checar(j["origem_setpoint"] == "fallback",
               "e cai para o setpoint fixo em vez de quebrar")

        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-07", "umidade_m3m3": 0.28})
        cliente.get("/api/v1/comando/esp32-07")
        r = cliente.get("/api/v1/comando/esp32-07").json()
        checar(r["acionar"] is True,
               "o controle CONTINUA operando sem o modelo")
        checar(r["degradado"] is True, "mas sinaliza que esta degradado")
        checar(r["setpoint_mm"] is not None, "com um setpoint valido de fallback")


def teste_solo_umido() -> None:
    print("\n[11] Solo acima da meta nao aciona")
    _, cliente = montar_app()
    with cliente:
        # 0,42 m³/m³ x 280 = 117,6 mm, acima do setpoint de ~104,8
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-08", "umidade_m3m3": 0.42})
        cliente.get("/api/v1/comando/esp32-08")
        j = cliente.get("/api/v1/comando/esp32-08").json()
        checar(j["acionar"] is False, "solo acima da meta NAO aciona")
        checar(j["motivo"] == "acima_do_setpoint", "com o motivo correto")


def teste_persistencia() -> None:
    print("\n[12] Estado de seguranca sobrevive ao reinicio")
    caminho = str(Path(tempfile.mkdtemp()) / "estado.json")
    servidor, cliente = montar_app(IRRIGACAO_ESTADO=caminho)
    with cliente:
        cliente.post("/api/v1/leitura",
                     json={"device_id": "esp32-09", "umidade_m3m3": 0.26})
        cliente.get("/api/v1/comando/esp32-09")
        cliente.get("/api/v1/comando/esp32-09")
        antes = cliente.get("/api/v1/status/esp32-09").json()
    # O lifespan grava ao sair do `with`.
    checar(Path(caminho).exists(), "o estado e gravado ao encerrar")
    checar(antes["consumo_dia_mm"] > 0, "houve consumo de cota a preservar")

    servidor2, cliente2 = montar_app(IRRIGACAO_ESTADO=caminho)
    with cliente2:
        depois = cliente2.get("/api/v1/status/esp32-09").json()
        checar(abs(depois["consumo_dia_mm"] - antes["consumo_dia_mm"]) < 1e-6,
               "a cota diaria ja gasta e restaurada apos o reinicio "
               "(senao um laco de restart encharcaria a lavoura)")
        checar(depois["irrigacoes_hoje"] == antes["irrigacoes_hoje"],
               "o contador de irrigacoes tambem sobrevive")


def teste_status_desconhecido() -> None:
    print("\n[13] Status de dispositivo desconhecido")
    _, cliente = montar_app()
    with cliente:
        r = cliente.get("/api/v1/status/nao-existe")
        checar(r.status_code == 404, "status de device desconhecido devolve 404")


def main() -> int:
    print("=" * 72)
    print("TESTES — backend FastAPI")
    print("=" * 72)

    teste_health()
    teste_sem_leitura()
    teste_ciclo_basico()
    teste_idempotencia()
    teste_confirmacao()
    teste_bomba_nao_ligou()
    teste_leitura_velha()
    teste_validacao()
    teste_autenticacao()
    teste_modelo_fora_do_ar()
    teste_solo_umido()
    teste_persistencia()
    teste_status_desconhecido()

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
