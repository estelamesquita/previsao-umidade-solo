#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes da persistencia em SQLite — rodam offline, sem rede e sem modelo.

Provam a lacuna que o banco veio fechar: depois de um reinicio do servidor,
o historico de leituras e o estado de seguranca (cota, irrigacoes, modo)
precisam continuar la. O "reinicio" e simulado recarregando o modulo
`servidor` apontando para o MESMO arquivo de banco.

    python teste_persistencia.py
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]    {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def montar(db: Path, estado: Path, **env: str):
    """Recarrega o servidor apontando para um banco especifico.

    Chamar duas vezes com o MESMO `db` simula desligar e religar o servidor:
    a configuracao e lida na importacao, entao cada 'boot' precisa de uma
    importacao limpa."""
    from fastapi.testclient import TestClient

    base = {
        "IRRIGACAO_DB": str(db),
        "IRRIGACAO_ESTADO": str(estado),
        "IRRIGACAO_CC_MM": "122.6",
        "IRRIGACAO_PM_MM": "71.8",
        "IRRIGACAO_PROFUNDIDADE_MM": "280.0",
        "IRRIGACAO_VAZAO_LPH": "400.0",
        "IRRIGACAO_AREA_M2": "50.0",
        "IRRIGACAO_PULSO_MIN_S": "5.0",
    }
    base.update(env)
    for k in list(os.environ):
        if k.startswith("IRRIGACAO_"):
            del os.environ[k]
    os.environ.update(base)

    import servidor
    importlib.reload(servidor)
    return servidor, TestClient(servidor.app)


DEV = "esp32-teste-01"


def teste_historico_sobrevive_reinicio() -> None:
    print("\n[1] O historico sobrevive ao reinicio")
    tmp = Path(tempfile.mkdtemp())
    db, estado = tmp / "irrigacao.db", tmp / "estado.json"

    # Primeiro boot: envia algumas leituras.
    _, c1 = montar(db, estado)
    with c1:
        for u in (80.0, 81.5, 83.0):
            r = c1.post("/api/v1/leitura",
                        json={"device_id": DEV, "umidade_mm": u})
            checar(r.status_code == 200, f"leitura {u} mm aceita")
        h = c1.get(f"/api/v1/historico/{DEV}").json()
        checar(len(h["pontos"]) == 3, "3 pontos no historico antes do reinicio")

    checar(db.exists(), "arquivo de banco foi criado em disco")

    # Segundo boot: MESMO banco, processo 'novo'.
    _, c2 = montar(db, estado)
    with c2:
        h = c2.get(f"/api/v1/historico/{DEV}").json()
        checar(len(h["pontos"]) == 3,
               "os 3 pontos continuam la depois do reinicio")
        umidades = [round(p["umidade_mm"], 1) for p in h["pontos"]]
        checar(umidades == [80.0, 81.5, 83.0],
               "valores e ordem cronologica preservados")


def teste_estado_sobrevive_reinicio() -> None:
    print("\n[2] Cota, irrigacoes e modo sobrevivem ao reinicio")
    tmp = Path(tempfile.mkdtemp())
    db, estado = tmp / "irrigacao.db", tmp / "estado.json"

    _, c1 = montar(db, estado)
    with c1:
        c1.post("/api/v1/leitura", json={"device_id": DEV, "umidade_mm": 80.0})
        c1.post(f"/api/v1/modo/{DEV}", json={"modo": "manual"})
        r = c1.post(f"/api/v1/acionar/{DEV}", json={"segundos": 120.0})
        checar(r.status_code == 200, "acionamento manual aceito")
        st1 = c1.get(f"/api/v1/status/{DEV}").json()
        checar(st1["irrigacoes_hoje"] >= 1, "ao menos 1 irrigacao registrada")
        checar(st1["consumo_dia_mm"] > 0.0, "cota diaria consumida > 0")
        consumo1 = st1["consumo_dia_mm"]

    _, c2 = montar(db, estado)
    with c2:
        st2 = c2.get(f"/api/v1/status/{DEV}").json()
        checar(st2["modo"] == "manual", "modo manual preservado")
        checar(st2["irrigacoes_hoje"] >= 1, "contador de irrigacoes preservado")
        checar(abs(st2["consumo_dia_mm"] - consumo1) < 1e-6,
               "cota diaria ja gasta reaplicada (nao zera num reinicio)")


def teste_sem_db_nao_persiste() -> None:
    print("\n[3] Sem IRRIGACAO_DB o comportamento antigo e preservado")
    from fastapi.testclient import TestClient

    tmp = Path(tempfile.mkdtemp())
    estado = tmp / "estado.json"
    for k in list(os.environ):
        if k.startswith("IRRIGACAO_"):
            del os.environ[k]
    os.environ["IRRIGACAO_ESTADO"] = str(estado)

    import servidor
    importlib.reload(servidor)
    with TestClient(servidor.app) as c:
        checar(servidor.banco is None, "banco fica None quando IRRIGACAO_DB ausente")
        h = c.get("/health").json()
        checar(h["banco"] is False, "/health reporta banco desativado")


def main() -> int:
    teste_historico_sobrevive_reinicio()
    teste_estado_sobrevive_reinicio()
    teste_sem_db_nao_persiste()
    print("\n" + "=" * 60)
    if FALHAS:
        print(f"{len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"  - {f}")
        return 1
    print("Todos os testes de persistencia passaram.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
