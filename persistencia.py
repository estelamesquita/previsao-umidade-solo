#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistencia em SQLite para o backend de irrigacao.

Fecha a lacuna nº 1 do projeto: o historico de leituras vivia so em RAM
(`deque`) e sumia a cada reinicio do servidor. Com `IRRIGACAO_DB` apontando
para um arquivo, cada leitura vira uma linha, e no boot o servidor:

  * recarrega o estado de SEGURANCA (cota diaria, irrigacoes, falhas, modo);
  * reidrata o `deque` de cada dispositivo, para os graficos do painel
    sobreviverem ao reinicio.

Por que SQLite e nao Postgres: o volume e minusculo (uma leitura a cada
poucos minutos), e um arquivo unico roda sem servico separado — ideal para
uma VPS gratuita de um unico no. A biblioteca `sqlite3` e da stdlib, entao
isto NAO adiciona nenhuma dependencia.

Duas decisoes que importam
--------------------------
* **O relogio e de parede (`time.time`), nao monotonico.** O `deque` guarda
  `time.time()` e o endpoint de historico compara com `time.time()`; gravar
  outra coisa quebraria a serie depois de um reinicio.
* **A conexao e protegida por um `threading.Lock`.** O servidor grava as
  leituras num thread do pool e salva o estado em outro; sem a trava, dois
  threads usando a mesma conexao SQLite ao mesmo tempo dao erro.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

LOG = logging.getLogger("irrigacao.persistencia")

# Um ponto do historico, na mesma ordem do `deque` do servidor:
# (t_epoch, umidade_mm, setpoint_mm | None, irrigado_mm | None)
Ponto = tuple[float, float, float | None, float | None]

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS leituras (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT NOT NULL,
    t           REAL NOT NULL,          -- epoch do SERVIDOR (relogio de parede)
    umidade_mm  REAL NOT NULL,
    setpoint_mm REAL,
    irrigado_mm REAL
);
CREATE INDEX IF NOT EXISTS ix_leituras_dev_t ON leituras (device_id, t);

CREATE TABLE IF NOT EXISTS estado (
    device_id      TEXT PRIMARY KEY,
    consumo_dia_mm REAL    NOT NULL DEFAULT 0,
    irrigacoes     INTEGER NOT NULL DEFAULT 0,
    falhas_bomba   INTEGER NOT NULL DEFAULT 0,
    modo           TEXT    NOT NULL DEFAULT 'auto',
    salvo_em       REAL    NOT NULL
);
"""


class Banco:
    """Camada fina sobre uma conexao SQLite. Todos os metodos sao sincronos
    e bloqueantes de proposito: o servidor os chama de um threadpool, nunca
    no laco de eventos."""

    def __init__(self, caminho: str | Path) -> None:
        self.caminho = Path(caminho)
        if self.caminho.parent and not self.caminho.parent.exists():
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: a conexao e usada por threads do pool do
        # servidor; a serializacao fica por conta de `self._lock`.
        self._conn = sqlite3.connect(str(self.caminho), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL deixa leitura e escrita concorrerem sem travar o arquivo;
            # NORMAL troca um fsync por requisicao por desempenho, aceitavel
            # para telemetria (perder o ultimo ponto num crash nao e critico).
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_ESQUEMA)
            self._conn.commit()
        LOG.info("banco de dados em %s", self.caminho)

    # -- escrita ----------------------------------------------------------

    def gravar_leituras(self, linhas: list[tuple[str, float, float,
                                                 float | None, float | None]]) -> None:
        """Insere um lote de (device_id, t, umidade_mm, setpoint_mm, irrigado_mm)."""
        if not linhas:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO leituras (device_id, t, umidade_mm, setpoint_mm, "
                "irrigado_mm) VALUES (?, ?, ?, ?, ?)",
                linhas,
            )
            self._conn.commit()

    def salvar_estado(self, dispositivos: dict[str, dict[str, Any]]) -> None:
        """Upsert do estado de seguranca de cada dispositivo."""
        if not dispositivos:
            return
        agora = time.time()
        linhas = [
            (did, float(d.get("consumo_dia_mm", 0.0)), int(d.get("irrigacoes", 0)),
             int(d.get("falhas_bomba", 0)), str(d.get("modo", "auto")), agora)
            for did, d in dispositivos.items()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO estado (device_id, consumo_dia_mm, irrigacoes, "
                "falhas_bomba, modo, salvo_em) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(device_id) DO UPDATE SET "
                "consumo_dia_mm=excluded.consumo_dia_mm, "
                "irrigacoes=excluded.irrigacoes, "
                "falhas_bomba=excluded.falhas_bomba, "
                "modo=excluded.modo, salvo_em=excluded.salvo_em",
                linhas,
            )
            self._conn.commit()

    def podar(self, mais_velho_que_t: float) -> int:
        """Remove leituras anteriores ao corte. Devolve quantas apagou."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM leituras WHERE t < ?",
                                     (mais_velho_que_t,))
            self._conn.commit()
            return cur.rowcount

    # -- leitura (boot) ---------------------------------------------------

    def carregar_estado(self) -> dict[str, Any]:
        """Devolve o estado no MESMO formato que `Registro.importar` espera:
        {"salvo_em": <mais recente>, "dispositivos": {device_id: {...}}}."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT device_id, consumo_dia_mm, irrigacoes, falhas_bomba, "
                "modo, salvo_em FROM estado"
            )
            linhas = cur.fetchall()
        dispositivos: dict[str, dict[str, Any]] = {}
        salvo_em = 0.0
        for did, consumo, irrig, falhas, modo, salvo in linhas:
            dispositivos[did] = {
                "consumo_dia_mm": consumo, "irrigacoes": irrig,
                "falhas_bomba": falhas, "modo": modo,
            }
            salvo_em = max(salvo_em, float(salvo))
        return {"salvo_em": salvo_em, "dispositivos": dispositivos}

    def carregar_historico(self, device_id: str, desde_t: float,
                           limite: int) -> list[Ponto]:
        """Ultimos `limite` pontos do dispositivo desde `desde_t`, em ordem
        cronologica (mais antigo primeiro), pronto para alimentar o `deque`."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT t, umidade_mm, setpoint_mm, irrigado_mm FROM leituras "
                "WHERE device_id = ? AND t >= ? ORDER BY t DESC LIMIT ?",
                (device_id, desde_t, limite),
            )
            linhas = cur.fetchall()
        linhas.reverse()
        return [(float(t), float(u),
                 None if s is None else float(s),
                 None if i is None else float(i)) for (t, u, s, i) in linhas]

    def dispositivos_com_historico(self, desde_t: float) -> list[str]:
        """device_ids que tem alguma leitura desde o corte."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT device_id FROM leituras WHERE t >= ?",
                (desde_t,),
            )
            return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
