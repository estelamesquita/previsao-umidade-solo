#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel do sistema de irrigacao (Streamlit).

Mostra a umidade do solo e a previsao do tempo, o estado da bomba e permite
alternar entre automatico e manual.

    pip install streamlit
    streamlit run dashboard.py

O painel e um CLIENTE do backend: toda decisao e todo estado vivem em
`servidor.py`. Isso nao e detalhe de arquitetura, e a diferenca entre um
botao que funciona e um enfeite — quem consulta o servidor e o ESP32, entao
um modo guardado na sessao do Streamlit deixaria a bomba irrigando sozinha
com a tela escrito "Manual".

Configure o endereco do servidor na barra lateral (ou em IRRIGACAO_API_URL).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paleta — slots 1 e 2 da paleta de referencia, sem alteracao.
# A documentacao dela certifica que os tres primeiros slots passam em todos
# os pares (CVD dE 9,2 claro / 9,4 escuro; visao normal 24,0 / 20,9).
# ---------------------------------------------------------------------------

SERIE_1 = "#2a78d6"      # azul    — umidade do solo, chuva
SERIE_2 = "#eb6834"      # laranja — setpoint, ETo
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

BOM = "#0ca30c"          # status: bomba ligada
AVISO = "#fab219"        # status: modo manual
SERIO = "#ec835a"        # status: operando sem o modelo
CRITICO = "#d03b3b"      # status: falha de bomba / offline

API_PADRAO = os.getenv("IRRIGACAO_API_URL", "http://127.0.0.1:8000")
DEVICE_PADRAO = os.getenv("IRRIGACAO_DEVICE", "esp32-horta-01")

st.set_page_config(page_title="Irrigação", page_icon="💧", layout="wide")


# ---------------------------------------------------------------------------
# Cliente do backend
# ---------------------------------------------------------------------------


class Api:
    def __init__(self, base: str, chave: str | None, timeout: float = 6.0) -> None:
        self.base = base.rstrip("/")
        self.headers = {"X-API-Key": chave} if chave else {}
        self.timeout = timeout

    def _url(self, rota: str) -> str:
        return f"{self.base}{rota}"

    def get(self, rota: str, **params) -> tuple[bool, dict | str]:
        try:
            r = requests.get(self._url(rota), headers=self.headers,
                             params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            return False, f"servidor inacessível: {exc.__class__.__name__}"
        return self._resolver(r)

    def post(self, rota: str, corpo: dict | None = None) -> tuple[bool, dict | str]:
        try:
            r = requests.post(self._url(rota), headers=self.headers,
                              json=corpo or {}, timeout=self.timeout)
        except requests.RequestException as exc:
            return False, f"servidor inacessível: {exc.__class__.__name__}"
        return self._resolver(r)

    @staticmethod
    def _resolver(r: requests.Response) -> tuple[bool, dict | str]:
        if r.status_code == 401:
            return False, "chave de API ausente ou inválida"
        if r.status_code == 404:
            return False, "dispositivo desconhecido pelo servidor"
        if not r.ok:
            try:
                return False, str(r.json().get("detail", r.text))[:200]
            except ValueError:
                return False, f"HTTP {r.status_code}"
        try:
            return True, r.json()
        except ValueError:
            return False, "resposta do servidor não é JSON"


# ---------------------------------------------------------------------------
# Previsao do tempo — buscada direto da Open-Meteo, com cache
# ---------------------------------------------------------------------------


@st.cache_data(ttl=1800, show_spinner=False)
def carregar_previsao(lat: float, lon: float, dias: int = 3) -> pd.DataFrame:
    """Previsao horaria. O cache de 30 min evita bater na API a cada rerun."""
    import requests as rq

    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "forecast_days": dias, "wind_speed_unit": "ms",
        "hourly": ("temperature_2m,relative_humidity_2m,precipitation,"
                   "et0_fao_evapotranspiration,wind_speed_10m"),
    }
    r = rq.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------


def _vestir(gr: alt.Chart, legenda: bool = False) -> alt.Chart:
    """Cromo comum: superficie clara, grade e eixos em fio de cabelo."""
    gr = (
        gr.configure_view(stroke=None, fill=SUPERFICIE)
        .configure_axis(
            grid=True, gridColor=GRADE, gridWidth=1, domainColor=EIXO,
            tickColor=EIXO, labelColor=MUDO, titleColor=TINTA_2,
            labelFontSize=11, titleFontSize=12, titleFontWeight="normal",
        )
    )
    if legenda:
        gr = gr.configure_legend(
            labelColor=TINTA_2, titleColor=TINTA_2, labelFontSize=12,
            orient="top", direction="horizontal", title=None, offset=4,
        )
    return gr


# Eixo de tempo legivel: sem isso, uma janela curta imprime fracoes de segundo.
EIXO_TEMPO = alt.Axis(format="%d/%m %H:%M", labelAngle=0, tickCount=6)


def grafico_umidade(hist: pd.DataFrame, cc: float, pm: float) -> alt.Chart:
    """Umidade do solo e setpoint, com as faixas agronomicas ao fundo.

    Duas series na MESMA unidade (mm), um unico eixo — nunca eixo duplo.
    Capacidade de campo e ponto de murcha sao anotacoes, nao series.

    A escala y e declarada UMA vez e aplicada a todas as camadas. Marcas de
    area (o `mark_rect` da faixa) forcam a inclusao do zero, e como o eixo e
    compartilhado na camada isso anula o dominio das linhas — o grafico sai
    de 0 a 120 com a serie achatada no rodape.
    """
    longo = hist.melt(
        id_vars="t",
        value_vars=[c for c in ("umidade_mm", "setpoint_mm") if c in hist],
        var_name="serie", value_name="mm",
    ).dropna(subset=["mm"])
    longo["serie"] = longo["serie"].map(
        {"umidade_mm": "Umidade do solo", "setpoint_mm": "Setpoint"}
    )

    piso = min(pm, float(longo["mm"].min())) - 3.0
    teto = max(cc, float(longo["mm"].max())) + 3.0
    escala_y = alt.Scale(domain=[piso, teto], nice=False, zero=False)
    eixo_y = alt.Y("mm:Q", title="mm de lâmina", scale=escala_y)

    cor = alt.Color(
        "serie:N",
        scale=alt.Scale(domain=["Umidade do solo", "Setpoint"],
                        range=[SERIE_1, SERIE_2]),
        legend=alt.Legend(title=None, symbolType="stroke", symbolStrokeWidth=2),
    )

    seletor = alt.selection_point(fields=["t"], nearest=True,
                                  on="pointerover", empty=False)

    anot = pd.DataFrame({
        "v": [cc, pm],
        "rotulo": ["Capacidade de campo", "Ponto de murcha"],
    })

    faixa = alt.Chart(pd.DataFrame({"y0": [pm], "y1": [cc]})).mark_rect(
        fill=SERIE_1, opacity=0.05,
    ).encode(
        y=alt.Y("y0:Q", scale=escala_y, title=None),
        y2=alt.Y2("y1:Q"),
    )

    regras = alt.Chart(anot).mark_rule(
        strokeWidth=1, color=EIXO, strokeDash=[2, 2],
    ).encode(y=alt.Y("v:Q", scale=escala_y, title=None))

    texto = alt.Chart(anot).mark_text(
        align="left", dx=8, dy=-6, fontSize=10, color=MUDO,
    ).encode(
        y=alt.Y("v:Q", scale=escala_y, title=None),
        text="rotulo:N", x=alt.value(10),
    )

    linhas = alt.Chart(longo).mark_line(
        strokeWidth=2, interpolate="monotone",
    ).encode(
        x=alt.X("t:T", title=None, axis=EIXO_TEMPO),
        y=eixo_y, color=cor,
        strokeDash=alt.condition(alt.datum.serie == "Setpoint",
                                 alt.value([5, 3]), alt.value([0])),
    )

    pontos = alt.Chart(longo).mark_point(size=60, filled=True).encode(
        x=alt.X("t:T", axis=EIXO_TEMPO), y=eixo_y, color=cor,
        opacity=alt.condition(seletor, alt.value(1), alt.value(0)),
    )

    # Alvo de hover largo: o ponteiro nao precisa acertar a linha de 2px.
    alvo = alt.Chart(longo).mark_rule(strokeWidth=12, opacity=0).encode(
        x=alt.X("t:T", axis=EIXO_TEMPO),
        tooltip=[
            alt.Tooltip("t:T", title="Momento", format="%d/%m %H:%M"),
            alt.Tooltip("serie:N", title="Série"),
            alt.Tooltip("mm:Q", title="mm", format=".2f"),
        ],
    ).add_params(seletor)

    return _vestir(
        alt.layer(faixa, regras, texto, linhas, pontos, alvo)
        .properties(height=300),
        legenda=True,
    )


def grafico_irrigacao(hist: pd.DataFrame) -> alt.Chart | None:
    """Pulsos aplicados. Compartilha o eixo x com o grafico de umidade."""
    eventos = hist.dropna(subset=["irrigado_mm"])
    eventos = eventos[eventos["irrigado_mm"] > 0]
    if eventos.empty:
        return None
    return _vestir(
        alt.Chart(eventos)
        .mark_bar(width=5, color=SERIE_1, cornerRadiusTopLeft=2,
                  cornerRadiusTopRight=2)
        .encode(
            x=alt.X("t:T", title=None, axis=EIXO_TEMPO),
            y=alt.Y("irrigado_mm:Q", title="mm aplicados"),
            tooltip=[
                alt.Tooltip("t:T", title="Momento", format="%d/%m %H:%M"),
                alt.Tooltip("irrigado_mm:Q", title="Lâmina (mm)", format=".2f"),
            ],
        )
        .properties(height=110)
    )


def grafico_balanco(prev: pd.DataFrame) -> alt.Chart:
    """Chuva e ETo previstas.

    As duas estao em mm, entao dividem UM eixo — sem eixo duplo, que e o
    erro mais comum em painel e inventa correlacao que o dado nao tem. As
    marcas diferem de proposito: chuva e evento (barra), ETo e demanda
    continua (linha).
    """
    chuva = alt.Chart(prev).mark_bar(width=3, color=SERIE_1).encode(
        x=alt.X("time:T", title=None, axis=EIXO_TEMPO),
        y=alt.Y("precipitation:Q", title="mm por hora"),
        tooltip=[
            alt.Tooltip("time:T", title="Momento", format="%d/%m %H:%M"),
            alt.Tooltip("precipitation:Q", title="Chuva (mm)", format=".2f"),
        ],
    )
    eto = alt.Chart(prev).mark_line(strokeWidth=2, color=SERIE_2,
                                    interpolate="monotone").encode(
        x="time:T",
        y="et0_fao_evapotranspiration:Q",
        tooltip=[
            alt.Tooltip("time:T", title="Momento", format="%d/%m %H:%M"),
            alt.Tooltip("et0_fao_evapotranspiration:Q", title="ETo (mm)",
                        format=".3f"),
        ],
    )
    legenda = alt.Chart(pd.DataFrame({
        "serie": ["Chuva prevista", "ETo prevista"],
    })).mark_point(size=0).encode(
        color=alt.Color("serie:N",
                        scale=alt.Scale(domain=["Chuva prevista", "ETo prevista"],
                                        range=[SERIE_1, SERIE_2]),
                        legend=alt.Legend(title=None)),
    )
    return _vestir(alt.layer(chuva, eto, legenda).properties(height=240),
                   legenda=True)


def grafico_temperatura(prev: pd.DataFrame) -> alt.Chart:
    """Temperatura prevista. Serie unica: o titulo a nomeia, sem legenda."""
    return _vestir(
        alt.Chart(prev)
        .mark_line(strokeWidth=2, color=SERIE_1, interpolate="monotone")
        .encode(
            x=alt.X("time:T", title=None, axis=EIXO_TEMPO),
            y=alt.Y("temperature_2m:Q", title="°C",
                    scale=alt.Scale(zero=False, nice=True)),
            tooltip=[
                alt.Tooltip("time:T", title="Momento", format="%d/%m %H:%M"),
                alt.Tooltip("temperature_2m:Q", title="Temperatura (°C)",
                            format=".1f"),
                alt.Tooltip("relative_humidity_2m:Q", title="UR (%)",
                            format=".0f"),
            ],
        )
        .properties(height=200)
    )


# ---------------------------------------------------------------------------
# Componentes de estado
# ---------------------------------------------------------------------------


def selo(cor: str, icone: str, titulo: str, detalhe: str) -> str:
    """Selo de estado. Icone + texto SEMPRE: a cor nunca carrega o sentido
    sozinha (daltonismo, impressao, modo de alto contraste)."""
    return (
        f'<div style="border-left:4px solid {cor};background:{SUPERFICIE};'
        f'border:1px solid {GRADE};border-left-width:4px;border-radius:10px;'
        f'padding:14px 16px">'
        f'<div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;'
        f'color:{MUDO}">{titulo}</div>'
        f'<div style="font-size:21px;font-weight:600;color:{TINTA};'
        f'margin-top:4px">{icone} {detalhe}</div></div>'
    )


# ---------------------------------------------------------------------------
# Aplicacao
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("💧 Painel de irrigação")

    with st.sidebar:
        st.header("Conexão")
        base = st.text_input("Servidor", API_PADRAO)
        chave = st.text_input("Chave de API", type="password",
                              help="Deixe vazio se o servidor subiu sem "
                                   "IRRIGACAO_API_KEY.")
        device = st.text_input("Dispositivo", DEVICE_PADRAO)

        st.header("Local")
        lat = st.number_input("Latitude", value=-23.5505, format="%.4f")
        lon = st.number_input("Longitude", value=-46.6333, format="%.4f")

        st.header("Exibição")
        janela_h = st.select_slider("Janela do histórico",
                                    options=[3, 6, 12, 24, 48, 72], value=24)
        auto = st.checkbox("Atualizar automaticamente", value=True)
        if st.button("Atualizar agora", width="stretch"):
            st.rerun()

    api = Api(base, chave or None, )

    ok, status = api.get(f"/api/v1/status/{device}")
    if not ok:
        st.error(f"Não foi possível falar com o servidor: **{status}**")
        st.info(
            "Confira se o backend está no ar:\n\n"
            "```bash\nuvicorn servidor:app --host 0.0.0.0 --port 8000\n```\n\n"
            "Se o dispositivo ainda não enviou nenhuma leitura, ele não existe "
            "para o servidor — faça o ESP32 publicar em `POST /api/v1/leitura` "
            "uma vez."
        )
        return

    assert isinstance(status, dict)

    # ---------------- estado ----------------
    modo = status["modo"]
    bomba = status["bomba_ligada"]
    restantes = status["segundos_restantes"]
    umidade = status.get("umidade_mm")
    setpoint = status.get("setpoint_mm")
    cc = status["capacidade_campo_mm"]
    pm = status["ponto_murcha_mm"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if bomba:
            st.markdown(selo(BOM, "⬤", "Bomba",
                             f"Ligada · {restantes:.0f} s"), unsafe_allow_html=True)
        else:
            st.markdown(selo(MUDO, "○", "Bomba", "Desligada"),
                        unsafe_allow_html=True)
    with c2:
        if modo == "manual":
            st.markdown(selo(AVISO, "✋", "Modo", "Manual"), unsafe_allow_html=True)
        else:
            st.markdown(selo(SERIE_1, "⚙", "Modo", "Automático"),
                        unsafe_allow_html=True)
    with c3:
        if not status["online"]:
            st.markdown(selo(CRITICO, "⚠", "Sensor", "Offline"),
                        unsafe_allow_html=True)
        else:
            idade = status.get("ultima_leitura_s") or 0
            st.markdown(selo(BOM, "✓", "Sensor", f"Há {idade:.0f} s"),
                        unsafe_allow_html=True)
    with c4:
        origem = status["origem_setpoint"]
        if origem == "modelo":
            st.markdown(selo(SERIE_1, "🧠", "Setpoint", "Do modelo"),
                        unsafe_allow_html=True)
        else:
            st.markdown(selo(SERIO, "▲", "Setpoint", "Regra fixa"),
                        unsafe_allow_html=True)

    if status["falhas_bomba"] > 0:
        st.warning(
            f"⚠ **{status['falhas_bomba']} falha(s) de acionamento** — a bomba "
            "não executou um comando, ou o dispositivo não confirmou. Verifique "
            "a válvula e a alimentação."
        )
    if origem != "modelo":
        st.info(
            "▲ Operando **sem o modelo preditivo**: o setpoint é uma fração fixa "
            "da água disponível. O controle segue funcionando — isto é "
            "degradação planejada, não falha."
        )

    st.divider()

    # ---------------- numeros ----------------
    m1, m2, m3, m4 = st.columns(4)
    if umidade is not None:
        disponivel = cc - pm
        frac = (umidade - pm) / disponivel if disponivel > 0 else 0.0
        m1.metric("Umidade atual", f"{umidade:.1f} mm",
                  f"{frac:.0%} da água útil")
    if setpoint is not None and umidade is not None:
        m2.metric("Setpoint", f"{setpoint:.1f} mm",
                  f"{umidade - setpoint:+.1f} mm", delta_color="off")
    m3.metric("Consumo do dia", f"{status['consumo_dia_mm']:.1f} mm",
              f"de {status['cota_diaria_mm']:.0f} mm", delta_color="off")
    m4.metric("Irrigações hoje", status["irrigacoes_hoje"],
              f"motivo: {status['ultimo_motivo']}", delta_color="off")

    # ---------------- controle ----------------
    st.subheader("Controle")
    ctl1, ctl2 = st.columns([1, 2])

    with ctl1:
        novo = "manual" if modo == "auto" else "auto"
        rotulo = ("✋ Passar para MANUAL" if novo == "manual"
                  else "⚙ Voltar ao AUTOMÁTICO")
        if st.button(rotulo, width="stretch", type="primary"):
            ok2, resp = api.post(f"/api/v1/modo/{device}", {"modo": novo})
            if ok2:
                st.success(f"Modo alterado para **{novo}**.")
                st.rerun()
            else:
                st.error(f"Falha ao alterar o modo: {resp}")

    with ctl2:
        if modo == "manual":
            seg = st.number_input("Duração do pulso (s)", min_value=1,
                                  max_value=3600, value=120, step=10)
            if st.button(f"💧 Ligar a bomba por {seg} s",
                         width="stretch", disabled=bomba):
                ok3, resp = api.post(f"/api/v1/acionar/{device}",
                                     {"segundos": float(seg)})
                if ok3 and isinstance(resp, dict):
                    st.success(
                        f"Pulso de **{resp['segundos']:.0f} s** enviado "
                        f"(comando `{resp['comando_id']}`). O ESP32 executa na "
                        "próxima consulta."
                    )
                    st.rerun()
                else:
                    st.error(f"Acionamento recusado: {resp}")
            st.caption(
                "O manual dispensa a decisão do controlador, **não os limites**: "
                "o pulso continua sujeito ao máximo do atuador, à cota diária e "
                "à capacidade de campo."
            )
        else:
            st.caption(
                "Em **automático** o controlador PID decide sozinho a partir do "
                "setpoint. Passe para manual para acionar a bomba por conta."
            )

    st.divider()

    # ---------------- graficos ----------------
    st.subheader("Umidade do solo")
    okh, hist = api.get(f"/api/v1/historico/{device}", horas=janela_h)
    if okh and isinstance(hist, dict) and hist["pontos"]:
        dfh = pd.DataFrame(hist["pontos"])
        dfh["t"] = pd.to_datetime(dfh["t"], unit="s")
        st.altair_chart(grafico_umidade(dfh, cc, pm), width="stretch")

        g = grafico_irrigacao(dfh)
        if g is not None:
            st.altair_chart(g, width="stretch")
        else:
            st.caption("Nenhuma irrigação registrada nesta janela.")

        with st.expander("Ver os dados em tabela"):
            # Toda leitura e alcancavel sem depender de passar o mouse.
            tabela = dfh.rename(columns={
                "t": "Momento", "umidade_mm": "Umidade (mm)",
                "setpoint_mm": "Setpoint (mm)", "irrigado_mm": "Aplicado (mm)",
            })
            st.dataframe(tabela.sort_values("Momento", ascending=False),
                         width="stretch", hide_index=True)
    else:
        st.info(
            f"Ainda não há histórico nas últimas {janela_h} h. O gráfico aparece "
            "assim que o dispositivo enviar leituras."
        )

    st.subheader("Previsão do tempo")
    try:
        prev = carregar_previsao(lat, lon)
    except Exception as exc:  # noqa: BLE001 — o painel nao pode cair por isso
        st.warning(f"Previsão indisponível no momento ({type(exc).__name__}).")
        prev = None

    if prev is not None and not prev.empty:
        st.caption("Chuva e ETo dividem um eixo porque ambas estão em mm — "
                   "a chuva entra no solo, a ETo sai dele.")
        st.altair_chart(grafico_balanco(prev), width="stretch")
        st.altair_chart(grafico_temperatura(prev), width="stretch")

        saldo = prev["precipitation"].sum() - prev["et0_fao_evapotranspiration"].sum()
        p1, p2, p3 = st.columns(3)
        p1.metric("Chuva prevista (72 h)", f"{prev['precipitation'].sum():.1f} mm")
        p2.metric("ETo prevista (72 h)",
                  f"{prev['et0_fao_evapotranspiration'].sum():.1f} mm")
        p3.metric("Saldo", f"{saldo:+.1f} mm",
                  "entrada supera a demanda" if saldo > 0 else "demanda supera a entrada",
                  delta_color="off")

        with st.expander("Ver a previsão em tabela"):
            st.dataframe(
                prev.rename(columns={
                    "time": "Momento", "temperature_2m": "Temp (°C)",
                    "relative_humidity_2m": "UR (%)",
                    "precipitation": "Chuva (mm)",
                    "et0_fao_evapotranspiration": "ETo (mm)",
                    "wind_speed_10m": "Vento (m/s)",
                }),
                width="stretch", hide_index=True,
            )

    st.caption(
        f"Servidor: `{base}` · dispositivo: `{device}` · "
        f"atualizado {datetime.now().strftime('%H:%M:%S')}"
    )

    if auto:
        # Fragmento com recarga propria: a pagina inteira nao pisca.
        st.empty()
        import time as _t
        _t.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
