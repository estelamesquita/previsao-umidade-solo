#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes de sanidade da pipeline de features — rodam offline, sem chamar a API.

O teste central e o de causalidade: perturbar uma linha futura da serie e
exigir que NENHUMA feature de instantes anteriores mude. Em pipeline de serie
temporal esse e o bug que passa silencioso pela revisao de codigo, infla as
metricas e so aparece em producao.

    python teste_vazamento.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from treinar_modelo_solo import (
    ConfigFeatures,
    ajustar_calibracao,
    alvo_para_nivel,
    construir_alvo,
    construir_features,
    estimar_limites_hidricos,
    predicoes_baseline,
    reconstruir_nivel,
)

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]    {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def serie_sintetica(n_horas: int = 2000, semente: int = 7) -> pd.DataFrame:
    """Serie horaria plausivel: secagem exponencial com pulsos de chuva."""
    rng = np.random.default_rng(semente)
    idx = pd.date_range("2024-01-01", periods=n_horas, freq="1h", tz="America/Sao_Paulo")
    hora = idx.hour.to_numpy()

    temp = 22.0 + 6.0 * np.sin(2 * np.pi * (hora - 9) / 24.0) + rng.normal(0, 1.0, n_horas)
    ur = np.clip(75.0 - 15.0 * np.sin(2 * np.pi * (hora - 9) / 24.0), 25.0, 100.0)
    vento = np.clip(rng.gamma(2.0, 1.2, n_horas), 0.1, None)
    rad = np.clip(800.0 * np.sin(np.pi * (hora - 6) / 12.0), 0.0, None)
    eto = np.clip(rad / 2.45e3 + 0.01 * (temp - 20.0), 0.0, None)
    chuva = rng.gamma(2.0, 1.5, n_horas) * (rng.random(n_horas) < 0.04)

    # Reservatorio simples: entra chuva, sai ETo, com escoamento no topo.
    agua = np.empty(n_horas)
    agua[0] = 90.0
    for i in range(1, n_horas):
        agua[i] = np.clip(agua[i - 1] + 0.8 * chuva[i] - 1.2 * eto[i], 70.0, 105.0)

    df = pd.DataFrame(
        {
            "temperature_2m": temp,
            "relative_humidity_2m": ur,
            "wind_speed_10m": vento,
            "shortwave_radiation": rad,
            "precipitation": chuva,
            "et0_fao_evapotranspiration": eto,
            "vpd_kpa": np.clip(0.6108 * np.exp(17.27 * temp / (temp + 237.3))
                               * (1 - ur / 100.0), 0, None),
            "agua_zona_raiz_mm": agua,
        },
        index=idx,
    )
    df.index.name = "time"
    df["balanco_horario_mm"] = df["precipitation"] - df["et0_fao_evapotranspiration"]
    return df


# ---------------------------------------------------------------------------


def teste_causalidade(cfg: ConfigFeatures) -> None:
    """Features em t nao podem depender de nada em t' > t."""
    print("\n[1] Causalidade das features (o teste que importa)")
    df = serie_sintetica()
    X_base = construir_features(df, cfg)

    corte = 1500
    df_perturbado = df.copy()
    # Perturbacao violenta em TODAS as colunas depois do corte.
    for coluna in df_perturbado.columns:
        df_perturbado.iloc[corte:, df_perturbado.columns.get_loc(coluna)] *= 7.5
    X_perturbado = construir_features(df_perturbado, cfg)

    passado_base = X_base.iloc[:corte]
    passado_perturbado = X_perturbado.iloc[:corte]
    iguais = passado_base.equals(passado_perturbado)

    if not iguais:
        difere = [
            c for c in passado_base.columns
            if not passado_base[c].equals(passado_perturbado[c])
        ]
        print(f"          features contaminadas: {difere}")
    checar(iguais, f"alterar t>={corte} nao mexe em nenhuma das "
                   f"{X_base.shape[1]} features de t<{corte}")

    # O futuro, por construcao, deve ter mudado — senao o teste nao testa nada.
    checar(
        not X_base.iloc[corte + 200:].equals(X_perturbado.iloc[corte + 200:]),
        "a perturbacao de fato altera as features posteriores (controle)",
    )


def teste_alvo_e_reconstrucao(cfg: ConfigFeatures) -> None:
    """O alvo olha exatamente h horas a frente e a reconstrucao e exata."""
    print("\n[2] Alvo e reconstrucao do nivel")
    df = serie_sintetica()
    h = 12

    alvo_nivel = construir_alvo(df, h, "nivel")
    checar(
        np.isclose(alvo_nivel.iloc[0], df["agua_zona_raiz_mm"].iloc[h]),
        f"alvo 'nivel' em t0 e igual a agua observada em t0+{h}h",
    )
    checar(
        alvo_nivel.iloc[-h:].isna().all(),
        f"as ultimas {h} horas ficam sem alvo (sem preenchimento inventado)",
    )

    alvo_delta = construir_alvo(df, h, "delta")
    checar(
        np.isclose(
            alvo_delta.iloc[0],
            df["agua_zona_raiz_mm"].iloc[h] - df["agua_zona_raiz_mm"].iloc[0],
        ),
        "alvo 'delta' e a variacao sobre o instante atual",
    )

    X = construir_features(df, cfg)
    valido = X.notna().all(axis=1) & alvo_delta.notna()
    Xv, yv = X.loc[valido], alvo_delta.loc[valido]
    reconstruido = alvo_para_nivel(yv, Xv, "delta")
    esperado = alvo_nivel.loc[valido].to_numpy()
    checar(
        np.allclose(reconstruido, esperado),
        "delta + agua_atual reconstroi exatamente o nivel observado",
    )
    checar(
        np.allclose(reconstruir_nivel(yv.to_numpy(), Xv, "delta"), esperado),
        "reconstruir_nivel e consistente com alvo_para_nivel",
    )
    checar(
        np.allclose(reconstruir_nivel(esperado, Xv, "nivel"), esperado),
        "formulacao 'nivel' passa a predicao adiante sem alterar",
    )


def teste_baselines(cfg: ConfigFeatures) -> None:
    """Baselines usam so informacao disponivel em t."""
    print("\n[3] Baselines")
    df = serie_sintetica()
    h = 12
    X = construir_features(df, cfg)
    y = construir_alvo(df, h, "delta")
    idx = X.loc[X.notna().all(axis=1) & y.notna()].index

    bases = predicoes_baseline(df, idx, h)
    checar("persistencia" in bases, "baseline de persistencia presente")
    checar(
        np.allclose(bases["persistencia"], df["agua_zona_raiz_mm"].reindex(idx)),
        "persistencia = agua observada em t",
    )
    checar("sazonal_24h" in bases, "baseline sazonal presente para h<=24")

    deslocamento = 24 - h
    esperado = df["agua_zona_raiz_mm"].shift(deslocamento).reindex(idx)
    checar(
        np.allclose(bases["sazonal_24h"], esperado, equal_nan=True),
        f"sazonal_24h usa t-{deslocamento}h, ja observado em t",
    )
    checar(
        predicoes_baseline(df, idx, 48).keys() == {"persistencia"},
        "para h>24 a baseline sazonal e omitida (seria nao observavel)",
    )


def teste_grade_e_limites(cfg: ConfigFeatures) -> None:
    """Buracos na serie nao podem virar lag silenciosamente errado."""
    print("\n[4] Grade temporal e limites hidricos")
    df = serie_sintetica(n_horas=600)

    limites = estimar_limites_hidricos(df["agua_zona_raiz_mm"])
    checar(
        limites["capacidade_campo_mm"] > limites["ponto_murcha_mm"],
        "capacidade de campo acima do ponto de murcha",
    )
    checar(
        np.isclose(
            limites["agua_disponivel_total_mm"],
            limites["capacidade_campo_mm"] - limites["ponto_murcha_mm"],
        ),
        "agua disponivel total e coerente com cc - pm",
    )
    try:
        estimar_limites_hidricos(pd.Series([50.0] * 100))
        checar(False, "serie constante e rejeitada")
    except ValueError:
        checar(True, "serie constante e rejeitada com ValueError")

    # Remove um bloco de horas: as features vizinhas devem ficar NaN, nunca
    # saltar o buraco e tratar t-30h como se fosse t-24h.
    com_buraco = df.drop(df.index[200:230])
    from treinar_modelo_solo import _validar_grade_horaria

    reindexado = _validar_grade_horaria(com_buraco)
    checar(
        len(reindexado) == len(df),
        "grade e reindexada para horaria contigua apos remocao de horas",
    )
    X_buraco = construir_features(reindexado, cfg)

    checar(
        X_buraco.iloc[200:230].isna().any(axis=1).all(),
        "toda hora faltante tem ao menos uma feature NaN (amostra descartada)",
    )
    checar(
        X_buraco.notna().all(axis=1).sum() > 0,
        "apos o buraco a serie volta a gerar amostras validas",
    )

    # O risco real do buraco nao e NaN, e o lag trocar de offset: sem
    # reindexar, `shift(24)` andaria 24 LINHAS e, com 30 horas ausentes,
    # pegaria t-54h achando que pegou t-24h. Com a grade contigua, um lag que
    # cai dentro do buraco tem de ser NaN, e um que cai fora tem de bater
    # exatamente com o valor daquele timestamp.
    agua = reindexado["agua_zona_raiz_mm"]
    alvo_dentro = X_buraco.index[240]            # t-24h == indice 216, no buraco
    checar(
        np.isnan(X_buraco.loc[alvo_dentro, "agua_lag24h_mm"]),
        "lag de 24h que cai dentro do buraco e NaN (nao pula para outra hora)",
    )
    alvo_fora = X_buraco.index[260]              # t-24h == indice 236, observado
    checar(
        np.isclose(
            X_buraco.loc[alvo_fora, "agua_lag24h_mm"],
            agua.loc[alvo_fora - pd.Timedelta(hours=24)],
        ),
        "lag de 24h fora do buraco bate com o valor em t-24h exatas",
    )
    checar(
        all(
            np.isclose(
                X_buraco.loc[alvo_fora, f"agua_lag{lag}h_mm"],
                agua.loc[alvo_fora - pd.Timedelta(hours=lag)],
                equal_nan=True,
            )
            for lag in cfg.lags_umidade
        ),
        f"todos os lags {list(cfg.lags_umidade)} referenciam o offset correto em horas",
    )


def teste_particao_temporal(cfg: ConfigFeatures) -> None:
    """Treino e holdout nao podem se sobrepor no tempo."""
    print("\n[5] Particao treino/holdout")
    from sklearn.model_selection import TimeSeriesSplit

    df = serie_sintetica()
    h = 24
    X = construir_features(df, cfg)
    y = construir_alvo(df, h, "delta")
    valido = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valido], y.loc[valido]

    corte = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:corte], X.iloc[corte:]
    checar(X_tr.index.max() < X_te.index.min(), "holdout e estritamente posterior ao treino")

    cv = TimeSeriesSplit(n_splits=5, gap=h)
    for i, (tr, va) in enumerate(cv.split(X_tr)):
        folga = (X_tr.index[va[0]] - X_tr.index[tr[-1]]).total_seconds() / 3600.0
        checar(
            folga > h,
            f"dobra {i}: folga de {folga:.0f}h entre fim do treino e inicio da "
            f"validacao (> h={h}h)",
        )
        checar(max(tr) < min(va), f"dobra {i}: indices de treino todos anteriores")


def teste_calibracao() -> None:
    """O fator de encolhimento tem de recuperar a teoria: skill = rho^2."""
    print("\n[6] Calibracao da amplitude")
    rng = np.random.default_rng(3)
    n = 20000
    sinal = rng.normal(0, 1, n)
    ruido = rng.normal(0, 1, n)

    # Observado tem o sinal; a previsao tem o sinal mas amplitude 3x maior.
    d_obs = sinal + ruido                 # rho com `sinal` = 1/sqrt(2)
    d_prev = 3.0 * sinal
    cal = ajustar_calibracao(d_obs, d_prev)

    rho = float(np.corrcoef(d_obs, d_prev)[0, 1])
    checar(np.isclose(cal["rho"], rho, atol=1e-9), "rho reportado bate com o empirico")
    # A escala otima e rho * sigma_obs / sigma_prev.
    esperado = rho * np.std(d_obs) / np.std(d_prev)
    checar(
        np.isclose(cal["escala"], esperado, rtol=1e-3),
        f"escala otima = rho*sigma_obs/sigma_prev ({cal['escala']:.4f})",
    )

    mse_zero = float(np.mean(d_obs**2))
    mse_bruto = float(np.mean((d_obs - d_prev) ** 2))
    mse_cal = float(np.mean((d_obs - cal["escala"] * d_prev) ** 2))
    checar(
        mse_bruto > mse_zero,
        "amplitude 3x excessiva e PIOR que prever zero (motiva a calibracao)",
    )
    checar(mse_cal < mse_zero, "apos calibrar, o modelo supera prever zero")
    skill = 1.0 - mse_cal / mse_zero
    checar(
        np.isclose(skill, cal["skill_teorico"], atol=5e-3),
        f"skill obtido ({skill:.4f}) == rho^2 ({cal['skill_teorico']:.4f})",
    )

    # Previsao sem informacao: a escala tem de colapsar para ~0.
    cal_ruido = ajustar_calibracao(d_obs, rng.normal(0, 1, n))
    checar(
        abs(cal_ruido["escala"]) < 0.05,
        f"previsao sem sinal recebe escala ~0 ({cal_ruido['escala']:+.4f})",
    )

    # Casos degenerados nao podem explodir.
    cal_const = ajustar_calibracao(d_obs, np.zeros(n))
    checar(cal_const["escala"] == 1.0, "previsao constante devolve escala=1.0 sem dividir por zero")
    cal_curto = ajustar_calibracao(d_obs[:10], d_prev[:10])
    checar(cal_curto["escala"] == 1.0, "amostra pequena devolve escala=1.0")

    # reconstruir_nivel deve aplicar a escala sobre a variacao, nao sobre o nivel.
    X = pd.DataFrame({"agua_atual_mm": [100.0, 100.0]})
    nivel = reconstruir_nivel(np.array([4.0, -4.0]), X, "delta", escala=0.5)
    checar(
        np.allclose(nivel, [102.0, 98.0]),
        "escala incide sobre a variacao prevista, preservando o nivel atual",
    )


def main() -> int:
    cfg = ConfigFeatures()
    print("=" * 72)
    print("TESTES DE SANIDADE — pipeline de umidade do solo")
    print("=" * 72)

    teste_causalidade(cfg)
    teste_alvo_e_reconstrucao(cfg)
    teste_baselines(cfg)
    teste_grade_e_limites(cfg)
    teste_particao_temporal(cfg)
    teste_calibracao()

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
