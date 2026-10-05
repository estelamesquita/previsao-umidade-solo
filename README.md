# Pipeline ETo + previsão de umidade do solo

Coleta meteorológica (Open-Meteo), cálculo de evapotranspiração de referência e
modelo de previsão de umidade do solo / déficit hídrico para +12h e +24h.

Resultado medido em São Paulo (3 anos de ERA5, holdout cronológico de 7 meses /
5.221 horas, zona de raízes 0–28 cm com 50,8 mm de água disponível):

| horizonte | modelo | RMSE | MAE | R² | skill vs persistência |
|---|---|---|---|---|---|
| +12h | Random Forest | 2,276 mm (0,0081 m³/m³) | 1,409 mm | 0,889 | +0,060 |
| +24h | HistGradientBoosting | 3,019 mm (0,0108 m³/m³) | 1,939 mm | 0,804 | +0,089 |

RMSE equivale a 4,5% e 5,9% da água disponível total. **Leia a seção 3 antes de
olhar o R²** — ele por si só não diz se o modelo serve.

```
coleta_open_meteo.py     coleta horária/diária + ETo (Hargreaves e Penman-Monteith)
treinar_modelo_solo.py   dataset supervisionado, treino, avaliação, export joblib
teste_vazamento.py       testes de causalidade das features (roda offline)
apresentacao.html        deck de 14 slides (abra no navegador; F = tela cheia, N = notas)
modelos_final/           bundle de +24h, predições do holdout e resumo das métricas
requirements.txt
```

O bundle de **+12h não está versionado**: são 85 MB (Random Forest com 400
árvores), acima do limite prático do GitHub. Regenere-o com o comando de treino
abaixo. O de +24h (790 KB) está incluído para que `--prever` funcione de imediato.

## Instalação

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

## Uso

```bash
# 1. conferir a coleta e a ETo na sua região
python coleta_open_meteo.py --lat -23.5505 --lon -46.6333 --dias 30

# 2. treinar (3 anos de ERA5, horizontes de 12h e 24h)
python treinar_modelo_solo.py --lat -23.5505 --lon -46.6333 --nome minha_fazenda --anos 3

# 3. validar a pipeline (sem rede)
python teste_vazamento.py

# 4. prever com um bundle salvo
python treinar_modelo_solo.py --prever modelos/solo_minha_fazenda_h24.joblib
```

Coordenadas também via `METEO_LAT` / `METEO_LON`.

---

## Três coisas que você precisa saber antes de usar isto para decidir irrigação

### 1. O alvo não é um sensor de campo

A umidade do solo vem do modelo de superfície do ERA5 (fonte `archive`) ou do
ICON (fonte `forecast`), não de medição in situo. O modelo treinado aqui aprende
a **emular o esquema de superfície da reanálise** — útil para capturar a
dinâmica (secagem por ETo, recarga por chuva) e para prototipar, mas não é o
mesmo que prever o que a sua sonda vai ler.

Para valer em campo, troque o alvo por leituras da sua sonda. A pipeline foi
escrita para isso: substitua `agua_zona_raiz_mm` em `preparar_serie()` pela
série do sensor, mantendo as features meteorológicas como forçantes. O resto
(features, CV temporal, baselines, export) não muda.

### 2. O esquema de camadas difere por endpoint, e o bundle valida isso

| fonte | camadas | histórico | uso |
|---|---|---|---|
| `archive` (ERA5) | 0-7, 7-28, 28-100, 100-255 cm | multi-anual | treino |
| `forecast` (ICON) | 0-1, 1-3, 3-9, 9-27, 27-81 cm | ~92 dias | operação |

Um modelo treinado num esquema **não** pode ser aplicado ao outro. O bundle
grava `fonte_dados` e `esquema_camadas`, e `prever_com_bundle()` coleta pela
mesma fonte do treino. Para um modelo verdadeiramente operacional use
`--fonte forecast` (mesmo esquema, mas só ~90 dias de treino); para aprender a
dinâmica com vários ciclos sazonais use `--fonte archive`.

### 3. Compare sempre com a persistência, não com o R²

Umidade do solo é fortemente autocorrelacionada: em 12h ela muda pouco. Um
modelo pode ter R² = 0,88 e ainda assim **empatar** com chutar "igual a agora".
Por isso o script reporta *skill score*:

```
skill = 1 − MSE_modelo / MSE_baseline     (≤ 0 → o modelo não serve)
```

Essa métrica guiou duas correções reais neste projeto.

**Correção 1 — formulação do alvo.** O primeiro treino deu R² = 0,735 com skill
de **−0,51**: pior que a persistência. Prever o nível absoluto obriga a árvore a
reproduzir "saída ≈ entrada" de forma escalonada, e Random Forest não extrapola
para faixas de umidade ausentes do treino. Com `--formulacao delta` (prever a
*variação* e reconstruir o nível) o RMSE caiu de 3,11 → 2,69 mm, a variância da
CV de ±1,85 → ±0,62, e as features importantes passaram a ser as fisicamente
corretas: chuva acumulada, VPD, balanço hídrico.

**Correção 2 — calibração da amplitude.** Com 3 anos de treino, o skill médio a
+12h ficou em **+0,0001**: empate exato com a persistência, ao lado de um
R² de 0,882. O diagnóstico mostrou que o modelo acerta a *direção* (ρ = 0,26 a
+12h, 0,33 a +24h) mas aposta amplitude excessiva para o sinal que tem. A teoria
é exata: para uma previsão com correlação ρ, o MSE de `a·d` é mínimo em
`a = ρ·σ_obs/σ_prev`, e o skill resultante é **ρ²**. Amplitude excessiva custa
MSE mesmo quando o sinal é real — era isso que anulava o ganho.

`ajustar_calibracao()` estima esse escalar com predições *out-of-fold do treino*
(nunca no holdout). Mas **só com as dobras finais**: as primeiras treinam com
uma fração pequena da série e em regimes antigos, e subestimam o ρ do modelo
final. Usando as 5 dobras o fator saiu 4,5× mais conservador que o ótimo e o
ganho foi de apenas +0,025; com as 2 finais (`--dobras-calibracao`, default 2):

| horizonte | ρ out-of-fold | escala | skill médio | teto ρ² | % do teto |
|---|---|---|---|---|---|
| +12h | 0,224 | 0,448 | **+0,060** | 0,065 | 92% |
| +24h | 0,225 | 0,344 | **+0,089** | 0,106 | 84% |

### 4. Erro médio e detecção de evento são objetivos em conflito

O skill médio é dominado pelas horas em que quase nada muda, e nelas a
persistência é imbatível por definição. Mas amplitude cheia, que detecta
eventos, destrói o desempenho nas horas calmas. A curva de trade-off (impressa
no relatório como **diagnóstico**, nunca para escolher a escala — isso seria
ajustar no teste) mostra o conflito a +24h:

```
 escala    RMSE     MAE  skill_med  skill_decil  skill_calmas
  0.000   3.162   2.019     0.0000       0.0000        0.0000   ← persistência pura
  0.344*  3.019   1.939     0.0887       0.0901       -0.5949   ← calibrado
  0.500   2.993   1.951     0.1043       0.1244       -1.4156
  1.000   3.081   2.112     0.0510       0.2066       -6.3624   ← saída crua
```

A saída crua tem o melhor skill de evento (+0,21) e é **6× pior que não fazer
nada** nas horas calmas, que são metade delas. A calibração equilibra: skill
positivo nas duas pontas, com dano modesto nas horas calmas.

Use a tabela para decidir o *objetivo*. Se a sua operação só age em evento,
`--sem-calibracao` maximiza a detecção; se você reporta RMSE/MAE, o default
calibrado é o certo.

Se o relatório imprimir que o modelo não agrega nem nos eventos, não coloque em
produção.

---

## Rigor temporal

Série temporal viola i.i.d.; a pipeline trata isso em cinco pontos:

- **Janelas retrospectivas.** Nenhum `rolling` olha adiante; o único `shift`
  negativo do código está em `construir_alvo()`.
- **Grade horária contígua.** `_validar_grade_horaria()` reindexa a série antes
  de qualquer lag. Sem isso, com 30 horas faltantes, `shift(24)` andaria 24
  *linhas* e pegaria t−54h achando que pegou t−24h.
- **`TimeSeriesSplit(gap=h)`.** A folga impede que o fim da janela de treino
  alcance o alvo da validação.
- **Holdout cronológico**, nunca amostra aleatória.
- **Métricas sempre na escala de nível** (mm de água), inclusive na seleção por
  CV, para que `delta` e `nivel` sejam comparáveis entre si e com as baselines.

`teste_vazamento.py` verifica isso empiricamente, com 39 asserções em série
sintética, sem rede. O teste decisivo multiplica por 7,5 **todas** as colunas
depois de um corte e exige que nenhuma das 63 features anteriores ao corte mude.
O bloco de calibração confirma numericamente que o skill obtido é ρ², que uma
previsão sem sinal recebe escala ≈ 0, e que casos degenerados (variância zero,
amostra pequena) devolvem escala 1,0 em vez de dividir por zero.

## Saídas

Um `.joblib` por horizonte, contendo modelo, contrato de features (nomes e
ordem), formulação do alvo, **fator de calibração**, esquema de camadas, limites
hídricos (capacidade de campo e ponto de murcha estimados por percentil),
métricas e versões de biblioteca. Salvar só o estimador é o erro clássico: na
inferência não haveria como saber a ordem das features, se a saída precisa ser
somada ao estado atual nem qual escala aplicar — e o erro seria de ~90 mm,
silencioso.

Mais `resumo_<nome>.json` e, com `--salvar-predicoes`, o CSV de
observado/previsto/baselines do holdout.

Nota de tamanho: o Random Forest exportado fica em ~87 MB com 3 anos de dados
(400 árvores, `min_samples_leaf=2`), contra ~800 KB do HistGradientBoosting. Se
o deploy tem restrição de tamanho ou latência, vale forçar
`--modelos hist_gradient_boosting` — no horizonte de 24h ele já foi o escolhido
pela CV de todo modo.
