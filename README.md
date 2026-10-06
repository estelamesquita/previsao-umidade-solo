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
controle_irrigacao.py    controlador PID e Fuzzy: setpoint + sensor -> pulso do atuador
servidor.py              backend FastAPI entre o ESP32 e o controle
dashboard.py             painel Streamlit: gráficos, status da bomba, auto/manual
firmware_esp32/          firmware do nó de campo (Arduino IDE)
no_virtual.py            ESP32 de mentira: roda o sistema inteiro sem hardware
teste_vazamento.py       testes de causalidade das features (roda offline)
teste_controle.py        testes do controlador com simulador de solo (roda offline)
teste_servidor.py        testes do backend (roda offline)
apresentacao.html        deck de 14 slides (abra no navegador; F = tela cheia, N = notas)
modelos_final/           bundle de +24h, predições do holdout e resumo das métricas
requirements.txt
```

O fluxo completo: a coleta calcula a **ETo**, o modelo prevê a **umidade futura**
e dela sai o **setpoint**, e o controlador converte setpoint + leitura do sensor
na **duração do pulso** da válvula.

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

## Controle do atuador

`controle_irrigacao.py` fecha a malha: recebe a umidade do sensor e o setpoint
dinâmico, devolve a duração do pulso em segundos e o duty cycle PWM.

```python
from controle_irrigacao import Atuador, ControladorPID, LimitesSeguranca

atuador = Atuador(vazao_lph=400.0, area_m2=50.0, eficiencia=0.85)
ctrl = ControladorPID(atuador, LimitesSeguranca(
    capacidade_campo_mm=122.6, ponto_murcha_mm=71.8))

d = ctrl.atualizar(umidade_mm=88.0, setpoint_mm=96.0)
if d.irrigar:
    acionar_valvula(segundos=d.pulso_s)     # ou PWM: d.duty
```

Tudo em **milímetros de lâmina**, a mesma unidade da ETo e da chuva (1 mm =
1 L/m²). A conversão para tempo fica isolada em `Atuador`, que guarda vazão,
área e eficiência — trocar o emissor não mexe nos ganhos.

### Três razões para um PID de livro-texto falhar aqui

**O solo já é um integrador.** A água se acumula; a planta não é um processo
auto-regulado. Ação integral forte sobre um processo integrador oscila, então
`ki` entra pequeno e só para corrigir erro sistemático (emissor entupido,
vazão real abaixo da nominal).

**O tempo morto é grande.** Infiltração e redistribuição levam de minutos a
horas até o sensor registrar a água. Um controlador que ignora isso segue
acionando enquanto espera. Tratado pela trava pós-irrigação e pelo termo de
**água em trânsito**, que desconta do erro o que já foi aplicado mas ainda não
foi lido — um preditor de Smith simplificado. O efeito, medido em simulação:

| | pico de umidade | água aplicada |
|---|---|---|
| sem o termo de trânsito | 121,1 mm (quase na cc) | 40,0 mm |
| **com o termo** | **95,1 mm** | **18,2 mm** |

Mesma meta, 2,2× menos água e sem encharcamento.

**O atuador é unidirecional.** Dá para adicionar água, não para remover. A
saturação em zero é permanente enquanto o solo estiver acima da meta, o que
torna o anti-windup obrigatório. A derivada é calculada sobre a **medida**,
nunca sobre o erro — o setpoint vem do modelo e muda sozinho, e derivada do
erro transformaria cada atualização num pico.

### Salvaguardas

Banda morta, trava pós-irrigação, cota diária, pulso mínimo e máximo do
atuador, e teto na capacidade de campo (água acima dela percola e leva
nutriente junto — irrigar além disso é perda irreversível). Toda recusa vem
com o motivo em `Decisao.motivo`, porque controlador sem observabilidade não
se depura em campo.

### Fuzzy como alternativa

`ControladorFuzzy` implementa o mesmo contrato com 15 regras Mamdani sobre
déficit relativo e tendência da umidade. A vantagem é expressar o conhecimento
agronômico direto ("déficit alto e solo secando rápido → pulso longo") sem
sintonizar ganhos; a desvantagem é não garantir erro nulo em regime. Nos testes
os dois convergem (erro final 0,93 mm no PID, 0,40 mm no fuzzy).

### Testes

```bash
python teste_controle.py
```

71 asserções, sem rede. Inclui um simulador de balanço hídrico com as duas
propriedades que tornam o processo difícil: tempo morto na infiltração e
drenagem irreversível acima da capacidade de campo. Verifica convergência,
ausência de overshoot, anti-windup sob saturação, derivada sem chute de
setpoint, e o comportamento diante de chuva.

## Backend para o ESP32

```bash
pip install fastapi "uvicorn[standard]"
uvicorn servidor:app --host 0.0.0.0 --port 8000
```

Dois endpoints de operação, mais `/health` e `/api/v1/status/{id}` para
diagnóstico. Documentação interativa em `/docs`.

```http
POST /api/v1/leitura
{"device_id": "esp32-horta-01", "umidade_m3m3": 0.29, "bateria_v": 3.7}

GET /api/v1/comando/esp32-horta-01
{"acionar": true, "segundos": 1800.0, "duty": 1.0,
 "comando_id": "77aca9a813b6", "motivo": "irrigar",
 "setpoint_mm": 104.82, "erro_mm": 23.62, "proximo_poll_s": 60}
```

### Dois laços de velocidades diferentes

O modelo precisa de 168 h de histórico e de uma chamada à Open-Meteo: leva
segundos e tem horizonte de 12 a 24 h. O ESP32 consulta a cada minuto. Rodar
o modelo por requisição seria lento e inútil — a previsão não muda em 60 s.

```
laço lento   (~1 h)      modelo → SETPOINT, em cache com TTL
laço rápido  (por poll)  setpoint + leitura → PID → PULSO
```

O modelo prevê a água em t+24 h; se antecipa **queda**, o setpoint sobe agora
para criar folga antes da demanda chegar. O ajuste é multiplicado por
`IRRIGACAO_CONFIANCA` (padrão 0,5), porque com o skill medido (0,060 e 0,089)
apostar o acionamento inteiro na previsão não se justifica.

**Se o modelo falhar, o controle continua.** O setpoint cai para uma fração
fixa da água disponível e a resposta vem com `degradado: true`. Um sistema de
irrigação que para porque a IA parou é pior que um temporizador burro.

### Decisões de protocolo pensadas para o ESP32

| decisão | por quê |
|---|---|
| relógio do **servidor** | ESP32 sem RTC perde a hora a cada reset, e carimbo errado corrompe o lockout e a cota diária |
| GET **idempotente** por 120 s | o retry do dispositivo devolve o mesmo `comando_id`, não uma segunda dose |
| `proximo_poll_s` na resposta | o dispositivo dorme o tempo certo em vez de consultar em laço |
| `comando_executado` no POST | fecha o ciclo e torna visível a bomba que não ligou |
| estado persistido em disco | sem isso, um laço de reinicialização zeraria a cota e encharcaria a lavoura |

Em qualquer dúvida — sem leitura, leitura vencida, erro interno — a resposta é
**não acionar**. Irrigar às cegas é o único erro verdadeiramente caro deste
sistema, e por isso até o handler de exceção genérico responde `acionar: false`.

### Segurança

O endpoint comanda uma bomba. Defina `IRRIGACAO_API_KEY` e o servidor passa a
exigir o header `X-API-Key`; sem ela, sobe aberto e registra um aviso no log.
Toda a configuração é por variável de ambiente (`IRRIGACAO_VAZAO_LPH`,
`IRRIGACAO_CC_MM`, `IRRIGACAO_BUNDLE`, …).

### Testes

```bash
python teste_servidor.py
```

52 asserções, sem rede e sem modelo treinado. O foco não é "o endpoint
responde 200": é que o servidor **nunca ligue a bomba por engano**. Cobre
sensor mudo, leitura vencida, modelo fora do ar, retry do dispositivo, corpo
malformado, autenticação e sobrevivência da cota diária a um reinício.

## Painel

```bash
pip install streamlit
streamlit run dashboard.py
```

Exige o backend no ar. Configure endereço, chave e dispositivo na barra lateral
(ou em `IRRIGACAO_API_URL` / `IRRIGACAO_DEVICE`).

Mostra umidade do solo e setpoint sobre a faixa de água útil, os pulsos
aplicados, a previsão do tempo, o estado da bomba e o seletor automático/manual.

### O botão de modo vive no servidor, não na sessão

Quem consulta o servidor é o **ESP32**, não o painel. Um modo guardado em
`st.session_state` deixaria a bomba irrigando sozinha com a tela escrito
"Manual" — um enfeite, não um controle. Por isso o modo é estado do servidor
(`POST /api/v1/modo/{id}`), sobrevive a reinício e cancela qualquer comando
pendente ao assumir o controle: quem passou para manual não quer uma decisão
antiga chegando depois.

O acionamento manual (`POST /api/v1/acionar/{id}`) dispensa a **decisão** do
controlador, nunca os **limites**. Um pedido de 1800 s com cota de 1 mm sai
cortado para 318 s; com o solo no teto de capacidade, é recusado. "Manual" não
é atalho para encharcar a lavoura.

### Decisões dos gráficos

**Nunca eixo duplo.** Chuva e ETo dividem um eixo porque ambas estão em mm — e
é essa a leitura: uma entra no solo, a outra sai. Temperatura, que está em °C,
vai para um gráfico próprio. Juntar escalas diferentes num eixo inventa
correlação que o dado não tem.

**Cor nunca sozinha.** Todo selo de estado leva ícone e texto, para funcionar
em daltonismo, impressão e alto contraste. As duas séries usam os slots 1 e 2
da paleta de referência sem alteração, cuja documentação certifica que os três
primeiros passam em todos os pares (CVD ΔE 9,2 claro / 9,4 escuro). *Não rodei
o validador — `node` não está disponível nesta máquina.*

**Toda leitura é alcançável sem o mouse.** Cada gráfico tem sua tabela
equivalente num expander; o tooltip enriquece, não é a única via.

## Rodar sem hardware nenhum

Três níveis, do mais isolado ao mais completo.

**1. Só o controlador, sem rede** — já incluso nos testes:

```bash
python teste_controle.py
```

Simulador de balanço hídrico com tempo morto de infiltração e drenagem
irreversível. É o lugar certo para avaliar qualidade de controle, porque
simula em tempo exato.

**2. O sistema inteiro, com um ESP32 de mentira:**

```bash
# servidor com as constantes de tempo encolhidas para a demonstração
IRRIGACAO_LOCKOUT_S=15 IRRIGACAO_TAU_TRANSITO_S=30 \
IRRIGACAO_PULSO_MAX_S=900 uvicorn servidor:app --port 8000

python no_virtual.py --aceleracao 200 --eto 7 --chuva 0.02
streamlit run dashboard.py
```

`no_virtual.py` fala o mesmo protocolo do firmware — POST da leitura, GET do
comando, confirmação do pulso no ciclo seguinte. **O servidor não sabe que é
simulação**: roda o mesmo controlador PID e devolve os mesmos comandos. O que
está sendo exercitado é o sistema de verdade; só o sensor, a bomba e o solo
são de mentira.

Com `--aceleracao 200` você vê um dia inteiro em alguns minutos, com o painel
atualizando ao vivo.

> **Ressalva:** o controlador no servidor mede `dt` pelo relógio real, enquanto
> o solo simulado anda acelerado. O termo derivativo, que é uma taxa por
> segundo, vê o solo mudando mais rápido que o próprio relógio. A aceleração
> serve para **ver** o laço funcionando, não para avaliar o controle — para isso
> use o nível 1, ou `--aceleracao 1`.

**3. O firmware C++ de verdade, sem placa:** o [Wokwi](https://wokwi.com) simula
um ESP32 com Wi-Fi no navegador. Cole o `.ino`, troque o sensor por um
potenciômetro (o capacitivo não existe lá) e aponte `API_BASE` para um servidor
acessível pela internet.

## Nó ESP32

`firmware_esp32/` fecha o circuito com o hardware: lê o sensor capacitivo v1.2,
envia a leitura e aciona o relé pelo tempo que a API devolver. Detalhes de
ligação e calibração no [README do firmware](firmware_esp32/README.md).

Três armadilhas que o firmware trata explicitamente, porque derrubam esse
projeto em campo:

**ADC2 não funciona com o Wi-Fi ligado.** No ESP32 o ADC2 é compartilhado com
o rádio — `analogRead()` em GPIO 0, 2, 4, 12–15 ou 25–27 devolve lixo assim que
a rede sobe. O sintoma engana: funciona perfeitamente até você conectar. O
sensor tem de ficar no ADC1 (32–36, 39), e o firmware avisa no Serial se não
estiver.

**A maioria dos módulos de relé aciona em nível baixo**, e entre o reset e o
`setup()` o pino flutua. Por isso o nível seguro é escrito **antes** de
`pinMode()` — com `pinMode` primeiro há uma janela em nível indefinido,
suficiente para o relé clicar e a bomba ligar sozinha no boot.

**`delay()` com o relé ligado é perigoso.** Um `delay(1800000)` prende o laço
por 30 minutos; se algo travar nesse intervalo, nada desliga a bomba. O tempo
é contado com `millis()`, `atualizarRele()` roda na primeira linha de todo
`loop()`, e há três camadas independentes: teto local de 30 min que ignora
duração absurda vinda da rede, watchdog de 30 s, e contagem que segue correndo
durante reconexão de Wi-Fi.

O nó confirma o pulso anterior no POST seguinte com o tempo que de fato
executou — é assim que o servidor descobre uma bomba que não ligou. E leitura
implausível (ADC travado em 0 ou 4095) **não é enviada**: telemetria falsa
levaria o controlador a irrigar às cegas.

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
