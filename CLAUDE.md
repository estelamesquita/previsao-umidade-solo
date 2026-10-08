# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sistema de irrigação automatizada: coleta meteorológica → ETo (FAO-56) → modelo
preditivo → setpoint → controlador PID → backend → ESP32 → bomba.
Código, comentários e mensagens de commit em **português**.

## Comandos

O ambiente virtual fica em `.venv/` (Windows): prefixe com
`./.venv/Scripts/python.exe`.

```bash
pip install -r requirements.txt
```

Testes — todos offline, sem rede e sem modelo treinado:

```bash
python teste_vazamento.py      # 42 asserções: causalidade das features
python teste_controle.py       # 71 asserções: controlador + simulador de solo
python teste_servidor.py       # 88 asserções: backend (requer httpx)
```

Rodar **um único teste**: os arquivos não usam pytest; são funções chamadas por
`main()`, e `checar()` acumula em `FALHAS`.

```bash
python -c "import teste_controle as t; t.teste_antiwindup(); print('FALHAS:', t.FALHAS)"
```

Pipeline e serviços:

```bash
python coleta_open_meteo.py --lat -23.55 --lon -46.63 --dias 30
python treinar_modelo_solo.py --lat -23.55 --lon -46.63 --anos 3
python treinar_modelo_solo.py --prever modelos_final/solo_sao_paulo_h24.joblib
uvicorn servidor:app --port 8000
streamlit run dashboard.py
python no_virtual.py --aceleracao 200 --eto 7
arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all firmware_esp32
```

## Arquitetura

### A unidade é milímetro de lâmina, em todo lugar

`1 mm = 1 L/m²`. ETo, chuva, água no solo e saída do controlador estão todas em
mm, o que mantém o balanço hídrico fechado e legível. As conversões ficam
isoladas nas bordas:

- `vwc_para_mm()` / `mm_para_vwc()` — m³/m³ do sensor ↔ mm
- `Atuador.segundos_para()` — mm ↔ tempo de válvula (vazão, área, eficiência)

Consequência prática: o ganho `kp` do PID é **adimensional** (fração do déficit
reposta por acionamento), e trocar o emissor não mexe em ganho nenhum.

### Dois laços de velocidades diferentes no servidor

O modelo precisa de 168 h de histórico e de uma chamada à Open-Meteo — leva
segundos. O ESP32 consulta a cada minuto.

```
laço lento  (~1 h, TTL)    modelo → SETPOINT           ServicoSetpoint
laço rápido (por poll)     setpoint + leitura → PULSO  ControladorPID
```

`ServicoSetpoint` degrada sozinho: sem modelo, sem internet ou com erro, o
setpoint vira fração fixa da água disponível e a resposta sai com
`degradado: true`. **O controle nunca para por causa do modelo.**

### Invariantes que não podem ser quebrados

1. **Alvo em variação, não em nível.** `construir_alvo(..., "delta")` prevê
   `água(t+h) − água(t)`; `reconstruir_nivel()` reconstrói. Prever o nível
   absoluto faz a árvore gastar capacidade reproduzindo a entrada — R² de 0,775
   com skill **−0,286** contra a persistência.

2. **O bundle carrega o contrato inteiro**, não só o estimador: ordem das
   features, `formulacao`, `escala_delta`, `esquema_camadas`, limites hídricos.
   Carregar só o modelo e inferir o resto erra por ~90 mm, silenciosamente.

3. **Esquemas de camada incompatíveis.** `archive` (ERA5: 0-7/7-28 cm) e
   `forecast` (ICON: 0-1/1-3/3-9/9-27 cm) não se misturam. O bundle grava qual
   usou e a inferência coleta pela mesma fonte.

4. **Em qualquer dúvida, não acionar.** Sem leitura, leitura vencida, erro
   interno — todos respondem `acionar: false`, inclusive o handler de exceção
   genérico do FastAPI.

5. **Métricas na escala de nível, sempre**, inclusive na seleção por CV. Só
   assim `delta` e `nivel` são comparáveis entre si e contra as baselines.

6. **Skill score, não R².** Umidade do solo é fortemente autocorrelacionada, e
   R² alto não distingue aprender de copiar. Resultado atual: skill +0,060 a
   +12 h e +0,089 a +24 h, contra teto teórico ρ² de 0,065 e 0,106.

### Rigor temporal (treino)

Janelas móveis só retrospectivas; o único `shift` negativo do projeto está em
`construir_alvo()`. A grade horária é reindexada **antes** de qualquer lag —
sem isso, um buraco de 30 h faz `shift(24)` pegar t−54 h. `TimeSeriesSplit` usa
`gap` igual ao horizonte, e o holdout é cronológico.

`teste_vazamento.py` prova isso empiricamente: multiplica por 7,5 **todas** as
colunas depois de um corte no tempo e exige que nenhuma das 63 features
anteriores mude.

### Controle

`Controlador` (ABC) concentra toda a segurança — banda morta, lockout, água em
trânsito, cota diária, teto de capacidade de campo, pulso mínimo e máximo. Só o
cálculo da lâmina bruta muda entre `ControladorPID` e `ControladorFuzzy`.

Três adaptações ao processo, que um PID de livro-texto erraria:

- **O solo já é integrador**, então `ki` é pequeno e só corrige erro sistemático
- **Tempo morto** tratado por lockout mais termo de água em trânsito (preditor
  de Smith simplificado). Sem ele: pico de 121,1 mm e 40,0 mm aplicados; com
  ele: 95,1 mm e 18,2 mm — mesma meta, 2,2× menos água
- **Derivada sobre a MEDIDA**, nunca sobre o erro: o setpoint vem do modelo e
  muda sozinho

### Modo auto/manual vive no servidor

Quem consulta o backend é o ESP32. Modo guardado em `st.session_state` seria
enfeite — a bomba seguiria irrigando com a tela escrito "Manual". Entrar em
manual cancela o comando pendente, e o modo é persistido.

O acionamento manual dispensa a **decisão** do controlador, nunca os
**limites**: pedido de 1800 s com cota de 1 mm sai cortado para 318 s.

### Protocolo ESP32

```
POST /api/v1/leitura          {device_id, umidade_m3m3|umidade_mm, rssi_dbm,
                               comando_executado, segundos_executados}
GET  /api/v1/comando/<id>  →  {acionar, segundos, comando_id, proximo_poll_s}
```

- O relógio é do **servidor** (ESP32 sem RTC perde a hora a cada reset)
- O GET é **idempotente** por 120 s: retry devolve o mesmo `comando_id`
- O nó confirma o pulso anterior no POST seguinte com o tempo que **de fato**
  executou — é assim que uma bomba que não ligou vira `falhas_bomba`

## Simulação sem hardware

`no_virtual.py` fala o mesmo protocolo do firmware. O servidor não sabe que é
simulação: roda o mesmo PID e devolve os mesmos comandos.

```bash
IRRIGACAO_LOCKOUT_S=15 IRRIGACAO_TAU_TRANSITO_S=30 IRRIGACAO_PULSO_MAX_S=900 uvicorn servidor:app --port 8000
python no_virtual.py --aceleracao 200 --eto 7
```

**Ressalva:** o controlador mede `dt` pelo relógio real enquanto o solo simulado
anda acelerado, então o termo derivativo vê o solo mudar mais rápido que o
próprio relógio. Aceleração serve para **ver** o laço, não para avaliar
qualidade de controle — para isso use `teste_controle.py`, que simula em tempo
exato.

## Convenções deste repositório

- **Configuração do servidor é toda por variável de ambiente** `IRRIGACAO_*`,
  lida na importação do módulo. Por isso `teste_servidor.py` recarrega o módulo
  com `importlib.reload()` a cada cenário.
- `no_virtual.py` importa `SoloSimulado` de `teste_controle.py` de propósito:
  duplicar a física criaria duas versões para divergirem.
- Identificadores e comentários **sem acento** (compatibilidade de console no
  Windows); strings visíveis ao usuário **com** acento.
- Mensagens de commit registram a decisão e o número que a justifica, não só o
  que mudou. `git log` é documentação de projeto aqui — leia antes de alterar
  algo que pareça estranho.
- Modelos `.joblib` são artefato de build: só o de +24 h (790 KB) é versionado,
  para `--prever` funcionar logo após o clone. O de +12 h tem 85 MB e fica fora.

## Firmware (`firmware_esp32/`)

Três armadilhas específicas do ESP32, detalhadas em `firmware_esp32/README.md`:

- **ADC2 não funciona com Wi-Fi ligado.** O sensor tem de ficar no ADC1
  (GPIO 32-36, 39). O firmware verifica em tempo de execução e avisa no Serial.
- **Nível seguro do relé escrito ANTES de `pinMode()`** — senão há uma janela em
  nível indefinido no boot, suficiente para a bomba ligar sozinha.
- **Nunca `delay()` com o relé ligado.** Contagem por `millis()`,
  `atualizarRele()` na primeira linha de todo `loop()`, teto local de 30 min e
  watchdog de 30 s.

## Estado atual e lacunas conhecidas

- `cc` e `pm` padrão vêm do **ERA5 para São Paulo**, não de medição no solo
  real. Calibrar isso é pré-requisito para uso em campo.
- O modelo versionado é de esquema `archive`; operação real exige treinar com
  `--fonte forecast`.
- O histórico do servidor vive em RAM (`deque`), perdido a cada reinício.
- Não há alerta de falha nem proteção contra reservatório vazio.
