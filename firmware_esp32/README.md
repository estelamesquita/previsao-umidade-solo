# Nó ESP32

Firmware do nó de campo: lê o sensor capacitivo, envia a leitura para o
backend e aciona o relé pelo tempo que a API devolver.

## Ligação

```
Sensor capacitivo v1.2        ESP32
  VCC  ──────────────────▶   GPIO25   (alimentado por GPIO — ver abaixo)
  GND  ──────────────────▶   GND
  AOUT ──────────────────▶   GPIO34   ◀── obrigatoriamente um pino do ADC1

Módulo relé (1 canal)         ESP32
  VCC  ──────────────────▶   5V (VIN)
  GND  ──────────────────▶   GND      ◀── terra COMUM
  IN   ──────────────────▶   GPIO26
```

A bomba vai nos contatos **COM/NO** do relé, com **fonte própria**. O ESP32
não alimenta bomba nenhuma — nem a de 5 V.

## As três armadilhas deste projeto

### 1. ADC2 não funciona com o Wi-Fi ligado

No ESP32 o ADC2 é compartilhado com o rádio. Assim que o Wi-Fi sobe,
`analogRead()` em GPIO 0, 2, 4, 12–15 ou 25–27 passa a devolver lixo ou a
travar. É o erro mais comum nesse projeto, e o sintoma engana: funciona
perfeitamente até você conectar na rede.

O sensor **tem** de ficar no ADC1: **GPIO 32, 33, 34, 35, 36 ou 39**. Os pinos
34–39 são só-entrada e não têm pull-up interno, o que é ideal para analógico.

O firmware avisa no Serial se o pino configurado não for do ADC1.

### 2. A maioria dos módulos de relé aciona em nível BAIXO

Inverter isso faz a bomba ligar quando você acha que desligou. Pior: entre o
reset e o `setup()`, o pino flutua e o relé pode fechar sozinho.

Por isso `releSeguroNoBoot()` escreve o nível seguro **antes** de `pinMode()` —
com `pinMode` primeiro há uma janela de microssegundos em nível indefinido,
suficiente para o relé clicar. Se o seu módulo for de nível alto, mude
`RELE_ATIVO_BAIXO` para `false`.

### 3. `delay()` com o relé ligado é perigoso

Um `delay(1800000)` prende o laço por 30 minutos. Se o Wi-Fi cair ou algo
travar nesse intervalo, **nada desliga a bomba**.

Aqui o tempo é contado com `millis()`, `atualizarRele()` roda na primeira
linha de todo `loop()`, e três camadas independentes protegem a lavoura:

| camada | o que faz |
|---|---|
| `PULSO_MAX_LOCAL_S` | corta qualquer duração absurda vinda da rede (30 min) |
| watchdog de 30 s | reseta se o firmware travar — e o boot deixa o pino seguro |
| contagem por `millis()` | segue correndo durante reconexão de Wi-Fi |

## Alimentação do sensor por GPIO

`PINO_SENSOR_VCC` liga o sensor só no instante da leitura. Sensor capacitivo
energizado 24 h por dia sofre eletrólise e a trilha corrói em semanas. Ligar
300 ms a cada 5 minutos resolve, e ainda economiza bateria. Use `-1` para
alimentar direto do 3V3.

## Calibração

Abra o **Monitor Serial a 115200** e digite:

```
cal seco        sensor no ar, limpo e seco
cal molhado     sensor na água até a linha marcada
ler             leitura imediata (cru e convertida)
parar           desliga a bomba agora
```

Os valores ficam na memória não volátil (NVS) e sobrevivem a reset e
regravação.

**Ressalva honesta:** o capacitivo mede capacitância, não umidade. A conversão
para m³/m³ usa a faixa `VWC_SECO`–`VWC_MOLHADO` e é uma aproximação. Para valor
agronômico de verdade, calibre contra amostragem gravimétrica no seu solo — o
resto do pipeline já está preparado para receber a série boa.

## Configuração

No topo do `.ino`:

```cpp
WIFI_SSID, WIFI_SENHA     sua rede
API_BASE                  "http://192.168.1.100:8000"  (sem barra no fim)
API_KEY                   igual ao IRRIGACAO_API_KEY do servidor, ou ""
DEVICE_ID                 "esp32-horta-01"
```

O `DEVICE_ID` é o que aparece no painel — o servidor registra o dispositivo na
primeira leitura recebida.

## Protocolo

```
POST /api/v1/leitura          {device_id, umidade_m3m3, rssi_dbm,
                               comando_executado, segundos_executados}
GET  /api/v1/comando/<id>  →  {acionar, segundos, comando_id, proximo_poll_s}
```

Dois detalhes que fecham o ciclo:

**O nó confirma o pulso anterior** no POST seguinte, com o tempo que de fato
executou. É assim que o servidor descobre uma bomba que não ligou — sem isso
ele assume que a água foi aplicada e a conta de balanço hídrico fica errada.

**O nó obedece `proximo_poll_s`.** O servidor pede consultas rápidas durante
irrigação e lentas em repouso, então o dispositivo não precisa adivinhar.

Leitura implausível (ADC travado em 0 ou 4095) **não é enviada**: telemetria
falsa levaria o controlador a irrigar às cegas. Melhor o silêncio, que o
servidor trata como `leitura_desatualizada` e não aciona.

## Bibliotecas

- **ArduinoJson** v7 (Gerenciador de Bibliotecas)
- Placa: **ESP32 Dev Module** (pacote `esp32` by Espressif)

Compilado e verificado com `arduino-cli` contra o core **esp32 3.3.12** e
**ArduinoJson 7.4.3**, com `--warnings all` e **zero avisos**:

```
Sketch uses 1045288 bytes (79%) of program storage space
Global variables use 48960 bytes (14%) of dynamic memory
```

Os 79% são quase todos pilha Wi-Fi e HTTPClient. Cabe no esquema de partição
padrão, inclusive com OTA, mas sem muita folga — se for adicionar bibliotecas
grandes, troque para "Minimal SPIFFS" em *Ferramentas › Partition Scheme*.

## Primeiro teste, sem bomba

1. Grave com `API_BASE` apontando para o seu PC
2. Suba o backend: `uvicorn servidor:app --host 0.0.0.0 --port 8000`
3. Abra o Monitor Serial e confirme `[WIFI] ok` e `[API] POST ok`
4. Abra o painel: `streamlit run dashboard.py`
5. Deixe a bomba **desconectada** e ouça o clique do relé no primeiro comando

Só ligue a bomba depois que o clique estiver acontecendo na hora certa.
