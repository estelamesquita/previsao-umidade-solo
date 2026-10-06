/*
 * firmware_esp32.ino — no de irrigacao
 *
 * Le um sensor capacitivo de umidade v1.2, envia a leitura para a API
 * FastAPI deste projeto e aciona o rele pelo tempo que a API devolver.
 *
 * Fluxo:
 *   POST /api/v1/leitura        envia a umidade (e confirma o pulso anterior)
 *   GET  /api/v1/comando/<id>   recebe {acionar, segundos, comando_id}
 *   -> liga o rele por "segundos", de forma NAO bloqueante
 *
 * ---------------------------------------------------------------------
 * LIGACAO
 *
 *   Sensor capacitivo v1.2        ESP32
 *     VCC  ------------------->   3V3   (5 V da mais excursao; veja nota)
 *     GND  ------------------->   GND
 *     AOUT ------------------->   GPIO34   <-- TEM de ser um pino do ADC1
 *
 *   Modulo rele (1 canal)         ESP32
 *     VCC  ------------------->   5V (VIN)  — o rele precisa de 5 V
 *     GND  ------------------->   GND       — terra COMUM com o ESP32
 *     IN   ------------------->   GPIO26
 *
 *   A bomba vai nos contatos do rele (COM/NO), com fonte PROPRIA. Nunca
 *   alimente a bomba pelo ESP32.
 *
 * ---------------------------------------------------------------------
 * TRES ARMADILHAS QUE DERRUBAM ESTE PROJETO
 *
 * 1. ADC2 NAO FUNCIONA COM O WI-FI LIGADO. No ESP32, o ADC2 (GPIO0, 2, 4,
 *    12-15, 25-27) e usado pelo radio: assim que o Wi-Fi sobe, analogRead
 *    nesses pinos passa a devolver lixo ou travar. O sensor TEM de ficar no
 *    ADC1: GPIO32, 33, 34, 35, 36 ou 39. GPIO34-39 sao so-entrada e nao tem
 *    pull-up interno, o que e perfeito para sensor analogico.
 *
 * 2. A MAIORIA DOS MODULOS DE RELE E ACIONADA EM NIVEL BAIXO. Trocar isso
 *    de lugar faz a bomba ligar quando voce acha que desligou. Pior: entre
 *    o reset e o setup(), o pino flutua e o rele pode fechar sozinho. Por
 *    isso o nivel seguro e escrito ANTES de pinMode().
 *
 * 3. NUNCA USE delay() COM O RELE LIGADO. Um delay(1800000) prende o laco
 *    por 30 min: se o Wi-Fi cair ou algo travar, nada desliga a bomba. Aqui
 *    o tempo e contado com millis(), o laco segue girando e um teto LOCAL
 *    desliga a bomba mesmo que o servidor peca um absurdo.
 *
 * ---------------------------------------------------------------------
 * BIBLIOTECAS (Arduino IDE -> Gerenciador de Bibliotecas)
 *   ArduinoJson (v7)  by Benoit Blanchon
 * Placa: "ESP32 Dev Module" (pacote esp32 by Espressif)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <esp_task_wdt.h>

// =====================================================================
// CONFIGURACAO — edite esta secao
// =====================================================================

static const char* WIFI_SSID  = "SUA_REDE_WIFI";
static const char* WIFI_SENHA = "SUA_SENHA_WIFI";

// Endereco do backend (servidor.py). SEM barra no fim.
static const char* API_BASE   = "http://192.168.1.100:8000";
// Deixe "" se o servidor subiu sem IRRIGACAO_API_KEY.
static const char* API_KEY    = "";
static const char* DEVICE_ID  = "esp32-horta-01";

// --- Pinos ---
// ATENCAO: o sensor TEM de ficar no ADC1 (32,33,34,35,36,39). Ver nota 1.
static const int PINO_SENSOR = 34;
static const int PINO_RELE   = 26;
// Alimenta o sensor por um GPIO para liga-lo so na leitura: evita a
// eletrolise que corroi a trilha quando ele fica energizado 24 h por dia.
// Use -1 para alimentar direto do 3V3.
static const int PINO_SENSOR_VCC = 25;

// --- Rele ---
// true  = modulo aciona em nivel BAIXO (o mais comum nos modulos chineses)
// false = modulo aciona em nivel ALTO
static const bool RELE_ATIVO_BAIXO = true;

// --- Teto de seguranca LOCAL ---
// O firmware nunca mantem a bomba ligada alem disto, aconteca o que
// acontecer com o servidor. E a ultima linha de defesa do sistema.
static const uint32_t PULSO_MAX_LOCAL_S = 1800;   // 30 min
static const uint32_t PULSO_MIN_LOCAL_S = 5;

// --- Calibracao do sensor (valores crus do ADC, 0..4095) ---
// SECO    = sensor no ar, limpo e seco
// MOLHADO = sensor dentro de um copo d'agua ate a linha marcada
// Sao apenas o ponto de partida: calibre com o comando "cal" no Monitor
// Serial e o firmware grava os seus valores na memoria nao volatil.
static const int ADC_SECO_PADRAO    = 3050;
static const int ADC_MOLHADO_PADRAO = 1380;

// Faixa de umidade volumetrica correspondente (m3/m3).
// A conversao de "cru" para VWC de verdade exige calibracao gravimetrica
// no SEU solo; esta faixa e uma aproximacao razoavel para solo agricola.
static const float VWC_SECO    = 0.05f;
static const float VWC_MOLHADO = 0.45f;

// --- Tempos ---
static const uint32_t INTERVALO_PADRAO_S = 300;   // sobrescrito pela API
static const uint32_t WDT_TIMEOUT_S      = 30;
static const uint16_t HTTP_TIMEOUT_MS    = 8000;
static const uint8_t  AMOSTRAS_ADC       = 15;    // impar: mediana exata

// =====================================================================
// ESTADO
// =====================================================================

Preferences prefs;

int adcSeco    = ADC_SECO_PADRAO;
int adcMolhado = ADC_MOLHADO_PADRAO;

// Rele / pulso em andamento
bool     releLigado        = false;
uint32_t pulsoInicioMs     = 0;
uint32_t pulsoDuracaoMs    = 0;
String   pulsoComandoId    = "";

// Confirmacao a enviar no proximo POST
String   ultimoComandoId   = "";
float    ultimoSegundosExec = -1.0f;

// Agendamento
uint32_t proximoCicloMs    = 0;
uint32_t intervaloS        = INTERVALO_PADRAO_S;

// =====================================================================
// RELE — toda a seguranca da bomba mora aqui
// =====================================================================

inline void releEscrever(bool ligar) {
  // XOR com a polaridade: um unico ponto decide o nivel eletrico, entao
  // trocar RELE_ATIVO_BAIXO nao exige cacar digitalWrite pelo codigo.
  digitalWrite(PINO_RELE, (ligar != RELE_ATIVO_BAIXO) ? HIGH : LOW);
}

void releSeguroNoBoot() {
  // A ORDEM IMPORTA: escrever o nivel seguro ANTES de virar saida faz o
  // pino ja nascer desligado. Com pinMode primeiro, ha uma janela de
  // microssegundos em nivel indefinido — suficiente para o rele clicar.
  releEscrever(false);
  pinMode(PINO_RELE, OUTPUT);
  releEscrever(false);
  releLigado = false;
}

void pararBomba(const char* motivo) {
  if (!releLigado) return;
  releEscrever(false);
  releLigado = false;

  uint32_t decorridoMs = millis() - pulsoInicioMs;
  ultimoSegundosExec = decorridoMs / 1000.0f;
  ultimoComandoId    = pulsoComandoId;
  pulsoComandoId     = "";

  Serial.printf("[RELE] desligado apos %.1f s (%s)\n",
                ultimoSegundosExec, motivo);
}

void iniciarBomba(uint32_t segundos, const String& comandoId) {
  if (releLigado) {
    Serial.println("[RELE] ja ha um pulso em andamento; ignorando");
    return;
  }
  // Teto local: nunca confia cegamente na duracao que veio da rede.
  if (segundos > PULSO_MAX_LOCAL_S) {
    Serial.printf("[RELE] servidor pediu %lu s; cortando para o teto local "
                  "de %lu s\n", (unsigned long)segundos,
                  (unsigned long)PULSO_MAX_LOCAL_S);
    segundos = PULSO_MAX_LOCAL_S;
  }
  if (segundos < PULSO_MIN_LOCAL_S) {
    Serial.printf("[RELE] pulso de %lu s abaixo do minimo; ignorado\n",
                  (unsigned long)segundos);
    return;
  }

  pulsoComandoId  = comandoId;
  pulsoDuracaoMs  = segundos * 1000UL;
  pulsoInicioMs   = millis();
  releLigado      = true;
  releEscrever(true);
  Serial.printf("[RELE] LIGADO por %lu s (comando %s)\n",
                (unsigned long)segundos, comandoId.c_str());
}

void atualizarRele() {
  if (!releLigado) return;
  // Subtracao de unsigned: correta mesmo no estouro de millis() (~49 dias).
  if ((millis() - pulsoInicioMs) >= pulsoDuracaoMs) {
    pararBomba("tempo cumprido");
  }
}

// =====================================================================
// SENSOR
// =====================================================================

int compararInt(const void* a, const void* b) {
  return (*(const int*)a) - (*(const int*)b);
}

int lerAdcBruto() {
  if (PINO_SENSOR_VCC >= 0) {
    digitalWrite(PINO_SENSOR_VCC, HIGH);
    delay(300);            // o capacitivo precisa estabilizar
  }

  int amostras[AMOSTRAS_ADC];
  for (uint8_t i = 0; i < AMOSTRAS_ADC; i++) {
    amostras[i] = analogRead(PINO_SENSOR);
    delay(12);
  }

  if (PINO_SENSOR_VCC >= 0) digitalWrite(PINO_SENSOR_VCC, LOW);

  // MEDIANA, nao media: o ADC do ESP32 solta picos isolados, e um unico
  // valor espurio arrasta a media. A mediana simplesmente o descarta.
  qsort(amostras, AMOSTRAS_ADC, sizeof(int), compararInt);
  return amostras[AMOSTRAS_ADC / 2];
}

float adcParaVwc(int cru) {
  // O capacitivo e INVERSO: quanto mais agua, MENOR a leitura.
  if (adcSeco == adcMolhado) return VWC_SECO;   // calibracao degenerada

  float f = (float)(adcSeco - cru) / (float)(adcSeco - adcMolhado);
  if (f < 0.0f) f = 0.0f;
  if (f > 1.0f) f = 1.0f;
  return VWC_SECO + f * (VWC_MOLHADO - VWC_SECO);
}

bool leituraPlausivel(int cru) {
  // 0 ou 4095 travados quase sempre significam fio solto ou pino errado
  // (tipicamente um pino do ADC2 com o Wi-Fi ligado — ver nota 1).
  if (cru <= 5 || cru >= 4090) {
    Serial.printf("[SENSOR] leitura suspeita (%d): confira a fiacao e se o "
                  "pino e mesmo do ADC1\n", cru);
    return false;
  }
  return true;
}

// =====================================================================
// CALIBRACAO (Monitor Serial)
// =====================================================================

void carregarCalibracao() {
  prefs.begin("irrig", true);
  adcSeco    = prefs.getInt("seco", ADC_SECO_PADRAO);
  adcMolhado = prefs.getInt("molhado", ADC_MOLHADO_PADRAO);
  prefs.end();
  Serial.printf("[CAL] seco=%d molhado=%d\n", adcSeco, adcMolhado);
}

void salvarCalibracao() {
  prefs.begin("irrig", false);
  prefs.putInt("seco", adcSeco);
  prefs.putInt("molhado", adcMolhado);
  prefs.end();
  Serial.printf("[CAL] gravado: seco=%d molhado=%d\n", adcSeco, adcMolhado);
}

void tratarSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd == "cal seco") {
    adcSeco = lerAdcBruto();
    salvarCalibracao();
  } else if (cmd == "cal molhado") {
    adcMolhado = lerAdcBruto();
    salvarCalibracao();
  } else if (cmd == "ler") {
    int cru = lerAdcBruto();
    Serial.printf("[SENSOR] cru=%d  vwc=%.3f m3/m3\n", cru, adcParaVwc(cru));
  } else if (cmd == "parar") {
    pararBomba("comando serial");
  } else if (cmd == "ajuda" || cmd == "?") {
    Serial.println(F(
      "comandos:\n"
      "  cal seco      sensor no ar  -> grava o ponto seco\n"
      "  cal molhado   sensor n'agua -> grava o ponto molhado\n"
      "  ler           leitura imediata\n"
      "  parar         desliga a bomba agora"));
  }
}

// =====================================================================
// WI-FI
// =====================================================================

bool conectarWifi(uint32_t limiteMs = 20000) {
  if (WiFi.status() == WL_CONNECTED) return true;

  Serial.printf("[WIFI] conectando a %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);          // latencia estavel; o no nao e a bateria
  WiFi.begin(WIFI_SSID, WIFI_SENHA);

  uint32_t inicio = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - inicio) < limiteMs) {
    // O laco segue chamando atualizarRele(): se a bomba estiver ligada,
    // ela NAO fica refem da reconexao do Wi-Fi.
    atualizarRele();
    esp_task_wdt_reset();
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WIFI] ok, IP %s, RSSI %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
    return true;
  }
  Serial.println("[WIFI] falhou");
  return false;
}

// =====================================================================
// API
// =====================================================================

void aplicarCabecalhos(HTTPClient& http) {
  http.addHeader("Content-Type", "application/json");
  if (strlen(API_KEY) > 0) http.addHeader("X-API-Key", API_KEY);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.setConnectTimeout(HTTP_TIMEOUT_MS);
}

bool enviarLeitura(float vwc, int cru) {
  if (WiFi.status() != WL_CONNECTED) return false;

  JsonDocument doc;
  doc["device_id"]    = DEVICE_ID;
  doc["umidade_m3m3"] = serialized(String(vwc, 4));
  doc["rssi_dbm"]     = WiFi.RSSI();

  // Confirma o pulso anterior: e assim que o servidor descobre uma bomba
  // que nao ligou. Sem isso, ele acha que a agua foi aplicada.
  if (ultimoComandoId.length() > 0 && ultimoSegundosExec >= 0.0f) {
    doc["comando_executado"]   = ultimoComandoId;
    doc["segundos_executados"] = serialized(String(ultimoSegundosExec, 1));
  }

  String corpo;
  serializeJson(doc, corpo);

  HTTPClient http;
  String url = String(API_BASE) + "/api/v1/leitura";
  if (!http.begin(url)) {
    Serial.println("[API] URL invalida");
    return false;
  }
  aplicarCabecalhos(http);

  int codigo = http.POST(corpo);
  bool ok = (codigo == 200);

  if (ok) {
    JsonDocument resp;
    if (deserializeJson(resp, http.getString()) == DeserializationError::Ok) {
      uint32_t p = resp["proximo_poll_s"] | 0;
      if (p >= 5 && p <= 3600) intervaloS = p;
    }
    Serial.printf("[API] POST ok (cru=%d vwc=%.3f) proximo em %lu s\n",
                  cru, vwc, (unsigned long)intervaloS);
    // So limpa a confirmacao depois que o servidor a recebeu.
    ultimoComandoId = "";
    ultimoSegundosExec = -1.0f;
  } else {
    Serial.printf("[API] POST falhou: %d %s\n", codigo,
                  http.errorToString(codigo).c_str());
  }

  http.end();
  return ok;
}

bool buscarComando() {
  if (WiFi.status() != WL_CONNECTED) return false;

  HTTPClient http;
  String url = String(API_BASE) + "/api/v1/comando/" + DEVICE_ID;
  if (!http.begin(url)) return false;
  if (strlen(API_KEY) > 0) http.addHeader("X-API-Key", API_KEY);
  http.setTimeout(HTTP_TIMEOUT_MS);
  http.setConnectTimeout(HTTP_TIMEOUT_MS);

  int codigo = http.GET();
  if (codigo != 200) {
    Serial.printf("[API] GET falhou: %d\n", codigo);
    http.end();
    return false;
  }

  JsonDocument doc;
  DeserializationError erro = deserializeJson(doc, http.getString());
  http.end();
  if (erro) {
    Serial.printf("[API] JSON invalido: %s\n", erro.c_str());
    return false;
  }

  bool        acionar = doc["acionar"] | false;
  float       seg     = doc["segundos"] | 0.0f;
  const char* cid     = doc["comando_id"] | "-";
  const char* motivo  = doc["motivo"] | "?";

  uint32_t p = doc["proximo_poll_s"] | 0;
  if (p >= 5 && p <= 3600) intervaloS = p;

  if (!acionar) {
    Serial.printf("[API] nao irrigar (%s)\n", motivo);
    return true;
  }

  // O servidor e idempotente dentro de uma janela: se o mesmo comando
  // voltar, nao e uma segunda dose — ignore se ja esta sendo executado.
  if (releLigado && pulsoComandoId == String(cid)) {
    Serial.println("[API] comando repetido; ja em execucao");
    return true;
  }

  iniciarBomba((uint32_t)(seg + 0.5f), String(cid));
  return true;
}

// =====================================================================
// SETUP / LOOP
// =====================================================================

void configurarWdt() {
  // O watchdog e o que salva a lavoura se o firmware travar com a bomba
  // ligada: o reset coloca o pino em nivel seguro no boot.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  // Atribuicao campo a campo em vez de inicializador designado: este e
  // extensao GNU fora do C++20 e a ordem dos campos da struct ja mudou
  // entre versoes do ESP-IDF.
  esp_task_wdt_config_t cfg = {};
  cfg.timeout_ms     = WDT_TIMEOUT_S * 1000;
  cfg.idle_core_mask = 0;
  cfg.trigger_panic  = true;
  // O core 3.x pode ja ter inicializado o WDT; nesse caso devolve
  // ESP_ERR_INVALID_STATE e seguimos com o timeout dele. Ignorar o retorno
  // e proposital — ficar sem watchdog seria pior que um timeout diferente.
  esp_task_wdt_init(&cfg);
#else
  esp_task_wdt_init(WDT_TIMEOUT_S, true);
#endif
  esp_task_wdt_add(NULL);
}

void setup() {
  // PRIMEIRA COISA: deixar a bomba em estado seguro, antes de qualquer
  // outra inicializacao que possa demorar ou falhar.
  releSeguroNoBoot();

  Serial.begin(115200);
  delay(300);
  Serial.println(F("\n=== no de irrigacao ==="));
  Serial.printf("device: %s\n", DEVICE_ID);

  if (PINO_SENSOR_VCC >= 0) {
    pinMode(PINO_SENSOR_VCC, OUTPUT);
    digitalWrite(PINO_SENSOR_VCC, LOW);
  }

  analogReadResolution(12);                       // 0..4095
  analogSetPinAttenuation(PINO_SENSOR, ADC_11db); // faixa ate ~3,3 V

  // Aviso em tempo de execucao para a armadilha mais cara do projeto.
  if (!(PINO_SENSOR == 32 || PINO_SENSOR == 33 || PINO_SENSOR == 34 ||
        PINO_SENSOR == 35 || PINO_SENSOR == 36 || PINO_SENSOR == 39)) {
    Serial.println(F("[ALERTA] PINO_SENSOR nao esta no ADC1. Com o Wi-Fi "
                     "ligado as leituras serao invalidas. Use 32-36 ou 39."));
  }

  configurarWdt();
  carregarCalibracao();
  Serial.println(F("digite 'ajuda' para os comandos de calibracao"));

  conectarWifi();
  proximoCicloMs = millis();     // primeiro ciclo imediato
}

void loop() {
  esp_task_wdt_reset();

  // ORDEM DELIBERADA: o rele e tratado PRIMEIRO, a cada volta. Nada
  // relacionado a rede pode adiar o desligamento da bomba.
  atualizarRele();
  tratarSerial();

  if ((int32_t)(millis() - proximoCicloMs) < 0) {
    delay(10);                   // cede tempo ao scheduler do FreeRTOS
    return;
  }

  if (WiFi.status() != WL_CONNECTED && !conectarWifi(15000)) {
    // Sem rede: tenta de novo em 30 s. A bomba, se ligada, continua sendo
    // contada e desligada pelo atualizarRele() acima.
    proximoCicloMs = millis() + 30000UL;
    return;
  }

  int cru = lerAdcBruto();
  if (leituraPlausivel(cru)) {
    float vwc = adcParaVwc(cru);
    if (enviarLeitura(vwc, cru)) {
      buscarComando();
    }
  } else {
    // Leitura ruim NAO vai para o servidor: telemetria falsa levaria o
    // controlador a irrigar as cegas. Melhor ficar em silencio.
    Serial.println("[CICLO] leitura descartada; nada enviado");
  }

  // Enquanto a bomba corre, consulta mais rapido para poder reagir a um
  // cancelamento vindo do painel.
  uint32_t espera = releLigado ? 15 : intervaloS;
  proximoCicloMs = millis() + espera * 1000UL;
}
