
#include <DHT.h>

// -------- Pin configuration --------
#define RAIN_PIN A0
#define SOIL_PIN A1
#define DHT_PIN 2
#define DHT_TYPE DHT11

#define GREEN_LED 5
#define YELLOW_LED 6
#define RED_LED 7
#define BUZZER_PIN 8

DHT dht(DHT_PIN, DHT_TYPE);

// -------- Settings --------
// Read sensors once every 5 seconds
const unsigned long SENSOR_INTERVAL = 5000;

unsigned long lastSensorRead = 0;

bool predictionActive = false;
bool manualAlertActive = false;
String currentRisk = "UNKNOWN";
String commandBuffer = "";

// -------- LED and buzzer control --------
void setIndicators(String risk) {
  currentRisk = risk;

  // While a manual alert is active, Red LED and Buzzer must stay ON.
  // Routine risk commands or sensor updates must not override or turn off the manual alert.
  if (manualAlertActive) {
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(RED_LED, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    return;
  }

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(YELLOW_LED, LOW);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  if (risk == "LOW") {
    digitalWrite(GREEN_LED, HIGH);
  }
  else if (risk == "MODERATE") {
    digitalWrite(YELLOW_LED, HIGH);
  }
  else if (risk == "HIGH") {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
  }
  // UNKNOWN: all outputs remain off.
}

// -------- Process dashboard commands --------
void processCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "START") {
    predictionActive = true;
    manualAlertActive = false;
    lastSensorRead = 0;
    setIndicators("UNKNOWN");
    Serial.println("STATUS,STARTED");
  }
  else if (command == "STOP") {
    predictionActive = false;
    manualAlertActive = false;
    setIndicators("UNKNOWN");
    Serial.println("STATUS,STOPPED");
  }
  else if (command == "MANUAL_ALERT") {
    manualAlertActive = true;
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(RED_LED, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    Serial.println("STATUS,MANUAL_ALERT_ON");
  }
  else if (command == "RESET_ALERT") {
    manualAlertActive = false;
    // Restore indicators to current automatic risk level if available; otherwise turn off.
    setIndicators(currentRisk);
    Serial.println("STATUS,MANUAL_ALERT_OFF");
  }
  else if (command == "LOW" ||
           command == "MODERATE" ||
           command == "HIGH" ||
           command == "UNKNOWN") {

    // Accept risk commands only while prediction is active.
    if (predictionActive) {
      setIndicators(command);
    }
  }
}

// -------- Read dashboard commands --------
void readDashboardCommands() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n') {
      processCommand(commandBuffer);
      commandBuffer = "";
    }
    else if (incoming != '\r') {
      if (commandBuffer.length() < 32) {
        commandBuffer += incoming;
      }
      else {
        commandBuffer = "";
      }
    }
  }
}

// -------- Send sensor data --------
void sendSensorReadings() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  int rainValue = digitalRead(RAIN_PIN);
  int soilValue = analogRead(SOIL_PIN);

  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("ERROR,DHT");
    return;
  }

  // Format expected by Python:
  // temperature,humidity,rain,soil
  Serial.print(temperature, 1);
  Serial.print(",");
  Serial.print(humidity, 1);
  Serial.print(",");
  Serial.print(rainValue);
  Serial.print(",");
  Serial.println(soilValue);
}

void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(RAIN_PIN, INPUT);
  pinMode(SOIL_PIN, INPUT);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  setIndicators("UNKNOWN");
}

void loop() {
  // Always listen for START, STOP, and risk commands.
  readDashboardCommands();

  // Send readings only when dashboard prediction is active.
  unsigned long now = millis();

  if (predictionActive &&
      now - lastSensorRead >= SENSOR_INTERVAL) {

    lastSensorRead = now;
    sendSensorReadings();
  }
}
