#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>
#include <CameraLibrary.h> // Example camera library for ESP8266

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* azureEndpoint = "YOUR_AZURE_API_ENDPOINT";
const char* apiKey = "YOUR_AZURE_API_KEY";

void setup(){
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi!");
}

void loop() {
  Camera.capture(); // Example function to capture image

  // Encode image data as base64
  String base64Image = Camera.getBase64Image();

  // Construct JSON payload
  StaticJsonDocument<200> doc;
  doc["image"] = base64Image;
  String payload;
  serializeJson(doc, payload);

  // Send HTTP POST request to Azure API
  HTTPClient http;
  http.begin(azureEndpoint);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("x-api-key", apiKey);
  int httpResponseCode = http.POST(payload);

  if (httpResponseCode > 0) {
    String response = http.getString();
    // Parse JSON response from Azure API (if applicable)
    StaticJsonDocument<200> jsonResult;
    deserializeJson(jsonResult, response);
    const char* detectedObject = jsonResult["object"];
    Serial.println(detectedObject);

    // Perform actions based on detected object (e.g., play sound)
    if (strcmp(detectedObject, "person") == 0) {
      // Play sound for detected person (example)
      tone(13, 1000, 1000); // Example tone generation (adjust pin and frequency)
    }
  } else {
    Serial.print("Error on HTTP request: ");
    Serial.println(httpResponseCode);
  }

  http.end();
  delay(5000); // Delay before capturing next image (adjust as needed)
}
