#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP8266Camera.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* azureEndpoint = "https://objectverify-cebkd.eastus2.inference.ml.azure.com/score";
const char* apiKey = "wvOtACFuhZMf5RlMJq7BkzErtyTRHyHI";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi!");
}

void loop() {
  camera_fb_t *fb = espCamera.capture();
  if (!fb) {
    Serial.println("Failed to capture image");
    return;
  }

  // Encode image data as base64
  String base64Image = base64_encode((const unsigned char*)fb->buf, fb->len);

  // Clean up camera framebuffer
  espCamera.fb_return(fb);

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

    // Generate audio output based on detected object (e.g., play sound)
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
