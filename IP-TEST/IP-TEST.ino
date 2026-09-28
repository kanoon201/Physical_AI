#include <WiFi.h>

const char* ssid     = "KUWIN-IOT";
const char* password = ""; // เว้นว่างไว้ ไม่ต้องใส่รหัสผ่าน

void setup() {
    Serial.begin(115200);
    
    // พิมพ์ค่า MAC Address ออกมาเพื่อนำไปลงทะเบียนในระบบ Smart KU
    Serial.print("ESP32 MAC Address: ");
    Serial.println(WiFi.macAddress());
    
    // เริ่มต้นการเชื่อมต่อ
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("");
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

void loop() {
    // โค้ดโปรเจกต์ของคุณ
}
