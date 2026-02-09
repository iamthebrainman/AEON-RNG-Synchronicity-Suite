"""ESP32 firmware generation helpers.

This module generates a reference Arduino/ESP32 sketch that sends hardware RNG
packets over UDP to the AEON host. The ESP32 firmware is experimental and
network reliability may vary by Wi‑Fi environment and router settings. The
template attempts to improve robustness (disables Wi‑Fi power save, scans for
BSSID/channel, uses explicit connect), but users should consult
`esp32_firmware/README.md` for troubleshooting and tuning (payload size,
send delay, retry strategy, host firewall/UDP buffer).
"""

import os

FIRMWARE_TEMPLATE = r"""
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_random.h"

// --- CONFIGURATION ---
const char* ssid = "{ssid}";
const char* password = "{password}";
const char* host_ip = "{host_ip}";
const uint16_t host_port = {host_port};
const uint32_t node_id = {node_id};

// --- INTERNAL ---
WiFiUDP udp;
uint32_t seq_num = 0;
#define PAYLOAD_SIZE 256 // Bytes of entropy per packet

// LED CONFIG
#ifndef LED_BUILTIN
  #define LED_PIN 2
#else
  #define LED_PIN LED_BUILTIN
#endif

void setup() {{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); // Start Off

  // 1. Initialize WiFi & Disable Power Save (Fix for some routers)
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(100);
  WiFi.setSleep(false); 

  Serial.println("Initializing Bulletproof Connection...");
  
  // 2. Scan for the network specifically to find Channel & BSSID
  Serial.print("Scanning for "); Serial.print(ssid); Serial.println("...");
  int n = WiFi.scanNetworks();
  int targetIdx = -1;
  
  for (int i = 0; i < n; ++i) {{
    if (WiFi.SSID(i) == ssid) {{
      targetIdx = i;
      break;
    }}
  }}
  
  // 3. Connect using explicit Channel/BSSID if found
  if (targetIdx >= 0) {{
    uint8_t bssid[6];
    memcpy(bssid, WiFi.BSSID(targetIdx), 6);
    int channel = WiFi.channel(targetIdx);
    
    Serial.print("Found! Connecting to BSSID: ");
    Serial.print(WiFi.BSSIDstr(targetIdx));
    Serial.print(" on Channel: ");
    Serial.println(channel);
    
    // Explicit connect bypasses ambiguity
    WiFi.begin(ssid, password, channel, bssid);
  }} else {{
    Serial.println("Network not found in scan! Trying blind connect...");
    WiFi.begin(ssid, password);
  }}
  
  // 4. Wait for connection
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {{
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Toggle
    delay(500);
    Serial.print(".");
    
    if (++tries % 10 == 0) {{
      Serial.print(" [Status: ");
      Serial.print(WiFi.status());
      Serial.print("] ");
      if (WiFi.status() == 6 && tries > 20) {{
         Serial.println("\nStuck on 6? Re-forcing begin...");
         WiFi.disconnect(); 
         delay(500);
         // Retry blind if specific failed
         WiFi.begin(ssid, password);
         tries = 0;
      }}
    }}
  }}
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
  
  // Connection success pattern: 5 rapid blinks
  for(int i=0; i<5; i++) {{
    digitalWrite(LED_PIN, HIGH); delay(50);
    digitalWrite(LED_PIN, LOW); delay(50);
  }}
}}

void loop() {{
  // Ensure we are still connected
  if (WiFi.status() != WL_CONNECTED) {{
    Serial.println("WiFi lost, reconnecting...");
    WiFi.disconnect();
    WiFi.reconnect();
    
    // Blink slowly while waiting for reconnect
    while (WiFi.status() != WL_CONNECTED) {{
       digitalWrite(LED_PIN, HIGH); delay(500);
       digitalWrite(LED_PIN, LOW); delay(500);
       Serial.print(".");
    }}
    Serial.println("\nReconnected!");
  }}

  uint8_t buffer[8 + PAYLOAD_SIZE];
  
  // Header
  memcpy(buffer, &node_id, 4);
  memcpy(buffer + 4, &seq_num, 4);
  
  // Payload (Hardware RNG)
  for (int i = 0; i < PAYLOAD_SIZE; i += 4) {{
    uint32_t r = esp_random();
    memcpy(buffer + 8 + i, &r, 4);
  }}
  
  // Blink on data push (brief flash)
  digitalWrite(LED_PIN, HIGH);
  udp.beginPacket(host_ip, host_port);
  udp.write(buffer, sizeof(buffer));
  udp.endPacket();
  digitalWrite(LED_PIN, LOW);
  
  seq_num++;
  delay(30); // Throttle slightly
}}
"""

def generate_firmware(ssid, password, host_ip, host_port=5000, node_id=0xAE01, output_path=None):
    if output_path is None:
        # Save in the esp32_firmware subfolder for Arduino IDE compatibility
        base_dir = os.path.join("esp32_firmware", "aeon_rng_node")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        output_path = os.path.join(base_dir, "aeon_rng_node.ino")

    code = FIRMWARE_TEMPLATE.format(
        ssid=ssid,
        password=password,
        host_ip=host_ip,
        host_port=host_port,
        node_id=node_id
    )
    
    with open(output_path, "w") as f:
        f.write(code.lstrip())
    
    # Return both relative and absolute path for UI clarity
    return os.path.abspath(output_path)

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 3:
        generate_firmware(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python firmware_gen.py <SSID> <PASSWORD> <HOST_IP>")
