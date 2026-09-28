// robot_control.h -- motors and the UDP command link, bolted onto the stock
// CameraWebServer sketch.
//
// WHY THIS EXISTS
// ---------------
// firmware/esp32cam_robot/ grew its own video stack: a UDP chunk transport, a
// camera drain task, an adaptive-quality loop and a beacon watchdog that
// reboots the board. All of that is switched off in MJPEG mode, but it is
// still there to go wrong. This variant keeps the stock CameraWebServer video
// path untouched -- byte-for-byte the code Espressif ships -- and adds only
// what is needed to drive a car:
//
//   * two PWM motor outputs
//   * a UDP listener on 44441 speaking the same `seq,cmd,v1,v2` protocol
//     robot_link.py already sends, so the PC side needs no changes at all
//   * a 150 ms failsafe that zeroes the motors when packets stop
//   * 1 Hz telemetry on 44444, which esp32_link_check.py looks for
//
// What is deliberately NOT here, compared with firmware/esp32cam_robot/:
//
//   * no UDP video path, so no adaptive quality loop and nothing that can
//     fight the /stream handler for the camera
//   * NO 120 s beacon reboot. The old sketch restarted the board when no UDP
//     beacon had arrived, which a browser or an HTTP-only run.py never sends;
//     the board came back on a new DHCP lease for no visible reason.
//   * no sensor power-down, which keyed on that same beacon
//
// The cost of dropping the UDP video path is broadcast discovery: this board
// must be reached by IP. `python run.py` (no --esp-ip) still finds it by
// scanning the subnet.

#pragma once

// Set to 0 to build the robot half out entirely: no motor PWM, no task, no UDP
// sockets. What is left is the stock CameraWebServer sketch plus the /status
// marker. Flash it that way to find out whether a picture problem comes from
// this file or from the camera path -- if the image is still wrong with this
// at 0, nothing in here is responsible.
#define ROBOT_ENABLE 1

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_camera.h"

// ==========================================
// HARDWARE
// ==========================================
// GPIO12 is a strapping pin (MTDI, flash voltage select) and must be low at
// boot. It is what the existing robot wiring uses, so it is kept -- but if the
// board ever refuses to boot with the motor driver attached, this is the first
// thing to suspect.
#define MOTOR_1A 12
#define MOTOR_1B 13
#define MOTOR_PWM_FREQ 50   // standard servo/ESC frame rate
#define MOTOR_PWM_RES 16
// 1000..2000 us out of a 20 ms period at 16-bit: 65535 * (us / 20000).
#define MAX_DUTY 3.27675

// ==========================================
// PROTOCOL (must match V2/robot_link.py)
// ==========================================
static const uint16_t CONTROL_PORT = 44441;    // PC -> ESP: seq,cmd,v1,v2
static const uint16_t DISCOVERY_PORT = 44444;  // both ways: beacon / telemetry

#define AUTOSTOP_INTERVAL 150    // ms with no drive packet -> motors off
#define LOST_GUI_TIMEOUT 90000   // ms with no contact -> mark PC gone
#define TELEMETRY_INTERVAL 1000  // ms between telemetry packets

// robot_link.py wraps its sequence number at 10000. A wrap looks like a huge
// backwards jump, so accept a low seq when the last one was near the top.
#define SEQ_LIMIT 10000
#define SEQ_CLOSE_UPPER 9000
#define SEQ_CLOSE_LOWER 1000

// Set by the framesize commands so the picture matches what the PC asked for.
extern const char *hostName;

#if defined(LED_GPIO_NUM)
// Both live in app_httpd.cpp. Reusing them means the UDP light command and the
// web page's LED slider drive the same PWM channel instead of fighting over
// GPIO4 -- the stock sketch already owns that pin via setupLedFlash().
extern int led_duty;
void enable_led(bool en);
#endif

static WiFiUDP udpControl;
static WiFiUDP udpDiscover;

static volatile int motorA = 0;  // us, 0 = stopped (not 1500 -- see failsafe)
static volatile int motorB = 0;
static volatile bool pcConnected = false;
static IPAddress serverIP;
static uint32_t lastControlSeq = 0;
static bool controlReset = true;
static unsigned long lastPacketTime = 0;
static unsigned long lastServerContact = 0;
static uint32_t telemetrySeq = 0;

// Applies a resolution preset the same way the old robot firmware's cmd 4..7
// did, so `RobotLink.set_framesize()` keeps working unchanged.
static void applyFramesize(framesize_t fs, int quality, const char *label) {
  sensor_t *s = esp_camera_sensor_get();
  if (!s) return;
  s->set_framesize(s, fs);
  s->set_quality(s, quality);
  Serial.printf(" [Robot] %s\n", label);
}

static void handleCommand(int cmd, int v1, int v2) {
  // v1/v2 repeat the command number on system commands as a crude checksum:
  // a corrupted drive packet must never be read as "reboot".
  switch (cmd) {
    case 0:  // drive: v1 = steer us, v2 = throttle us
      motorA = v1;
      motorB = v2;
      break;
    case 1:  // light: 0..65535 from the PC, 8-bit on the stock LED channel
#if defined(LED_GPIO_NUM)
      led_duty = constrain(v1 >> 8, 0, 255);
      enable_led(true);
#endif
      break;
    case 3:
      if (v1 == 3 && v2 == 3) {
        Serial.println(" [Robot] RESTART");
        delay(50);
        ESP.restart();
      }
      break;
    case 4:
      if (v1 == 4 && v2 == 4) applyFramesize(FRAMESIZE_QVGA, 14, "QVGA 320x240 q14");
      break;
    case 5:
      if (v1 == 5 && v2 == 5) applyFramesize(FRAMESIZE_VGA, 12, "VGA 640x480 q12");
      break;
    case 6:
      if (v1 == 6 && v2 == 6) applyFramesize(FRAMESIZE_SVGA, 11, "SVGA 800x600 q11");
      break;
    case 7:
      if (v1 == 7 && v2 == 7) applyFramesize(FRAMESIZE_XGA, 14, "XGA 1024x768 q14");
      break;
    case 8:  // "video over MJPEG" -- already the only transport here
      break;
    case 9:
      Serial.println(" [Robot] UDP video requested, but this build is MJPEG only");
      break;
    default:
      break;
  }
}

static void robotTask(void *pv) {
  udpControl.begin(CONTROL_PORT);
  udpDiscover.begin(DISCOVERY_PORT);
  Serial.printf(" [Robot] control UDP on %u, discovery on %u\n", CONTROL_PORT,
                DISCOVERY_PORT);

  lastServerContact = millis();

  while (true) {
    // 1. DISCOVERY -- any packet on 44444 tells us where the PC is.
    int discSize = udpDiscover.parsePacket();
    if (discSize) {
      IPAddress newIP = udpDiscover.remoteIP();
      lastServerContact = millis();
      if (!pcConnected || newIP != serverIP) {
        serverIP = newIP;
        Serial.print(" [Robot] PC at ");
        Serial.println(serverIP);
      }
      pcConnected = true;
      // A new PC restarts its sequence at 0, which would otherwise look like a
      // flood of stale packets until it caught up with lastControlSeq.
      controlReset = true;
      while (udpDiscover.available()) udpDiscover.read();
    }

    // 2. COMMANDS
    int packetSize = udpControl.parsePacket();
    if (packetSize) {
      char buf[128];
      int len = udpControl.read(buf, sizeof(buf) - 1);
      if (len > 0) {
        buf[len] = 0;
        String line = String(buf);
        int c1 = line.indexOf(',');
        int c2 = line.indexOf(',', c1 + 1);
        int c3 = line.lastIndexOf(',');
        if (c1 > 0 && c2 > c1 && c3 > c2) {
          uint32_t seq = line.substring(0, c1).toInt();
          int cmd = line.substring(c1 + 1, c2).toInt();
          int v1 = line.substring(c2 + 1, c3).toInt();
          int v2 = line.substring(c3 + 1).toInt();

          if (controlReset || seq > lastControlSeq ||
              (lastControlSeq > SEQ_CLOSE_UPPER && seq < SEQ_CLOSE_LOWER)) {
            lastControlSeq = seq;
            controlReset = false;
            lastPacketTime = millis();
            lastServerContact = millis();
            handleCommand(cmd, v1, v2);
          }
        }
      }
    }

    // 3. FAILSAFE. The PC re-sends an unchanged drive command every 50 ms, so
    // silence for 150 ms means the link is gone, not that nothing changed.
    if (millis() - lastPacketTime > AUTOSTOP_INTERVAL) {
      motorA = 0;
      motorB = 0;
    }

    if (pcConnected && millis() - lastServerContact > LOST_GUI_TIMEOUT) {
      Serial.println(" [Robot] lost PC contact");
      pcConnected = false;
    }
    // Note: no reboot on silence. The stock video path has no beacon, so a
    // browser session would trip any such watchdog.

    // 4. DRIVE THE PINS
    ledcWrite(MOTOR_1A, (int)(motorA * MAX_DUTY));
    ledcWrite(MOTOR_1B, (int)(motorB * MAX_DUTY));

    // 5. TELEMETRY -- what esp32_link_check.py listens for.
    static unsigned long lastTele = 0;
    if (pcConnected && millis() - lastTele > TELEMETRY_INTERVAL) {
      lastTele = millis();
      if (WiFi.status() == WL_CONNECTED) {
        char chipStr[10];
        snprintf(chipStr, sizeof(chipStr), "%08lX", (unsigned long)(uint32_t)ESP.getEfuseMac());
        udpDiscover.beginPacket(serverIP, DISCOVERY_PORT);
        udpDiscover.printf("%u,%s,RSSI:%d,SSID:%s,NAME:%s",
                           (unsigned)telemetrySeq, chipStr, WiFi.RSSI(),
                           WiFi.SSID().c_str(), hostName);
        udpDiscover.endPacket();
        telemetrySeq = (telemetrySeq + 1) % SEQ_LIMIT;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// Called from setup() once WiFi is up and the camera server is running.
void robot_setup() {
#if !ROBOT_ENABLE
  Serial.println(" [Robot] ROBOT_ENABLE=0 -- motors and UDP control are OUT of this build");
  return;
#else
  ledcAttach(MOTOR_1A, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
  ledcAttach(MOTOR_1B, MOTOR_PWM_FREQ, MOTOR_PWM_RES);
  ledcWrite(MOTOR_1A, 0);
  ledcWrite(MOTOR_1B, 0);

  // Core 1: core 0 runs the WiFi stack and the httpd stream task, and this
  // loop wakes 200x a second.
  // 8192, not 4096: this task parses with Arduino String and calls
  // udpDiscover.printf() with WiFi.SSID(), and the shipping firmware gives its
  // equivalent task 8192. A FreeRTOS stack lives in internal RAM next to the
  // camera driver's DMA descriptors, so undersizing it corrupts something far
  // away from the code that overflowed.
  xTaskCreatePinnedToCore(robotTask, "Robot", 8192, NULL, 3, NULL, 1);
#endif
}
