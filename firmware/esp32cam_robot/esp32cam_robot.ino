#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_system.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_log.h>
#include <esp_task_wdt.h>
#include "esp_http_server.h"
#include "esp_timer.h"

const char *hostName = "my-robot-cam";

// --- 1. NETWORK & PORTS ---
#define WIFICOUNT 1
// Real credentials live in wifi_secrets.h, which is gitignored. Copy
// wifi_secrets.example.h to wifi_secrets.h and edit it before flashing.
#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#else
#define WIFI_SSID "KUWIN-IOT"
#define WIFI_PASSWORD ""
#endif
const char *ssids[WIFICOUNT] = {WIFI_SSID};
const char *passs[WIFICOUNT] = {WIFI_PASSWORD};

// HTTP video, same layout as CameraWebServer: control on 80, stream on 81.
// Two separate httpd instances on purpose -- /status still answers while
// the stream holds its slot, which is what makes live diagnosis possible.
#define HTTP_CTRL_PORT 80
#define HTTP_STREAM_PORT 81

const int controlPort = 44441;
const int videoPort = 44442;
const int discoveryPort = 44444;

// --- 2. TIMEOUTS & INTERVALS ---
#define BEACON_TIMEOUT 120000     // 120s reboot if no server contact
#define LOST_GUI_TIMEOUT 90000    // 90s to mark GUI as disconnected
#define TELEMETRY_INTERVAL 1000   // 1s between telemetry packets
#define WATCHDOG_TIMEOUT_S 5      // 5s hardware watchdog
#define AUTOSTOP_INTERVAL 150     // 150ms engine failsafe
#define CHUNK_DELAY_US 1000       // 1ms to let OS breathe without stalling
#define POWER_SAVE_DELAY 500      // 500ms sleep loop when disconnected

#define VIDEO_CHUNK_SIZE 1400     // Safer max payload for MTU
#define CHUNK_YIELD_EVERY 4       // Yield to the scheduler every N chunks, not every chunk
#define CHUNK_GAP_US 500          // Rough air-time pacing; too small floods the WiFi TX queue
#define MIN_FRAME_INTERVAL 15     // Floor only; tx_ready is the real pacer
#define QUALITY_BEST 10           // Lower = better image (esp32-camera is inverted)
#define QUALITY_WORST 28          // Never degrade past this, even on a bad link
#define SEQ_LIMIT 10000
#define SEQ_CLOSE_UPPER 9000
#define SEQ_CLOSE_LOWER 1000

// --- 3. HARDWARE CONFIG ---
#define LED_PIN 4
#define MOTOR_1A 12
#define MOTOR_1B 13
#define PWM_RES 16
#define PWM_FREQ 50
#define MAX_DUTY 3.27675

// State
IPAddress serverIP;
bool pcConnected = false;
uint32_t lastControlSeq = 0xFFFFFFFF;
uint32_t telemetry_seq = 0;
unsigned long lastPacketTime = 0;
unsigned long lastServerContact = 0;

WiFiUDP videoUDP;
WiFiUDP udpControl;
WiFiUDP udpDiscover;

volatile int previous1A = 0;
volatile int previous1B = 0;
volatile int oldlight = 0;
// Set once the browser page drives the light, so loop() stops
// overwriting it with the "no beacon" blink.
volatile bool light_override = false;
volatile bool CAMCONFIG = false;
volatile bool controlReset = true;
volatile framesize_t current_framesize = FRAMESIZE_VGA;
volatile int current_quality = 12;
volatile int video_frame_interval = MIN_FRAME_INTERVAL; // Floor; tx_ready paces the rest

// VIDEO TRANSPORT. MJPEG-over-TCP is the default: TCP retransmits, so a frame
// either arrives byte-exact or the socket stalls -- it can never arrive
// half-decoded, and there is no per-chunk loss for the adaptive loop to chase.
// The UDP chunk path is kept as a fallback (cmd 9) because it is the only one
// that can broadcast-discover the PC without knowing an IP first.
volatile bool video_udp_mode = false;
volatile bool http_streaming = false;
// Health counters for /status. Sampled by link_monitor.py every few seconds
// to tell a board that degrades with uptime (heap shrinking, chip heating,
// frames growing) from a radio link that degrades on its own.
volatile uint32_t stream_frames_sent = 0;   // MJPEG frames handed to httpd since boot
volatile uint32_t stream_last_len = 0;      // size of the last one, bytes
httpd_handle_t camera_httpd = NULL;
httpd_handle_t stream_httpd = NULL;

// ==========================================
// CAMERA INIT
// ==========================================
void updateCameraSettings() {
  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_framesize(s, current_framesize);
    s->set_quality(s, current_quality);

    // Motion blur is the real enemy on a moving car: the sensor is rolling
    // shutter, and left alone AEC picks a long exposure and smears everything.
    // Bias darker so the shutter stays short, and let gain make up the light.
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);                      // DSP-assisted metering
    s->set_ae_level(s, -1);                 // darker bias => shorter shutter
    s->set_gain_ctrl(s, 1);
    s->set_gainceiling(s, GAINCEILING_4X);  // cap gain so noise stays sane

    // These are all off by driver default and all worth having.
    s->set_lenc(s, 1);      // lens correction: kills corner vignetting
    s->set_bpc(s, 1);       // black pixel correction
    s->set_wpc(s, 1);       // white pixel correction (hot pixels)
    s->set_raw_gma(s, 1);   // gamma curve: much better tonal range
    s->set_dcw(s, 1);       // better downscaling

    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);   // 0=auto. Pin to 1 (sunny) if AWB hunts outdoors.

    s->set_contrast(s, 1);  // slight bump survives JPEG compression better
    s->set_saturation(s, 0);
    // NOTE: set_sharpness()/set_denoise() are unimplemented for OV2640 and
    // silently return -1. Don't bother.

    Serial.printf("Camera Mode Updated: FS %d, Q %d\n", current_framesize,
                  current_quality);
  }
}

void initCamera() {
  CAMCONFIG = false;
  Serial.printf("Init Camera Driver...\n");
  esp_camera_deinit();
  delay(100);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000; // Full rate: more sensor throughput
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = current_framesize;
  config.jpeg_quality = current_quality;
  // With GRAB_LATEST, every extra buffer is another stale frame queued ahead
  // of you. 6 was adding ~6 frames of steering latency; 2 is what you want.
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    delay(1000);
    ESP.restart();
  }

  Serial.println("Stabilizing Sensor...");
  for (int i = 0; i < 15; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      esp_camera_fb_return(fb);
      delay(20);
    }
    esp_task_wdt_reset();
  }

  // esp_camera_init() only applies what is in camera_config_t: framesize,
  // quality, buffers. Everything the sensor actually needs to look right
  // and stay fast -- the -1 exposure bias that keeps the shutter short,
  // the gain ceiling, lens/black-pixel/gamma correction -- lives in
  // updateCameraSettings(), and until this call it only ran when a
  // framesize command or the adaptive loop happened to fire. A board that
  // was left alone therefore streamed with AEC free to pick a long
  // exposure, which in a dim room drops the SENSOR's own framerate to a
  // couple of fps no matter how much WiFi headroom there is.
  updateCameraSettings();

  Serial.println("Camera Init Success");
  CAMCONFIG = true;
}

// ==========================================
// HTTP VIDEO SERVER (the CameraWebServer transport)
// ==========================================
// Byte-for-byte the same wire format CameraWebServer serves, so the PC side
// needs no new receiver: mjpeg_video_source.py reads /stream on 81, and
// camera_control.py drives /control, /capture and /status on 80 exactly as
// it does against the stock sketch.

#define STREAM_BOUNDARY_TOKEN "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" STREAM_BOUNDARY_TOKEN;
static const char *STREAM_BOUNDARY = "\r\n--" STREAM_BOUNDARY_TOKEN "\r\n";
static const char *STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

static esp_err_t http_stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK)
    return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  http_streaming = true;
  Serial.println(" [HTTP] Stream client connected");

  while (true) {
    // An open stream IS contact with the PC. Without this the beacon
    // watchdog in loopcommand() reboots the board every 120s whenever the PC
    // is watching video over HTTP and never sends a UDP beacon.
    lastServerContact = millis();

    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println(" [HTTP] Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART,
                             (unsigned)fb->len, (int)fb->timestamp.tv_sec,
                             (int)fb->timestamp.tv_usec);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    if (res == ESP_OK) {
      stream_frames_sent++;
      stream_last_len = fb->len;
    }

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK)
      break; // client went away
  }

  http_streaming = false;
  Serial.println(" [HTTP] Stream client gone");
  return res;
}

static esp_err_t http_capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t http_status_handler(httpd_req_t *req) {
  sensor_t *s = esp_camera_sensor_get();
  char buf[1024];
  // "robot":1 is the marker autodrive.py checks for: it says this board has
  // motors, which the stock CameraWebServer sketch does not. Without it the
  // PC has no way to tell the two firmwares apart over HTTP.
  int n = snprintf(
      buf, sizeof(buf),
      "{\"robot\":1,\"video_mode\":\"%s\",\"framesize\":%d,\"quality\":%d,"
      "\"awb\":%d,\"awb_gain\":%d,\"aec\":%d,\"aec2\":%d,\"agc\":%d,"
      "\"ae_level\":%d,\"vflip\":%d,\"hmirror\":%d,\"special_effect\":%d,"
      "\"brightness\":%d,\"contrast\":%d,\"saturation\":%d,"
      "\"lenc\":%d,\"raw_gma\":%d,\"dcw\":%d,\"light\":%d,"
      "\"rssi\":%d,\"pc_connected\":%d,"
      // Health block. temp_c is the ESP32's internal sensor: its absolute
      // value is unreliable, its TREND over a session is not.
      "\"uptime_s\":%lu,\"heap_free\":%u,\"heap_min_free\":%u,"
      "\"heap_largest\":%u,\"psram_free\":%u,\"temp_c\":%.1f,"
      "\"stream_frames\":%lu,\"stream_last_len\":%lu}",
      video_udp_mode ? "udp" : "mjpeg", s ? (int)s->status.framesize : -1,
      s ? s->status.quality : -1, s ? s->status.awb : -1,
      s ? s->status.awb_gain : -1, s ? s->status.aec : -1,
      s ? s->status.aec2 : -1, s ? s->status.agc : -1,
      s ? s->status.ae_level : 0, s ? s->status.vflip : 0,
      s ? s->status.hmirror : 0, s ? s->status.special_effect : 0,
      s ? s->status.brightness : 0, s ? s->status.contrast : 0,
      s ? s->status.saturation : 0, s ? s->status.lenc : 0,
      s ? s->status.raw_gma : 0, s ? s->status.dcw : 0, oldlight, WiFi.RSSI(),
      pcConnected ? 1 : 0,
      (unsigned long)(millis() / 1000),
      (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
      (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
      (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
      (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
      temperatureRead(),
      (unsigned long)stream_frames_sent, (unsigned long)stream_last_len);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, buf, n);
}

static esp_err_t http_control_handler(httpd_req_t *req) {
  char query[128];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }
  char var[32], val_s[32];
  if (httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(query, "val", val_s, sizeof(val_s)) != ESP_OK) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }
  int val = atoi(val_s);

  sensor_t *s = esp_camera_sensor_get();
  if (!s) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  int res = 0;

  // framesize and quality are mirrored into the globals so the UDP fallback
  // and updateCameraSettings() stay in step with whatever HTTP just set.
  if (!strcmp(var, "framesize")) {
    current_framesize = (framesize_t)val;
    res = s->set_framesize(s, (framesize_t)val);
  } else if (!strcmp(var, "quality")) {
    current_quality = val;
    res = s->set_quality(s, val);
  } else if (!strcmp(var, "contrast"))
    res = s->set_contrast(s, val);
  else if (!strcmp(var, "brightness"))
    res = s->set_brightness(s, val);
  else if (!strcmp(var, "saturation"))
    res = s->set_saturation(s, val);
  else if (!strcmp(var, "gainceiling"))
    res = s->set_gainceiling(s, (gainceiling_t)val);
  else if (!strcmp(var, "colorbar"))
    res = s->set_colorbar(s, val);
  else if (!strcmp(var, "awb"))
    res = s->set_whitebal(s, val);
  else if (!strcmp(var, "agc"))
    res = s->set_gain_ctrl(s, val);
  else if (!strcmp(var, "aec"))
    res = s->set_exposure_ctrl(s, val);
  else if (!strcmp(var, "hmirror"))
    res = s->set_hmirror(s, val);
  else if (!strcmp(var, "vflip"))
    res = s->set_vflip(s, val);
  else if (!strcmp(var, "awb_gain"))
    res = s->set_awb_gain(s, val);
  else if (!strcmp(var, "agc_gain"))
    res = s->set_agc_gain(s, val);
  else if (!strcmp(var, "aec_value"))
    res = s->set_aec_value(s, val);
  else if (!strcmp(var, "aec2"))
    res = s->set_aec2(s, val);
  else if (!strcmp(var, "dcw"))
    res = s->set_dcw(s, val);
  else if (!strcmp(var, "bpc"))
    res = s->set_bpc(s, val);
  else if (!strcmp(var, "wpc"))
    res = s->set_wpc(s, val);
  else if (!strcmp(var, "raw_gma"))
    res = s->set_raw_gma(s, val);
  else if (!strcmp(var, "lenc"))
    res = s->set_lenc(s, val);
  else if (!strcmp(var, "special_effect"))
    res = s->set_special_effect(s, val);
  else if (!strcmp(var, "wb_mode"))
    res = s->set_wb_mode(s, val);
  else if (!strcmp(var, "ae_level"))
    res = s->set_ae_level(s, val);
  else if (!strcmp(var, "light")) {
    // The flash LED is on GPIO4 and PWM_RES is 16-bit, so full scale is 65535.
    // The browser page caps its slider well below that: this LED is bright
    // enough to pull the 5V rail down, and a browning-out radio is the first
    // thing that breaks on this board.
    if (val < 0) val = 0;
    if (val > 65535) val = 65535;
    oldlight = val;
    // Without this, loop() keeps blinking the "no beacon" heartbeat over the
    // value we just set -- an HTTP-only client never sets pcConnected.
    light_override = true;
    ledcWrite(LED_PIN, oldlight);
  } else if (!strcmp(var, "video_mode")) {
    // 0 = MJPEG over HTTP (default), 1 = UDP chunks. Same switch as UDP cmd 8/9.
    video_udp_mode = (val != 0);
    Serial.printf(" [Video] transport -> %s (via HTTP)\n",
                  video_udp_mode ? "UDP" : "MJPEG");
  } else if (!strcmp(var, "restart")) {
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, NULL, 0);
    delay(200);
    ESP.restart();
  } else {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }

  if (res < 0) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, NULL, 0);
}

// ==========================================
// HTTP INDEX PAGE (browser UI on port 80)
// ==========================================
// Served from "/" so the board can be opened in a browser the way
// CameraWebServer can, without pulling in that sketch's ~20 KB gzipped
// camera_index.h. The page is split in two halves so the framesize <option>
// values can be emitted between them from the real framesize_t constants:
// `val` in /control?var=framesize is a raw enum index, and the esp32 core has
// inserted new entries into that enum before (128X128, 320X320), which is
// exactly how hardcoded indices go stale.
//
// Motor control is deliberately NOT on this page. /control has no auth, and a
// browser widget that drives the car would let anyone on the network do the
// same. Driving stays on the UDP command port, where autodrive.py's arming
// gate is.

static const char INDEX_HTML_HEAD[] PROGMEM = R"rawliteral(
<title>ESP32-CAM Robot</title>
<style>
:root{--bg:#12141a;--panel:#1b1e26;--line:#2b3040;--fg:#e6e8ee;--dim:#8b93a7;--accent:#4da3ff;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
h1{margin:0;font-size:16px;font-weight:600}
.tag{font-size:12px;color:var(--dim)}
.wrap{display:flex;gap:16px;padding:16px;align-items:flex-start;flex-wrap:wrap}
.video{flex:1 1 420px;min-width:300px}
.side{flex:0 1 320px;min-width:280px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.panel h2{margin:0 0 10px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);font-weight:600}
#stream{width:100%;display:block;border-radius:10px;background:#000;min-height:180px}
.row{display:flex;align-items:center;gap:8px;margin:7px 0}
.row label{flex:0 0 96px;color:var(--dim);font-size:13px}
.row input[type=range]{flex:1;min-width:0}
.row select{flex:1;min-width:0;background:#12151c;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px}
.val{flex:0 0 46px;text-align:right;font-variant-numeric:tabular-nums;color:var(--accent)}
button{background:#252a36;color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:8px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}
button.on{background:var(--accent);color:#0b0d12;border-color:var(--accent)}
button.danger:hover{border-color:#ff6b6b;color:#ff6b6b}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.toggles{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:6px;margin-top:10px}
.toggles button{padding:6px;font-size:12px}
.note{background:#241f14;border:1px solid #4a3a1a;color:var(--warn);border-radius:8px;padding:9px 11px;font-size:12.5px;margin:0 16px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
td{padding:3px 0;color:var(--dim)}
td:last-child{text-align:right;color:var(--fg);font-variant-numeric:tabular-nums}
</style>
<header>
  <h1>ESP32-CAM Robot</h1>
  <span class="tag" id="host"></span>
  <span class="tag" id="fpsTag">-- fps</span>
</header>
<p class="note">The camera serves <b>one stream client at a time</b>. Press Stop (or close this tab) before running <code>run.py</code> / <code>autodrive.py</code>, or Python gets no frames.</p>
<div class="wrap">
  <div class="video">
    <img id="stream" alt="stream stopped">
    <div class="btns" style="margin-top:8px">
      <button id="btnStart" class="on">Start stream</button>
      <button id="btnStop">Stop stream</button>
      <button id="btnSnap">Snapshot</button>
    </div>
  </div>
  <div class="side">
    <div class="panel">
      <h2>Picture</h2>
      <div class="row"><label>Resolution</label><select id="framesize"></select></div>
      <div class="row"><label>Quality</label><input type="range" id="quality" min="10" max="63"><span class="val" id="qualityV"></span></div>
      <div class="row"><label>AE level</label><input type="range" id="ae_level" min="-2" max="2"><span class="val" id="ae_levelV"></span></div>
      <div class="row"><label>Brightness</label><input type="range" id="brightness" min="-2" max="2"><span class="val" id="brightnessV"></span></div>
      <div class="row"><label>Contrast</label><input type="range" id="contrast" min="-2" max="2"><span class="val" id="contrastV"></span></div>
      <div class="row"><label>Saturation</label><input type="range" id="saturation" min="-2" max="2"><span class="val" id="saturationV"></span></div>
      <div class="toggles" id="toggles"></div>
    </div>
    <div class="panel">
      <h2>Robot</h2>
      <div class="row"><label>Light</label><input type="range" id="light" min="0" max="16000" step="250" value="0"><span class="val" id="lightV">0</span></div>
      <div class="row"><label>Transport</label><select id="video_mode"><option value="0">MJPEG / HTTP</option><option value="1">UDP chunks</option></select></div>
      <div class="btns"><button id="btnRestart" class="danger">Restart board</button></div>
    </div>
    <div class="panel">
      <h2>Status</h2>
      <table id="statusTable"></table>
    </div>
  </div>
</div>
)rawliteral";

static const char INDEX_HTML_TAIL[] PROGMEM = R"rawliteral(
<script>
var $=function(id){return document.getElementById(id)};
var host=location.hostname, ctrl='http://'+host, streamURL='http://'+host+':81/stream';
$('host').textContent=host;

function set(v,val){return fetch(ctrl+'/control?var='+v+'&val='+val).catch(function(){})}

// The resolution list comes from FS, which the firmware emits from the real
// framesize_t constants. Hardcoding the indices breaks whenever the esp32
// core inserts a new entry into that enum -- it has done so before.
var sel=$('framesize');
FS.forEach(function(f){var o=document.createElement('option');o.value=f[0];o.textContent=f[1];sel.appendChild(o)});
sel.onchange=function(){set('framesize',sel.value)};
$('video_mode').onchange=function(){set('video_mode',$('video_mode').value)};

['quality','ae_level','brightness','contrast','saturation'].forEach(function(v){
  var el=$(v);
  el.oninput=function(){$(v+'V').textContent=el.value};
  el.onchange=function(){set(v,el.value)};
});
var lt=$('light');
lt.oninput=function(){$('lightV').textContent=lt.value};
lt.onchange=function(){set('light',lt.value)};

var TOGGLES=[['awb','AWB'],['awb_gain','AWB gain'],['aec','AEC'],['aec2','AEC2'],['agc','AGC'],
             ['vflip','V-Flip'],['hmirror','H-Mirror'],['lenc','Lens corr'],['raw_gma','Gamma'],['dcw','DCW']];
var tgEl={};
TOGGLES.forEach(function(t){
  var b=document.createElement('button');
  b.textContent=t[1]; b.dataset.v='0';
  b.onclick=function(){var nv=(b.dataset.v==='1')?0:1; set(t[0],nv).then(function(){mark(b,nv)})};
  $('toggles').appendChild(b); tgEl[t[0]]=b;
});
function mark(b,v){b.dataset.v=v?'1':'0'; b.classList.toggle('on',!!v)}

var img=$('stream'), running=false;
function start(){
  if(running)return;
  running=true; img.src=streamURL+'?t='+Date.now();
  $('btnStart').classList.add('on'); $('btnStop').classList.remove('on');
}
function stop(){
  running=false; img.removeAttribute('src'); img.alt='stream stopped';
  $('btnStart').classList.remove('on'); $('btnStop').classList.add('on');
  $('fpsTag').textContent='-- fps';
}
$('btnStart').onclick=start;
$('btnStop').onclick=stop;
$('btnSnap').onclick=function(){window.open(ctrl+'/capture?t='+Date.now(),'_blank')};
$('btnRestart').onclick=function(){if(confirm('Restart the board?'))set('restart',1)};
// Dropping the stream on unload frees the single client slot immediately, so
// run.py can connect right after the tab closes instead of waiting for a
// TCP timeout.
window.addEventListener('pagehide',stop);

// fps from the img load events: one fires per MJPEG part.
var fc=0, ft=Date.now();
img.addEventListener('load',function(){
  fc++;
  var d=(Date.now()-ft)/1000;
  if(d>=2){$('fpsTag').textContent=(fc/d).toFixed(1)+' fps'; fc=0; ft=Date.now()}
});

var ROWS=[['framesize','Framesize'],['quality','Quality'],['video_mode','Transport'],
          ['rssi','RSSI (dBm)'],['pc_connected','UDP beacon'],['robot','Robot firmware']];
function refresh(){
  fetch(ctrl+'/status').then(function(r){return r.json()}).then(function(s){
    var h='';
    ROWS.forEach(function(r){if(s[r[0]]!==undefined)h+='<tr><td>'+r[1]+'</td><td>'+s[r[0]]+'</td></tr>'});
    $('statusTable').innerHTML=h;
    ['quality','ae_level','brightness','contrast','saturation'].forEach(function(v){
      if(s[v]!==undefined&&document.activeElement!==$(v)){$(v).value=s[v]; $(v+'V').textContent=s[v]}
    });
    if(s.framesize!==undefined&&document.activeElement!==sel)sel.value=s.framesize;
    if(s.video_mode!==undefined)$('video_mode').value=(s.video_mode==='udp')?'1':'0';
    TOGGLES.forEach(function(t){if(s[t[0]]!==undefined)mark(tgEl[t[0]],s[t[0]])});
  }).catch(function(){});
}
refresh();
setInterval(refresh,2000);
start();
</script>
)rawliteral";

static esp_err_t http_index_handler(httpd_req_t *req) {
  char fs[192];
  int n = snprintf(fs, sizeof(fs),
                   "<script>var FS=[[%d,'QVGA 320x240'],[%d,'CIF 400x296'],"
                   "[%d,'VGA 640x480'],[%d,'SVGA 800x600'],[%d,'XGA 1024x768']];"
                   "</script>",
                   (int)FRAMESIZE_QVGA, (int)FRAMESIZE_CIF, (int)FRAMESIZE_VGA,
                   (int)FRAMESIZE_SVGA, (int)FRAMESIZE_XGA);

  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Content-Encoding", "identity");
  esp_err_t res = httpd_resp_send_chunk(req, INDEX_HTML_HEAD,
                                        strlen_P(INDEX_HTML_HEAD));
  if (res == ESP_OK)
    res = httpd_resp_send_chunk(req, fs, n);
  if (res == ESP_OK)
    res = httpd_resp_send_chunk(req, INDEX_HTML_TAIL,
                                strlen_P(INDEX_HTML_TAIL));
  if (res == ESP_OK)
    res = httpd_resp_send_chunk(req, NULL, 0); // end of response
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 8;
  config.stack_size = 8192;

  httpd_uri_t index_uri = {"/", HTTP_GET, http_index_handler, NULL};
  httpd_uri_t status_uri = {"/status", HTTP_GET, http_status_handler, NULL};
  httpd_uri_t control_uri = {"/control", HTTP_GET, http_control_handler, NULL};
  httpd_uri_t capture_uri = {"/capture", HTTP_GET, http_capture_handler, NULL};
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, http_stream_handler, NULL};

  config.server_port = HTTP_CTRL_PORT;
  config.ctrl_port = 32768;
  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
    httpd_register_uri_handler(camera_httpd, &control_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    Serial.printf(" [HTTP] control server on port %d\n", HTTP_CTRL_PORT);
  } else {
    Serial.println(" [HTTP] control server FAILED to start");
  }

  // A second httpd instance, not a second URI on the first one: the stream
  // handler never returns while a client is attached, and would otherwise
  // block /status and /control for as long as video is running.
  config.server_port = HTTP_STREAM_PORT;
  config.ctrl_port = 32769;
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.printf(" [HTTP] stream server on port %d/stream\n",
                  HTTP_STREAM_PORT);
  } else {
    Serial.println(" [HTTP] stream server FAILED to start");
  }
}

// ==========================================
// TASKS: VIDEO STREAM (PSRAM Double-Buffered)
// ==========================================
uint8_t *tx_buffer = NULL;
volatile size_t tx_frame_size = 0;
volatile bool tx_ready = false;
volatile uint32_t current_seq_global = 0;

void cameraDrainTask(void *pvParameters) {
  esp_task_wdt_add(NULL);
  unsigned long lastCam = 0;
  static bool sensor_sleeping = false;

  while (true) {
    esp_task_wdt_reset();
    if (!CAMCONFIG) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // MJPEG mode: the httpd stream handler owns the camera. Grabbing frames
    // here as well would fight it over the two frame buffers and burn core 0
    // -- the core the WiFi stack runs on -- producing pictures nobody sends.
    // This also keeps the sensor power-down below out of the way: it keys on
    // the UDP beacon, which an HTTP-only client never sends.
    if (!video_udp_mode) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (!pcConnected) {
      if (!sensor_sleeping) {
        sensor_t *s = esp_camera_sensor_get();
        if (s) {
          s->set_reg(s, 0xff, 0xff, 0x01); // bank 1
          s->set_reg(s, 0x12, 0xff, 0x40); // Power down
          sensor_sleeping = true;
          Serial.println(" [Power] Sensor Sleeping...");
        }
      }
      vTaskDelay(pdMS_TO_TICKS(POWER_SAVE_DELAY));
      continue;
    } else {
      if (sensor_sleeping) {
        sensor_t *s = esp_camera_sensor_get();
        if (s) {
          s->set_reg(s, 0xff, 0xff, 0x01); // bank 1
          s->set_reg(s, 0x12, 0xff, 0x00); // Power up
          sensor_sleeping = false;
          Serial.println(" [Power] Sensor Waking Up (Stabilizing)...");
          for (int i = 0; i < 5; i++) {
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb)
              esp_camera_fb_return(fb);
            vTaskDelay(pdMS_TO_TICKS(60));
            esp_task_wdt_reset();
          }
          Serial.println(" [Power] Sensor Ready.");
        }
      }
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue; 
    }

    if (!tx_ready && (millis() - lastCam > (unsigned long)video_frame_interval)) {
      size_t frame_size_loc = fb->len;
      if (frame_size_loc > 1024 * 1024) {
        Serial.println(" [Video] Frame too large for tx_buffer, dropping!");
      } else {
        memcpy(tx_buffer, fb->buf, frame_size_loc);
        lastCam = millis();
        // Must be a frame COUNTER, not millis(). The PC computes delivery% as
        // (seq - last_seq - 1); a timestamp makes every gap look like ~50 lost
        // frames, which drove the adaptive logic to max compression forever.
        static uint32_t frame_counter = 0;
        current_seq_global = ++frame_counter;
        tx_frame_size = frame_size_loc;
        tx_ready = true;
      }
    }

    esp_camera_fb_return(fb); 
  }
}

void streamTXTask(void *pvParameters) {
  esp_task_wdt_add(NULL);
  
  while (true) {
    esp_task_wdt_reset();

    if (!video_udp_mode) {
      tx_ready = false;
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (!tx_ready) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    if (WiFi.status() != WL_CONNECTED) {
      tx_ready = false; 
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    size_t frame_size = tx_frame_size;
    uint32_t current_seq = current_seq_global;
    unsigned long tx_start = millis();

    size_t chunk_size = VIDEO_CHUNK_SIZE;
    int total_chunks = (frame_size + chunk_size - 1) / chunk_size;

    static int debug_counter = 0;
    if (debug_counter++ % 30 == 0) {
      Serial.printf(" [Video] Sending frame %u (%d chunks) to %s\n",
                    current_seq, total_chunks, serverIP.toString().c_str());
    }

    uint8_t header[8];
    memcpy(header, &current_seq, 4);
    uint16_t tc = (uint16_t)total_chunks;
    memcpy(header + 4, &tc, 2);

    for (int i = 0; i < total_chunks; i++) {
        uint16_t chunk_id = (uint16_t)i;
        memcpy(header + 6, &chunk_id, 2);

        size_t offset = i * chunk_size;
        size_t current_chunk_len = (frame_size - offset > chunk_size)
                                       ? chunk_size
                                       : (frame_size - offset);

        if (videoUDP.beginPacket(serverIP, videoPort)) {
          videoUDP.write(header, 8);
          videoUDP.write(tx_buffer + offset, current_chunk_len);
          videoUDP.endPacket();
        }
        // A 2ms sleep per chunk was the hard framerate ceiling: a 40KB VGA
        // frame is ~29 chunks, so this alone cost ~58ms/frame. Yield
        // periodically instead -- enough to keep the WiFi stack fed without
        // paying the tick cost on every packet.
        if ((i % CHUNK_YIELD_EVERY) == (CHUNK_YIELD_EVERY - 1)) {
          vTaskDelay(1);
          esp_task_wdt_reset();
        } else {
          delayMicroseconds(CHUNK_GAP_US);
        }
    }
    esp_task_wdt_reset();

    tx_ready = false; 

    unsigned long actual_send_ms = millis() - tx_start;
    // int auto_interval = actual_send_ms + 50;
    // if (auto_interval < 50)
    //   auto_interval = 50;
    // video_frame_interval = auto_interval;

    // static int interval_log = 0;
    // if (interval_log++ % 30 == 0) {
    //   Serial.printf(
    //       " [Video] %d chunks, sent in %lums, next interval=%dms (~%d FPS)\n",
    //       total_chunks, actual_send_ms, auto_interval, 1000 / auto_interval);
    // }
  }
}

// ==========================================
// TASK: CONTROL & DISCOVERY
// ==========================================
void loopcommand(void *pvParameters) {
  esp_task_wdt_add(NULL); 

  // Standard IPv4 Binding for local network
  videoUDP.begin(videoPort);
  udpControl.begin(controlPort);
  udpDiscover.begin(discoveryPort);

  pcConnected = false;
  lastServerContact = millis();

  while (true) {
    esp_task_wdt_reset(); 
    
    // 1. ALWAYS LISTEN FOR DISCOVERY
    int discSize = udpDiscover.parsePacket();
    if (discSize) {
      IPAddress newIP = udpDiscover.remoteIP();
      serverIP = newIP;
      pcConnected = true;
      controlReset = true; 
      lastServerContact = millis();

      static IPAddress lastKnownIP;
      if (newIP != lastKnownIP) {
        Serial.print("PC CONNECTED/CHANGED: ");
        Serial.println(serverIP);
        for (int i = 0; i < 3; i++) {
          ledcWrite(LED_PIN, 50);
          delay(50);
          ledcWrite(LED_PIN, 0);
          delay(50);
        }
        lastKnownIP = newIP;
        serverIP = newIP;
        pcConnected = true;
        lastServerContact = millis();
        lastPacketTime = millis();
      }

      char discBuf[128];
      int discLen = udpDiscover.read(discBuf, sizeof(discBuf) - 1);
      if (discLen > 0) {
        discBuf[discLen] = '\0';
        // Only the UDP path can lose part of a frame, so only the UDP path
        // has a delivery figure worth reacting to. Over TCP the PC always
        // reports ~100%, and letting that run would walk the quality down to
        // 10 and hand the radio bigger frames than it can carry.
        char *deliveryStr = strstr(discBuf, "delivery:");
        if (deliveryStr && video_udp_mode) {
          int delivery = atoi(deliveryStr + 9);
          static int low_delivery_count = 0;
          static int high_delivery_count = 0;

          if (delivery < 50) {
            low_delivery_count++;
            high_delivery_count = 0;
            if (low_delivery_count >= 3 && current_quality < QUALITY_WORST) {
              current_quality += 3;
              if (current_quality > QUALITY_WORST) current_quality = QUALITY_WORST;
              updateCameraSettings();
              Serial.printf(" [Adaptive] Delivery %d%% LOW -> quality=%d (more compressed)\n",
                            delivery, current_quality);
              low_delivery_count = 0;
            }
          } else if (delivery > 80) {
            high_delivery_count++;
            low_delivery_count = 0;
            if (high_delivery_count >= 5 && current_quality > QUALITY_BEST) {
              current_quality -= 2;
              if (current_quality < QUALITY_BEST) current_quality = QUALITY_BEST;
              updateCameraSettings();
              Serial.printf(" [Adaptive] Delivery %d%% HIGH -> quality=%d (better image)\n",
                            delivery, current_quality);
              high_delivery_count = 0;
            }
          } else {
            low_delivery_count = 0;
            high_delivery_count = 0;
          }
        }
      } else {
        while (udpDiscover.available())
          udpDiscover.read();
      }
    }

    // 2. LISTEN FOR COMMANDS
    int packetSize = udpControl.parsePacket();
    if (packetSize) {
      lastPacketTime = millis();
      lastServerContact = millis();
      char packetBuffer[255];
      int len = udpControl.read(packetBuffer, 255);
      if (len > 0) {
        packetBuffer[len] = 0;
        String line = String(packetBuffer);

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
            if (cmd == 0) {
              previous1A = v1;
              previous1B = v2;
            } else if (cmd == 1) {
              oldlight = v1;
            } else if (cmd == 3 && v1 == 3 && v2 == 3) {
              Serial.println("COMMAND: RESET ENGINE");
              ESP.restart();
            } else if (cmd == 4 && v1 == 4 && v2 == 4) {
              Serial.println("COMMAND: Q - FAR  : QVGA 320x240 q14");
              current_framesize = FRAMESIZE_QVGA;
              current_quality = 14;
              updateCameraSettings();
            } else if (cmd == 5 && v1 == 5 && v2 == 5) {
              Serial.println("COMMAND: W - NORM : VGA 640x480 q12");
              current_framesize = FRAMESIZE_VGA;
              current_quality = 12;
              updateCameraSettings();
            } else if (cmd == 6 && v1 == 6 && v2 == 6) {
              Serial.println("COMMAND: E - HIGH : SVGA 800x600 q11");
              current_framesize = FRAMESIZE_SVGA;
              current_quality = 11;
              updateCameraSettings();
            } else if (cmd == 8 && v1 == 8 && v2 == 8) {
              Serial.println("COMMAND: VIDEO -> MJPEG over HTTP (port 81)");
              video_udp_mode = false;
              tx_ready = false;
            } else if (cmd == 9 && v1 == 9 && v2 == 9) {
              Serial.println("COMMAND: VIDEO -> UDP chunks (port 44442)");
              video_udp_mode = true;
            } else if (cmd == 7 && v1 == 7 && v2 == 7) {
              Serial.println("COMMAND: R - MAX  : XGA 1024x768 q14 (low fps)");
              current_framesize = FRAMESIZE_XGA;
              current_quality = 14;
              updateCameraSettings();
            }
          }
        }
      }
    }

    // 3. FAILSAFE
    if (millis() - lastPacketTime > AUTOSTOP_INTERVAL) {
      previous1A = 0;
      previous1B = 0;
    }

    if (pcConnected && millis() - lastServerContact > LOST_GUI_TIMEOUT) {
      Serial.println("Warning: Lost GUI contact.");
      pcConnected = false;
    }

    if (millis() - lastServerContact > BEACON_TIMEOUT) {
      Serial.println("CRITICAL: No beacon. Rebooting...");
      ESP.restart();
    }

    // 4. TELEMETRY
    static unsigned long lastTele = 0;
    if (pcConnected && millis() - lastTele > TELEMETRY_INTERVAL) {
      lastTele = millis();
      if (WiFi.status() == WL_CONNECTED) {
        uint32_t chipId = (uint32_t)ESP.getEfuseMac();
        char chipStr[10];
        sprintf(chipStr, "%08X", chipId);
        udpDiscover.beginPacket(serverIP, discoveryPort);
        udpDiscover.printf("%d,%s,RSSI:%d,SSID:%s,NAME:%s", telemetry_seq,
                           chipStr, WiFi.RSSI(), WiFi.SSID().c_str(), hostName);
        udpDiscover.endPacket();

        static int tele_count = 0;
        if (tele_count++ % 5 == 0) {
          Serial.printf(" [Telemetry] Sent to %s on port %d\n",
                        serverIP.toString().c_str(), discoveryPort);
        }

        telemetry_seq = (telemetry_seq + 1) % SEQ_LIMIT;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

// ==========================================
// SETUP & LOOP
// ==========================================
void setup() {
  setCpuFrequencyMhz(240); 
  Serial.begin(115200);

  esp_log_level_set("cam_hal", ESP_LOG_NONE);

  Serial.println("\n\n--- ESP32 CAM ROBOT STARTED ---");

  ledcAttach(LED_PIN, PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR_1A, PWM_FREQ, PWM_RES);
  ledcAttach(MOTOR_1B, PWM_FREQ, PWM_RES);

  ledcWrite(LED_PIN, 100);

  Serial.println("Init Camera Driver...");
  current_framesize = FRAMESIZE_VGA;
  current_quality = 12;
  initCamera();
  vTaskDelay(pdMS_TO_TICKS(500));
  Serial.println("Camera Ready.");

  bool connected = false;
  for (int i = 0; i < WIFICOUNT; i++) {
    Serial.printf("\nAttempting WiFi %d: %s", i, ssids[i]);
    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_STA);
    // Station power-save parks the radio between beacons and adds tens to
    // >100ms of jitter. For a live control link it must be off.
    WiFi.setSleep(false);
    WiFi.setTxPower(WIFI_POWER_19_5dBm); // max TX: more range, fewer retries
    WiFi.begin(ssids[i], passs[i]);
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 20) {
      delay(500);
      Serial.print(".");
      retry++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      connected = true;
      Serial.println("\nCONNECTED!");
      break;
    }
  }

  if (!connected) {
    Serial.println("\nAll WiFi failed. Restarting...");
    delay(2000);
    ESP.restart();
  }

  Serial.print("Local IP Address: ");
  Serial.println(WiFi.localIP());

  startCameraServer();
  Serial.printf(" [Video] transport at boot: %s\n",
                video_udp_mode ? "UDP chunks" : "MJPEG over HTTP");

  esp_task_wdt_config_t twdt_config = {
      .timeout_ms = WATCHDOG_TIMEOUT_S * 1000,
      .idle_core_mask = 0,
      .trigger_panic = true,
  };

  if (esp_task_wdt_init(&twdt_config) != ESP_OK) {
    esp_task_wdt_reconfigure(&twdt_config);
  }

  tx_buffer = (uint8_t *)heap_caps_malloc(1024 * 1024, MALLOC_CAP_SPIRAM);
  if (!tx_buffer) {
    Serial.println("CRITICAL: Failed to allocate TX double buffer in PSRAM!");
    delay(1000);
    ESP.restart();
  }

  xTaskCreatePinnedToCore(loopcommand, "Ctl", 8192, NULL, 3, NULL, 1);
  xTaskCreatePinnedToCore(cameraDrainTask, "Cam", 8192, NULL, 3, NULL, 0);
  xTaskCreatePinnedToCore(streamTXTask, "VTx", 8192, NULL, 2, NULL, 0);

  esp_task_wdt_add(NULL); 

  ledcWrite(LED_PIN, 0);
}

void loop() {
  if (!pcConnected && !light_override) {
    static bool state = 0;
    ledcWrite(LED_PIN, state ? 50 : 0);
    state = !state;
    vTaskDelay(pdMS_TO_TICKS(200));
  } else {
    ledcWrite(LED_PIN, oldlight);
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  ledcWrite(MOTOR_1A, (previous1A * MAX_DUTY));
  ledcWrite(MOTOR_1B, (previous1B * MAX_DUTY));
  esp_task_wdt_reset(); 
}