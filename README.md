# Physical AI — Vision-Guided Autonomous Robot on ESP32-CAM

An ESP32-CAM streams video to a host PC. The PC finds the blue cones marking the track,
works out where the middle of the lane is, and steers the car back onto it — closing the
loop over Wi-Fi.

```
video → cone_detection → lane_center → autopilot → robot_link → motors
```

Every stage is a separate file that runs standalone, so each can be tuned on its own.
**Exactly one script moves the physical car: `autodrive.py`.**

> Thai documentation: [README.th.md](README.th.md) — longer, with tuning notes and
> troubleshooting.

---

## Design notes

**The pipeline is split by decision, not by convenience.** `cone_detection.py` only answers
"where are the cones", `lane_center.py` only answers "where should we be", `autopilot.py` is
the only file that decides how the car moves, and `robot_link.py` is the only one that talks
to the motors. Tuning the detector cannot accidentally change the steering law.

**Nothing moves until you arm it.** `autodrive.py` starts disarmed and stays that way until
SPACE is pressed, so launching it can never send the car across the room. `X` is an emergency
stop, `Q` stops the car on the way out, and `--dry-run` runs the full decision chain while
sending nothing.

**Tuned values live in `cone_config.json`, not in the code** — HSV bounds, contour area and
aspect limits, ROI, steering gain, base throttle, turn slowdown, throttle ramp, and how many
lost frames trigger a stop. Press `W` while running to write the current values back.

**White balance and exposure are locked before driving.** Auto-WB re-tints the frame as the
car turns, which moves blue out of the HSV window mid-corner. `camera_setup.prepare_camera()`
locks both, and both `run.py` and `autodrive.py` call it — an earlier version that skipped it
booted at QVGA 320×240, and since `min_contour_area` is in absolute pixels, a detector tuned
at VGA stopped seeing cones at a quarter of the area.

**Link failures are diagnosed, not guessed at.** `link_monitor.py` watches an MJPEG stream for
tens of minutes and attributes degradation to a specific layer — sensor, radio, heap
exhaustion, board reboot, or the host — which turns "the robot stopped working" into something
actionable.

---

## Files

| File | Role |
|---|---|
| `run.py` | **Start here.** Finds the camera on the LAN, sets resolution, opens the stream, shows live video. `--detect` adds cone detection and lane centre without driving. |
| `cone_detection.py` | Finds blue cones by HSV threshold plus contour shape filtering (`ConeDetector.detect()`). Runs standalone for tuning. |
| `lane_center.py` | Pairs left and right cones and computes the lane centre (`LaneFinder.update()`), with memory across frames when a cone drops out. |
| `autopilot.py` | P-controller: lateral deviation → steering and throttle. The only file that decides motion. |
| `robot_link.py` | Sends UDP commands to the board. |
| `autodrive.py` | **Moves the car.** Ties the whole chain together. |
| `camera_setup.py` | Scan → set resolution → lock AWB → flip. Shared by `run.py` and `autodrive.py`. |
| `camera_control.py` | Resolution / quality / flip and the AWB and exposure lock, over `/control`. |
| `mjpeg_video_source.py` | Reads `http://<ip>:81/stream`, on its own thread, keeping only the newest frame and reconnecting itself. |
| `esp32_video_source.py` | Reads the UDP stream from the robot firmware. |
| `video_source.py` | Picks a source from `--source` (`webserver` / `esp32` / an `http://` URL). |
| `link_monitor.py` | Long-run link diagnostics — says which layer is degrading. |
| `esp32_link_check.py` | Checks the UDP link to the robot firmware and reports where it breaks. |
| `capture_esp32_samples.py` | Saves frames as `.jpg` for offline tuning. |
| `cone_config.json` | Tuned HSV, filter, lane and control values. |
| `CameraWebServer/` | Firmware in use — MJPEG over HTTP. |
| `firmware/esp32cam_robot/` | Robot firmware — the only one that drives motors. |
| `firmware/esp32cam_webserver_robot/` | Combined web server plus motor control. |

---

## Running it

```bash
pip install -r requirements.txt     # opencv-python and numpy; everything else is stdlib
```

```bash
python run.py                       # find the camera and show the stream
python run.py --detect              # add cone detection and lane centre, no driving
```

```bash
python autodrive.py --esp-ip <camera-ip> --dry-run    # full decision chain, sends nothing
python autodrive.py --esp-ip <camera-ip>              # drives — SPACE to arm, X to stop
```

Keys: `SPACE` arm/disarm · `X` emergency stop · `W` write tuned config · `Q` quit ·
`v`/`h` flip · `s` save frame

### Firmware

Open the sketch in `CameraWebServer/` (or `firmware/esp32cam_robot/` for the driving board)
in the Arduino IDE, then before flashing:

```bash
cp CameraWebServer/wifi_secrets.example.h CameraWebServer/wifi_secrets.h
# edit wifi_secrets.h with your SSID and password
```

`wifi_secrets.h` is gitignored. Each sketch reads it behind a `__has_include` guard, so a
missing file still compiles — it just joins a placeholder network.

---

## Status

**Phase 1 — complete.** Cone detection, lane centring, and closed-loop driving all work on a
tuned course.

**Phase 2 — in progress.** Hand-tuned HSV thresholds are brittle: they hold on the course they
were tuned on and break when lighting or surface changes. The robot is being run on real
courses to collect an on-board image dataset, to train a learned detector that generalises
across conditions instead of being re-tuned for each one.

## Tech

Python 3.13 · OpenCV · NumPy · C++ (Arduino, ESP32-CAM) · MJPEG over HTTP · UDP
