"""
Autonomous driving: cones -> lane center -> motors
====================================================
Ties the whole chain together and actually moves the car:

    video -> cone_detection -> lane_center -> autopilot -> robot_link

    python autodrive.py --esp-ip 192.168.8.191
    python autodrive.py --esp-ip 192.168.8.191 --dry-run   # decide, don't send
    python autodrive.py --esp-ip 192.168.8.191 --flip both --framesize svga

THIS ONE MOVES A PHYSICAL ROBOT. It starts DISARMED and stays that way
until you press SPACE, so launching it can never make the car take off
across the room. Read the safety notes before the first run.

    SPACE : arm / disarm
    X     : emergency stop (disarms and sends neutral)
    Q     : quit (stops the car on the way out)
    W     : write the tuned config

SAME CAMERA SETUP AS run.py
--------------------------
Resolution and the AWB/exposure lock go through camera_setup.prepare_camera(),
the same call run.py makes. Skipping it is what made an earlier version of
this script look blurry next to run.py: CameraWebServer boots at QVGA
320x240 and stays there unless something asks for more, and min_contour_area
is in absolute pixels, so a detector tuned at VGA stops seeing cones at a
quarter of the area. On --source esp32 the resolution goes out as a UDP
command instead (RobotLink.set_framesize), because that firmware has no
HTTP control endpoint — and for the same reason its AWB cannot be locked
from here at all.

REQUIRES THE ROBOT FIRMWARE
---------------------------
`firmware/esp32cam_robot/` — the CameraWebServer sketch has no motor
control, so --source webserver can see cones but will never drive. This
script defaults to --source esp32 for that reason.

THE SAFETY LAYERS, AND WHY EACH ONE EXISTS
------------------------------------------
1. Starts disarmed. Nothing moves until a deliberate keypress.
2. Emergency stop on X, on Q, on the window closing, and in the `finally`
   of the main loop — including when an exception unwinds it. A crash in
   the vision code must not leave the throttle where it was.
3. Stale-frame cutout. If frames stop arriving for `--stale-timeout`
   seconds the car is stopped and disarmed: no video means the autopilot
   is steering by a frozen picture, which is worse than not steering.
4. Lane-lost stop, inside autopilot.
5. The firmware's own 150 ms failsafe underneath all of it — if this
   process dies outright, the motors zero on their own.

Layer 5 is the only one that survives the PC losing power, so treat the
others as convenience and keep a hand near the robot's power switch on
the first runs anyway.
"""

import argparse
import sys
import time

import cv2

from autopilot import Autopilot, REASON_DISARMED
from camera_control import RESOLUTIONS
from camera_setup import (FIRST_FRAME_TIMEOUT, FLIP_MODES, apply_flip,
                          flip_frame, prepare_camera)
from cone_detection import ConeDetector, build_trackbars, print_saved, read_trackbars, save_config
from lane_center import LaneFinder, draw_lane
from camera_control import is_robot_firmware
from robot_link import RobotLink
from video_source import make_stream_source

WINDOW = "Autodrive"
CONTROL_WIN = "Tuning (w=write config)"


def draw_hud(frame, armed, steer, throttle, reason, dry_run, fps):
    h, w = frame.shape[:2]
    bar_h = 54
    # Solid band rather than text straight on the video: the state has to
    # be readable against a moving background at a glance.
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)

    if dry_run:
        state, color = "DRY RUN", (0, 200, 255)
    elif armed:
        state, color = "ARMED", (0, 0, 255)
    else:
        state, color = "DISARMED", (0, 200, 0)

    cv2.putText(frame, state, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"steer {steer:+.2f}  throttle {throttle:.2f}  {reason}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:4.1f} fps   SPACE=arm  X=stop  Q=quit",
                (w - 330, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # A red border while armed — impossible to miss from across a room.
    if armed and not dry_run:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Drive the robot by following blue cones")
    parser.add_argument("--esp-ip", default=None,
                        help="Robot IP. Optional for --source esp32 (it broadcast-discovers), "
                             "but REQUIRED to send control packets.")
    parser.add_argument("--source", default="webserver",
                        help="How video arrives: 'webserver' = MJPEG over HTTP (default, what "
                             "the robot firmware now streams), 'esp32' = the legacy UDP chunk "
                             "path. Either can drive; control always goes out over UDP.")
    parser.add_argument("--config", default="cone_config.json", help="Shared detector/lane/autopilot config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run the full loop and show what it would command, but send nothing")
    parser.add_argument("--framesize", default="vga", choices=sorted(RESOLUTIONS),
                        help="Resolution to put the camera in (default vga, same as run.py). "
                             "Re-tune min_contour_area after changing it — it is in absolute pixels.")
    parser.add_argument("--flip", default="none", choices=sorted(FLIP_MODES),
                        help="Flip the image: 'v' upside down, 'h' mirrored, 'both', or 'none' (default)")
    parser.add_argument("--quality", type=int, default=None,
                        help="JPEG quality 10 (best/biggest) .. 63 (worst/smallest). "
                             "The main framerate knob: CameraWebServer boots at 10, and "
                             "16-18 roughly halves the bytes per frame. --source webserver only.")
    parser.add_argument("--ae-level", type=int, default=None, choices=[-2, -1, 0, 1, 2],
                        help="Exposure bias. -1 keeps the shutter short, which cuts motion "
                             "blur and stops the sensor's own framerate collapsing in a dim "
                             "room. Applied before the AWB/exposure lock.")
    parser.add_argument("--no-lock", action="store_true",
                        help="Leave the camera's auto white balance / exposure running (locked by default)")
    parser.add_argument("--stale-timeout", type=float, default=1.0,
                        help="Stop and disarm if no new frame arrives for this many seconds")
    args = parser.parse_args()

    if not args.dry_run and not args.esp_ip:
        print("Refusing to arm without --esp-ip: control packets need a destination.")
        print("Pass --esp-ip <robot ip>, or --dry-run to test the vision loop with no robot.")
        return 1

    # The transport no longer identifies the sketch. firmware/esp32cam_robot/
    # serves MJPEG on port 81 exactly like CameraWebServer does, and MJPEG is
    # now its default video path -- so "--source webserver" says nothing about
    # whether there are motors on the other end. Ask the board instead: the
    # robot firmware reports "robot":1 in /status, the stock sketch does not.
    # Refusing on the transport alone would block the normal way to drive;
    # trusting it would let a stock CameraWebServer board be armed.
    if args.source.lower() != "esp32" and not args.dry_run:
        if not is_robot_firmware(args.esp_ip):
            print(f"{args.esp_ip} is not running the robot firmware "
                  f"(no \"robot\" marker in /status), so it has no motors.")
            print("Flash firmware/esp32cam_robot/, or add --dry-run to test vision only.")
            return 1
        print(f"Robot firmware confirmed at {args.esp_ip}; video over MJPEG, control over UDP.")

    detector = ConeDetector(config_path=args.config)
    lane = LaneFinder(detector.cfg)          # one shared config dict
    autopilot = Autopilot(detector.cfg)
    build_trackbars(CONTROL_WIN, detector.cfg)

    # Camera setup BEFORE the stream opens: the ESP32-CAM serves one client,
    # and the /capture that verifies the resolution change can't run while
    # the MJPEG stream holds the slot.
    esp_ip = prepare_camera(args.source, args.esp_ip, args.framesize,
                            lock=not args.no_lock, quality=args.quality,
                            ae_level=args.ae_level)
    if esp_ip is None and args.source.lower() == "webserver":
        return 1

    link = None
    if not args.dry_run:
        link = RobotLink(args.esp_ip)
        link.start()
        # The robot firmware takes its resolution as a UDP command — there
        # is no HTTP endpoint for prepare_camera() to use. Sent before the
        # stream starts so the sensor is already reconfigured by the time
        # frames matter; it also re-applies the .ino's sensor settings.
        if args.source.lower() == "esp32":
            link.set_framesize(args.framesize)
            print(f"Asked the robot firmware for {args.framesize.upper()}.")
    elif args.source.lower() == "esp32":
        print(f"--dry-run: no control link, so the camera stays at whatever the "
              f"firmware booted with (VGA q12) rather than {args.framesize.upper()}.")

    vflip, hmirror, flip_in_sensor = apply_flip(args.source, esp_ip, args.flip)

    src = make_stream_source(args.source, esp_ip=esp_ip)
    if src is None:
        print(f"Not a live video source: {args.source}")
        return 1
    src.start()

    armed = False
    steer = throttle = 0.0
    reason = REASON_DISARMED
    fps = 0.0
    last_tick = time.time()
    last_frame_at = time.time()
    seen_first_frame = False
    started_at = time.time()
    warned_no_video = False

    print("Waiting for video ...")
    print("SPACE arms the car. X = emergency stop, Q = quit, W = write config.")

    last_count = -1

    def handle_key(key):
        """Every key, in ONE place. Returns True to quit.

        Called on both loop paths deliberately. The loop only does real
        work on a new frame and idles on cv2.waitKey(5) otherwise, so at
        8 fps the idle path runs ~25x more often than the fresh-frame one.
        waitKey consumes the keystroke wherever it is called, so a handler
        that only exists on the fresh-frame path drops ~96% of what you
        press — which on this script meant SPACE often not arming and X
        not always e-stopping on the first try. Safety keys must not
        depend on a frame having just arrived."""
        nonlocal armed, steer, throttle, reason

        if key == 255:                      # nothing pressed, the common case
            return False
        if key == ord('q'):
            return True
        if key == ord(' '):
            armed = not armed
            autopilot.reset()
            if armed:
                print("[ARMED] car is live")
            else:
                if link:
                    link.stop()
                print("[DISARMED] stopped")
        if key == ord('x'):
            armed = False
            autopilot.reset()
            steer = throttle = 0.0
            if link:
                link.stop()
            print("[E-STOP] stopped and disarmed")
        if key == ord('w'):
            save_config(detector.cfg, args.config)
            print_saved(detector.cfg, args.config)
        return False

    try:
        while True:
            now = time.time()

            # Only do work when a NEW frame has actually arrived. get_frame()
            # keeps handing back the last picture forever, so processing on
            # every pass means running detection ~470x/s on ~20 real frames
            # — a saturated core that starves the receive thread, which on
            # the lossy UDP path shows up as a stuttering picture.
            count = src.frame_count
            fresh = count != last_count

            if not fresh:
                # Still drive: this is deduplicated internally and only puts
                # a packet on the wire every 50 ms, which is what keeps the
                # firmware's 150 ms motor failsafe from tripping when the
                # video is slower than that.
                if armed and link:
                    link.drive(steer, throttle)

                if armed and seen_first_frame and (now - last_frame_at) > args.stale_timeout:
                    armed = False
                    steer = throttle = 0.0
                    reason = "no video - disarmed"
                    if link:
                        link.stop()
                    print(f"[safety] No new frames for {args.stale_timeout:.1f}s — stopped and disarmed.")

                if (not warned_no_video and not seen_first_frame
                        and now - started_at > FIRST_FRAME_TIMEOUT):
                    print(f"Still no frames after {FIRST_FRAME_TIMEOUT:.0f}s. The ESP32-CAM "
                          "serves ONE stream client — close any browser tab showing the "
                          "camera page, then restart this.")
                    warned_no_video = True

                # 5 ms, not 1: gives the GIL back to the receive thread
                # instead of spinning on it.
                if handle_key(cv2.waitKey(5) & 0xFF):
                    break
                continue

            last_count = count
            frame = src.get_frame()
            if frame is None:
                continue

            # Only when the sensor didn't do it for us — flipping twice
            # undoes the work. Before detection, so the ROI (which blanks
            # the TOP of the mask) covers the right half of the picture.
            if not flip_in_sensor:
                frame = flip_frame(frame, vflip, hmirror)

            # dt between real frames, so the throttle ramp is measured in
            # wall-clock seconds rather than in loop iterations.
            dt = now - last_tick
            last_tick = now
            last_frame_at = now
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)
            if not seen_first_frame:
                seen_first_frame = True
                print(f"Video up: {frame.shape[1]}x{frame.shape[0]}. Press SPACE when the track is clear.")

            read_trackbars(CONTROL_WIN, detector.cfg)
            cones, mask, display = detector.detect(frame)
            lane_result = lane.update(cones, frame.shape[1], frame.shape[0])
            draw_lane(display, lane_result)

            if armed:
                steer, throttle, reason = autopilot.update(lane_result, dt)
                if link:
                    link.drive(steer, throttle)
            else:
                # Keep showing what it WOULD do while disarmed — that is how
                # you check the gains before trusting it with the motors.
                steer, throttle, reason = autopilot.update(lane_result, dt)
                reason = f"{reason} (disarmed)"
                autopilot.reset()

            draw_hud(display, armed, steer, throttle, reason, args.dry_run, fps)
            cv2.imshow(WINDOW, display)
            cv2.imshow("Mask", mask)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        # Runs on a normal quit AND on an exception on the way out — the
        # car must not be left with a throttle command standing.
        if link:
            link.stop()
            time.sleep(0.15)   # let the stop burst actually leave the socket
            link.close()
        src.stop()
        cv2.destroyAllWindows()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
