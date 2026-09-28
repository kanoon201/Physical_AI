"""
One-command runner
===================
Everything in a single process: finds the ESP32-CAM on the network, puts
it in the resolution we want, opens its stream, and shows the live video.

    python run.py                          # auto-find the camera
    python run.py --esp-ip 192.168.8.191   # skip the scan if you know the IP
    python run.py --source esp32           # robot firmware (UDP) instead
    python run.py --flip both              # camera mounted upside down
    python run.py --detect                 # blue cone detection + lane center

Controls:
    'v' : flip vertically (toggle)
    'h' : mirror horizontally (toggle)
    's' : save the current frame as a .jpg
    'w' : (--detect only) write the tuned values to cone_config.json
    'q' : quit

Without --detect this is video only. With it, cones are found, the middle
of the gap between them is worked out, and both are drawn — but nothing is
driven. See cone_detection.py and lane_center.py.
"""

import argparse
import os
import sys
import time

import cv2

from camera_control import RESOLUTIONS
from camera_setup import (FIRST_FRAME_TIMEOUT, FLIP_MODES, apply_flip,
                          flip_frame, prepare_camera, push_flip_to_sensor,
                          set_robot_framesize)
from video_source import make_stream_source

SNAPSHOT_DIR = "esp32_samples"


# ============================================================
# LINK BENCHMARK
# ============================================================

def run_bench(src, seconds):
    """Framerate with nothing else in the way.

    Worth having because "the video is slow" has two completely different
    causes and they need opposite fixes. If this number is high and the
    normal window is slow, the loop is the problem. If this number is the
    same low number, the camera or the WiFi is the problem and no amount
    of tuning on this end will help — drop --quality or --framesize.

    For reference, the per-frame CPU cost of the whole detection pipeline
    at VGA is around 6 ms, and this receiver alone does four figures of
    fps against a local server. Anything in single digits here is the
    link, not Python."""
    print(f"Benchmarking for {seconds:g}s — no window, no detection ...")
    # Bounded: the robot reboots itself after 120s with no beacon and comes
    # back on a different DHCP lease, so "no frames yet" is a state this can
    # legitimately sit in forever. Say so instead of hanging.
    waited_until = time.time() + FIRST_FRAME_TIMEOUT
    while src.frame_count == 0:
        if time.time() > waited_until:
            src.stop()
            print(f"No frames at all after {FIRST_FRAME_TIMEOUT:.0f}s.")
            print("Run esp32_link_check.py — the robot may have rebooted onto a "
                  "different IP, which it does after 120s with no PC talking to it.")
            return 1
        time.sleep(0.05)

    t0 = time.time()
    f0, b0 = src.frame_count, src.bytes_received
    time.sleep(seconds)
    elapsed = time.time() - t0
    frames = src.frame_count - f0
    byte_delta = src.bytes_received - b0

    src.stop()
    if frames == 0:
        print("No frames arrived at all.")
        return 1

    fps = frames / elapsed
    kbps = byte_delta / elapsed / 1024.0
    print(f"  {frames} frames in {elapsed:.1f}s  ->  {fps:.1f} fps")
    print(f"  {kbps:.0f} KB/s ({kbps * 8 / 1024:.1f} Mbit/s), "
          f"average frame {byte_delta / frames / 1024:.1f} KB")
    if fps < 10:
        print("  Link-limited. Raise --quality (16-18) or drop --framesize, "
              "and re-tune min_contour_area afterwards.")
    return 0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Find the ESP32-CAM, open its stream, show the live video, and optionally detect blue cones")
    parser.add_argument("--esp-ip", default=None, help="Camera IP. Omit to scan the local subnet for it.")
    parser.add_argument("--source", default="webserver",
                        help="Video transport: 'webserver' = MJPEG over HTTP (both sketches "
                             "speak it), 'esp32' = the legacy UDP chunk path (robot firmware only)")
    parser.add_argument("--framesize", default="vga", choices=sorted(RESOLUTIONS),
                        help="Resolution to put the camera in (default vga)")
    parser.add_argument("--flip", default="none", choices=sorted(FLIP_MODES),
                        help="Flip the image: 'v' upside down, 'h' mirrored, 'both', or 'none' (default)")
    parser.add_argument("--detect", action="store_true",
                        help="Run blue cone detection on the feed, with live tuning trackbars")
    parser.add_argument("--config", default="cone_config.json", help="Detector config JSON (with --detect)")
    parser.add_argument("--quality", type=int, default=None,
                        help="JPEG quality 10 (best/biggest) .. 63 (worst/smallest). "
                             "The main framerate knob: CameraWebServer boots at 10, and "
                             "16-18 roughly halves the bytes per frame. --source webserver only.")
    parser.add_argument("--ae-level", type=int, default=None, choices=[-2, -1, 0, 1, 2],
                        help="Exposure bias. -1 keeps the shutter short, which cuts motion "
                             "blur and stops the sensor's own framerate collapsing in a dim "
                             "room. Applied before the AWB/exposure lock.")
    parser.add_argument("--no-lock", action="store_true",
                        help="Leave the camera's auto white balance / exposure running (they are locked by default)")
    parser.add_argument("--out", default=SNAPSHOT_DIR, help="Folder that 's' saves frames into")
    parser.add_argument("--bench", type=float, default=None, metavar="SECONDS",
                        help="Measure the link instead of showing it: no window, no detection, "
                             "just fps / KB per second / average frame size for this long. "
                             "Use it to tell a slow camera apart from a slow PC.")
    args = parser.parse_args()

    # Resolution + AWB/exposure lock, before the stream opens. The UDP
    # firmware can broadcast-discover itself and has no HTTP control, so
    # prepare_camera() is a no-op there and just hands esp_ip back.
    esp_ip = prepare_camera(args.source, args.esp_ip, args.framesize,
                            lock=not args.no_lock, quality=args.quality,
                            ae_level=args.ae_level)
    if esp_ip is None and args.source.lower() == "webserver":
        return 1

    # The robot firmware has no HTTP endpoint, so its resolution is a UDP
    # command instead. Without this, --framesize silently did nothing on
    # --source esp32 and the board stayed at whatever it booted with.
    if args.source.lower() == "esp32":
        set_robot_framesize(esp_ip, args.framesize)

    vflip, hmirror, flip_in_sensor = apply_flip(args.source, esp_ip, args.flip)

    src = make_stream_source(args.source, esp_ip=esp_ip)
    if src is None:
        print(f"--source must be 'esp32', 'webserver', or an http:// stream URL (got: {args.source})")
        return 1
    src.start()

    if args.bench is not None:
        return run_bench(src, args.bench)

    detector = None
    control_win = "Cone tuning (w=write config)"
    if args.detect:
        from cone_detection import ConeDetector, build_trackbars, print_saved, read_trackbars, save_config
        from lane_center import LaneFinder, draw_lane
        detector = ConeDetector(config_path=args.config)
        # Shares the detector's dict, so the trackbars and the saved JSON
        # cover the lane settings too.
        lane = LaneFinder(detector.cfg)
        build_trackbars(control_win, detector.cfg)

    os.makedirs(args.out, exist_ok=True)
    saved_count = len([f for f in os.listdir(args.out) if f.lower().endswith(".jpg")])

    print("Waiting for the first frame ...")
    started_at = time.time()
    warned = False
    got_first_frame = False
    fps = 0.0
    last_tick = time.time()
    last_frame = None

    last_count = -1

    def handle_key(key):
        """Every key, in ONE place. Returns True to quit.

        This is called on both loop paths on purpose. The loop only does
        real work when a new frame arrives and idles on cv2.waitKey(5) the
        rest of the time, so at 8 fps the idle path runs ~25x more often
        than the fresh-frame path. A handler that lives only on the
        fresh-frame path therefore swallows ~96% of what you press —
        waitKey consumes the keystroke either way, and the idle branch used
        to throw away everything that wasn't 'q'. That is what made 'v'
        and 'h' look like they did nothing."""
        nonlocal vflip, hmirror, flip_in_sensor, saved_count

        if key == 255:                    # nothing pressed, the common case
            return False
        if key == ord('q'):
            return True
        if key in (ord('v'), ord('h')):
            if key == ord('v'):
                vflip = not vflip
            else:
                hmirror = not hmirror
            flip_in_sensor = push_flip_to_sensor(args.source, esp_ip, vflip, hmirror)
            where = "sensor" if flip_in_sensor else "software"
            print(f"Flip: vflip={int(vflip)} hmirror={int(hmirror)} ({where})")
        if key == ord('w') and detector is not None:
            save_config(detector.cfg, args.config)
            print_saved(detector.cfg, args.config)
        if key == ord('s'):
            if last_frame is None:
                print("Nothing to save yet — no frame has arrived.")
            else:
                filename = os.path.join(args.out, f"esp32_sample_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(filename, last_frame)
                saved_count += 1
                print(f"Saved: {filename}")
        return False

    try:
        while True:
            # get_frame() hands back the last picture forever, so without
            # this check the loop re-runs detection and imshow hundreds of
            # times per second on frames it has already seen. That saturates
            # a core and starves the receive thread — which on the UDP path
            # drops chunks and makes the video visibly stutter.
            count = src.frame_count
            if count == last_count:
                if handle_key(cv2.waitKey(5) & 0xFF):   # 5ms yields to the receiver
                    break
                if not warned and not got_first_frame and time.time() - started_at > FIRST_FRAME_TIMEOUT:
                    print(f"Still no frames after {FIRST_FRAME_TIMEOUT:.0f}s. The ESP32-CAM only "
                          "serves ONE stream client — close any browser tab showing the camera page.")
                    warned = True
                continue
            last_count = count

            frame = src.get_frame()
            if frame is None:
                continue

            if not got_first_frame:
                got_first_frame = True
                print(f"Streaming {frame.shape[1]}x{frame.shape[0]}. "
                      f"'v'/'h' flip, 's' saves a frame into ./{args.out}/, 'q' quits.")
                if detector is not None:
                    print("Adjust the trackbars until Mask shows ONLY the cones white, then press 'w'.")

            if not flip_in_sensor:
                frame = flip_frame(frame, vflip, hmirror)
            # What 's' writes out: the frame as processed, flip included.
            last_frame = frame

            now = time.time()
            # Smoothed so the readout doesn't flicker on every frame.
            fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-6)
            last_tick = now

            if detector is not None:
                read_trackbars(control_win, detector.cfg)
                cones, mask, display = detector.detect(frame)
                lane_result = lane.update(cones, frame.shape[1], frame.shape[0])
                draw_lane(display, lane_result)
                cv2.imshow("Mask", mask)
            else:
                display = frame.copy()

            flip_txt = ("V" if vflip else "") + ("H" if hmirror else "") or "-"
            # The detector writes its own line at y=20, so sit below it.
            info_y = 40 if detector is not None else 20
            cv2.putText(display, f"{frame.shape[1]}x{frame.shape[0]}  {fps:4.1f} fps  "
                                 f"flip:{flip_txt}  saved:{saved_count}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("ESP32-CAM Live Feed", display)

            if handle_key(cv2.waitKey(1) & 0xFF):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
