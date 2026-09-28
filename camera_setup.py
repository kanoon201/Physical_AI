"""
Camera discovery and startup setup
====================================
Everything that has to happen BEFORE the video stream opens, in one place
so `run.py` and `autodrive.py` cannot drift apart on it.

This module exists because they did drift: run.py set the resolution and
locked the auto-exposure loops, autodrive.py did neither, and the result
was that the exact same receiver code produced a soft 320x240 picture
under autodrive and a sharp 640x480 one under run.py. CameraWebServer
boots at QVGA — CameraWebServer.ino drops it there right after init for
speed — so "just open the stream" gets you the small picture.

    esp_ip = prepare_camera("webserver", args.esp_ip, "vga", lock=True,
                            quality=16, ae_level=-1)
    src = make_stream_source("webserver", esp_ip=esp_ip)

ORDER MATTERS
-------------
Call prepare_camera() before src.start(). The ESP32-CAM serves ONE client,
and /capture (which set_resolution uses to verify the size actually took)
cannot run while the MJPEG stream is holding the slot.

THE TWO FIRMWARES SET UP DIFFERENTLY
------------------------------------
CameraWebServer has the HTTP /control endpoint, so everything here works.
Since the MJPEG switchover the robot firmware serves the same endpoints, so
--source webserver now prepares either sketch. Only the legacy UDP video
path (--source esp32) still bypasses this: on it
all — its resolution is a UDP command instead (RobotLink.set_framesize)
and its AWB cannot be locked from this end: updateCameraSettings() in the
.ino re-enables set_whitebal/set_awb_gain every time. So on --source
esp32 prepare_camera() does nothing and says so.
"""

import socket
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from camera_control import (RESOLUTIONS, lock_image_settings, set_ae_level,
                            set_flip, set_quality, set_resolution)

STREAM_PORT = 81
FIRST_FRAME_TIMEOUT = 15.0

# --flip name -> (vertical, horizontal)
FLIP_MODES = {
    "none": (False, False),
    "v": (True, False),
    "h": (False, True),
    "both": (True, True),
}


# ============================================================
# CAMERA DISCOVERY
# ============================================================

def local_subnet_prefix():
    """Best guess at the /24 this PC is on. The UDP 'connect' sends no
    packets — it just makes the OS pick the outbound interface."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()
    return ip.rsplit(".", 1)[0]


def _probe(ip, timeout=0.4):
    """A host is the camera if its stream port is open. Plain socket check
    only — actually opening /stream would consume the camera's single
    client slot, and we'd have to drop it again a moment later."""
    try:
        with socket.create_connection((ip, STREAM_PORT), timeout=timeout):
            return ip
    except Exception:
        return None


def discover_camera(prefix, workers=128):
    print(f"Scanning {prefix}.1-254 for an ESP32-CAM on port {STREAM_PORT} ...")
    hosts = [f"{prefix}.{i}" for i in range(1, 255)]
    found = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_probe, hosts):
            if result:
                found.append(result)
    return found


def resolve_camera_ip(explicit_ip):
    if explicit_ip:
        return explicit_ip

    prefix = local_subnet_prefix()
    if not prefix:
        print("Could not work out this PC's subnet — pass --esp-ip manually.")
        return None

    found = discover_camera(prefix)
    if not found:
        print("No camera found. Check that:")
        print("  - the ESP32-CAM is powered and finished booting")
        print("  - it joined the same WiFi as this PC")
        print("  - CameraWebServer.ino OR firmware/esp32cam_robot/ is flashed")
        print("    (both serve the stream on port 81 now)")
        print("Otherwise read the IP off the Serial Monitor and pass --esp-ip.")
        return None
    if len(found) > 1:
        print(f"Multiple candidates found: {', '.join(found)} — using {found[0]}.")
        print("Pass --esp-ip if that's the wrong one.")
    print(f"Found camera at {found[0]}")
    return found[0]


# ============================================================
# STARTUP SETUP
# ============================================================

def prepare_camera(source, esp_ip, framesize="vga", lock=True,
                   quality=None, ae_level=None):
    """Find the camera and put it in a known state before the stream opens.

    Returns the resolved IP (which may be one this function discovered),
    or None if --source webserver was asked for and no camera turned up —
    the caller should give up in that case.

    On --source esp32 this is a no-op that returns esp_ip unchanged: that
    firmware has no HTTP control endpoint. Use RobotLink.set_framesize()
    for its resolution."""
    if source.lower() != "webserver":
        return esp_ip

    esp_ip = resolve_camera_ip(esp_ip)
    if not esp_ip:
        return None

    want_w, want_h = RESOLUTIONS[framesize][0]
    actual = set_resolution(esp_ip, framesize)
    if actual is None:
        print("Camera found but not answering on port 80 — leaving the resolution as-is.")
    elif actual != (want_w, want_h):
        print(f"Camera refused {framesize.upper()} ({want_w}x{want_h}); "
              f"running at {actual[0]}x{actual[1]} instead.")
    else:
        print(f"Camera set to {framesize.upper()} ({want_w}x{want_h}).")

    # JPEG quality: 10 (best) .. 63 (worst), and the biggest framerate knob
    # on this end. CameraWebServer boots at 10, which is the LARGEST frames
    # it can produce — on a link that tops out around 2 Mbit/s in a normal
    # room, VGA at q10 is a handful of frames per second. q16-18 roughly
    # halves the bytes for a picture a blue-blob detector cannot tell apart.
    if quality is not None:
        if set_quality(esp_ip, quality):
            print(f"JPEG quality set to {quality} (10=best/biggest, 63=worst/smallest).")
        else:
            print(f"Camera did not accept quality {quality}.")

    # Before the lock, while AEC is still running and can act on it.
    if ae_level is not None:
        if set_ae_level(esp_ip, ae_level):
            print(f"Exposure bias set to {ae_level:+d} "
                  f"({'shorter shutter' if ae_level < 0 else 'longer shutter'}).")
        else:
            print(f"Camera did not accept ae_level {ae_level}.")

    # After a settle pause, so the auto loops freeze on a properly adjusted
    # image of the real scene rather than a half-converged one.
    if lock:
        print("Locking white balance / exposure (2s settle — point the camera at the scene) ...")
        if lock_image_settings(esp_ip):
            print("Locked. Colors will now stay put; re-run with --no-lock to undo.")
        else:
            print("Camera did not accept the lock — colors will drift as the light changes.")

    return esp_ip


def set_robot_framesize(esp_ip, framesize):
    """Put the ROBOT firmware into a resolution, over its UDP control port.

    prepare_camera() cannot do this: that firmware serves no HTTP at all.
    The command is cmd 4/5/6/7, and the .ino answers it by calling
    updateCameraSettings(), so it also re-applies the sensor settings as a
    side effect.

    Opens a link, sends, and closes it again — no drive commands are ever
    sent, so this is safe to call from a viewer like run.py. Returns True
    if the command left the socket."""
    if not esp_ip:
        print("Can't set the robot's resolution without --esp-ip; leaving it alone.")
        return False

    from robot_link import RobotLink
    link = RobotLink(esp_ip)
    link.start()
    try:
        link.set_framesize(framesize)
        time.sleep(0.2)          # let the sender thread actually transmit
    finally:
        link.close()
    print(f"Asked the robot firmware for {framesize.upper()} on {esp_ip}.")
    return True


# ============================================================
# FLIP
# ============================================================

def flip_frame(frame, vflip, hmirror):
    """Software flip, for when the sensor can't do it for us (the UDP
    firmware has no HTTP control endpoint) or refuses to."""
    if vflip and hmirror:
        return cv2.flip(frame, -1)
    if vflip:
        return cv2.flip(frame, 0)
    if hmirror:
        return cv2.flip(frame, 1)
    return frame


def push_flip_to_sensor(source, esp_ip, vflip, hmirror):
    """Try to move the flip into the sensor. Returns True if the camera
    took it, in which case the frames already arrive flipped and doing it
    again in OpenCV would undo the work.

    Only the CameraWebServer firmware has the HTTP endpoint. The UDP one
    doesn't serve port 80 at all, so asking it would just burn two HTTP
    timeouts before falling back — skip straight to software there."""
    if not esp_ip or source.lower() != "webserver":
        return False
    return set_flip(esp_ip, vflip=vflip, hmirror=hmirror)


def apply_flip(source, esp_ip, flip_mode):
    """Resolve a --flip name into (vflip, hmirror, done_in_sensor) and push
    it to the camera. Always pushed, even for 'none' — that clears a flip
    left in the sensor by an earlier run (it survives until reboot)."""
    vflip, hmirror = FLIP_MODES[flip_mode]
    in_sensor = push_flip_to_sensor(source, esp_ip, vflip, hmirror)
    if vflip or hmirror:
        where = "in the sensor" if in_sensor else "in software"
        print(f"Flip: {flip_mode} ({where}).")
    return vflip, hmirror, in_sensor
