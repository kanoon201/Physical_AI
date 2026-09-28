"""
Blue cone detection (color + shape, no steering)
==================================================
Finds BLUE traffic cones in a BGR frame and hands back where they are.
Nothing here decides how to drive — that's a later step on purpose, so
the detection can be judged on its own.

Why blue rather than orange:
  - Orange sits on the H=0/179 seam, so it needs two inRange passes
    stitched together. Blue sits mid-range (~100) and takes one.
  - The things that ruin an orange detector indoors — skin, cardboard,
    wooden floors and desks — all live at H 5-25, right on top of an
    orange cone. Almost nothing indoors is saturated blue.

Color alone is not enough, though, so every blob has to survive four
cheap filters (area, aspect, solidity, ROI) before it counts as a cone.
See DEFAULT_CONFIG below for what each one throws away.

    python cone_detection.py --source webserver --esp-ip 192.168.8.191
    python cone_detection.py --source esp32
    python cone_detection.py --source esp32_samples/esp32_sample_123.jpg

Controls:
    trackbars : adjust until the Mask window shows ONLY the cones white
    'w'       : write the current values to cone_config.json
    'q'       : quit

Lock the camera's AWB/exposure BEFORE tuning (run.py does it for you, or
call camera_control.lock_image_settings). HSV values tuned while auto
white balance is running describe the light, not the cone.

Usage as a module:
    from cone_detection import ConeDetector
    det = ConeDetector()
    cones, mask, debug = det.detect(bgr_frame)
    for c in cones:
        print(c.cx, c.cy, c.area, c.bbox)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

CONFIG_PATH = "cone_config.json"

# Starting point for a blue cone under white indoor light. These are a
# place to begin tuning, not an answer — every camera and every room
# shifts them.
DEFAULT_CONFIG = {
    # Hue 90-130 covers cyan-blue through to blue-violet. One range is
    # enough; blue is nowhere near the H=0/179 wrap that forces orange
    # and red to use two.
    "hsv_lower": [90, 80, 40],
    "hsv_upper": [130, 255, 255],

    # --- FILTERS ---
    # Area is in ABSOLUTE PIXELS, so it is tied to the resolution the
    # camera happens to be sending. 1500 px is ~0.5% of a VGA frame but
    # ~2% of QVGA — re-tune it if you change --framesize.
    "min_contour_area": 1500,
    "max_contour_area": 60000,

    # width / height of the bounding box. A cone is taller than it is
    # wide, so anything square or landscape is something else. This is
    # what throws out jeans, chair backs and monitors.
    "min_aspect_ratio": 0.2,
    "max_aspect_ratio": 1.3,

    # contour area / convex hull area. A cone is a solid, convex-ish blob
    # (~0.85+). Ragged noise and things seen through gaps score low.
    "min_solidity": 0.75,

    # Ignore everything above this fraction of the frame height. Cones
    # sit on the floor; the sky through a window, monitors, and people's
    # clothing are all above the horizon. Cheapest filter here by far —
    # set it to 0.0 to disable.
    "roi_top_ratio": 0.4,

    # Kernel for the open/close pass. Blue suffers more than most colors
    # from JPEG chroma subsampling, so distant cones arrive speckled and
    # need the CLOSE to knit them back together.
    "morph_kernel_size": 5,
}


def load_config(path=CONFIG_PATH):
    """Missing or partial config falls back to the defaults key by key, so
    an old file doesn't crash the detector when a new filter is added."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read {path} ({e}) — using defaults.")
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ============================================================
# DETECTION
# ============================================================

class Cone:
    __slots__ = ("cx", "cy", "area", "bbox")

    def __init__(self, cx, cy, area, bbox):
        self.cx = cx
        self.cy = cy
        self.area = area
        self.bbox = bbox  # (x, y, w, h)

    def __repr__(self):
        return f"Cone(cx={self.cx}, cy={self.cy}, area={self.area:.0f})"


def build_mask(bgr_frame, cfg):
    """BGR in, binary mask out. Takes BGR straight from cv2.imdecode — not
    an RGB frame built for display, which swaps the red and blue channels
    and so turns a blue cone into an orange one."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(cfg["hsv_lower"], dtype=np.uint8),
                       np.array(cfg["hsv_upper"], dtype=np.uint8))

    k = max(1, int(cfg["morph_kernel_size"]))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # kill speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close JPEG gaps

    # Blank the ROI in the mask rather than cropping the frame: contour
    # coordinates stay in full-frame space, so nothing downstream has to
    # remember to add an offset back.
    top = int(mask.shape[0] * float(cfg["roi_top_ratio"]))
    if top > 0:
        mask[:top, :] = 0
    return mask


def find_cones(mask, cfg):
    """Every blob in the mask that survives all four filters, biggest
    first (biggest is usually nearest)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cones = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg["min_contour_area"] or area > cfg["max_contour_area"]:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if aspect < cfg["min_aspect_ratio"] or aspect > cfg["max_aspect_ratio"]:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (area / hull_area) < cfg["min_solidity"]:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cones.append(Cone(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]),
                          area, (x, y, w, h)))

    cones.sort(key=lambda c: -c.area)
    return cones


class ConeDetector:
    def __init__(self, config_path=CONFIG_PATH):
        self.cfg = load_config(config_path)

    def detect(self, bgr_frame, draw=True):
        """Returns (cones, mask, debug_frame). cones is largest-first;
        debug_frame is None when draw=False."""
        mask = build_mask(bgr_frame, self.cfg)
        cones = find_cones(mask, self.cfg)
        debug = self.draw(bgr_frame, cones) if draw else None
        return cones, mask, debug

    def draw(self, bgr_frame, cones):
        out = bgr_frame.copy()
        h, w = out.shape[:2]

        top = int(h * float(self.cfg["roi_top_ratio"]))
        if top > 0:
            # Dim the ignored band so it's obvious why a cone up there
            # isn't being found.
            out[:top, :] = (out[:top, :] * 0.35).astype(np.uint8)
            cv2.line(out, (0, top), (w, top), (80, 80, 80), 1)

        for i, c in enumerate(cones):
            x, y, cw, ch = c.bbox
            # Largest cone in green, the rest in yellow.
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.rectangle(out, (x, y), (x + cw, y + ch), color, 2)
            cv2.circle(out, (c.cx, c.cy), 4, color, -1)
            cv2.putText(out, f"{c.area:.0f}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(out, f"cones: {len(cones)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out


# ============================================================
# TUNING UI
# ============================================================

def _nothing(_):
    pass


TRACKBARS = (
    ("H min", "hsv_lower", 0, 179),
    ("S min", "hsv_lower", 1, 255),
    ("V min", "hsv_lower", 2, 255),
    ("H max", "hsv_upper", 0, 179),
    ("S max", "hsv_upper", 1, 255),
    ("V max", "hsv_upper", 2, 255),
)


def build_trackbars(window, cfg):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 420, 340)
    for name, key, idx, maximum in TRACKBARS:
        cv2.createTrackbar(name, window, int(cfg[key][idx]), maximum, _nothing)
    # Upper bound sized for VGA — a cone covers ~4x the pixels it does at
    # QVGA, so a range that felt generous there is not enough here.
    cv2.createTrackbar("min area", window, int(cfg["min_contour_area"]), 8000, _nothing)
    cv2.createTrackbar("solidity %", window, int(cfg["min_solidity"] * 100), 100, _nothing)
    cv2.createTrackbar("roi top %", window, int(cfg["roi_top_ratio"] * 100), 90, _nothing)


def read_trackbars(window, cfg):
    for name, key, idx, _maximum in TRACKBARS:
        cfg[key][idx] = cv2.getTrackbarPos(name, window)
    cfg["min_contour_area"] = max(1, cv2.getTrackbarPos("min area", window))
    cfg["min_solidity"] = cv2.getTrackbarPos("solidity %", window) / 100.0
    cfg["roi_top_ratio"] = cv2.getTrackbarPos("roi top %", window) / 100.0


def print_saved(cfg, path):
    print(f"Saved to {path}: hsv {cfg['hsv_lower']}..{cfg['hsv_upper']}  "
          f"min_area={cfg['min_contour_area']}  solidity={cfg['min_solidity']:.2f}  "
          f"roi_top={cfg['roi_top_ratio']:.2f}")


# ============================================================
# STANDALONE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Detect blue cones in a live feed or a still image")
    parser.add_argument("--source", default="webserver",
                        help="'webserver', 'esp32', an http:// stream URL, or a path to an image")
    parser.add_argument("--esp-ip", default=None, help="Camera IP (needed for --source webserver)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to config JSON")
    args = parser.parse_args()

    from video_source import make_stream_source

    det = ConeDetector(config_path=args.config)
    control_win = "Controls (w=write config, q=quit)"
    build_trackbars(control_win, det.cfg)

    src = make_stream_source(args.source, esp_ip=args.esp_ip)

    # Not a live source — treat it as an image on disk. Handy for
    # re-tuning against saved frames with no camera plugged in.
    if src is None:
        still = cv2.imread(args.source)
        if still is None:
            print(f"Not a live source and not a readable image: {args.source}")
            return 1
        try:
            while True:
                read_trackbars(control_win, det.cfg)
                _cones, mask, debug = det.detect(still)
                cv2.imshow("Cone Detection", debug)
                cv2.imshow("Mask", mask)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('w'):
                    save_config(det.cfg, args.config)
                    print_saved(det.cfg, args.config)
        finally:
            cv2.destroyAllWindows()
        return 0

    src.start()
    print("Waiting for the first frame ... ('w' writes the config, 'q' quits)")
    try:
        while True:
            frame = src.get_frame()
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            read_trackbars(control_win, det.cfg)
            _cones, mask, debug = det.detect(frame)
            cv2.imshow("Cone Detection", debug)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('w'):
                save_config(det.cfg, args.config)
                print_saved(det.cfg, args.config)
    finally:
        src.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
