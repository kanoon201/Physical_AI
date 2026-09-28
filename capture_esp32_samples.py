"""
ESP32-CAM Sample Capture Tool
===============================
Grabs frames directly from the ESP32-CAM's live video stream — the same
compressed, low-res feed the robot actually sees — and saves the ones
you pick to disk as .jpg files. Use these for any offline work instead of
webcam photos, since webcam images look nothing like what the ESP32-CAM
actually produces.

Usage:
    python capture_esp32_samples.py --esp-ip 192.168.8.191
    python capture_esp32_samples.py                        # broadcast-discover instead
    python capture_esp32_samples.py --source webserver --esp-ip 192.168.8.191   # CameraWebServer sketch

Controls:
    's' = save the current frame to ./esp32_samples/
    'q' = quit
"""

import cv2
import os
import argparse
import time
from video_source import make_stream_source

OUTPUT_DIR = "esp32_samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esp-ip", default=None, help="ESP32-CAM IP address. Required for --source webserver; optional for --source esp32 (omit to broadcast-discover instead)")
    parser.add_argument("--source", default="webserver",
                        help="Video transport: 'webserver' = MJPEG over HTTP (default; both "
                             "sketches speak it), 'esp32' = the legacy UDP chunk path")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Folder to save captured frames into")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    src = make_stream_source(args.source, esp_ip=args.esp_ip)
    if src is None:
        print(f"--source must be 'esp32', 'webserver', or an http:// stream URL (got: {args.source})")
        return
    src.start()
    print("Connecting to ESP32-CAM... (waiting for first frame — make sure it's powered on)")
    print(f"Press 's' to save a frame into ./{args.out}/, 'q' to quit.")

    saved_count = len([f for f in os.listdir(args.out) if f.lower().endswith(".jpg")])

    try:
        while True:
            frame = src.get_frame()
            if frame is None:
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                continue

            display = frame.copy()
            cv2.putText(display, f"Saved: {saved_count}  (press 's' to save, 'q' to quit)",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("ESP32-CAM - Sample Capture", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = os.path.join(args.out, f"esp32_sample_{int(time.time()*1000)}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                print(f"Saved: {filename}")
    finally:
        src.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
