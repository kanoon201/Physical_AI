"""
MJPEG HTTP Video Source (for the stock CameraWebServer sketch)
================================================================
The CameraWebServer sketch serves an MJPEG stream over HTTP on port 81
(`http://<ip>:81/stream`) instead of the UDP protocol that
sketch_jul13a.ino uses. This module receives that stream and exposes the
exact same interface as ESP32VideoSource, so run.py and
capture_esp32_samples.py can use either camera firmware without any
other change.

Frames are pulled on a background thread and only the newest one is kept.
That matters: an MJPEG socket buffers frames, so reading it synchronously
in the main loop makes the video lag further behind reality the longer
you run.

Usage as a module:
    from mjpeg_video_source import MJPEGVideoSource
    src = MJPEGVideoSource("192.168.8.191")      # or a full http:// URL
    src.start()
    frame = src.get_frame()   # BGR numpy array, or None until the first frame
    ...
    src.stop()

Usage standalone (just to check the feed comes through):
    python mjpeg_video_source.py --esp-ip 192.168.8.191

NOTE: the ESP32-CAM only serves ONE stream client at a time. Close the
browser tab showing the camera page before running this, otherwise no
frames will arrive.
"""

import argparse
import collections
import threading
import time
import urllib.request

import cv2
import numpy as np

STREAM_PORT = 81
STREAM_PATH = "/stream"

SOI = b"\xff\xd8"  # JPEG start-of-image
EOI = b"\xff\xd9"  # JPEG end-of-image


def build_stream_url(target):
    """Accepts either a bare IP/host ('192.168.8.191') or a full URL, and
    returns the MJPEG stream URL. A bare host gets the CameraWebServer
    defaults (port 81, /stream) filled in."""
    if target.startswith("http://") or target.startswith("https://"):
        return target
    return f"http://{target}:{STREAM_PORT}{STREAM_PATH}"


class MJPEGVideoSource:
    def __init__(self, target, timeout=5.0, reconnect_delay=1.0, max_buffer=2_000_000):
        """
        target: IP/host of the ESP32-CAM, or a full stream URL.
        max_buffer: safety cap on the reassembly buffer. If the stream
                    desyncs and no EOI ever shows up, this stops the
                    buffer from growing until the process runs out of RAM.
        """
        self.url = build_stream_url(target)
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_buffer = max_buffer
        self._running = False
        self._latest_frame = None
        self._frame_count = 0
        # Total JPEG bytes accepted. Only useful for telling a camera-limited
        # framerate apart from a PC-limited one: divide by elapsed time and
        # compare against what the link can actually carry.
        self.bytes_received = 0
        # How many times the stream had to be re-opened. A --bench run
        # that looks slow but shows reconnects is a dropping link, not a
        # camera that cannot produce frames.
        self.reconnects = 0
        # perf_counter() of each accepted frame, newest last. Stutter is a
        # frame-SPACING problem that average fps hides completely, and this
        # is the only place the spacing can be measured without a polling
        # loop whose own sleep granularity (15 ms on Windows) blurs it.
        self.arrival_times = collections.deque(maxlen=4096)
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1.0)

    def get_frame(self):
        """Returns the latest decoded BGR frame, or None if nothing received yet."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    @property
    def frame_count(self):
        return self._frame_count

    def _recv_loop(self):
        while self._running:
            try:
                stream = urllib.request.urlopen(self.url, timeout=self.timeout)
            except Exception as e:
                # Camera not up yet, wrong IP, or another client already
                # holds the single available stream slot. Keep retrying so
                # the tool survives a camera reboot mid-session.
                self.reconnects += 1
                print(f"[mjpeg] cannot open {self.url}: {e} (retrying)")
                time.sleep(self.reconnect_delay)
                continue

            buf = b""
            try:
                while self._running:
                    # read1(), NOT read(). urlopen() hands back an
                    # HTTPResponse wrapped in a BufferedReader, and
                    # BufferedReader.read(n) blocks until it has all n bytes.
                    # An MJPEG response has no overall Content-Length, so that
                    # is the path taken: a JPEG that is already complete sits
                    # unnoticed until ~4KB of the NEXT frame turns up, which
                    # is exactly one frame interval of delay and makes frames
                    # arrive in pairs. read1() returns whatever one underlying
                    # read produced, so a finished frame is handed over the
                    # moment it lands.
                    chunk = stream.read1(4096)
                    if not chunk:
                        break  # stream ended — fall through and reconnect
                    buf += chunk

                    # Pull out every complete JPEG sitting in the buffer.
                    # The multipart headers between parts are skipped
                    # implicitly by searching for the SOI marker.
                    while True:
                        start = buf.find(SOI)
                        if start < 0:
                            break
                        end = buf.find(EOI, start + 2)
                        if end < 0:
                            break
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                             cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._latest_frame = frame
                                self._frame_count += 1
                                self.bytes_received += len(jpg)
                                self.arrival_times.append(time.perf_counter())

                    if len(buf) > self.max_buffer:
                        buf = b""
            except Exception as e:
                if self._running:
                    print(f"[mjpeg] stream error: {e} (reconnecting)")
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

            if self._running:
                time.sleep(self.reconnect_delay)


def main():
    parser = argparse.ArgumentParser(
        description="Preview the CameraWebServer MJPEG feed (no detection, just the video)")
    parser.add_argument("--esp-ip", required=True,
                        help="ESP32-CAM IP (e.g. 192.168.8.191) or a full stream URL")
    args = parser.parse_args()

    src = MJPEGVideoSource(args.esp_ip)
    src.start()
    print(f"Connecting to {src.url} ...")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = src.get_frame()
            if frame is not None:
                cv2.imshow("ESP32-CAM MJPEG Feed", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
