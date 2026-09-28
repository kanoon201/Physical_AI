"""
Video source selection
=======================
One place that decides which live-camera receiver to use, so every tool
here (run.py, capture_esp32_samples.py) accepts the same --source values
and they stay consistent.

These two names pick a TRANSPORT, not a sketch:

  --source webserver   MJPEG over HTTP (port 81, /stream)
  --source esp32       JPEG in UDP chunks (ports 44444/44442)

firmware/esp32cam_robot/ speaks both and defaults to MJPEG, because TCP
retransmits: a frame either arrives byte-exact or the socket stalls, and it
can never arrive half-decoded the way a UDP frame with a missing chunk does.
Its UDP path is kept as a fallback -- it is the only one that can find the PC
by broadcast without being told an IP first. CameraWebServer speaks only
MJPEG and has no motors, so --source webserver alone does not mean "can
drive"; autodrive.py checks /status for the robot marker.

Both hand back plain BGR frames, so whatever consumes them doesn't care
which one is behind it. A full 'http://...' URL is also accepted and
treated as an MJPEG stream.
"""

STREAM_SOURCES = ("esp32", "webserver")


def is_stream_source(source):
    s = source.lower()
    return s in STREAM_SOURCES or s.startswith("http://") or s.startswith("https://")


def make_stream_source(source, esp_ip=None):
    """Build (but do not start) the receiver for a live-camera source.
    Returns None if `source` isn't a live-camera source — the caller
    should fall back to cv2.VideoCapture / cv2.imread for that."""
    s = source.lower()

    if s == "esp32":
        from esp32_video_source import ESP32VideoSource
        return ESP32VideoSource(esp_ip=esp_ip)

    if s == "webserver" or s.startswith("http://") or s.startswith("https://"):
        from mjpeg_video_source import MJPEGVideoSource
        target = source if s.startswith("http") else esp_ip
        if not target:
            raise ValueError("--source webserver needs --esp-ip (the ESP32-CAM's IP address)")
        return MJPEGVideoSource(target)

    return None
