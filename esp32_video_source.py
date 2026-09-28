"""
ESP32-CAM Video Source (standalone receiver)
==============================================
The webcam on your PC is a completely different sensor/lens/compression
pipeline than the ESP32-CAM — colors, sharpness, and noise characteristics
don't match, so nothing tested on webcam footage carries over to the
robot's real feed. This module receives frames DIRECTLY from the
ESP32-CAM's UDP stream (the same stream 2026_07_13.py uses), so anything
run against it sees exactly what the robot sees.

Protocol (matches sketch_jul13a.ino):
  - The ESP32 listens on DISCOVERY_PORT for ANY UDP packet and starts
    streaming video to whichever IP that packet came from. We send a
    small beacon periodically so it knows we exist.
  - Video arrives on VIDEO_PORT as JPEG frames fragmented into UDP
    chunks: an 8-byte header (4-byte little-endian seq, 2-byte
    total_chunks, 2-byte chunk_id) followed by the chunk payload.

THE BEACON CARRIES A MEASUREMENT, NOT A CONSTANT
------------------------------------------------
`delivery:<pct>` in the beacon is the other half of the firmware's
adaptive quality loop, and both halves have to be real for it to work.
The firmware raises compression after 3 consecutive readings under 50%
and lowers it after 5 consecutive over 80%.

This used to send a hardcoded `delivery:100`. On a lossy link that is not
a harmless white lie — it is a signal pointing the wrong way. The
firmware reads a perfect link, lowers the compression, and produces
BIGGER frames, which need more chunks, which lose more packets. Measured
on an iPhone hotspot: 22% chunk loss, frames grown to 32 chunks, half of
them arriving incomplete, 1.2 usable fps.

So the loss is measured here instead: the sequence number is a frame
counter, so a gap between two consecutive COMPLETE frames is exactly the
number of frames that did not survive. That ratio, smoothed, is what goes
out in the beacon.

Usage as a module:
    from esp32_video_source import ESP32VideoSource
    src = ESP32VideoSource(esp_ip="192.168.8.191")  # or None to broadcast
    src.start()
    frame = src.get_frame()   # BGR numpy array, or None until first frame arrives
    ...
    src.stop()

Usage standalone (just to check the feed comes through):
    python esp32_video_source.py --esp-ip 192.168.8.191
    python esp32_video_source.py            # broadcast discovery instead
"""

import socket
import time
import threading
import argparse
import cv2
import numpy as np

DISCOVERY_PORT = 44444
VIDEO_PORT = 44442
HOST_IP = "0.0.0.0"

# How far a sequence number may run BACKWARDS and still be treated as
# ordinary UDP reordering rather than the ESP having restarted. A few
# frames is reordering; hundreds is a new boot with a fresh counter.
REORDER_WINDOW = 30


class ESP32VideoSource:
    def __init__(self, esp_ip=None, beacon_interval=0.5):
        """
        esp_ip: IP of the ESP32-CAM if known (recommended — faster to
                connect and avoids flooding the LAN with broadcasts).
                Leave None to broadcast on the local subnet instead
                (slower to establish, works without knowing the IP).
        """
        self.esp_ip = esp_ip
        self.beacon_interval = beacon_interval
        self._running = False
        self._latest_frame = None
        self._frame_count = 0
        # Total JPEG bytes of frames that arrived COMPLETE. Chunks belonging
        # to frames that were never fully reassembled are not counted, which
        # is what makes this comparable with the MJPEG source.
        self.bytes_received = 0
        # Percent of frames that arrive complete, EMA-smoothed. Starts
        # optimistic so a healthy link isn't throttled during the first
        # second, before there is anything to measure.
        self.stream_delivery_pct = 100.0
        self._last_complete_seq = None
        # Counted per beacon window, not per frame: "3 of the 4 frames the
        # camera sent got here" is the number the firmware's loop is written
        # against. Averaging 1/gap per frame instead flatters a lossy link —
        # gaps of 1,1,2 average to 83% when only 3 of 4 frames arrived.
        self._window_delivered = 0
        self._window_expected = 0
        self._lock = threading.Lock()
        self._threads = []

    def start(self):
        self._running = True
        t1 = threading.Thread(target=self._beacon_loop, daemon=True)
        t2 = threading.Thread(target=self._recv_loop, daemon=True)
        t1.start()
        t2.start()
        self._threads = [t1, t2]

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)

    def get_frame(self):
        """Returns the latest decoded BGR frame, or None if nothing received yet."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    @property
    def frame_count(self):
        return self._frame_count

    def _note_delivery(self, seq):
        """Fold one completed frame into the current window. Lock held.

        The sequence number is a frame counter, so `seq - prev` is exactly
        how many frames the camera sent since the last one that made it
        here — one of which is this frame, and the rest are losses."""
        prev = self._last_complete_seq
        self._last_complete_seq = seq

        # First frame, or the ESP rebooted and restarted its counter.
        # Re-baseline instead of reporting a fake collapse.
        if prev is None or seq <= prev or (seq - prev) > 1000:
            return

        self._window_delivered += 1
        self._window_expected += seq - prev

    def _delivery_snapshot(self):
        """Close the window, fold it into the EMA, return the percentage.

        A window with nothing in it keeps the previous figure rather than
        reporting 0: no frames at all means the link is down or the sensor
        is asleep, and answering that with maximum compression would be
        treating a disconnection as a bandwidth problem."""
        with self._lock:
            if self._window_expected > 0:
                instant = 100.0 * self._window_delivered / self._window_expected
                self.stream_delivery_pct = (0.6 * self.stream_delivery_pct
                                            + 0.4 * instant)
                self._window_delivered = 0
                self._window_expected = 0
            return int(round(max(0.0, min(100.0, self.stream_delivery_pct))))

    def _beacon_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self._running:
                # Rebuilt every time: this is the feedback signal the
                # firmware's compression loop runs on, so a stale or
                # invented number steers it wrong.
                msg = f"PC_HERE,delivery:{self._delivery_snapshot()}".encode()
                try:
                    if self.esp_ip:
                        s.sendto(msg, (self.esp_ip, DISCOVERY_PORT))
                    else:
                        s.sendto(msg, ('<broadcast>', DISCOVERY_PORT))
                except Exception:
                    pass
                time.sleep(self.beacon_interval)

    def _recv_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
        except Exception:
            pass
        srv.bind((HOST_IP, VIDEO_PORT))
        srv.settimeout(0.5)

        frames = {}
        last_processed_seq = 0

        while self._running:
            try:
                data, addr = srv.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                continue

            if len(data) < 8:
                continue

            seq = int.from_bytes(data[0:4], 'little')
            total = int.from_bytes(data[4:6], 'little')
            chunk_id = int.from_bytes(data[6:8], 'little')
            payload = data[8:]

            if seq < last_processed_seq:
                if last_processed_seq - seq < REORDER_WINDOW:
                    continue        # ordinary UDP reordering, genuinely stale

                # A big jump backwards means the ESP restarted and its frame
                # counter began again at 1 — which it does on its own after
                # 120s with no beacon (BEACON_TIMEOUT in the .ino), and after
                # every reflash. Holding on to the old high-water mark here
                # discards every new frame until the fresh counter climbs past
                # it: at a few fps that is minutes of "camera connected, no
                # picture" for no reason. Re-baseline instead.
                print(f"[esp32] sequence restarted ({last_processed_seq} -> {seq}); "
                      "the camera rebooted. Resyncing.")
                frames.clear()
                last_processed_seq = 0
                with self._lock:
                    self._last_complete_seq = None
                    self.stream_delivery_pct = 100.0
                    self._window_delivered = 0
                    self._window_expected = 0

            if seq not in frames:
                if frames:
                    for k in list(frames.keys()):
                        if k < seq:
                            del frames[k]
                frames[seq] = {'total': total, 'chunks': {}}

            frames[seq]['chunks'][chunk_id] = payload

            if len(frames[seq]['chunks']) == frames[seq]['total']:
                entry = frames.pop(seq)
                try:
                    full_frame = b"".join(entry['chunks'][i] for i in range(entry['total']))
                except KeyError:
                    continue

                if full_frame.startswith(b'\xff\xd8') and full_frame.endswith(b'\xff\xd9'):
                    np_arr = np.frombuffer(full_frame, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._latest_frame = frame
                            self._frame_count += 1
                            self.bytes_received += len(full_frame)
                            self._note_delivery(seq)
                        last_processed_seq = seq

        srv.close()


def main():
    parser = argparse.ArgumentParser(description="Preview the raw ESP32-CAM feed (no controller UI, just the video)")
    parser.add_argument("--esp-ip", default=None, help="ESP32-CAM IP (skip to broadcast-discover instead)")
    args = parser.parse_args()

    src = ESP32VideoSource(esp_ip=args.esp_ip)
    src.start()
    print("Waiting for frames... (make sure the ESP32-CAM is powered on and on the same WiFi)")
    print("Press 'q' to quit.")

    last_seen_count = -1
    try:
        while True:
            frame = src.get_frame()
            if frame is not None:
                if src.frame_count != last_seen_count:
                    last_seen_count = src.frame_count
                cv2.imshow("ESP32-CAM Live Feed", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
