"""
Robot control link (UDP)
==========================
The sending half of the robot protocol, lifted out of the pygame
controller in ../Aj-Piya_code/2026_07_13.py and stripped to just the
transport: sequence numbers, de-duplication, batching, failsafe.
Nothing here decides where to drive — see autopilot.py for that.

    link = RobotLink("192.168.8.191")
    link.start()
    link.drive(steer=0.2, throttle=0.35)   # both -1..1
    link.stop()                            # neutral, sent hard
    link.close()

WHICH FIRMWARE
--------------
This talks to `firmware/esp32cam_robot/` (the sketch_jul13a lineage).
CameraWebServer has no motor control at all, so `--source webserver` can
see cones but can never drive. Flash the robot firmware and use
`--source esp32` before expecting anything to move.

THE PROTOCOL (must match the .ino byte for byte)
------------------------------------------------
Plain text UDP to port 44441: `seq,cmd,v1,v2`

  cmd 0  drive   v1 = steer, v2 = throttle, both 1000..2000us, 1500 = neutral
  cmd 1  light   v1 = 0..65535 PWM
  cmd 3  reboot  v1 = v2 = 3
  cmd 4..7       resolution preset QVGA / VGA / SVGA / XGA

`v1`/`v2` repeat the cmd number on the system commands as a crude
checksum — the firmware refuses cmd 3 unless v1 and v2 are also 3, so a
corrupted packet can't reboot the robot mid-drive.

The firmware ignores any packet whose seq is not newer than the last one
it accepted (with a wrap window near SEQ_LIMIT), which is what makes UDP
reordering harmless.

TWO TIMEOUTS THAT WILL BITE
---------------------------
- The firmware zeroes the motors 150 ms after the last control packet.
  That is a safety feature, not a bug: it means a crashed PC stops the
  car. But it also means "keep driving straight" has to be re-sent
  continuously, so drive() re-sends an unchanged command every 50 ms.
- The ESP reboots after 120 s with no discovery beacon. The video source
  already beacons, so as long as video is running this is covered.
"""

import queue
import socket
import threading
import time

CONTROL_PORT = 44441
DISCOVERY_PORT = 44444

SEQ_LIMIT = 10000          # matches SEQ_LIMIT in the .ino
COMMAND_QUEUE_SIZE = 5
COMMAND_TX_INTERVAL = 0.01
DRIVE_RESEND_INTERVAL = 0.05   # < the firmware's 150 ms failsafe, with margin
LIGHT_RESEND_INTERVAL = 1.0

CMD_DRIVE = 0
CMD_LIGHT = 1
CMD_REBOOT = 3
CMD_FRAMESIZE = {"qvga": 4, "vga": 5, "svga": 6, "xga": 7}

NEUTRAL_US = 1500
MAX_LIGHT = 65535


def ev(v):
    """-1..+1  ->  1000..2000 us, 1500 = neutral. Same mapping the pygame
    controller used; the firmware multiplies by MAX_DUTY (3.27675) to get
    a 16-bit duty at 50 Hz, i.e. a standard servo/ESC pulse."""
    half = (2000 - 1000) * 0.5
    return 1000 + int(half + (max(-1.0, min(1.0, v)) * half))


class RobotLink:
    def __init__(self, esp_ip, port=CONTROL_PORT):
        if not esp_ip:
            raise ValueError("RobotLink needs the robot's IP address")
        self.esp_ip = esp_ip
        self.port = port
        self._q = queue.Queue(maxsize=COMMAND_QUEUE_SIZE)
        self._seq = 0
        self._running = False
        self._thread = None
        self._sock = None
        self._lock = threading.Lock()

        self._last_drive = None
        self._last_drive_at = 0.0
        self._last_light = None
        self._last_light_at = 0.0
        self.packets_sent = 0

    # --- lifecycle ---

    def start(self):
        if self._running:
            return
        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._thread.start()

    def close(self):
        """Always stop() before this. Closing the socket with the car still
        rolling leaves it to coast until the firmware's 150 ms failsafe."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        try:
            self.stop()
        finally:
            self.close()

    # --- commands ---

    def drive(self, steer, throttle):
        """steer/throttle in -1..+1. Re-sent every DRIVE_RESEND_INTERVAL
        even when unchanged, so the firmware's failsafe doesn't fire."""
        v1, v2 = ev(steer), ev(throttle)
        now = time.time()
        if (v1, v2) != self._last_drive or (now - self._last_drive_at) > DRIVE_RESEND_INTERVAL:
            self._last_drive = (v1, v2)
            self._last_drive_at = now
            self._enqueue(CMD_DRIVE, v1, v2)

    def stop(self, burst=4):
        """Neutral, sent immediately, several times.

        Deliberately bypasses the queue. Two reasons, both found the hard
        way: the sender collapses a batch of drive packets down to the
        newest one, which would swallow all but one of the burst; and a
        stop that waits its turn behind a backlog is not a stop. This also
        drops whatever drive commands were still queued, so a stale
        steering angle can't be transmitted after the stop."""
        self._drain()
        self._last_drive = (NEUTRAL_US, NEUTRAL_US)
        self._last_drive_at = time.time()
        for _ in range(burst):
            self._send_now(CMD_DRIVE, NEUTRAL_US, NEUTRAL_US)

    def _drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _send_now(self, cmd, v1, v2):
        if self._sock is None:
            return
        with self._lock:
            msg = f"{self._seq},{cmd},{v1},{v2}".encode()
            self._seq = (self._seq + 1) % SEQ_LIMIT
        try:
            self._sock.sendto(msg, (self.esp_ip, self.port))
            self.packets_sent += 1
        except OSError:
            pass

    def light(self, value):
        value = int(max(0, min(MAX_LIGHT, value)))
        now = time.time()
        if value != self._last_light or (now - self._last_light_at) > LIGHT_RESEND_INTERVAL:
            self._last_light = value
            self._last_light_at = now
            # Twice: a lost light packet would otherwise stay lost for a
            # whole second, and it is visible to the user.
            self._enqueue(CMD_LIGHT, value, value, burst=2)

    def reboot(self):
        self._enqueue(CMD_REBOOT, 3, 3, burst=2)

    def set_framesize(self, name):
        cmd = CMD_FRAMESIZE.get(name.lower())
        if cmd is None:
            raise ValueError(f"unknown framesize '{name}' (have: {', '.join(CMD_FRAMESIZE)})")
        self._enqueue(cmd, cmd, cmd, burst=2)

    # --- plumbing ---

    def _enqueue(self, cmd, v1, v2, burst=1):
        for _ in range(burst):
            with self._lock:
                msg = f"{self._seq},{cmd},{v1},{v2}".encode()
                self._seq = (self._seq + 1) % SEQ_LIMIT
            try:
                self._q.put_nowait(msg)
            except queue.Full:
                # A full queue means the sender is starved; the newest
                # drive command matters more than the backlog, so drop the
                # oldest rather than this one.
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass

    def _sender_loop(self):
        while self._running:
            try:
                first = self._q.get(timeout=COMMAND_TX_INTERVAL)
            except queue.Empty:
                continue

            batch = [first]
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break

            # Collapse the batch to ONE drive packet — the newest. Sending
            # a queued backlog of stale steering angles makes the car chase
            # positions it has already left.
            out, latest_drive = [], None
            for msg in batch:
                try:
                    if int(msg.split(b",")[1]) == CMD_DRIVE:
                        latest_drive = msg
                        continue
                except (IndexError, ValueError):
                    pass
                out.append(msg)
            if latest_drive is not None:
                out.append(latest_drive)

            for msg in out:
                try:
                    self._sock.sendto(msg, (self.esp_ip, self.port))
                    self.packets_sent += 1
                except OSError:
                    pass
