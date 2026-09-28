"""
Long-run link monitor: WHY does the video get worse the longer it runs?
=======================================================================
"Stutters sometimes, and more the longer the camera is left on" has at
least five different causes, and a single --bench run cannot tell them
apart because they all look like "fewer frames per second". What tells
them apart is the TREND of several numbers over 10-30 minutes:

    frame KB grows, KB/s flat        -> the sensor. JPEGs get bigger as the
                                        OV2640 heats up (more noise), and a
                                        link that was carrying N fps of 20 KB
                                        frames now carries fewer 30 KB ones.
    KB/s falls, frame KB flat,       -> the radio. Same picture, fewer bytes
    ping rises / loses replies          delivered; the ESP32 heating up or a
                                        PSU sagging does exactly this.
    heap_largest shrinks steadily    -> firmware heap fragmentation; the
                                        board will eventually stall or reboot.
    uptime_s goes backwards          -> the board rebooted (power / watchdog).
    gap p95 rises, everything above  -> this PC. Something else is eating
    stays flat                          the core (Task Manager).

So this opens the stream exactly like run.py does, then every --interval
seconds prints one line: what the PC received, how it was spaced, what the
board says about itself (/status: RSSI, heap, temperature, uptime), and a
ping to the board. It writes the same rows to a CSV and finishes with a
verdict comparing the first windows against the last ones.

    python link_monitor.py --esp-ip 192.168.8.191                 # 20 min
    python link_monitor.py --esp-ip 192.168.8.191 --minutes 40 --interval 15
    python link_monitor.py --esp-ip 192.168.8.191 --quality 16 --ae-level -1

--detect runs the same cone detection + lane finder as run.py --detect on
every frame that arrives (and --show also draws the windows), and adds two
columns: the fps the PC actually processed and the detect time per frame.
If proc fps tracks the received fps and det ms stays far below the frame
interval, detection is not what slows the picture down.

The health fields (heap / temp / uptime) need the current
firmware/esp32cam_robot/ sketch; an older board or the stock
CameraWebServer sketch shows '-' for them and the rest still works.

Ctrl+C ends the run early and still prints the verdict.
"""

import argparse
import csv
import re
import statistics
import subprocess
import sys
import threading
import time

from camera_control import RESOLUTIONS, read_status
from camera_setup import FIRST_FRAME_TIMEOUT, prepare_camera
from mjpeg_video_source import MJPEGVideoSource

# /status keys copied into every row when the firmware provides them.
HEALTH_KEYS = ("rssi", "uptime_s", "heap_free", "heap_largest", "psram_free",
               "temp_c", "stream_frames", "quality", "framesize")

COLUMNS = ["t_s", "fps", "kbps", "frame_kb", "gap_p50_ms", "gap_p95_ms",
           "gap_max_ms", "reconnects", "ping_avg_ms", "ping_max_ms",
           "ping_lost", "proc_fps", "det_ms_avg", "det_ms_max"] + list(HEALTH_KEYS)


# ============================================================
# MEASUREMENTS
# ============================================================

def ping(ip, count=3, timeout_ms=1000):
    """(avg_ms, max_ms, lost) from the per-reply lines of Windows/Unix ping.

    The per-reply lines, not the summary: on Windows a dead host makes the
    PC answer 'Destination host unreachable' for itself, and the summary
    then counts those as received with 0% loss."""
    if sys.platform.startswith("win"):
        cmd = ["ping", "-n", str(count), "-w", str(timeout_ms), ip]
    else:
        cmd = ["ping", "-c", str(count), "-W", str(max(1, timeout_ms // 1000)), ip]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=count * (timeout_ms / 1000.0) + 3).stdout
    except Exception:
        return None, None, count
    times = [float(m) for m in re.findall(r"time[=<]\s*([\d.]+)\s*ms", out)]
    if not times:
        return None, None, count
    return statistics.mean(times), max(times), count - len(times)


def gap_stats(times):
    """Frame spacing in ms over a window's worth of arrival timestamps:
    (median, 95th percentile, worst). None if fewer than two frames."""
    if len(times) < 2:
        return None, None, None
    gaps = sorted((b - a) * 1000.0 for a, b in zip(times, times[1:]))
    return (statistics.median(gaps),
            gaps[min(len(gaps) - 1, int(len(gaps) * 0.95))],
            gaps[-1])


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


class Processor:
    """Runs the run.py --detect pipeline on every new frame, on its own
    thread, and keeps per-window counters. Gated on frame_count exactly
    like run.py, so it costs nothing while no new frame has arrived."""

    def __init__(self, src, config_path, show):
        from cone_detection import ConeDetector
        from lane_center import LaneFinder, draw_lane
        self.src = src
        self.show = show
        self.detector = ConeDetector(config_path=config_path)
        self.lane = LaneFinder(self.detector.cfg)
        self.draw_lane = draw_lane
        self.lock = threading.Lock()
        self.processed = 0
        self.det_ms = []
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def take(self):
        """(processed frames, [detect ms]) since the last call."""
        with self.lock:
            n, ms = self.processed, self.det_ms
            self.processed, self.det_ms = 0, []
        return n, ms

    def _loop(self):
        import cv2
        last = -1
        while self._running:
            count = self.src.frame_count
            if count == last:
                # Same idle path as run.py: waitKey when a window is up
                # (it also pumps the GUI), a plain sleep otherwise.
                if self.show:
                    cv2.waitKey(5)
                else:
                    time.sleep(0.005)
                continue
            last = count
            frame = self.src.get_frame()
            if frame is None:
                continue
            t = time.perf_counter()
            cones, mask, display = self.detector.detect(frame)
            result = self.lane.update(cones, frame.shape[1], frame.shape[0])
            self.draw_lane(display, result)
            ms = (time.perf_counter() - t) * 1000.0
            if self.show:
                cv2.imshow("Mask", mask)
                cv2.imshow("ESP32-CAM Live Feed", display)
                cv2.waitKey(1)
            with self.lock:
                self.processed += 1
                self.det_ms.append(ms)
        if self.show:
            cv2.destroyAllWindows()


# ============================================================
# VERDICT
# ============================================================

def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def verdict(rows):
    """Compare the first few windows with the last few and say what moved.

    Ratios rather than absolutes: what matters is which numbers drifted
    over the session, not what any of them is."""
    if len(rows) < 4:
        print("\nToo few windows for a trend. Run longer (--minutes) or "
              "shorten --interval.")
        return
    n = max(2, len(rows) // 4)
    head, tail = rows[:n], rows[-n:]
    print(f"\n=== Verdict: first {n} windows vs last {n} windows ===")

    def trend(key, label, spec=".1f", unit=""):
        a, b = _mean(head, key), _mean(tail, key)
        if a is None or b is None:
            return None
        pct = (b - a) / a * 100 if a else 0.0
        print(f"  {label:<18} {a:{spec}}{unit} -> {b:{spec}}{unit}  ({pct:+.0f}%)")
        return pct

    fps_pct = trend("fps", "fps")
    trend("proc_fps", "processed fps")
    trend("det_ms_avg", "detect ms", ".1f", " ms")
    kbps_pct = trend("kbps", "KB/s", ".0f")
    frame_pct = trend("frame_kb", "frame size", ".1f", " KB")
    gap95_pct = trend("gap_p95_ms", "gap p95", ".0f", " ms")
    ping_pct = trend("ping_avg_ms", "ping avg", ".0f", " ms")
    lost = sum(r["ping_lost"] for r in rows if r.get("ping_lost") is not None)
    recon = rows[-1]["reconnects"] - rows[0]["reconnects"]
    trend("rssi", "RSSI", ".0f", " dBm")
    trend("temp_c", "chip temp", ".0f", " C")
    heap_pct = trend("heap_largest", "heap largest", ".0f", " B")
    trend("heap_free", "heap free", ".0f", " B")

    uptimes = [r["uptime_s"] for r in rows if r.get("uptime_s") is not None]
    rebooted = any(b < a for a, b in zip(uptimes, uptimes[1:]))

    print()
    findings = []
    recv, proc = _mean(rows, "fps"), _mean(rows, "proc_fps")
    det = _mean(rows, "det_ms_avg")
    if proc is not None and recv and det is not None:
        if proc < recv * 0.9:
            findings.append(f"Detection kept up with only {proc / recv * 100:.0f}% of the "
                            f"frames received ({det:.1f} ms per frame). The PC is the "
                            "bottleneck here — lower the resolution or the detector's "
                            "morphology kernel.")
        else:
            findings.append(f"Detection processed every frame received ({det:.1f} ms per "
                            "frame against a frame interval of "
                            f"{1000 / recv:.0f} ms) — it is not what slows the picture.")
    if rebooted:
        findings.append("The board REBOOTED during the run (uptime went backwards). "
                        "Power supply first: 5V >= 1A on a short thick cable.")
    if recon:
        findings.append(f"The stream reconnected {recon}x — TCP gave up on the "
                        "socket. That is the radio or a reboot, not the sensor.")
    if fps_pct is not None and fps_pct > -10 and (gap95_pct is None or gap95_pct < 30):
        findings.append("Nothing got worse over this run. If it still looked "
                        "worse on screen, the drift is in what runs on top "
                        "(--detect, another program) — run again with the "
                        "window open beside it.")
    else:
        if frame_pct is not None and frame_pct > 15 and (kbps_pct is None or kbps_pct > -10):
            findings.append("Frames GREW while KB/s stayed put: the link carried the "
                            "same bytes but each frame cost more. That is the sensor "
                            "(noise rises with temperature; JPEG size follows). Try "
                            "--quality 16-20, shade/cool the board, and re-check "
                            "AWB/AEC lock left the gain at a sane value.")
        if kbps_pct is not None and kbps_pct < -15 and (frame_pct is None or frame_pct < 10):
            findings.append("KB/s FELL with the same frame size: the radio is "
                            "delivering less. " +
                            ("Ping got worse too, so it is the link, not the board's "
                             "sender. " if (ping_pct or 0) > 30 or lost else "") +
                            "Order of suspects: antenna jumper, power supply (a "
                            "heating ESP32 at 19.5 dBm TX browns out first), "
                            "distance, the AP's 2.4 GHz channel.")
        if heap_pct is not None and heap_pct < -25:
            findings.append("Largest free heap block shrank steadily: the firmware "
                            "is fragmenting its heap. Expect a stall or watchdog "
                            "reboot eventually; this is a firmware bug to chase "
                            "(String use in loopcommand, httpd allocations).")
        if gap95_pct is not None and gap95_pct > 30 and (kbps_pct or 0) > -10 and (frame_pct or 0) < 10:
            findings.append("Frame spacing got more uneven while bytes, frame size "
                            "and ping all held: that is this PC. Check what else "
                            "is running (Task Manager -> CPU) and whether Windows "
                            "throttled the WiFi adapter's power.")
        if not findings:
            findings.append("fps dropped but no single number explains it cleanly. "
                            "Look at the CSV: which column moved first?")
    for f in findings:
        print(" *", f)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Watch the MJPEG link for minutes and report what degrades")
    parser.add_argument("--esp-ip", required=True, help="Camera IP")
    parser.add_argument("--minutes", type=float, default=20, help="How long to watch (default 20)")
    parser.add_argument("--interval", type=float, default=10, help="Seconds per row (default 10)")
    parser.add_argument("--framesize", default="vga", choices=sorted(RESOLUTIONS))
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument("--ae-level", type=int, default=None, choices=[-2, -1, 0, 1, 2])
    parser.add_argument("--no-lock", action="store_true",
                        help="Leave AWB/AEC/AGC running (locked by default, as run.py does)")
    parser.add_argument("--no-prepare", action="store_true",
                        help="Do not touch camera settings; watch it as it is")
    parser.add_argument("--detect", action="store_true",
                        help="Run cone detection + lane finder on every frame, like run.py --detect")
    parser.add_argument("--show", action="store_true",
                        help="With --detect: also draw the two windows, so their cost is included")
    parser.add_argument("--config", default="cone_config.json", help="Detector config (with --detect)")
    parser.add_argument("--csv", default=None,
                        help="Where to write the rows (default link_monitor_<time>.csv)")
    args = parser.parse_args()

    esp_ip = args.esp_ip
    if not args.no_prepare:
        esp_ip = prepare_camera("webserver", esp_ip, args.framesize,
                                lock=not args.no_lock, quality=args.quality,
                                ae_level=args.ae_level)
        if esp_ip is None:
            return 1

    src = MJPEGVideoSource(esp_ip)
    src.start()
    print("Waiting for the first frame ...")
    waited_until = time.time() + FIRST_FRAME_TIMEOUT
    while src.frame_count == 0:
        if time.time() > waited_until:
            src.stop()
            print(f"No frames after {FIRST_FRAME_TIMEOUT:.0f}s. Close any browser tab on the "
                  "camera page; the ESP32-CAM serves ONE stream client.")
            return 1
        time.sleep(0.05)

    proc = None
    if args.detect:
        proc = Processor(src, args.config, args.show)
        proc.start()

    csv_path = args.csv or f"link_monitor_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Watching {esp_ip} for {args.minutes:g} min, one row per {args.interval:g}s "
          f"-> {csv_path}. Ctrl+C stops early.\n")
    print(f"{'t':>5} {'fps':>5} {'KB/s':>5} {'KB/f':>5} | {'gap p50':>7} {'p95':>5} {'max':>5} | "
          f"{'ping':>5} {'max':>5} {'lost':>4} | "
          + (f"{'proc':>5} {'det ms':>6} {'max':>5} | " if proc else "")
          + f"{'rssi':>4} {'temp':>4} {'heapMax':>7} {'uptime':>6} rec")

    rows = []
    t0 = time.perf_counter()
    end_at = t0 + args.minutes * 60
    f_prev, b_prev, t_prev = src.frame_count, src.bytes_received, t0
    arrivals_prev_len = 0

    try:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            while time.perf_counter() < end_at:
                time.sleep(args.interval)
                now = time.perf_counter()
                f_now, b_now = src.frame_count, src.bytes_received
                elapsed = now - t_prev
                frames = f_now - f_prev
                nbytes = b_now - b_prev

                # Arrival times of just this window's frames. The deque is
                # bounded, so take the tail; frames beyond its capacity in
                # one window would mean >400 fps, which this link cannot do.
                times = list(src.arrival_times)
                times = [t for t in times if t > t_prev]
                p50, p95, pmax = gap_stats(times)

                status = read_status(esp_ip, timeout=2.0) or {}
                pavg, pmax_ping, lost = ping(esp_ip)

                row = {
                    "t_s": round(now - t0),
                    "fps": frames / elapsed,
                    "kbps": nbytes / elapsed / 1024.0,
                    "frame_kb": (nbytes / frames / 1024.0) if frames else None,
                    "gap_p50_ms": p50, "gap_p95_ms": p95, "gap_max_ms": pmax,
                    "reconnects": src.reconnects,
                    "ping_avg_ms": pavg, "ping_max_ms": pmax_ping, "ping_lost": lost,
                    "proc_fps": None, "det_ms_avg": None, "det_ms_max": None,
                }
                if proc is not None:
                    n_proc, det_ms = proc.take()
                    row["proc_fps"] = n_proc / elapsed
                    if det_ms:
                        row["det_ms_avg"] = statistics.mean(det_ms)
                        row["det_ms_max"] = max(det_ms)
                for k in HEALTH_KEYS:
                    row[k] = status.get(k)
                rows.append(row)
                writer.writerow(row)
                fh.flush()

                print(f"{row['t_s']:5d} {row['fps']:5.1f} {row['kbps']:5.0f} {fmt(row['frame_kb'], '5.1f')} | "
                      f"{fmt(p50, '7.0f')} {fmt(p95, '5.0f')} {fmt(pmax, '5.0f')} | "
                      f"{fmt(pavg, '5.0f')} {fmt(pmax_ping, '5.0f')} {lost:4d} | "
                      + (f"{row['proc_fps']:5.1f} {fmt(row['det_ms_avg'], '6.1f')} "
                         f"{fmt(row['det_ms_max'], '5.0f')} | " if proc else "")
                      + f"{fmt(row['rssi'], '4')} {fmt(row['temp_c'], '4.0f')} "
                      f"{fmt(row['heap_largest'], '7')} {fmt(row['uptime_s'], '6')} {src.reconnects}")

                f_prev, b_prev, t_prev = f_now, b_now, now
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        if proc is not None:
            proc.stop()
        src.stop()

    verdict(rows)
    print(f"\nRows written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
