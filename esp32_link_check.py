"""
Robot link check (why is there no video?)
===========================================
Answers one question the normal tools cannot: when nothing shows up, is it
because the ESP never heard us, or because it is answering into a wall?

    python esp32_link_check.py --esp-ip 172.20.10.3
    python esp32_link_check.py                        # broadcast instead

WHAT IT ACTUALLY TESTS
----------------------
The robot firmware powers the sensor DOWN until a PC appears — that is
what `[Power] Sensor Sleeping...` on the serial monitor means. It is
waiting, not broken. Any UDP packet arriving on port 44444 wakes it, and
it then streams to whatever IP that packet came from.

So there are three separate things that can be broken, and they need
different fixes. This listens on both ports at once to tell them apart:

  PC -> ESP  44444   our beacon           (nothing here means bad IP/subnet)
  ESP -> PC  44444   telemetry, 1 Hz      (arrives => the ESP sees us)
  ESP -> PC  44442   video chunks         (missing while telemetry arrives
                                           => a firewall is eating the video)

That last case is the one worth having a tool for. Telemetry and video
come back on different ports, so Windows Firewall can happily allow one
and drop the other, which looks exactly like a dead camera.

REQUIRES THE ROBOT FIRMWARE (firmware/esp32cam_robot/). CameraWebServer
speaks HTTP and knows nothing about any of these ports — use run.py
--source webserver for that one.
"""

import argparse
import select
import socket
import sys
import time

DISCOVERY_PORT = 44444
VIDEO_PORT = 44442
BEACON = b"PC_HERE,delivery:100"
BEACON_INTERVAL = 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Check the UDP link to the robot firmware and say what is broken")
    parser.add_argument("--esp-ip", default=None,
                        help="Robot IP (from the serial monitor). Omit to broadcast.")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="How long to listen (default 8)")
    args = parser.parse_args()

    disc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    disc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    disc.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # Bound, not just connected: the firmware sends its telemetry back to
    # this same port, and an unbound socket would never see it.
    disc.bind(("0.0.0.0", DISCOVERY_PORT))

    video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video.bind(("0.0.0.0", VIDEO_PORT))

    target = args.esp_ip or "<broadcast>"
    print(f"Beaconing to {target}:{DISCOVERY_PORT}, listening on "
          f"{DISCOVERY_PORT} (telemetry) and {VIDEO_PORT} (video) "
          f"for {args.seconds:g}s ...\n")

    telemetry, chunks, video_bytes = 0, 0, 0
    senders = set()
    last_telemetry = None
    deadline = time.time() + args.seconds
    next_beacon = 0.0

    try:
        while time.time() < deadline:
            now = time.time()
            if now >= next_beacon:
                try:
                    disc.sendto(BEACON, (target, DISCOVERY_PORT))
                except OSError as e:
                    print(f"Could not send the beacon: {e}")
                    print("Wrong subnet, or no route to that address.")
                    return 1
                next_beacon = now + BEACON_INTERVAL

            ready, _, _ = select.select([disc, video], [], [], 0.2)
            for s in ready:
                data, addr = s.recvfrom(2048)
                if s is video:
                    chunks += 1
                    video_bytes += len(data)
                    senders.add(addr[0])
                elif data != BEACON:      # our own broadcast comes back to us
                    telemetry += 1
                    senders.add(addr[0])
                    last_telemetry = (addr[0], data.decode("utf-8", "replace").strip())
    finally:
        disc.close()
        video.close()

    # --- verdict ---
    print(f"telemetry packets (port {DISCOVERY_PORT}): {telemetry}")
    print(f"video chunks      (port {VIDEO_PORT}): {chunks}"
          f"   ({video_bytes / 1024:.0f} KB)")
    if last_telemetry:
        print(f"last telemetry from {last_telemetry[0]}: {last_telemetry[1]}")
    print()

    if telemetry == 0 and chunks == 0:
        print("NOTHING came back. The ESP never heard the beacon, or its reply")
        print("never reached this PC. In order of likelihood:")
        print(f"  1. Wrong IP. Use the one the serial monitor printed"
              f"{'' if args.esp_ip else ' — and pass it with --esp-ip'}.")
        print("  2. This PC is not on the same network as the robot. On an")
        print("     iPhone hotspot both have to be joined to the SAME hotspot.")
        print("  3. Windows Firewall is blocking inbound UDP for python.exe.")
        print("     Allow python on PRIVATE networks (a hotspot usually gets")
        print("     classified as Public, where inbound UDP is blocked).")
        print("  4. Broadcast discovery does not work on some hotspots at all.")
        print("     Always pass --esp-ip there.")
        return 1

    if telemetry > 0 and chunks == 0:
        print("The ESP HEARD US — telemetry is arriving — but no video came back.")
        print(f"Only port {VIDEO_PORT} is blocked, which is almost always the")
        print("firewall: telemetry and video arrive on different ports, so one")
        print("can be allowed while the other is dropped.")
        print("  - Allow python.exe inbound UDP on the Private profile, or")
        print(f"  - open UDP {VIDEO_PORT} specifically.")
        print("Also check the serial monitor: it should have stopped saying")
        print("'Sensor Sleeping' and started printing '[Video] Sending frame'.")
        return 1

    if chunks > 0:
        fps_hint = ""
        if video_bytes:
            fps_hint = f", {video_bytes / 1024 / args.seconds:.0f} KB/s"
        print(f"LINK IS UP{fps_hint}. Video chunks are arriving, so the camera,")
        print("the network and the firewall are all fine.")
        print("Next:")
        ip = args.esp_ip or (sorted(senders)[0] if senders else "<robot ip>")
        print(f"  python run.py --source esp32 --esp-ip {ip}")
        print(f"  python run.py --source esp32 --esp-ip {ip} --bench 10")
        return 0

    print("Telemetry only, and no video after the wait — try a longer --seconds.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
