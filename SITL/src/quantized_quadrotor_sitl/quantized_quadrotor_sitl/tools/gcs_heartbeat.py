from __future__ import annotations

import argparse
import signal
import time

from pymavlink import mavutil


RUNNING = True


def _handle_signal(signum, frame):
    del signum, frame
    global RUNNING
    RUNNING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send MAVLink GCS heartbeats to PX4 SITL.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18570)
    parser.add_argument("--rate-hz", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    heartbeat_period = 1.0 / max(args.rate_hz, 1.0e-6)
    print(
        f"Starting GCS heartbeat helper on udpout:{args.host}:{args.port} at {args.rate_hz:.2f} Hz",
        flush=True,
    )
    connection = mavutil.mavlink_connection(
        f"udpout:{args.host}:{args.port}",
        source_system=255,
        source_component=0,
    )

    while RUNNING:
        connection.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        time.sleep(heartbeat_period)

    print("Stopping GCS heartbeat helper", flush=True)


if __name__ == "__main__":
    main()
