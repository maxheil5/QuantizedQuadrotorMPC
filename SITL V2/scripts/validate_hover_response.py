#!/usr/bin/env python3
"""Validate learned hover behavior against ground-truth odometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _import_ros() -> tuple[Any, Any, Any]:
    try:
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "rospy is required. Source /opt/ros/noetic/setup.bash and your workspace first."
        ) from exc
    return rospy, PoseStamped, Odometry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _v2_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


@dataclass(frozen=True)
class HoverValidationConfig:
    target_x_m: float = 0.0
    target_y_m: float = 0.0
    target_z_m: float = 1.0
    target_yaw_rad: float = 0.0
    duration_s: float = 15.0
    publish_rate_hz: float = 5.0
    settle_window_s: float = 5.0
    z_tolerance_m: float = 0.15
    xy_tolerance_m: float = 0.30
    odometry_topic: str = "/firefly/ground_truth/odometry"
    pose_topic: str = "/firefly/command/pose"
    output_root: str | None = None


@dataclass(frozen=True)
class Sample:
    t_s: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_hover_validation")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, samples: list[Sample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(samples[0]).keys()) if samples else list(asdict(Sample(0, 0, 0, 0, 0, 0, 0)).keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


def _rms(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def _latest_subset(samples: list[Sample], window_s: float) -> list[Sample]:
    if not samples:
        return []
    cutoff = samples[-1].t_s - window_s
    return [sample for sample in samples if sample.t_s >= cutoff]


def _time_to_first_z_band(samples: list[Sample], target_z_m: float, tolerance_m: float) -> float | None:
    for sample in samples:
        if abs(sample.z_m - target_z_m) <= tolerance_m:
            return sample.t_s
    return None


def _environment_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
    }


def _make_pose_message(PoseStamped: Any, config: HoverValidationConfig) -> Any:
    message = PoseStamped()
    message.header.frame_id = "world"
    message.pose.position.x = config.target_x_m
    message.pose.position.y = config.target_y_m
    message.pose.position.z = config.target_z_m
    message.pose.orientation.x = 0.0
    message.pose.orientation.y = 0.0
    message.pose.orientation.z = math.sin(config.target_yaw_rad / 2.0)
    message.pose.orientation.w = math.cos(config.target_yaw_rad / 2.0)
    return message


def _compute_metrics(samples: list[Sample], config: HoverValidationConfig) -> dict[str, Any]:
    if not samples:
        return {
            "success": False,
            "reason": "no_samples",
        }

    xy_errors = [
        math.hypot(sample.x_m - config.target_x_m, sample.y_m - config.target_y_m)
        for sample in samples
    ]
    z_errors = [sample.z_m - config.target_z_m for sample in samples]
    final = samples[-1]
    tail = _latest_subset(samples, min(config.settle_window_s, config.duration_s))
    tail_xy_errors = [
        math.hypot(sample.x_m - config.target_x_m, sample.y_m - config.target_y_m)
        for sample in tail
    ]
    tail_z_errors = [sample.z_m - config.target_z_m for sample in tail]

    mean_tail_z_error = sum(tail_z_errors) / len(tail_z_errors) if tail_z_errors else math.nan
    mean_tail_xy_error = sum(tail_xy_errors) / len(tail_xy_errors) if tail_xy_errors else math.nan

    success = (
        abs(final.z_m - config.target_z_m) <= config.z_tolerance_m
        and max(xy_errors) <= config.xy_tolerance_m
        and abs(mean_tail_z_error) <= config.z_tolerance_m
        and mean_tail_xy_error <= config.xy_tolerance_m
    )

    return {
        "success": success,
        "sample_count": len(samples),
        "duration_s": samples[-1].t_s,
        "target": {
            "x_m": config.target_x_m,
            "y_m": config.target_y_m,
            "z_m": config.target_z_m,
            "yaw_rad": config.target_yaw_rad,
        },
        "initial_position_m": {
            "x": samples[0].x_m,
            "y": samples[0].y_m,
            "z": samples[0].z_m,
        },
        "final_position_m": {
            "x": final.x_m,
            "y": final.y_m,
            "z": final.z_m,
        },
        "position_rmse_m": {
            "xy": _rms(xy_errors),
            "z": _rms(z_errors),
        },
        "position_error_tail_mean_m": {
            "xy": mean_tail_xy_error,
            "z": mean_tail_z_error,
        },
        "max_position_error_m": {
            "xy": max(xy_errors),
            "z": max(abs(value) for value in z_errors),
        },
        "time_to_first_z_band_s": _time_to_first_z_band(samples, config.target_z_m, config.z_tolerance_m),
        "tolerances": {
            "xy_m": config.xy_tolerance_m,
            "z_m": config.z_tolerance_m,
        },
    }


def run_validation(config: HoverValidationConfig) -> Path:
    rospy, PoseStamped, Odometry = _import_ros()

    if not rospy.get_param("/use_sim_time", False):
        rospy.logwarn("/use_sim_time is false. This validator expects Gazebo sim time.")

    output_root = Path(config.output_root) if config.output_root else (_v2_root() / "results" / "runtime_logs" / "hover_validation")
    run_dir = output_root / _timestamp_run_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_root = _repo_root()
    samples: list[Sample] = []
    latest_odometry: Any | None = None

    def _handle_odometry(message: Any) -> None:
        nonlocal latest_odometry
        latest_odometry = message

    rospy.init_node("hover_validation", anonymous=True)
    pose_pub = rospy.Publisher(config.pose_topic, PoseStamped, queue_size=1)
    odom_sub = rospy.Subscriber(config.odometry_topic, Odometry, _handle_odometry, queue_size=10)
    del odom_sub

    pose_message = _make_pose_message(PoseStamped, config)
    pose_publish_period_s = 1.0 / config.publish_rate_hz

    wait_start = time.monotonic()
    while latest_odometry is None and not rospy.is_shutdown():
        pose_message.header.stamp = rospy.Time.now()
        pose_pub.publish(pose_message)
        rospy.sleep(min(0.1, pose_publish_period_s))
        if time.monotonic() - wait_start > 10.0:
            raise SystemExit(f"Timed out waiting for odometry on {config.odometry_topic}.")

    monotonic_start = time.monotonic()
    next_publish_time = monotonic_start
    while not rospy.is_shutdown():
        now = time.monotonic()
        elapsed = now - monotonic_start
        if elapsed > config.duration_s:
            break

        if now >= next_publish_time:
            pose_message.header.stamp = rospy.Time.now()
            pose_pub.publish(pose_message)
            next_publish_time += pose_publish_period_s

        if latest_odometry is not None:
            samples.append(
                Sample(
                    t_s=elapsed,
                    x_m=float(latest_odometry.pose.pose.position.x),
                    y_m=float(latest_odometry.pose.pose.position.y),
                    z_m=float(latest_odometry.pose.pose.position.z),
                    vx_mps=float(latest_odometry.twist.twist.linear.x),
                    vy_mps=float(latest_odometry.twist.twist.linear.y),
                    vz_mps=float(latest_odometry.twist.twist.linear.z),
                )
            )
        rospy.sleep(0.02)

    metrics = _compute_metrics(samples, config)

    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "environment.json", _environment_payload(repo_root))
    _write_csv(run_dir / "odometry.csv", samples)

    print(f"hover_validation_run_dir={run_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return run_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-x", type=float, default=HoverValidationConfig.target_x_m)
    parser.add_argument("--target-y", type=float, default=HoverValidationConfig.target_y_m)
    parser.add_argument("--target-z", type=float, default=HoverValidationConfig.target_z_m)
    parser.add_argument("--target-yaw", type=float, default=HoverValidationConfig.target_yaw_rad)
    parser.add_argument("--duration", type=float, default=HoverValidationConfig.duration_s)
    parser.add_argument("--publish-rate", type=float, default=HoverValidationConfig.publish_rate_hz)
    parser.add_argument("--settle-window", type=float, default=HoverValidationConfig.settle_window_s)
    parser.add_argument("--z-tolerance", type=float, default=HoverValidationConfig.z_tolerance_m)
    parser.add_argument("--xy-tolerance", type=float, default=HoverValidationConfig.xy_tolerance_m)
    parser.add_argument("--odometry-topic", default=HoverValidationConfig.odometry_topic)
    parser.add_argument("--pose-topic", default=HoverValidationConfig.pose_topic)
    parser.add_argument("--output-root")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    config = HoverValidationConfig(
        target_x_m=args.target_x,
        target_y_m=args.target_y,
        target_z_m=args.target_z,
        target_yaw_rad=args.target_yaw,
        duration_s=args.duration,
        publish_rate_hz=args.publish_rate,
        settle_window_s=args.settle_window,
        z_tolerance_m=args.z_tolerance,
        xy_tolerance_m=args.xy_tolerance,
        odometry_topic=args.odometry_topic,
        pose_topic=args.pose_topic,
        output_root=args.output_root,
    )
    run_validation(config)


if __name__ == "__main__":
    main()
