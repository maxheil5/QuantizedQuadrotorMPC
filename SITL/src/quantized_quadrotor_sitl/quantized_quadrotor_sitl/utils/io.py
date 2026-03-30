from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_datetime_to_nearest_minutes(value: datetime, minutes: int) -> datetime:
    if minutes <= 0:
        raise ValueError("minutes must be positive")
    rounded = value
    discard = timedelta(
        minutes=rounded.minute % minutes,
        seconds=rounded.second,
        microseconds=rounded.microsecond,
    )
    rounded -= discard
    if discard >= timedelta(minutes=minutes / 2):
        rounded += timedelta(minutes=minutes)
    return rounded


def format_sitl_run_name(value: datetime | None = None) -> str:
    current = value if value is not None else datetime.now().astimezone()
    rounded = round_datetime_to_nearest_minutes(current, 10)
    return f"{rounded.month}-{rounded.day}-{rounded.year % 100:02d}_{rounded.hour:02d}{rounded.minute:02d}"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _refresh_latest_symlink(latest_path: Path, run_dir: Path) -> None:
    if latest_path.is_symlink() or latest_path.is_file():
        latest_path.unlink()
    elif latest_path.exists():
        legacy_path = _unique_path(latest_path.with_name("latest_legacy"))
        latest_path.rename(legacy_path)
    latest_path.symlink_to(run_dir.name)


def create_sitl_results_directory(results_dir: Path, timestamp: datetime | None = None) -> Path:
    resolved = results_dir
    if resolved.name != "latest":
        return ensure_dir(resolved)
    root = ensure_dir(resolved.parent)
    run_dir = ensure_dir(_unique_path(root / format_sitl_run_name(timestamp)))
    _refresh_latest_symlink(root / "latest", run_dir)
    return run_dir


def create_run_directory(root: Path, profile_name: str, tag: str | None = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{tag}" if tag else ""
    run_dir = ensure_dir(root / profile_name / f"{timestamp}{suffix}")
    latest = root / profile_name / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez(path, **arrays)
