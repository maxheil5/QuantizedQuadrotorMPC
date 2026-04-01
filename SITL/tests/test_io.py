from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys


SITL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SITL_ROOT / "src" / "quantized_quadrotor_sitl"
for candidate in (SITL_ROOT, PACKAGE_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from quantized_quadrotor_sitl.utils.io import (  # noqa: E402
    create_sitl_results_directory,
    format_sitl_run_name,
)


def test_format_sitl_run_name_rounds_to_nearest_ten_minutes():
    assert format_sitl_run_name(datetime(2026, 3, 30, 12, 24, 59)) == "3-30-26_1220"
    assert format_sitl_run_name(datetime(2026, 3, 30, 12, 25, 0)) == "3-30-26_1230"
    assert format_sitl_run_name(datetime(2026, 3, 30, 23, 57, 0)) == "3-31-26_0000"
    assert format_sitl_run_name(datetime(2026, 4, 1, 14, 52, 0), seed_suffix=2141444) == "4-1-26_1450_1444"


def test_create_sitl_results_directory_creates_timestamped_run_and_latest_symlink(tmp_path: Path):
    results_dir = tmp_path / "results" / "sitl" / "latest"

    run_dir = create_sitl_results_directory(results_dir, datetime(2026, 3, 30, 12, 24, 59))

    assert run_dir.name == "3-30-26_1220"
    assert run_dir.is_dir()
    latest = results_dir.parent / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


def test_create_sitl_results_directory_appends_seed_suffix_when_requested(tmp_path: Path):
    results_dir = tmp_path / "results" / "sitl" / "latest"

    run_dir = create_sitl_results_directory(results_dir, datetime(2026, 4, 1, 14, 52, 0), seed_suffix=2141444)

    assert run_dir.name == "4-1-26_1450_1444"
    assert run_dir.is_dir()
    latest = results_dir.parent / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


def test_create_sitl_results_directory_preserves_existing_latest_directory_and_avoids_collisions(tmp_path: Path):
    root = tmp_path / "results" / "sitl"
    latest = root / "latest"
    latest.mkdir(parents=True)
    (latest / "runtime_log.csv").write_text("legacy\n", encoding="utf-8")

    first = create_sitl_results_directory(latest, datetime(2026, 3, 30, 12, 24, 59))
    second = create_sitl_results_directory(latest, datetime(2026, 3, 30, 12, 24, 59))

    assert first.name == "3-30-26_1220"
    assert second.name == "3-30-26_1220_01"
    assert (root / "latest_legacy").is_dir()
    assert latest.is_symlink()
    assert latest.resolve() == second.resolve()


def test_create_sitl_results_directory_preserves_seeded_names_when_avoiding_collisions(tmp_path: Path):
    root = tmp_path / "results" / "sitl"
    latest = root / "latest"

    first = create_sitl_results_directory(latest, datetime(2026, 4, 1, 14, 52, 0), seed_suffix=2141444)
    second = create_sitl_results_directory(latest, datetime(2026, 4, 1, 14, 52, 0), seed_suffix=2141444)

    assert first.name == "4-1-26_1450_1444"
    assert second.name == "4-1-26_1450_1444_01"
    assert latest.is_symlink()
    assert latest.resolve() == second.resolve()
