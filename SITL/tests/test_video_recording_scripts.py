from __future__ import annotations

from pathlib import Path


def test_run_sitl_experiment_script_supports_gazebo_video_recording():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sitl_experiment.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "AUTO_GAZEBO_VIDEO_RECORDING" in script_text
    assert "maybe_start_gazebo_video_recording" in script_text
    assert "gazebo_recording.mp4" in script_text
    assert "ffmpeg" in script_text


def test_setup_linux_script_installs_video_recording_dependencies():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "setup_linux.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "ffmpeg" in script_text
    assert "xdotool" in script_text
    assert "x11-utils" in script_text
