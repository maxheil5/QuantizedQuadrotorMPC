from __future__ import annotations

from pathlib import Path


def test_run_sitl_experiment_script_supports_gazebo_video_recording():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sitl_experiment.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "AUTO_GAZEBO_VIDEO_RECORDING" in script_text
    assert "SITL_PRINT_RUN_OUTPUTS" in script_text
    assert "maybe_start_gazebo_video_recording" in script_text
    assert "stop_gazebo_video_recording" in script_text
    assert "stop_simulation_stack" in script_text
    assert "setsid env" in script_text
    assert "Stopping PX4/Gazebo simulation stack..." in script_text
    assert "request_graceful_shutdown" in script_text
    assert "Received ${signal}; stopping the simulation and finalizing run artifacts..." in script_text
    assert "Second ${signal} received; forcing immediate shutdown." in script_text
    assert "finalize_gazebo_video_recording" in script_text
    assert "print_run_output_summary" in script_text
    assert "xdotool search --onlyvisible" in script_text
    assert "VIDEO_RECORDER_CAPTURE_SOURCE" in script_text
    assert "PREVIOUS_ACTIVE_RUN_DIR" in script_text
    assert "RUN_LAUNCH_EPOCH" in script_text
    assert "metadata_path.stat().st_mtime >= launch_epoch" in script_text
    assert 'setsar=1' in script_text
    assert "Stored files:" in script_text
    assert "gazebo_recording.mp4" in script_text
    assert ".recording.mkv" in script_text
    assert "ffmpeg" in script_text


def test_setup_linux_script_installs_video_recording_dependencies():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "setup_linux.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "ffmpeg" in script_text
    assert "xdotool" in script_text
    assert "x11-utils" in script_text
