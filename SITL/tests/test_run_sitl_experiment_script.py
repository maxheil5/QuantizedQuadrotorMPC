from __future__ import annotations

from pathlib import Path


def test_run_sitl_experiment_script_supports_graceful_shutdown_and_summary():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sitl_experiment.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "AUTO_DRIFT_ANALYSIS" in script_text
    assert "SITL_PRINT_RUN_OUTPUTS" in script_text
    assert "stop_simulation_stack" in script_text
    assert "setsid env" in script_text
    assert "Stopping PX4/Gazebo simulation stack..." in script_text
    assert "request_graceful_shutdown" in script_text
    assert "Received ${signal}; stopping the simulation and finalizing run artifacts. Do not press Ctrl-C again unless you want to abort cleanup." in script_text
    assert "Second ${signal} received; forcing immediate shutdown." in script_text
    assert "print_run_output_summary" in script_text
    assert "Canonical run folder name:" in script_text
    assert "Use this exact folder name for upload and analysis. Do not rename it." in script_text
    assert "Stored files:" in script_text
    assert "Required files: complete" in script_text
    assert "Missing required files:" in script_text
    assert "Host cleanup before rerun after invalid-runtime results:" in script_text
    assert "bash ./scripts/cleanup_sitl_processes.sh" in script_text
    assert "quantized_quadrotor_sitl.experiments.sitl_postrun_analysis" in script_text
