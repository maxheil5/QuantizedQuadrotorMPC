from __future__ import annotations

from pathlib import Path


def test_standard_anchor_trial_script_uses_standard_anchor_config_and_backfill():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_sitl_standard_anchor_trial.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "sitl_runtime_sitl_retrain_edmd_anchor_h8.yaml" in script_text
    assert "handle_interrupt" in script_text
    assert "waiting for the SITL runner to finalize artifacts" in script_text
    assert "SITL_PRINT_RUN_OUTPUTS=0" in script_text
    assert "--run-dir \"${run_dir}\"" in script_text
    assert "--base-dir \"${ROOT_DIR}\"" in script_text
    assert "print_run_output_summary" in script_text
    assert "Canonical run folder name:" in script_text
    assert "Use this exact folder name for upload and analysis. Do not rename it." in script_text
    assert "Stored files:" in script_text
    assert "Required files: complete" in script_text
    assert "Missing required files:" in script_text
    assert "Root milestone summary:" in script_text
    assert "Root milestone summary row for this run: present" in script_text
    assert "WARNING: root milestone_summary.csv was not updated for this run." in script_text
    assert "Host cleanup before rerun after invalid-runtime results:" in script_text
    assert "bash ./scripts/cleanup_sitl_processes.sh" in script_text
    assert "quantized_quadrotor_sitl.experiments.sitl_postrun_analysis" in script_text
    assert "milestone_summary_contains_run" in script_text
