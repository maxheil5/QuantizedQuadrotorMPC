from __future__ import annotations

from quantized_quadrotor_sitl.utils.host import runtime_host_snapshot


def test_runtime_host_snapshot_includes_expected_keys():
    snapshot = runtime_host_snapshot(headless=True)

    assert snapshot["headless"] is True
    assert "hostname" in snapshot
    assert "cpu_model" in snapshot
    assert "cpu_count" in snapshot
    assert "load_average_1m" in snapshot
    assert "load_average_5m" in snapshot
    assert "load_average_15m" in snapshot
    assert "available_memory_bytes" in snapshot
