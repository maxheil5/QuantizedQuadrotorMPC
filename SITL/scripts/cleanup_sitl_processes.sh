#!/usr/bin/env bash
set -euo pipefail

pkill -f MicroXRCEAgent >/dev/null 2>&1 || true
pkill -f px4 >/dev/null 2>&1 || true
pkill -f "gz sim" >/dev/null 2>&1 || true
pkill -f quantized_quadrotor_sitl.ros.telemetry_adapter_node >/dev/null 2>&1 || true
pkill -f quantized_quadrotor_sitl.ros.controller_node >/dev/null 2>&1 || true
pkill -f quantized_quadrotor_sitl.tools.gcs_heartbeat >/dev/null 2>&1 || true
