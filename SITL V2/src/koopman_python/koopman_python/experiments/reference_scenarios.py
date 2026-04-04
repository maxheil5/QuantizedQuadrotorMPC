"""Deterministic offline reference scenarios for V2 learned-MPC validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from koopman_python.dynamics.srb import pack_state


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_name: str
    duration_s: float
    reference_type: str
    position_m: np.ndarray | None = None
    start_m: np.ndarray | None = None
    end_m: np.ndarray | None = None
    center_m: np.ndarray | None = None
    radius_m: float | None = None
    angular_rate_rad_s: float | None = None
    yaw_rad: float | None = None
    yaw_mode: str | None = None


def _rotation_z(yaw_rad: float) -> np.ndarray:
    c = cos(yaw_rad)
    s = sin(yaw_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


SCENARIOS: dict[str, ScenarioDefinition] = {
    "hover_5s": ScenarioDefinition(
        scenario_name="hover_5s",
        duration_s=5.0,
        reference_type="hover",
        position_m=np.array([0.0, 0.0, 1.0], dtype=float),
        yaw_rad=0.0,
    ),
    "line_tracking": ScenarioDefinition(
        scenario_name="line_tracking",
        duration_s=5.0,
        reference_type="line",
        start_m=np.array([0.0, 0.0, 1.0], dtype=float),
        end_m=np.array([2.0, 0.0, 1.0], dtype=float),
        yaw_rad=0.0,
    ),
    "circle_tracking": ScenarioDefinition(
        scenario_name="circle_tracking",
        duration_s=5.0,
        reference_type="circle",
        center_m=np.array([0.0, 0.0, 1.0], dtype=float),
        radius_m=1.0,
        angular_rate_rad_s=0.8,
        yaw_mode="tangent",
    ),
}

RANDOM_REFERENCE_SCENARIO = "random_mpc_reference"


def available_scenarios() -> list[str]:
    return list(SCENARIOS.keys()) + [RANDOM_REFERENCE_SCENARIO]


def get_scenario_definition(scenario_name: str) -> ScenarioDefinition:
    try:
        return SCENARIOS[scenario_name]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario_name: {scenario_name}") from exc


def generate_reference_states(scenario_name: str, time_grid: np.ndarray) -> np.ndarray:
    """Generate a deterministic 18 x T reference state matrix for a named scenario."""

    definition = get_scenario_definition(scenario_name)
    t = np.asarray(time_grid, dtype=float).reshape(-1)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time_grid must be a 1D array with at least two samples.")

    states = []
    duration = max(definition.duration_s, np.finfo(float).eps)
    for t_s in t:
        if definition.reference_type == "hover":
            position = np.asarray(definition.position_m, dtype=float)
            velocity = np.zeros(3, dtype=float)
            yaw = float(definition.yaw_rad or 0.0)
            rotation = _rotation_z(yaw)
            angular_velocity = np.zeros(3, dtype=float)
        elif definition.reference_type == "line":
            start = np.asarray(definition.start_m, dtype=float)
            end = np.asarray(definition.end_m, dtype=float)
            progress = np.clip(t_s / duration, 0.0, 1.0)
            position = start + progress * (end - start)
            velocity = (end - start) / duration
            yaw = float(definition.yaw_rad or 0.0)
            rotation = _rotation_z(yaw)
            angular_velocity = np.zeros(3, dtype=float)
        elif definition.reference_type == "circle":
            center = np.asarray(definition.center_m, dtype=float)
            radius = float(definition.radius_m)
            omega = float(definition.angular_rate_rad_s)
            angle = omega * t_s
            position = center + np.array([radius * cos(angle), radius * sin(angle), 0.0], dtype=float)
            velocity = np.array([-radius * omega * sin(angle), radius * omega * cos(angle), 0.0], dtype=float)
            if definition.yaw_mode == "tangent":
                yaw = float(np.arctan2(velocity[1], velocity[0]))
                angular_velocity = np.array([0.0, 0.0, omega], dtype=float)
            else:
                yaw = 0.0
                angular_velocity = np.zeros(3, dtype=float)
            rotation = _rotation_z(yaw)
        else:
            raise ValueError(f"Unsupported reference_type: {definition.reference_type}")

        states.append(pack_state(position, velocity, rotation, angular_velocity))

    return np.column_stack(states)
