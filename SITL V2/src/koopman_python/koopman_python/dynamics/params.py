"""Fresh V2 parameter source of truth for the SRB learned-model path.

The MATLAB dynamics source remains the algorithmic reference, but the active
V2 vehicle should match the Gazebo model we actually fly in RotorS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


STATE_LAYOUT = "x(3), dx(3), R(9), wb(3)"
CONTROL_LAYOUT = "Fb, Mbx, Mby, Mbz"


@dataclass(frozen=True)
class SrbParameters:
    """Minimal physical parameter set used by the MATLAB SRB model."""

    profile_name: str
    source_summary: str
    mass: float
    inertia_diag: Tuple[float, float, float]
    gravity: float = 9.81
    arm_length: float | None = None
    motor_constant: float | None = None
    moment_constant: float | None = None
    rotor_count: int | None = None
    drag_coefficients: Tuple[float, float, float] | None = None

    @property
    def J(self) -> np.ndarray:
        return np.diag(np.asarray(self.inertia_diag, dtype=float))

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["J"] = self.J.copy()
        if self.drag_coefficients is not None:
            payload["drag_coefficients"] = np.asarray(
                self.drag_coefficients, dtype=float
            )
        payload["state_layout"] = STATE_LAYOUT
        payload["control_layout"] = CONTROL_LAYOUT
        return payload


MATLAB_REFERENCE_PARAMS = SrbParameters(
    profile_name="matlab_reference",
    source_summary="MATLAB/dynamics/get_params.m",
    mass=4.34,
    inertia_diag=(0.0820, 0.0845, 0.1377),
    gravity=9.81,
)

# RotorS Firefly physical values from rotors_description/urdf/firefly.xacro.
ROTORS_FIREFLY_XACRO_PARAMS = SrbParameters(
    profile_name="rotors_firefly_xacro",
    source_summary="ethz-asl/rotors_simulator rotors_description/urdf/firefly.xacro",
    mass=1.5,
    inertia_diag=(0.0347563, 0.0458929, 0.0977),
    gravity=9.81,
    arm_length=0.215,
    motor_constant=8.54858e-06,
    moment_constant=0.016,
    rotor_count=6,
)

# The Lee controller in RotorS commonly runs with the same inertia but a
# slightly heavier mass parameter. We keep it explicit because it is exactly
# the sort of mismatch we want to surface early in V2.
ROTORS_FIREFLY_LEE_CONTROLLER_PARAMS = SrbParameters(
    profile_name="rotors_firefly_lee_controller",
    source_summary="RotorS Firefly lee_position_controller runtime parameters",
    mass=1.56779,
    inertia_diag=(0.0347563, 0.0458929, 0.0977),
    gravity=9.81,
    arm_length=0.215,
    motor_constant=8.54858e-06,
    moment_constant=0.016,
    rotor_count=6,
)

ROTORS_FIREFLY_LINEAR_MPC_RUNTIME_PARAMS = SrbParameters(
    profile_name="rotors_firefly_linear_mpc_runtime",
    source_summary="Verified from /firefly/mav_linear_mpc and /firefly/PID_attitude_controller on Ubuntu",
    mass=1.52,
    inertia_diag=(0.034756, 0.045893, 0.0977),
    gravity=9.81,
    arm_length=0.2156,
    motor_constant=8.54858e-06,
    moment_constant=0.016,
    rotor_count=6,
    drag_coefficients=(0.01, 0.01, 0.0),
)


PARAMETER_PROFILES = {
    MATLAB_REFERENCE_PARAMS.profile_name: MATLAB_REFERENCE_PARAMS,
    ROTORS_FIREFLY_XACRO_PARAMS.profile_name: ROTORS_FIREFLY_XACRO_PARAMS,
    ROTORS_FIREFLY_LEE_CONTROLLER_PARAMS.profile_name: ROTORS_FIREFLY_LEE_CONTROLLER_PARAMS,
    ROTORS_FIREFLY_LINEAR_MPC_RUNTIME_PARAMS.profile_name: ROTORS_FIREFLY_LINEAR_MPC_RUNTIME_PARAMS,
}

DEFAULT_PROFILE = ROTORS_FIREFLY_LINEAR_MPC_RUNTIME_PARAMS.profile_name


def available_profiles() -> Iterable[str]:
    return tuple(PARAMETER_PROFILES)


def compare_parameter_sets(
    lhs: SrbParameters, rhs: SrbParameters
) -> Dict[str, float]:
    """Return a compact numeric diff for audit/debug output."""

    lhs_inertia = np.asarray(lhs.inertia_diag, dtype=float)
    rhs_inertia = np.asarray(rhs.inertia_diag, dtype=float)
    return {
        "mass_delta": rhs.mass - lhs.mass,
        "gravity_delta": rhs.gravity - lhs.gravity,
        "ixx_delta": rhs_inertia[0] - lhs_inertia[0],
        "iyy_delta": rhs_inertia[1] - lhs_inertia[1],
        "izz_delta": rhs_inertia[2] - lhs_inertia[2],
        "arm_length_delta": (rhs.arm_length or 0.0) - (lhs.arm_length or 0.0),
    }


def get_params(profile_name: str = DEFAULT_PROFILE) -> Dict[str, object]:
    """Return a MATLAB-shaped parameter dictionary for the selected profile."""

    try:
        params = PARAMETER_PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(PARAMETER_PROFILES))
        raise ValueError(
            f"Unknown parameter profile '{profile_name}'. Available: {choices}"
        ) from exc
    return params.as_dict()
