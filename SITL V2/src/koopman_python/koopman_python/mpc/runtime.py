"""Runtime learned-MPC helpers for online SITL V2 integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from koopman_python.dynamics.params import DEFAULT_PROFILE, get_params
from koopman_python.dynamics.srb import pack_state, unpack_state
from koopman_python.edmd.basis import lift_state, lift_trajectory, lifted_state_dimension
from koopman_python.edmd.fit import EdmdModel
from koopman_python.mpc.qp import build_qp_structure, form_qp, shift_warm_start, solve_box_qp
from koopman_python.training.random_trajectories import rk4_step


@dataclass(frozen=True)
class LearnedMpcRuntimeConfig:
    """Online learned-MPC configuration for one-step receding-horizon control."""

    pred_horizon: int = 10
    control_lower_bound: float = -50.0
    control_upper_bound: float = 50.0
    qp_max_iter: int = 100
    qp_tol: float = 1e-8
    parameter_profile: str = DEFAULT_PROFILE
    control_time_step: float = 1e-2
    max_roll_pitch_rad: float = 0.35


@dataclass(frozen=True)
class LearnedMpcStepResult:
    """One online MPC solve result."""

    control: np.ndarray
    solve_time_ms: float
    solve_iterations: int
    solve_converged: bool
    qp_objective_value: float


@dataclass(frozen=True)
class RollPitchYawrateThrustCommand:
    """Hover-first low-level command derived from the learned body wrench."""

    roll_rad: float
    pitch_rad: float
    yaw_rate_rad_s: float
    thrust_newton: float
    predicted_state: np.ndarray


def load_edmd_model(model_path: str | Path) -> EdmdModel:
    """Load an EDMD model stored by the offline runners."""

    path = Path(model_path)
    payload = np.load(path, allow_pickle=False)
    if "n_basis" in payload:
        n_basis = int(np.asarray(payload["n_basis"]).reshape(()))
    else:
        z_dim = int(np.asarray(payload["A"]).shape[0])
        remainder = z_dim - 24
        if remainder < 0 or remainder % 9 != 0:
            raise ValueError(f"Cannot infer n_basis from lifted dimension {z_dim}.")
        n_basis = remainder // 9
        if lifted_state_dimension(n_basis) != z_dim:
            raise ValueError(f"Inferred n_basis={n_basis} does not match lifted dimension {z_dim}.")
    return EdmdModel(
        K=np.asarray(payload["K"], dtype=float),
        A=np.asarray(payload["A"], dtype=float),
        B=np.asarray(payload["B"], dtype=float),
        C=np.asarray(payload["C"], dtype=float),
        Z1=np.asarray(payload["Z1"], dtype=float),
        Z2=np.asarray(payload["Z2"], dtype=float),
        n_basis=n_basis,
    )


def _infer_n_basis_from_model(model: EdmdModel) -> int:
    return int(model.n_basis)


class LearnedMpcController:
    """Reusable one-step learned-MPC controller for online ROS integration."""

    def __init__(
        self,
        model: EdmdModel,
        config: LearnedMpcRuntimeConfig | None = None,
    ) -> None:
        self.model = model
        self.config = LearnedMpcRuntimeConfig() if config is None else config
        self.params = get_params(self.config.parameter_profile)
        self.qp_structure = build_qp_structure(
            model=model,
            horizon=self.config.pred_horizon,
            lower_bound=self.config.control_lower_bound,
            upper_bound=self.config.control_upper_bound,
        )
        self._warm_start: np.ndarray | None = None

    def reset(self) -> None:
        self._warm_start = None

    def solve(self, current_state: np.ndarray, reference_horizon: np.ndarray) -> LearnedMpcStepResult:
        current = np.asarray(current_state, dtype=float).reshape(18)
        reference = np.asarray(reference_horizon, dtype=float)
        if reference.ndim != 2 or reference.shape != (18, self.config.pred_horizon):
            raise ValueError("reference_horizon must be an 18 x pred_horizon matrix.")

        z_current = lift_state(current, _infer_n_basis_from_model(self.model))
        z_reference = lift_trajectory(reference, _infer_n_basis_from_model(self.model))
        qp = form_qp(self.qp_structure, z_current=z_current, z_reference=z_reference)

        import time

        solve_start = time.perf_counter()
        solve_result = solve_box_qp(
            qp,
            initial_guess=self._warm_start,
            max_iter=self.config.qp_max_iter,
            tol=self.config.qp_tol,
        )
        solve_time_ms = (time.perf_counter() - solve_start) * 1000.0

        self._warm_start = shift_warm_start(solve_result.solution, self.qp_structure.control_dim)
        full_solution = solve_result.solution
        control = full_solution[: self.qp_structure.control_dim]
        objective = 0.5 * float(full_solution @ (qp.G @ full_solution)) + float(qp.F @ full_solution)
        return LearnedMpcStepResult(
            control=control,
            solve_time_ms=solve_time_ms,
            solve_iterations=solve_result.iterations,
            solve_converged=solve_result.converged,
            qp_objective_value=objective,
        )


def _rotation_matrix_to_roll_pitch_yaw(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return ZYX roll, pitch, yaw from a body-to-world rotation matrix."""

    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    pitch = float(np.arcsin(-np.clip(r[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) <= 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))
    else:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    return roll, pitch, yaw


def control_to_roll_pitch_yawrate_thrust(
    current_state: np.ndarray,
    control: np.ndarray,
    *,
    parameter_profile: str = DEFAULT_PROFILE,
    control_time_step: float = 1e-2,
    max_roll_pitch_rad: float = 0.35,
) -> RollPitchYawrateThrustCommand:
    """Map learned body-force/body-moment control to a hover-first low-level command."""

    state = np.asarray(current_state, dtype=float).reshape(18)
    command = np.asarray(control, dtype=float).reshape(4)
    params = get_params(parameter_profile)
    predicted_state = rk4_step(
        state=state,
        control=command,
        dt=control_time_step,
        params=params,
        time_s=0.0,
    )
    _position, _velocity, rotation, angular_velocity = unpack_state(predicted_state)
    roll, pitch, _yaw = _rotation_matrix_to_roll_pitch_yaw(rotation)
    roll = float(np.clip(roll, -max_roll_pitch_rad, max_roll_pitch_rad))
    pitch = float(np.clip(pitch, -max_roll_pitch_rad, max_roll_pitch_rad))
    thrust_newton = float(max(command[0], 0.0))
    yaw_rate = float(angular_velocity[2])
    return RollPitchYawrateThrustCommand(
        roll_rad=roll,
        pitch_rad=pitch,
        yaw_rate_rad_s=yaw_rate,
        thrust_newton=thrust_newton,
        predicted_state=predicted_state,
    )


def build_constant_reference_horizon(
    position_xyz: np.ndarray,
    yaw_rad: float,
    horizon: int,
) -> np.ndarray:
    """Build a constant pose reference horizon for hover-first online tests."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    state = pack_state(
        position=np.asarray(position_xyz, dtype=float).reshape(3),
        velocity=np.zeros(3, dtype=float),
        rotation=rotation,
        angular_velocity=np.zeros(3, dtype=float),
    )
    return np.repeat(state.reshape(18, 1), horizon, axis=1)
