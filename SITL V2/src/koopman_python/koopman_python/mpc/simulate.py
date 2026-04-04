"""Closed-loop learned MPC rollout for the V2 EDMD controller."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from koopman_python.dynamics.params import DEFAULT_PROFILE, get_params
from koopman_python.dynamics.srb import pack_state
from koopman_python.edmd.basis import lift_state
from koopman_python.edmd.fit import EdmdModel
from koopman_python.edmd.evaluate import vee_map
from koopman_python.mpc.qp import build_qp, solve_box_qp
from koopman_python.training.random_trajectories import rk4_step


@dataclass(frozen=True)
class MpcSimulationConfig:
    pred_horizon: int = 10
    sim_time_step: float = 1e-3
    sim_duration: float = 1.2
    use_projected_qp: bool = True
    control_lower_bound: float = -50.0
    control_upper_bound: float = 50.0
    qp_max_iter: int = 2000
    qp_tol: float = 1e-8
    parameter_profile: str = DEFAULT_PROFILE


@dataclass(frozen=True)
class MpcSimulationResult:
    t: np.ndarray
    X: np.ndarray
    U: np.ndarray
    X_ref: np.ndarray
    Z: np.ndarray
    Z_ref: np.ndarray
    solve_times_ms: np.ndarray


def observed_state_to_full_state(observed: np.ndarray) -> np.ndarray:
    """Convert C*z output back into the 18-state SRB layout."""

    flat = np.asarray(observed, dtype=float).reshape(24)
    x = flat[0:3]
    dx = flat[3:6]
    rotation = flat[6:15].reshape(3, 3, order="F").T
    wb_hat = flat[15:24].reshape(3, 3, order="F").T
    wb = vee_map(wb_hat)
    return pack_state(x, dx, rotation, wb)


def simulate_closed_loop(
    model: EdmdModel,
    initial_lifted_state: np.ndarray,
    lifted_reference: np.ndarray,
    reference_states: np.ndarray,
    config: MpcSimulationConfig | None = None,
    params: dict[str, object] | None = None,
) -> MpcSimulationResult:
    """Port of MATLAB/mpc/sim_MPC.m using a lightweight box-QP solver."""

    cfg = MpcSimulationConfig() if config is None else config
    dynamics_params = get_params(cfg.parameter_profile) if params is None else params

    Z = np.asarray(initial_lifted_state, dtype=float).reshape(-1)
    Z_ref = np.asarray(lifted_reference, dtype=float)
    X_ref = np.asarray(reference_states, dtype=float)
    if Z_ref.ndim != 2:
        raise ValueError("lifted_reference must be a z_dim x T matrix.")
    if X_ref.ndim != 2 or X_ref.shape[0] != 18:
        raise ValueError("reference_states must be an 18 x T matrix.")

    max_iter = int(np.floor(cfg.sim_duration / cfg.sim_time_step))
    required_ref_cols = max_iter + cfg.pred_horizon
    if Z_ref.shape[1] < required_ref_cols:
        raise ValueError("lifted_reference must contain at least MAX_ITER + pred_horizon columns.")
    if X_ref.shape[1] < max_iter:
        raise ValueError("reference_states must contain at least MAX_ITER columns.")

    t_values = []
    X_values = []
    U_values = []
    Xd_values = []
    Z_values = []
    solve_times_ms = []

    current_time = 0.0
    for ii in range(max_iter):
        z_ref_horizon = Z_ref[:, ii : ii + cfg.pred_horizon]
        qp = build_qp(
            model=model,
            z_current=Z,
            z_reference=z_ref_horizon,
            horizon=cfg.pred_horizon,
            lower_bound=cfg.control_lower_bound,
            upper_bound=cfg.control_upper_bound,
        )

        solve_start = time.perf_counter()
        zval = solve_box_qp(qp, max_iter=cfg.qp_max_iter, tol=cfg.qp_tol)
        solve_times_ms.append((time.perf_counter() - solve_start) * 1000.0)
        Ut = zval[:4]

        Xt_obs = model.C @ Z
        Xt = observed_state_to_full_state(Xt_obs)
        Xt_next = rk4_step(
            state=Xt,
            control=Ut,
            dt=cfg.sim_time_step,
            params=dynamics_params,
            time_s=current_time,
        )
        Z = lift_state(Xt_next, model.n_basis)

        current_time += cfg.sim_time_step
        t_values.append(current_time)
        X_values.append(Xt_next)
        U_values.append(Ut)
        Xd_values.append(X_ref[:, ii])
        Z_values.append(Z)

    return MpcSimulationResult(
        t=np.asarray(t_values, dtype=float),
        X=np.vstack(X_values),
        U=np.vstack(U_values),
        X_ref=np.vstack(Xd_values),
        Z=np.column_stack(Z_values),
        Z_ref=Z_ref[:, :max_iter],
        solve_times_ms=np.asarray(solve_times_ms, dtype=float),
    )
