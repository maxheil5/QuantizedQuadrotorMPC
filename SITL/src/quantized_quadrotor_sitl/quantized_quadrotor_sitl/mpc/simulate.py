from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy import sparse
from scipy.integrate import solve_ivp

from ..core.config import MPCConfig
from ..core.types import EDMDModel, MPCSimulation
from ..dynamics.params import get_params
from ..dynamics.srb import dynamics_srb
from ..edmd.basis import lift_state
from ..utils.state import decode_lifted_prefix
from .qp import get_qp


FloatArray = NDArray[np.float64]


def solve_qp(f_vector: FloatArray, g_matrix: FloatArray, a_ineq, b_ineq: FloatArray) -> FloatArray:
    g_symmetric = 0.5 * (g_matrix + g_matrix.T)
    try:
        import osqp

        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix(g_symmetric + 1.0e-9 * np.eye(g_symmetric.shape[0])),
            q=f_vector,
            A=a_ineq,
            l=-np.inf * np.ones_like(b_ineq),
            u=b_ineq,
            verbose=False,
        )
        result = problem.solve()
        if result.x is not None:
            return np.asarray(result.x, dtype=float)
    except Exception:
        pass

    objective = lambda u: 0.5 * float(u.T @ g_symmetric @ u) + float(f_vector.T @ u)
    jacobian = lambda u: g_symmetric @ u + f_vector
    linear_constraint = optimize.LinearConstraint(a_ineq.toarray(), -np.inf, b_ineq)
    initial_guess = np.zeros_like(f_vector)
    result = optimize.minimize(
        objective,
        initial_guess,
        jac=jacobian,
        method="trust-constr",
        constraints=[linear_constraint],
        options={"maxiter": 200, "verbose": 0},
    )
    if not result.success:
        raise RuntimeError(f"QP solve failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def sim_mpc(
    model: EDMDModel,
    lifted_state0: FloatArray,
    lifted_reference: FloatArray,
    physical_reference: FloatArray,
    config: MPCConfig,
) -> MPCSimulation:
    params = get_params()
    dt_sim = config.sim_timestep
    horizon = config.pred_horizon
    tstart = 0.0
    tend = dt_sim

    t_log: list[float] = []
    x_log: list[FloatArray] = []
    u_log: list[FloatArray] = []
    x_ref_log: list[FloatArray] = []

    z = np.asarray(lifted_state0, dtype=float).copy()
    for step in range(config.max_iter):
        z_ref = lifted_reference[:, step : step + horizon]
        f_vector, g_matrix, a_ineq, b_ineq = get_qp(model, z, z_ref, horizon, config)
        zval = solve_qp(f_vector, g_matrix, a_ineq, b_ineq)
        u_t = zval[:4]

        decoded = model.C @ z
        _, _, r_matrix, wb = decode_lifted_prefix(decoded)
        x_state = np.concatenate(
            [
                decoded[0:3],
                decoded[3:6],
                r_matrix.reshape(-1, order="F"),
                wb,
            ]
        )

        solution = solve_ivp(
            lambda t, state: dynamics_srb(t, state, u_t, params),
            (tstart, tend),
            x_state,
            method="RK45",
            t_eval=np.array([tend]),
            rtol=1.0e-3,
            atol=1.0e-6,
        )
        next_state = solution.y[:, -1]
        z = lift_state(next_state, model.n_basis)

        tstart = tend
        tend = tstart + dt_sim

        t_log.append(float(solution.t[-1]))
        x_log.append(next_state)
        u_log.append(u_t)
        x_ref_log.append(physical_reference[:, step])

    return MPCSimulation(
        t=np.asarray(t_log, dtype=float),
        X=np.vstack(x_log),
        U=np.vstack(u_log),
        X_ref=np.vstack(x_ref_log),
    )
