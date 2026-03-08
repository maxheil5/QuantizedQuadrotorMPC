from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import logm

from ..core.types import RMSEBreakdown
from .linear_algebra import vee_map


FloatArray = NDArray[np.float64]


def normalized_rmse(estimate: FloatArray, reference: FloatArray) -> float:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    est = np.asarray(estimate, dtype=float).reshape(-1)
    denominator = np.sqrt(np.mean(ref**2))
    if denominator == 0.0:
        return float(np.sqrt(np.mean((ref - est) ** 2)))
    return float(np.sqrt(np.mean((ref - est) ** 2)) / denominator)


def rmse(decoded_prediction: FloatArray, decoded_reference: FloatArray) -> RMSEBreakdown:
    x_ref: list[FloatArray] = []
    dx_ref: list[FloatArray] = []
    theta_ref: list[FloatArray] = []
    wb_ref: list[FloatArray] = []
    x_val: list[FloatArray] = []
    dx_val: list[FloatArray] = []
    theta_val: list[FloatArray] = []
    wb_val: list[FloatArray] = []
    for idx in range(decoded_prediction.shape[1]):
        r_true = decoded_reference[6:15, idx].reshape(3, 3, order="F").T
        wb_hat_true = decoded_reference[15:24, idx].reshape(3, 3, order="F")
        r_pred = decoded_prediction[6:15, idx].reshape(3, 3, order="F").T
        wb_hat_pred = decoded_prediction[15:24, idx].reshape(3, 3, order="F")

        x_ref.append(decoded_reference[0:3, idx])
        dx_ref.append(decoded_reference[3:6, idx])
        theta_ref.append(vee_map(logm(r_true)))
        wb_ref.append(vee_map(wb_hat_true))

        x_val.append(decoded_prediction[0:3, idx])
        dx_val.append(decoded_prediction[3:6, idx])
        theta_val.append(vee_map(logm(r_pred)))
        wb_val.append(vee_map(wb_hat_pred))

    return RMSEBreakdown(
        x=normalized_rmse(np.column_stack(x_val), np.column_stack(x_ref)),
        dx=normalized_rmse(np.column_stack(dx_val), np.column_stack(dx_ref)),
        theta=normalized_rmse(np.column_stack(theta_val), np.column_stack(theta_ref)),
        wb=normalized_rmse(np.column_stack(wb_val), np.column_stack(wb_ref)),
    )


def position_tracking_rmse(position: FloatArray, reference: FloatArray) -> float:
    return normalized_rmse(np.asarray(position, dtype=float), np.asarray(reference, dtype=float))

