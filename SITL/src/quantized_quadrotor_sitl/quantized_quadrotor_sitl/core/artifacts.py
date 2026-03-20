from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import EDMDModel


def load_edmd_artifact(path: Path) -> tuple[EDMDModel, dict[str, np.ndarray]]:
    payload = np.load(path)
    model = EDMDModel(
        A=np.asarray(payload["A"], dtype=float),
        B=np.asarray(payload["B"], dtype=float),
        C=np.asarray(payload["C"], dtype=float),
        Z1=np.asarray(payload["Z1"], dtype=float),
        Z2=np.asarray(payload["Z2"], dtype=float),
        n_basis=int(np.asarray(payload["n_basis"]).reshape(-1)[0]),
    )
    metadata = {
        key: np.asarray(payload[key], dtype=float)
        for key in ("x_train_min", "x_train_max", "u_train_min", "u_train_max")
        if key in payload
    }
    return model, metadata


def coarsen_edmd_model(model: EDMDModel, step_multiple: int) -> EDMDModel:
    if step_multiple <= 1:
        return model

    a_power = np.eye(model.A.shape[0], dtype=float)
    b_accum = np.zeros_like(model.B, dtype=float)
    for _ in range(step_multiple):
        b_accum += a_power @ model.B
        a_power = a_power @ model.A

    return EDMDModel(
        A=a_power,
        B=b_accum,
        C=model.C,
        Z1=model.Z1,
        Z2=model.Z2,
        n_basis=model.n_basis,
    )
