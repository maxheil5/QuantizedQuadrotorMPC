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
    for key in ("u_train_mean", "u_train_std", "u_trim"):
        if key in payload:
            metadata[key] = np.asarray(payload[key], dtype=float)
    return model, metadata
