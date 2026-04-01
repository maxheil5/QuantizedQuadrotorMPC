from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import EDMDModel


def _scalar_bool(payload: np.lib.npyio.NpzFile, key: str, default: bool = False) -> bool:
    if key not in payload:
        return default
    return bool(np.asarray(payload[key]).reshape(-1)[0])


def _scalar_str(payload: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key]).reshape(-1)[0]
    return str(value)


def load_edmd_artifact(path: Path) -> tuple[EDMDModel, dict[str, object]]:
    with np.load(path) as payload:
        bias = np.asarray(payload["bias"], dtype=float).reshape(-1) if "bias" in payload else None
        affine_enabled = _scalar_bool(payload, "affine_enabled", default=bias is not None)
        model = EDMDModel(
            A=np.asarray(payload["A"], dtype=float),
            B=np.asarray(payload["B"], dtype=float),
            C=np.asarray(payload["C"], dtype=float),
            Z1=np.asarray(payload["Z1"], dtype=float),
            Z2=np.asarray(payload["Z2"], dtype=float),
            n_basis=int(np.asarray(payload["n_basis"]).reshape(-1)[0]),
            bias=bias,
            affine_enabled=affine_enabled,
        )
        metadata = {
            key: np.asarray(payload[key], dtype=float)
            for key in ("x_train_min", "x_train_max", "u_train_min", "u_train_max")
            if key in payload
        }
        for key in ("u_train_mean", "u_train_std", "u_trim"):
            if key in payload:
                metadata[key] = np.asarray(payload[key], dtype=float)
        if "state_trim" in payload:
            metadata["state_trim"] = np.asarray(payload["state_trim"], dtype=float)
        metadata["residual_enabled"] = _scalar_bool(payload, "residual_enabled", default=False)
        state_coordinates = _scalar_str(payload, "state_coordinates")
        if state_coordinates is not None:
            metadata["state_coordinates"] = state_coordinates
        state_trim_mode = _scalar_str(payload, "state_trim_mode")
        if state_trim_mode is not None:
            metadata["state_trim_mode"] = state_trim_mode
    return model, metadata
