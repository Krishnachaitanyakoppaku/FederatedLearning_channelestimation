"""Common metrics used across centralized and federated experiments."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def to_complex(two_channel_array: np.ndarray) -> np.ndarray:
    """Convert real/imag stacked channels (N,2,H,W) into complex form (N,H,W)."""
    if two_channel_array.ndim != 4 or two_channel_array.shape[1] != 2:
        raise ValueError(
            f"Expected array shape (N, 2, H, W), got {two_channel_array.shape}."
        )
    return two_channel_array[:, 0] + 1j * two_channel_array[:, 1]


def nmse(H_true: np.ndarray, H_pred: np.ndarray) -> Dict[str, float]:
    """Compute NMSE in both linear and dB domains."""
    if H_true.shape != H_pred.shape:
        raise ValueError(f"Shape mismatch: H_true={H_true.shape}, H_pred={H_pred.shape}")

    numerator = np.linalg.norm(H_true - H_pred) ** 2
    denominator = np.linalg.norm(H_true) ** 2
    if denominator <= 0:
        raise ValueError("Reference channel power is zero, NMSE undefined.")

    nmse_linear = float(numerator / denominator)
    nmse_db = float(10.0 * math.log10(max(nmse_linear, 1e-30)))
    return {"nmse_linear": nmse_linear, "nmse_db": nmse_db}


def nmse_from_two_channel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute NMSE from stacked real/imag arrays."""
    h_true = to_complex(y_true)
    h_pred = to_complex(y_pred)
    return nmse(h_true, h_pred)
