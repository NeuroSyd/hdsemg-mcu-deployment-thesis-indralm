"""Per-recording normalisation for Stage 1 preprocessing."""

import numpy as np


def zscore_normalize(signal: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalise a (n_channels, n_samples) array along `axis`.

    Normalisation scope is per-subject-per-session (each call receives one
    full recording), matching the Week 6 framework decision, rather than a
    global scope across the dataset.
    """
    mean = signal.mean(axis=axis, keepdims=True)
    std = signal.std(axis=axis, keepdims=True)
    return (signal - mean) / (std + eps)
