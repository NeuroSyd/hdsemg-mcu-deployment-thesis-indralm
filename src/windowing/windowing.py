"""
Sliding-window segmentation for Stage 2.

200 ms windows at 50% overlap, matching CEMHSEY's own published LDA
benchmark protocol, applied to both retained datasets for comparability.
"""

import numpy as np


def compute_window_params(window_ms: float, overlap: float, fs: float) -> tuple[int, int]:
    """Return (window_len_samples, step_samples) for the given window length and overlap fraction."""
    window_len = int(round((window_ms / 1000.0) * fs))
    step = int(round(window_len * (1.0 - overlap)))
    return window_len, step


def sliding_windows(signal: np.ndarray, window_len: int, step: int) -> np.ndarray:
    """
    Segment a (n_channels, n_samples) array into overlapping windows.

    Returns an array of shape (n_windows, n_channels, window_len). Trailing
    samples that do not fill a full window are dropped.
    """
    n_channels, n_samples = signal.shape
    n_windows = 1 + (n_samples - window_len) // step if n_samples >= window_len else 0

    windows = np.empty((n_windows, n_channels, window_len), dtype=signal.dtype)
    for i in range(n_windows):
        start = i * step
        windows[i] = signal[:, start:start + window_len]
    return windows
