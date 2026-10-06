"""Normalisation for Stage 1 preprocessing."""

import numpy as np


def zscore_normalize(signal: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score a (n_channels, n_samples) array using ITS OWN statistics (scope:
    per recording). Kept for comparison runs and unit tests; the Stage 1
    pipeline uses `zscore_with_stats` with session-wide statistics instead.
    """
    mean = signal.mean(axis=axis, keepdims=True)
    std = signal.std(axis=axis, keepdims=True)
    return (signal - mean) / (std + eps)


class RunningChannelStats:
    """
    Accumulate per-channel mean/std over many (n_channels, n_samples)
    recordings in float64, without holding them all in memory at once.
    """

    def __init__(self, n_channels: int):
        self.n = 0
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sumsq = np.zeros(n_channels, dtype=np.float64)

    def update(self, signal: np.ndarray) -> None:
        x = signal.astype(np.float64, copy=False)
        self.n += x.shape[1]
        self.sum += x.sum(axis=1)
        self.sumsq += np.square(x).sum(axis=1)

    @property
    def mean(self) -> np.ndarray:
        return self.sum / self.n

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.sumsq / self.n - self.mean ** 2, 0.0))


def zscore_with_stats(signal: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score a (n_channels, n_samples) array with precomputed per-channel
    statistics. Using one mean/std per channel for a whole subject-session
    (the Week 6 framework decision) preserves the relative amplitude between
    trials and between channels, which per-recording z-scoring erases (it
    forces every channel of every trial to unit RMS).
    """
    return (signal - mean[:, None]) / (std[:, None] + eps)
