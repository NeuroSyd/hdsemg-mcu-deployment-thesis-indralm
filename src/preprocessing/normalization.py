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


def resolve_group_stats(scope, subject_id, session, own_stats, reference_session, reference_store):
    """
    Pick the (mean, std) used to z-score one subject-session group.

    scope "per_subject_session": the group's own statistics (the original behaviour).
    scope "reference_session":   the statistics of the subject's reference session (default
        session 1, the training session) are applied to every other session, as a deployed
        device would do. The reference session must be processed first for each subject;
        its statistics are stored in `reference_store`.
    """
    if scope != "reference_session":
        return own_stats
    if str(session) == str(reference_session):
        reference_store[subject_id] = own_stats
        return own_stats
    if subject_id not in reference_store:
        raise ValueError(
            f"{subject_id}: reference session {reference_session} has not been processed, "
            f"so session {session} cannot be normalised with its statistics"
        )
    return reference_store[subject_id]


def calibration_indices(n_trials: int, n_cal: int, scheme: str) -> np.ndarray:
    """
    Which trials (0-based, in recording order) a device would use to estimate normalisation
    statistics without labels. Hyser stores trials in gesture order (6 in a row per gesture),
    so the two schemes behave very differently:
      "first":  the first n_cal trials (a start-up calibration; sees only the first few gestures)
      "spread": n_cal trials evenly spaced over the session (a short mixed calibration routine)
    """
    n_cal = max(0, min(int(n_cal), n_trials))
    if n_cal == 0:
        return np.array([], dtype=int)
    if scheme == "first":
        return np.arange(n_cal)
    if scheme == "spread":
        return np.unique(np.round(np.linspace(0, n_trials - 1, n_cal)).astype(int))
    raise ValueError(f"Unknown calibration scheme '{scheme}'")
