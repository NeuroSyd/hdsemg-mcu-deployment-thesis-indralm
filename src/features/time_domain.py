"""
Classical time-domain sEMG features, per Phinyomark & Scheme (2018).

Used only for the LDA baseline; the deep architectures (1D CNN, GRU, TCN)
train directly on windowed raw channels and do not use this module.

Every function takes a window of shape (n_channels, window_len) and returns
one value per channel, shape (n_channels,).
"""

import numpy as np


def mean_absolute_value(window: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(window), axis=-1)


def root_mean_square(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(window ** 2, axis=-1))


def waveform_length(window: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(window, axis=-1)), axis=-1)


def zero_crossings(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Count sign changes between consecutive samples, ignoring changes smaller than `threshold`."""
    sign_change = np.diff(np.sign(window), axis=-1) != 0
    large_enough = np.abs(np.diff(window, axis=-1)) >= threshold
    return np.sum(sign_change & large_enough, axis=-1)


def slope_sign_changes(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Count changes in the sign of the first-order slope, ignoring changes smaller than `threshold`."""
    slope = np.diff(window, axis=-1)
    sign_change = np.diff(np.sign(slope), axis=-1) != 0
    large_enough = np.abs(np.diff(slope, axis=-1)) >= threshold
    return np.sum(sign_change & large_enough, axis=-1)


FEATURE_FUNCTIONS = {
    "mav": mean_absolute_value,
    "rms": root_mean_square,
    "wl": waveform_length,
    "zc": zero_crossings,
    "ssc": slope_sign_changes,
}


def extract_feature_vector(window: np.ndarray, feature_names: list[str], cfg: dict) -> np.ndarray:
    """
    Compute the requested features for one window and concatenate them into
    a single 1D vector, ordered [feature_0 x all channels, feature_1 x all
    channels, ...].
    """
    parts = []
    for name in feature_names:
        func = FEATURE_FUNCTIONS[name]
        if name == "zc":
            parts.append(func(window, cfg.get("zc_threshold", 0.0)))
        elif name == "ssc":
            parts.append(func(window, cfg.get("ssc_threshold", 0.0)))
        else:
            parts.append(func(window))
    return np.concatenate(parts)
