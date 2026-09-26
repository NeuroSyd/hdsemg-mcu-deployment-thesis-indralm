"""
Filter design and application for Stage 1 preprocessing.

Both retained datasets (Hyser, CEMHSEY) are processed with the same filter
chain: a Butterworth bandpass followed by a notch filter at mains frequency.
Hyser ships a pre-filtered signal variant already; this module is what
produces the equivalent filtering for the raw variant and for CEMHSEY, so
both datasets enter Stage 2 through an identical filtering path.

Signal convention used throughout this pipeline: arrays are shaped
(n_channels, n_samples), filtering applied along the last axis.
"""

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def design_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4):
    """Design a Butterworth bandpass filter, returned in second-order-sections form."""
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    return butter(order, [low, high], btype="bandpass", output="sos")


def design_notch_sos(freq_hz: float, fs: float, quality_factor: float = 30.0):
    """Design an IIR notch filter at `freq_hz`, returned in second-order-sections form."""
    b, a = iirnotch(freq_hz, quality_factor, fs)
    return tf2sos(b, a)


def apply_filter_chain(signal: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    """
    Apply the bandpass + notch chain to a (n_channels, n_samples) array.

    `cfg` is the parsed preprocessing config (see configs/preprocessing.yaml):
    expects `cfg["bandpass"]` with low_hz/high_hz/order and `cfg["notch"]`
    with freq_hz/quality_factor. Zero-phase filtering (filtfilt) is used so
    gesture-onset timing in the labels is not shifted.
    """
    bp_cfg = cfg["bandpass"]
    notch_cfg = cfg["notch"]

    bandpass_sos = design_bandpass_sos(bp_cfg["low_hz"], bp_cfg["high_hz"], fs, bp_cfg["order"])
    notch_sos = design_notch_sos(notch_cfg["freq_hz"], fs, notch_cfg["quality_factor"])

    filtered = sosfiltfilt(bandpass_sos, signal, axis=-1)
    filtered = sosfiltfilt(notch_sos, filtered, axis=-1)
    return filtered
