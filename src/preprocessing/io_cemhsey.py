"""CEMHSEY signal loading for Stage 1 (MATLAB v5 .mat, one variable `data_sEMG`)."""

import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FS = 2048


def read_cemhsey_trial(path: str | Path, retries: int = 5):
    """Return (signal float32 of shape (320, n_samples), fs). Retries for cloud-synced folders."""
    for attempt in range(retries):
        try:
            d = loadmat(str(path), variable_names=["data_sEMG"])
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))
    x = d["data_sEMG"]
    if x.ndim != 2 or x.shape[0] != 320:
        raise ValueError(f"Unexpected data_sEMG shape {x.shape} in {path}")
    return x.astype(np.float32), FS
