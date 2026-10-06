"""
Windowed manifest rows -> (X, y, trial_id) arrays for Stage 3 training.

The windowing manifest carries each trial's `gesture_label` (1 to 34, parsed
from Hyser's label_<task>.txt in src/common/labels.py). Models use 0-indexed
classes, so labels are shifted down by one here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_trial_label(row: pd.Series) -> int:
    """0-indexed class for one trial (one windowed manifest row)."""
    return int(row["gesture_label"]) - 1


def build_dataset(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load every window from the given manifest rows into a single dataset.

    Returns:
        X: (n_windows, n_channels, window_len), float32
        y: (n_windows,) class per window, propagated from its trial
        trial_id: (n_windows,) row position in `manifest` identifying the
                  source trial, used by metrics.per_trial_majority_vote_accuracy
                  to regroup window-level predictions back to trial level.
    """
    X_parts, y_parts, trial_id_parts = [], [], []

    for trial_idx, (_, row) in enumerate(manifest.iterrows()):
        windows = np.load(Path(row["windowed_path"]))["X"].astype(np.float32, copy=False)
        X_parts.append(windows)
        y_parts.append(np.full(windows.shape[0], load_trial_label(row)))
        trial_id_parts.append(np.full(windows.shape[0], trial_idx))

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(trial_id_parts, axis=0),
    )
