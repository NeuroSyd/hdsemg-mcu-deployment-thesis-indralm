"""
Windowed manifest rows -> (X, y, trial_id) arrays for Stage 3 training.

Trial-level gesture labels are not yet attached anywhere upstream: the
windowing manifest (src/windowing/pipeline.py) segments signals into windows
but does not yet parse Hyser's label_<task>.txt files, since their column
layout has not been confirmed against the Hyser documentation. load_trial_label
below is the single point where that gets wired in once the label file
format is confirmed; everything else in this module is independent of it.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_trial_label(row: pd.Series) -> int:
    """
    Return the gesture label for one trial (one windowed manifest row).

    Pending: parse the matching label_<task_type>.txt for this row's
    subject_id/session and look up the label for row["sample_index"].
    """
    raise NotImplementedError(
        "Label lookup depends on the Hyser label_<task>.txt column format, "
        "still to be confirmed - see src/windowing/pipeline.py docstring."
    )


def build_dataset(manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load every window from the given manifest rows into a single dataset.

    Returns:
        X: (n_windows, n_channels, window_len)
        y: (n_windows,) gesture label per window, propagated from its trial
        trial_id: (n_windows,) index into `manifest` identifying the source
                  trial, used by metrics.per_trial_majority_vote_accuracy
                  to regroup window-level predictions back to trial level.
    """
    X_parts, y_parts, trial_id_parts = [], [], []

    for trial_idx, row in manifest.iterrows():
        windows = np.load(Path(row["windowed_path"]))["X"]
        label = load_trial_label(row)

        X_parts.append(windows)
        y_parts.append(np.full(windows.shape[0], label))
        trial_id_parts.append(np.full(windows.shape[0], trial_idx))

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(trial_id_parts, axis=0),
    )
