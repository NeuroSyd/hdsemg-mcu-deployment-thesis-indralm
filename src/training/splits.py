"""
Subject-dependent train/val/test splitting for Stage 3.

Split happens at trial granularity (one manifest row = one WFDB sample/trial)
within each subject, never at the individual window level, so overlapping
windows from the same trial cannot be split across train and test. That
split-by-trial rule is what stops the 50% window overlap leaking information
between sets. "Subject-dependent" means every subject contributes trials to
train, val and test alike (a subject-independent protocol would hold out
whole subjects; not used here per the Week 6 framework decision).

When the manifest carries a `gesture_label` column the split is also
stratified: each (subject, gesture) group of repetitions is split on its own,
so every gesture appears in every split whenever it has at least 3 trials.
Hyser has only ~6 repetitions per gesture, so an unstratified random split
would regularly leave whole gestures out of the test set.
"""

import numpy as np
import pandas as pd


def _split_indices(indices: np.ndarray, train_f: float, val_f: float, rng) -> tuple:
    indices = indices.copy()
    rng.shuffle(indices)
    n = len(indices)
    if n >= 3:
        n_val = max(1, int(round(n * val_f)))
        n_test = max(1, n - int(round(n * train_f)) - n_val)
        n_train = n - n_val - n_test
    else:
        n_train, n_val = n, 0
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def subject_dependent_split(
    manifest: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
    stratify_col: str | None = "gesture_label",
) -> dict[str, pd.DataFrame]:
    """
    Split a windowed-stage manifest into train/val/test, per subject
    (and per gesture when `stratify_col` is present in the manifest).

    Returns {"train": df, "val": df, "test": df}, each a subset of rows from
    `manifest`, with the manifest's original index kept in column `manifest_index`.
    """
    fractions_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fractions_sum, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions_sum}")

    rng = np.random.default_rng(random_seed)
    parts = {"train": [], "val": [], "test": []}

    group_cols = ["subject_id"]
    if stratify_col and stratify_col in manifest.columns:
        group_cols.append(stratify_col)

    for _, group in manifest.groupby(group_cols, sort=True):
        tr, va, te = _split_indices(group.index.to_numpy(), train_fraction, val_fraction, rng)
        parts["train"].append(group.loc[tr])
        parts["val"].append(group.loc[va])
        parts["test"].append(group.loc[te])

    out = {}
    for name, frames in parts.items():
        df = pd.concat(frames)
        df.index.name = "manifest_index"
        out[name] = df.reset_index()
    return out
