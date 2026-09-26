"""
Subject-dependent train/val/test splitting for Stage 3.

Split happens at trial granularity (one manifest row = one WFDB sample/trial)
within each subject, not at the individual window level, so overlapping
windows from the same trial never end up split across train and test - that
split-by-trial rule is what keeps the 50% window overlap from leaking
information between sets. This is also why the split is "subject-dependent":
every subject contributes trials to train, val, and test alike, rather than
holding whole subjects out (that would be a subject-independent protocol,
not used here per the Week 6 framework decision).
"""

import numpy as np
import pandas as pd


def subject_dependent_split(
    manifest: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, pd.DataFrame]:
    """
    Split a windowed-stage manifest into train/val/test, per subject.

    Returns {"train": df, "val": df, "test": df}, each a subset of rows from
    `manifest` with the original index preserved.
    """
    fractions_sum = train_fraction + val_fraction + test_fraction
    if not np.isclose(fractions_sum, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions_sum}")

    rng = np.random.default_rng(random_seed)
    train_rows, val_rows, test_rows = [], [], []

    for _, subject_rows in manifest.groupby("subject_id"):
        indices = subject_rows.index.to_numpy()
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))

        train_rows.append(subject_rows.loc[indices[:n_train]])
        val_rows.append(subject_rows.loc[indices[n_train:n_train + n_val]])
        test_rows.append(subject_rows.loc[indices[n_train + n_val:]])

    return {
        "train": pd.concat(train_rows).reset_index(drop=True),
        "val": pd.concat(val_rows).reset_index(drop=True),
        "test": pd.concat(test_rows).reset_index(drop=True),
    }
