"""
Evaluation metrics for Stage 3.

Accuracy is reported two ways per the Week 6 framework decision: per-window
(every window scored independently) and per-trial via majority vote (all of
a trial's window predictions collapsed to one label before scoring).
"""

import numpy as np


def per_window_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of individually-scored windows predicted correctly."""
    return float(np.mean(y_true == y_pred))


def per_trial_majority_vote_accuracy(
    trial_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Collapse window-level predictions to one label per trial by majority
    vote, then score against that trial's true label.

    `trial_id` groups windows belonging to the same trial (see
    dataset.build_dataset); all windows sharing a trial_id must also share
    the same y_true value.
    """
    correct = 0
    total = 0
    for tid in np.unique(trial_id):
        mask = trial_id == tid
        votes = y_pred[mask]
        majority = np.bincount(votes).argmax()
        true_label = y_true[mask][0]

        correct += int(majority == true_label)
        total += 1

    return correct / total
