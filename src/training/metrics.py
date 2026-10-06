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


def trial_majority_vote(trial_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """
    Collapse window predictions to one label per trial by majority vote.

    Returns (trial_true, trial_pred), one entry per unique trial_id. All
    windows sharing a trial_id must share the same y_true value.
    """
    trial_true, trial_pred = [], []
    for tid in np.unique(trial_id):
        mask = trial_id == tid
        trial_pred.append(np.bincount(y_pred[mask]).argmax())
        trial_true.append(y_true[mask][0])
    return np.array(trial_true), np.array(trial_pred)


def per_trial_majority_vote_accuracy(
    trial_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Accuracy after collapsing each trial's windows by majority vote."""
    trial_true, trial_pred = trial_majority_vote(trial_id, y_true, y_pred)
    return float(np.mean(trial_true == trial_pred))
