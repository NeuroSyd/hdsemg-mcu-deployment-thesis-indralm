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


def per_session_accuracies(sessions_by_trial, trial_id: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Accuracy per session (CEMHSEY: per day) for a set of trials. `sessions_by_trial[i]` is the
    session of the trial with trial_id == i. Returns {session: {per_window_accuracy, per_trial_accuracy, n_trials}}.
    """
    sess = np.asarray(sessions_by_trial).astype(str)[trial_id]
    out = {}
    for s in sorted(set(sess), key=lambda v: int(v)):
        m = sess == s
        out[s] = {
            "per_window_accuracy": per_window_accuracy(y_true[m], y_pred[m]),
            "per_trial_accuracy": per_trial_majority_vote_accuracy(trial_id[m], y_true[m], y_pred[m]),
            "n_trials": int(len(np.unique(trial_id[m]))),
        }
    return out


def trial_mean_probability(trial_id: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Mean of a trial's window probabilities, one row per unique trial_id, in the same
    order as `trial_majority_vote`. Stored so layered / soft-voting classifiers can be
    evaluated later without re-running inference.
    """
    return np.stack([y_prob[trial_id == t].mean(axis=0) for t in np.unique(trial_id)])


def expand_proba(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    """Place a classifier's predict_proba columns (ordered by `classes`) into an (n, n_classes) array."""
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, np.asarray(classes, dtype=int)] = proba
    return out
