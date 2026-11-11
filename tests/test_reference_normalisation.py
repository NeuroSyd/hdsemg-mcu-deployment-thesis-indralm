"""Tests for reference-session normalisation and stored probabilities (numpy only). Run: pytest -q tests"""

import numpy as np
import pytest

from src.preprocessing.normalization import RunningChannelStats, resolve_group_stats, zscore_with_stats
from src.training.metrics import expand_proba, trial_majority_vote, trial_mean_probability


def _stats(x):
    s = RunningChannelStats(x.shape[0]); s.update(x)
    return s.mean, s.std


def test_default_scope_keeps_own_statistics():
    own = (np.zeros(2), np.ones(2))
    assert resolve_group_stats("per_subject_session", "s01", 2, own, 1, {}) is own


def test_reference_scope_applies_session1_stats_to_session2():
    rng = np.random.default_rng(0)
    s1 = rng.normal(0, 1.0, (4, 5000)); s2 = rng.normal(0, 3.0, (4, 5000))  # amplitude drift on day 2
    store = {}
    m1, sd1 = resolve_group_stats("reference_session", "s01", 1, _stats(s1), 1, store)
    m2, sd2 = resolve_group_stats("reference_session", "s01", 2, _stats(s2), 1, store)
    assert np.allclose(sd2, sd1)                                   # session 2 uses session 1 statistics
    z2 = zscore_with_stats(s2, m2, sd2)
    assert 2.5 < z2.std() < 3.5                                    # drift preserved, not normalised away
    own = zscore_with_stats(s2, *_stats(s2))
    assert abs(own.std() - 1.0) < 1e-3                             # the original scope removes it


def test_reference_scope_is_per_subject_and_needs_reference_first():
    store = {}
    with pytest.raises(ValueError):
        resolve_group_stats("reference_session", "s02", 2, (np.zeros(1), np.ones(1)), 1, store)
    resolve_group_stats("reference_session", "s01", 1, (np.zeros(1), np.ones(1)), 1, store)
    with pytest.raises(ValueError):                                # s01's reference must not leak to s02
        resolve_group_stats("reference_session", "s02", 2, (np.zeros(1), np.ones(1)), 1, store)


def test_trial_probability_matches_majority_vote_order():
    tid = np.array([5, 5, 2, 2, 2])
    prob = np.array([[.9, .1], [.7, .3], [.2, .8], [.4, .6], [.1, .9]])
    tp = trial_mean_probability(tid, prob)
    _, pred = trial_majority_vote(tid, np.array([0, 0, 1, 1, 1]), prob.argmax(-1))
    assert tp.shape == (2, 2) and np.array_equal(tp.argmax(-1), pred)   # row 0 is trial 2, row 1 is trial 5
    assert np.allclose(tp[1], [.8, .2])


def test_expand_proba_fills_missing_classes_with_zero():
    out = expand_proba(np.array([[.3, .7]]), [0, 3], 5)
    assert out.shape == (1, 5) and np.isclose(out[0, 3], .7) and out[0, 1] == 0


def test_calibration_indices_schemes():
    from src.preprocessing.normalization import calibration_indices
    assert calibration_indices(200, 0, "first").size == 0
    assert np.array_equal(calibration_indices(200, 5, "first"), np.arange(5))
    sp = calibration_indices(200, 10, "spread")
    assert len(sp) == 10 and sp[0] == 0 and sp[-1] == 199 and np.all(np.diff(sp) > 0)
    assert len(calibration_indices(6, 40, "spread")) == 6          # capped at the number of trials
    with pytest.raises(ValueError):
        calibration_indices(10, 3, "random")
