"""Tests for the CEMHSEY protocol helpers (no dataset files needed)."""

import numpy as np
import pytest

from src.extraction.extract_cemhsey import FS, GESTURES, gesture_segments, parse_days
from src.training.splits import make_split
from src.windowing.windowing import compute_window_params, sliding_windows


def test_gesture_segments_follow_protocol():
    segs = gesture_segments(369000)
    assert [s[0] for s in segs] == list(range(1, 12))
    assert len(segs) == len(GESTURES) == 11
    assert segs[0][1] == 5 * FS and segs[0][2] - segs[0][1] == 10 * FS
    assert segs[1][1] - segs[0][1] == 15 * FS          # 10 s hold + 5 s rest
    assert segs[-1][2] == 165 * FS


def test_gesture_segments_reject_short_trial():
    with pytest.raises(ValueError):
        gesture_segments(100 * FS)


def test_window_counts_per_hold():
    sig = np.zeros((4, 10 * FS))
    wl, step = compute_window_params(200, 0.5, FS)
    assert (wl, step) == (410, 205)
    assert sliding_windows(sig, wl, step).shape[0] == 98
    wl, step = compute_window_params(200, 0.0, FS)
    assert sliding_windows(sig, wl, step).shape[0] == 49


def test_parse_days():
    assert parse_days("1-3,5") == [1, 2, 3, 5]


def test_by_trial_split_has_no_overlap():
    import pandas as pd
    rows = [{"subject_id": "S4", "session": 1, "trial": t, "gesture_label": g, "windowed_path": f"{t}_{g}"}
            for t in range(1, 6) for g in range(1, 12)]
    sp = make_split(pd.DataFrame(rows), {"method": "by_trial", "train_trials": [1, 2, 3],
                                         "val_trials": [4], "test_trials": [5]})
    assert set(sp["train"]["trial"]) == {1, 2, 3} and set(sp["val"]["trial"]) == {4} and set(sp["test"]["trial"]) == {5}
    assert len(sp["train"]) == 33 and len(sp["test"]) == 11
