"""
Sanity tests for Stages 1 to 3. Run from the repo root:  pytest -q tests

These exist to give concrete, re-runnable evidence that each stage does what
the framework says it does (and are small enough to run in seconds without
the dataset).
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.filters import apply_filter_chain
from src.preprocessing.normalization import RunningChannelStats, zscore_with_stats
from src.training.metrics import per_trial_majority_vote_accuracy
from src.training.splits import subject_dependent_split
from src.windowing.windowing import compute_window_params, sliding_windows

FS = 2048
CFG = {"bandpass": {"low_hz": 10, "high_hz": 500, "order": 4}, "notch": {"freq_hz": 50, "quality_factor": 30}}


def _tone(freq, seconds=2.0):
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * freq * t)[None, :]


def _rms(x):
    core = x[:, FS // 2: -FS // 2]  # ignore filter edge effects
    return float(np.sqrt(np.mean(core ** 2)))


def test_bandpass_passes_emg_band_and_blocks_drift():
    assert _rms(apply_filter_chain(_tone(100), FS, CFG)) > 0.65     # in band: ~0.707 kept
    assert _rms(apply_filter_chain(_tone(2), FS, CFG)) < 0.02       # motion artefact removed


def test_notch_removes_50hz_mains():
    assert _rms(apply_filter_chain(_tone(50), FS, CFG)) < 0.05


def test_window_params_match_framework():
    assert compute_window_params(200, 0.5, FS) == (410, 205)       # 200 ms, 100 ms stride


def test_window_count_and_no_boundary_crossing():
    sig = np.arange(256 * 2048, dtype=np.float32).reshape(256, 2048)
    w = sliding_windows(sig, 410, 205)
    assert w.shape == (8, 256, 410)                                   # 1 s trial -> 8 windows
    assert np.array_equal(w[3], sig[:, 3 * 205: 3 * 205 + 410])       # windows are exact slices
    assert 7 * 205 + 410 <= 2048                                      # last window stays in the trial


def test_session_zscore_uses_shared_stats_and_keeps_relative_amplitude():
    rng = np.random.default_rng(0)
    quiet, loud = rng.normal(0, 1, (4, 3000)), rng.normal(0, 5, (4, 3000))
    stats = RunningChannelStats(4)
    stats.update(quiet), stats.update(loud)
    q = zscore_with_stats(quiet, stats.mean, stats.std)
    l = zscore_with_stats(loud, stats.mean, stats.std)
    assert l.std() > 3 * q.std()          # per-recording z-score would make these equal


def test_running_stats_match_numpy():
    rng = np.random.default_rng(1)
    parts = [rng.normal(2, 3, (5, 1000)) for _ in range(4)]
    stats = RunningChannelStats(5)
    for p in parts:
        stats.update(p)
    full = np.concatenate(parts, axis=1)
    assert np.allclose(stats.mean, full.mean(axis=1)) and np.allclose(stats.std, full.std(axis=1))


def _fake_manifest(n_gestures=5, reps=6):
    rows = [{"subject_id": "subject01", "gesture_label": g, "windowed_path": f"g{g}_r{r}.npz"}
            for g in range(1, n_gestures + 1) for r in range(reps)]
    return pd.DataFrame(rows)


def test_split_is_trial_level_disjoint_and_stratified():
    sp = subject_dependent_split(_fake_manifest(), 0.7, 0.15, 0.15, 42)
    sets = {k: set(v["windowed_path"]) for k, v in sp.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sum(len(v) for v in sets.values()) == 30                  # nothing lost or duplicated
    for part in ("train", "val", "test"):
        assert sp[part]["gesture_label"].nunique() == 5               # every gesture in every split
    assert len(sp["train"]) == 20 and len(sp["val"]) == 5 and len(sp["test"]) == 5


def test_majority_vote_accuracy():
    trial = np.array([0, 0, 0, 1, 1, 1])
    y_true = np.array([3, 3, 3, 4, 4, 4])
    y_pred = np.array([3, 3, 9, 4, 9, 9])                             # trial 0 right, trial 1 wrong
    assert per_trial_majority_vote_accuracy(trial, y_true, y_pred) == pytest.approx(0.5)
