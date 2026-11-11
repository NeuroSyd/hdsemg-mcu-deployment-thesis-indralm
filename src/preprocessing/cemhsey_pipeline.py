"""
Stage 1 + 2a for CEMHSEY, one day at a time (keeps disk and RAM small).

Per trial: zero-phase 10-500 Hz Butterworth bandpass + 50 Hz notch on the whole
continuous trial (done before cutting, so there are no edge artefacts inside a
gesture), then z-score with per-channel statistics taken from the TRAIN trials
of the training day only (Day 1, trials in `split.stats_trials`). Every later day is
normalised with those same Day 1 statistics, as a deployed device would be.
Then each trial is cut into its 11 gesture holds (fixed protocol timing) and each hold is
windowed (200 ms). One .npz per (trial, gesture) holds the windows, so a
manifest row is one gesture performance, the same unit Hyser's "trial" is.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.extract_cemhsey import GESTURES, gesture_segments
from src.preprocessing.filters import apply_filter_chain
from src.preprocessing.io_cemhsey import read_cemhsey_trial
from src.preprocessing.normalization import RunningChannelStats
from src.windowing.windowing import compute_window_params, sliding_windows


def filter_trial(path, cfg: dict, chunk: int = 32) -> np.ndarray:
    """Load and filter one trial in channel chunks (caps peak RAM at roughly 1.5 GB)."""
    signal, fs = read_cemhsey_trial(path)
    out = np.empty(signal.shape, dtype=np.float32)
    for i in range(0, signal.shape[0], chunk):
        out[i:i + chunk] = apply_filter_chain(signal[i:i + chunk].astype(np.float64), fs, cfg)
    return out


def window_trial(norm: np.ndarray, row: dict, cfg: dict, overlap: float, out_dir: Path) -> list[dict]:
    fs = cfg["sampling_rate_hz"]
    window_len, step = compute_window_params(cfg["windowing"]["window_ms"], overlap, fs)
    dtype = np.dtype(cfg["windowing"].get("dtype", "float16"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, a, b in gesture_segments(norm.shape[1], fs):
        w = sliding_windows(norm[:, a:b], window_len, step).astype(dtype)
        path = out_dir / f"{row['subject_id']}_Day{row['session']}_Trial{row['trial']}_G{label:02d}_windows.npz"
        np.savez(path, X=w)
        rows.append({**row, "gesture_label": label, "gesture_name": GESTURES[label - 1],
                     "windowed_path": str(path), "n_windows": int(w.shape[0]),
                     "window_len_samples": window_len, "step_samples": step, "overlap": overlap})
    return rows


def process_day(rows: list[dict], day: int, cfg: dict, work: Path) -> list[dict]:
    """Filter, normalise and window every trial of one day. Returns the windowed manifest rows."""
    train_day = cfg["train_day"]
    train_trials = set(cfg["split"]["stats_trials"])
    work.mkdir(parents=True, exist_ok=True)
    stats_path = work / "norm_stats.npz"
    out_dir = work / "windows" / f"day{day:02d}"
    overlap = cfg["windowing"]["train_day_overlap"] if day == train_day else cfg["windowing"]["eval_day_overlap"]

    held = {}
    if day == train_day:
        stats = None
        for row in rows:
            if row["trial"] in train_trials:
                f = filter_trial(row["file_path"], cfg)
                stats = stats or RunningChannelStats(f.shape[0])
                stats.update(f)
                held[row["trial"]] = f
        if stats is None:
            raise RuntimeError("No training trials found on the training day")
        np.savez(stats_path, mean=stats.mean, std=stats.std, n_samples=stats.n)
    if not stats_path.exists():
        raise RuntimeError(f"{stats_path} missing: process the training day (Day {train_day}) first")
    s = np.load(stats_path)
    mean = s["mean"].astype(np.float32)[:, None]
    std = s["std"].astype(np.float32)[:, None]

    out = []
    for row in sorted(rows, key=lambda r: r["trial"]):
        print(f"  Day {day} trial {row['trial']}: filter, normalise, window", flush=True)
        f = held.pop(row["trial"], None)
        if f is None:
            f = filter_trial(row["file_path"], cfg)
        f -= mean
        f /= (std + 1e-8)
        if not np.isfinite(f).all():
            raise RuntimeError(f"Non-finite values after normalisation in {row['filename']}")
        out += window_trial(f, row, cfg, overlap, out_dir)
        del f
    return out
