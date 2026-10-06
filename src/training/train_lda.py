"""
Stage 3 LDA baseline: classical time-domain features + Linear Discriminant Analysis.

Uses exactly the same trial-level, gesture-stratified split as the deep
models (src/training/train.py), so its accuracy is a fixed reference figure
for the Stage 3 comparative evaluation. Features (MAV, RMS, ZC, WL, SSC per
channel, concatenated) come from src/features/time_domain.py.

With 5 features x 256 channels = 1280 dimensions and only a few hundred
training windows per subject, plain LDA is ill-conditioned, so features are
standardised on the training split and LDA uses automatic (Ledoit-Wolf)
covariance shrinkage. This is a standard regularised-LDA setup and is
logged in metrics.json.

Usage:
    python src/training/train_lda.py --config configs/training.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest
from src.features.time_domain import extract_feature_vector
from src.training.metrics import (
    per_trial_majority_vote_accuracy,
    per_window_accuracy,
    trial_majority_vote,
)
from src.training.splits import subject_dependent_split
from src.training.train import select_trials


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def featurise(trials, feat_cfg: dict):
    """Windows of every trial -> (features, y, trial_id). y is 0-indexed."""
    F, y, tid = [], [], []
    for i, (_, row) in enumerate(trials.iterrows()):
        windows = np.load(row["windowed_path"])["X"]
        for w in windows:
            F.append(extract_feature_vector(w, feat_cfg["features"], feat_cfg))
        y.append(np.full(len(windows), int(row["gesture_label"]) - 1))
        tid.append(np.full(len(windows), i))
    return np.asarray(F, dtype=np.float64), np.concatenate(y), np.concatenate(tid)


def run_lda(config_path: str | Path, seed: int | None = None) -> dict:
    cfg = load_config(config_path)
    base_seed = cfg["split"]["random_seed"]
    if seed is not None:
        cfg["split"] = {**cfg["split"], "random_seed": seed}
    feat_cfg = load_config("configs/features.yaml")

    data_cfg = cfg.get("data", {})
    manifest = select_trials(load_manifest(cfg["input_manifest"]), data_cfg)
    train_sessions = {str(s) for s in data_cfg.get("train_sessions", manifest["session"].unique())}
    in_scope = manifest[manifest["session"].astype(str).isin(train_sessions)].reset_index(drop=True)

    s = cfg["split"]
    splits = subject_dependent_split(
        in_scope, s["train_fraction"], s["val_fraction"], s["test_fraction"], s["random_seed"]
    )
    # LDA has no validation loop, so val trials are folded into training.
    train_trials = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    t0 = time.time()
    F_tr, y_tr, _ = featurise(train_trials, feat_cfg)
    F_te, y_te, tid_te = featurise(splits["test"], feat_cfg)
    feat_seconds = time.time() - t0

    scaler = StandardScaler().fit(F_tr)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(scaler.transform(F_tr), y_tr)
    y_pred = lda.predict(scaler.transform(F_te))

    trial_true, trial_pred = trial_majority_vote(tid_te, y_te, y_pred)
    metrics = {
        "architecture": "lda",
        "variant": "",
        "seed": cfg["split"]["random_seed"],
        "dataset": cfg["dataset"],
        "subjects": sorted(in_scope["subject_id"].unique().tolist()),
        "task_type": data_cfg.get("task_type"),
        "train_sessions": sorted(train_sessions),
        "n_trials": {"train+val": int(len(train_trials)), "test": int(len(splits["test"]))},
        "n_test_windows": int(len(y_te)),
        "per_window_accuracy": per_window_accuracy(y_te, y_pred),
        "per_trial_accuracy": per_trial_majority_vote_accuracy(tid_te, y_te, y_pred),
        "feature_dim": int(F_tr.shape[1]),
        "feature_seconds": round(feat_seconds, 1),
        "setup": "StandardScaler + LDA(lsqr, shrinkage=auto)",
    }

    if cfg.get("cross_session_eval"):
        other = manifest[~manifest["session"].astype(str).isin(train_sessions)].reset_index(drop=True)
        if len(other):
            F_o, y_o, tid_o = featurise(other, feat_cfg)
            p_o = lda.predict(scaler.transform(F_o))
            metrics["cross_session"] = {
                "eval_sessions": sorted(other["session"].astype(str).unique().tolist()),
                "n_trials": int(len(other)),
                "per_window_accuracy": per_window_accuracy(y_o, p_o),
                "per_trial_accuracy": per_trial_majority_vote_accuracy(tid_o, y_o, p_o),
            }

    run_name = "lda" + ("" if cfg["split"]["random_seed"] == base_seed else f"_seed{cfg['split']['random_seed']}")
    out = Path(cfg["output_dir"]) / run_name / cfg["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test_predictions.npz", y_true=y_te, y_pred=y_pred, trial_id=tid_te,
             trial_true=trial_true, trial_pred=trial_pred)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"LDA baseline: per-window {metrics['per_window_accuracy']:.4f}, "
          f"per-trial {metrics['per_trial_accuracy']:.4f}")
    if "cross_session" in metrics:
        cs = metrics["cross_session"]
        print(f"  Cross-session (preview): per-window {cs['per_window_accuracy']:.4f}, per-trial {cs['per_trial_accuracy']:.4f}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--seed", type=int, default=None, help="Override split.random_seed (repeat-split runs)")
    a = p.parse_args()
    run_lda(a.config, a.seed)


if __name__ == "__main__":
    main()
