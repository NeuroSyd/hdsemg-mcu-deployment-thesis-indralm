"""
Calibration test (Hyser): how much of session 2 does a device need to see (no labels) to recover the
accuracy lost when session 2 is normalised with session 1 statistics?

For each subject it re-scores the SAVED models (results/<root>/<subj>/<run>/hyser/best.keras, so no
retraining) on session 2 after z-scoring session 2 with statistics estimated from:
    N = 0           session 1 statistics (calibration-free; must reproduce metrics.json)
    N = 5..80       statistics from N session 2 trials, scheme "first" (recording order) or "spread" (evenly spaced)
    all             statistics from all session 2 trials (the original own-session normalisation)
Every variant is evaluated on ALL session 2 trials. Only unlabelled signal statistics are used for
calibration (no labels). Models and session 1 statistics are exactly those of the training run.

Needs: the saved checkpoints (subjects run with --keep-checkpoints), the per-subject windowed manifest in
manifests/per_subject, and raw data (downloaded from PhysioNet again, then deleted unless --keep-raw).

Usage (repo root, hyser_env active):
    python scripts/calibration_test.py --subjects 03,07,09,12
    python scripts/calibration_test.py --subjects 01,03-05,07-18 --skip-download
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.preprocessing.filters import apply_filter_chain  # noqa: E402
from src.preprocessing.io_hyser import read_hyser_record  # noqa: E402
from src.preprocessing.normalization import RunningChannelStats, calibration_indices, zscore_with_stats  # noqa: E402
from src.training.metrics import per_trial_majority_vote_accuracy, per_window_accuracy  # noqa: E402
from src.windowing.windowing import compute_window_params, sliding_windows  # noqa: E402

S3_BASE = "s3://physionet-open/hd-semg/2.0.0/pr_dataset/"
RUNS = {"1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out += [f"{i:02d}" for i in range(int(a), int(b) + 1)]
        elif part:
            out.append(f"{int(part):02d}")
    return out


def filtered(path, cfg):
    sig, fs, _ = read_hyser_record(path)
    return apply_filter_chain(sig, fs, cfg).astype(np.float32)


def stats_of(arrays):
    st = RunningChannelStats(arrays[0].shape[0])
    for a in arrays:
        st.update(a)
    return st.mean, st.std


def run_subject(sub, a, cfg, win):
    import tensorflow as tf
    t0 = time.time()
    man = pd.read_csv(REPO / "manifests" / "per_subject" / f"hyser_windowed_manifest_{sub}.csv")
    man = man[man["task_type"] == "dynamic"]
    raw_root = a.raw_root.expanduser().resolve()
    if not a.skip_download:
        for sess in (1, 2):
            folder = f"subject{sub}_session{sess}"
            subprocess.run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors", f"{S3_BASE}{folder}/",
                            str(raw_root / folder), "--exclude", "*", "--include", "dynamic_raw_sample*",
                            "--include", "label_dynamic.txt"], check=True)

    def rows(sess):
        d = man[man["session"].astype(int) == sess].copy()
        d["sample_index"] = d["sample_index"].astype(float).astype(int)
        d = d.sort_values("sample_index")
        d["path"] = [str(raw_root / f"subject{sub}_session{sess}" / f) for f in d["filename"]]
        return d

    r1, r2 = rows(1), rows(2)
    # session 1 statistics, streamed (identical rows and filter chain to training)
    st1 = RunningChannelStats(256)
    for p in r1["path"]:
        st1.update(filtered(p, cfg))
    mean1, std1 = st1.mean, st1.std
    sig2 = [filtered(p, cfg) for p in r2["path"]]
    y = (r2["gesture_label"].astype(int).to_numpy() - 1)
    n_tr = len(sig2)

    variants = [("N=0 (session 1 stats)", 0, "none", (mean1, std1))]
    for scheme in ("first", "spread"):
        for n in a.n_cal:
            idx = calibration_indices(n_tr, n, scheme)
            variants.append((f"N={n} {scheme}", n, scheme, stats_of([sig2[i] for i in idx])))
    variants.append(("all session 2 (own-session stats)", n_tr, "all", stats_of(sig2)))

    wl, step = win
    models = {m: tf.keras.models.load_model(REPO / a.results_root / sub / d / "hyser" / "best.keras", compile=False)
              for m, d in RUNS.items()}
    out = []
    for name, n, scheme, (mu, sd) in variants:
        Xs, tid = [], []
        for i, s in enumerate(sig2):
            w = sliding_windows(zscore_with_stats(s, mu, sd).astype(np.float32), wl, step)
            Xs.append(w); tid.append(np.full(len(w), i))
        X, tid = np.concatenate(Xs), np.concatenate(tid)
        yw = y[tid]
        for m, model in models.items():
            pred = model.predict(X, batch_size=64, verbose=0).argmax(-1)
            out.append(dict(subject=sub, model=m, variant=name, n_cal=n, scheme=scheme,
                            per_trial=per_trial_majority_vote_accuracy(tid, yw, pred) * 100,
                            per_window=per_window_accuracy(yw, pred) * 100))
        del X
    df = pd.DataFrame(out)
    # check: N=0 must reproduce the stored cross-session accuracy of the training run
    for m, d in RUNS.items():
        stored = json.load(open(REPO / a.results_root / sub / d / "hyser" / "metrics.json"))["cross_session"]["per_trial_accuracy"] * 100
        got = df[(df.model == m) & (df.n_cal == 0) & (df.scheme == "none")]["per_trial"].iloc[0]
        flag = "OK" if abs(stored - got) < 1.0 else "MISMATCH (check raw data / filters)"
        print(f"  check {m}: stored {stored:.1f} vs recomputed {got:.1f}  {flag}", flush=True)
    outdir = REPO / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"subject{sub}.csv", index=False)
    if not a.keep_raw:
        for sess in (1, 2):
            shutil.rmtree(raw_root / f"subject{sub}_session{sess}", ignore_errors=True)
    print(f"  subject{sub} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def summarise(a):
    files = sorted((REPO / a.out).glob("subject*.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    n_sub = df.subject.nunique()
    order = list(dict.fromkeys(df.sort_values(["scheme", "n_cal"])["variant"]))
    t = df.groupby(["variant", "model"])["per_trial"].mean().unstack()
    t = t.loc[[v for v in ["N=0 (session 1 stats)"] + [x for x in order if x.startswith("N=") and "first" in x]
               + [x for x in order if x.startswith("N=") and "spread" in x] + ["all session 2 (own-session stats)"] if v in t.index]]
    t["mean_3_nets"] = t.mean(axis=1)
    t.round(1).to_csv(REPO / a.out / "summary_per_trial.csv")
    print(f"\n=== calibration test: cross-session per-trial accuracy (%), mean over {n_sub} subjects ===")
    print(t.round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="03,07,09,12")
    ap.add_argument("--results-root", default="results/hyser_refnorm")
    ap.add_argument("--out", default="reports/calibration_test")
    ap.add_argument("--raw-root", type=Path, default=Path("~/hyser_data/hyser"))
    ap.add_argument("--n-cal", type=lambda s: [int(x) for x in s.split(",")], default=[5, 10, 20, 40, 80])
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(REPO / "configs" / "preprocessing.yaml"))
    wcfg = yaml.safe_load(open(REPO / "configs" / "windowing.yaml"))
    win = compute_window_params(wcfg["window_ms"], wcfg["overlap"], wcfg["sampling_rate_hz"])
    failed = []
    for sub in parse_subjects(a.subjects):
        if (REPO / a.out / f"subject{sub}.csv").exists() and not a.force:
            print(f"subject{sub}: already done, skipping (use --force to redo)"); continue
        print(f"\n===== subject{sub} =====", flush=True)
        try:
            run_subject(sub, a, cfg, win)
        except Exception as e:  # keep going; report at the end
            print(f"  !! subject{sub} failed: {e!r}", flush=True); failed.append(sub)
    summarise(a)
    print("\nFinished. Failed subjects:", failed or "none")


if __name__ == "__main__":
    main()
