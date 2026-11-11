"""Train and score the grouped sequential classifier for Hyser subjects (needs windowed data).

For each subject and each saved flat network (results/<root>/<subj>/<run>/hyser/best.keras) it
    1. extracts penultimate features of session 1 train/val/test and of session 2,
       with session 2 normalised by session 1 statistics (s1stats) and, label-free,
       by statistics re-estimated from N spread unlabelled session 2 trials (cal{N})
    2. trains the group head and the within-group specialist heads on session 1
       training trials (grouping per subject from reports/sequential_offline/groups.json,
       learned without that subject's session 2)
    3. fits the labelled last-layer baseline with k = 1..K labelled session 2 trials per gesture
    4. saves trial-level probabilities of every condition for scripts/evaluate_sequential_trained.py

No network weights are changed. Session 2 labels are used only for the labelled baseline
shots and for scoring; shot trials are excluded from scoring for every condition.

Usage (repo root, after the windowed data of the subject exist, e.g. via run_hyser_sequential.py):
    python scripts/run_sequential_trained.py --subjects 01
    python scripts/run_sequential_trained.py --subjects 01,03-05 --groupings D1-5,H5 --n-cal 10 --shots 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.sequential import groups as G                  # noqa: E402
from src.sequential import heads as H                   # noqa: E402
from src.preprocessing.normalization import calibration_indices  # noqa: E402
from src.training.splits import make_split              # noqa: E402

RUNS = {"1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [f"{i:02d}" for i in range(int(a), int(b) + 1)]
        elif part:
            out.append(f"{int(part):02d}")
    return out


def grouping_for(name, sub, groups_json):
    if name in G.ANATOMICAL:
        return G.ANATOMICAL[name]
    return np.asarray(groups_json["data"][sub][name])


def labels(rows):
    return rows["gesture_label"].astype(int).to_numpy() - 1


def run_subject(sub, a, split_cfg, groups_json):
    import tensorflow as tf
    t0 = time.time()
    man = pd.read_csv(REPO / a.manifest_dir / f"hyser_windowed_manifest_{sub}.csv")
    man = man[man["task_type"] == "dynamic"].reset_index(drop=True)
    missing = [p for p in man["windowed_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"subject{sub}: {len(missing)} windowed files missing (run the data stages first)")

    s1 = man[man["session"].astype(str) == "1"].reset_index(drop=True)
    s2 = man[man["session"].astype(str) == "2"].copy()
    s2["sample_index"] = s2["sample_index"].astype(float).astype(int)
    s2 = s2.sort_values("sample_index").reset_index(drop=True)        # recording order
    sp = make_split(s1, split_cfg)                                      # identical to the flat training split
    y2 = labels(s2)
    n2 = len(s2)
    cal_idx = calibration_indices(n2, a.n_cal, "spread")   # same rule as the Week 13 calibration test
    shots = H.pick_shots(y2, a.shots, seed=a.seed)
    eval_mask = np.ones(n2, bool)
    eval_mask[shots[a.shots]] = False

    out_dir = REPO / a.out / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"y2": y2, "eval_mask": eval_mask, "cal_idx": cal_idx,
            "y1_test": labels(sp["test"])}
    for k, idx in shots.items():
        save[f"shots{k}"] = idx
    info = {"subject": sub, "n_cal": a.n_cal, "shots": a.shots, "groupings": a.groupings, "runs": {}}

    for name, run in RUNS.items():
        ckpt = REPO / a.results_root / sub / run / "hyser" / "best.keras"
        if not ckpt.exists():
            print(f"  {name}: no checkpoint at {ckpt}, skipped", flush=True)
            continue
        model = tf.keras.models.load_model(ckpt, compile=False)
        two = H.split_model(model)
        tr, va, te = (sp[k]["windowed_path"].tolist() for k in ("train", "val", "test"))
        f_tr, p_tr, t_tr = H.extract(two, tr)
        f_va, p_va, t_va = H.extract(two, va)
        f_te, p_te, t_te = H.extract(two, te)
        ytr, yva = labels(sp["train"])[t_tr], labels(sp["val"])[t_va]
        paths2 = s2["windowed_path"].tolist()
        f2, p2, t2 = H.extract(two, paths2)
        mean_c, std_c = H.channel_stats([paths2[i] for i in cal_idx])
        f2c, p2c, _ = H.extract(two, paths2, renorm=(mean_c, std_c))
        key = run
        tm = lambda p, t, n: H.trial_mean(p, t, n)               # noqa: E731
        save[f"{key}__s1test__flat"] = tm(p_te, t_te, len(te))
        save[f"{key}__s1stats__flat"] = tm(p2, t2, n2)
        save[f"{key}__cal__flat"] = tm(p2c, t2, n2)
        rinfo = {"feature_dim": int(f_tr.shape[1]), "heads": {}}

        for gname in a.groupings:
            gmap = grouping_for(gname, sub, groups_json)
            gh = H.GroupedHeads(gmap).fit(f_tr, ytr, f_va, yva)
            for cond, f, t, n in (("s1test", f_te, t_te, len(te)), ("s1stats", f2, t2, n2), ("cal", f2c, t2, n2)):
                pg, pw = gh.proba(f)
                save[f"{key}__{cond}__{gname}__group"] = tm(pg, t, n)
                save[f"{key}__{cond}__{gname}__within"] = tm(pw, t, n)
            rinfo["heads"][gname] = {"params": gh.n_params(), "group_C": gh.group.C,
                                     "K": int(G.n_groups(gmap))}

        # labelled last-layer baseline (session 1 stats input): ll0 = refit on session 1 only (control)
        n_tr = len(ytr)
        for k in range(0, a.shots + 1):
            if k == 0:
                f_fit, y_fit, w_fit = f_tr, ytr, None
            else:
                sel = np.isin(t2, shots[k])
                f_s, y_s = f2[sel], y2[t2[sel]]
                f_fit = np.concatenate([f_tr, f_s])
                y_fit = np.concatenate([ytr, y_s])
                w_fit = np.concatenate([np.ones(n_tr), np.full(len(y_s), n_tr / max(1, len(y_s)))])
            head = H.Head(np.arange(34)).fit(f_fit, y_fit, None, None, w_fit)
            save[f"{key}__ll{k}__flat"] = tm(head.proba(f2), t2, n2)
        info["runs"][run] = rinfo
        print(f"  {name}: done", flush=True)
        tf.keras.backend.clear_session()

    np.savez_compressed(out_dir / "seq_predictions.npz", **save)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(out_dir / "seq_info.json", "w"), indent=1)
    print(f"  subject{sub} sequential heads done in {info['seconds'] / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="01-20")
    ap.add_argument("--results-root", default="results/hyser_refnorm", help="where the flat best.keras are")
    ap.add_argument("--manifest-dir", default="manifests/per_subject")
    ap.add_argument("--groups", default="reports/sequential_offline/groups.json")
    ap.add_argument("--groupings", default="D1-5,H5")
    ap.add_argument("--n-cal", type=int, default=10, help="unlabelled session 2 trials for label-free calibration")
    ap.add_argument("--shots", type=int, default=2, help="max labelled session 2 trials per gesture (baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--training-config", default="configs/training.yaml")
    ap.add_argument("--out", default="results/hyser_seq")
    a = ap.parse_args()
    a.groupings = [g.strip() for g in a.groupings.split(",") if g.strip()]
    split_cfg = yaml.safe_load(open(REPO / a.training_config))["split"]
    groups_json = json.load(open(REPO / a.groups))
    failed = []
    for sub in parse_subjects(a.subjects):
        print(f"\n===== subject{sub} (sequential heads) =====", flush=True)
        try:
            run_subject(sub, a, split_cfg, groups_json)
        except Exception as e:  # keep going with the next subject
            print(f"  !! subject{sub} failed: {e}", flush=True)
            failed.append(sub)
    print("\nFinished. Failed subjects:", failed or "none")


if __name__ == "__main__":
    main()
