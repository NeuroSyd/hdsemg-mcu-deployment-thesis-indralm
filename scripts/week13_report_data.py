"""Data tables behind the Week 13 progress report (figures and tables). numpy/pandas only.
Run: python scripts/week13_report_data.py   ->  reports/week13/*.csv"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "week13"; OUT.mkdir(parents=True, exist_ok=True)
SUBJ = [f"{i:02d}" for i in range(1, 21)]
RUNS = {"1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8", "LDA": "lda"}

# 1. normalisation effect per subject (per-trial cross-session, %)
def cross(root, s, d):
    return json.load(open(ROOT / root / s / d / "hyser" / "metrics.json"))["cross_session"]["per_trial_accuracy"] * 100
rows = []
for s in SUBJ:
    for m, d in RUNS.items():
        rows.append(dict(subject=s, model=m, own_session=cross("results/hyser_all", s, d), session1_stats=cross("results/hyser_refnorm", s, d)))
pd.DataFrame(rows).to_csv(OUT / "norm_per_subject_model.csv", index=False)

# 2. calibration curves (4 subjects)
cal = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "reports" / "calibration_test").glob("subject*.csv")))
cal.groupby(["subject", "variant", "n_cal", "scheme"], as_index=False)["per_trial"].mean().to_csv(OUT / "calibration_per_subject.csv", index=False)

# 3. layered results, per subject, with paired bootstrap CIs over subjects
def load_lc(root):
    sys.argv = [sys.argv[0], "--results-root", root, "--out", "reports/week13/_tmp"]
    spec = importlib.util.spec_from_file_location("lc", ROOT / "scripts" / "layered_classifier_check.py")
    lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc); return lc
rng = np.random.default_rng(0)
def paired_ci(d):
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    return d.mean(), *np.percentile(b, [2.5, 97.5]), int((d > 0).sum()), int((d < 0).sum())
res = []
for setting, root in [("session 1 stats (calibration-free)", "results/hyser_refnorm"), ("own-session stats (recalibrated)", "results/hyser_all")]:
    lc = load_lc(root); data = lc.load(); N = lc.NETS
    R = {m: np.stack([lc.per_class_recall(data[(s, "cross")][0], data[(s, "cross")][1][m]) for s in lc.SUBJECTS]) for m in N}
    acc = {k: [] for k in ["GRU alone", "Hard vote", "Weighted hard vote", "Oracle (any net right)"] + (["Soft vote (mean probability)"] if root.endswith("refnorm") else [])}
    for i, s in enumerate(lc.SUBJECTS):
        true, p = data[(s, "cross")]
        acc["GRU alone"].append((p["GRU"] == true).mean())
        acc["Hard vote"].append((lc.vote(np.stack([p[m] for m in N])) == true).mean())
        oth = [j for j in range(20) if j != i]
        W = np.stack([np.nan_to_num(np.nanmean(R[m][oth], 0), nan=0.5) for m in N]) + 0.01
        acc["Weighted hard vote"].append((lc.vote(np.stack([p[m] for m in N]), W) == true).mean())
        acc["Oracle (any net right)"].append(np.stack([p[m] == true for m in N]).any(0).mean())
        if "Soft vote (mean probability)" in acc:
            P = np.stack([lc.PROBS[(s, "cross")][m] for m in N]); acc["Soft vote (mean probability)"].append((P.mean(0).argmax(-1) == true).mean())
    base = np.array(acc["GRU alone"]) * 100
    for k, v in acc.items():
        v = np.array(v) * 100; d = v - base
        mu, lo, hi, up, dn = paired_ci(d) if k != "GRU alone" else (0, 0, 0, 0, 0)
        res.append(dict(setting=setting, method=k, mean=v.mean(), sd=v.std(ddof=1), diff_vs_gru=mu, ci_lo=lo, ci_hi=hi, subjects_better=up, subjects_worse=dn))
pd.DataFrame(res).to_csv(OUT / "layered_results.csv", index=False)
# cascade curve (calibration-free) from existing csv if present
print(pd.DataFrame(res).round(1).to_string(index=False))
import shutil; shutil.rmtree(OUT / "_tmp", ignore_errors=True)
