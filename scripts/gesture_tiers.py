"""Rank the 34 Hyser gestures easiest to hardest with bootstrap CIs over subjects, and tier them.
Accuracy per gesture = mean over the 3 networks (1D CNN, TCN, GRU) of the per-subject per-trial recall,
then mean over subjects. CI = 95% bootstrap over the 20 subjects. Tier: easy if CI lower > grand mean,
hard if CI upper < grand mean, else middle (cannot be separated from average).
Run: python scripts/gesture_tiers.py --results-root results/hyser_refnorm --out reports/layered_check_refnorm --split cross"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default="results/hyser_refnorm"); ap.add_argument("--out", default="reports/layered_check_refnorm")
ap.add_argument("--split", default="cross", choices=["cross", "same"])
a = ap.parse_args()
sys.argv = [sys.argv[0], "--results-root", a.results_root, "--out", a.out]
spec = importlib.util.spec_from_file_location("lc", ROOT / "scripts" / "layered_classifier_check.py")
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
data = lc.load()
NAMES = {1:"Thumb ext",2:"Index ext",3:"Middle ext",4:"Ring ext",5:"Little ext",6:"Wrist flex",7:"Wrist ext",8:"Wrist radial",9:"Wrist ulnar",10:"Wrist pron",11:"Wrist sup",12:"Thumb+index ext",13:"Index+middle ext",14:"Wrist flex+close",15:"Wrist ext+close",16:"Wrist radial+close",17:"Wrist ulnar+close",18:"Wrist pron+close",19:"Wrist sup+close",20:"Wrist flex+open",21:"Wrist ext+open",22:"Wrist radial+open",23:"Wrist ulnar+open",24:"Wrist pron+open",25:"Wrist sup+open",26:"Thumb+index+middle ext",27:"Index+middle+ring ext",28:"Middle+ring+little ext",29:"Index to little ext",30:"Hand close",31:"Hand open",32:"Thumb+index pinch",33:"Thumb+index+middle pinch",34:"Thumb+middle pinch"}
M = np.stack([np.stack([lc.per_class_recall(data[(s, a.split)][0], data[(s, a.split)][1][m]) for s in lc.SUBJECTS]) for m in lc.NETS])  # (3,20,34)
subj = np.nanmean(M, 0)                      # (20,34) mean over networks
mean = np.nanmean(subj, 0) * 100
rng = np.random.default_rng(0)
boots = np.array([np.nanmean(subj[rng.integers(0, 20, 20)], 0) for _ in range(5000)]) * 100
lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
grand = np.nanmean(mean)
tier = np.where(lo > grand, "easy", np.where(hi < grand, "hard", "middle"))
df = pd.DataFrame(dict(gesture=range(1, 35), name=[NAMES[g] for g in range(1, 35)], mean=mean, ci_lo=lo, ci_hi=hi, tier=tier,
                       n_subjects=[(~np.isnan(subj[:, g])).sum() for g in range(34)])).sort_values("mean", ascending=False)
df["rank_easiest_first"] = range(1, 35)
tag = f"gesture_tiers_{a.split}.csv"
df.to_csv(lc.OUT / tag, index=False)
print(f"grand mean {grand:.1f}%  |  tiers: " + ", ".join(f"{t}={int((tier==t).sum())}" for t in ["easy", "middle", "hard"]))
print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
