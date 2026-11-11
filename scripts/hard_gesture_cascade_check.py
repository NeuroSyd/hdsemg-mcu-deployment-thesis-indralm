"""Cascade check: stage 1 = one network on all 34 gestures; stage 2 = weighted vote of
1D CNN, TCN, GRU, used ONLY when stage 1 predicts a 'hard' gesture. Hard set and per-gesture
weights come from OTHER subjects' cross-session recall (leave-one-subject-out). Hard labels only.
Run: python scripts/hard_gesture_cascade_check.py"""
import numpy as np, pandas as pd
from pathlib import Path
import importlib.util
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lc", ROOT / "scripts" / "layered_classifier_check.py")
lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
data = lc.load(); S = lc.SUBJECTS; N = lc.NETS; C = lc.NCLS
R = {m: np.stack([lc.per_class_recall(data[(s, "cross")][0], data[(s, "cross")][1][m]) for s in S]) for m in N}
rows = []
for stage1 in N:
    for k in [0, 5, 10, 15, 20, 34]:
        acc, acc_hard_true, base_hard_true = [], [], []
        for i, s in enumerate(S):
            oth = [j for j in range(20) if j != i]
            rec = np.nanmean(np.mean([R[m][oth] for m in N], 0), 0)
            hard = np.argsort(rec)[:k]
            W = np.stack([np.nan_to_num(np.nanmean(R[m][oth], 0), nan=0.5) for m in N]) + 0.01
            true, p = data[(s, "cross")]
            vote = lc.vote(np.stack([p[m] for m in N]), W)
            use2 = np.isin(p[stage1], hard)
            pred = np.where(use2, vote, p[stage1])
            acc.append((pred == true).mean())
            th = np.isin(true, hard)
            if th.any():
                acc_hard_true.append((pred[th] == true[th]).mean()); base_hard_true.append((p[stage1][th] == true[th]).mean())
        rows.append(dict(stage1=stage1, hard_k=k, end_to_end=np.mean(acc) * 100,
                         on_hard_gestures=np.mean(acc_hard_true) * 100 if acc_hard_true else np.nan,
                         stage1_alone_on_hard=np.mean(base_hard_true) * 100 if base_hard_true else np.nan))
df = pd.DataFrame(rows); df.to_csv(lc.OUT / "hard_gesture_cascade_cross.csv", index=False)
print(df.round(1).to_string(index=False))
