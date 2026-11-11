"""Compare cross-session accuracy: session 2 normalised with its own statistics (results/hyser_all)
versus session 1 statistics (results/hyser_refnorm). Reads metrics.json only.
Run: python scripts/compare_refnorm.py [--subjects 02,06,19,20]"""
import argparse, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--subjects", default="02,06,19,20")
ap.add_argument("--base", default="results/hyser_all"); ap.add_argument("--new", default="results/hyser_refnorm")
a = ap.parse_args()
subs = [s.strip().zfill(2) for s in a.subjects.split(",")]
MODELS = {"LDA": "lda", "1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}
def get(root, s, d, key):
    f = ROOT / root / s / d / "hyser" / "metrics.json"
    return json.load(open(f))["cross_session"][key] * 100 if f.exists() else np.nan
for key, label in [("per_trial_accuracy", "per trial"), ("per_window_accuracy", "per window")]:
    print(f"\n=== cross-session accuracy, {label} (%), subjects {','.join(subs)} ===")
    print(f"{'subject':8s}" + "".join(f"{m:>22s}" for m in MODELS))
    tot = {m: [[], []] for m in MODELS}
    for s in subs:
        line = f"{s:8s}"
        for m, d in MODELS.items():
            b, n = get(a.base, s, d, key), get(a.new, s, d, key)
            if not (np.isnan(b) or np.isnan(n)):   # like-for-like: only subjects finished on both sides
                tot[m][0].append(b); tot[m][1].append(n)
            line += f"{b:10.1f} -> {n:6.1f}  "
        print(line)
    print(f"{'MEAN':8s}" + "".join(f"{np.nanmean(tot[m][0]):10.1f} -> {np.nanmean(tot[m][1]):6.1f}  " for m in MODELS))
    print("(own-session stats -> session-1 stats; MEAN uses only subjects finished in both runs)")
