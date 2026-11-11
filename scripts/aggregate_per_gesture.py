"""
Per-gesture accuracy across subjects (Omid's feedback: publish accuracy for each of the 34 hand movements).

Reads each subject's saved predictions:
    results/hyser_all/<NN>/<run>/hyser/test_predictions.npz            same session (1 held-out trial per gesture)
    results/hyser_all/<NN>/<run>/hyser/cross_session_predictions.npz   session 2 (needs a re-run with the patched train scripts)
and writes, for each of same / cross:
    reports/per_gesture_long.csv        one row per subject x model x gesture x split
    reports/per_gesture_<split>.csv     34 gestures x 4 models (mean over subjects) with trial counts
    reports/per_gesture_<split>.md      the same as a Markdown table
    reports/per_gesture_matrix_<model>_<split>.csv   20 subjects x 34 gestures, per-trial accuracy %
    reports/figures/per_gesture/heatmap_<model>_<split>.png   the same matrix as a heatmap

Usage (repo root):  python scripts/aggregate_per_gesture.py
Gesture numbers are Hyser's label numbers 1 to 34. Names come from Fig. 2 of the Hyser paper (GESTURE_NAMES below).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"lda": "LDA", "cnn1d": "1D CNN", "tcn": "TCN", "gru_pool8": "GRU"}
# Gesture names from Fig. 2 of Jiang et al., IEEE TNSRE 2021 (Hyser paper); "ext" = extension
GESTURE_NAMES: dict[int, str] = {
    1: "Thumb ext",
    2: "Index ext",
    3: "Middle ext",
    4: "Ring ext",
    5: "Little ext",
    6: "Wrist flex",
    7: "Wrist ext",
    8: "Wrist radial",
    9: "Wrist ulnar",
    10: "Wrist pronation",
    11: "Wrist supination",
    12: "Thumb+index ext",
    13: "Index+middle ext",
    14: "Wrist flex + close",
    15: "Wrist ext + close",
    16: "Wrist radial + close",
    17: "Wrist ulnar + close",
    18: "Wrist pron + close",
    19: "Wrist sup + close",
    20: "Wrist flex + open",
    21: "Wrist ext + open",
    22: "Wrist radial + open",
    23: "Wrist ulnar + open",
    24: "Wrist pron + open",
    25: "Wrist sup + open",
    26: "Thumb+index+middle ext",
    27: "Index+middle+ring ext",
    28: "Middle+ring+little ext",
    29: "Index to little ext",
    30: "Hand close",
    31: "Hand open",
    32: "Thumb+index pinch",
    33: "Thumb+index+middle pinch",
    34: "Thumb+middle pinch",
}
FILES = {"same": "test_predictions.npz", "cross": "cross_session_predictions.npz"}


def gname(g: int) -> str:
    return f"G{g:02d}" + (f" {GESTURE_NAMES[g]}" if g in GESTURE_NAMES else "")


def find(root: Path, sub: str, run: str, fname: str):
    for cand in (root / sub / run / "hyser" / fname, root / sub / run / fname):
        if cand.exists():
            return cand
    return None


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run, label in MODELS.items():
            for split, fname in FILES.items():
                f = find(root, sub_dir.name, run, fname)
                if f is None:
                    continue
                d = np.load(f)
                yt, yp, tid = d["y_true"], d["y_pred"], d["trial_id"]
                tt, tp = d["trial_true"], d["trial_pred"]
                for g in np.unique(tt):
                    tm = tt == g
                    wm = yt == g
                    rows.append({
                        "subject": f"S{sub_dir.name}", "model": label, "split": split, "gesture": int(g) + 1,
                        "n_trials": int(tm.sum()), "trials_correct": int((tp[tm] == tt[tm]).sum()),
                        "n_windows": int(wm.sum()), "windows_correct": int((yp[wm] == yt[wm]).sum()),
                    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/hyser_all"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    a = ap.parse_args()
    df = collect(a.results)
    if df.empty:
        raise SystemExit("no prediction files found")
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "per_gesture_long.csv", index=False)
    (a.out / "figures" / "per_gesture").mkdir(parents=True, exist_ok=True)

    for split in df.split.unique():
        d = df[df.split == split]
        g = d.groupby(["gesture", "model"]).agg(tc=("trials_correct", "sum"), nt=("n_trials", "sum"),
                                                wc=("windows_correct", "sum"), nw=("n_windows", "sum"),
                                                nsub=("subject", "nunique")).reset_index()
        g["trial_acc"] = 100 * g.tc / g.nt
        g["window_acc"] = 100 * g.wc / g.nw
        wide = g.pivot(index="gesture", columns="model", values="trial_acc")[[m for m in MODELS.values() if m in set(g.model)]]
        wide.insert(0, "trials", g.groupby("gesture").nt.first())
        wide.insert(0, "name", [gname(i) for i in wide.index])
        wide.round(1).to_csv(a.out / f"per_gesture_{split}.csv")
        (a.out / f"per_gesture_{split}.md").write_text(wide.round(1).to_markdown())
        print(f"\n== {split} session, per-trial accuracy % by gesture (mean over {g.nsub.max()} subjects) ==")
        print(wide.round(1).to_string())
        for label in sorted(set(d.model)):
            m = d[d.model == label]
            hm = m.assign(acc=100 * m.trials_correct / m.n_trials).pivot(index="subject", columns="gesture", values="acc")
            mat = hm.copy(); mat.columns = [gname(c) for c in mat.columns]
            mat.round(1).to_csv(a.out / f"per_gesture_matrix_{label.replace(' ', '_')}_{split}.csv")
            fig, ax = plt.subplots(figsize=(11, 5.2))
            im = ax.imshow(hm.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, fontsize=7)
            ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index, fontsize=7)
            ax.set_xlabel("Gesture (Hyser label 1 to 34)"); ax.set_ylabel("Subject")
            ax.set_title(f"{label}, {split} session: per-trial accuracy (%) by subject and gesture", fontsize=10, loc="left")
            fig.colorbar(im, ax=ax, label="Accuracy (%)")
            fig.tight_layout()
            fig.savefig(a.out / "figures" / "per_gesture" / f"heatmap_{label.replace(' ', '_')}_{split}.png", dpi=160)
            plt.close(fig)
    print("\nwrote", a.out / "per_gesture_long.csv")


if __name__ == "__main__":
    main()
