"""
Solution A (gesture grouping), first test: relabel EXISTING predictions, no retraining.

The saved models predict 34 classes. Here the true and predicted gesture of every trial are mapped
to a group id and accuracy is recomputed. This is a lower bound on what a model trained on grouped
labels would reach (a retrained model cannot predict a removed or merged-away class).

Groupings tested
  none      the 34 gestures (reference)
  A1        anatomical: wrist + hand close (G14-19) and wrist + hand open (G20-25) join their wrist
            movement (G6-11)                                           -> 22 classes
  A1b       A1, plus the three/four finger extensions G26-29 join G12  -> 18 classes
  A2-s1     data-driven. Start from 34 classes and repeatedly merge the pair of classes with the most
            confusion, down to K classes. The confusion comes ONLY from session 1 (same-session test
            trials, pooled over subjects and over the models given), so nothing from session 2 is used.
  A2-s2     same, but the confusion comes from session 2. OPTIMISTIC: it uses the test data to choose
            the groups. Shown only to bound the effect.

Usage (repo root):
    python scripts/test_gesture_grouping.py
    python scripts/test_gesture_grouping.py --results results/hyser_all --ks 26,22,18,14,10
Outputs go to reports/grouping/: grouping_accuracy.csv, groups_A2.json
"""
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = {"LDA": "lda", "1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}
NN = ["1D CNN", "TCN", "GRU"]


def find(root, sub, run, name):
    for p in (root / sub / run / "hyser" / name, root / sub / run / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{root}/{sub}/{run}/.../{name}")


def load(root):
    subs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    data = {}
    for m, run in MODELS.items():
        data[m] = {}
        for s in subs:
            a = np.load(find(root, s, run, "test_predictions.npz"))
            b = np.load(find(root, s, run, "cross_session_predictions.npz"))
            data[m][s] = ((a["trial_true"], a["trial_pred"]), (b["trial_true"], b["trial_pred"]))
    return subs, data


def anatomical(level):
    mp = list(range(34))
    for i in range(13, 19): mp[i] = i - 8      # G14-19 -> G6-11
    for i in range(19, 25): mp[i] = i - 14     # G20-25 -> G6-11
    if level == "A1b":
        for i in (25, 26, 27, 28): mp[i] = 11  # G26-29 -> G12
    return np.array(mp)


def confusion(data, models, subs, which):
    C = np.zeros((34, 34))
    for m in models:
        for s in subs:
            t, p = data[m][s][which]
            for a, b in zip(t, p): C[a, b] += 1
    return C


def greedy_groups(C, k_target):
    """Merge the pair with the largest symmetric confusion until k_target classes remain."""
    groups = [[i] for i in range(34)]
    cur = C.copy()
    while len(groups) > k_target:
        sym = cur + cur.T
        np.fill_diagonal(sym, -1)
        i, j = np.unravel_index(np.argmax(np.triu(sym, 1) + np.tril(np.full_like(sym, -1))), sym.shape)
        if i > j: i, j = j, i
        cur[i, :] += cur[j, :]; cur[:, i] += cur[:, j]
        cur = np.delete(np.delete(cur, j, 0), j, 1)
        groups[i] += groups[j]; del groups[j]
    mp = np.zeros(34, int)
    for gi, g in enumerate(groups):
        for x in g: mp[x] = gi
    return mp, groups


def acc(data, m, subs, which, mp):
    return float(np.mean([(mp[data[m][s][which][0]] == mp[data[m][s][which][1]]).mean() * 100 for s in subs]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "results" / "hyser_all")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "grouping")
    ap.add_argument("--ks", default="30,26,22,18,14,10,6")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    subs, data = load(a.results)
    print(f"{len(subs)} subjects loaded")
    ks = [int(k) for k in a.ks.split(",")]
    maps = {"none": np.arange(34), "A1": anatomical("A1"), "A1b": anatomical("A1b")}
    groups_out = {}
    for tag, which in (("A2-s1", 0), ("A2-s2", 1)):
        C = confusion(data, NN, subs, which)
        for k in ks:
            mp, g = greedy_groups(C, k)
            maps[f"{tag} K={k}"] = mp
            groups_out[f"{tag} K={k}"] = [[f"G{x + 1}" for x in grp] for grp in g if len(grp) > 1]
    rows = ["grouping,classes,model,session1_acc,session2_acc"]
    print(f"{'grouping':14s} cls " + " ".join(f"{m:>14s}" for m in MODELS) + "   (session 2 accuracy, %)")
    for name, mp in maps.items():
        line = f"{name:14s} {len(set(mp)):3d} "
        for m in MODELS:
            s1, s2 = acc(data, m, subs, 0, mp), acc(data, m, subs, 1, mp)
            rows.append(f"{name},{len(set(mp))},{m},{s1:.2f},{s2:.2f}")
            line += f" {s2:13.1f}"
        print(line)
    (a.out / "grouping_accuracy.csv").write_text("\n".join(rows) + "\n")
    json.dump(groups_out, open(a.out / "groups_A2.json", "w"), indent=1)
    print("written to", a.out)


if __name__ == "__main__":
    main()
