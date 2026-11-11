"""Feasibility check for a layered classifier (cascade / per-gesture weighted ensemble).

Uses only the stored trial-level predictions (results/hyser_all/<subj>/<model>/hyser/*.npz).
No retraining. Hard labels only (no probabilities are stored), so voting is hard voting.

Selection rule: any per-gesture weights used for a subject come from OTHER subjects only
(leave-one-subject-out), so nothing from the target subject's own test data is used.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--results-root", default="results/hyser_all", help="e.g. results/hyser_refnorm for the normalisation re-run")
_ap.add_argument("--out", default="reports/layered_check")
_args, _ = _ap.parse_known_args()
RES = ROOT / _args.results_root
OUT = ROOT / _args.out
PROBS = {}  # (subject, split) -> {model: (n_trials, 34) mean window probability}; filled when saved by train.py
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"LDA": "lda", "1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}
NETS = ["1D CNN", "TCN", "GRU"]
SUBJECTS = [f"{i:02d}" for i in range(1, 21)]
SPLITS = {"same": "test_predictions.npz", "cross": "cross_session_predictions.npz"}
NCLS = 34
# Table G1 groups (gesture number 1..34)
GROUPS = ([0] * 5 + [1] * 6 + [2] * 2 + [3] * 6 + [4] * 6 + [5] * 4 + [6] * 2 + [7] * 3)
rng = np.random.default_rng(0)


def load():
    data = {}
    for s in SUBJECTS:
        for split, fn in SPLITS.items():
            true, preds = None, {}
            for m, d in MODELS.items():
                z = np.load(RES / s / d / "hyser" / fn, allow_pickle=True)
                if true is None:
                    true = z["trial_true"]
                assert np.array_equal(true, z["trial_true"]), (s, split, m, "trial misalignment")
                preds[m] = z["trial_pred"]
                if "trial_prob" in z.files:
                    PROBS.setdefault((s, split), {})[m] = z["trial_prob"]
            data[(s, split)] = (true, preds)
    return data


def vote(preds_stack, weights=None):
    """preds_stack: (n_models, n_trials). weights: (n_models, NCLS) or None. Ties -> first model."""
    n_m, n_t = preds_stack.shape
    out = np.zeros(n_t, dtype=int)
    for t in range(n_t):
        score = np.zeros(NCLS)
        for k in range(n_m):
            score[preds_stack[k, t]] += 1.0 if weights is None else weights[k, preds_stack[k, t]]
        best = score.max()
        cands = np.flatnonzero(score == best)
        # tie-break: the first model's vote if it is a candidate, else lowest class id
        out[t] = preds_stack[0, t] if preds_stack[0, t] in cands else cands[0]
    return out


def per_class_recall(true, pred):
    r = np.full(NCLS, np.nan)
    for c in range(NCLS):
        m = true == c
        if m.any():
            r[c] = (pred[m] == c).mean()
    return r


def main():
    data = load()
    rows, recs = [], []
    # ---------- accuracy, oracle, error overlap ----------
    for split in SPLITS:
        for s in SUBJECTS:
            true, p = data[(s, split)]
            correct = {m: p[m] == true for m in MODELS}
            net_c = np.stack([correct[m] for m in NETS])
            row = dict(split=split, subject=s, n_trials=len(true))
            for m in MODELS:
                row[f"acc_{m}"] = correct[m].mean()
            row["oracle_nets"] = net_c.any(0).mean()
            row["oracle_all4"] = np.stack(list(correct.values())).any(0).mean()
            row["all_nets_wrong"] = (~net_c).all(0).mean()
            # independence expectation for all-three-wrong
            row["all_nets_wrong_if_indep"] = np.prod([1 - correct[m].mean() for m in NETS])
            row["hard_vote_nets"] = (vote(np.stack([p[m] for m in NETS])) == true).mean()
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subject_level_oracle_vote.csv", index=False)
    summ = df.groupby("split")[[c for c in df.columns if c not in ("split", "subject", "n_trials")]].mean() * 100
    summ.to_csv(OUT / "summary_oracle_vote.csv")
    print("=== MEAN OVER 20 SUBJECTS (percent, per trial) ===")
    print(summ.round(1).T.to_string())

    # pairwise error overlap (cross), pooled
    pw = []
    for split in SPLITS:
        for a, b in combinations(NETS + ["LDA"], 2):
            both, ea, eb = 0, 0, 0
            for s in SUBJECTS:
                true, p = data[(s, split)]
                wa, wb = p[a] != true, p[b] != true
                both += (wa & wb).sum(); ea += wa.sum(); eb += wb.sum()
            pw.append(dict(split=split, A=a, B=b, P_B_wrong_given_A_wrong=both / ea,
                           P_A_wrong_given_B_wrong=both / eb, jaccard=both / (ea + eb - both)))
    pw = pd.DataFrame(pw)
    pw.to_csv(OUT / "pairwise_error_overlap.csv", index=False)
    print("\n=== PAIRWISE ERROR OVERLAP (pooled trials) ===")
    print(pw.round(3).to_string(index=False))

    # ---------- per-gesture recall tables ----------
    R = {}  # (split, model) -> (20, 34) recall (per-trial, per-subject)
    for split in SPLITS:
        for m in MODELS:
            R[(split, m)] = np.stack([per_class_recall(data[(s, split)][0], data[(s, split)][1][m]) for s in SUBJECTS])
    rank_rows = []
    for split in SPLITS:
        for m in MODELS:
            mat = R[(split, m)]
            mean = np.nanmean(mat, 0)
            boots = np.array([np.nanmean(mat[rng.integers(0, 20, 20)], 0) for _ in range(2000)])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
            for g in range(NCLS):
                rank_rows.append(dict(split=split, model=m, gesture=g + 1, mean=mean[g] * 100,
                                      ci_lo=lo[g] * 100, ci_hi=hi[g] * 100, n_subjects=int((~np.isnan(mat[:, g])).sum())))
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(OUT / "per_gesture_ranking_with_ci.csv", index=False)
    print("\n=== CROSS-SESSION: 8 hardest / 5 easiest gestures (mean over models 1D CNN,TCN,GRU) ===")
    cs = rk[(rk.split == "cross") & rk.model.isin(NETS)].groupby("gesture")["mean"].mean().sort_values()
    print("hardest:", [(int(g), round(v, 1)) for g, v in cs.head(8).items()])
    print("easiest:", [(int(g), round(v, 1)) for g, v in cs.tail(5).items()])
    # rank correlation same vs cross, and between models (cross)
    def sp(a, b):
        return pd.Series(a).rank().corr(pd.Series(b).rank())
    for m in NETS:
        a = rk[(rk.split == "same") & (rk.model == m)].sort_values("gesture")["mean"].values
        b = rk[(rk.split == "cross") & (rk.model == m)].sort_values("gesture")["mean"].values
        print(f"Spearman same-vs-cross gesture ranking, {m}: {sp(a, b):.2f}")
    for a, b in combinations(NETS, 2):
        x = rk[(rk.split == "cross") & (rk.model == a)].sort_values("gesture")["mean"].values
        y = rk[(rk.split == "cross") & (rk.model == b)].sort_values("gesture")["mean"].values
        print(f"Spearman cross-session ranking {a} vs {b}: {sp(x, y):.2f}")

    # ---------- per-gesture weighted vote (leave-one-subject-out weights) ----------
    wres = []
    for wsrc in ["same", "cross"]:
        accs = []
        for i, s in enumerate(SUBJECTS):
            others = [j for j in range(20) if j != i]
            W = np.stack([np.nan_to_num(np.nanmean(R[(wsrc, m)][others], 0), nan=0.5) for m in NETS])
            W = W + 0.01  # avoid all-zero scores
            true, p = data[(s, "cross")]
            pred = vote(np.stack([p[m] for m in NETS]), W)
            accs.append((pred == true).mean())
        wres.append(dict(weights_from=f"other subjects, {wsrc}-session", acc=np.mean(accs) * 100,
                         sd=np.std(accs, ddof=1) * 100))
    # best single network per subject chosen LOSO on cross (upper-ish reference) and overall best
    wres = pd.DataFrame(wres)
    wres.to_csv(OUT / "per_gesture_weighted_vote_cross.csv", index=False)
    print("\n=== PER-GESTURE WEIGHTED HARD VOTE on cross-session (mean over 20 subjects) ===")
    print(wres.round(1).to_string(index=False))
    print("reference: best single net mean = %.1f, plain hard vote = %.1f, oracle(any net correct) = %.1f" % (
        max(summ.loc["cross", f"acc_{m}"] for m in NETS), summ.loc["cross", "hard_vote_nets"], summ.loc["cross", "oracle_nets"]))

    # ---------- where do the errors go? (cascade headroom) ----------
    grp = np.array(GROUPS)
    err_rows = []
    pairs = {}
    for m in NETS:
        tot = within = 0
        for s in SUBJECTS:
            true, p = data[(s, "cross")]
            wrong = p[m] != true
            tot += wrong.sum()
            within += (wrong & (grp[true] == grp[p[m]])).sum()
            for t, q in zip(true[wrong], p[m][wrong]):
                pairs[(int(t) + 1, int(q) + 1)] = pairs.get((int(t) + 1, int(q) + 1), 0) + 1
        err_rows.append(dict(model=m, errors=int(tot), pct_errors_within_same_group=100 * within / tot))
    er = pd.DataFrame(err_rows)
    er.to_csv(OUT / "error_within_group.csv", index=False)
    print("\n=== CROSS-SESSION ERRORS THAT STAY INSIDE THE TRUE GESTURE'S GROUP (Table G1) ===")
    print(er.round(1).to_string(index=False))
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:12]
    pd.DataFrame([dict(true=k[0], predicted=k[1], count=v) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])]).to_csv(OUT / "confusion_pairs_nets_pooled.csv", index=False)
    print("top confused (true->pred, count), 3 nets pooled:", top)

    # ---------- subject concentration of errors ----------
    sub = df[df.split == "cross"].set_index("subject")
    mean_net = sub[[f"acc_{m}" for m in NETS]].mean(1).sort_values()
    errs = (1 - mean_net) * sub.loc[mean_net.index, "n_trials"]
    print("\n=== SUBJECT CONCENTRATION (cross, mean of 3 nets) ===")
    print("weakest 5 subjects:", [(s, round(v * 100, 1)) for s, v in mean_net.head(5).items()])
    print("share of all errors from weakest 5 subjects: %.1f%%" % (100 * errs.head(5).sum() / errs.sum()))

    # ---------- gesture-only headroom: what if the hardest k gestures were perfect? ----------
    print("\n=== HEADROOM IF THE k HARDEST GESTURES (ranked on OTHER subjects) WERE SOLVED, cross, best net GRU ===")
    for k in [5, 10]:
        accs = []
        for i, s in enumerate(SUBJECTS):
            others = [j for j in range(20) if j != i]
            rec = np.nanmean(np.mean([R[("cross", m)][others] for m in NETS], 0), 0)
            hard = np.argsort(rec)[:k]
            true, p = data[(s, "cross")]
            pred = p["GRU"].copy()
            pred[np.isin(true, hard)] = true[np.isin(true, hard)]
            accs.append((pred == true).mean())
        print(f"k={k}: GRU {summ.loc['cross','acc_GRU']:.1f} -> {np.mean(accs)*100:.1f} (perfect specialist upper bound)")

    # ---------- soft voting (only when probabilities were saved by the patched train scripts) ----------
    have = all(all(m in PROBS.get((s, "cross"), {}) for m in NETS) for s in SUBJECTS)
    if have:
        print("\n=== SOFT VOTING on cross-session (probabilities available) ===")
        out_rows = []
        for wsrc in [None, "same", "cross"]:
            accs = []
            for i, s in enumerate(SUBJECTS):
                true, _ = data[(s, "cross")]
                P = np.stack([PROBS[(s, "cross")][m] for m in NETS])
                if wsrc is None:
                    score = P.mean(0)
                else:
                    others = [j for j in range(20) if j != i]
                    W = np.stack([np.nan_to_num(np.nanmean(R[(wsrc, m)][others], 0), nan=0.5) for m in NETS]) + 0.01
                    score = (P * W[:, None, :]).sum(0)
                accs.append((score.argmax(-1) == true).mean())
            out_rows.append(dict(weights="none (mean probability)" if wsrc is None else f"other subjects, {wsrc}-session",
                                 acc=np.mean(accs) * 100, sd=np.std(accs, ddof=1) * 100))
        sv = pd.DataFrame(out_rows)
        sv.to_csv(OUT / "soft_voting_cross.csv", index=False)
        print(sv.round(1).to_string(index=False))
    else:
        print("\n(no saved probabilities in this results root: soft voting skipped)")


if __name__ == "__main__":
    main()
