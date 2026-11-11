"""Offline study of the grouped sequential classifier (no retraining).

Uses the stored trial probabilities of the three networks trained on session 1
(results/hyser_refnorm: session 2 is normalised with SESSION 1 statistics, the
calibration-free case a device without recalibration would face).

For every subject it compares, on all session 2 trials:
    flat      34-class decision of one network or of an arbiter across networks
    seq       two-stage decision: group first, then gesture inside the group,
              with an optional confidence gate that falls back to flat
under several groupings:
    anatomical: H5 (five joint-based groups), G8 (Hyser gesture families), A1
    data-driven, learned without the target subject's session 2:
        D1-K   soft confusion of session 1 trials of the OTHER subjects
        D2-K   soft confusion of session 2 trials of the OTHER subjects
               (population day-2 errors; uses other people's labels only)

No threshold or group is tuned on the target subject's session 2.

Outputs (reports/sequential_offline/):
    per_subject.csv        one row per subject x arbiter x grouping x decoder
    summary.csv            means and paired comparison with the flat decision
    groups.json            the groupings (data-driven ones per left-out subject)
    gate_sweep.csv         accuracy and routing rate against tau_group
    risk_coverage.csv      selective accuracy against coverage
    figures/*.png

Usage (repo root):
    python scripts/sequential_offline.py
    python scripts/sequential_offline.py --results-root results/hyser_refnorm --out reports/sequential_offline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sequential import groups as G          # noqa: E402
from src.sequential import decode as D          # noqa: E402

NETS = {"1D CNN": "cnn1d", "TCN": "tcn", "GRU": "gru_pool8"}
SUBJECTS = [f"{i:02d}" for i in range(1, 21)]
WITHIN_COST = 0.5
TAU_FIXED = 0.5            # stage 1 gate fixed in advance (not tuned)
DATA_K = (5, 8)
MAX_GROUP = 12             # cap on data-driven group size
rng = np.random.default_rng(0)


def load(res: Path):
    data = {}
    for s in SUBJECTS:
        d = {}
        for split, fn in (("same", "test_predictions.npz"), ("cross", "cross_session_predictions.npz")):
            y, probs = None, {}
            for name, run in NETS.items():
                z = np.load(res / s / run / "hyser" / fn)
                if y is None:
                    y = z["trial_true"]
                assert np.array_equal(y, z["trial_true"]), (s, split, name)
                probs[name] = z["trial_prob"].astype(np.float64)
            d[split] = (y, probs)
        data[s] = d
    return data


def arbiters(probs):
    nets = [probs[n] for n in NETS]
    return {
        "GRU": probs["GRU"],
        "1D CNN": probs["1D CNN"],
        "TCN": probs["TCN"],
        "Soft vote": D.soft_vote(nets),
        "Arbiter (conf.-weighted)": D.confidence_weighted_vote(nets),
    }


def data_groupings(data, target):
    """Data-driven groupings for one left-out subject."""
    others = [s for s in SUBJECTS if s != target]
    out = {}
    for tag, split in (("D1", "same"), ("D2", "cross")):
        ys, ps = [], []
        for s in others:
            y, probs = data[s][split]
            ys.append(y)
            ps.append(D.soft_vote([probs[n] for n in NETS]))
        conf = G.soft_confusion(ys, ps)
        for k in DATA_K:
            out[f"{tag}-{k}"] = G.cluster_groups(conf, k, MAX_GROUP)
    return out


def evaluate(y, p, gmap, tau_group=0.0, tau_gesture=0.0):
    flat = p.argmax(1)
    dec = D.sequential_decode(p, gmap, tau_group, tau_gesture)
    conf_flat = p.max(1)
    conf_seq = dec.conf_group * dec.conf_gesture
    res = {}
    for name, yhat, ghat, conf, deferred in (
        ("flat", flat, gmap[flat], conf_flat, None),
        ("seq", dec.gesture, dec.group, conf_seq, dec.deferred),
    ):
        acc = D.gesture_accuracy(y, yhat) if deferred is None else float(np.mean((y == yhat) & ~deferred))
        res[name] = dict(
            gesture_acc=100 * acc,
            group_acc=100 * D.group_accuracy(y, ghat, gmap),
            containment=100 * D.error_containment(y, yhat, gmap),
            severity=100 * D.severity_score(y, yhat, gmap, WITHIN_COST, ghat, deferred),
            acc_at_80=100 * D.accuracy_at_coverage(y, yhat, conf, 0.8),
            acc_at_50=100 * D.accuracy_at_coverage(y, yhat, conf, 0.5),
            routed=100 * (dec.routed.mean() if name == "seq" else 0.0),
            deferred=100 * (deferred.mean() if deferred is not None else 0.0),
        )
    return res


def paired(a, b):
    """Mean of a-b over subjects, bootstrap 95% CI, Wilcoxon p, wins/losses."""
    d = np.asarray(a) - np.asarray(b)
    boots = rng.choice(d, (10000, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), lo, hi, p, int((d > 0).sum()), int((d < 0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results/hyser_refnorm")
    ap.add_argument("--out", default="reports/sequential_offline")
    args = ap.parse_args()
    res_root, out = ROOT / args.results_root, ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data = load(res_root)
    rows, saved_groups = [], {"anatomical": {k: v.tolist() for k, v in G.ANATOMICAL.items()}, "data": {}}

    for s in SUBJECTS:
        groupings = dict(G.ANATOMICAL)
        dg = data_groupings(data, s)
        groupings.update(dg)
        saved_groups["data"][s] = {k: v.tolist() for k, v in dg.items()}
        for split in ("same", "cross"):
            y, probs = data[s][split]
            for arb, p in arbiters(probs).items():
                for gname, gmap in groupings.items():
                    for dec_name, tg in (("ungated", 0.0), ("gated", TAU_FIXED)):
                        r = evaluate(y, p, gmap, tau_group=tg)
                        for kind in ("flat", "seq"):
                            if kind == "flat" and dec_name == "gated":
                                continue
                            rows.append(dict(subject=s, split=split, arbiter=arb, grouping=gname,
                                             K=G.n_groups(gmap), decoder="flat" if kind == "flat" else f"seq-{dec_name}",
                                             **r[kind]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_subject.csv", index=False)
    json.dump(saved_groups, open(out / "groups.json", "w"), indent=1)

    # ---------------- summary with paired tests against flat (same arbiter, same grouping)
    summ = []
    metrics = ["gesture_acc", "group_acc", "containment", "severity", "acc_at_80", "acc_at_50", "routed"]
    for (split, arb, gname), g in df.groupby(["split", "arbiter", "grouping"]):
        flat = g[g.decoder == "flat"].set_index("subject")
        for dec in ("flat", "seq-ungated", "seq-gated"):
            cur = g[g.decoder == dec].set_index("subject").loc[flat.index]
            row = dict(split=split, arbiter=arb, grouping=gname, K=int(cur.K.iloc[0]), decoder=dec)
            for m in metrics:
                row[m] = cur[m].mean()
                row[m + "_sd"] = cur[m].std()
            if dec != "flat":
                for m in ("gesture_acc", "severity", "group_acc"):
                    md, lo, hi, p, w, l = paired(cur[m], flat[m])
                    row.update({f"d_{m}": md, f"d_{m}_lo": lo, f"d_{m}_hi": hi, f"p_{m}": p,
                                f"wins_{m}": w, f"losses_{m}": l})
            summ.append(row)
    summary = pd.DataFrame(summ)
    # arbiter comparison against GRU alone, flat decision
    arb_rows = []
    for split in ("same", "cross"):
        base = df[(df.split == split) & (df.decoder == "flat") & (df.grouping == "H5") & (df.arbiter == "GRU")].set_index("subject")
        for arb in df.arbiter.unique():
            cur = df[(df.split == split) & (df.decoder == "flat") & (df.grouping == "H5") & (df.arbiter == arb)].set_index("subject").loc[base.index]
            md, lo, hi, p, w, l = paired(cur.gesture_acc, base.gesture_acc)
            arb_rows.append(dict(split=split, arbiter=arb, gesture_acc=cur.gesture_acc.mean(), sd=cur.gesture_acc.std(),
                                 d_vs_gru=md, lo=lo, hi=hi, p=p, wins=w, losses=l))
    pd.DataFrame(arb_rows).to_csv(out / "arbiter_vs_gru.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    # ---------------- gate sweep and risk-coverage for the main arbiter
    taus = np.round(np.linspace(0, 0.95, 20), 3)
    sweep, rc = [], []
    main_arb = "Soft vote"
    for s in SUBJECTS:
        y, probs = data[s]["cross"]
        p = arbiters(probs)[main_arb]
        groupings = dict(G.ANATOMICAL)
        groupings.update({k: np.array(v) for k, v in saved_groups["data"][s].items()})
        for gname in ("H5", "G8", "A1", "D1-5", "D1-8", "D2-5", "D2-8"):
            gmap = groupings[gname]
            for t in taus:
                dec = D.sequential_decode(p, gmap, tau_group=t)
                sweep.append(dict(subject=s, grouping=gname, tau=t,
                                  gesture_acc=100 * D.gesture_accuracy(y, dec.gesture),
                                  routed=100 * dec.routed.mean()))
            dec = D.sequential_decode(p, gmap)
            q = D.group_probs(p, gmap)
            for cov in np.linspace(0.1, 1.0, 10):
                rc.append(dict(subject=s, grouping=gname, coverage=round(cov, 2),
                               flat_gesture=100 * D.accuracy_at_coverage(y, p.argmax(1), p.max(1), cov),
                               seq_gesture=100 * D.accuracy_at_coverage(y, dec.gesture, dec.conf_group * dec.conf_gesture, cov),
                               seq_group=100 * D.accuracy_at_coverage(gmap[y], q.argmax(1), q.max(1), cov)))
    pd.DataFrame(sweep).to_csv(out / "gate_sweep.csv", index=False)
    pd.DataFrame(rc).to_csv(out / "risk_coverage.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
