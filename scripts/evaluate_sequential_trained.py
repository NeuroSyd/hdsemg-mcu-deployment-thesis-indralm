"""Aggregate the trained sequential classifier and the adaptation ladder (Hyser, session 2).

Reads results/hyser_seq/<subj>/seq_predictions.npz (scripts/run_sequential_trained.py) and writes
reports/sequential_trained/:
    per_subject.csv   subject x model x condition x decoder metrics
    ladder.csv        accuracy against user effort: no adaptation, label-free (N unlabelled
                      trials), labelled last layer with 1 and 2 trials per gesture, each with
                      the flat and the grouped sequential decoder
    summary.csv       means, paired differences against the flat calibration-free decision
    figures/fig_ladder.png
and, with --latex-dir, LaTeX table rows tab_ladder.tex and tab_seq_trained.tex for the report.

All metrics use only the scored session 2 trials (labelled baseline shots removed for every condition).
The ensemble ("Soft vote") averages the flat, group and within-group probabilities over the three networks.

Usage:
    python scripts/evaluate_sequential_trained.py
    python scripts/evaluate_sequential_trained.py --latex-dir ../ProgressReport/generated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.sequential import decode as D   # noqa: E402
from src.sequential import groups as G   # noqa: E402
from src.sequential import heads as H    # noqa: E402

RUNS = {"cnn1d": "1D CNN", "tcn": "TCN", "gru_pool8": "GRU"}
TAU = 0.5
rng = np.random.default_rng(0)


def gmap_for(name, sub, groups_json):
    if name in G.ANATOMICAL:
        return G.ANATOMICAL[name]
    return np.asarray(groups_json["data"][sub][name])


def metrics(y, yhat, gmap, ghat=None):
    ghat = gmap[yhat] if ghat is None else ghat
    return dict(gesture_acc=100 * D.gesture_accuracy(y, yhat),
                group_acc=100 * D.group_accuracy(y, ghat, gmap),
                containment=100 * D.error_containment(y, yhat, gmap),
                severity=100 * D.severity_score(y, yhat, gmap, 0.5, ghat))


def models_in(z):
    return [r for r in RUNS if f"{r}__s1stats__flat" in z.files]


def get(z, runs, key):
    """Mean over networks of an array stored per run (soft vote when len(runs) > 1)."""
    return np.mean([z[f"{r}__{key}"] for r in runs], axis=0)


def tex_rows(lines):
    """Join rows; ASCII minus before a digit becomes a LaTeX math minus (no dashes in the text)."""
    import re
    return re.sub(r"(?<![A-Za-z$])-(?=\d)", "$-$", "\n".join(lines)) + "\n"


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    boots = rng.choice(d, (10000, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return d.mean(), lo, hi, p, int((d > 0).sum()), int((d < 0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-root", default="results/hyser_seq")
    ap.add_argument("--groups", default="reports/sequential_offline/groups.json")
    ap.add_argument("--grouping", default="D1-5", help="main grouping for the ladder")
    ap.add_argument("--out", default="reports/sequential_trained")
    ap.add_argument("--latex-dir", default=None)
    a = ap.parse_args()
    out = REPO / a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    groups_json = json.load(open(REPO / a.groups))

    rows = []
    for f in sorted((REPO / a.seq_root).glob("*/seq_predictions.npz")):
        sub = f.parent.name
        z = np.load(f)
        info = json.load(open(f.parent / "seq_info.json"))
        runs = models_in(z)
        y2, m = z["y2"], z["eval_mask"]
        y = y2[m]
        for model_name, rs in [(RUNS[r], [r]) for r in runs] + [("Soft vote", runs)]:
            for gname in info["groupings"]:
                gmap = gmap_for(gname, sub, groups_json)
                k = G.n_groups(gmap)
                # conditions: (label, flat key, grouped key or None)
                conds = [("No adaptation", "s1stats__flat", "s1stats"),
                         (f"Label-free, {info['n_cal']} unlabelled trials", "cal__flat", "cal"),
                         ("Labelled last layer, 1 trial per gesture", "ll1__flat", None),
                         ("Labelled last layer, 2 trials per gesture", "ll2__flat", None)]
                for cname, fkey, gkey in conds:
                    if f"{rs[0]}__{fkey}" not in z.files:
                        continue
                    p = get(z, rs, fkey)[m]
                    yhat = p.argmax(1)
                    rows.append(dict(subject=sub, model=model_name, grouping=gname, K=k, condition=cname,
                                     decoder="flat", **metrics(y, yhat, gmap)))
                    if gkey is None:
                        continue
                    pg = get(z, rs, f"{gkey}__{gname}__group")[m]
                    pw = get(z, rs, f"{gkey}__{gname}__within")[m]
                    for dname, tau in (("sequential", 0.0), ("sequential, gated", TAU)):
                        dec = D.sequential_decode(p, gmap, tau_group=tau, p_group=pg,
                                                  p_within=H.within_list(pw, gmap))
                        rows.append(dict(subject=sub, model=model_name, grouping=gname, K=k, condition=cname,
                                         decoder=dname, routed=100 * dec.routed.mean(),
                                         **metrics(y, dec.gesture, gmap, dec.group)))
    if not rows:
        sys.exit(f"no results under {a.seq_root}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_subject.csv", index=False)

    # paired comparison with the flat, calibration-free decision of the same model and grouping
    summ = []
    for (mdl, g, cond, dec), d in df.groupby(["model", "grouping", "condition", "decoder"]):
        base = df[(df.model == mdl) & (df.grouping == g) & (df.condition == "No adaptation") &
                  (df.decoder == "flat")].set_index("subject")
        cur = d.set_index("subject").loc[base.index]
        row = dict(model=mdl, grouping=g, condition=cond, decoder=dec, n_subjects=len(cur))
        for k in ("gesture_acc", "group_acc", "containment", "severity"):
            row[k] = cur[k].mean()
            row[k + "_sd"] = cur[k].std()
        md, lo, hi, p, w, l = paired(cur.gesture_acc, base.gesture_acc)
        row.update(d_gesture=md, d_lo=lo, d_hi=hi, p=p, wins=w, losses=l)
        md, lo, hi, p, w, l = paired(cur.severity, base.severity)
        row.update(d_severity=md, d_sev_lo=lo, d_sev_hi=hi, p_severity=p)
        summ.append(row)
    summary = pd.DataFrame(summ)
    summary.to_csv(out / "summary.csv", index=False)

    lad = summary[(summary.model == "Soft vote") & (summary.grouping == a.grouping)].copy()
    lad.to_csv(out / "ladder.csv", index=False)

    # figure: accuracy against user effort (labelled trials needed), flat and sequential
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    canon = ["No adaptation", "Label-free", "Labelled last layer, 1", "Labelled last layer, 2"]
    present = list(dict.fromkeys(lad.condition))
    order = [c for k in canon for c in present if c.startswith(k)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(order))
    for dec, col, off in (("flat", "#2a78d6", -0.18), ("sequential, gated", "#eb6834", 0.18)):
        v = [lad[(lad.condition == c) & (lad.decoder == dec)].gesture_acc.mean() for c in order]
        e = [lad[(lad.condition == c) & (lad.decoder == dec)].gesture_acc_sd.mean() / np.sqrt(
            max(1, lad.n_subjects.max())) for c in order]
        ax.bar(x + off, v, width=0.34, color=col, label="Flat 34-class" if dec == "flat" else "Grouped sequential")
        ax.errorbar(x + off, v, yerr=e, fmt="none", ecolor="#52514e", lw=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(", ", ",\n") for c in order], fontsize=8)
    ax.set_ylabel("Day 2 gesture accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig_ladder.png", dpi=300)
    plt.close(fig)

    if a.latex_dir:
        ld = Path(a.latex_dir)
        ld.mkdir(parents=True, exist_ok=True)
        lines = []
        for c in order:
            for dec in ("flat", "sequential, gated"):
                r = lad[(lad.condition == c) & (lad.decoder == dec)]
                if r.empty:
                    continue
                r = r.iloc[0]
                diff = "reference" if (c == "No adaptation" and dec == "flat") else \
                    f"{r.d_gesture:+.1f} ({r.d_lo:+.1f} to {r.d_hi:+.1f})"
                short = (c.replace("Labelled last layer, ", "Labelled, ").replace(" trials per gesture", " per gesture")
                         .replace(" trial per gesture", " per gesture").replace(" unlabelled trials", " unlabelled"))
                lines.append(f"{short} & {'Flat' if dec == 'flat' else 'Sequential (gated)'} & "
                             f"{r.gesture_acc:.1f} $\\pm$ {r.gesture_acc_sd:.1f} & {r.group_acc:.1f} & "
                             f"{r.severity:.1f} & {diff} \\\\")
        (ld / "tab_ladder.tex").write_text(tex_rows(lines))
        per_model = summary[(summary.grouping == a.grouping) & (summary.condition == "No adaptation")]
        lines = []
        for mdl in ["1D CNN", "TCN", "GRU", "Soft vote"]:
            for dec in ("flat", "sequential", "sequential, gated"):
                r = per_model[(per_model.model == mdl) & (per_model.decoder == dec)]
                if r.empty:
                    continue
                r = r.iloc[0]
                diff = "reference" if dec == "flat" else f"{r.d_gesture:+.1f} ({r.d_lo:+.1f} to {r.d_hi:+.1f})"
                lines.append(f"{mdl} & {dict(flat='Flat', sequential='Sequential').get(dec, 'Gated')} & {r.gesture_acc:.1f} & {r.group_acc:.1f} & "
                             f"{r.containment:.1f} & {r.severity:.1f} & {diff} \\\\")
        (ld / "tab_seq_trained.tex").write_text(tex_rows(lines))
        n = int(lad.n_subjects.max())
        (ld / "numbers_seq_trained.tex").write_text(f"\\newcommand{{\\SeqTrainedN}}{{{n}}}\n")
        import shutil
        shutil.copy(out / "figures" / "fig_ladder.png", ld / "fig_ladder.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
