"""
Collect per-subject Stage 3 results into one table + figure for the cross-subject summary.

    python scripts/aggregate_hyser.py                       # reads results/hyser_all/*, optional subject 01 from results/checkpoints
    python scripts/aggregate_hyser.py --results results/hyser_all --out reports/hyser_all_subjects

Writes <out>.csv (one row per subject x model), <out>_summary.csv/.md (mean +- SD over subjects)
and <out>_boxplot.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = {"lda": "LDA (baseline)", "cnn1d": "1D CNN", "tcn": "TCN", "gru_pool8": "GRU (pool x8)"}


def collect(root: Path, subject_label=None):
    rows = []
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run, label in MODELS.items():
            f = sub_dir / run / "hyser" / "metrics.json"
            if not f.exists():  # subject 01 proof-run layout has no dataset sub-folder
                f = sub_dir / run / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            cs = m.get("cross_session", {})
            rows.append({
                "subject": subject_label or f"subject{sub_dir.name}",
                "model": label,
                "per_window_%": 100 * m["per_window_accuracy"],
                "per_trial_%": 100 * m["per_trial_accuracy"],
                "xsession_window_%": 100 * cs.get("per_window_accuracy", float("nan")),
                "xsession_trial_%": 100 * cs.get("per_trial_accuracy", float("nan")),
                "parameters": m.get("parameters"),
                "MFLOPs": (m.get("flops_per_inference") or float("nan")) / 1e6,
                "latency_ms": m.get("latency_ms_desktop_cpu"),
                "train_s": m.get("train_seconds"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/hyser_all"))
    ap.add_argument("--out", type=Path, default=Path("reports/hyser_all_subjects"))
    a = ap.parse_args()
    df = pd.DataFrame(collect(a.results))
    if df.empty:
        raise SystemExit(f"No metrics.json found under {a.results}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{a.out}.csv", index=False)

    metrics = ["per_window_%", "per_trial_%", "xsession_window_%", "xsession_trial_%"]
    order = list(MODELS.values())
    g = df.groupby("model")
    summ = pd.DataFrame({m: g[m].mean().round(1).astype(str) + " ± " + g[m].std().round(1).astype(str) for m in metrics})
    summ.insert(0, "n_subjects", df.groupby("model")["subject"].nunique())
    summ = summ.reindex([m for m in order if m in summ.index])
    summ.to_csv(f"{a.out}_summary.csv")
    Path(f"{a.out}_summary.md").write_text(summ.to_markdown())
    print(summ.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (col, title) in zip(axes, [("per_trial_%", "Same-session (session 1 test), per-trial"),
                                       ("xsession_trial_%", "Cross-session (train S1, test S2), per-trial")]):
        data = [df[df.model == m][col].dropna() for m in order if m in set(df.model)]
        ax.boxplot(data, labels=[m for m in order if m in set(df.model)], showmeans=True)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle(f"Hyser, subject-dependent, {df.subject.nunique()} subjects", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{a.out}_boxplot.png", dpi=160)
    print(f"Wrote {a.out}.csv, _summary.csv/.md, _boxplot.png")


if __name__ == "__main__":
    main()
