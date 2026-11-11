"""Figures for the multi-subject Hyser report. Reads reports/hyser_all_subjects.csv (from aggregate_hyser.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

CSV = Path("reports/hyser_all_subjects.csv")
OUT = Path("reports/figures/hyser_all"); OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["LDA (baseline)", "1D CNN", "TCN", "GRU (pool x8)"]
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # fixed categorical order, slots 1-4
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

d = pd.read_csv(CSV)
n = d.subject.nunique()
def piv(c): return d.pivot(index="subject", columns="model", values=c)[ORDER]

# Figure A: paired per-subject accuracy, same-session vs cross-session
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (col, title) in zip(axes, [("per_trial_%", "Same session (session 1 held-out trials)"),
                                   ("xsession_trial_%", "Cross session (train S1, test S2)")]):
    p = piv(col)
    x = np.arange(len(ORDER))
    for _, row in p.iterrows():
        ax.plot(x, row.values, color="#b9b8b2", lw=0.7, alpha=0.8, zorder=1)
    for i, m in enumerate(ORDER):
        ax.scatter(np.full(n, i), p[m], s=26, color=COL[m], edgecolor=SURF, linewidth=1.0, zorder=3)
        mu = p[m].mean()
        ax.hlines(mu, i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
        ax.text(i + 0.32, mu, f"{mu:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(x, ORDER); ax.set_xlim(-0.5, len(ORDER) - 0.3); ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Per-trial accuracy (%)")
fig.text(0.01, 0.01, f"Dots: one subject each (n={n}). Grey lines join the same subject across models. Black bar: mean.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "figA_paired_accuracy.png", dpi=170); plt.close(fig)

# Figure B: cross-session per-trial accuracy by subject, 1D CNN with LDA marker
p = piv("xsession_trial_%").sort_values("1D CNN", ascending=False)
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(p))
ax.bar(x, p["1D CNN"], width=0.55, color=COL["1D CNN"], zorder=2)
ax.scatter(x, p["LDA (baseline)"], marker="D", s=28, color=COL["LDA (baseline)"], edgecolor=SURF, linewidth=1, zorder=3, label="LDA (baseline)")
ax.scatter(x, p["GRU (pool x8)"], marker="o", s=28, color=COL["GRU (pool x8)"], edgecolor=INK2, linewidth=0.6, zorder=3, label="GRU (pool x8)")
ax.set_xticks(x, [s.replace("subject", "S") for s in p.index], fontsize=8)
ax.set_ylim(0, 100); ax.set_ylabel("Cross-session per-trial accuracy (%)")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=COL["1D CNN"], label="1D CNN (bar)"),
           Line2D([], [], marker="o", ls="", color=COL["GRU (pool x8)"], markeredgecolor=INK2, markeredgewidth=0.6, label="GRU (pool x8)"),
           Line2D([], [], marker="D", ls="", color=COL["LDA (baseline)"], label="LDA (baseline)")]
ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "figB_xsession_by_subject.png", dpi=170); plt.close(fig)
print("wrote", list(OUT.glob("*.png")))
