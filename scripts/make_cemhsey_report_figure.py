"""Figure 3 of the Week 10 and 11 report: CEMHSEY S4 accuracy by test day. Run from the repo root:
    python scripts/make_cemhsey_report_figure.py
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "results/cemhsey/S4"
RUNS = [("lda", "LDA", "#2a78d6"), ("cnn1d", "1D CNN", "#eb6834"), ("tcn", "TCN", "#1baf7a"), ("gru_pool8", "GRU", "#eda100")]
INK, GRID = "#222222", "#d9d9d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, key, title in zip(axes, ["per_window_accuracy", "per_trial_accuracy"],
                          ["Per window", "Per gesture (majority vote over windows)"]):
    for run, label, col in RUNS:
        m = json.load(open(f"{R}/{run}/cemhsey/metrics.json"))
        ps = m["cross_session"]["per_session"]
        days = [1] + sorted(int(d) for d in ps)
        vals = [100 * m[key]] + [100 * ps[str(d)][key] for d in days[1:]]
        ax.plot(days, vals, color=col, lw=2, zorder=2)
        ax.plot(days[1:], vals[1:], "o", color=col, ms=6, mec="white", mew=1.2, zorder=3)
        ax.plot(days[:1], vals[:1], "o", color="white", ms=7, mec=col, mew=2, zorder=4)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks(range(1, 12)); ax.set_xlabel("Test day")
    ax.set_ylim(0, 102); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Accuracy (%)")
handles = [Line2D([0], [0], color=c, lw=2, marker="o", ms=6, mec="white", label=l) for _, l, c in RUNS]
handles.append(Line2D([0], [0], color="none", marker="o", ms=7, mfc="white", mec="#555555", mew=2, label="Day 1 = held-out trial 5 (within-day)"))
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
import os
os.makedirs("reports/figures/cemhsey", exist_ok=True)
fig.savefig("reports/figures/cemhsey/figC_cemhsey_by_day.png", dpi=170)
print("ok")
