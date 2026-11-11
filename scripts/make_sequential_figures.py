"""Figures for the offline sequential classifier study.

Reads reports/sequential_offline/{summary,risk_coverage,groups}.* and writes
reports/sequential_offline/figures/fig_seq_groups.png and fig_seq_coverage.png.
Run after scripts/sequential_offline.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "reports" / "sequential_offline"
FIG = IN / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
})
ORDER = ["A1", "G8", "H5", "D1-5", "D1-8", "D2-5"]
LABEL = {"A1": "A1 (22)", "G8": "G8 (8)", "H5": "H5 (5)", "D1-5": "D1-5 (5)",
         "D1-8": "D1-8 (8)", "D2-5": "D2-5 (5)"}


def chance_containment(gmap):
    gmap = np.asarray(gmap)
    size = np.bincount(gmap)
    return 100 * np.mean([(size[gmap[i]] - 1) / 33 for i in range(34)])


def main():
    s = pd.read_csv(IN / "summary.csv")
    flat = s[(s.split == "cross") & (s.arbiter == "Soft vote") & (s.decoder == "flat")].set_index("grouping")
    groups = json.load(open(IN / "groups.json"))
    chance = {}
    for g in ORDER:
        if g in groups["anatomical"]:
            chance[g] = chance_containment(groups["anatomical"][g])
        else:
            chance[g] = np.mean([chance_containment(v[g]) for v in groups["data"].values()])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    ax = axes[0]
    vals = [flat.loc[g, "group_acc"] for g in ORDER]
    ax.barh(y, vals, height=0.55, color=BLUE)
    ax.axvline(flat.loc["H5", "gesture_acc"], color=INK2, lw=1, ls="--", label="34-gesture accuracy")
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Group-level accuracy, day 2 (%)")
    ax.set_title("(a) Group decided correctly", fontsize=10, loc="left")

    ax = axes[1]
    obs = [flat.loc[g, "containment"] for g in ORDER]
    ch = [chance[g] for g in ORDER]
    ax.barh(y + 0.15, obs, height=0.28, color=BLUE, label="Observed")
    ax.barh(y - 0.15, ch, height=0.28, color=ORANGE, label="Chance (random error)")
    for yi, v in zip(y, obs):
        ax.text(v + 1, yi + 0.15, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Errors inside the true group (%)")
    ax.set_title("(b) Error containment", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[g] for g in ORDER])
    for a in axes:
        a.grid(axis="x", color=GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_seq_groups.png", dpi=300)
    plt.close(fig)

    rc = pd.read_csv(IN / "risk_coverage.csv")
    m = rc.groupby(["grouping", "coverage"]).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    cov = m.loc["D1-5"].index.values * 100
    ax.plot(cov, m.loc["D1-5"].flat_gesture, color=BLUE, lw=2, marker="o", ms=4, label="Gesture (34 classes)")
    ax.plot(cov, m.loc["D1-5"].seq_group, color=ORANGE, lw=2, marker="o", ms=4, label="Group, D1-5")
    ax.plot(cov, m.loc["H5"].seq_group, color=AQUA, lw=2, marker="o", ms=4, label="Group, H5")
    ax.set_xlabel("Coverage: share of trials kept, most confident first (%)")
    ax.set_ylabel("Accuracy on kept trials (%)")
    ax.set_xlim(5, 102)
    ax.set_ylim(30, 100)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_seq_coverage.png", dpi=300)
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
