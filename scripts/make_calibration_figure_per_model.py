"""Figure 2 split by architecture: cross-session accuracy per trial vs number of unlabelled session 2 trials
used to estimate the normalisation statistics. One panel per network (1D CNN, GRU, TCN) plus single-panel files."""
import glob
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "calibration_per_model"
OUT.mkdir(parents=True, exist_ok=True)
d = pd.concat(pd.read_csv(f, dtype={"subject": str}) for f in sorted(glob.glob(str(ROOT / "reports/calibration_test/subject*.csv"))))
XPOS = {0: 0, 5: 1, 10: 2, 20: 3, 40: 4, 80: 5, "all": 6}
d["x"] = [6 if s == "all" else XPOS[n] for n, s in zip(d.n_cal, d.scheme)]
BLUE, ORANGE = "#2b6bd6", "#e8602c"
MODELS = ["TCN", "1D CNN", "GRU"]
N_SUB = d.subject.nunique()

def panel(ax, m, show_ylabel=True):
    g = d[d.model == m]
    sp = g[g.scheme.isin(["none", "spread", "all"])]
    fi = g[g.scheme.isin(["none", "first", "all"])]
    for sub in g.subject.unique():
        for df, c in ((sp, BLUE), (fi, ORANGE)):
            s = df[df.subject == sub].sort_values("x")
            ax.plot(s.x, s.per_trial, color=c, alpha=0.18, lw=1)
    for df, c, lab, off in ((sp, BLUE, "Calibration trials spread over the session", 4), (fi, ORANGE, "Calibration trials = first N of the session", -13)):
        mu = df.groupby("x").per_trial.mean().sort_index()
        ax.plot(mu.index, mu.values, color=c, lw=3, marker="o", ms=7, label=lab, zorder=3)
        for x, v in mu.items():
            if x in (0, 6) and c == ORANGE:
                continue
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(0, off), ha="center", fontsize=9, color=c)
    ax.set_title(m, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 90); ax.set_yticks(range(0, 91, 10))
    ax.set_xticks(range(7)); ax.set_xticklabels(["0\n(session 1\nstats)", "5", "10", "20", "40", "80", "all\n(own-session\nstats)"], fontsize=9)
    ax.grid(axis="y", color="#e3e3e3"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if show_ylabel: ax.set_ylabel("Accuracy per trial (%)")
    ax.set_xlabel("Session 2 trials used (no labels)\nto estimate the normalisation statistics", fontsize=10)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
for i, (ax, m) in enumerate(zip(axes, MODELS)):
    panel(ax, m, i == 0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"Cross-session accuracy per trial vs unlabelled session 2 trials used for normalisation (thick: mean of {N_SUB} subjects, thin: each subject)", fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(OUT / "fig2_calibration_3panel.png", dpi=200); plt.close(fig)

for m in MODELS:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    panel(ax, m); ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"fig2_calibration_{m.replace(' ', '')}.png", dpi=200); plt.close(fig)

t = d.groupby(["model", "scheme", "n_cal"]).per_trial.mean().round(1)
print(d[d.scheme.isin(["first", "spread"])].groupby(["model", "scheme", "n_cal"]).per_trial.mean().round(1).unstack(0))
