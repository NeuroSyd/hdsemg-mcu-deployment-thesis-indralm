"""
Evidence figures and summary table for a Stage 0-3 progress report.

Reads the pipeline's own outputs (manifests, preprocessed arrays, results/
metrics) and writes PNG figures plus a markdown/CSV summary, so every number
and image in a progress report can be regenerated with one command:

    python src/evaluation/progress_figures.py --subject 01

Outputs go to reports/figures/subject<NN>/ and reports/subject<NN>_stage0_3_summary.{md,csv}.
Figures whose inputs are missing (e.g. a model not trained yet) are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.filters import apply_filter_chain
from src.preprocessing.io_hyser import read_hyser_record

# Categorical slots in fixed order (reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Run keys = results/checkpoints/<key>/<dataset>/. "gru_pool8" is the GRU with an 8x average
# pool on the time axis (pool factor chosen on validation accuracy); "gru" is the untuned GRU on
# the raw 410-step sequence, kept as the reference it was tuned against.
MODEL_ORDER = ["lda", "cnn1d", "tcn", "gru_pool8", "gru"]
DEEP_ORDER = ["cnn1d", "tcn", "gru_pool8", "gru"]
MODEL_LABEL = {"lda": "LDA (baseline)", "cnn1d": "1D CNN", "tcn": "TCN",
               "gru_pool8": "GRU, time pooled x8", "gru": "GRU, raw 410 steps"}
MODEL_COLOR = {"lda": BLUE, "cnn1d": ORANGE, "tcn": AQUA, "gru_pool8": YELLOW, "gru": MAGENTA}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def save(fig, out: Path, name: str):
    fig.savefig(out / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out / name)


# ---------------------------------------------------------------- Stage 0
def fig_dataset_overview(man: pd.DataFrame, out: Path):
    sessions = sorted(man["session"].unique())
    fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 2.6 * len(sessions)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sess in zip(axes, sessions):
        counts = man[man["session"] == sess]["gesture_label"].value_counts().reindex(range(1, 35), fill_value=0)
        colors = [BLUE if c == 6 else ORANGE for c in counts.values]
        ax.bar(counts.index, counts.values, color=colors, width=0.65)
        ax.axhline(6, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.set_ylim(0, 7.5)
        ax.set_ylabel("Trials")
        n_missing = int((counts == 0).sum())
        ax.set_title(f"Session {sess}: {int(counts.sum())} dynamic trials"
                     + (f", {n_missing} gesture with no usable trial" if n_missing else ""))
        for g, c in counts.items():
            if c != 6:
                ax.text(g, c + 0.2, str(c), ha="center", fontsize=8, color=INK)
    axes[-1].set_xlabel("Gesture ID (dashed line = 6 repetitions, the nominal protocol)")
    axes[-1].set_xticks(range(1, 35, 1))
    axes[-1].tick_params(axis="x", labelsize=7)
    save(fig, out, "fig1_stage0_trials_per_gesture.png")


# ---------------------------------------------------------------- Stage 1
def fig_filtering(pre_cfg: dict, raw_path: Path, out: Path):
    sig, fs, _ = read_hyser_record(raw_path)
    filt = apply_filter_chain(sig, fs, pre_cfg)
    f, p_raw = welch(sig, fs, nperseg=1024, axis=-1)
    _, p_flt = welch(filt, fs, nperseg=1024, axis=-1)
    p_raw, p_flt = p_raw.mean(0), p_flt.mean(0)

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.35, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(f, p_raw, color=INK2, lw=1.4, label="Raw")
    ax.semilogy(f, p_flt, color=BLUE, lw=1.4, label="After bandpass + notch")
    ax.set_xlim(0, 600)
    band = (f > 15) & (f < 595)
    ax.set_ylim(p_flt[band].min() / 5, p_raw.max() * 3)
    ax.axvspan(0, pre_cfg["bandpass"]["low_hz"], color=GRID, alpha=0.8, lw=0)
    ax.axvspan(pre_cfg["bandpass"]["high_hz"], 600, color=GRID, alpha=0.8, lw=0)
    ax.axvline(50, color=ORANGE, lw=1, ls=(0, (4, 3)))
    ax.text(54, 0.95, "50 Hz mains", color=INK, fontsize=8, transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density, mean of 256 channels (V²/Hz)")
    ax.set_title("Spectrum before vs after filtering")
    ax.legend(frameon=False, loc="upper right")

    t = np.arange(sig.shape[1]) / fs * 1000
    ch = 100
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, sig[ch] * 1e3, color=INK2, lw=0.8)
    a1.set_ylabel("mV"); a1.set_title(f"Channel {ch}, raw"); a1.tick_params(labelbottom=False)
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, filt[ch] * 1e3, color=BLUE, lw=0.8)
    a2.set_ylabel("mV"); a2.set_xlabel("Time (ms)"); a2.set_title("Same channel, filtered")
    save(fig, out, "fig2_stage1_filtering.png")


def fig_spatial_rms(man: pd.DataFrame, raw_dir: Path, out: Path, gestures=(1, 12, 30)):
    hdr = None
    rows = man[man["session"] == man["session"].min()]
    fig, axes = plt.subplots(len(gestures), 4, figsize=(9, 2.3 * len(gestures)))
    vmax = 0
    maps = {}
    for g in gestures:
        row = rows[rows["gesture_label"] == g].iloc[0]
        x = np.load(row["preprocessed_path"])
        rms = np.sqrt((x ** 2).mean(axis=1))
        if hdr is None:
            import wfdb
            hdr = wfdb.rdheader(str(Path(row["file_path"]).with_suffix("")))
        grids = {}
        for ci, name in enumerate(hdr.sig_name):
            arr, r, c = name.split("-")
            grids.setdefault(arr, np.full((8, 8), np.nan))[int(r) - 1, int(c) - 1] = rms[ci]
        maps[g] = grids
        vmax = max(vmax, np.nanmax([np.nanmax(v) for v in grids.values()]))
    order = ["ED", "EP", "FD", "FP"]
    for i, g in enumerate(gestures):
        for j, arr in enumerate(order):
            ax = axes[i, j]
            im = ax.imshow(maps[g][arr], cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(arr, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Gesture {g}", fontsize=10)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("RMS of normalised signal (σ units)")
    fig.suptitle("Spatial activation per 8×8 array after Stage 1 (one trial per gesture)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    save(fig, out, "fig3_stage1_spatial_rms.png")


# ---------------------------------------------------------------- Stage 2
def fig_windowing(man: pd.DataFrame, win_cfg: dict, out: Path):
    row = man.iloc[0]
    x = np.load(row["preprocessed_path"])
    fs = win_cfg["sampling_rate_hz"]
    wl = int(round(win_cfg["window_ms"] / 1000 * fs))
    step = int(round(wl * (1 - win_cfg["overlap"])))
    env = np.abs(x).mean(0)
    k = 41
    env = np.convolve(env, np.ones(k) / k, mode="same")
    t = np.arange(x.shape[1]) / fs * 1000
    n_win = 1 + (x.shape[1] - wl) // step

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, env, color=INK2, lw=1.4)
    ax.set_ylabel("Mean |signal| over 256 ch (σ units)")
    ax.set_xlabel("Time within one 1 s dynamic trial (ms)")
    base = env.min()
    span = env.max() - env.min()
    for i in range(n_win):
        s0, s1 = i * step / fs * 1000, (i * step + wl) / fs * 1000
        lane = i % 2
        y = base - span * (0.12 + 0.12 * lane)
        ax.plot([s0, s1], [y, y], color=BLUE if lane == 0 else ORANGE, lw=5, solid_capstyle="butt")
        if lane == 0:
            ax.text((s0 + s1) / 2, y + span * 0.05, f"W{i + 1}", ha="center", va="bottom", fontsize=8, color=INK)
        else:
            ax.text((s0 + s1) / 2, y - span * 0.05, f"W{i + 1}", ha="center", va="top", fontsize=8, color=INK)
    ax.set_ylim(base - span * 0.55, env.max() + span * 0.1)
    ax.set_xlim(0, t[-1])
    ax.set_title(f"{win_cfg['window_ms']} ms windows, {int(win_cfg['overlap'] * 100)}% overlap: "
                 f"{n_win} windows per trial, none crossing the trial boundary "
                 f"(gesture {int(row['gesture_label'])})")
    save(fig, out, "fig4_stage2_windowing.png")


# ---------------------------------------------------------------- Stage 3
def load_metrics(results_dir: Path, dataset: str) -> dict:
    out = {}
    for m in MODEL_ORDER:
        p = results_dir / m / dataset / "metrics.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def load_all_runs(results_dir: Path, dataset: str) -> dict[str, list[dict]]:
    """All repeat-split runs per model key (base run + <key>_seed<N> runs)."""
    runs = {}
    for m in MODEL_ORDER:
        found = []
        for p in sorted(results_dir.glob(f"{m}*/{dataset}/metrics.json")):
            name = p.parent.parent.name
            if name == m or (name.startswith(m + "_seed") and name[len(m) + 5:].isdigit()):
                found.append(json.loads(p.read_text()))
        if found:
            runs[m] = found
    return runs


def fig_multiseed(runs: dict, out: Path):
    models = [m for m in MODEL_ORDER if m in runs and len(runs[m]) >= 3]
    if len(models) < 2:
        return
    n_runs = min(len(runs[m]) for m in models)
    panels = [("Held-out trials, same session", lambda r, k: r[k]),
              ("Preview: untouched session 2", lambda r, k: r["cross_session"][k])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True)
    n = len(models)
    w = 0.8 / n
    for ax, (title, get) in zip(axes, panels):
        for i, m in enumerate(models):
            for j, key in enumerate(("per_window_accuracy", "per_trial_accuracy")):
                vals = np.array([get(r, key) for r in runs[m]]) * 100
                x = j + (i - (n - 1) / 2) * w
                ax.bar(x, vals.mean(), width=w * 0.88, color=MODEL_COLOR[m], label=MODEL_LABEL[m] if j == 0 else None)
                ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), color=INK, lw=1, capsize=3)
                ax.scatter(np.full(len(vals), x), vals, s=9, color=INK, zorder=3, linewidths=0)
                ax.text(x, min(vals.mean() + vals.std(ddof=1) + 2.5, 104), f"{vals.mean():.0f}", ha="center", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Per window", "Per trial (majority vote)"])
        ax.set_ylim(0, 110); ax.set_title(title); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy (%), mean ± SD")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=8)
    fig.suptitle(f"Repeated random trial splits (n = {n_runs} per model; dots = individual splits)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.0)
    save(fig, out, "fig9_stage3_repeated_splits.png")


def _bars(ax, models, key_fn, title):
    x = np.arange(2)
    n = len(models)
    w = 0.78 / n
    for i, m in enumerate(models):
        vals = [key_fn(m, "per_window_accuracy"), key_fn(m, "per_trial_accuracy")]
        pos = x + (i - (n - 1) / 2) * w
        ax.bar(pos, np.array(vals) * 100, width=w * 0.88, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        for px, v in zip(pos, vals):
            ax.text(px, v * 100 + 1.2, f"{v * 100:.0f}", ha="center", fontsize=8, color=INK)
    ax.axhline(100 / 34, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(-0.48, 100 / 34 + 1.5, "chance (2.9%)", fontsize=7.5, color=INK2, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(["Per window", "Per trial (majority vote)"])
    ax.set_ylim(0, 108); ax.set_ylabel("Accuracy (%)"); ax.set_title(title)
    ax.grid(axis="x", visible=False)


def fig_model_comparison(metrics: dict, out: Path):
    models = [m for m in MODEL_ORDER if m in metrics]
    if len(models) < 2:
        return
    has_cs = all("cross_session" in metrics[m] for m in models)
    fig, axes = plt.subplots(1, 2 if has_cs else 1, figsize=(11 if has_cs else 6, 4), sharey=True)
    axes = np.atleast_1d(axes)
    _bars(axes[0], models, lambda m, k: metrics[m][k], "Held-out trials, same session")
    if has_cs:
        _bars(axes[1], models, lambda m, k: metrics[m]["cross_session"][k], "Preview: untouched session 2")
        axes[1].set_ylabel("")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    save(fig, out, "fig5_stage3_model_comparison.png")


def fig_training_curves(results_dir: Path, dataset: str, out: Path):
    hist = {}
    for m in DEEP_ORDER:
        p = results_dir / m / dataset / "history.json"
        if p.exists():
            hist[m] = json.loads(p.read_text())
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m, h in hist.items():
        ep = np.arange(1, len(h["val_accuracy"]) + 1)
        for ax, key in zip(axes, ("val_accuracy", "val_loss")):
            ax.plot(ep, h[key], color=MODEL_COLOR[m], lw=1.8)
            ax.text(ep[-1] + 0.5, h[key][-1], MODEL_LABEL[m], color=INK, fontsize=8, va="center")
    axes[0].set_title("Validation accuracy"); axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.margins(x=0.12)
    axes[0].set_ylim(0, 1)
    save(fig, out, "fig6_stage3_training_curves.png")


def fig_confusion(results_dir: Path, metrics: dict, dataset: str, out: Path):
    deep = [m for m in DEEP_ORDER if m in metrics]
    if not deep:
        return
    best = max(deep, key=lambda m: metrics[m]["per_window_accuracy"])
    d = np.load(results_dir / best / dataset / "test_predictions.npz")
    cm = np.zeros((34, 34))
    for t, p in zip(d["y_true"], d["y_pred"]):
        cm[t, p] += 1
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(0, 34, 3)); ax.set_xticklabels(range(1, 35, 3), fontsize=7)
    ax.set_yticks(range(0, 34, 3)); ax.set_yticklabels(range(1, 35, 3), fontsize=7)
    ax.set_xlabel("Predicted gesture"); ax.set_ylabel("True gesture")
    ax.set_title(f"{MODEL_LABEL[best]}: per-window confusion, held-out trials")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fraction of true class")
    save(fig, out, "fig7_stage3_confusion_best_model.png")


def fig_accuracy_by_window_position(results_dir: Path, metrics: dict, dataset: str, out: Path):
    """Per-window accuracy by window position inside the 1 s trial (W1 = first 200 ms)."""
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    n_models = 0
    for m in MODEL_ORDER:
        p = results_dir / m / dataset / "test_predictions.npz"
        if m not in metrics or not p.exists():
            continue
        d = np.load(p)
        df = pd.DataFrame({"t": d["trial_id"], "ok": d["y_true"] == d["y_pred"]})
        df["pos"] = df.groupby("t").cumcount() + 1
        acc = df.groupby("pos")["ok"].mean() * 100
        ax.plot(acc.index, acc.values, color=MODEL_COLOR[m], lw=1.8, marker="o", ms=4, label=MODEL_LABEL[m])
        n_models += 1
    if not n_models:
        plt.close(fig)
        return
    ax.set_xlabel("Window position within the 1 s trial (W1 = 0 to 200 ms, W8 = 800 to 1000 ms)")
    ax.set_ylabel("Per-window accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy rises through the trial: early windows are mostly rest but carry the gesture label")
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    save(fig, out, "fig8_stage3_accuracy_by_window_position.png")


def export_runs(results_dir: Path, dataset: str, dest: Path):
    """Copy the small per-run JSON evidence (metrics + training history) next to the reports."""
    import shutil
    n = 0
    for p in sorted(results_dir.glob(f"*/{dataset}")):
        for fn in ("metrics.json", "history.json"):
            src = p / fn
            if src.exists():
                (dest / p.parent.name).mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest / p.parent.name / fn)
                n += 1
    print("exported", n, "result files to", dest)


def write_summary(metrics: dict, man: pd.DataFrame, subject: str, reports: Path, runs: dict | None = None):
    rows = []
    for m in MODEL_ORDER:
        if m not in metrics:
            continue
        x = metrics[m]
        cs = x.get("cross_session", {})
        rows.append({
            "model": MODEL_LABEL[m],
            "per_window_acc_%": round(x["per_window_accuracy"] * 100, 1),
            "per_trial_acc_%": round(x["per_trial_accuracy"] * 100, 1),
            "xsession_window_%": round(cs["per_window_accuracy"] * 100, 1) if cs else None,
            "xsession_trial_%": round(cs["per_trial_accuracy"] * 100, 1) if cs else None,
            "parameters": x.get("parameters"),
            "MFLOPs/inference": round(x["flops_per_inference"] / 1e6, 1) if x.get("flops_per_inference") else None,
            "latency_ms_desktop": round(x["latency_ms_desktop_cpu"], 2) if x.get("latency_ms_desktop_cpu") else None,
            "epochs": x.get("epochs_run"),
            "train_s": x.get("train_seconds"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(reports / f"subject{subject}_stage0_3_summary.csv", index=False)
    first = next(x for x in metrics.values() if "train" in x["n_trials"])
    lines = [
        f"# Subject {subject}: Stage 0 to 3 proof run",
        "",
        f"- Stage 0: {len(man)} dynamic trials (session counts: "
        + ", ".join(f"S{s}={n}" for s, n in man.groupby('session').size().items()) + "), all SHA256-verified.",
        f"- Stage 2: {int(man['n_windows'].sum())} windows of {int(man['window_len_samples'].iloc[0])} samples "
        f"(256 ch), step {int(man['step_samples'].iloc[0])}.",
        f"- Stage 3 split (session 1 only): {first['n_trials']} trials, trial-level, gesture-stratified, seed 42; "
        f"{first['n_test_windows']} test windows.",
        "",
        "## Primary split (seed 42)",
        "",
        df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False),
        "",
    ]
    if runs:
        agg = []
        for m in MODEL_ORDER:
            rs = runs.get(m, [])
            if len(rs) < 2:
                continue
            def ms(f):
                v = np.array([f(r) for r in rs]) * 100
                return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"
            agg.append({
                "model": MODEL_LABEL[m], "splits": len(rs),
                "per_window_%": ms(lambda r: r["per_window_accuracy"]),
                "per_trial_%": ms(lambda r: r["per_trial_accuracy"]),
                "xsession_window_%": ms(lambda r: r["cross_session"]["per_window_accuracy"]),
                "xsession_trial_%": ms(lambda r: r["cross_session"]["per_trial_accuracy"]),
            })
        if agg:
            adf = pd.DataFrame(agg)
            adf.to_csv(reports / f"subject{subject}_stage0_3_repeated_splits.csv", index=False)
            lines += ["## Repeated random trial splits (mean ± SD)", "", adf.to_markdown(index=False), ""]
    (reports / f"subject{subject}_stage0_3_summary.md").write_text("\n".join(lines))
    print("wrote summary for", len(rows), "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="01")
    ap.add_argument("--dataset", default="hyser")
    ap.add_argument("--results", type=Path, default=Path("results/checkpoints"))
    ap.add_argument("--manifest", type=Path, default=Path("manifests/hyser_windowed_manifest.csv"))
    args = ap.parse_args()

    out = Path("reports/figures") / f"subject{args.subject}"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest)
    man = man[man["subject_id"] == f"subject{args.subject}"].reset_index(drop=True)
    pre_cfg = load_yaml("configs/preprocessing.yaml")
    win_cfg = load_yaml("configs/windowing.yaml")

    fig_dataset_overview(man, out)
    first_raw = man[(man["session"] == man["session"].min())].iloc[0]["file_path"]
    fig_filtering(pre_cfg, Path(first_raw), out)
    fig_spatial_rms(man, Path("data/raw/hyser"), out)
    fig_windowing(man, win_cfg, out)

    metrics = load_metrics(args.results, args.dataset)
    if metrics:
        fig_model_comparison(metrics, out)
        fig_training_curves(args.results, args.dataset, out)
        fig_confusion(args.results, metrics, args.dataset, out)
        fig_accuracy_by_window_position(args.results, metrics, args.dataset, out)
        runs = load_all_runs(args.results, args.dataset)
        fig_multiseed(runs, out)
        write_summary(metrics, man, args.subject, Path("reports"), runs)
        export_runs(args.results, args.dataset, Path("reports/results") / f"subject{args.subject}")


if __name__ == "__main__":
    main()
