"""
Run Stage 0 to 3 on one CEMHSEY GESTURE subject, one day at a time, to keep disk use small.

Per day: (0) copy that day's 5 trials out of GESTURE_S<n>.zip, (1) filter + normalise with Day 1
statistics, (2a) cut into the 11 gesture holds and window, then delete the raw .mat files.
Then (3) train LDA, 1D CNN, TCN and GRU (input_pool=8) on Day 1 trials 1-3 (val trial 4; LDA uses
trials 1-4), test on Day 1 trial 5, and score every other processed day as cross-day.

Windows are kept in a work folder OUTSIDE the repo (default ~/cemhsey/work), because they are
several GB and must not sync to OneDrive. Safe to re-run: finished days are skipped.

Usage (from the repo root, inside your venv):
    python scripts/run_cemhsey.py --zip ~/cemhsey/GESTURE_S4.zip --days 1-3 --skip-training   # pilot
    caffeinate -i python scripts/run_cemhsey.py --zip ~/cemhsey/GESTURE_S4.zip --days 1-11     # full run
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.extraction.extract_cemhsey import extract_day, parse_days  # noqa: E402
from src.preprocessing.cemhsey_pipeline import process_day  # noqa: E402

RUNS = [
    ("cnn1d", ["--architecture", "cnn1d"]),
    ("tcn", ["--architecture", "tcn"]),
    ("gru_pool8", ["--architecture", "gru", "--variant", "pool8", "--model-opt", "input_pool=8"]),
]


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--subject", type=int, default=4)
    ap.add_argument("--days", default="1-11")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "cemhsey.yaml")
    ap.add_argument("--work-root", type=Path, default=Path("~/cemhsey/work"))
    ap.add_argument("--results-root", default="results/cemhsey")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--delete-windows", action="store_true", help="delete the windowed data after training")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    sid = f"S{a.subject}"
    work = a.work_root.expanduser() / sid
    raw = work / "raw"
    man_dir = REPO / "manifests" / "cemhsey"
    man_dir.mkdir(parents=True, exist_ok=True)
    zip_path = a.zip.expanduser()
    days = parse_days(a.days)
    if 1 not in days and not (work / "norm_stats.npz").exists():
        sys.exit("Day 1 must be processed first (it provides the normalisation statistics).")

    free_gb = shutil.disk_usage(work.parent if work.parent.exists() else Path.home()).free / 1e9
    print(f"Free disk: {free_gb:.0f} GB (full run needs about 12 GB for windows plus 1.3 GB per day temporarily)")

    t0 = time.time()
    for day in sorted(days, key=lambda d: (d != cfg["train_day"], d)):  # training day first
        part = work / "manifests" / f"day{day:02d}.csv"
        if part.exists():
            print(f"Day {day}: already processed, skipping")
            continue
        print(f"\n===== {sid} Day {day} =====", flush=True)
        rows = extract_day(zip_path, a.subject, day, raw)
        if not rows:
            print(f"  no trials found for day {day}")
            continue
        out = process_day(rows, day, cfg, work)
        part.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(part, index=False)
        shutil.rmtree(raw / sid / f"D{day}", ignore_errors=True)
        print(f"  Day {day} done ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    parts = sorted((work / "manifests").glob("day*.csv"))
    manifest = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    manifest_path = man_dir / f"{sid}_windowed_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nWindowed manifest: {manifest_path} ({len(manifest)} gesture segments, "
          f"{manifest['session'].nunique()} days)")
    if a.skip_training:
        print("Skipping Stage 3 (--skip-training).")
        return

    results_dir = REPO / a.results_root / sid
    cfg_dir = work / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(open(REPO / "configs" / "training.yaml"))
    base.update(dataset="cemhsey", input_manifest=str(manifest_path),
                model_config_path="configs/models_cemhsey.yaml",
                output_dir=str(results_dir), cross_session_eval=True)
    base["data"] = {"task_type": "gesture", "train_sessions": [cfg["train_day"]]}
    base["split"] = dict(cfg["split"])
    train_cfg = cfg_dir / "training.yaml"
    yaml.safe_dump(base, open(train_cfg, "w"), sort_keys=False)

    run([sys.executable, "src/training/train_lda.py", "--config", train_cfg])
    for _, extra in RUNS:
        run([sys.executable, "src/training/train.py", "--config", train_cfg, *extra])

    if not a.keep_checkpoints:
        for p in results_dir.rglob("best.keras"):
            p.unlink()
    if a.delete_windows:
        shutil.rmtree(work / "windows", ignore_errors=True)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Results: {results_dir}")


if __name__ == "__main__":
    main()
