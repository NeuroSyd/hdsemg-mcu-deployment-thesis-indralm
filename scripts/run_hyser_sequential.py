"""One command for the trained sequential classifier on Hyser, subject by subject.

For each subject:
    0. download the dynamic raw files of both sessions (PhysioNet S3, no sign-in) and verify them
    1. filter and z-score with SESSION 1 statistics for both sessions (the calibration-free setting)
    2. window (200 ms, 50 % overlap)
    3. train the flat 1D CNN, TCN and GRU only if no checkpoint exists yet (the pilots S02, S06,
       S19 and S20); this is the slow, fan-loud step and is identical to the Week 13 runs
    4. train the group and specialist heads and the labelled baseline (scripts/run_sequential_trained.py)
    5. delete raw files and windows (unless --keep-raw / --keep-work); checkpoints are kept

Results go to results/hyser_seq/<subj>/. Afterwards run
    python scripts/evaluate_sequential_trained.py

Usage (repo root, hyser_env active, raw data OUTSIDE OneDrive):
    python scripts/run_hyser_sequential.py --subjects 01 --dry-run         # print the commands only
    python scripts/run_hyser_sequential.py --subjects 01                   # one subject first, check the log
    python scripts/run_hyser_sequential.py --subjects 01-20                # then all (skips finished subjects)
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("rhs", REPO / "scripts" / "run_hyser_subjects.py")
rhs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rhs)


def data_stages(sub, a, cfgdir, man, work, raw_root):
    if not a.skip_download:
        for sess in (1, 2):
            folder = f"subject{sub}_session{sess}"
            rhs.run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors",
                     f"{rhs.S3_BASE}{folder}/", raw_root / folder,
                     "--exclude", "*", "--include", "dynamic_raw_sample*", "--include", "label_dynamic.txt"],
                    a.dry_run, retries=4)
    rhs.run([sys.executable, "src/extraction/extract_hyser.py", "--out", raw_root, "--subjects", sub,
             "--sig-types", "raw", "--task-types", "dynamic", "--workers", str(a.workers),
             "--manifest-out", man / f"hyser_manifest_{sub}.csv"], a.dry_run)
    pre = rhs.load_yaml("preprocessing.yaml")
    pre.update(subjects=[sub], sessions=[1, 2], task_types=["dynamic"], signal_source="raw",
               input_manifest=str(man / f"hyser_manifest_{sub}.csv"),
               output_dir=str(work / "preprocessed"),
               output_manifest=str(man / f"hyser_preprocessed_manifest_{sub}.csv"))
    pre["normalization"] = {**pre["normalization"], "scope": "reference_session", "reference_session": 1}
    win = rhs.load_yaml("windowing.yaml")
    win.update(input_manifest=pre["output_manifest"], output_dir=str(work / "windows"),
               output_manifest=str(man / f"hyser_windowed_manifest_{sub}.csv"))
    rhs.write_yaml(cfgdir / "preprocessing.yaml", pre, a.dry_run)
    rhs.write_yaml(cfgdir / "windowing.yaml", win, a.dry_run)
    rhs.run([sys.executable, "src/preprocessing/pipeline.py", "--config", cfgdir / "preprocessing.yaml"], a.dry_run)
    rhs.run([sys.executable, "src/windowing/pipeline.py", "--config", cfgdir / "windowing.yaml"], a.dry_run)
    return win


def process(sub, a):
    t0 = time.time()
    print(f"\n===== subject{sub} =====", flush=True)
    done = REPO / a.out / sub / "seq_predictions.npz"
    if done.exists() and not a.force:
        print("  sequential results exist, skipping (use --force to redo)")
        return
    raw_root = a.raw_root.expanduser().resolve()
    work = REPO / a.work_root / sub
    cfgdir = work / "configs"
    man = REPO / "manifests" / "per_subject"
    win = data_stages(sub, a, cfgdir, man, work, raw_root)

    # flat networks: train only where the checkpoint is missing
    results_dir = REPO / a.results_root / sub
    missing = [(rn, extra) for rn, extra in rhs.RUNS
               if not (results_dir / rn / "hyser" / "best.keras").exists()]
    if missing:
        trn = rhs.load_yaml("training.yaml")
        trn.update(input_manifest=win["output_manifest"], output_dir=str(results_dir))
        trn["data"] = {**trn.get("data", {}), "subjects": [sub]}
        rhs.write_yaml(cfgdir / "training.yaml", trn, a.dry_run)
        print(f"  training flat networks without a checkpoint: {[m for m, _ in missing]} (slow step)", flush=True)
        for _, extra in missing:
            rhs.run([sys.executable, "src/training/train.py", "--config", cfgdir / "training.yaml", *extra], a.dry_run)

    rhs.run([sys.executable, "scripts/run_sequential_trained.py", "--subjects", sub,
             "--results-root", a.results_root, "--groupings", a.groupings,
             "--n-cal", str(a.n_cal), "--shots", str(a.shots), "--out", a.out], a.dry_run)

    if not a.dry_run:
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        if not a.keep_raw:
            for sess in (1, 2):
                shutil.rmtree(raw_root / f"subject{sub}_session{sess}", ignore_errors=True)
    print(f"  subject{sub} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="01-20")
    ap.add_argument("--raw-root", type=Path, default=Path("~/hyser_data/hyser"),
                    help="where raw files go; keep OUTSIDE OneDrive")
    ap.add_argument("--work-root", default="data/processed/hyser_seq")
    ap.add_argument("--results-root", default="results/hyser_refnorm")
    ap.add_argument("--out", default="results/hyser_seq")
    ap.add_argument("--groupings", default="D1-5,H5")
    ap.add_argument("--n-cal", type=int, default=10)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.skip_download and shutil.which("aws") is None and not a.dry_run:
        sys.exit("aws CLI not found. Install with: brew install awscli (or use --skip-download)")
    failed = []
    for sub in rhs.parse_subjects(a.subjects):
        try:
            process(sub, a)
        except Exception as e:
            print(f"  !! subject{sub} failed: {e}. Continuing.", flush=True)
            failed.append(sub)
    print("\nFinished. Failed subjects:", failed or "none")


if __name__ == "__main__":
    main()
