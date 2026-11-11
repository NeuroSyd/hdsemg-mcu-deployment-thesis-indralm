"""
Run Stage 0 to 3 for many Hyser subjects, one subject at a time, keeping disk use small.

For each subject it: (0) fetches only dynamic RAW signals + dynamic labels (both sessions)
from PhysioNet's S3 mirror and verifies them with extract_hyser.py, (1-2) filters and windows
them, (3) trains LDA, 1D CNN, TCN and GRU (input_pool=8), then deletes the raw files and all
intermediate arrays. Only small result files (metrics.json etc.) are kept.

Same protocol as the subject 01 proof run: train on session 1 (trial-level, gesture-stratified
70/15/15 split, seed 42), cross-session preview on session 2.

Usage (from the repo root):
    python scripts/run_hyser_subjects.py --subjects 02-20 --raw-root ~/hyser_data
    python scripts/run_hyser_subjects.py --subjects 02,03 --skip-download       # raw files already there
    python scripts/run_hyser_subjects.py --subjects 02 --dry-run                # print commands only
Gesture grouping (Solution A): add --class-map A1,A2-s1:22 --results-root results/hyser_grouped to retrain on
grouped labels. Raw data is downloaded and windowed once; each spec trains into <results-root>/<spec>/<subject>.
Deployment-like normalisation test: add --norm-reference-session 1 --results-root results/hyser_refnorm (session 2 is
z-scored with session 1 statistics; session 1 and the seed are unchanged, so only cross-session results should move).
Re-running skips subjects whose results already exist (use --force to redo).
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.training.class_maps import safe_name, resolve_class_map  # noqa: E402
S3_BASE = "s3://physionet-open/hd-semg/2.0.0/pr_dataset/"
RUNS = [  # (run_name, extra train.py args)
    ("cnn1d", ["--architecture", "cnn1d"]),
    ("tcn", ["--architecture", "tcn"]),
    ("gru_pool8", ["--architecture", "gru", "--variant", "pool8", "--model-opt", "input_pool=8"]),
]


def parse_subjects(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += [f"{i:02d}" for i in range(int(a), int(b) + 1)]
        elif part:
            out.append(f"{int(part):02d}")
    return out


def run(cmd, dry, retries=1, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return
    for attempt in range(1, retries + 1):
        try:
            subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, **kw)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            print(f"  ...command failed (attempt {attempt}/{retries}), retrying in 10 s", flush=True)
            time.sleep(10)


def write_yaml(path: Path, data: dict, dry: bool):
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(name):
    with open(REPO / "configs" / name) as f:
        return yaml.safe_load(f)


def process_subject(sub, a):
    t0 = time.time()
    print(f"\n===== subject{sub} =====", flush=True)
    specs = [x.strip() for x in a.class_map.split(",") if x.strip()] if a.class_map else [None]

    def rdir(spec):
        return REPO / a.results_root / (safe_name(spec) if spec else "") / sub

    if not a.force:
        specs = [sp for sp in specs if not (rdir(sp) / "gru_pool8" / "hyser" / "metrics.json").exists()]
        if not specs:
            print("  results already exist, skipping (use --force to redo)")
            return
    results_dir = rdir(specs[0])
    raw_root = a.raw_root.expanduser().resolve()
    work = REPO / a.work_root / sub
    cfgdir = work / "configs"
    man = REPO / "manifests" / "per_subject"

    # Stage 0: download dynamic raw only + dynamic labels, verify hashes, write manifest
    if not a.skip_download:
        for sess in (1, 2):
            folder = f"subject{sub}_session{sess}"
            run(["aws", "s3", "sync", "--no-sign-request", "--only-show-errors",
                 f"{S3_BASE}{folder}/", raw_root / folder,
                 "--exclude", "*", "--include", "dynamic_raw_sample*", "--include", "label_dynamic.txt"], a.dry_run, retries=4)
    run([sys.executable, "src/extraction/extract_hyser.py", "--out", raw_root, "--subjects", sub,
         "--sig-types", "raw", "--task-types", "dynamic", "--workers", str(a.workers),
         "--manifest-out", man / f"hyser_manifest_{sub}.csv"], a.dry_run)

    # Per-subject configs (same values as configs/*.yaml, only paths and scope change)
    pre = load_yaml("preprocessing.yaml")
    pre.update(subjects=[sub], sessions=[1, 2], task_types=["dynamic"], signal_source="raw",
               input_manifest=str(man / f"hyser_manifest_{sub}.csv"),
               output_dir=str(work / "preprocessed"),
               output_manifest=str(man / f"hyser_preprocessed_manifest_{sub}.csv"))
    if a.norm_reference_session:
        # deployment-like normalisation: every session uses the reference (training) session's statistics
        pre["normalization"] = {**pre["normalization"], "scope": "reference_session",
                                "reference_session": a.norm_reference_session}
    win = load_yaml("windowing.yaml")
    win.update(input_manifest=pre["output_manifest"], output_dir=str(work / "windows"),
               output_manifest=str(man / f"hyser_windowed_manifest_{sub}.csv"))
    trn = load_yaml("training.yaml")
    trn.update(input_manifest=win["output_manifest"], output_dir=str(results_dir))
    trn["data"] = {**trn.get("data", {}), "subjects": [sub]}
    for name, cfg in (("preprocessing", pre), ("windowing", win)):
        write_yaml(cfgdir / f"{name}.yaml", cfg, a.dry_run)
    for sp in specs:
        write_yaml(cfgdir / f"training{'_' + safe_name(sp) if sp else ''}.yaml",
                   {**trn, "output_dir": str(rdir(sp))}, a.dry_run)

    # Stages 1, 2a, 3
    run([sys.executable, "src/preprocessing/pipeline.py", "--config", cfgdir / "preprocessing.yaml"], a.dry_run)
    run([sys.executable, "src/windowing/pipeline.py", "--config", cfgdir / "windowing.yaml"], a.dry_run)
    for sp in specs:
        tcfg = cfgdir / f"training{'_' + safe_name(sp) if sp else ''}.yaml"
        cm = ["--class-map", sp] if sp else []
        run([sys.executable, "src/training/train_lda.py", "--config", tcfg, *cm], a.dry_run)
        for _, extra in RUNS:
            run([sys.executable, "src/training/train.py", "--config", tcfg, *extra, *cm], a.dry_run)

    # Keep results; remove bulky files
    if not a.dry_run:
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        if not a.keep_raw:
            for sess in (1, 2):
                shutil.rmtree(raw_root / f"subject{sub}_session{sess}", ignore_errors=True)
        for sp in specs:
            for p in rdir(sp).rglob("best.keras"):  # model weights are not needed for the cross-subject summary
                if not a.keep_checkpoints:
                    p.unlink()
    print(f"  subject{sub} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="02-20")
    ap.add_argument("--raw-root", type=Path, default=Path("~/hyser_data/hyser"),
                    help="where raw files go; keep OUTSIDE OneDrive")
    ap.add_argument("--work-root", default="data/processed/hyser_all",
                    help="temporary per-subject intermediates (deleted after each subject)")
    ap.add_argument("--results-root", default="results/hyser_all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--class-map", default=None,
                    help="comma-separated gesture groupings, e.g. A1,A2-s1:22 (see src/training/class_maps.py)")
    ap.add_argument("--norm-reference-session", type=int, default=None,
                    help="Normalise all sessions with this session's statistics (e.g. 1) instead of each session's own. "
                         "Use with a separate --results-root, e.g. results/hyser_refnorm")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.skip_download is False and shutil.which("aws") is None and not a.dry_run:
        sys.exit("aws CLI not found. Install with: brew install awscli (or use --skip-download)")
    failed = []
    for sub in parse_subjects(a.subjects):
        try:
            process_subject(sub, a)
        except subprocess.CalledProcessError as e:
            print(f"  !! subject{sub} failed: {e}. Continuing with the next subject.", flush=True)
            failed.append(sub)
    print("\nFinished. Failed subjects:", failed or "none")


if __name__ == "__main__":
    main()
