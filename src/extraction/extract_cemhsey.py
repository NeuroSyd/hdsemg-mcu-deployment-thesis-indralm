"""
Stage 0 - Extraction: CEMHSEY GESTURE sub-dataset (Zenodo, CC BY 4.0).

Source: S. Yang, C. Chen, D. Li, X. Zhu, "A Consecutive Multi-Day High-Density
Surface Electromyography Dataset Comprising 7 Grasps and 11 Gestures",
Scientific Data 2025, doi:10.1038/s41597-025-05733-y. Part II (GESTURE_S1..S6),
Zenodo record 15070187.

Dataset facts used here (all from the paper):
  - 320 channels (5 grids x 64), 2048 Hz, one variable `data_sEMG` per .mat file
  - 11 days x 5 trials per subject, each trial about 180 s
  - Per trial: 5 s rest, then 11 gestures of 10 s with 5 s rest between them
    (fixed order, see GESTURES). The .mat files carry NO label channel, so
    labels come from this fixed timing (`gesture_segments`).

This script copies the requested days out of the downloaded GESTURE_S<n>.zip
into a working folder and writes a manifest row per trial. The zip itself is
verified separately against Zenodo's md5 (md5 -r GESTURE_S4.zip).

Usage:
    python src/extraction/extract_cemhsey.py --zip ~/cemhsey/GESTURE_S4.zip --subject 4 \
        --days 1-2 --out ~/cemhsey/work/raw --manifest-out manifests/cemhsey/S4_manifest.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import zipfile_deflate64 as zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FS = 2048
N_CHANNELS = 320
N_DAYS = 11
N_TRIALS = 5
GESTURES = ["HC", "HO", "WF", "WE", "RD", "UD", "FP", "FS", "2F-PpP", "3F-PpP", "LatP"]
LEAD_REST_S = 5.0
HOLD_S = 10.0
GAP_REST_S = 5.0


def gesture_segments(n_samples: int, fs: float = FS) -> list[tuple[int, int, int]]:
    """
    (label 1..11, start_sample, end_sample) for the 11 gesture holds of one trial.
    Gesture k holds from 5 + 15*(k-1) s for 10 s, so the last one ends at 165 s.
    """
    segments = []
    for k in range(len(GESTURES)):
        start = int(round((LEAD_REST_S + k * (HOLD_S + GAP_REST_S)) * fs))
        end = start + int(round(HOLD_S * fs))
        if end > n_samples:
            raise ValueError(f"Trial too short ({n_samples} samples) for gesture {k + 1}")
        segments.append((k + 1, start, end))
    return segments


def parse_days(spec: str) -> list[int]:
    days = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            days += list(range(int(a), int(b) + 1))
        elif part:
            days.append(int(part))
    return days


def extract_day(zip_path: Path, subject: int, day: int, out_dir: Path) -> list[dict]:
    """
    Copy one day's trial files out of the zip; return one manifest row per trial found.

    The Zenodo zips are zip64 and use Deflate64. Python's zipfile and macOS's unzip (also Homebrew's)
    cannot read them, so this uses the `zipfile-deflate64` package (pip install zipfile-deflate64).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for trial in range(1, N_TRIALS + 1):
            member = f"S{subject}/D{day}/S{subject}_Day{day}_Trial{trial}.mat"
            if member not in names:
                print(f"  ! missing in zip: {member}", flush=True)
                continue
            target = Path(z.extract(member, out_dir))
            rows.append(
                {
                    "dataset": "cemhsey",
                    "subject_id": f"S{subject}",
                    "session": day,  # one day = one session, so the Stage 3 cross-session code is cross-day
                    "task_type": "gesture",
                    "trial": trial,
                    "sample_index": trial,
                    "filename": Path(member).name,
                    "file_path": str(target),
                    "file_kind": "signal",
                    "sig_type": "raw",
                    "size_bytes": target.stat().st_size,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description="Stage 0: extract CEMHSEY GESTURE trials from the Zenodo zip.")
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--days", default="1-11")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for day in parse_days(a.days):
        rows += extract_day(a.zip.expanduser(), a.subject, day, a.out.expanduser())
    a.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.manifest_out, index=False)
    print(f"Extracted {len(rows)} trials. Manifest: {a.manifest_out}")


if __name__ == "__main__":
    main()
