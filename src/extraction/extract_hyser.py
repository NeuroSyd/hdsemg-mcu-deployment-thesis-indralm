"""
Stage 0 - Extraction: Hyser HD-sEMG dataset (PhysioNet, v2.0.0, PR sub-dataset)

Pulls the Pattern Recognition (PR) sub-dataset of Hyser directly from
PhysioNet (20 subjects, 2 sessions each, pr_dataset/ folder ~37 GB),
verifies file integrity against PhysioNet's published SHA256SUMS.txt, and
writes a manifest row per file recording subject ID, session, task type,
signal type, and sample index.

No format conversion happens here: files are kept in their native WFDB
(.dat/.hea) and label .txt form on disk, so this extraction step stays
auditable against the original PhysioNet release. Filtering and
normalisation happen in Stage 1 (src/preprocessing/), not here.

PhysioNet's hd-semg/2.0.0 release has five top-level sub-datasets
(1dof_dataset, mvc_dataset, ndof_dataset, pr_dataset, random_dataset,
135 GB total). This script only pulls pr_dataset (~37 GB), the one this
thesis uses, by filtering PhysioNet's dataset-wide SHA256SUMS.txt down to
entries under "pr_dataset/".

Within pr_dataset, files are organised as:
    pr_dataset/subject_<II>_session_<J>/<taskType>_<sigType>_sample<k>.<ext>
    pr_dataset/subject_<II>_session_<J>/label_<taskType>.txt
where taskType is "dynamic" or "maintenance", sigType is "raw" or
"preprocess" (Hyser ships a pre-filtered version already), and ext is
.dat (signal) or .hea (WFDB header).

Usage:
    python src/extraction/extract_hyser.py --out data/raw/hyser
    python src/extraction/extract_hyser.py --out data/raw/hyser --sig-types preprocess
    python src/extraction/extract_hyser.py --out data/raw/hyser --subjects 01,02,03

Source:
    N. Jiang, C. Dai, X. Liu, and C. Fan, "Open Access Dataset and Toolbox
    of High-Density Surface Electromyogram Recordings," v2.0.0, PhysioNet,
    2021/2023. https://physionet.org/content/hd-semg/2.0.0/
"""

import argparse
import hashlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

import pandas as pd

PHYSIONET_BASE_URL = "https://physionet.org/files/hd-semg/2.0.0/"
SHA256SUMS_URL = PHYSIONET_BASE_URL + "SHA256SUMS.txt"
PR_PREFIX = "pr_dataset/"

# pr_dataset/subject_<II>_session_<J>/<filename>
FOLDER_PATTERN = re.compile(
    r"^pr_dataset/subject(?P<subject>\d+)_session(?P<session>\d+)/(?P<filename>.+)$"
)
# <taskType>_<sigType>_sample<k>.<ext>   e.g. dynamic_raw_sample3.dat
DATA_FILENAME_PATTERN = re.compile(
    r"^(?P<task>dynamic|maintenance)_(?P<sig>raw|preprocess)_sample(?P<sample>\d+)\.(?P<ext>dat|hea)$",
    re.IGNORECASE,
)
# label_<taskType>.txt   e.g. label_dynamic.txt
LABEL_FILENAME_PATTERN = re.compile(
    r"^label_(?P<task>dynamic|maintenance)\.txt$", re.IGNORECASE
)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA256 hash of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    """Download a single file atomically (.part then rename), skipping if it exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_err = None
    for _ in range(retries):
        try:
            urlretrieve(url, tmp)
            tmp.replace(dest)
            return
        except URLError as e:
            last_err = e
    raise RuntimeError(f"Failed to download {url}: {last_err}") from last_err


def load_pr_dataset_hashes(work_dir: Path) -> dict[str, str]:
    """
    Download PhysioNet's dataset-wide SHA256SUMS.txt and return only the
    entries under pr_dataset/, as {relative_path: expected_hash}.
    """
    sums_path = work_dir / "SHA256SUMS.txt"
    print("Fetching SHA256SUMS.txt (dataset-wide, ~7 MB) ...")
    download_file(SHA256SUMS_URL, sums_path)

    hashes = {}
    with open(sums_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts
            rel_path = rel_path.strip().lstrip("*")  # sha256sum binary-mode lines prefix path with "*"
            if rel_path.startswith(PR_PREFIX):
                hashes[rel_path] = expected_hash
    return hashes


def parse_pr_path(rel_path: str) -> dict | None:
    """Parse a pr_dataset/ relative path into subject/session/task/sig/sample fields."""
    folder_match = FOLDER_PATTERN.match(rel_path)
    if not folder_match:
        return None

    subject_id = f"subject{int(folder_match.group('subject')):02d}"
    session = folder_match.group("session")
    filename = folder_match.group("filename")

    data_match = DATA_FILENAME_PATTERN.match(filename)
    if data_match:
        return {
            "subject_id": subject_id,
            "session": session,
            "task_type": data_match.group("task").lower(),
            "sig_type": data_match.group("sig").lower(),
            "sample_index": data_match.group("sample"),
            "file_kind": "signal" if data_match.group("ext").lower() == "dat" else "header",
            "filename": filename,
        }

    label_match = LABEL_FILENAME_PATTERN.match(filename)
    if label_match:
        return {
            "subject_id": subject_id,
            "session": session,
            "task_type": label_match.group("task").lower(),
            "sig_type": None,
            "sample_index": None,
            "file_kind": "label",
            "filename": filename,
        }

    return None  # unrecognised file (e.g. readme) - skip rather than fail


def extract_hyser(
    out_dir: Path,
    sig_types: set[str] | None = None,
    subjects: set[str] | None = None,
    verify_hashes: bool = True,
    workers: int = 12,
) -> pd.DataFrame:
    """
    Download Hyser's pr_dataset from PhysioNet into `out_dir` (preserving
    the subject_XX_session_Y/ folder layout), verify each file's hash, and
    return a manifest DataFrame with one row per downloaded file.

    sig_types: e.g. {"preprocess"} to skip raw signals and save bandwidth.
               None downloads both raw and preprocess.
    subjects:  e.g. {"01", "02"} to restrict to specific subjects for a
               quick test run. None downloads all 20 subjects.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_hashes = load_pr_dataset_hashes(out_dir)
    if not all_hashes:
        print(
            "Warning: no pr_dataset/ entries found in SHA256SUMS.txt. "
            "Check PHYSIONET_BASE_URL / PR_PREFIX are still correct on PhysioNet's end.",
            file=sys.stderr,
        )

    # 1) Filter first, so only the requested files are touched.
    jobs = []
    for rel_path, expected_hash in all_hashes.items():
        parsed = parse_pr_path(rel_path)
        if parsed is None:
            continue
        if subjects and parsed["subject_id"].replace("subject", "") not in subjects:
            continue
        if sig_types and parsed["sig_type"] is not None and parsed["sig_type"] not in sig_types:
            continue
        jobs.append((rel_path, expected_hash, parsed))

    total = len(jobs)
    print(f"{total} files to fetch with {workers} parallel workers.")
    counter = {"n": 0}
    lock = threading.Lock()

    # 2) Download + verify in parallel (I/O bound, so threads work well).
    def fetch(job):
        rel_path, expected_hash, parsed = job
        dest_path = out_dir / rel_path[len(PR_PREFIX):]
        download_file(PHYSIONET_BASE_URL + rel_path, dest_path)
        if verify_hashes and sha256_of_file(dest_path) != expected_hash:
            dest_path.unlink()  # corrupt or stale file: redownload once
            download_file(PHYSIONET_BASE_URL + rel_path, dest_path)
            actual_hash = sha256_of_file(dest_path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Hash mismatch for {rel_path}: "
                    f"expected {expected_hash}, got {actual_hash}."
                )
        with lock:
            counter["n"] += 1
            print(f"[{counter['n']}/{total}] OK {rel_path}")
        return {
            "dataset": "hyser",
            "subject_id": parsed["subject_id"],
            "session": parsed["session"],
            "day": None,  # Hyser uses sessions, not days (that's CEMHSEY)
            "task_type": parsed["task_type"],
            "sig_type": parsed["sig_type"],
            "sample_index": parsed["sample_index"],
            "file_kind": parsed["file_kind"],
            "filename": parsed["filename"],
            "file_path": str(dest_path),
            "sha256": expected_hash,
        }

    manifest_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch, j) for j in jobs]
        for fut in as_completed(futures):
            manifest_rows.append(fut.result())  # re-raises any worker error
    manifest_rows.sort(key=lambda r: r["file_path"])

    return pd.DataFrame(manifest_rows)


def main():
    parser = argparse.ArgumentParser(description="Extract Hyser HD-sEMG pr_dataset from PhysioNet.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/hyser"),
        help="Directory to download raw Hyser files into (default: data/raw/hyser)",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("manifests/hyser_manifest.csv"),
        help="Where to write the Hyser manifest CSV (default: manifests/hyser_manifest.csv)",
    )
    parser.add_argument(
        "--sig-types",
        type=str,
        default=None,
        help="Comma-separated signal types to download: raw,preprocess (default: both)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subject numbers to restrict to, e.g. 01,02 (default: all 20)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA256 hash verification against PhysioNet's published sums (not recommended).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Parallel download threads (default: 12; keep <=16 to be polite to PhysioNet)",
    )
    args = parser.parse_args()

    sig_types = set(s.strip().lower() for s in args.sig_types.split(",")) if args.sig_types else None
    subjects = set(s.strip() for s in args.subjects.split(",")) if args.subjects else None

    manifest = extract_hyser(
        args.out,
        sig_types=sig_types,
        subjects=subjects,
        verify_hashes=not args.no_verify,
        workers=args.workers,
    )

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_out, index=False)

    print(f"\nExtracted {len(manifest)} files.")
    print(f"Manifest written to {args.manifest_out}")
    if not manifest.empty:
        print(f"Subjects found: {sorted(manifest['subject_id'].unique())}")
        print(f"Sessions found: {sorted(manifest['session'].unique())}")
        print(f"Task types found: {sorted(manifest['task_type'].dropna().unique())}")


if __name__ == "__main__":
    main()