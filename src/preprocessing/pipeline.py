"""
Stage 1 orchestration: raw signal -> filtered, normalised signal.

Reads the extraction manifest (Stage 0 output), applies the filter chain and
per-recording z-score normalisation to every signal file, writes each result
as a .npy array under the configured output directory, and produces a new
manifest that Stage 2 (windowing) reads as its input.

Usage:
    python src/preprocessing/pipeline.py --config configs/preprocessing.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import filter_manifest, load_manifest, write_manifest
from src.preprocessing.filters import apply_filter_chain
from src.preprocessing.io_hyser import read_hyser_record
from src.preprocessing.normalization import RunningChannelStats, zscore_normalize, zscore_with_stats


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def filter_recording(dat_path: str, cfg: dict) -> np.ndarray:
    """Load and filter a single Hyser recording (no normalisation yet)."""
    signal, record_fs, _ = read_hyser_record(dat_path)
    return apply_filter_chain(signal, record_fs, cfg).astype(np.float32)


def preprocess_recording(dat_path: str, fs: float, cfg: dict) -> np.ndarray:
    """Load, filter, and z-score one recording with its own statistics (per_recording scope)."""
    return zscore_normalize(filter_recording(dat_path, cfg)).astype(np.float32)


def run_preprocessing(config_path: str | Path) -> None:
    cfg = load_config(config_path)

    manifest = load_manifest(cfg["input_manifest"])
    subjects = set(cfg["subjects"]) if cfg.get("subjects") else None
    sessions = {str(x) for x in cfg["sessions"]} if cfg.get("sessions") else None
    signal_rows = filter_manifest(
        manifest,
        subjects=subjects,
        sessions=sessions,
        sig_type=cfg["signal_source"],
        file_kind="signal",
    )
    if cfg.get("task_types"):
        signal_rows = signal_rows[signal_rows["task_type"].isin(cfg["task_types"])].reset_index(drop=True)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    scope = cfg["normalization"]["scope"]
    if scope not in ("per_subject_session", "per_recording"):
        raise ValueError(f"Unknown normalization scope '{scope}'")

    output_rows = []
    done = 0
    # One group = one subject-session. Pass 1 filters every recording in the
    # group while accumulating per-channel statistics; pass 2 normalises with
    # those session-wide statistics. (Filtered float32 arrays for one session
    # are ~0.4 GB for 202 one-second trials, so they are held in memory.)
    for (subject_id, session), group in signal_rows.groupby(["subject_id", "session"], sort=True):
        filtered, stats = [], None
        for _, row in group.iterrows():
            f = filter_recording(row["file_path"], cfg)
            if stats is None:
                stats = RunningChannelStats(f.shape[0])
            stats.update(f)
            filtered.append(f)

        mean, std = stats.mean, stats.std
        stats_path = output_dir / subject_id / f"session{session}" / "norm_stats.npz"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path, mean=mean, std=std, n_samples=stats.n)

        for (_, row), f in zip(group.iterrows(), filtered):
            done += 1
            print(f"[{done}/{len(signal_rows)}] Preprocessing {row['filename']} ...")
            if scope == "per_subject_session":
                processed = zscore_with_stats(f, mean, std).astype(np.float32)
            else:
                processed = zscore_normalize(f).astype(np.float32)

            out_name = f"{row['subject_id']}_session{row['session']}_{Path(row['filename']).stem}.npy"
            out_path = output_dir / row["subject_id"] / f"session{row['session']}" / out_name
            np.save(out_path, processed)

            output_rows.append(
                {
                    **row.to_dict(),
                    "preprocessed_path": str(out_path),
                    "bandpass_low_hz": cfg["bandpass"]["low_hz"],
                    "bandpass_high_hz": cfg["bandpass"]["high_hz"],
                    "notch_freq_hz": cfg["notch"]["freq_hz"],
                    "normalization": f"{cfg['normalization']['method']}_{scope}",
                    "norm_stats_path": str(stats_path),
                }
            )
        del filtered

    import pandas as pd

    write_manifest(pd.DataFrame(output_rows), cfg["output_manifest"])
    print(f"\nPreprocessed {len(output_rows)} recordings.")
    print(f"Manifest written to {cfg['output_manifest']}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: filter and normalise raw HD-sEMG recordings.")
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.yaml"))
    args = parser.parse_args()
    run_preprocessing(args.config)


if __name__ == "__main__":
    main()
