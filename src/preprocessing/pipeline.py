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
from src.preprocessing.normalization import zscore_normalize


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess_recording(dat_path: str, fs: float, cfg: dict) -> np.ndarray:
    """Load, filter, and normalise a single Hyser recording."""
    signal, record_fs, _ = read_hyser_record(dat_path)
    filtered = apply_filter_chain(signal, record_fs, cfg)
    return zscore_normalize(filtered)


def run_preprocessing(config_path: str | Path) -> None:
    cfg = load_config(config_path)

    manifest = load_manifest(cfg["input_manifest"])
    signal_rows = filter_manifest(
        manifest, sig_type=cfg["signal_source"], file_kind="signal"
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for i, row in signal_rows.iterrows():
        print(f"[{i + 1}/{len(signal_rows)}] Preprocessing {row['filename']} ...")

        processed = preprocess_recording(row["file_path"], cfg["sampling_rate_hz"], cfg)

        out_name = f"{row['subject_id']}_session{row['session']}_{Path(row['filename']).stem}.npy"
        out_path = output_dir / row["subject_id"] / f"session{row['session']}" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, processed)

        output_rows.append(
            {
                **row.to_dict(),
                "preprocessed_path": str(out_path),
                "bandpass_low_hz": cfg["bandpass"]["low_hz"],
                "bandpass_high_hz": cfg["bandpass"]["high_hz"],
                "notch_freq_hz": cfg["notch"]["freq_hz"],
                "normalization": cfg["normalization"]["method"],
            }
        )

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
