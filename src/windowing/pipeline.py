"""
Stage 2a orchestration: preprocessed signal -> windowed tensors.

Reads the Stage 1 manifest, segments each recording into overlapping
windows, and saves each recording's windows as a single .npz (array `X` of
shape (n_windows, n_channels, window_len)) so Stage 3 training and Stage 2b
feature extraction share one windowed representation.

Gesture-label association per window is left for the dataset-specific label
loader (label_<task>.txt for Hyser) to attach once its column layout is
confirmed against the Hyser documentation; window boundaries alone are
produced here.

Usage:
    python src/windowing/pipeline.py --config configs/windowing.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest, write_manifest
from src.windowing.windowing import compute_window_params, sliding_windows


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_windowing(config_path: str | Path) -> None:
    cfg = load_config(config_path)

    manifest = load_manifest(cfg["input_manifest"])
    window_len, step = compute_window_params(
        cfg["window_ms"], cfg["overlap"], cfg["sampling_rate_hz"]
    )

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for i, row in manifest.iterrows():
        print(f"[{i + 1}/{len(manifest)}] Windowing {row['preprocessed_path']} ...")

        signal = np.load(row["preprocessed_path"])
        windows = sliding_windows(signal, window_len, step)

        out_path = output_dir / f"{Path(row['preprocessed_path']).stem}_windows.npz"
        np.savez(out_path, X=windows)

        output_rows.append(
            {
                **row.to_dict(),
                "windowed_path": str(out_path),
                "n_windows": windows.shape[0],
                "window_len_samples": window_len,
                "step_samples": step,
            }
        )

    write_manifest(pd.DataFrame(output_rows), cfg["output_manifest"])
    print(f"\nWindowed {len(output_rows)} recordings.")
    print(f"Manifest written to {cfg['output_manifest']}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2a: segment preprocessed signals into windows.")
    parser.add_argument("--config", type=Path, default=Path("configs/windowing.yaml"))
    args = parser.parse_args()
    run_windowing(args.config)


if __name__ == "__main__":
    main()
