"""
Stage 2b orchestration: windowed tensors -> classical feature matrix (LDA baseline only).

Reads the Stage 2a manifest, computes the configured feature set for every
window of every recording, and writes one combined feature matrix (rows =
windows, columns = features) as CSV.

Usage:
    python src/features/pipeline.py --config configs/features.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest
from src.features.time_domain import extract_feature_vector


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_feature_extraction(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    manifest = load_manifest(cfg["input_manifest"])

    rows = []
    for i, row in manifest.iterrows():
        print(f"[{i + 1}/{len(manifest)}] Extracting features from {row['windowed_path']} ...")

        windows = np.load(row["windowed_path"])["X"]
        for window_idx in range(windows.shape[0]):
            vector = extract_feature_vector(windows[window_idx], cfg["features"], cfg)
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "session": row["session"],
                    "windowed_path": row["windowed_path"],
                    "window_index": window_idx,
                    **{f"feat_{j}": v for j, v in enumerate(vector)},
                }
            )

    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nExtracted features for {len(rows)} windows.")
    print(f"Feature matrix written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2b: compute classical time-domain features for the LDA baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    args = parser.parse_args()
    run_feature_extraction(args.config)


if __name__ == "__main__":
    main()
