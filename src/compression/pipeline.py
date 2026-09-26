"""
Stage 4 orchestration: trained checkpoint -> PTQ -> QAT -> pruning.

Each architecture is compressed independently through all three steps, and
accuracy is measured after every step, producing the retention curve the
Stage 3/4 comparison needs. Architecture selection for MCU deployment
(Stage 5) happens after this, on the post-compression numbers, per the
scope change agreed after the Week 6 report: an architecture that wins
uncompressed is not guaranteed to still win after quantisation and pruning.

Usage:
    python src/compression/pipeline.py --config configs/compression.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest
from src.compression.pruning import apply_pruning, strip_pruning
from src.compression.quantization import (
    apply_quantization_aware_training,
    make_representative_dataset,
    post_training_quantize,
)
from src.training.dataset import build_dataset
from src.training.metrics import per_trial_majority_vote_accuracy, per_window_accuracy
from src.training.splits import subject_dependent_split


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_keras_model(model: tf.keras.Model, X, y, trial_id) -> dict:
    y_pred = model.predict(X).argmax(axis=-1)
    return {
        "window_accuracy": per_window_accuracy(y, y_pred),
        "trial_accuracy": per_trial_majority_vote_accuracy(trial_id, y, y_pred),
    }


def compress_architecture(architecture: str, cfg: dict, splits: dict) -> list[dict]:
    """Run one architecture through PTQ, QAT, and pruning, returning one result row per stage."""
    checkpoint_path = Path(cfg["checkpoint_dir"]) / architecture / cfg["dataset"] / "best.keras"
    model = tf.keras.models.load_model(checkpoint_path)

    X_train, y_train, _ = build_dataset(splits["train"])
    X_test, y_test, trial_id_test = build_dataset(splits["test"])

    output_dir = Path(cfg["output_dir"]) / architecture
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Stage 4a: post-training quantisation, on the Stage 3 checkpoint directly.
    representative_dataset = make_representative_dataset(
        X_train, cfg["quantization"]["post_training"]["representative_dataset_size"]
    )
    ptq_bytes = post_training_quantize(model, representative_dataset)
    ptq_path = output_dir / "ptq.tflite"
    ptq_path.write_bytes(ptq_bytes)
    rows.append({"architecture": architecture, "stage": "ptq", "artifact_path": str(ptq_path),
                 "size_bytes": len(ptq_bytes)})

    # Stage 4b: quantisation-aware fine-tuning on top of the PTQ-calibrated model.
    qat_cfg = cfg["quantization"]["quantization_aware_training"]
    qat_model = apply_quantization_aware_training(model)
    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=qat_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    qat_model.fit(X_train, y_train, epochs=qat_cfg["epochs"], verbose=0)
    qat_metrics = evaluate_keras_model(qat_model, X_test, y_test, trial_id_test)
    rows.append({"architecture": architecture, "stage": "qat", **qat_metrics})

    # Stage 4c: pruning fine-tuning on top of the QAT model.
    prune_cfg = cfg["pruning"]
    steps_per_epoch = len(X_train) // 32
    pruned_model = apply_pruning(
        qat_model,
        prune_cfg["target_sparsity"],
        begin_step=0,
        end_step=steps_per_epoch * prune_cfg["epochs"],
    )
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=prune_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    pruned_model.fit(
        X_train, y_train, epochs=prune_cfg["epochs"], verbose=0,
        callbacks=[tfmot_update_pruning_step()],
    )
    pruned_model = strip_pruning(pruned_model)
    pruned_metrics = evaluate_keras_model(pruned_model, X_test, y_test, trial_id_test)

    pruned_path = output_dir / "pruned.keras"
    pruned_model.save(pruned_path)
    rows.append({"architecture": architecture, "stage": "pruned", "artifact_path": str(pruned_path),
                 **pruned_metrics})

    return rows


def tfmot_update_pruning_step():
    import tensorflow_model_optimization as tfmot
    return tfmot.sparsity.keras.UpdatePruningStep()


def run_compression(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    manifest = load_manifest(cfg["input_manifest"])

    split_cfg = {"train_fraction": 0.7, "val_fraction": 0.15, "test_fraction": 0.15, "random_seed": 42}
    splits = subject_dependent_split(manifest, **split_cfg)

    all_rows = []
    for architecture in cfg["architectures"]:
        print(f"Compressing {architecture} ...")
        all_rows.extend(compress_architecture(architecture, cfg, splits))

    output_path = Path(cfg["retention_manifest"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    print(f"\nAccuracy retention manifest written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: compress each architecture through PTQ, QAT, and pruning.")
    parser.add_argument("--config", type=Path, default=Path("configs/compression.yaml"))
    args = parser.parse_args()
    run_compression(args.config)


if __name__ == "__main__":
    main()
