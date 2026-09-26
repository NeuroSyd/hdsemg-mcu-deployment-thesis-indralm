"""
Stage 3 orchestration: windowed tensors -> trained checkpoint + evaluation.

Loads the Stage 2a manifest, applies the subject-dependent split, builds the
requested architecture through the Stage 3 model registry, trains it, and
reports both per-window and per-trial-majority-vote accuracy on the held-out
test split. One run trains one architecture; the Stage 3 comparison across
1D CNN / GRU / TCN is driven by running this script once per architecture
name in configs/training.yaml.

Usage:
    python src/training/train.py --config configs/training.yaml
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest
from src.models.registry import build_model
from src.training.dataset import build_dataset
from src.training.metrics import per_trial_majority_vote_accuracy, per_window_accuracy
from src.training.splits import subject_dependent_split


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    model_cfg = load_config(cfg["model_config_path"])

    manifest = load_manifest(cfg["input_manifest"])
    split_cfg = cfg["split"]
    splits = subject_dependent_split(
        manifest,
        split_cfg["train_fraction"],
        split_cfg["val_fraction"],
        split_cfg["test_fraction"],
        split_cfg["random_seed"],
    )

    print("Loading windowed tensors ...")
    X_train, y_train, _ = build_dataset(splits["train"])
    X_val, y_val, _ = build_dataset(splits["val"])
    X_test, y_test, trial_id_test = build_dataset(splits["test"])

    architecture = cfg["architecture"]
    model = build_model(
        architecture,
        tuple(model_cfg["input_shape"]),
        model_cfg["num_classes"],
        model_cfg[architecture],
    )

    train_cfg = cfg["training"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_dir = Path(cfg["output_dir"]) / architecture / cfg["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=train_cfg["early_stopping_patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best.keras"), save_best_only=True
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_test).argmax(axis=-1)
    window_acc = per_window_accuracy(y_test, y_pred)
    trial_acc = per_trial_majority_vote_accuracy(trial_id_test, y_test, y_pred)

    print(f"\n{architecture} on {cfg['dataset']}:")
    print(f"  Per-window accuracy: {window_acc:.4f}")
    print(f"  Per-trial (majority vote) accuracy: {trial_acc:.4f}")
    print(f"  Checkpoint saved to {output_dir / 'best.keras'}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: train and evaluate one architecture.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
