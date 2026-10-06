"""
Stage 3 orchestration: windowed tensors -> trained checkpoint + evaluation.

Loads the Stage 2a manifest, applies the subject-dependent (trial-level,
gesture-stratified) split, builds the requested architecture through the
model registry, trains it, and reports per-window and per-trial
(majority-vote) accuracy on the held-out test split. One run trains one
architecture; the Stage 3 comparison across 1D CNN / GRU / TCN is driven by
running this script once per architecture.

Everything needed for the progress report is written next to the checkpoint:
metrics.json (accuracies, parameter count, FLOPs, desktop latency),
history.json (loss/accuracy curves) and test_predictions.npz (for confusion
matrices).

Optionally (cfg["cross_session_eval"]) the trained model is also scored on
every trial of the sessions NOT used for training, as an early preview of
the Stage 6 cross-session evaluation. That preview never influences
training or model selection.

Usage:
    python src/training/train.py --config configs/training.yaml
    python src/training/train.py --config configs/training.yaml --architecture gru --epochs 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.manifest import load_manifest
from src.models.registry import build_model
from src.training.dataset import build_dataset, load_trial_label
from src.training.metrics import (
    per_trial_majority_vote_accuracy,
    per_window_accuracy,
    trial_majority_vote,
)
from src.training.splits import subject_dependent_split


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def select_trials(manifest, data_cfg: dict):
    """Apply the task_type / sessions scope from the training config."""
    out = manifest
    if data_cfg.get("subjects"):
        out = out[out["subject_id"].str.replace("subject", "", regex=False).isin(data_cfg["subjects"])]
    if data_cfg.get("task_type"):
        out = out[out["task_type"] == data_cfg["task_type"]]
    return out.reset_index(drop=True)


def _gru_flops(model) -> int:
    """
    Analytic FLOPs for the GRU layers. TensorFlow's profiler does not count
    ops inside the recurrent while-loop, so they are added separately:
    per time step, 3 gates x (in*h + h*h) multiply-accumulates, x2 for FLOPs.
    """
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GRU):
            _, steps, n_in = layer.input.shape
            h = layer.units
            total += 2 * steps * 3 * (n_in * h + h * h)
    return total


def count_flops(model, input_shape) -> int | None:
    """Float-op count for one inference (batch of 1): profiler graph ops + analytic GRU ops."""
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(
            tf.TensorSpec([1, *input_shape], tf.float32)
        )
        frozen = convert_variables_to_constants_v2(fn)
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        info = tf.compat.v1.profiler.profile(frozen.graph, options=opts)
        return int(info.total_float_ops) + _gru_flops(model)
    except Exception as e:  # profiling is best-effort evidence, never fatal
        print(f"FLOPs profiling skipped: {e}")
        return None


def measure_latency_ms(model, input_shape, n_runs: int = 50) -> float:
    """Median single-window desktop inference latency (batch 1, CPU)."""
    x = np.random.randn(1, *input_shape).astype(np.float32)
    fn = tf.function(lambda t: model(t, training=False))
    for _ in range(5):
        fn(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(x).numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def predict_trials(model, trials, batch_size: int):
    """Stream trial by trial (keeps RAM low). Returns y_true, y_pred, trial_id."""
    y_true, y_pred, tid = [], [], []
    for i, (_, row) in enumerate(trials.iterrows()):
        w = np.load(row["windowed_path"])["X"].astype(np.float32)
        p = model.predict(w, batch_size=batch_size, verbose=0).argmax(-1)
        y_pred.append(p)
        y_true.append(np.full(len(p), load_trial_label(row)))
        tid.append(np.full(len(p), i))
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(tid)


def run_training(
    config_path: str | Path,
    architecture: str | None = None,
    epochs: int | None = None,
    seed: int | None = None,
    variant: str | None = None,
    model_opts: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    model_cfg = load_config(cfg["model_config_path"])
    architecture = architecture or cfg["architecture"]
    train_cfg = dict(cfg["training"])
    if epochs:
        train_cfg["epochs"] = epochs
    if model_opts:
        model_cfg[architecture] = {**model_cfg[architecture], **model_opts}
    base_seed = cfg["split"]["random_seed"]
    if seed is not None:
        cfg["split"] = {**cfg["split"], "random_seed": seed}

    tf.keras.utils.set_random_seed(cfg["split"]["random_seed"])

    data_cfg = cfg.get("data", {})
    manifest = select_trials(load_manifest(cfg["input_manifest"]), data_cfg)
    train_sessions = {str(s) for s in data_cfg.get("train_sessions", manifest["session"].unique())}
    in_scope = manifest[manifest["session"].astype(str).isin(train_sessions)].reset_index(drop=True)

    split_cfg = cfg["split"]
    splits = subject_dependent_split(
        in_scope,
        split_cfg["train_fraction"],
        split_cfg["val_fraction"],
        split_cfg["test_fraction"],
        split_cfg["random_seed"],
    )

    # Leakage guard: no trial may appear in more than one split.
    keys = {k: set(v["windowed_path"]) for k, v in splits.items()}
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"]), \
        "Trial leakage between splits"
    print({k: len(v) for k, v in splits.items()}, "trials per split")

    print("Loading windowed tensors ...")
    X_train, y_train, _ = build_dataset(splits["train"])
    X_val, y_val, _ = build_dataset(splits["val"])
    print(f"train {X_train.shape}, val {X_val.shape}")

    input_shape = tuple(model_cfg["input_shape"])
    assert X_train.shape[1:] == input_shape, f"Model input {input_shape} != data {X_train.shape[1:]}"

    model = build_model(architecture, input_shape, model_cfg["num_classes"], model_cfg[architecture])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    run_name = architecture + (f"_{variant}" if variant else "")
    if cfg["split"]["random_seed"] != base_seed:
        run_name += f"_seed{cfg['split']['random_seed']}"
    output_dir = Path(cfg["output_dir"]) / run_name / cfg["dataset"]
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=train_cfg["early_stopping_patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(str(output_dir / "best.keras"), save_best_only=True),
    ]

    t0 = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.time() - t0
    del X_train, y_train, X_val, y_val

    # Held-out test split (trials never seen in training or validation).
    y_test, y_pred, trial_id_test = predict_trials(model, splits["test"], train_cfg["batch_size"])
    window_acc = per_window_accuracy(y_test, y_pred)
    trial_acc = per_trial_majority_vote_accuracy(trial_id_test, y_test, y_pred)
    trial_true, trial_pred = trial_majority_vote(trial_id_test, y_test, y_pred)

    metrics = {
        "architecture": architecture,
        "variant": variant or "",
        "seed": cfg["split"]["random_seed"],
        "model_config": model_cfg[architecture],
        "dataset": cfg["dataset"],
        "subjects": sorted(in_scope["subject_id"].unique().tolist()),
        "task_type": data_cfg.get("task_type"),
        "train_sessions": sorted(train_sessions),
        "n_trials": {k: int(len(v)) for k, v in splits.items()},
        "n_test_windows": int(len(y_test)),
        "per_window_accuracy": window_acc,
        "per_trial_accuracy": trial_acc,
        "epochs_run": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "train_seconds": round(train_seconds, 1),
        "parameters": int(model.count_params()),
        "flops_per_inference": count_flops(model, input_shape),
        "latency_ms_desktop_cpu": measure_latency_ms(model, input_shape),
    }

    np.savez(
        output_dir / "test_predictions.npz",
        y_true=y_test, y_pred=y_pred, trial_id=trial_id_test,
        trial_true=trial_true, trial_pred=trial_pred,
    )
    with open(output_dir / "history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)

    # Optional: preview of Stage 6, scoring on sessions never used for training.
    if cfg.get("cross_session_eval"):
        other = manifest[~manifest["session"].astype(str).isin(train_sessions)].reset_index(drop=True)
        if len(other):
            yt, yp, tid = predict_trials(model, other, train_cfg["batch_size"])
            metrics["cross_session"] = {
                "eval_sessions": sorted(other["session"].astype(str).unique().tolist()),
                "n_trials": int(len(other)),
                "per_window_accuracy": per_window_accuracy(yt, yp),
                "per_trial_accuracy": per_trial_majority_vote_accuracy(tid, yt, yp),
            }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{architecture} on {cfg['dataset']}:")
    print(f"  Per-window accuracy: {window_acc:.4f}")
    print(f"  Per-trial (majority vote) accuracy: {trial_acc:.4f}")
    if "cross_session" in metrics:
        cs = metrics["cross_session"]
        print(f"  Cross-session (preview) per-window {cs['per_window_accuracy']:.4f}, per-trial {cs['per_trial_accuracy']:.4f}")
    print(f"  Outputs saved to {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 3: train and evaluate one architecture.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--architecture", choices=["cnn1d", "gru", "tcn"], default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override split.random_seed (repeat-split runs)")
    parser.add_argument("--variant", default=None, help="Label for a model-config variant, e.g. pool8")
    parser.add_argument("--model-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a key of this architecture's model config, e.g. input_pool=8")
    args = parser.parse_args()
    opts = {}
    for kv in args.model_opt:
        k, v = kv.split("=", 1)
        opts[k] = yaml.safe_load(v)
    run_training(args.config, args.architecture, args.epochs, args.seed, args.variant, opts or None)


if __name__ == "__main__":
    main()
