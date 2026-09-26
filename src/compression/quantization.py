"""
Quantisation for Stage 4: post-training quantisation (PTQ) first, then
quantisation-aware training (QAT) on top of the PTQ-calibrated model.
"""

from collections.abc import Iterator

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def make_representative_dataset(X: np.ndarray, n_samples: int):
    """
    Build the representative_dataset generator TFLiteConverter needs to
    calibrate PTQ activation ranges, sampling `n_samples` trials from `X`.
    """
    indices = np.random.default_rng(0).choice(len(X), size=min(n_samples, len(X)), replace=False)

    def representative_dataset() -> Iterator[list[np.ndarray]]:
        for i in indices:
            yield [X[i : i + 1].astype(np.float32)]

    return representative_dataset


def post_training_quantize(model: tf.keras.Model, representative_dataset) -> bytes:
    """
    Convert a trained Keras model to a full-integer-quantised TFLite model.

    Full-integer (not just weight) quantisation is used since the MCU
    deployment target (TensorFlow Lite for Microcontrollers) requires an
    integer-only model.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def apply_quantization_aware_training(model: tf.keras.Model) -> tf.keras.Model:
    """
    Wrap a trained Keras model for QAT, inserting fake-quantisation nodes so
    subsequent fine-tuning adapts the weights to quantised inference before
    the final PTQ export.
    """
    return tfmot.quantization.keras.quantize_model(model)
