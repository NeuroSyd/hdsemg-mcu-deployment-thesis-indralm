"""
Magnitude-based pruning for Stage 4, applied last in the PTQ -> QAT ->
pruning order, on top of the already quantisation-aware-trained model.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def apply_pruning(
    model: tf.keras.Model, target_sparsity: float, begin_step: int, end_step: int
) -> tf.keras.Model:
    """
    Wrap a model for magnitude-based pruning, ramping sparsity from 0 to
    `target_sparsity` on a polynomial decay schedule between `begin_step`
    and `end_step` (both measured in training steps, not epochs).
    """
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=target_sparsity,
        begin_step=begin_step,
        end_step=end_step,
    )
    return tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)


def strip_pruning(model: tf.keras.Model) -> tf.keras.Model:
    """Remove the pruning wrapper after fine-tuning, leaving a plain Keras model with the sparsified weights."""
    return tfmot.sparsity.keras.strip_pruning(model)
