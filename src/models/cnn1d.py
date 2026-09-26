"""
1D CNN architecture.

All three architectures (this module, gru.py, tcn.py) accept the same
input tensor shape, so any accuracy/latency/size difference measured in
Stage 3 is attributable to architecture choice alone, per the project
framework decision.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn1d(input_shape: tuple[int, int], num_classes: int, config: dict) -> tf.keras.Model:
    """
    Build a stacked Conv1D classifier.

    `input_shape` is (n_channels, window_len); Keras Conv1D expects
    channels-last, so the input is transposed to (window_len, n_channels)
    inside the model.

    `config` (see configs/models.yaml, key "cnn1d"): filters (list of ints,
    one per conv block), kernel_size, pool_size, dropout.
    """
    n_channels, window_len = input_shape
    inputs = layers.Input(shape=(n_channels, window_len), name="raw_window")
    x = layers.Permute((2, 1))(inputs)  # -> (window_len, n_channels)

    for n_filters in config["filters"]:
        x = layers.Conv1D(n_filters, config["kernel_size"], padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(config["pool_size"])(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="cnn1d")
