"""
Temporal Convolutional Network, sharing the same input tensor convention as
cnn1d.py and gru.py.

Implements the standard TCN residual block: two dilated causal Conv1D
layers with weight normalisation left out for MCU-deployment simplicity,
each block's dilation doubling per the configured schedule, with a 1x1
convolution on the skip path when channel counts differ.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def _residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float):
    skip = x

    y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(dropout)(y)

    y = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(dropout)(y)

    if skip.shape[-1] != filters:
        skip = layers.Conv1D(filters, 1, padding="same")(skip)

    return layers.Add()([skip, y])


def build_tcn(input_shape: tuple[int, int], num_classes: int, config: dict) -> tf.keras.Model:
    """
    Build a TCN classifier.

    `config` (see configs/models.yaml, key "tcn"): filters, kernel_size,
    dilations (list of ints, one residual block per entry), dropout.
    """
    n_channels, window_len = input_shape
    inputs = layers.Input(shape=(n_channels, window_len), name="raw_window")
    x = layers.Permute((2, 1))(inputs)  # -> (window_len, n_channels)

    for dilation_rate in config["dilations"]:
        x = _residual_block(x, config["filters"], config["kernel_size"], dilation_rate, config["dropout"])

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="tcn")
