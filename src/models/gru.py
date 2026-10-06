"""GRU architecture, sharing the same input tensor convention as cnn1d.py and tcn.py."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_gru(input_shape: tuple[int, int], num_classes: int, config: dict) -> tf.keras.Model:
    """
    Build a stacked GRU classifier.

    `config` (see configs/models.yaml, key "gru"): units (list of ints, one
    per GRU layer), dropout, bidirectional, and optional `input_pool` (int,
    default 1): average-pool the time axis by this factor before the GRU.
    The model input tensor is unchanged (still the shared raw window); pooling
    only shortens the sequence the recurrent layers have to unroll
    (410 steps -> 410 // input_pool), which is the usual fix when a GRU
    memorises long raw sEMG sequences instead of generalising.
    """
    n_channels, window_len = input_shape
    inputs = layers.Input(shape=(n_channels, window_len), name="raw_window")
    x = layers.Permute((2, 1))(inputs)  # -> (window_len, n_channels) as the GRU time axis
    pool = int(config.get("input_pool", 1))
    if pool > 1:
        x = layers.AveragePooling1D(pool)(x)

    units_list = config["units"]
    for i, units in enumerate(units_list):
        return_sequences = i < len(units_list) - 1
        gru_layer = layers.GRU(units, return_sequences=return_sequences)
        if config.get("bidirectional", False):
            gru_layer = layers.Bidirectional(gru_layer)
        x = gru_layer(x)

    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="gru")
