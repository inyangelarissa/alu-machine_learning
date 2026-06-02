#!/usr/bin/env python3
"""Creates a layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing output of previous layer
        n: number of nodes in the layer
        activation: activation function
        keep_prob: probability a node is kept

    Returns:
        Output tensor of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(0.0)

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer
    )

    output = layer(prev)

    # Apply dropout (rate = 1 - keep_prob)
    output = tf.layers.dropout(output, rate=1 - keep_prob)

    return output
