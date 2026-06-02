#!/usr/bin/env python3
"""Creates a layer with L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes in the layer
        activation: activation function
        lambtha: L2 regularization parameter

    Returns:
        Output tensor of the layer
    """
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
        ),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha)
    )

    return layer(prev)
