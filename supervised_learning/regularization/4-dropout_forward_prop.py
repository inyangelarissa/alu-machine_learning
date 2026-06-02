#!/usr/bin/env python3
"""Forward propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: input data of shape (nx, m)
        weights: dictionary of weights and biases
        L: number of layers
        keep_prob: probability of keeping a node active

    Returns:
        cache: dictionary containing activations and dropout masks
    """
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        A_prev = cache["A{}".format(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            t = np.exp(Z)
            cache["A{}".format(i)] = t / np.sum(t, axis=0,
                                                keepdims=True)
        else:
            A = np.tanh(Z)

            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A = A / keep_prob

            cache["D{}".format(i)] = D
            cache["A{}".format(i)] = A

    return cache
