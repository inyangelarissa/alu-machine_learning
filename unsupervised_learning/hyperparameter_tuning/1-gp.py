#!/usr/bin/env python3
"""Gaussian Process with update"""

import numpy as np


class GaussianProcess:
    """
    Noiseless 1D Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        sqdist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1)
            - 2 * np.dot(X1, X2.T)
        )

        return (self.sigma_f ** 2) * np.exp(-sqdist / (2 * self.l ** 2))

    def predict(self, X_s):
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)

        K_inv = np.linalg.inv(K + 1e-10 * np.eye(K.shape[0]))

        mu = K_s.T @ K_inv @ self.Y
        mu = mu.reshape(-1)

        cov = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diag(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates Gaussian Process with new data point

        Args:
            X_new: (1,) new input
            Y_new: (1,) new output
        """

        # reshape to (1, 1)
        X_new = X_new.reshape(1, -1)
        Y_new = Y_new.reshape(1, -1)

        # append new data
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)

        # recompute kernel
        self.K = self.kernel(self.X, self.X)
