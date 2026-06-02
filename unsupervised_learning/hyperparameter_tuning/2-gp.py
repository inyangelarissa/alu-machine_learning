#!/usr/bin/env python3
"""Gaussian Process"""
import numpy as np


class GaussianProcess:
    """Gaussian Process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix"""
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1)
                  + np.sum(X2**2, 1)
                  - 2 * np.matmul(X1, X2.T))

        return self.sigma_f**2 * np.exp(
            -0.5 / self.l**2 * sqdist
        )

    def update(self, X_new, Y_new):
        """updates the Gaussian Process with a new sample point"""

        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)

        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        self.K = self.kernel(self.X, self.X)
