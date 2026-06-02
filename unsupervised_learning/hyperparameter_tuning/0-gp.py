#!/usr/bin/env python3
"""Gaussian Process (noiseless 1D)"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian Process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Constructor

        Args:
            X_init: inputs already sampled (t, 1)
            Y_init: outputs for X_init (t, 1)
            l: length parameter
            sigma_f: standard deviation of function output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates RBF covariance kernel matrix

        Args:
            X1: (m, 1)
            X2: (n, 1)

        Returns:
            Kernel matrix (m, n)
        """
        # squared distance (m, n)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \ np.sum(X2**2, 1) - \
                 2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-sqdist / (2 * self.l ** 2))
  
