from scipy.stats import norm
import numpy as np

def acquisition(self):
    """
    Calculates the next best sample location using
    the Expected Improvement acquisition function.

    Returns:
        X_next: numpy.ndarray of shape (1,)
        EI: numpy.ndarray of shape (ac_samples,)
    """

    mu, sigma = self.gp.predict(self.X_s)

    sigma = sigma.reshape(-1)

    if self.minimize:
        Y_best = np.min(self.gp.Y)
        improvement = Y_best - mu - self.xsi
    else:
        Y_best = np.max(self.gp.Y)
        improvement = mu - Y_best - self.xsi

    with np.errstate(divide='ignore'):
        Z = improvement / sigma

        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        EI[sigma == 0.0] = 0.0

    X_next = self.X_s[np.argmax(EI)]

    return X_next, EI
