from scipy.stats import ncx2
import numpy as np


def srmr(kappa, theta, sigma, r, V, K, T):
    alpha = kappa * theta
    # parameters for non-central chi square
    x = 4 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * T)))
    df = 4 * alpha / sigma ** 2
    nc = x * np.exp(-kappa * T) * V
    # compute for call price
    term1 = np.exp(-kappa * T) * V * (1 - ncx2.cdf(x * K, df + 4, nc))
    term2 = (alpha / kappa) * (1 - np.exp(-kappa * T)) * (1 - ncx2.cdf(x * K, df + 2, nc))
    term3 = K * (1 - ncx2.cdf(x * K, df, nc))
    call = np.exp(-r * T) * (term1 + term2 - term3)
    return call
