from sympy import symbols, Derivative, Function, exp
from sympy import diff, lambdify
import numpy as np
from scipy.stats import ncx2


def approx(v0, T, r, K, sig, kappa, m, gamma):
    t, v, sigma = symbols("t v sigma")

    def aux(sigma):
        """
        auxiliary model
        :param sigma: nuiance parameter sigma
        :return:
        """
        alpha = kappa
        beta = kappa * m
        # for non-central chi square
        x = 4 * beta / (sigma ** 2 * (1 - np.exp(-beta * T)))
        df = 4 * alpha / sigma ** 2
        nc = x * np.exp(-beta * T) * v0
        # compute for call price
        term1 = np.exp(-beta * T) * v0 * (1 - ncx2.cdf(x * K, df + 4, nc))
        term2 = (alpha / beta) * (1 - np.exp(-beta * T)) * (1 - ncx2.cdf(x * K, df + 2, nc))
        term3 = K * (1 - ncx2.cdf(x * K, df, nc))
        call = np.exp(-r * T) * (term1 + term2 - term3)
        return call
