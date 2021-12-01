from scipy.stats import ncx2
import numpy as np
import matplotlib.pyplot as plt
from srmr import srmr


def delta(kappa, theta, sigma, r, V, K, T):
    x = 4 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * T)))
    df = 4 * kappa * theta / sigma ** 2
    nc = x * np.exp(-kappa * T) * V
    coef = np.exp(-(kappa + r) * T)
    tmp = (1 - ncx2.cdf(x * K, df + 4, nc)) - 2 * ncx2.pdf(x * K, df + 4, nc)
    return coef * tmp


def gamma(kappa, theta, sigma, r, V, K, T):
    x = 4 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * T)))
    df = 4 * kappa * theta / sigma ** 2
    nc = x * np.exp(-kappa * T) * V
    coef = x * np.exp(-(2 * kappa + r) * T)
    return coef * ncx2.pdf(x * K, df + 4, nc)


if __name__ == '__main__':
    # parameter settings, equal to Griinbichler's paper
    theta = 0.15
    kappa = 4
    sigma = np.sqrt(0.133)
    r = 0.05
    K = 0.15
    T1 = 0.3
    T2 = 0.5
    T3 = 0.1
    step_size = 0.002
    v1 = np.arange(0.01, 0.51, step_size)
    v2 = np.arange(0.01 + step_size, 0.51 - step_size, step_size)

    # comparison
    p = srmr(kappa, theta, sigma, r, v1, K, T1)
    delta2 = delta(kappa, theta, sigma, r, v2, K, T1)  # our formula
    delta1 = (p[2:] - p[:-2]) / step_size / 2  # fdm

    v3 = np.arange(0.01 + 2 * step_size, 0.51 - 2 * step_size, step_size)
    gamma1 = gamma(kappa, theta, sigma, r, v3, K, T1)  # our formula
    gamma2 = (delta1[2:] - delta1[:-2]) / step_size / 2

    plt.figure(1)
    plt.xlabel('vol')
    plt.ylabel('delta')
    plt.plot(v2, delta2, label='analytical delta')
    plt.plot(v2, delta1, label='Finite Difference delta')
    plt.legend()

    plt.figure(2)
    plt.xlabel('vol')
    plt.ylabel('delta diff')
    plt.plot(v2, (delta2 - delta1))

    plt.figure(3)
    plt.xlabel('vol')
    plt.ylabel('gamma')
    plt.plot(v3, gamma1, label='analytical gamma')
    plt.plot(v3, gamma2, label='Finite Difference gamma')
    plt.legend()

    plt.figure(4)
    plt.xlabel('vol')
    plt.ylabel('gamma diff')
    plt.plot(v3, (gamma1 - gamma2))
    plt.show()
