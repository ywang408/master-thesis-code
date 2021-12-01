from scipy.stats import ncx2
import numpy as np
import matplotlib.pyplot as plt


def srmr(alpha, beta, sigma, r, V, K, T):
    # for non-central chi square
    x = 4 * beta / (sigma ** 2 * (1 - np.exp(-beta * T)))
    df = 4 * alpha / sigma ** 2
    nc = x * np.exp(-beta * T) * V
    # compute for call price
    term1 = np.exp(-beta * T) * V * (1 - ncx2.cdf(x * K, df + 4, nc))
    term2 = (alpha / beta) * (1 - np.exp(-beta * T)) * (1 - ncx2.cdf(x * K, df + 2, nc))
    term3 = K * (1 - ncx2.cdf(x * K, df, nc))
    call = np.exp(-r * T) * (term1 + term2 - term3)
    return call


def delta(alpha, beta, sigma, r, V, K, T):
    x = 4 * beta / (sigma ** 2 * (1 - np.exp(-beta * T)))
    df = 4 * alpha / sigma ** 2
    nc = x * np.exp(-beta * T) * V
    coef = np.exp(-(beta + r) * T)
    tmp = (1 - ncx2.cdf(x * K, df + 4, nc)) - 2 * ncx2.pdf(x * K, df + 4, nc)
    return coef * tmp


def gamma(alpha, beta, sigma, r, V, K, T):
    x = 4 * beta / (sigma ** 2 * (1 - np.exp(-beta * T)))
    df = 4 * alpha / sigma ** 2
    nc = x * np.exp(-beta * T) * V
    coef = x * np.exp(-(2 * kappa + r) * T)
    return coef * ncx2.pdf(x * K, df + 4, nc)


if __name__ == '__main__':
    # parameter settings, equal to Griinbichler's paper
    m = 0.15
    kappa = 4
    alpha = kappa * m
    sigma = np.sqrt(0.133)
    r = 0.05
    K = 0.15
    T1 = 0.3
    T2 = 0.5
    T3 = 0.1
    step_size = 0.002
    v1 = np.arange(0.01, 0.51, step_size)
    v2 = np.arange(0.01+step_size, 0.51-step_size, step_size)

    # comparison
    p = srmr(alpha, kappa, sigma, r, v1, K, T1)
    delta2 = delta(alpha, kappa, sigma, r, v2, K, T1) # our formula
    delta1 = (p[2:] - p[:-2])/step_size/2 # fdm


    v3 = np.arange(0.01+2*step_size, 0.51-2*step_size, step_size)
    gamma1 = gamma(alpha, kappa, sigma, r, v3, K, T1) # our formula
    gamma2 = (delta1[2:] - delta1[:-2])/step_size/2

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
