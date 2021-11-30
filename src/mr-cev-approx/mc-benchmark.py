import numpy as np
from scipy.stats import ncx2
import matplotlib.pyplot as plt


def mc(v, K, T, r, kappa, theta, sigma, gamma, n, m):
    """
    monte carlo simulation for mean reverting cev model
    :param v:
    :param K:
    :param T:
    :param r:
    :param kappa:
    :param theta:
    :param sigma:
    :param gamma:
    :param n:
    :param m:
    :return:
    """
    dt = T / n
    v = np.ones(m) * v
    for i in range(n):
        z1 = np.random.normal(0, 1, size=m)
        v = v + kappa * (theta - v) * dt + sigma * np.power(v, gamma) * z1 * np.sqrt(dt)
        v = np.maximum(v, 0)

    payoffs = np.maximum(v - K, 0)
    p = payoffs.mean() * np.exp(-r * T)
    return p


if __name__ == '__main__':
    # square root mean reverting model as test

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


    m = 0.15
    kappa = 4
    alpha = kappa * m
    sigma = np.sqrt(0.133)
    r = 0.05
    K = 0.15
    T1 = 0.3
    T2 = 0.5
    T3 = 0.1
    V = np.arange(0.01, 0.5, 0.01)
    benchmark1 = srmr(alpha, kappa, sigma, r, V, K, T1)
    benchmark2 = srmr(alpha, kappa, sigma, r, V, K, T2)
    benchmark3 = srmr(alpha, kappa, sigma, r, V, K, T3)
    test1 = []
    test2 = []
    test3 = []

    for v in V:
        tmp1 = mc(v, K, T1, r, kappa, m, sigma, 0.5, 100, 10000)
        tmp2 = mc(v, K, T2, r, kappa, m, sigma, 0.5, 100, 10000)
        tmp3 = mc(v, K, T3, r, kappa, m, sigma, 0.5, 100, 10000)
        test1.append(tmp1)
        test2.append(tmp2)
        test3.append(tmp3)

    # plots
    plt.figure(1)
    plt.plot(V, test1, label='test')
    plt.plot(V, benchmark1,label='benchmark')
    plt.legend()
    plt.figure(2)
    plt.plot(V, test2, label='test')
    plt.plot(V, benchmark2,label='benchmark')
    plt.legend()
    plt.figure(3)
    plt.plot(V, test3, label='test')
    plt.plot(V, benchmark3,label='benchmark')
    plt.legend()
    plt.show()