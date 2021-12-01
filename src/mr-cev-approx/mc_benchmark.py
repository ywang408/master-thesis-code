import numpy as np
import matplotlib.pyplot as plt
from srmr import srmr


def mc(v, K, T, r, kappa, theta, sigma, gamma, n, m):
    """
    monte carlo simulation for mean reverting cev model
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
    # parameter settings
    theta = 0.15
    kappa = 4
    sigma = np.sqrt(0.133)
    r = 0.05
    K = 0.15
    T1 = 0.3
    T2 = 0.5
    T3 = 0.1
    V = np.arange(0.01, 0.5, 0.01)

    # analytical result as benchmark
    benchmark1 = srmr(kappa, theta, sigma, r, V, K, T1)
    benchmark2 = srmr(kappa, theta, sigma, r, V, K, T2)
    benchmark3 = srmr(kappa, theta, sigma, r, V, K, T3)

    # monte carlo result
    test1 = []
    test2 = []
    test3 = []
    for v in V:
        tmp1 = mc(v, K, T1, r, kappa, theta, sigma, 0.5, 200, 50000)
        tmp2 = mc(v, K, T2, r, kappa, theta, sigma, 0.5, 200, 50000)
        tmp3 = mc(v, K, T3, r, kappa, theta, sigma, 0.5, 200, 50000)
        test1.append(tmp1)
        test2.append(tmp2)
        test3.append(tmp3)

    # plots
    plt.figure(1)
    plt.title(f"T={T1}")
    plt.xlabel('vol')
    plt.ylabel('price')
    plt.plot(V, test1, label='monte carlo')
    plt.plot(V, benchmark1, label='benchmark')
    plt.legend()

    plt.figure(2)
    plt.title(f"T={T2}")
    plt.xlabel('vol')
    plt.ylabel('price')
    plt.plot(V, test2, label='monte carlo')
    plt.plot(V, benchmark2, label='benchmark')
    plt.legend()

    plt.figure(3)
    plt.title(f"T={T3}")
    plt.xlabel('vol')
    plt.ylabel('price')
    plt.plot(V, test3, label='monte carlo')
    plt.plot(V, benchmark3, label='benchmark')
    plt.legend()
    plt.show()
