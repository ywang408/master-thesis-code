import numpy as np


def mc(s, v, K, T, r, kappa, theta, sigma, rho, n, m):
    dt = T / n
    s = np.ones(m) * s
    v = np.ones(m) * v
    for i in range(n):
        z1 = np.random.normal(0, 1, size=m)
        z2 = np.random.normal(0, 1, size=m)
        z3 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        s = s + r * s * dt + v ** 0.5 * s * z1 * np.sqrt(dt)
        v = v + kappa * (theta - v) * dt + sigma * v ** 0.5 * z3 * np.sqrt(dt)
        v = np.maximum(v, 0)

    payoffs = np.maximum(s - K, 0)
    p = payoffs.mean() * np.exp(-r * T)
    return p
