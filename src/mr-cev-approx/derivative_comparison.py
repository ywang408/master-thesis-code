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


def de2t(kappa, theta, sigma, r, V, K, T):
    """
    it's theta greek, to avoid conflicts set it name to de2t
    """
    x = 4 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * T)))
    df = 4 * kappa * theta / sigma ** 2
    nc = x * np.exp(-kappa * T) * V

    def dq(nu):
        coef = -kappa * x * np.exp(-kappa * T) / (1 - np.exp(-kappa * T))
        return coef * (-K * ncx2.pdf(x * K, nu, nc) + V * ncx2.pdf(x * K, nu + 2, nc))

    term1 = -(kappa + r) * np.exp(-(kappa + r) * T) * V * (1 - ncx2.cdf(x * K, df + 4, nc))
    term2 = np.exp(-(kappa + r) * T) * V * dq(df + 4)
    term3 = -r * theta * np.exp(-r * T) * (1 - ncx2.cdf(x * K, df + 2, nc)) \
            + theta * np.exp(-r * T) * dq(df + 2)
    term4 = (kappa + r) * theta * np.exp(-(kappa + r) * T) * (1 - ncx2.cdf(x * K, df + 2, nc)) \
            - theta * np.exp(-(kappa + r) * T) * dq(df + 2)
    term5 = r * np.exp(-r * T) * K * (1 - ncx2.cdf(x * K, df, nc)) - np.exp(-r * T) * K * dq(df)
    return term1 + term2 + term3 + term4 + term5


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
    plt.title("delta comparison")
    plt.xlabel('vol')
    plt.ylabel('delta')
    plt.plot(v2, delta2, label='analytical delta')
    plt.plot(v2, delta1, label='Finite Difference delta')
    plt.legend()

    plt.figure(2)
    plt.title("delta diff")
    plt.xlabel('vol')
    plt.ylabel('delta diff')
    plt.plot(v2, (delta2 - delta1))

    plt.figure(3)
    plt.title("gamma comparison")
    plt.xlabel('vol')
    plt.ylabel('gamma')
    plt.plot(v3, gamma1, label='analytical gamma')
    plt.plot(v3, gamma2, label='Finite Difference gamma')
    plt.legend()

    plt.figure(4)
    plt.title("gamma diff")
    plt.xlabel('vol')
    plt.ylabel('gamma diff')
    plt.plot(v3, (gamma1 - gamma2))

    # theta test
    t_vec1 = np.arange(0.1, 0.5, step_size)
    t_vec2 = np.arange(0.1 + step_size, 0.5 - step_size, step_size)
    V = 0.15
    p2 = srmr(kappa, theta, sigma, r, V, K, t_vec1)
    theta1 = de2t(kappa, theta, sigma, r, V, K, t_vec2)  # our formula
    theta2 = (p2[2:] - p2[:-2]) / step_size / 2
    plt.figure(5)
    plt.title("theta comparison")
    plt.xlabel('t')
    plt.ylabel('theta')
    plt.plot(t_vec2, theta1, label='analytical theta')
    plt.plot(t_vec2, theta2, label='Finite Difference theta')
    plt.legend()

    plt.figure(6)
    plt.title("theta diff")
    plt.xlabel('t')
    plt.ylabel('theta diff')

    plt.plot(t_vec2, (theta2 - theta1))
    plt.show()
