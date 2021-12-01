from sympy import symbols, Derivative, Function, exp
from sympy import diff, factorial, lambdify
from srmr import srmr
from scipy.stats import ncx2
import pickle as pkl


def approx(N):
    t, v = symbols("t v")
    r, K = symbols("r K")
    kappa, theta, gamma = symbols("kappa theta gamma")
    sigma, sigma0 = symbols("sigma sigma0")  # nuisance parameter: sigma0

    # ncx2 parameters, sigma use nuisance param sigma0
    x = 4 * kappa / (sigma0 ** 2 * (1 - exp(-kappa * t)))
    df = 4 * kappa * theta / sigma0 ** 2
    nc = x * exp(-kappa * t) * v

    # ncx2 funcs
    p1 = Function('p1')  # nu+2
    p2 = Function('p2')  # nu+4
    p3 = Function('p3')  # nu+6
    # # ncx2 approx
    p1a, p2a, p3a = symbols('p1a, p2a, p3a')
    # x_ = x.subs([(t, T), (v, v0)])
    # nc_ = nc.subs([(t, T), (v, v0)])
    # p1_approx = ncx2.pdf(x_ * K, df + 2, nc_)
    # p2_approx = ncx2.pdf(x_ * K, df + 4, nc_)
    # p3_approx = ncx2.pdf(x_ * K, df + 6, nc_)
    # transformation if df=nu
    p0 = df / x / K * p2(v) + df / x / K * p1(v)
    p4 = x * K / nc * p2(v) - (df + 4) / nc * p3(v)
    # ncx2 diffs to t
    coef_t = -kappa * x * exp(-kappa * t) / 2 / (1 - exp(-kappa * t))
    p1t = coef_t * (-(K + v) * p1(v) + K * p0 + v * p2(v))
    p2t = coef_t * (-(K + v) * p2(v) + K * p1(v) + v * p3(v))
    p3t = coef_t * (-(K + v) * p3(v) + K * p2(v) + v * p4)
    # ncx2 diffs to v
    coef_v = x * exp(-kappa * t) / 2
    p1v = coef_v * (-p1(v) + p2(v))
    p2v = coef_v * (-p2(v) + p3(v))
    p3v = coef_v * (-p3(v) + p4)

    # first mis-pricing term: delta0
    Gamma = x * exp(-(2 * kappa + r) * t) * p2(v)
    delta0 = (sigma ** 2 * v ** (2 * gamma - 1) - sigma0 ** 2) / 2 * v * Gamma

    # infinitesimal generator
    def L(f):
        def dt(expr):
            expr = expr.subs([(p1(v), p1(t)), (p2(v), p2(t)), (p3(v), p3(t))])
            ft = diff(expr, t)
            ft = ft.subs(Derivative(p1(t), t), p1t)
            ft = ft.subs(Derivative(p2(t), t), p2t)
            ft = ft.subs(Derivative(p3(t), t), p3t)
            return ft

        def dv(expr):
            fv = diff(expr, v)
            fv = fv.subs(Derivative(p1(v), v), p1v)
            fv = fv.subs(Derivative(p2(v), v), p2v)
            fv = fv.subs(Derivative(p3(v), v), p3v)
            return fv

        ft = dt(f)
        fv = dv(f)
        fv2 = dv(fv)
        inf_gen = ft + kappa * (theta - v) * fv + sigma ** 2 / 2 * v ** (2 * gamma) * fv2
        return inf_gen - r * f

    # calculate mis-pricing terms
    ds = [delta0]
    for i in range(1, N + 1):
        tmp_ds = L(ds[i - 1])
        ds.append(tmp_ds)

    for i in range(len(ds)):
        ds[i] = ds[i].subs([(p1(v), p1a), (p2(v), p2a), (p3(v), p3a)])
        ds[i] = lambdify([v, t, r, kappa, theta, gamma, sigma, sigma0, p1a, p2a, p3a], ds[i])
    return ds


deltas = approx(4)
pkl.dump(deltas, open('./deltas.pkl', 'wb'))
