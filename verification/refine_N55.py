"""
Resolve the borderline N=55, phi=2 case: is the all-repulsive solution genuinely feasible?

Take the LP solution u (all components > 0), then Newton-refine it onto the affine set
A u = -F by minimal-norm least-squares corrections, and check that positivity survives
with machine-precision residuals.  Also re-examine with an unscaled uniform-margin LP.
"""

import numpy as np
from scipy.optimize import minimize, linprog

def params(beta):
    phi = beta
    return np.sinh(phi) / phi * beta, 1.0 / np.cosh(phi / 2.0)

def c_coeffs(pos, beta_phi):
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    K = np.exp(-d2 / (2.0 * beta_phi))
    Kinv = np.linalg.inv(K)
    C = (2.0 / beta_phi**2) * (Kinv * K)
    np.fill_diagonal(C, 0.0)
    return C

def pair_list(N):
    return [(a, b) for a in range(N) for b in range(a + 1, N)]

def constraint_matrix(pos):
    N = pos.shape[0]
    pairs = pair_list(N)
    A = np.zeros((2 * N, len(pairs)))
    for p, (a, b) in enumerate(pairs):
        d = pos[b] - pos[a]
        A[2 * a:2 * a + 2, p] += d
        A[2 * b:2 * b + 2, p] -= d
    return A, pairs

def make_vtotal(N, beta_phi, omega_phi):
    def fj(v):
        pos = v.reshape(N, 2)
        diffl = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diffl * diffl, axis=2)
        K = np.exp(-d2 / (2.0 * beta_phi))
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100, np.zeros_like(v)
        Kinv = np.linalg.inv(K)
        S = Kinv * K
        val = 0.5 * omega_phi**2 * np.sum(pos * pos) - logdet / beta_phi
        grad = omega_phi**2 * pos + (2.0 / beta_phi**2) * np.einsum('ab,abj->aj', S, diffl)
        return val, grad.ravel()
    return fj

def table_seed(seed):
    rng = np.random.RandomState(1000 + seed)
    shells = [(3, 0.429), (9, 1.123), (9, 1.742), (17, 2.436), (17, 3.332)]
    pts = []
    for n_s, r_s in shells:
        off = rng.uniform(0, 2 * np.pi)
        for k2 in range(n_s):
            ang = 2 * np.pi * k2 / n_s + off + rng.randn() * 0.02
            pts.append([(r_s + rng.randn() * 0.02) * np.cos(ang),
                        (r_s + rng.randn() * 0.02) * np.sin(ang)])
    return np.array(pts)


for N, beta in ((45, 2.0), (55, 2.0)):
    beta_phi, omega_phi = params(beta)
    fjac = make_vtotal(N, beta_phi, omega_phi)
    best = (np.inf, None)
    if N == 55:
        seeds = [table_seed(s) for s in range(8)]
    else:
        # reuse generic seeding
        def generic_seed(Nn, seed):
            rng = np.random.RandomState(seed)
            x0 = np.zeros((Nn, 2)); idx = 0; r = 0.0
            for s in range(int(np.ceil(np.sqrt(2 * Nn))) + 1):
                n_in = min(s + 1, Nn - idx)
                if n_in <= 0: break
                if s == 0:
                    x0[idx] = [0, 0]; idx += 1; r = 0.7
                else:
                    r += 0.55 + rng.randn() * 0.03
                    for k2 in range(n_in):
                        ang = 2 * np.pi * k2 / n_in + rng.randn() * 0.05 + seed * 0.3
                        x0[idx] = [r * np.cos(ang), r * np.sin(ang)]; idx += 1
                if idx >= Nn: break
            return x0
        seeds = [generic_seed(N, s) for s in range(40)]
    for x0 in seeds:
        res = minimize(lambda v: fjac(v)[0], x0.ravel(), jac=lambda v: fjac(v)[1],
                       method='L-BFGS-B', options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best[0]:
            best = (res.fun, res.x.reshape(N, 2))
    pos = best[1]
    print(f"\n===== N={N}, beta={beta}:  V_total={best[0]:.6f}")

    C = c_coeffs(pos, beta_phi)
    A, pairs = constraint_matrix(pos)
    P = len(pairs)
    c = np.array([C[a, b] for (a, b) in pairs])
    F = A @ c
    r_scale = np.abs(c) + 1e-3 * np.median(np.abs(c))

    # LP 1: scaled margin (as before)
    A_eq = np.hstack([A, (A @ r_scale)[:, None]])
    colnorm = np.linalg.norm(A_eq, axis=0); colnorm[colnorm == 0] = 1
    obj = np.zeros(P + 1); obj[-1] = -1.0 / colnorm[-1]
    res1 = linprog(obj, A_eq=A_eq / colnorm, b_eq=-F,
                   bounds=[(0, None)] * P + [(None, None)], method='highs')
    x1 = res1.x / colnorm
    u = x1[:P] + x1[-1] * r_scale
    print(f"LP(scaled margin): status={res1.status}, delta={x1[-1]:.3e}, "
          f"min u/r_scale={np.min(u / r_scale):.3e}")

    # Newton refinement onto A u = -F (minimal-norm correction), positivity watch
    for it in range(4):
        resid = A @ u + F
        rel = np.max(np.abs(resid)) / np.max(np.abs(F))
        print(f"   iter {it}: residual(rel)={rel:.2e}, min u={np.min(u):.3e}, "
              f"min u/r_scale={np.min(u / r_scale):.3e}, all>0: {bool(np.all(u > 0))}")
        if rel < 1e-14:
            break
        du, *_ = np.linalg.lstsq(A, -resid, rcond=None)
        u = u + du
    resid = A @ u + F
    print(f"   final: residual(rel)={np.max(np.abs(resid))/np.max(np.abs(F)):.2e}, "
          f"min u={np.min(u):.3e}, min u/r_scale={np.min(u/r_scale):.3e}, "
          f"ALL REPULSIVE: {bool(np.all(u > 0))}")

    # LP 2 cross-check: maximize uniform ABSOLUTE margin t (u >= t) with unscaled A
    A_eq2 = np.hstack([A, (A @ np.ones(P))[:, None]])
    obj2 = np.zeros(P + 1); obj2[-1] = -1.0
    res2 = linprog(obj2, A_eq=A_eq2, b_eq=-F,
                   bounds=[(0, None)] * P + [(None, None)], method='highs')
    if res2.status == 0:
        t = res2.x[-1]
        u2 = res2.x[:P] + t
        resid2 = np.max(np.abs(A @ u2 + F)) / np.max(np.abs(F))
        print(f"LP(absolute margin): t={t:.3e}, min u2={np.min(u2):.3e}, residual={resid2:.2e}")
    else:
        print(f"LP(absolute margin): status={res2.status}")
