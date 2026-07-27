"""
Rigorous (certificate-based) resolution of the all-repulsive-decomposition question.

Equality formulation.  A central Newton-third-law decomposition with coefficients c'
reproduces the statistical force field iff  A c' = F  (A assembles pair vectors into
total forces).  All-repulsive means c' < 0, i.e. u := -c' > 0 with  A u = -F.

  * Primal LP  (all-repulsive existence):   max delta  s.t.  A(delta*r + v) = -F, v >= 0.
      delta_opt > 0  ->  strictly all-repulsive decomposition exists (explicit u returned).
  * Dual certificate (non-existence):  find lambda in R^{2N} (a virtual displacement
    field xi_a) with
        (A^T lambda)_p = (x_b - x_a).(xi_a - xi_b)  <= 0   for every pair p=(a,b)
        lambda^T F  =  sum_a xi_a . F_a  < 0 .
    For any u >= 0 with A u = -F :  lambda^T F = -sum_p (A^T lambda)_p u_p >= 0,
    a contradiction -- so no all-repulsive (even weakly, u>=0) decomposition exists.
    Physical reading: an infinitesimal motion that does not decrease any pairwise
    distance, along which the statistical force field nevertheless does negative work.
    (Purely repulsive pair forces always do non-negative work under such a motion.)
    The raw LP certificate is repaired into a STRICTLY verifiable one by adding a small
    multiple of the dilation field xi_a = x_a, which is strictly interior to the cone:
    (x_b-x_a).(x_a-x_b) = -r_ab^2 < 0.

Scan: closed shells N = 3,6,10,15,21,28,36,45,55 at phi=2, plus N=55 at phi=1, 0.5.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linprog

rng_global = np.random.default_rng(7)


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


def generic_seed(N, seed):
    rng = np.random.RandomState(seed)
    x0 = np.zeros((N, 2)); idx = 0
    r = 0.0
    for s in range(int(np.ceil(np.sqrt(2 * N))) + 1):
        n_in = min(s + 1, N - idx)
        if n_in <= 0:
            break
        if s == 0:
            x0[idx] = [0, 0]; idx += 1; r = 0.7
        else:
            r += 0.55 + rng.randn() * 0.03
            for k2 in range(n_in):
                ang = 2 * np.pi * k2 / n_in + rng.randn() * 0.05 + seed * 0.3
                x0[idx] = [r * np.cos(ang), r * np.sin(ang)]; idx += 1
        if idx >= N:
            break
    return x0


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


def find_min(N, beta, n_generic, n_table=0):
    beta_phi, omega_phi = params(beta)
    fj = make_vtotal(N, beta_phi, omega_phi)
    best = (np.inf, None)
    seeds = [('g', s) for s in range(n_generic)] + [('t', s) for s in range(n_table)]
    for kind, s in seeds:
        x0 = generic_seed(N, s) if kind == 'g' else table_seed(s)
        res = minimize(lambda v: fj(v)[0], x0.ravel(), jac=lambda v: fj(v)[1],
                       method='L-BFGS-B', options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best[0]:
            best = (res.fun, res.x.reshape(N, 2))
    return best


def analyze(N, beta, n_generic=40, n_table=0):
    beta_phi, _ = params(beta)
    f, pos = find_min(N, beta, n_generic, n_table)
    C = c_coeffs(pos, beta_phi)
    A, pairs = constraint_matrix(pos)
    P = len(pairs)
    c = np.array([C[a, b] for (a, b) in pairs])
    F_flat = A @ c                                    # total statistical forces (2N,)
    n_att = int(np.sum(c > 0))
    diffnorm = np.array([np.linalg.norm(pos[b] - pos[a]) for (a, b) in pairs])
    mags = np.abs(c) * diffnorm
    kmax = int(np.argmax(mags))
    strongest = 'attractive' if c[kmax] > 0 else 'repulsive'

    out = {'N': N, 'beta': beta, 'V_total': float(f), 'pairs': P, 'n_att': n_att,
           'n_rep': P - n_att, 'strongest': strongest, 'Fmax': float(mags[kmax])}

    # ---------- primal: max delta  s.t.  A v + delta (A r) = -F,  v >= 0
    r_scale = np.abs(c) + 1e-3 * np.median(np.abs(c))
    A_eq = np.hstack([A, (A @ r_scale)[:, None]])
    # column scaling for conditioning
    colnorm = np.linalg.norm(A_eq, axis=0); colnorm[colnorm == 0] = 1.0
    A_eq_s = A_eq / colnorm
    obj = np.zeros(P + 1); obj[-1] = -1.0 * (1.0 / colnorm[-1])
    res = linprog(obj, A_eq=A_eq_s, b_eq=-F_flat,
                  bounds=[(0, None)] * P + [(None, None)], method='highs')
    delta_opt = None
    if res.status == 0:
        x = res.x / colnorm
        delta_opt = float(x[-1])
        out['delta_opt'] = delta_opt
        if delta_opt > 1e-9:
            u = x[:P] + delta_opt * r_scale
            resid = float(np.max(np.abs(A @ (-u) - F_flat)) / np.max(np.abs(F_flat)))
            out.update(verdict='all-repulsive EXISTS', u_min=float(np.min(u)),
                       primal_residual=resid)
            return out, pos
    else:
        out['delta_opt'] = f'LP status {res.status}'

    # ---------- dual certificate: min lambda.F  s.t.  A^T lambda <= 0, |lambda|<=1
    resc = linprog(F_flat / np.max(np.abs(F_flat)),
                   A_ub=A.T, b_ub=np.zeros(P),
                   bounds=[(-1, 1)] * (2 * len(pos)), method='highs')
    if resc.status != 0:
        out['verdict'] = f'certificate LP failed (status {resc.status})'
        return out, pos
    lam = resc.x
    # repair: add small multiple of the dilation field (strict interior of the cone)
    dil = pos.ravel()
    g = A.T @ lam
    viol = float(np.max(g))
    if viol > 0:
        theta = 2.0 * viol / np.min(diffnorm**2)
        lam = lam - theta * dil          # dilation gives (A^T dil)_p = +r_p^2 -> subtract
    else:
        theta = 0.0
    # NOTE: (A^T dil)_p = (x_b-x_a).(x_a-x_b) = -r_p^2  with our A convention; check sign numerically
    gd = A.T @ dil
    if np.max(gd) < 0:                    # dilation is interior with negative sign
        lam = resc.x + (2.0 * viol / np.min(-gd) if viol > 0 else 0.0) * dil
    g_fin = A.T @ lam
    work = float(lam @ F_flat)
    out.update(cert_max_pairdot=float(np.max(g_fin)), cert_work=work, cert_theta=theta)
    if np.max(g_fin) < 0 and work < 0:
        out['verdict'] = 'NO all-repulsive: DECOMPOSITION-INDEPENDENT attraction (verified certificate)'
    elif np.max(g_fin) <= 0 and work < 0:
        out['verdict'] = 'NO all-repulsive (certificate with zero-margin pairs)'
    else:
        out['verdict'] = 'undecided (certificate repair failed)'
    return out, pos


print(f"{'N':>3} {'beta':>5} {'att/rep':>10} {'strongest':>10} {'delta_opt':>12}  verdict")
print("-" * 100)
RES = []
cases = [(3, 2.0, 40, 0), (6, 2.0, 40, 0), (10, 2.0, 40, 0), (15, 2.0, 40, 0),
         (21, 2.0, 40, 0), (28, 2.0, 40, 0), (36, 2.0, 40, 0), (45, 2.0, 40, 0),
         (55, 2.0, 40, 10), (55, 1.0, 40, 0), (55, 0.5, 40, 0)]
for N, beta, ng, nt in cases:
    out, pos = analyze(N, beta, ng, nt)
    RES.append(out)
    d = out.get('delta_opt')
    dstr = f"{d:+.3e}" if isinstance(d, float) else str(d)
    print(f"{N:>3} {beta:>5.2f} {out['n_att']:>4}/{out['n_rep']:<5} {out['strongest']:>10} "
          f"{dstr:>12}  {out['verdict']}", flush=True)
    extra = []
    if 'primal_residual' in out:
        extra.append(f"u_min={out['u_min']:.2e}, residual={out['primal_residual']:.1e}")
    if 'cert_work' in out:
        extra.append(f"cert: max pair-dot={out['cert_max_pairdot']:+.3e}, work={out['cert_work']:+.3e}")
    if extra:
        print(f"{'':>34}  [{'; '.join(extra)}]", flush=True)

output_path = Path(__file__).resolve().parent / "certificate_results.json"
with output_path.open("w") as fh:
    json.dump(RES, fh, indent=1)
print(f"\n{output_path.name} written")
