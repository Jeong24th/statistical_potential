"""
Verification of Supplementary Note 7 claims + LP test for decomposition-independent attraction.

Checks:
  A. Analytic canonical pairwise forces reproduce -grad V_stat (implementation validation).
  B. Dimension of the self-stress space (null central decompositions) for generic 2D configs
     equals (N-2)(N-3)/2  -> uniqueness for N=3, non-uniqueness for N>=4.
  C. Explicit N=4 counterexample: affine-dependence self-stress mu_a mu_b leaves every total
     force invariant while flipping the sign of an individual pair force.
  D. Reproduce the paper's V_total minima (phi=2) for N=6, 10, 55 (units hbar=m=omega=1),
     validated against Table 1 shell radii and bond counts.
  E. LP test at each minimum: does ANY central Newton-third-law decomposition of the same
     statistical force field make ALL pairs repulsive?  If infeasible, the existence of
     attraction is decomposition-independent.

Conventions follow PRL_PRE_v1/CODE (vtotal_minimize.py):
  units hbar=m=omega=1;  phi=beta;  beta_phi=sinh(phi)/phi*beta;  omega_phi=1/cosh(phi/2)
  K_ab=exp(-|x_a-x_b|^2/(2 beta_phi));  F_{b->a}=c_ab (x_b-x_a),  c_ab=(2/beta_phi^2) Kinv_ab K_ab
  attractive  <=>  c_ab > 0.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linprog

rng_global = np.random.default_rng(12345)
OUT = {}


# ----------------------------------------------------------------------
def params(beta):
    phi = beta
    beta_phi = np.sinh(phi) / phi * beta
    omega_phi = 1.0 / np.cosh(phi / 2.0)
    return beta_phi, omega_phi


def kernel(pos, beta_phi):
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * beta_phi)), d2


def V_stat(pos, beta_phi):
    K, _ = kernel(pos, beta_phi)
    s, ld = np.linalg.slogdet(K)
    return -ld / beta_phi if s > 0 else 1e100


def c_coeffs(pos, beta_phi):
    """Canonical coefficients c_ab (symmetric, zero diagonal): F_{b->a} = c_ab (x_b - x_a)."""
    K, _ = kernel(pos, beta_phi)
    Kinv = np.linalg.inv(K)
    C = (2.0 / beta_phi**2) * (Kinv * K)
    np.fill_diagonal(C, 0.0)
    return C


def F_stat_pairsum(pos, beta_phi):
    C = c_coeffs(pos, beta_phi)
    diff = pos[None, :, :] - pos[:, None, :]          # diff[a,b] = x_b - x_a
    return np.einsum('ab,abj->aj', C, diff)


def F_stat_numgrad(pos, beta_phi, h=1e-6):
    N = pos.shape[0]
    F = np.zeros_like(pos)
    for a in range(N):
        for j in range(2):
            pp = pos.copy(); pp[a, j] += h
            pm = pos.copy(); pm[a, j] -= h
            F[a, j] = -(V_stat(pp, beta_phi) - V_stat(pm, beta_phi)) / (2 * h)
    return F


# ----------------------------------------------------------------------
def pair_list(N):
    return [(a, b) for a in range(N) for b in range(a + 1, N)]


def constraint_matrix(pos):
    """A: (2N) x P map from symmetric pair coefficients {dc_p} to total-force perturbations.
    Null space of A = self-stress space = ambiguity of central decompositions."""
    N = pos.shape[0]
    pairs = pair_list(N)
    P = len(pairs)
    A = np.zeros((2 * N, P))
    for p, (a, b) in enumerate(pairs):
        d = pos[b] - pos[a]
        A[2 * a:2 * a + 2, p] += d
        A[2 * b:2 * b + 2, p] -= d
    return A, pairs


def nullspace(A, rtol=1e-10):
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = (s[0] if s.size else 1.0) * rtol
    rank = int(np.sum(s > tol))
    return Vt[rank:].T, rank, s


# ----------------------------------------------------------------------
#  A. implementation validation
# ----------------------------------------------------------------------
print("=" * 72)
print("A. Canonical pairwise sum vs numerical gradient of V_stat")
beta_phi2, omega_phi2 = params(2.0)
errsA = []
for N in (4, 6):
    pos = rng_global.normal(size=(N, 2)) * 1.2
    Fa = F_stat_pairsum(pos, beta_phi2)
    Fn = F_stat_numgrad(pos, beta_phi2)
    err = np.max(np.abs(Fa - Fn)) / max(1.0, np.max(np.abs(Fn)))
    errsA.append(err)
    print(f"   N={N}: rel. max deviation = {err:.2e}")
OUT['A_max_rel_err'] = float(max(errsA))
assert max(errsA) < 1e-6

# ----------------------------------------------------------------------
#  B. self-stress dimension for generic configurations
# ----------------------------------------------------------------------
print("=" * 72)
print("B. Self-stress dimension, generic 2D configurations vs (N-2)(N-3)/2")
OUT['B'] = {}
for N in range(3, 9):
    pos = rng_global.normal(size=(N, 2))
    A, _ = constraint_matrix(pos)
    W, rank, _ = nullspace(A)
    S_pred = (N - 2) * (N - 3) // 2
    print(f"   N={N}: nullity={W.shape[1]}  predicted={S_pred}  rank(A)={rank} (2N-3={2*N-3})")
    OUT['B'][N] = {'nullity': int(W.shape[1]), 'predicted': S_pred}
    assert W.shape[1] == S_pred

# ----------------------------------------------------------------------
#  C. explicit N=4 counterexample via affine dependence
# ----------------------------------------------------------------------
print("=" * 72)
print("C. N=4 explicit self-stress (mu_a mu_b) and sign flip of an individual bond")
pos4 = rng_global.normal(size=(4, 2)) * 1.1
M = np.vstack([pos4.T, np.ones(4)])                  # 3x4, null vector = affine dependence
_, _, Vt = np.linalg.svd(M)
mu = Vt[-1]
res_aff = np.max(np.abs(M @ mu))
print(f"   affine dependence residual |sum mu|,|sum mu x| = {res_aff:.2e}")

pairs4 = pair_list(4)
dc = np.array([mu[a] * mu[b] for (a, b) in pairs4])   # self-stress candidate
A4, _ = constraint_matrix(pos4)
res_null = np.max(np.abs(A4 @ dc))
print(f"   max |Sum_b dc_ab (x_b - x_a)| over particles = {res_null:.2e}  (must be ~0)")
assert res_null < 1e-12

C4 = c_coeffs(pos4, beta_phi2)
c4 = np.array([C4[a, b] for (a, b) in pairs4])
k = int(np.argmax(np.abs(dc)))
t = -2.0 * c4[k] / dc[k]                              # flips the sign of bond k
c4_mod = c4 + t * dc
diff = pos4[None, :, :] - pos4[:, None, :]
Cmod = np.zeros((4, 4))
for p, (a, b) in enumerate(pairs4):
    Cmod[a, b] = Cmod[b, a] = c4_mod[p]
F_orig = F_stat_pairsum(pos4, beta_phi2)
F_mod = np.einsum('ab,abj->aj', Cmod, diff)
dF = np.max(np.abs(F_orig - F_mod))
print(f"   bond {pairs4[k]}: c = {c4[k]:+.3e}  ->  c' = {c4_mod[k]:+.3e}  (sign flipped)")
print(f"   max |change of ANY total force F_a| = {dF:.2e}")
OUT['C'] = {'null_residual': float(res_null), 'total_force_change': float(dF),
            'c_before': float(c4[k]), 'c_after': float(c4_mod[k])}
assert dF < 1e-10

# ----------------------------------------------------------------------
#  D. reproduce paper minima (phi=2)
# ----------------------------------------------------------------------
print("=" * 72)
print("D. V_total minima at phi=2 (units hbar=m=omega=1)")


def make_vtotal(N, beta_phi, omega_phi):
    def fun_jac(v):
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
    return fun_jac


def generic_seed(N, seed):
    rng = np.random.RandomState(seed)
    x0 = np.zeros((N, 2)); idx = 0
    max_shell = int(np.ceil(np.sqrt(2 * N))); r = 0.0
    for s in range(max_shell + 1):
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
    """3+9+9+17+17 shells at Table-1 radii (N=55)."""
    rng = np.random.RandomState(1000 + seed)
    shells = [(3, 0.429), (9, 1.123), (9, 1.742), (17, 2.436), (17, 3.332)]
    pts = []
    for n_s, r_s in shells:
        off = rng.uniform(0, 2 * np.pi)
        for k2 in range(n_s):
            ang = 2 * np.pi * k2 / n_s + off + rng.randn() * 0.02
            rr = r_s + rng.randn() * 0.02
            pts.append([rr * np.cos(ang), rr * np.sin(ang)])
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


def shell_report(pos):
    r = np.sort(np.linalg.norm(pos, axis=1))
    shells, start = [], 0
    for i in range(1, len(r)):
        if r[i] - r[i - 1] > 0.15:
            shells.append((i - start, float(np.mean(r[start:i])))); start = i
    shells.append((len(r) - start, float(np.mean(r[start:]))))
    return shells


minima = {}
for N, n_gen, n_tab in ((6, 60, 0), (10, 60, 0), (55, 60, 20)):
    f, pos = find_min(N, 2.0, n_gen, n_tab)
    minima[N] = pos
    sh = shell_report(pos)
    print(f"   N={N}: V_total={f:.6f}  shells={[(n, round(rr, 3)) for n, rr in sh]}")
    OUT.setdefault('D', {})[N] = {'V_total': float(f), 'shells': sh}

# validation vs paper (Table 1, N=55)
sh55 = shell_report(minima[55])
table1 = [(3, 0.429), (9, 1.123), (9, 1.742), (17, 2.436), (17, 3.332)]
ok_struct = [n for n, _ in sh55] == [n for n, _ in table1]
rmsd = float(np.sqrt(np.mean([(sh55[i][1] - table1[i][1]) ** 2 for i in range(len(table1))]))) if ok_struct else None
print(f"   N=55 vs Table 1: structure match={ok_struct}, shell-radius RMSD={rmsd}")
OUT['D'][55]['table1_match'] = ok_struct
OUT['D'][55]['table1_rmsd'] = rmsd

# ----------------------------------------------------------------------
#  E. LP test: can ALL pairs be made repulsive by adding a self-stress?
# ----------------------------------------------------------------------
print("=" * 72)
print("E. LP: existence of an all-repulsive central decomposition (c' = c + W s < 0 ?)")
OUT['E'] = {}
beta_phi, omega_phi = params(2.0)
for N in (6, 10, 55):
    pos = minima[N]
    C = c_coeffs(pos, beta_phi)
    pairs = pair_list(N)
    c = np.array([C[a, b] for (a, b) in pairs])
    n_att = int(np.sum(c > 0)); n_rep = int(np.sum(c < 0))
    # strongest bond (paper convention: |F| = |c| * r_ab)
    diff = pos[None, :, :] - pos[:, None, :]
    mags = np.array([abs(C[a, b]) * np.linalg.norm(pos[b] - pos[a]) for (a, b) in pairs])
    kmax = int(np.argmax(mags))
    strongest = ('attractive' if c[kmax] > 0 else 'repulsive', float(mags[kmax]))

    A, _ = constraint_matrix(pos)
    W, rank, _ = nullspace(A)
    S = W.shape[1]

    # row scaling (positive diag): preserves sign feasibility
    r_scale = np.abs(c) + 1e-3 * np.median(np.abs(c))
    c_s = c / r_scale
    W_s = W / r_scale[:, None]

    # max-margin LP: minimize -delta  s.t.  W_s s + delta*1 <= -c_s
    P = len(pairs)
    A_ub = np.hstack([W_s, np.ones((P, 1))])
    obj = np.zeros(S + 1); obj[-1] = -1.0
    res = linprog(obj, A_ub=A_ub, b_ub=-c_s, bounds=[(None, None)] * (S + 1), method='highs')
    delta_opt = res.x[-1] if res.status == 0 else None

    # independent feasibility cross-check
    res2 = linprog(np.zeros(S), A_ub=W_s, b_ub=-c_s - 1e-9,
                   bounds=[(None, None)] * S, method='highs')
    feas = {0: 'feasible', 2: 'infeasible'}.get(res2.status, f'status{res2.status}')

    verdict = ('ALL-REPULSIVE DECOMPOSITION EXISTS (attraction scheme-dependent)'
               if (delta_opt is not None and delta_opt > 1e-9) or res2.status == 0
               else 'NO all-repulsive decomposition: attraction is DECOMPOSITION-INDEPENDENT')
    print(f"   N={N}: pairs={P}, self-stress dim={S} (rank A={rank}), canonical att/rep={n_att}/{n_rep}")
    print(f"          strongest bond: {strongest[0]}  |F|={strongest[1]:.3e}")
    print(f"          LP max-margin delta={delta_opt if delta_opt is None else f'{delta_opt:+.3e}'}"
          f"  | strict-feasibility: {feas}")
    print(f"          --> {verdict}")
    OUT['E'][N] = {'pairs': P, 'stress_dim': int(S), 'rankA': int(rank),
                   'n_att': n_att, 'n_rep': n_rep,
                   'strongest_type': strongest[0], 'strongest_mag': strongest[1],
                   'delta_opt': None if delta_opt is None else float(delta_opt),
                   'strict_feasibility': feas, 'verdict': verdict}

output_path = Path(__file__).resolve().parent / "results.json"
with output_path.open("w") as fh:
    json.dump(OUT, fh, indent=1)
print("=" * 72)
print(f"{output_path.name} written")
