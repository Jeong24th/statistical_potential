"""
Figure SM6: Shell radii at V_total min vs |Psi_0| max for closed-shell N at phi=2.
Vectorized Slater determinant using precomputed 1D HO wavefunctions.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import hermite
from math import factorial
from multiprocessing import Pool, cpu_count
import time

MAX_WORKERS = max(1, int(cpu_count() * 0.7))
PHI = 2.0

# ==================== 2D HO states ====================

def get_2d_ho_states(N):
    states = []
    for E in range(200):
        for nx in range(E + 1):
            ny = E - nx
            states.append((nx, ny))
            if len(states) >= N:
                return states[:N]
    return states[:N]

# Precompute hermite polynomial coefficients
_hermite_cache = {}
def get_hermite(n):
    if n not in _hermite_cache:
        _hermite_cache[n] = hermite(n)
    return _hermite_cache[n]

_norm_cache = {}
def get_norm(n):
    if n not in _norm_cache:
        _norm_cache[n] = (np.pi**0.5 * 2**n * factorial(n))**(-0.5)
    return _norm_cache[n]

def ho_wf_array(n, x_arr):
    """Evaluate phi_n(x) for array of x values. Returns array."""
    H = get_hermite(n)
    norm = get_norm(n)
    return norm * H(x_arr) * np.exp(-0.5 * x_arr * x_arr)

def ho_wf_deriv_array(n, x_arr):
    """d/dx phi_n(x) = sqrt(n/2)*phi_{n-1}(x) - sqrt((n+1)/2)*phi_{n+1}(x)"""
    result = -np.sqrt((n + 1) / 2.0) * ho_wf_array(n + 1, x_arr)
    if n > 0:
        result += np.sqrt(n / 2.0) * ho_wf_array(n - 1, x_arr)
    return result

def build_slater_matrix(pos, states):
    """Build Slater matrix M[i,j] = phi_i(r_j), fully vectorized."""
    N = len(pos)
    xs = pos[:, 0]  # (N,)
    ys = pos[:, 1]  # (N,)
    M = np.zeros((N, N))
    for i, (nx, ny) in enumerate(states):
        M[i, :] = ho_wf_array(nx, xs) * ho_wf_array(ny, ys)
    return M

# ==================== V_total ====================

def compute_K(pos, sigma2):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.exp(-np.sum(diff**2, axis=2) / (2.0 * sigma2))

def vtotal_and_grad(x, N, wp2, beta_phi, sigma2):
    pos = x.reshape(N, 2)
    K = compute_K(pos, sigma2)
    sign, logdet = np.linalg.slogdet(K)
    if sign <= 0:
        return 1e10, np.zeros_like(x)
    val = 0.5 * wp2 * np.sum(pos * pos) - logdet / beta_phi
    Kinv = np.linalg.inv(K)
    W = Kinv * K
    diff = pos[None, :, :] - pos[:, None, :]
    c = 2.0 / (sigma2 * beta_phi)
    grad = wp2 * pos - c * np.einsum('ab,abd->ad', W, diff)
    return val, grad.flatten()

# ==================== |Psi_0| ====================

def psi_neg_logdet_and_grad(x, N, states):
    pos = x.reshape(N, 2)
    xs, ys = pos[:, 0], pos[:, 1]
    M = build_slater_matrix(pos, states)
    sign, logdet = np.linalg.slogdet(M)
    if sign == 0:
        return 1e10, np.zeros_like(x)
    val = -logdet

    Minv = np.linalg.inv(M)  # (N, N)
    grad = np.zeros((N, 2))
    # dM[i,j]/dx_j = dphi_nx(x_j)/dx * phi_ny(y_j)
    # dM[i,j]/dy_j = phi_nx(x_j) * dphi_ny(y_j)/dy
    # d(-logdet)/dx_j = -sum_i Minv[j,i] * dM[i,j]/dx_j
    for i, (nx, ny) in enumerate(states):
        dMdx_i = ho_wf_deriv_array(nx, xs) * ho_wf_array(ny, ys)  # (N,)
        dMdy_i = ho_wf_array(nx, xs) * ho_wf_deriv_array(ny, ys)  # (N,)
        # Minv[j, i] for all j
        grad[:, 0] -= Minv[:, i] * dMdx_i
        grad[:, 1] -= Minv[:, i] * dMdy_i

    return val, grad.flatten()

# ==================== Seeds ====================

def make_seeds(N, n_seeds=80):
    cfgs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2 * N))); r = 0.0
        for s in range(ms + 1):
            ns = min(s + 1, N - idx)
            if ns <= 0: break
            if s == 0: x0[idx] = [0, 0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn() * 0.03
                for k in range(ns):
                    a = 2*np.pi*k/ns + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        cfgs.append(x0.flatten())
    return cfgs

# ==================== Worker ====================

def worker(N):
    print(f"  Starting N={N}...", flush=True)
    t0 = time.time()

    beta_phi = np.sinh(PHI)
    omega_phi = 1.0 / np.cosh(PHI / 2.0)
    sigma2 = beta_phi
    wp2 = omega_phi ** 2
    states = get_2d_ho_states(N)

    cfgs = make_seeds(N, 80)

    # --- V_total min ---
    best_v, best_xv = 1e10, None
    for cfg in cfgs:
        try:
            f0, g0 = vtotal_and_grad(cfg, N, wp2, beta_phi, sigma2)
            res = minimize(lambda x: vtotal_and_grad(x, N, wp2, beta_phi, sigma2)[0],
                           cfg, method='L-BFGS-B',
                           jac=lambda x: vtotal_and_grad(x, N, wp2, beta_phi, sigma2)[1],
                           options={'maxiter': 30000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_v:
                best_v, best_xv = res.fun, res.x
        except:
            pass

    # --- |Psi_0| max ---
    best_p, best_xp = 1e10, None
    for cfg in cfgs:
        try:
            res = minimize(lambda x: psi_neg_logdet_and_grad(x, N, states)[0],
                           cfg, method='L-BFGS-B',
                           jac=lambda x: psi_neg_logdet_and_grad(x, N, states)[1],
                           options={'maxiter': 30000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_p:
                best_p, best_xp = res.fun, res.x
        except:
            pass

    dt = time.time() - t0
    print(f"  N={N} done in {dt:.1f}s", flush=True)
    return N, best_xv, best_xp

# ==================== Shell radii ====================

def extract_shells(pos, tol=0.15):
    radii = np.sort(np.sqrt(np.sum(pos**2, axis=1)))
    shells = []
    i = 0
    while i < len(radii):
        cnt = 1
        while i + cnt < len(radii) and abs(radii[i + cnt] - radii[i]) < tol:
            cnt += 1
        shells.append((cnt, np.mean(radii[i:i + cnt])))
        i += cnt
    return shells

# ==================== MAIN ====================

if __name__ == '__main__':
    closed_shell_N = [3, 6, 10, 15, 21, 28, 36, 45, 55]

    print(f"Fig SM6: N = {closed_shell_N}, phi = {PHI}")
    print(f"Using {MAX_WORKERS} processes")
    t0 = time.time()

    with Pool(processes=MAX_WORKERS) as pool:
        results = list(pool.imap_unordered(worker, closed_shell_N))

    results.sort(key=lambda t: t[0])
    print(f"\nTotal: {time.time() - t0:.1f}s\n")

    all_data = {}
    for N, xv, xp in results:
        pv = xv.reshape(N, 2) if xv is not None else None
        pp = xp.reshape(N, 2) if xp is not None else None
        sv = extract_shells(pv) if pv is not None else []
        sp = extract_shells(pp) if pp is not None else []
        all_data[N] = (sv, sp)

        print(f"N={N}:")
        print(f"  V_total shells: {[(n, f'{r:.3f}') for n, r in sv]}")
        print(f"  |Psi_0| shells: {[(n, f'{r:.3f}') for n, r in sp]}")
        rv = np.array([r for _, r in sv])
        rp = np.array([r for _, r in sp])
        if len(rv) == len(rp) and len(rv) > 0:
            rmsd = np.sqrt(np.mean((rv - rp)**2))
            print(f"  RMSD = {rmsd:.4f} a_0")
        print()

    # ==================== PLOT ====================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.size'] = 13

    fig, ax = plt.subplots(figsize=(11, 7))

    x_ticks = np.arange(len(closed_shell_N))
    offset = 0.12

    for idx, N in enumerate(closed_shell_N):
        sv, sp = all_data[N]
        for _, r in sv:
            ax.scatter(idx - offset, r, marker='s', s=90, c='royalblue', zorder=3,
                       edgecolors='navy', linewidths=0.5)
        for _, r in sp:
            ax.scatter(idx + offset, r, marker='^', s=90, c='tomato', zorder=3,
                       edgecolors='darkred', linewidths=0.5)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(n) for n in closed_shell_N])
    ax.set_xlabel(r'$N$', fontsize=15)
    ax.set_ylabel(r'Shell radius $/a_0$', fontsize=15)
    ax.set_ylim(-0.1, 3.8)
    ax.set_title(r'Shell radii at $\varphi = 2$', fontsize=15)

    ax.legend(handles=[
        Line2D([0], [0], marker='s', color='w', markerfacecolor='royalblue', markersize=10,
               markeredgecolor='navy', label=r'$V_{\mathrm{total}}$ min'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='tomato', markersize=10,
               markeredgecolor='darkred', label=r'$|\Psi_0|$ max'),
    ], fontsize=12, loc='upper left')

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1\fig_SM_shell_radii.pdf'
    plt.savefig(out, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {out}")
