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

def make_seeds(N, n_seeds=300):
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

    cfgs = make_seeds(N, 300)

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

def extract_shells(pos, gap_tol=0.25):
    """Cluster particles into shells by gap-based segmentation:
    sort radii, split where consecutive gap exceeds gap_tol.
    Robust to within-shell spread (unlike first-element threshold)."""
    radii = np.sort(np.sqrt(np.sum(pos**2, axis=1)))
    if len(radii) == 0:
        return []
    shells = []
    start = 0
    for i in range(1, len(radii)):
        if radii[i] - radii[i-1] > gap_tol:
            shells.append((i - start, float(np.mean(radii[start:i]))))
            start = i
    shells.append((len(radii) - start, float(np.mean(radii[start:]))))
    return shells

# ==================== MAIN ====================

if __name__ == '__main__':
    closed_shell_N = [3, 6, 10, 15, 21, 28, 36, 45, 55]

    import os
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(cache_dir, 'shell_radii_cache.npz')

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        d = np.load(cache_file)
        all_data = {}
        for N in closed_shell_N:
            sv_arr = d[f'sv_{N}']
            sp_arr = d[f'sp_{N}']
            sv = [(int(round(s[0])), float(s[1])) for s in sv_arr]
            sp = [(int(round(s[0])), float(s[1])) for s in sp_arr]
            all_data[N] = (sv, sp)
    else:
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

        cache_payload = {}
        for N in closed_shell_N:
            sv, sp = all_data[N]
            cache_payload[f'sv_{N}'] = np.array(sv) if sv else np.array([]).reshape(0, 2)
            cache_payload[f'sp_{N}'] = np.array(sp) if sp else np.array([]).reshape(0, 2)
        cache_payload['Ns'] = np.array(closed_shell_N)
        np.savez(cache_file, **cache_payload)
        print(f"Cached results to {cache_file}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 11

    # Two-panel: main scatter (top) + per-shell residual (bottom)
    fig, (ax, axR) = plt.subplots(2, 1, figsize=(9, 6.2),
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
                                  sharex=True)

    x_ticks = np.arange(len(closed_shell_N))
    offset = 0.06  # tighter pairing

    # Color-code each pair by shell index (inner -> dark, outer -> light)
    cmap = plt.get_cmap('plasma')
    max_shells = max(max(len(all_data[N][0]), len(all_data[N][1]))
                     for N in closed_shell_N)
    rmsd_per_N = []
    for idx, N in enumerate(closed_shell_N):
        sv, sp = all_data[N]
        rs_v = [r for _, r in sv]
        rs_p = [r for _, r in sp]
        # Plot pairs with thin connecting line per shell (inner-to-outer order)
        n_pairs = min(len(rs_v), len(rs_p))
        for k in range(n_pairs):
            c = cmap(k / max(max_shells - 1, 1) * 0.95)
            ax.plot([idx - offset, idx + offset], [rs_v[k], rs_p[k]],
                    color=c, lw=1.2, alpha=0.85, zorder=2)
            ax.scatter(idx - offset, rs_v[k], marker='s', s=85, c=[c],
                       edgecolors='black', linewidths=0.3, zorder=3)
            ax.scatter(idx + offset, rs_p[k], marker='^', s=85, c=[c],
                       edgecolors='black', linewidths=0.3, zorder=3)
        # Any leftover shells (mismatched count): plot uncoupled
        for k in range(n_pairs, len(rs_v)):
            c = cmap(k / max(max_shells - 1, 1) * 0.95)
            ax.scatter(idx - offset, rs_v[k], marker='s', s=85, c=[c],
                       edgecolors='black', linewidths=0.3, zorder=3)
        for k in range(n_pairs, len(rs_p)):
            c = cmap(k / max(max_shells - 1, 1) * 0.95)
            ax.scatter(idx + offset, rs_p[k], marker='^', s=85, c=[c],
                       edgecolors='black', linewidths=0.3, zorder=3)

        # RMSD over paired shells
        if n_pairs > 0:
            rv = np.array(rs_v[:n_pairs]); rp = np.array(rs_p[:n_pairs])
            rmsd = float(np.sqrt(np.mean((rv - rp)**2)))
        else:
            rmsd = float('nan')
        rmsd_per_N.append(rmsd)

    # Top panel cosmetics
    ax.set_ylabel(r'Shell radius $/a_0$')
    ax.set_ylim(-0.15, 3.9)
    ax.grid(True, axis='y', alpha=0.2)
    ax.legend(handles=[
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', label=r'$V_{\mathrm{total}}$ min'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8,
               markeredgecolor='black', label=r'$|\Psi_0|$ max'),
    ], fontsize=10, loc='upper left', framealpha=0.9)
    ax.set_title(r'Shell radii at $\varphi = 2$ (color: shell index, inner $\to$ outer)',
                 fontsize=11)

    # Bottom panel: per-N RMSD (paired shells)
    rmsd_arr = np.array(rmsd_per_N)
    bars = axR.bar(x_ticks, rmsd_arr, width=0.6,
                   color=['#9b9bd2' if not np.isnan(v) else 'lightgray' for v in rmsd_arr],
                   edgecolor='black', linewidth=0.5)
    for x, v in zip(x_ticks, rmsd_arr):
        if not np.isnan(v):
            axR.text(x, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    axR.set_ylabel(r'RMSD $/a_0$', fontsize=10)
    axR.set_xlabel(r'$N$ (closed shells)')
    axR.set_xticks(x_ticks)
    axR.set_xticklabels([str(n) for n in closed_shell_N])
    axR.set_ylim(0, max(0.025, np.nanmax(rmsd_arr) * 1.4))
    axR.grid(True, axis='y', alpha=0.2)

    plt.subplots_adjust(left=0.10, right=0.97, bottom=0.08, top=0.94)
    out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1_2\fig_SM_shell_radii.pdf'
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out}")
