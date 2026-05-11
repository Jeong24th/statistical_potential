"""
Phase diagram: ATT vs REP dominance in (N, varphi) space.
For each (N, beta), minimize V_total and check whether the
globally strongest pairwise force is attractive or repulsive.

X-axis is varphi = beta*hbar*omega (so low T sits on the right);
this expands the melting-relevant region compared to the 1/T axis.

Caches results to phase_diagram_cache.npz so the heavy compute
only runs once.

Parallelized with multiprocessing.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── Scan parameters ───────────────────────────────────────────
N_values = list(range(2, 56))
beta_values = [0.5, 4.0/7.0, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7,
               2.0, 2.3, 2.5, 2.7, 3.0]
N_SEEDS = 1000
MAX_WORKERS = max(1, int(cpu_count() * 0.7))

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'phase_diagram_cache.npz')

# ── V_total + analytic gradient (per-process) ────────────────
class VtotalCached:
    def __init__(self, N, bp, wp):
        self.N = N; self.bp = bp; self.wp2 = wp * wp
        self.coeff = 2.0 / (bp * bp); self.inv_2s2 = 1.0 / (2.0 * bp)
        self._ck = None; self._cv = None
    def _c(self, v):
        k = v.tobytes()
        if self._ck == k: return self._cv
        N = self.N; pos = v.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        K = np.exp(-self.inv_2s2 * d2)
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            r = (1e100, np.zeros_like(v)); self._cv = r; self._ck = k; return r
        Kinv = np.linalg.inv(K); S = Kinv * K
        val = 0.5 * self.wp2 * np.sum(pos * pos) - logdet / self.bp
        grad = self.wp2 * pos + self.coeff * np.einsum('ab,abj->aj', S, diff)
        r = (val, grad.ravel()); self._cv = r; self._ck = k; return r
    def fun(self, v): return self._c(v)[0]
    def jac(self, v): return self._c(v)[1]

def _seeds(N, n_seeds):
    cfgs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2 * N))); r = 0.0
        for s in range(ms + 1):
            ni = s + 1
            if idx + ni > N: ni = N - idx
            if ni <= 0: break
            if s == 0: x0[idx] = [0, 0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn() * 0.03
                for k in range(ni):
                    a = 2*np.pi*k/ni + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        cfgs.append(x0.flatten())
    return cfgs

def analyze_pair(args):
    """Compute ratio = max|F_att| / max|F_rep| for a single (N, beta)."""
    N, beta = args
    phi_p = beta
    beta_phi = np.sinh(phi_p) / phi_p * beta
    omega_phi = 1.0 / np.cosh(phi_p / 2.0)
    sigma2 = beta_phi
    obj = VtotalCached(N, beta_phi, omega_phi)
    best_f, best_x = np.inf, None
    for cfg in _seeds(N, N_SEEDS):
        res = minimize(obj.fun, cfg, jac=obj.jac, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.reshape(N, 2)
    pc = best_x
    d2_pc = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
    K = np.exp(-d2_pc / (2.0 * sigma2))
    Kinv = np.linalg.inv(K)
    fmax_att, fmax_rep = 0.0, 0.0
    for a in range(N):
        for b in range(a+1, N):
            coeff = Kinv[a, b] * K[a, b]
            f = (2.0 / sigma2) * (pc[b]-pc[a]) * coeff / beta_phi
            mag = np.linalg.norm(f)
            dr = pc[b] - pc[a]
            dot = np.dot(f, dr / np.linalg.norm(dr))
            if dot > 0:
                fmax_att = max(fmax_att, mag)
            else:
                fmax_rep = max(fmax_rep, mag)
    ratio = fmax_att / fmax_rep if fmax_rep > 0 else 999.0
    return (N, beta, ratio)

# ── Compute (or load from cache) ──────────────────────────────
def compute_grid():
    cached = None
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE)
        cN = list(cached['N_values'])
        cB = list(cached['beta_values'])
        cS = int(cached['n_seeds']) if 'n_seeds' in cached.files else -1
        if cN == N_values and cB == beta_values and cS == N_SEEDS:
            return cached['ratio']
        print(f"Cache parameters differ (cached n_seeds={cS}, current={N_SEEDS}) - recomputing.", flush=True)

    pairs = [(N, b) for N in N_values for b in beta_values]
    print(f"Phase diagram: {len(pairs)} (N, beta) pairs, {N_SEEDS} seeds each", flush=True)
    print(f"Workers: {MAX_WORKERS}", flush=True)
    t0 = time.time()

    ratio = np.zeros((len(N_values), len(beta_values)))
    done = 0
    with Pool(processes=MAX_WORKERS) as pool:
        for N, beta, r in pool.imap_unordered(analyze_pair, pairs, chunksize=1):
            i = N_values.index(N); j = beta_values.index(beta)
            ratio[i, j] = r
            done += 1
            if done % 10 == 0 or done == len(pairs):
                elapsed = time.time() - t0
                rem = elapsed * (len(pairs) - done) / max(done, 1)
                print(f"  {done}/{len(pairs)}  elapsed={elapsed/60:.1f}m  ETA={rem/60:.1f}m",
                      flush=True)

    np.savez(CACHE_FILE,
             N_values=np.array(N_values),
             beta_values=np.array(beta_values),
             n_seeds=np.array(N_SEEDS),
             ratio=ratio)
    print(f"Saved cache to {CACHE_FILE}", flush=True)
    return ratio

if __name__ == '__main__':
    ratio = compute_grid()

    # ═══════════════════════════════════════════════════════════
    #  PLOT — x-axis is k_B T / (hbar*omega) = 1/varphi, so that
    #  lower temperatures are on the LEFT.
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.11, top=0.95)

    for i, N in enumerate(N_values):
        for j, beta in enumerate(beta_values):
            phi = beta  # varphi = beta * hbar * omega; we use natural units
            T_norm = 1.0 / phi  # k_B T / (hbar*omega)
            r = ratio[i, j]
            if r > 1:
                color = '#CC0000'; marker = 's'
            else:
                color = '#2255CC'; marker = 'o'
            size = 25 + 40 * min(abs(np.log10(max(r, 1e-3))), 2)
            ax.scatter(T_norm, N, c=color, marker=marker, s=size,
                       edgecolors='k', linewidths=0.3, zorder=5)

    closed = [3, 6, 10, 15, 21, 28, 36, 45, 55]
    for Nc in closed:
        ax.axhline(Nc, color='grey', ls=':', lw=0.4, alpha=0.4)

    ax.set_xlabel(r'$k_{\rm B}T/\hbar\omega$')
    ax.set_ylabel(r'$N$')
    ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    ax.set_ylim(1.5, 56)
    # 1/phi ranges over [1/3.0, 1/0.5] = [0.333, 2.0]; pad a bit.
    ax.set_xlim(0.30, 2.10)

    leg = [Line2D([0], [0], marker='s', color='w', markerfacecolor='#CC0000',
                  markeredgecolor='k', ms=8,
                  label=r'$\max|F_{\rm att}| > \max|F_{\rm rep}|$'),
           Line2D([0], [0], marker='o', color='w', markerfacecolor='#2255CC',
                  markeredgecolor='k', ms=8,
                  label=r'$\max|F_{\rm rep}| > \max|F_{\rm att}|$')]
    ax.legend(handles=leg, fontsize=8, loc='upper right', framealpha=0.9)

    out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1_2'
    fig.savefig(f'{out}\\fig_phase_diagram.pdf', dpi=600, bbox_inches='tight')
    fig.savefig(f'{out}\\fig_phase_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved fig_phase_diagram.pdf / .png")
    print("Done")
