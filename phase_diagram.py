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
from pathlib import Path
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

SCRIPT_DIR = Path(__file__).resolve().parent
MANUSCRIPT_DIR = SCRIPT_DIR.parents[1] / 'Nature_Comm'
OUTPUT_DIR = MANUSCRIPT_DIR / 'figures' if MANUSCRIPT_DIR.exists() else SCRIPT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Scan parameters ───────────────────────────────────────────
N_values = list(range(2, 56))
beta_values = [0.5, 4.0/7.0, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7,
               2.0, 2.3, 2.5, 2.7, 3.0]
N_SEEDS = 1000
MAX_WORKERS = max(1, int(cpu_count() * 0.7))

CACHE_FILE = SCRIPT_DIR / 'phase_diagram_cache.npz'

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
    fig.set_size_inches(6.0, 5.0)
    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.12, top=0.96)

    T_values = 1.0 / np.asarray(beta_values)
    order = np.argsort(T_values)
    T_values = T_values[order]
    ratio_T = ratio[:, order]
    T_min, T_max = 0.30, 2.10
    closed = [3, 6, 10, 15, 21, 28, 36, 45, 55]

    def crossover_temperature(row):
        """First low-T attraction-to-repulsion crossing on heating."""
        if row[0] <= 1.0:
            return np.nan
        repulsive = np.flatnonzero(row <= 1.0)
        if not len(repulsive):
            return T_values[-1]
        k = int(repulsive[0])
        t1, t2 = T_values[k - 1], T_values[k]
        r1, r2 = row[k - 1], row[k]
        return t1 + (1.0 - r1) * (t2 - t1) / (r2 - r1)

    Tc = np.asarray([crossover_temperature(row) for row in ratio_T])

    ax.set_facecolor('#e9f1fa')
    for N, tc in zip(N_values, Tc):
        if np.isfinite(tc):
            ax.fill_betweenx([N - 0.42, N + 0.42], T_min, tc,
                             color='#efb3b3', lw=0, zorder=2)

    # Draw T_c(N) only across consecutive particle numbers; N is discrete.
    finite = np.isfinite(Tc)
    start = None
    for i in range(len(N_values) + 1):
        if i < len(N_values) and finite[i]:
            if start is None:
                start = i
        elif start is not None:
            stop = i
            ax.plot(Tc[start:stop], np.asarray(N_values[start:stop]),
                    color='#8b0000', lw=1.2, zorder=4)
            start = None

    open_mask = finite & ~np.isin(N_values, closed)
    closed_mask = finite & np.isin(N_values, closed)
    ax.scatter(Tc[open_mask], np.asarray(N_values)[open_mask], s=20,
               color='#b30000', edgecolor='white', linewidth=0.35, zorder=5)
    ax.scatter(Tc[closed_mask], np.asarray(N_values)[closed_mask], s=34,
               marker='D', facecolor='white', edgecolor='#8b0000',
               linewidth=1.0, zorder=6)

    for Nc in closed:
        ax.axhline(Nc, color='0.35', ls=':', lw=0.55, alpha=0.55, zorder=1)
        ax.plot(T_max - 0.025, Nc, marker='D', ms=3.6,
                mfc='white', mec='0.25', mew=0.7, zorder=6)

    ax.text(0.43, 47.5, 'attraction-dominated', color='#8b0000',
            fontsize=9, rotation=90, ha='center', va='center')
    ax.text(1.50, 31, 'repulsion-dominated', color='#2255CC',
            fontsize=9, ha='center', va='center')
    ax.annotate(r'$T_{\rm c}(N)$', xy=(Tc[-1], N_values[-1]),
                xytext=(0.76, 53.0), fontsize=9, color='#8b0000',
                arrowprops=dict(arrowstyle='-', color='#8b0000', lw=0.8))

    ax.set_xlabel(r'$k_{\rm B}T/\hbar\omega$')
    ax.set_ylabel(r'$N$')
    ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    ax.set_ylim(1.5, 56)
    ax.set_xlim(T_min, T_max)

    leg = [
        Line2D([0], [0], color='#efb3b3', lw=7,
               label=r'$\max|F_{\rm att}|>\max|F_{\rm rep}|$'),
        Line2D([0], [0], color='#e9f1fa', lw=7,
               label=r'$\max|F_{\rm rep}|>\max|F_{\rm att}|$'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
               markeredgecolor='0.25', ms=5, label='closed shell'),
    ]
    ax.legend(handles=leg, fontsize=8, loc='upper right', framealpha=0.94)

    fig.savefig(OUTPUT_DIR / 'fig_phase_diagram.pdf', dpi=600, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'fig_phase_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved fig_phase_diagram.pdf / .png to {OUTPUT_DIR}")
    print("Done")
