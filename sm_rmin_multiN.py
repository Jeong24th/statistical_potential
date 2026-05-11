"""
SM: r_min order parameter vs temperature for multiple N values.
Shows structural transitions (G->I->P) across different particle numbers.
Optimized with analytic gradient and caching.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 11,
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

def get_params(beta):
    phi = beta
    bp = np.sinh(phi) / phi * beta
    wp = 1.0 / np.cosh(phi / 2.0)
    return bp, wp

class VtotalCached:
    """V_total with analytic gradient, cached to avoid recomputation."""
    def __init__(self, N, bp, wp):
        self.N = N
        self.bp = bp
        self.wp2 = wp * wp
        self.coeff = 2.0 / (bp * bp)
        self.inv_2s2 = 1.0 / (2.0 * bp)
        self._cache_v = None
        self._cache_key = None

    def _compute(self, v):
        key = v.tobytes()
        if self._cache_key == key:
            return self._cache_v
        N = self.N
        pos = v.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        K = np.exp(-self.inv_2s2 * d2)
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            self._cache_v = (1e100, np.zeros_like(v))
            self._cache_key = key
            return self._cache_v
        Kinv = np.linalg.inv(K)
        S = Kinv * K
        val = 0.5 * self.wp2 * np.sum(pos * pos) - logdet / self.bp
        grad_stat = self.coeff * np.einsum('ab,abj->aj', S, diff)
        grad = self.wp2 * pos + grad_stat
        self._cache_v = (val, grad.ravel())
        self._cache_key = key
        return self._cache_v

    def fun(self, v):
        return self._compute(v)[0]

    def jac(self, v):
        return self._compute(v)[1]

def find_min(N, bp, wp, n_seeds=300):
    obj = VtotalCached(N, bp, wp)
    best_f, best_x = np.inf, None
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2*N))); r = 0.0
        for si in range(ms + 1):
            ni = si + 1
            if idx + ni > N: ni = N - idx
            if ni <= 0: break
            if si == 0: x0[idx] = [0, 0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn() * 0.03
                for k in range(ni):
                    a = 2*np.pi*k/ni + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        res = minimize(obj.fun, x0.ravel(), jac=obj.jac,
                       method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.reshape(N, 2)
    return np.min(np.linalg.norm(best_x, axis=1))

# ── Scan ──
import os
N_values = [28, 36, 45, 55]
# Refined beta grid (denser near transitions in 0.6-1.7); ~24 points
betas = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
         1.00, 1.10, 1.20, 1.35, 1.50, 1.60, 1.65, 1.70, 1.80, 2.00,
         2.20, 2.50, 2.80, 3.00]
colors = {28: '#2255CC', 36: '#22AA44', 45: '#CC8800', 55: '#CC0000'}
markers = {28: 'o', 36: 's', 45: '^', 55: 'D'}

# Cache results so plotting can be re-run without recomputing
cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'rmin_multiN_cache.npz')

# Use 500 seeds (originally 300); enough to suppress most noise at competing-minima
# points. Cache file lets the user bump this and rerun individual N values cheaply.
N_SEEDS = 500

results = {}
if os.path.exists(cache_file):
    print(f"Loading cache from {cache_file}", flush=True)
    cached = np.load(cache_file, allow_pickle=True)
    cached_betas = cached['betas']
    if list(cached_betas) == list(betas):
        for N in N_values:
            key = f'N{N}'
            if key in cached.files:
                results[N] = cached[key]
        print("  (cache valid)", flush=True)
    else:
        print("  (cache stale: beta grid changed)", flush=True)

for N in N_values:
    if N in results:
        print(f"N={N}: cached, skipping compute", flush=True)
        continue
    print(f"N={N}, seeds={N_SEEDS} ...", flush=True)
    rmin_vals = []
    for beta in betas:
        bp, wp = get_params(beta)
        rm = find_min(N, bp, wp, n_seeds=N_SEEDS)
        rmin_vals.append(rm)
        print(f"  beta={beta:.3f}  r_min={rm:.4f}", flush=True)
    results[N] = np.array(rmin_vals)
    # Persist incrementally so a long crash leaves partial data
    np.savez(cache_file, betas=np.array(betas),
             **{f'N{n}': results[n] for n in results})

# ── Plot ──
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
plt.subplots_adjust(left=0.13, right=0.95, bottom=0.13, top=0.95)
T_vals = [1.0 / b for b in betas]
for N in N_values:
    if N not in results:
        continue
    ax.plot(T_vals, results[N], marker=markers[N], color=colors[N],
            ms=5, lw=1.4, label=rf'$N={N}$')
ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel(r'$r_{\min}\,/\,a_0$')
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=10, framealpha=0.9)

out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1_2'
fig.savefig(f'{out}\\fig_SM_rmin_multiN.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_SM_rmin_multiN.png', dpi=300, bbox_inches='tight')
print(f"\nSaved fig_SM_rmin_multiN.pdf / .png")
print("Done")
