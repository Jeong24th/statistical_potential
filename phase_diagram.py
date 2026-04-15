"""
Phase diagram: ATT vs REP dominance in (N, T) space.
For each (N, beta), minimize V_total and check whether the
globally strongest pairwise force is attractive or repulsive.
Now covers N=2..55 with analytic gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── Scan parameters ───────────────────────────────────────────
N_values = list(range(2, 56))
beta_values = [0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.5, 2.7, 3.0]

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

def analyze(N, beta):
    phi_p = beta
    beta_phi = np.sinh(phi_p) / phi_p * beta
    omega_phi = 1.0 / np.cosh(phi_p / 2.0)
    sigma2 = beta_phi

    obj = VtotalCached(N, beta_phi, omega_phi)

    best_f, best_x = np.inf, None
    n_seeds = 100
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2*N))); r = 0.0
        for s in range(ms+1):
            ni = s+1
            if idx+ni > N: ni = N-idx
            if ni <= 0: break
            if s == 0: x0[idx] = [0,0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn()*0.03
                for k in range(ni):
                    a = 2*np.pi*k/ni + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        res = minimize(obj.fun, x0.ravel(), jac=obj.jac, method='L-BFGS-B',
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

    ratio = fmax_att / fmax_rep if fmax_rep > 0 else 999
    return ratio

# ── Compute ───────────────────────────────────────────────────
results = np.zeros((len(N_values), len(beta_values)))

for i, N in enumerate(N_values):
    for j, beta in enumerate(beta_values):
        print(f"  N={N:3d}, beta={beta:.1f} ...", end='', flush=True)
        try:
            ratio = analyze(N, beta)
            results[i, j] = ratio
            tag = 'ATT' if ratio > 1 else 'REP'
            print(f"  {tag} ({ratio:.3f})")
        except Exception as e:
            results[i, j] = 0.0
            print(f"  FAIL ({e})")

# ═══════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════
print("\nPlotting ...", flush=True)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.10, top=0.95)

for i, N in enumerate(N_values):
    for j, beta in enumerate(beta_values):
        T = 1.0 / beta
        ratio = results[i, j]
        if ratio > 1:
            color = '#CC0000'
            marker = 's'
        else:
            color = '#2255CC'
            marker = 'o'
        size = 25 + 40 * min(abs(np.log10(max(ratio, 1e-3))), 2)
        ax.scatter(T, N, c=color, marker=marker, s=size,
                   edgecolors='k', linewidths=0.3, zorder=5)

# Mark closed shells
closed = [3, 6, 10, 15, 21, 28, 36, 45, 55]
for Nc in closed:
    ax.axhline(Nc, color='grey', ls=':', lw=0.4, alpha=0.4)

ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel(r'$N$')
ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
ax.set_ylim(1.5, 56)

from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='s', color='w', markerfacecolor='#CC0000',
              markeredgecolor='k', ms=8, label=r'$\max|F_{\rm att}| > \max|F_{\rm rep}|$'),
       Line2D([0],[0], marker='o', color='w', markerfacecolor='#2255CC',
              markeredgecolor='k', ms=8, label=r'$\max|F_{\rm rep}| > \max|F_{\rm att}|$')]
ax.legend(handles=leg, fontsize=8, loc='upper right', framealpha=0.9)

out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig.savefig(f'{out}\\fig_phase_diagram.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_phase_diagram.png', dpi=300, bbox_inches='tight')
print("Saved fig_phase_diagram.pdf / .png")
print("Done")
