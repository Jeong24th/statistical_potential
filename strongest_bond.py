"""
For each particle, find the partner exerting the strongest force on it.
Draw only these "strongest bonds", colored by attractive (red) / repulsive (blue).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 55
beta = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

phi_p = beta
beta_phi = np.sinh(phi_p) / phi_p * beta
omega_phi = 1.0 / np.cosh(phi_p / 2.0)
sigma2 = beta_phi

print(f"N={N}, beta={beta}, sigma2={sigma2:.4f}")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ── V_total minimization ──────────────────────────────────────
def V_total(pos_flat):
    pos = pos_flat.reshape(N, 2)
    Vh = 0.5 * omega_phi**2 * np.sum(pos**2)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    Vs = -ld / beta_phi if s > 0 else 1e10
    return Vh + Vs

class _VTotalCached:
    """Value + analytical gradient, cached so L-BFGS-B never recomputes."""
    def __init__(self):
        self._x = None
        self._val = None
        self._grad = None

    def _compute(self, pos_flat):
        pos = pos_flat.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        K = np.exp(-d2 / (2.0 * sigma2))
        sign, logdet = np.linalg.slogdet(K)
        Kinv = np.linalg.inv(K)
        self._val = 0.5 * omega_phi**2 * np.sum(pos * pos) - logdet / beta_phi
        self._grad = (omega_phi**2 * pos
                      + (2.0 / (sigma2 * beta_phi))
                      * np.einsum('ab,abj->aj', Kinv * K, diff)).ravel()
        self._x = pos_flat.copy()

    def val(self, pos_flat):
        if self._x is None or not np.array_equal(pos_flat, self._x):
            self._compute(pos_flat)
        return self._val

    def grad(self, pos_flat):
        if self._x is None or not np.array_equal(pos_flat, self._x):
            self._compute(pos_flat)
        return self._grad

_vtc = _VTotalCached()

def V_grad(v):
    return _vtc.grad(v)

print("Finding V_total minimum ...", flush=True)
best_f, best_x = np.inf, None
n_seeds = 300
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
    res = minimize(V_total, x0.ravel(), jac=V_grad, method='L-BFGS-B',
                   options={'maxiter': 30000, 'ftol': 1e-15})
    if res.fun < best_f:
        best_f, best_x = res.fun, res.x.reshape(N, 2)

pc = best_x[np.argsort(np.linalg.norm(best_x, axis=1))]
print(f"V_total min = {best_f:.4f}")

# ── Compute all pairwise forces ───────────────────────────────
d2_pc = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
K = np.exp(-d2_pc / (2.0 * sigma2))
Kinv = np.linalg.inv(K)

# Force magnitude and type for each pair
force_mag = np.zeros((N, N))
force_att = np.zeros((N, N), dtype=bool)

for a in range(N):
    for b in range(a+1, N):
        coeff = Kinv[a, b] * K[a, b]
        f = (2.0 / sigma2) * (pc[b]-pc[a]) * coeff / beta_phi
        mag = np.linalg.norm(f)
        dr = pc[b] - pc[a]
        dot = np.dot(f, dr / np.linalg.norm(dr))
        att = dot > 0
        force_mag[a, b] = mag
        force_mag[b, a] = mag
        force_att[a, b] = att
        force_att[b, a] = att

# For each particle, find the partner with strongest force
strongest_partner = np.zeros(N, dtype=int)
strongest_mag = np.zeros(N)
strongest_att = np.zeros(N, dtype=bool)

for a in range(N):
    mags = force_mag[a].copy()
    mags[a] = 0  # exclude self
    b = np.argmax(mags)
    strongest_partner[a] = b
    strongest_mag[a] = mags[b]
    strongest_att[a] = force_att[a, b]

# Unique bonds (avoid drawing a→b and b→a twice)
bonds = set()
for a in range(N):
    b = strongest_partner[a]
    bonds.add((min(a,b), max(a,b)))

n_att_bonds = sum(1 for a,b in bonds if force_att[a,b])
n_rep_bonds = sum(1 for a,b in bonds if not force_att[a,b])
print(f"\nStrongest bonds: {len(bonds)} unique ({n_att_bonds} attractive, {n_rep_bonds} repulsive)")

# Print summary
print("\nPer-particle strongest partner:")
for a in range(N):
    b = strongest_partner[a]
    tag = "ATT" if strongest_att[a] else "REP"
    dist = np.linalg.norm(pc[a] - pc[b])
    print(f"  {a:2d} -> {b:2d}  |F|={strongest_mag[a]:.2f}  {tag}  dist={dist:.3f}")

# ═══════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════
print("\nPlotting ...", flush=True)
fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5))
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95)

lim = 4.5 if N > 10 else 2.5
fmax_bond = max(force_mag[a, b] for a, b in bonds)

for (a, b) in bonds:
    att = force_att[a, b]
    col = '#CC0000' if att else '#2255CC'
    rel = force_mag[a, b] / fmax_bond
    lw = 1.0 + 2.5 * rel
    alpha = 0.6 + 0.35 * rel
    ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
            color=col, lw=lw, alpha=alpha, zorder=3, solid_capstyle='round')

for a in range(N):
    ax.plot(pc[a,0], pc[a,1], '*', color='black',
            ms=5 if N > 10 else 7,
            markeredgecolor='black', markeredgewidth=0.3, zorder=6)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$')
ax.set_ylabel(r'$y/a_0$')
ax.text(0.05, 0.95, rf'$N={N}$', transform=ax.transAxes,
        fontsize=11, va='top', ha='left', fontweight='bold')

from matplotlib.lines import Line2D
leg = [Line2D([0],[0], color='#CC0000', lw=2.5, label='Attractive'),
       Line2D([0],[0], color='#2255CC', lw=2.5, label='Repulsive')]
ax.legend(handles=leg, fontsize=9, loc='upper right', framealpha=0.9)

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
btag = f'beta{int(beta)}' if beta == int(beta) else f'beta{beta}'
fig.savefig(f'{out}\\fig_strongest_N{N}_{btag}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_strongest_N{N}_{btag}.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_strongest_N{N}_{btag}.pdf / .png")

# No-label version
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom=True, labelleft=True)
fig.savefig(f'{out}\\fig_strongest_N{N}_{btag}_nolabel.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_strongest_N{N}_{btag}_nolabel.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_strongest_N{N}_{btag}_nolabel.pdf / .png")
print("Done")
