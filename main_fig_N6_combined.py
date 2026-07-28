"""
Combined main-text Fig.1 for N=6 at beta=2 (phi=2): 2x2 grid

  (a) one-body density rho(r)         (b) conditional |Psi_0|^2 map
  (c) V_total minimum + pairwise      (d) dominant force per particle
      forces                              (strongest bond)

All four panels share the same Pauli-crystal positions (V_total minimum).
Output: fig_main_N6_combined.pdf (and .png) in the parent (manuscript) folder.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.optimize import minimize
from math import factorial

# ── Parameters (N=6, beta=2 = phi=2) ─────────────────────────
N = 6
beta = 2.0
hbar, m_p, omega = 1.0, 1.0, 1.0
phi_p = beta
beta_phi = np.sinh(phi_p) / phi_p * beta
omega_phi = 1.0 / np.cosh(phi_p / 2.0)
sigma2 = beta_phi

print(f"N={N}, phi={phi_p}, beta_phi={beta_phi:.4f}, omega_phi={omega_phi:.4f}")

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.55,
    'ytick.major.width': 0.55,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── Polar Slater basis ───────────────────────────────────────
def build_states(Nq):
    states = []; E = 1
    while len(states) < Nq:
        for n_r in range(E):
            am = E - 1 - 2 * n_r
            if am < 0: continue
            if am == 0: states.append((n_r, 0))
            else: states.append((n_r, am)); states.append((n_r, -am))
            if len(states) >= Nq: break
        E += 1
    return states[:Nq]

def polar_wf(n_r, m, x, y):
    r2 = x**2 + y**2; r = np.sqrt(r2); am = abs(m)
    th = np.arctan2(y, x)
    norm = np.sqrt(2.0*factorial(n_r)/factorial(n_r+am))/np.sqrt(np.pi)
    if m == 0: norm /= np.sqrt(2.0)
    L = genlaguerre(n_r, am)
    rad = r**am * L(r2) * np.exp(-r2/2.0)
    if m > 0: ang = np.cos(m*th)
    elif m < 0: ang = np.sin(am*th)
    else: ang = 1.0
    return norm * rad * ang

states = build_states(N)

def slater_mat(pos):
    S = np.empty((N, N))
    for i, (nr, m) in enumerate(states):
        S[i] = polar_wf(nr, m, pos[:, 0], pos[:, 1])
    return S

def log_det_slater(pos):
    s, ld = np.linalg.slogdet(slater_mat(pos))
    return ld if s != 0 else -1e30

# ── V_total + analytical gradient ────────────────────────────
def V_total(pos_flat):
    pos = pos_flat.reshape(N, 2)
    Vh = 0.5 * m_p * omega_phi**2 * np.sum(pos**2)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    return Vh + (-ld/beta_phi if s > 0 else 1e10)

class _Cached:
    def __init__(self):
        self.coeff = 2.0 / (sigma2 * beta_phi)
        self.inv_2s2 = 1.0 / (2.0 * sigma2)
        self._k = None; self._v = None
    def _c(self, v):
        k = v.tobytes()
        if self._k == k: return self._v
        pos = v.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff*diff, axis=2)
        K = np.exp(-self.inv_2s2 * d2)
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            self._v = (1e100, np.zeros_like(v)); self._k = k; return self._v
        Kinv = np.linalg.inv(K); S = Kinv * K
        val = 0.5*omega_phi**2*np.sum(pos*pos) - logdet/beta_phi
        grad = omega_phi**2*pos + self.coeff * np.einsum('ab,abj->aj', S, diff)
        self._v = (val, grad.ravel()); self._k = k; return self._v
    def fun(self, v): return self._c(v)[0]
    def jac(self, v): return self._c(v)[1]

obj = _Cached()

# ── Find V_total minimum (Pauli crystal) ─────────────────────
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
        if s == 0: x0[idx] = [0, 0]; idx += 1; r = 0.7
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
pc = best_x[np.argsort(np.linalg.norm(best_x, axis=1))]
print(f"V_total min = {best_f:.6f}")

# ── Pairwise forces at minimum ───────────────────────────────
print("Computing pairwise forces ...", flush=True)
d2_pc = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
K = np.exp(-d2_pc / (2.0 * sigma2))
Kinv = np.linalg.inv(K)
forces = {}
for a in range(N):
    for b in range(a+1, N):
        coeff = Kinv[a, b] * K[a, b]
        f = (2.0 / sigma2) * (pc[b]-pc[a]) * coeff / beta_phi
        dr = pc[b] - pc[a]
        dot = np.dot(f, dr / np.linalg.norm(dr))
        forces[(a, b)] = {'mag': np.linalg.norm(f), 'att': dot > 0}

# Strongest partner per particle
force_mag = np.zeros((N, N)); force_att = np.zeros((N, N), dtype=bool)
for (a, b), v in forces.items():
    force_mag[a, b] = v['mag']; force_mag[b, a] = v['mag']
    force_att[a, b] = v['att']; force_att[b, a] = v['att']
strongest_bonds = set()
for a in range(N):
    mg = force_mag[a].copy(); mg[a] = 0
    b = int(np.argmax(mg))
    strongest_bonds.add((min(a, b), max(a, b)))

# ── 1-body density rho(x,y) ──────────────────────────────────
gn = 200
gr = 3.2
xg = np.linspace(-gr, gr, gn); yg = np.linspace(-gr, gr, gn)
Xg, Yg = np.meshgrid(xg, yg)
print("Computing 1-body density ...", flush=True)
rho = np.zeros((gn, gn))
for n_r, m in states:
    rho += polar_wf(n_r, m, Xg, Yg)**2

# ── Conditional density: -ln|Psi_0|^2 with N-1 fixed at pc ───
print("Computing conditional |Psi_0|^2 ...", flush=True)
gnD = 150
xgD = np.linspace(-gr, gr, gnD); ygD = np.linspace(-gr, gr, gnD)
XgD, YgD = np.meshgrid(xgD, ygD)
vary = 0  # vary the central particle
Pg = np.empty((gnD, gnD))
for iy in range(gnD):
    for ix in range(gnD):
        p = pc.copy(); p[vary] = [xgD[ix], ygD[iy]]
        Pg[iy, ix] = 2 * log_det_slater(p)
neg_ln_P = -Pg
neg_ln_P -= np.min(neg_ln_P)

# ── V_total contour: vary one particle (consistent with current Fig.1) ──
print("Computing V_total contour ...", flush=True)
gnV = 150
xgV = np.linspace(-gr, gr, gnV); ygV = np.linspace(-gr, gr, gnV)
XgV, YgV = np.meshgrid(xgV, ygV)
def V_stat_only(pos):
    d2 = np.sum((pos[:, None, :]-pos[None, :, :])**2, axis=2)
    K2 = np.exp(-d2/(2.0*sigma2))
    s, ld = np.linalg.slogdet(K2)
    return -ld/beta_phi if s > 0 else 1e10
Vg = np.empty((gnV, gnV))
for iy in range(gnV):
    for ix in range(gnV):
        p = pc.copy(); p[vary] = [xgV[ix], ygV[iy]]
        Vh = 0.5*m_p*omega_phi**2*np.sum(p**2)
        Vg[iy, ix] = Vh + V_stat_only(p)
Vg -= np.min(Vg)

# ═══════════════════════════════════════════════════════════════
#  PLOT — 2x2 grid
# ═══════════════════════════════════════════════════════════════
print("Plotting ...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.97,
                    wspace=0.22, hspace=0.22)

lim = gr

# ── (a) one-body density ─────────────────────────────────────
ax = axes[0, 0]
lvl_rho = np.linspace(0, np.max(rho), 22)
ax.contourf(Xg, Yg, rho, levels=lvl_rho, cmap='YlOrRd', alpha=0.9)
ax.contour(Xg, Yg, rho, levels=lvl_rho[::3], colors='k', linewidths=0.2, alpha=0.3)
for a in range(N):
    ax.plot(pc[a, 0], pc[a, 1], '*', color='black', ms=7,
            markeredgecolor='white', markeredgewidth=0.4, zorder=6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$'); ax.set_ylabel(r'$y/a_0$')
ax.text(0.04, 0.96, r'(a) one-body density $\rho(\vec{r})$',
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

# ── (b) conditional density ──────────────────────────────────
ax = axes[0, 1]
vmaxP = np.percentile(neg_ln_P, 95)
lvl_P = np.linspace(0, vmaxP, 22)
ax.contourf(XgD, YgD, neg_ln_P, levels=lvl_P, cmap='RdYlBu_r',
            extend='max', alpha=0.85)
ax.contour(XgD, YgD, neg_ln_P, levels=lvl_P[::2], colors='k',
           linewidths=0.2, alpha=0.3)
for a in range(N):
    ax.plot(pc[a, 0], pc[a, 1], '*', color='black', ms=5.5,
            markeredgecolor='black', markeredgewidth=0.3, zorder=6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$'); ax.set_ylabel(r'$y/a_0$')
ax.text(0.04, 0.96, r'(b) conditional $-\ln|\Psi_0|^2$',
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

# ── (c) V_total + pairwise forces ────────────────────────────
ax = axes[1, 0]
vmaxV = np.percentile(Vg, 95)
lvl_V = np.linspace(0, vmaxV, 22)
ax.contourf(XgV, YgV, Vg, levels=lvl_V, cmap='RdYlBu_r',
            extend='max', alpha=0.85)
ax.contour(XgV, YgV, Vg, levels=lvl_V[::2], colors='k',
           linewidths=0.2, alpha=0.2)
fmax = max(v['mag'] for v in forces.values())
for (a, b), v in forces.items():
    col = '#CC0000' if v['att'] else '#2255CC'
    rel = np.sqrt(v['mag'] / fmax)
    lw = 0.4 + 2.0 * rel
    al = 0.3 + 0.6 * rel
    ax.plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]],
            color=col, lw=lw, alpha=al, zorder=3, solid_capstyle='round')
for a in range(N):
    ax.plot(pc[a, 0], pc[a, 1], '*', color='black', ms=5.5,
            markeredgecolor='black', markeredgewidth=0.3, zorder=6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$'); ax.set_ylabel(r'$y/a_0$')
ax.text(0.04, 0.96, r'(c) $V_{\rm total}$ minimum + pairwise forces',
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

# ── (d) strongest bond per particle ──────────────────────────
ax = axes[1, 1]
fmax_b = max(force_mag[a, b] for a, b in strongest_bonds)
for (a, b) in strongest_bonds:
    att = force_att[a, b]
    col = '#CC0000' if att else '#2255CC'
    rel = force_mag[a, b] / fmax_b
    lw = 1.0 + 2.5 * rel
    al = 0.6 + 0.35 * rel
    ax.plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]],
            color=col, lw=lw, alpha=al, zorder=3, solid_capstyle='round')
for a in range(N):
    ax.plot(pc[a, 0], pc[a, 1], '*', color='black', ms=6,
            markeredgecolor='black', markeredgewidth=0.3, zorder=6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$'); ax.set_ylabel(r'$y/a_0$')
ax.text(0.04, 0.96, r'(d) dominant force on each particle',
        transform=ax.transAxes, fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

# N label is conveyed through caption.

out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fig.savefig(os.path.join(out, 'fig_main_N6_combined.pdf'),
            dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(out, 'fig_main_N6_combined.png'),
            dpi=300, bbox_inches='tight')
print(f"Saved fig_main_N6_combined.pdf / .png to {out}")
print("Done")
