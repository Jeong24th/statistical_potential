"""
FIG — V_total contour + pairwise force lines
Minimizes V_total(X) = (1/2) m omega_phi^2 X.X + V_stat(beta_phi, X)
for N fermions in 2D harmonic oscillator.

Usage: python generate_fig.py <beta> [<N>]
Default: beta=1, N=6
Units: hbar = m = omega = 1
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Parameters ────────────────────────────────────────────────
beta = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6

hbar, m_p, omega = 1.0, 1.0, 1.0
phi_param = omega * beta * hbar
beta_phi = np.sinh(phi_param) / phi_param * beta
omega_phi = omega / np.cosh(phi_param / 2.0)
sigma2 = beta_phi * hbar**2 / m_p  # kernel width for V_stat

print(f"N={N}, beta={beta}")
print(f"  phi={phi_param:.4f}, beta_phi={beta_phi:.4f}, omega_phi={omega_phi:.4f}, sigma2={sigma2:.4f}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── V_total ───────────────────────────────────────────────────
def V_total(pos_flat):
    pos = pos_flat.reshape(N, 2)
    V_harm = 0.5 * m_p * omega_phi**2 * np.sum(pos**2)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    V_stat = -ld / beta_phi if s > 0 else 1e10
    return V_harm + V_stat

def V_total_grad(pos_flat):
    eps = 1e-6
    f0 = V_total(pos_flat)
    g = np.empty_like(pos_flat)
    for i in range(len(pos_flat)):
        vp = pos_flat.copy()
        vp[i] += eps
        g[i] = (V_total(vp) - f0) / eps
    return g

# ── V_stat only (for conditional map) ─────────────────────────
def V_stat_at(pos):
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    return -ld / beta_phi if s > 0 else 1e10

# ── Find V_total minimum ─────────────────────────────────────
print("Finding V_total minimum ...", flush=True)

best_f, best_x = np.inf, None
n_seeds = 40 if N <= 10 else 12

for seed in range(n_seeds):
    rng = np.random.RandomState(seed)
    x0 = np.zeros((N, 2))
    idx = 0
    max_shell = int(np.ceil(np.sqrt(2*N)))
    r = 0.0
    for s in range(max_shell + 1):
        n_in_shell = s + 1
        if idx + n_in_shell > N:
            n_in_shell = N - idx
        if n_in_shell <= 0:
            break
        if s == 0:
            x0[idx] = [0, 0]
            idx += 1
            r = 0.7
        else:
            r += 0.55 + rng.randn() * 0.03
            for k in range(n_in_shell):
                angle = 2*np.pi*k/n_in_shell + rng.randn()*0.05 + seed*0.3
                x0[idx] = [r*np.cos(angle), r*np.sin(angle)]
                idx += 1
        if idx >= N:
            break

    # L-BFGS-B with numerical gradient
    res = minimize(V_total, x0.ravel(), jac=V_total_grad,
                   method='L-BFGS-B', options={'maxiter': 30000, 'ftol': 1e-15})
    if res.fun < best_f:
        best_f, best_x = res.fun, res.x.reshape(N, 2)
    if seed < 5 or res.fun <= best_f:
        print(f"  seed {seed}: V_total={res.fun:.6f} {'*' if res.fun <= best_f else ''}", flush=True)

pc = best_x.copy()
radii = np.linalg.norm(pc, axis=1)
order = np.argsort(radii)
pc = pc[order]

print(f"\nV_total minimum = {best_f:.6f}")
radii_sorted = np.linalg.norm(pc, axis=1)
print("Shell structure:")
shell_start = 0
for i in range(1, N):
    if radii_sorted[i] - radii_sorted[i-1] > 0.15:
        n_s = i - shell_start
        r_m = np.mean(radii_sorted[shell_start:i])
        print(f"  {n_s:2d} particles, r ~ {r_m:.3f}")
        shell_start = i
n_s = N - shell_start
r_m = np.mean(radii_sorted[shell_start:])
print(f"  {n_s:2d} particles, r ~ {r_m:.3f}")

# Decompose
V_h = 0.5 * m_p * omega_phi**2 * np.sum(pc**2)
V_s = best_f - V_h
print(f"  V_harm = {V_h:.6f}, V_stat = {V_s:.6f}")

# ── Pairwise statistical forces ───────────────────────────────
print("\nComputing forces ...", flush=True)
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
        forces[(a, b)] = {'f': f, 'mag': np.linalg.norm(f), 'attractive': dot > 0}

n_att = sum(1 for v in forces.values() if v['attractive'])
n_rep = sum(1 for v in forces.values() if not v['attractive'])
print(f"  {n_att} attractive, {n_rep} repulsive pairs (total {len(forces)})")

# ── Conditional V_total map (center particle) ─────────────────
ci = np.argmin(np.linalg.norm(pc, axis=1))
vary = ci
gn = 150 if N <= 10 else 120
gr = 3.2 if N <= 10 else 6.0
xg = np.linspace(-gr, gr, gn)
yg = np.linspace(-gr, gr, gn)
Xg, Yg = np.meshgrid(xg, yg)

print(f"Computing conditional V_total map (particle {vary}) ...", flush=True)
Vg = np.empty((gn, gn))
for iy in range(gn):
    if iy % 30 == 0:
        print(f"  row {iy}/{gn}", flush=True)
    for ix in range(gn):
        p = pc.copy()
        p[vary] = [xg[ix], yg[iy]]
        # V_total = V_harm + V_stat
        V_h_grid = 0.5 * m_p * omega_phi**2 * np.sum(p**2)
        V_s_grid = V_stat_at(p)
        Vg[iy, ix] = V_h_grid + V_s_grid
Vg -= np.min(Vg)

# ═══════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════
print("Plotting ...", flush=True)
if N <= 10:
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.95)
    ms_dot = 4
else:
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95)
    ms_dot = 2.5

lim = gr
vmax = np.percentile(Vg, 88)
lvl = np.linspace(0, vmax, 22)
cf = ax.contourf(Xg, Yg, Vg, levels=lvl, cmap='RdYlBu_r',
                 extend='max', alpha=0.45)
ax.contour(Xg, Yg, Vg, levels=lvl[::2], colors='k',
           linewidths=0.2, alpha=0.2)

# Force lines (log-scale width mapping)
fmax = max(v['mag'] for v in forces.values())
fmin = min(v['mag'] for v in forces.values())
if fmin <= 0 or fmin == fmax:
    fmin = fmax * 1e-6
log_range = np.log(fmax / fmin)
for (a, b), v in forces.items():
    col = '#CC0000' if v['attractive'] else '#2255CC'
    rel = np.log(max(v['mag'], fmin) / fmin) / log_range  # 0 to 1
    if N <= 10:
        lw = 0.6 + 2.0 * rel
    else:
        lw = 0.4 + 1.6 * rel
    alpha = 0.45 + 0.5 * rel
    ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
            color=col, lw=lw, alpha=alpha, zorder=3, solid_capstyle='round')

# Particles
for a in range(N):
    ax.plot(pc[a,0], pc[a,1], 'o', color='black', ms=ms_dot,
            markeredgecolor='black', markeredgewidth=0.4, zorder=6)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$')
ax.set_ylabel(r'$y/a_0$')
ax.text(0.05, 0.95, rf'$N={N}$', transform=ax.transAxes,
        fontsize=11, va='top', ha='left', fontweight='bold')

# ── Save ──
out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
btag = f'beta{int(beta)}' if beta == int(beta) else f'beta{beta}'
fig.savefig(f'{out}\\fig_N{N}_{btag}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_N{N}_{btag}.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_N{N}_{btag}.pdf / .png")

# No-label version
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom=True, labelleft=True)
fig.savefig(f'{out}\\fig_N{N}_{btag}_nolabel.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_N{N}_{btag}_nolabel.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_N{N}_{btag}_nolabel.pdf / .png")
print("Done")
