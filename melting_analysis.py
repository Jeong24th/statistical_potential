"""
Melting analysis: force magnitude vs temperature.
For each beta, minimize V_total and compute pairwise statistical forces.
Track total force magnitude, attractive/repulsive decomposition.

Usage: python melting_force.py [<N>]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# ── V_total and forces ────────────────────────────────────────
def get_params(beta):
    phi = beta
    beta_phi = np.sinh(phi) / phi * beta
    omega_phi = 1.0 / np.cosh(phi / 2.0)
    sigma2 = beta_phi
    return beta_phi, omega_phi, sigma2

def V_total_func(pos_flat, omega_phi, sigma2, beta_phi):
    pos = pos_flat.reshape(N, 2)
    Vh = 0.5 * omega_phi**2 * np.sum(pos**2)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    Vs = -ld / beta_phi if s > 0 else 1e10
    return Vh + Vs

def grad_func(pos_flat, omega_phi, sigma2, beta_phi):
    eps = 1e-6
    f0 = V_total_func(pos_flat, omega_phi, sigma2, beta_phi)
    g = np.empty_like(pos_flat)
    for i in range(len(pos_flat)):
        vp = pos_flat.copy(); vp[i] += eps
        g[i] = (V_total_func(vp, omega_phi, sigma2, beta_phi) - f0) / eps
    return g

def find_minimum(beta):
    beta_phi, omega_phi, sigma2 = get_params(beta)
    obj = lambda v: V_total_func(v, omega_phi, sigma2, beta_phi)
    grd = lambda v: grad_func(v, omega_phi, sigma2, beta_phi)

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
        res = minimize(obj, x0.ravel(), jac=grd, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.reshape(N, 2)
    # Structured seeds for N > 10
    if N > 10:
        # G-type (one at origin)
        for seed in range(5):
            rng = np.random.RandomState(seed + 100)
            x0 = np.zeros((N, 2)); x0[0] = [0, 0]
            for i in range(1, N):
                angle = 2*np.pi*rng.rand() + seed*0.5
                r = 0.5 + 3.0*np.sqrt(i/N) + rng.randn()*0.08
                x0[i] = [r*np.cos(angle), r*np.sin(angle)]
            res = minimize(obj, x0.ravel(), jac=grd, method='L-BFGS-B',
                           options={'maxiter': 30000, 'ftol': 1e-15})
            if res.fun < best_f: best_f, best_x = res.fun, res.x.reshape(N, 2)
        # I-type (two near center)
        for seed in range(5):
            rng = np.random.RandomState(seed + 200)
            x0 = np.zeros((N, 2))
            ang0 = rng.rand()*2*np.pi
            x0[0] = [0.3*np.cos(ang0), 0.3*np.sin(ang0)]
            x0[1] = [-0.3*np.cos(ang0), -0.3*np.sin(ang0)]
            for i in range(2, N):
                angle = 2*np.pi*rng.rand()
                r = 0.6 + 2.8*np.sqrt((i-2)/(N-2)) + rng.randn()*0.08
                x0[i] = [r*np.cos(angle), r*np.sin(angle)]
            res = minimize(obj, x0.ravel(), jac=grd, method='L-BFGS-B',
                           options={'maxiter': 30000, 'ftol': 1e-15})
            if res.fun < best_f: best_f, best_x = res.fun, res.x.reshape(N, 2)
        # P-type (triangle at center)
        for seed in range(5):
            rng = np.random.RandomState(seed + 300)
            x0 = np.zeros((N, 2))
            for k in range(3):
                angle = 2*np.pi*k/3 + rng.randn()*0.1
                x0[k] = [0.43*np.cos(angle), 0.43*np.sin(angle)]
            for i in range(3, N):
                angle = 2*np.pi*rng.rand()
                r = 0.8 + 2.5*np.sqrt((i-3)/(N-3)) + rng.randn()*0.08
                x0[i] = [r*np.cos(angle), r*np.sin(angle)]
            res = minimize(obj, x0.ravel(), jac=grd, method='L-BFGS-B',
                           options={'maxiter': 30000, 'ftol': 1e-15})
            if res.fun < best_f: best_f, best_x = res.fun, res.x.reshape(N, 2)
    return best_x, best_f, beta_phi, omega_phi, sigma2

def compute_forces(pc, sigma2, beta_phi):
    d2 = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
    K = np.exp(-d2 / (2.0 * sigma2))
    Kinv = np.linalg.inv(K)

    F_att_total = 0.0
    F_rep_total = 0.0
    F_all_total = 0.0
    n_att, n_rep = 0, 0
    F_max = 0.0
    F_max_is_att = False
    F_net_radial = np.zeros(N)  # net radial force on each particle

    for a in range(N):
        for b in range(a+1, N):
            coeff = Kinv[a, b] * K[a, b]
            f = (2.0 / sigma2) * (pc[b]-pc[a]) * coeff / beta_phi
            mag = np.linalg.norm(f)
            dr = pc[b] - pc[a]
            dot = np.dot(f, dr / np.linalg.norm(dr))

            F_all_total += mag
            if mag > F_max:
                F_max = mag
                F_max_is_att = (dot > 0)

            if dot > 0:  # attractive
                F_att_total += mag; n_att += 1
            else:
                F_rep_total += mag; n_rep += 1

            # Net radial force on each particle
            r_a = np.linalg.norm(pc[a])
            r_b = np.linalg.norm(pc[b])
            if r_a > 0.01:
                rhat_a = pc[a] / r_a
                F_net_radial[a] += np.dot(-f, rhat_a)  # force on a from b
            if r_b > 0.01:
                rhat_b = pc[b] / r_b
                F_net_radial[b] += np.dot(f, rhat_b)   # force on b from a

    return {
        'F_att': F_att_total, 'F_rep': F_rep_total, 'F_total': F_all_total,
        'F_max': F_max, 'F_max_is_att': F_max_is_att,
        'n_att': n_att, 'n_rep': n_rep,
        'F_net_radial_mean': np.mean(np.abs(F_net_radial)),
        'F_net_radial_outer': np.mean(np.abs(F_net_radial[np.linalg.norm(pc, axis=1) > 0.5])),
    }

# ── Scan beta ─────────────────────────────────────────────────
betas = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 1.65, 1.7, 1.8, 2.0, 2.5, 3.0]
results = []

print(f"N={N}, scanning beta ...", flush=True)
for beta in betas:
    print(f"  beta={beta:.1f} ...", flush=True)
    try:
        pc, Vmin, beta_phi, omega_phi, sigma2 = find_minimum(beta)
        finfo = compute_forces(pc, sigma2, beta_phi)
        finfo['beta'] = beta
        finfo['T'] = 1.0 / beta
        finfo['Vmin'] = Vmin
        finfo['omega_phi'] = omega_phi
        finfo['shell_r'] = np.mean(np.linalg.norm(pc, axis=1))
        finfo['r_min'] = np.min(np.linalg.norm(pc, axis=1))
        results.append(finfo)
        print(f"    V={Vmin:.4f}, F_max={finfo['F_max']:.4f}, "
              f"att={finfo['n_att']}, rep={finfo['n_rep']}")
    except Exception as e:
        print(f"    FAILED: {e}")

# ── Print table ───────────────────────────────────────────────
print(f"\n{'beta':>5s} {'T':>5s} {'F_total':>9s} {'F_att':>9s} {'F_rep':>9s} "
      f"{'F_max':>8s} {'n_att':>5s} {'n_rep':>5s} {'F_att/F_rep':>11s}")
print('-'*75)
for r in results:
    ratio = r['F_att']/r['F_rep'] if r['F_rep'] > 0 else 0
    print(f"{r['beta']:5.1f} {r['T']:5.2f} {r['F_total']:9.4f} {r['F_att']:9.4f} "
          f"{r['F_rep']:9.4f} {r['F_max']:8.4f} {r['n_att']:5d} {r['n_rep']:5d} {ratio:11.4f}")

# ═══════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════
betas_arr = np.array([r['beta'] for r in results])
T_arr = 1.0 / betas_arr

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
plt.subplots_adjust(hspace=0.35, wspace=0.35)

# Table II temperatures (N=55 only)
table_II_T = [2.0, 1.0, 1.0/1.5, 0.5, 1.0/3.0] if N > 10 else []

def add_table_lines(ax):
    for T in table_II_T:
        ax.axvline(T, color='grey', ls=':', lw=0.7, alpha=0.5, zorder=0)

# Panel label helper
def panel_label(ax, label):
    ax.text(0.04, 0.94, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

# (a) Total, attractive, repulsive force vs T
ax = axes[0, 0]
ax.plot(T_arr, [r['F_total'] for r in results], 'ko-', ms=5, lw=1.8, label='Total')
ax.plot(T_arr, [r['F_att'] for r in results], 'rs-', ms=5, lw=1.5, label='Attractive')
ax.plot(T_arr, [r['F_rep'] for r in results], 'b^-', ms=5, lw=1.5, label='Repulsive')
ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel(r'$\sum |F_{b\to a}|$')
ax.legend(fontsize=9, framealpha=0.9)
ax.set_title(rf'$N={N}$', fontsize=12)
ax.set_yscale('log')
panel_label(ax, '(a)')
add_table_lines(ax)

# (b) r_min order parameter
ax = axes[0, 1]
rmin_vals = [r['r_min'] for r in results]
ax.plot(T_arr, rmin_vals, 'go-', ms=6, lw=1.8)
ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel(r'$r_{\min}\,/\,a_0$')
ax.set_ylim(bottom=0)
panel_label(ax, '(b)')
add_table_lines(ax)

# (c) Max force — colored by ATT/REP
ax = axes[1, 0]
fmax_vals = [r['F_max'] for r in results]
ax.plot(T_arr, fmax_vals, '-', color='grey', lw=1.0, zorder=1)
for i, r in enumerate(results):
    is_att = r.get('F_max_is_att', False)
    col = '#CC0000' if is_att else '#2255CC'
    marker = 's' if is_att else 'o'
    ax.plot(T_arr[i], fmax_vals[i], marker, color=col, ms=7,
            markeredgecolor='k', markeredgewidth=0.4, zorder=5)
ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel(r'$\max\,|F_{b\to a}|$')
from matplotlib.lines import Line2D
leg_max = [Line2D([0],[0], marker='s', color='w', markerfacecolor='#CC0000',
                  markeredgecolor='k', ms=6, label='Attractive'),
           Line2D([0],[0], marker='o', color='w', markerfacecolor='#2255CC',
                  markeredgecolor='k', ms=6, label='Repulsive')]
if N <= 10:
    ax.text(0.95, 0.90, 'all repulsive', transform=ax.transAxes,
            fontsize=9, ha='right', va='top', color='#2255CC', fontstyle='italic')
else:
    ax.legend(handles=leg_max, fontsize=7, loc='best', framealpha=0.9)
ax.set_yscale('log')
if N <= 10:
    ax.set_ylim(1e-1, 1e1)
panel_label(ax, '(c)')
add_table_lines(ax)

# (d) Number of attractive pairs
ax = axes[1, 1]
total_pairs = N*(N-1)//2
ax.plot(T_arr, [r['n_att'] for r in results], 'rs-', ms=5, lw=1.5, label='Attractive')
ax.plot(T_arr, [r['n_rep'] for r in results], 'b^-', ms=5, lw=1.5, label='Repulsive')
ax.axhline(total_pairs/2, color='grey', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax.set_ylabel('Number of pairs')
ax.legend(fontsize=9, framealpha=0.9)
panel_label(ax, '(d)')
add_table_lines(ax)

out = r'C:\Users\park\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig.savefig(f'{out}\\melting_force_N{N}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\melting_force_N{N}.png', dpi=300, bbox_inches='tight')
print(f"\nSaved melting_force_N{N}.pdf / .png")

# ── SM figure: Attraction/Repulsion ratio (moved from main Fig 3) ──
fig_sm, ax_sm = plt.subplots(1, 1, figsize=(4, 3))
ratios = [r['F_att']/r['F_rep'] if r['F_rep']>0 else 0 for r in results]
ax_sm.plot(T_arr, ratios, 'go-', ms=6, lw=1.8)
ax_sm.axhline(1.0, color='grey', ls='--', lw=0.8)
ax_sm.set_xlabel(r'$k_{\rm B}T\,/\,\hbar\omega$')
ax_sm.set_ylabel(r'$\sum|F_{\rm att}|\,/\,\sum|F_{\rm rep}|$')
ax_sm.set_title(rf'$N={N}$', fontsize=11)
ax_sm.set_ylim(bottom=0)
fig_sm.savefig(f'{out}\\fig_SM_ratio_N{N}.pdf', dpi=600, bbox_inches='tight')
print(f"Saved fig_SM_ratio_N{N}.pdf")

print("Done")
