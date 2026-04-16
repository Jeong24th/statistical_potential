"""
SM figure: |Psi_0|^2 conditional map only, in V_total contour style.
Same RdYlBu_r colormap with black dots at Pauli crystal positions.
Usage: python fig_SM_density.py [N]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.optimize import minimize
from math import factorial

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

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

# ── Polar basis ───────────────────────────────────────────────
def build_states(N):
    states = []; E = 1
    while len(states) < N:
        for n_r in range(E):
            abs_m = E-1-2*n_r
            if abs_m < 0: continue
            if abs_m == 0: states.append((n_r, 0))
            else: states.append((n_r, abs_m)); states.append((n_r, -abs_m))
            if len(states) >= N: break
        E += 1
    return states[:N]

def polar_wf(n_r, m, x, y):
    r2 = x**2+y**2; r = np.sqrt(r2); abs_m = abs(m)
    theta = np.arctan2(y, x)
    norm = np.sqrt(2.0*factorial(n_r)/factorial(n_r+abs_m))/np.sqrt(np.pi)
    if m == 0: norm /= np.sqrt(2.0)
    L = genlaguerre(n_r, abs_m)
    radial = r**abs_m * L(r2) * np.exp(-r2/2.0)
    if m > 0: angular = np.cos(m*theta)
    elif m < 0: angular = np.sin(abs_m*theta)
    else: angular = 1.0
    return norm * radial * angular

states = build_states(N)
def slater_mat(pos):
    S = np.empty((N, N))
    for i,(nr,m) in enumerate(states): S[i] = polar_wf(nr, m, pos[:,0], pos[:,1])
    return S
def log_det_slater(pos):
    s,ld = np.linalg.slogdet(slater_mat(pos)); return ld if s!=0 else -1e30

# ── Find Pauli crystal (max |Psi_0|) ─────────────────────────
beta = 2.0
phi_p = beta
beta_phi = np.sinh(phi_p) / phi_p * beta
sigma2 = beta_phi

print(f"N={N}", flush=True)
print("Finding Pauli crystal (max |Psi_0|) ...", flush=True)
def neg_ld(v): return -2*log_det_slater(v.reshape(N,2))
def neg_g(v):
    eps = 1e-7; f0 = neg_ld(v); g = np.empty_like(v)
    for i in range(len(v)):
        vp = v.copy(); vp[i] += eps; g[i] = (neg_ld(vp) - f0) / eps
    return g

best_f, best_x = np.inf, None
n_seeds = 300
for seed in range(n_seeds):
    rng = np.random.RandomState(seed)
    x0 = np.zeros((N,2)); idx=0
    ms=int(np.ceil(np.sqrt(2*N))); r=0.0
    for s in range(ms+1):
        ni=s+1
        if idx+ni>N: ni=N-idx
        if ni<=0: break
        if s==0: x0[idx]=[0,0]; idx+=1; r=0.7
        else:
            r+=0.55+rng.randn()*0.03
            for k in range(ni):
                a=2*np.pi*k/ni+rng.randn()*0.05+seed*0.3
                x0[idx]=[r*np.cos(a),r*np.sin(a)]; idx+=1
        if idx>=N: break
    res = minimize(neg_ld, x0.ravel(), jac=neg_g, method='L-BFGS-B',
                   options={'maxiter':30000,'ftol':1e-14})
    if res.fun < best_f: best_f, best_x = res.fun, res.x.reshape(N,2)

pc = best_x[np.argsort(np.linalg.norm(best_x, axis=1))]
print(f"Pauli crystal found")

# ── |Psi_0|^2 conditional map ─────────────────────────────────
vary = 0
gn = 150 if N <= 10 else 150
gr = 3.2 if N <= 10 else 6.0
xg = np.linspace(-gr, gr, gn)
yg = np.linspace(-gr, gr, gn)
Xg, Yg = np.meshgrid(xg, yg)

print("Computing |Psi_0|^2 conditional map ...", flush=True)
Pg = np.empty((gn, gn))
for iy in range(gn):
    for ix in range(gn):
        p = pc.copy(); p[vary] = [xg[ix], yg[iy]]
        Pg[iy,ix] = 2*log_det_slater(p)
Pn = np.exp(Pg - np.max(Pg))

# ═══════════════════════════════════════════════════════════════
#  PLOT — single panel, V_total style (RdYlBu_r, inverted)
# ═══════════════════════════════════════════════════════════════
print("Plotting ...", flush=True)
fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
plt.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.96)

lim = gr

# Use -ln|Psi_0|^2 as effective potential landscape (same structure as V_stat)
neg_ln_P = -Pg  # Pg = 2*log_det = ln|Psi_0|^2, so -Pg = -ln|Psi_0|^2
neg_ln_P -= np.min(neg_ln_P)
vmax = np.percentile(neg_ln_P, 88)
lvl = np.linspace(0, vmax, 22)
ax.contourf(Xg, Yg, neg_ln_P, levels=lvl, cmap='RdYlBu_r', extend='max', alpha=0.45)
ax.contour(Xg, Yg, neg_ln_P, levels=lvl[::2], colors='k', linewidths=0.25, alpha=0.35)

for a in range(N):
    ax.plot(pc[a,0], pc[a,1], '*', color='black',
            ms=3.5 if N > 10 else 5.5,
            markeredgecolor='black', markeredgewidth=0.3, zorder=6)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$')
ax.set_ylabel(r'$y/a_0$')
ax.text(0.05, 0.95, rf'$N={N}$', transform=ax.transAxes,
        fontsize=11, va='top', ha='left', fontweight='bold')

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig.savefig(f'{out}\\fig_SM_density_N{N}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_SM_density_N{N}.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_SM_density_N{N}.pdf / .png")
print("Done")
