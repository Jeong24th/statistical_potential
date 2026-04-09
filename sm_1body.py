"""
SM figure: 1-body density rho(r) for ground state of N fermions in 2D HO.
rho(r) = sum_i |phi_i(r)|^2 for occupied states.
Overlaid with Pauli crystal positions from V_total minimum.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from scipy.optimize import minimize
from math import factorial

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
beta = 2.0

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

# ── 1-body density: rho(x,y) = sum_i |phi_i(x,y)|^2 ─────────
gn = 200 if N <= 10 else 150
gr = 3.5 if N <= 10 else 6.5
xg = np.linspace(-gr, gr, gn)
yg = np.linspace(-gr, gr, gn)
Xg, Yg = np.meshgrid(xg, yg)

print(f"N={N}", flush=True)
print("Computing 1-body density ...", flush=True)
rho = np.zeros((gn, gn))
for n_r, m in states:
    rho += polar_wf(n_r, m, Xg, Yg)**2

# ── Find V_total minimum for crystal positions ────────────────
phi_p = beta
beta_phi = np.sinh(phi_p)/phi_p*beta
omega_phi = 1.0/np.cosh(phi_p/2.0)
sigma2 = beta_phi

def V_total_at(pos):
    Vh = 0.5*omega_phi**2*np.sum(pos**2)
    d2 = np.sum((pos[:,None,:]-pos[None,:,:])**2, axis=2)
    K = np.exp(-d2/(2.0*sigma2))
    s,ld = np.linalg.slogdet(K)
    return Vh+(-ld/beta_phi if s>0 else 1e10)

def Vt_flat(v): return V_total_at(v.reshape(N,2))
def Vt_grad(v):
    eps=1e-6; f0=Vt_flat(v); g=np.empty_like(v)
    for i in range(len(v)): vp=v.copy(); vp[i]+=eps; g[i]=(Vt_flat(vp)-f0)/eps
    return g

print("Finding V_total minimum ...", flush=True)
best_f, best_x = np.inf, None
n_seeds = 40 if N <= 10 else 12
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
    res = minimize(Vt_flat, x0.ravel(), jac=Vt_grad, method='L-BFGS-B',
                   options={'maxiter':30000,'ftol':1e-15})
    if res.fun < best_f: best_f, best_x = res.fun, res.x.reshape(N,2)

pc = best_x[np.argsort(np.linalg.norm(best_x, axis=1))]

# ═══════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════
print("Plotting ...", flush=True)
fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.5) if N > 10 else (3.6, 3.2))
if N > 10:
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95)
else:
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.95)

lim = gr

# Smooth 1-body density contour
lvl = np.linspace(0, np.max(rho)*0.95, 22)
ax.contourf(Xg, Yg, rho, levels=lvl, cmap='YlOrRd', extend='max', alpha=0.6)
ax.contour(Xg, Yg, rho, levels=lvl[::3], colors='k', linewidths=0.2, alpha=0.3)

# Crystal positions
for a in range(N):
    ax.plot(pc[a,0], pc[a,1], 'o', color='black',
            ms=3.5 if N > 10 else 5,
            markeredgecolor='white', markeredgewidth=0.6, zorder=6)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(r'$x/a_0$')
ax.set_ylabel(r'$y/a_0$')
ax.text(0.05, 0.95, rf'$N={N}$', transform=ax.transAxes,
        fontsize=11, va='top', ha='left', fontweight='bold')

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'
fig.savefig(f'{out}\\fig_SM_1body_N{N}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_SM_1body_N{N}.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_SM_1body_N{N}.pdf / .png")
print("Done")
