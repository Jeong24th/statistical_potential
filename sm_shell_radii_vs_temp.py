"""
SM: Inner and outer shell radii vs temperature for N=55
Plots average radius of innermost shell and outermost shell as a function of phi.
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

N = 55
hbar, m_p, omega = 1.0, 1.0, 1.0

def get_params(beta):
    phi = omega * beta * hbar
    bp = np.sinh(phi) / phi * beta
    wp = omega / np.cosh(phi / 2.0)
    s2 = bp * hbar**2 / m_p
    return bp, wp, s2

def make_Vt(bp, wp, s2):
    def Vt(v):
        pos = v.reshape(N, 2)
        Vh = 0.5 * m_p * wp**2 * np.sum(pos**2)
        d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
        K = np.exp(-d2 / (2.0 * s2))
        s_, ld = np.linalg.slogdet(K)
        Vs = -ld / bp if s_ > 0 else 1e10
        return Vh + Vs
    def Vg(v):
        pos = v.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]          # (N, N, 2)
        d2 = np.sum(diff**2, axis=2)                       # (N, N)
        K = np.exp(-d2 / (2.0 * s2))
        Kinv = np.linalg.inv(K)
        g_stat = (2.0 / (s2 * bp)) * np.einsum('ab,abj->aj', Kinv * K, diff)
        g = m_p * wp**2 * pos + g_stat
        return g.ravel()
    return Vt, Vg

def find_min(bp, wp, s2, n_seeds=300):
    Vt, Vg = make_Vt(bp, wp, s2)
    bf, bx = np.inf, None
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2*N))); r = 0.0
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
        res = minimize(Vt, x0.ravel(), jac=Vg, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < bf: bf, bx = res.fun, res.x.reshape(N, 2)
    # Structured seeds
    for seed in range(5):
        rng = np.random.RandomState(seed + 100)
        x0 = np.zeros((N, 2)); x0[0] = [0, 0]
        for i in range(1, N):
            angle = 2*np.pi*rng.rand() + seed*0.5
            r = 0.5 + 3.0*np.sqrt(i/N) + rng.randn()*0.08
            x0[i] = [r*np.cos(angle), r*np.sin(angle)]
        res = minimize(Vt, x0.ravel(), jac=Vg, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < bf: bf, bx = res.fun, res.x.reshape(N, 2)
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
        res = minimize(Vt, x0.ravel(), jac=Vg, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < bf: bf, bx = res.fun, res.x.reshape(N, 2)
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
        res = minimize(Vt, x0.ravel(), jac=Vg, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < bf: bf, bx = res.fun, res.x.reshape(N, 2)
    return bx[np.argsort(np.linalg.norm(bx, axis=1))]

def get_shells(pc):
    r = np.linalg.norm(pc, axis=1)
    shells = []
    start = 0
    for i in range(1, N):
        if r[i] - r[i-1] > 0.15:
            shells.append((start, i))
            start = i
    shells.append((start, N))
    return shells

# ===== Scan =====
snapshot_betas = [0.40, 0.70, 0.80, 1.00, 1.50, 2.00]
betas_scan = sorted(set(
    list(np.round(np.arange(0.10, 0.60, 0.05), 3)) +
    list(np.round(np.arange(0.60, 0.80, 0.02), 3)) +
    list(np.round(np.arange(0.80, 1.55, 0.05), 3)) +
    list(np.round(np.arange(1.55, 1.75, 0.02), 3)) +
    list(np.round(np.arange(1.75, 2.60, 0.10), 3)) +
    snapshot_betas
))

results = []
print(f"Scanning {len(betas_scan)} beta values for N={N}...")
for i, beta in enumerate(betas_scan):
    bp, wp, s2 = get_params(beta)
    pc = find_min(bp, wp, s2, n_seeds=300)
    radii = np.linalg.norm(pc, axis=1)

    shells = get_shells(pc)
    # Innermost shell
    s0, e0 = shells[0]
    r_inner = np.mean(radii[s0:e0])
    # Outermost shell
    sL, eL = shells[-1]
    r_outer = np.mean(radii[sL:eL])

    results.append((beta, r_inner, r_outer, len(shells)))

    if i % 5 == 0:
        print(f"  [{i+1}/{len(betas_scan)}] beta={beta:.2f}  r_inner={r_inner:.3f}  r_outer={r_outer:.3f}  n_shells={len(shells)}", flush=True)

print("Scan complete.", flush=True)

# ===== Plot =====
betas_arr = np.array([r[0] for r in results])
r_inner_arr = np.array([r[1] for r in results])
r_outer_arr = np.array([r[2] for r in results])

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
plt.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.93)

ax.plot(betas_arr, r_inner_arr, 's-', color='#CC0000', ms=4, lw=1.0,
        markeredgecolor='k', markeredgewidth=0.3, label=r'Innermost shell')
ax.plot(betas_arr, r_outer_arr, 'o-', color='#2255CC', ms=4, lw=1.0,
        markeredgecolor='k', markeredgewidth=0.3, label=r'Outermost shell')

ax.set_xlabel(r'$\beta\hbar\omega$')
ax.set_ylabel(r'Shell radius $/\, a_0$')
ax.legend(fontsize=9)
ax.set_xlim(0, 2.65)

# Transition lines
ax.axvline(0.67, color='gray', ls='--', lw=0.6, alpha=0.5)
ax.axvline(1.63, color='gray', ls='--', lw=0.6, alpha=0.5)

out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1\OLD'
fig.savefig(f'{out}\\fig_SM_shell_radii_vs_temp.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_SM_shell_radii_vs_temp.png', dpi=300, bbox_inches='tight')
print(f"Saved to OLD/fig_SM_shell_radii_vs_temp.pdf/.png")
plt.close(fig)
print("Done.")
