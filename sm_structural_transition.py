"""
SM: Structural phase transition of the V_total global minimum (C -> D -> T)
Generates fig_SM_rmin_order.pdf and fig_SM_struct_configs.pdf
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
        eps = 1e-6; f0 = Vt(v); g = np.empty_like(v)
        for i in range(len(v)):
            vp = v.copy(); vp[i] += eps; g[i] = (Vt(vp) - f0) / eps
        return g
    return Vt, Vg

def find_min(bp, wp, s2, n_seeds=25):
    Vt, Vg = make_Vt(bp, wp, s2)
    bf, bx = np.inf, None
    # Random seeds
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
    # C-type seeds (one at origin)
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
    # D-type seeds (two near center)
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
    # T-type seeds (triangle at center)
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
    return bf, bx

def classify(pc):
    radii = np.linalg.norm(pc, axis=1)
    sorted_r = np.sort(radii)
    r_min = sorted_r[0]
    if r_min < 0.05:
        return 'G', r_min
    elif sorted_r[2] - sorted_r[0] > 0.15:
        return 'I', r_min
    else:
        return 'P', r_min

# ===== Scan =====
snapshot_betas = [0.40, 0.70, 0.80, 1.00, 1.50, 2.00]
betas_scan = sorted(set(
    list(np.round(np.arange(0.10, 0.60, 0.05), 3)) +
    list(np.round(np.arange(0.60, 0.80, 0.02), 3)) +
    list(np.round(np.arange(0.80, 1.55, 0.05), 3)) +
    list(np.round(np.arange(1.55, 1.75, 0.02), 3)) +
    list(np.round(np.arange(1.75, 2.60, 0.10), 3)) +
    snapshot_betas  # ensure all snapshot betas are included
))

results = []
configs = {}

print(f"Scanning {len(betas_scan)} beta values for N={N}...")
for i, beta in enumerate(betas_scan):
    bp, wp, s2 = get_params(beta)
    vf, pc = find_min(bp, wp, s2, n_seeds=25)
    phase, r_min = classify(pc)
    results.append((beta, vf, r_min, phase))

    for sb in snapshot_betas:
        if abs(beta - sb) < 0.005:
            configs[sb] = pc.copy()

    if i % 5 == 0:
        print(f"  [{i+1}/{len(betas_scan)}] beta={beta:.2f}  r_min={r_min:.3f}  ({phase})", flush=True)

print("Scan complete.", flush=True)

# ===== Figure 1: r_min order parameter =====
out = r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1'

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
plt.subplots_adjust(left=0.14, right=0.95, bottom=0.15, top=0.93)

color_map = {'G': '#2255CC', 'I': '#228B22', 'P': '#CC0000'}
marker_map = {'G': 'o', 'I': 's', 'P': '^'}
labels_done = set()

for beta, vf, r_min, phase in results:
    lbl = None
    if phase not in labels_done:
        lbl = {'G': 'Gas-like (G)', 'I': 'Intermediate (I)', 'P': 'Pauli crystal (P)'}[phase]
        labels_done.add(phase)
    ax.plot(beta, r_min, marker_map[phase], color=color_map[phase], ms=5,
            label=lbl, zorder=5, markeredgecolor='k', markeredgewidth=0.3)

ax.set_xlabel(r'$\beta\hbar\omega$')
ax.set_ylabel(r'$r_{\min}\,/\,a_0$')
ax.legend(fontsize=9, loc='center right')
ax.set_xlim(0, 2.65)
ax.set_ylim(-0.02, 0.52)

ax.axvline(0.67, color='gray', ls='--', lw=0.6, alpha=0.5)
ax.axvline(1.63, color='gray', ls='--', lw=0.6, alpha=0.5)

fig.savefig(f'{out}\\fig_SM_rmin_order.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_SM_rmin_order.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_rmin_order.pdf/.png")
plt.close(fig)

# ===== Figure 2: Configuration snapshots =====
fig2, axes = plt.subplots(2, 3, figsize=(10, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.35)

for idx, beta in enumerate(snapshot_betas):
    ax = axes[idx // 3, idx % 3]
    pc = configs[beta]
    phase, r_min = classify(pc)

    # Find V_total for title
    vf = None
    for b, v, rm, ph in results:
        if abs(b - beta) < 0.001:
            vf = v; break

    ax.plot(pc[:, 0], pc[:, 1], 'ko', ms=3.5, markeredgewidth=0.3)
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')

    title = r'$\beta\hbar\omega=%.2f$  (%s)' % (beta, phase)
    if vf is not None:
        title += r'  $V=%.2f$' % vf
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$x/a_0$', fontsize=9)
    ax.set_ylabel(r'$y/a_0$', fontsize=9)

fig2.savefig(f'{out}\\fig_SM_struct_configs.pdf', dpi=600, bbox_inches='tight')
fig2.savefig(f'{out}\\fig_SM_struct_configs.png', dpi=300, bbox_inches='tight')
print("Saved fig_SM_struct_configs.pdf/.png")
plt.close(fig2)
print("All done.")
