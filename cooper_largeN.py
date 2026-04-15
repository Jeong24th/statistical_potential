"""
Large-N Pauli crystal with spin: analytical gradient + Fibonacci spiral init.
Pushes N as high as possible with reliable convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
import time

# ── Parameters ───────────────────────────────────────────────────
N     = 120
beta  = 2.0
phi_p = beta
beta_phi  = np.sinh(phi_p) / phi_p * beta
omega_phi = 1.0 / np.cosh(phi_p / 2.0)
sigma2    = beta_phi

print(f"N={N}, beta={beta:.1f}, beta_phi={beta_phi:.4f}, "
      f"omega_phi={omega_phi:.4f}, sigma2={sigma2:.4f}")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
})

# ── Combined V_total + analytical gradient ───────────────────────
def V_and_grad(pos_flat, gs):
    pos = pos_flat.reshape(N, 2)
    Vh     = 0.5 * omega_phi**2 * np.sum(pos**2)
    grad_h = omega_phi**2 * pos

    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K  = np.exp(-d2 / (2.0 * sigma2))
    K += 1e-14 * np.eye(N)                              # tiny regularization

    s, ld = np.linalg.slogdet(K)
    if s <= 0:
        return 1e10, np.zeros(2*N)

    Vs = -ld / (gs * beta_phi)

    Kinv = np.linalg.inv(K)
    W    = Kinv * K
    np.fill_diagonal(W, 0.0)
    Wsum = W.sum(axis=1, keepdims=True)
    grad_s = (2.0 / (gs * beta_phi * sigma2)) * (Wsum * pos - W @ pos)

    return Vh + Vs, (grad_h + grad_s).ravel()


def make_init(seed, scale=1.0):
    """Fibonacci spiral scaled to estimated crystal radius."""
    rng = np.random.RandomState(seed)
    # Estimated max radius from N=55 data: R ~ 3.33 * (N/55)^{0.4}
    R_est = 3.5 * (N / 55.0)**0.4 * scale
    golden = np.pi * (3.0 - np.sqrt(5.0))               # golden angle
    x0 = np.zeros((N, 2))
    for i in range(N):
        r = R_est * np.sqrt((i + 0.5) / N)              # uniform area density
        theta = i * golden + rng.randn() * 0.08 + seed * 0.5
        x0[i] = [r * np.cos(theta), r * np.sin(theta)]
    return x0


def find_minimum(gs, n_seeds=300):
    best_f, best_x = np.inf, None
    t0 = time.time()
    for seed in range(n_seeds):
        # Try two initialization scales per seed
        for sc in [0.9, 1.1]:
            x0 = make_init(seed, scale=sc)
            res = minimize(lambda v: V_and_grad(v, gs),
                           x0.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': 50000, 'ftol': 1e-12,
                                    'gtol': 1e-7, 'maxfun': 100000})
            if res.fun < best_f:
                best_f, best_x = res.fun, res.x.reshape(N, 2)
            dt = time.time() - t0
            tag = '*' if res.fun <= best_f else ' '
            print(f"  seed {seed:2d} sc={sc:.1f}: V={res.fun:.4f}  "
                  f"nit={res.nit:5d}  {tag}  [{dt:.1f}s]", flush=True)
    return best_x[np.argsort(np.linalg.norm(best_x, axis=1))], best_f


def compute_forces(pc, gs):
    d2   = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
    K    = np.exp(-d2 / (2.0 * sigma2))
    K   += 1e-14 * np.eye(N)
    Kinv = np.linalg.inv(K)
    W    = Kinv * K
    np.fill_diagonal(W, 0.0)

    fmag = np.zeros((N, N))
    fatt = np.zeros((N, N), dtype=bool)
    coeff = 2.0 / (gs * beta_phi * sigma2)
    for a in range(N):
        for b in range(a+1, N):
            f   = coeff * (pc[b] - pc[a]) * W[a, b]
            mag = np.linalg.norm(f)
            dr  = pc[b] - pc[a]
            dot = np.dot(f, dr / np.linalg.norm(dr))
            fmag[a, b] = fmag[b, a] = mag
            fatt[a, b] = fatt[b, a] = (dot > 0)
    return fmag, fatt


def find_strongest_bonds(fmag, fatt):
    partner = np.zeros(N, dtype=int)
    smag    = np.zeros(N)
    satt    = np.zeros(N, dtype=bool)
    for a in range(N):
        m = fmag[a].copy(); m[a] = 0
        b = np.argmax(m)
        partner[a] = b; smag[a] = m[b]; satt[a] = fatt[a, b]
    bonds = set()
    for a in range(N):
        bonds.add((min(a, partner[a]), max(a, partner[a])))
    return partner, smag, satt, bonds


def find_cooper_pairs(partner, fatt):
    pairs, seen = [], set()
    for a in range(N):
        b = partner[a]
        if partner[b] == a and fatt[a, b]:
            p = (min(a,b), max(a,b))
            if p not in seen:
                pairs.append(p); seen.add(p)
    return pairs


def shell_structure(pc):
    radii = np.linalg.norm(pc, axis=1)
    gap   = 0.15 if N <= 100 else 0.12
    shells, cur = [], [0]
    for i in range(1, N):
        if radii[i] - radii[i-1] > gap:
            shells.append(cur); cur = []
        cur.append(i)
    shells.append(cur)
    return [(len(s), np.mean(radii[s])) for s in shells]


def antipodal_angle(pc, a, b):
    ra, rb = np.linalg.norm(pc[a]), np.linalg.norm(pc[b])
    if ra < 1e-10 or rb < 1e-10:
        return 0.0
    cos_th = np.clip(np.dot(pc[a], pc[b]) / (ra * rb), -1, 1)
    return np.arccos(cos_th)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
results = {}

for gs in [1, 2]:
    label = "spinless" if gs == 1 else "spin-1/2"
    print(f"\n{'='*65}")
    print(f"  g_s = {gs}  ({label})   N = {N}")
    print(f"{'='*65}")

    t0 = time.time()
    pc, vmin = find_minimum(gs)
    dt = time.time() - t0
    print(f"\n  V_total min = {vmin:.6f}  [total {dt:.1f}s]")

    print("  Computing forces ...", flush=True)
    fmag, fatt = compute_forces(pc, gs)
    partner, smag, satt, bonds = find_strongest_bonds(fmag, fatt)
    cooper = find_cooper_pairs(partner, fatt)
    sinfo = shell_structure(pc)

    shell_str = "+".join(str(n) for n, _ in sinfo)
    n_att = sum(1 for a, b in bonds if fatt[a, b])
    n_rep = sum(1 for a, b in bonds if not fatt[a, b])

    print(f"  Shell structure : {shell_str}")
    print(f"  Shell radii     : {', '.join(f'{r:.2f}' for _, r in sinfo)}")
    print(f"  Strongest bonds : {len(bonds)} unique "
          f"({n_att} attractive, {n_rep} repulsive)")
    print(f"  Cooper pairs    : {len(cooper)}")

    if cooper:
        print(f"\n  Cooper pair details:")
        for a, b in cooper:
            ra = np.linalg.norm(pc[a])
            rb = np.linalg.norm(pc[b])
            dist = np.linalg.norm(pc[a] - pc[b])
            theta = antipodal_angle(pc, a, b)
            print(f"    ({a:3d},{b:3d})  r_a={ra:.3f}  r_b={rb:.3f}  "
                  f"dist={dist:.3f}  theta/pi={theta/np.pi:.4f}  "
                  f"|F|={fmag[a,b]:.2e}")

    results[gs] = dict(pc=pc, vmin=vmin, fmag=fmag, fatt=fatt,
                       partner=partner, bonds=bonds, cooper=cooper,
                       sinfo=sinfo, n_att=n_att, n_rep=n_rep)


# ═══════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════
print("\n\nPlotting ...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.12, left=0.06, right=0.97, bottom=0.10, top=0.88)

for idx, gs in enumerate([1, 2]):
    ax = axes[idx]
    R  = results[gs]
    pc, bonds = R['pc'], R['bonds']
    fmag, fatt = R['fmag'], R['fatt']
    cooper = R['cooper']
    n_att, n_rep = R['n_att'], R['n_rep']

    rmax = np.max(np.linalg.norm(pc, axis=1))
    lim  = rmax * 1.25
    fmax_bond = max(fmag[a, b] for a, b in bonds) if bonds else 1.0

    for (a, b) in bonds:
        att = fatt[a, b]
        col = '#CC0000' if att else '#2255CC'
        rel = fmag[a, b] / fmax_bond
        lw  = 0.5 + 2.0 * rel
        alp = 0.4 + 0.5 * rel
        ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
                color=col, lw=lw, alpha=alp, zorder=3,
                solid_capstyle='round')

    for (a, b) in cooper:
        ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
                color='#00AA44', lw=4.0, alpha=0.8, zorder=4,
                solid_capstyle='round')

    ms = max(1.5, 4.5 - N/80)
    for a in range(N):
        ax.plot(pc[a,0], pc[a,1], 'o', color='black', ms=ms,
                markeredgewidth=0.2, zorder=6)

    ttl = (rf'$g_s=1$ (spinless)' if gs == 1
           else rf'$g_s=2$ (spin-$\frac{{1}}{{2}}$)')
    ax.set_title(ttl, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x\,/\,a_0$')
    if idx == 0:
        ax.set_ylabel(r'$y\,/\,a_0$')

    shell_str = "+".join(str(n) for n, _ in R['sinfo'])
    ax.text(0.03, 0.97, rf'$N={N}$, $\varphi={beta:.0f}$',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold')
    ax.text(0.03, 0.90, f'shells: {shell_str}',
            transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.03, 0.83, f'{n_att}a / {n_rep}r',
            transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.03, 0.76, f'Cooper: {len(cooper)}',
            transform=ax.transAxes, fontsize=9, va='top',
            color='#00AA44', fontweight='bold')
    if cooper:
        angles = [antipodal_angle(pc, a, b)/np.pi for a, b in cooper]
        ax.text(0.03, 0.69,
                rf'$\langle\theta\rangle/\pi={np.mean(angles):.3f}$',
                transform=ax.transAxes, fontsize=8, va='top', color='#00AA44')

leg = [Line2D([0],[0], color='#CC0000', lw=2.5, label='Attractive'),
       Line2D([0],[0], color='#2255CC', lw=2.5, label='Repulsive'),
       Line2D([0],[0], color='#00AA44', lw=3.5, label='Cooper pair')]
fig.legend(handles=leg, fontsize=9, loc='upper center', ncol=3,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.99))

out = (r'C:\Users\user\Dropbox\PROJECTS\STAT_Physics'
       r'\IDENTICAL_id\Statistical Potential\Manuscript\Pauli_v1')
fig.savefig(f'{out}\\fig_cooper_N{N}.pdf', dpi=600, bbox_inches='tight')
fig.savefig(f'{out}\\fig_cooper_N{N}.png', dpi=300, bbox_inches='tight')
print(f"Saved fig_cooper_N{N}.pdf / .png")
print("Done.")
