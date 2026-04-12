"""
Pauli crystal with spin degeneracy g_s and Cooper pair identification.
Compares g_s=1 (spinless) vs g_s=2 (spin-1/2) for N=55 at phi=2.

V_total = (1/2) omega_phi^2 |X|^2 - ln(det K) / (g_s * beta_phi)

Cooper pair := mutual strongest bond that is attractive.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize

# ── Parameters ───────────────────────────────────────────────────
N     = 55
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

# ── Core functions ───────────────────────────────────────────────
def V_total(pos_flat, gs):
    pos = pos_flat.reshape(N, 2)
    Vh = 0.5 * omega_phi**2 * np.sum(pos**2)
    d2 = np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=2)
    K  = np.exp(-d2 / (2.0 * sigma2))
    s, ld = np.linalg.slogdet(K)
    Vs = -ld / (gs * beta_phi) if s > 0 else 1e10
    return Vh + Vs

def V_grad(pos_flat, gs):
    eps = 1e-6; f0 = V_total(pos_flat, gs)
    g = np.empty_like(pos_flat)
    for i in range(len(pos_flat)):
        vp = pos_flat.copy(); vp[i] += eps
        g[i] = (V_total(vp, gs) - f0) / eps
    return g


def find_minimum(gs, n_seeds=12):
    """Find V_total minimum using multi-seed L-BFGS-B."""
    best_f, best_x = np.inf, None
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2*N))); r = 0.0
        for s in range(ms+1):
            ni = s + 1
            if idx + ni > N: ni = N - idx
            if ni <= 0: break
            if s == 0:
                x0[idx] = [0, 0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn()*0.03
                for k in range(ni):
                    a = 2*np.pi*k/ni + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        res = minimize(lambda v: V_total(v, gs),
                       x0.ravel(),
                       jac=lambda v: V_grad(v, gs),
                       method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.reshape(N, 2)
        print(f"  seed {seed:2d}: V={res.fun:.4f}  "
              f"{'*' if res.fun <= best_f else ' '}", flush=True)
    return best_x[np.argsort(np.linalg.norm(best_x, axis=1))], best_f


def compute_forces(pc, gs):
    """Compute all pairwise statistical forces."""
    d2 = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
    K    = np.exp(-d2 / (2.0 * sigma2))
    Kinv = np.linalg.inv(K)

    fmag = np.zeros((N, N))
    fatt = np.zeros((N, N), dtype=bool)

    for a in range(N):
        for b in range(a+1, N):
            coeff = Kinv[a, b] * K[a, b]
            f   = (2.0 / sigma2) * (pc[b] - pc[a]) * coeff / (gs * beta_phi)
            mag = np.linalg.norm(f)
            dr  = pc[b] - pc[a]
            dot = np.dot(f, dr / np.linalg.norm(dr))
            att = dot > 0
            fmag[a, b] = fmag[b, a] = mag
            fatt[a, b] = fatt[b, a] = att
    return fmag, fatt


def find_strongest_bonds(fmag, fatt):
    """For each particle find the partner with the strongest force."""
    partner = np.zeros(N, dtype=int)
    smag    = np.zeros(N)
    satt    = np.zeros(N, dtype=bool)
    for a in range(N):
        m = fmag[a].copy(); m[a] = 0
        b = np.argmax(m)
        partner[a] = b; smag[a] = m[b]; satt[a] = fatt[a, b]
    bonds = set()
    for a in range(N):
        b = partner[a]
        bonds.add((min(a, b), max(a, b)))
    return partner, smag, satt, bonds


def find_cooper_pairs(partner, fatt):
    """Cooper pair = mutual strongest bond that is attractive."""
    pairs = []
    seen  = set()
    for a in range(N):
        b = partner[a]
        if partner[b] == a and fatt[a, b]:
            p = (min(a, b), max(a, b))
            if p not in seen:
                pairs.append(p)
                seen.add(p)
    return pairs


def shell_structure(pc):
    """Detect shell boundaries by radial gaps."""
    radii = np.linalg.norm(pc, axis=1)  # already sorted
    shells, cur = [], [0]
    for i in range(1, N):
        if radii[i] - radii[i-1] > 0.15:
            shells.append(cur); cur = []
        cur.append(i)
    shells.append(cur)
    info = [(len(s), np.mean(radii[s])) for s in shells]
    return shells, info


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════
results = {}

for gs in [1, 2]:
    label = "spinless" if gs == 1 else "spin-1/2"
    print(f"\n{'='*60}")
    print(f"  g_s = {gs}  ({label})")
    print(f"{'='*60}")

    pc, vmin = find_minimum(gs)
    print(f"\n  V_total min = {vmin:.6f}")

    fmag, fatt = compute_forces(pc, gs)
    partner, smag, satt, bonds = find_strongest_bonds(fmag, fatt)
    cooper = find_cooper_pairs(partner, fatt)

    shells, sinfo = shell_structure(pc)
    shell_str = "+".join(str(n) for n, _ in sinfo)
    radii_str = ", ".join(f"{r:.3f}" for _, r in sinfo)

    n_att = sum(1 for a, b in bonds if fatt[a, b])
    n_rep = sum(1 for a, b in bonds if not fatt[a, b])

    print(f"  Shell structure : {shell_str}")
    print(f"  Shell radii     : {radii_str}")
    print(f"  Strongest bonds : {len(bonds)} unique "
          f"({n_att} attractive, {n_rep} repulsive)")
    print(f"  Cooper pairs    : {len(cooper)}")
    if cooper:
        for a, b in cooper:
            ra = np.linalg.norm(pc[a])
            rb = np.linalg.norm(pc[b])
            dist = np.linalg.norm(pc[a] - pc[b])
            print(f"    ({a:2d},{b:2d})  r_a={ra:.3f}  r_b={rb:.3f}  "
                  f"dist={dist:.3f}  |F|={fmag[a,b]:.4f}")

    print(f"\n  Per-particle strongest partner:")
    for a in range(N):
        b = partner[a]
        tag = "ATT" if satt[a] else "REP"
        dist = np.linalg.norm(pc[a] - pc[b])
        print(f"    {a:2d} -> {b:2d}  |F|={smag[a]:.4f}  {tag}  dist={dist:.3f}")

    results[gs] = dict(pc=pc, vmin=vmin, fmag=fmag, fatt=fatt,
                       partner=partner, smag=smag, satt=satt,
                       bonds=bonds, cooper=cooper,
                       shells=shells, sinfo=sinfo,
                       n_att=n_att, n_rep=n_rep)


# ═══════════════════════════════════════════════════════════════
#  PLOT : side-by-side g_s=1 vs g_s=2
# ═══════════════════════════════════════════════════════════════
print("\n\nPlotting ...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
plt.subplots_adjust(wspace=0.12, left=0.06, right=0.97, bottom=0.11, top=0.90)

for idx, gs in enumerate([1, 2]):
    ax = axes[idx]
    R  = results[gs]
    pc, bonds = R['pc'], R['bonds']
    fmag, fatt = R['fmag'], R['fatt']
    cooper = R['cooper']
    n_att, n_rep = R['n_att'], R['n_rep']

    lim = 5.5
    fmax_bond = max(fmag[a, b] for a, b in bonds) if bonds else 1.0

    # Strongest bonds
    for (a, b) in bonds:
        att = fatt[a, b]
        col = '#CC0000' if att else '#2255CC'
        rel = fmag[a, b] / fmax_bond
        lw  = 1.0 + 2.5 * rel
        alp = 0.5 + 0.4 * rel
        ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
                color=col, lw=lw, alpha=alp, zorder=3,
                solid_capstyle='round')

    # Cooper pairs (thick green highlight)
    for (a, b) in cooper:
        ax.plot([pc[a,0], pc[b,0]], [pc[a,1], pc[b,1]],
                color='#00AA44', lw=4.0, alpha=0.7, zorder=4,
                solid_capstyle='round')

    # Particles
    for a in range(N):
        ax.plot(pc[a,0], pc[a,1], 'o', color='black', ms=3.5,
                markeredgecolor='black', markeredgewidth=0.3, zorder=6)

    # Labels
    label = r'$g_s=1$ (spinless)' if gs == 1 else r'$g_s=2$ (spin-$\frac{1}{2}$)'
    ax.set_title(label, fontsize=12)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$x\,/\,a_0$')
    if idx == 0:
        ax.set_ylabel(r'$y\,/\,a_0$')

    # Info text
    shell_str = "+".join(str(n) for n, _ in R['sinfo'])
    txt  = f"{n_att}a / {n_rep}r"
    txt2 = f"Cooper: {len(cooper)}"
    ax.text(0.03, 0.97, rf'$N={N}$, $\varphi={beta:.0f}$',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold')
    ax.text(0.03, 0.89, f'shells: {shell_str}',
            transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.03, 0.82, txt,
            transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.03, 0.75, txt2,
            transform=ax.transAxes, fontsize=8, va='top',
            color='#00AA44', fontweight='bold')

# Legend
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
