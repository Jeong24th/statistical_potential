"""
N=3 schematic: geometric origin of emergent attraction.

Two side-by-side panels illustrating Eq. (12)-(13) of the main text:
  Left:  third particle OUTSIDE the antipodal sphere of (x_1, x_2)
         => F_{2->1} repulsive (reduces to N=2 limit)
  Right: third particle INSIDE the antipodal sphere
         (r_{23}^2 + r_{31}^2 < r_{12}^2)
         => F_{2->1} switches sign, becomes attractive

The "antipodal sphere" is the circle with x_1, x_2 as diameter endpoints.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.13, top=0.92, wspace=0.18)

x1 = np.array([-1.0, 0.0])
x2 = np.array([+1.0, 0.0])
center = 0.5 * (x1 + x2)
radius = 0.5 * np.linalg.norm(x2 - x1)
arrow_kw = dict(arrowstyle='-|>', mutation_scale=14, lw=1.6, zorder=5)

def panel(ax, x3, attractive, title):
    # Antipodal sphere (circle on diameter x1-x2)
    circ_fill = Circle(center, radius, fill=True, fc='#FFE8E8',
                       ec='none', alpha=0.35, zorder=1)
    ax.add_patch(circ_fill)
    circ_edge = Circle(center, radius, fill=False, ec='#CC4444',
                       ls='--', lw=1.2, alpha=0.85, zorder=2)
    ax.add_patch(circ_edge)
    # Diameter line
    ax.plot([x1[0], x2[0]], [x1[1], x2[1]], color='gray',
            ls=':', lw=0.8, zorder=2)
    # Particles 1, 2
    ax.plot(*x1, 'o', ms=10, color='#1f4e8c',
            markeredgecolor='black', markeredgewidth=0.6, zorder=6)
    ax.plot(*x2, 'o', ms=10, color='#1f4e8c',
            markeredgecolor='black', markeredgewidth=0.6, zorder=6)
    ax.text(x1[0]-0.05, x1[1]-0.45, r'$\vec{x}_1$', fontsize=11,
            ha='center', va='top')
    ax.text(x2[0]+0.05, x2[1]-0.45, r'$\vec{x}_2$', fontsize=11,
            ha='center', va='top')
    # Particle 3
    ax.plot(*x3, 'o', ms=10, color='#1f4e8c',
            markeredgecolor='black', markeredgewidth=0.6, zorder=6)
    ax.text(x3[0]+0.18, x3[1]+0.18, r'$\vec{x}_3$', fontsize=11,
            ha='left', va='bottom')
    # Distance lines from x3
    for xt in (x1, x2):
        ax.plot([x3[0], xt[0]], [x3[1], xt[1]],
                color='gray', ls='-', lw=0.5, alpha=0.6, zorder=2)
    # Force on particle 1 from 2 (along x1->x2 direction or reverse)
    direction = (x2 - x1) / np.linalg.norm(x2 - x1)
    if attractive:
        # F_{2->1} pulls 1 toward 2 (rightward)
        start = x1 + 0.18 * direction
        end = start + 0.55 * direction
        col = '#CC0000'
        # Place label outside the sphere (above) so it does not occlude x_3
        flabel_xy = (start[0]-0.05, 1.20)
        flabel_text = r'$\vec{F}_{2\to 1}$ (attractive)'
    else:
        # F_{2->1} pushes 1 away from 2 (leftward)
        start = x1 - 0.18 * direction
        end = start - 0.55 * direction
        col = '#1f4e8c'
        flabel_xy = (start[0]-0.40, start[1]+0.30)
        flabel_text = r'$\vec{F}_{2\to 1}$ (repulsive)'
    arrow = FancyArrowPatch(start, end, color=col, **arrow_kw)
    ax.add_patch(arrow)
    ax.text(*flabel_xy, flabel_text, fontsize=9, color=col,
            ha='left', va='bottom', zorder=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    # By symmetry, mirror force on particle 2 (smaller arrow)
    if attractive:
        start2 = x2 - 0.18 * direction
        end2 = start2 - 0.55 * direction
    else:
        start2 = x2 + 0.18 * direction
        end2 = start2 + 0.55 * direction
    arrow2 = FancyArrowPatch(start2, end2, color=col, **arrow_kw)
    ax.add_patch(arrow2)

    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.6, 1.8); ax.set_aspect('equal')
    ax.set_xlabel(r'$x/\ell$'); ax.set_ylabel(r'$y/\ell$')
    ax.set_title(title, fontsize=10)

# Left panel: x_3 OUTSIDE the antipodal sphere
x3_out = np.array([0.0, 1.4])
# verify: r23^2 + r31^2 vs r12^2
r12 = np.linalg.norm(x2 - x1)
r23 = np.linalg.norm(x3_out - x2)
r31 = np.linalg.norm(x3_out - x1)
assert r23**2 + r31**2 > r12**2, 'left panel must be outside sphere'
panel(axes[0], x3_out, attractive=False,
      title=r'(a) $\vec{x}_3$ outside: $r_{23}^2 + r_{31}^2 > r_{12}^2$')

# Right panel: x_3 INSIDE the antipodal sphere
x3_in = np.array([0.0, 0.45])
r23 = np.linalg.norm(x3_in - x2)
r31 = np.linalg.norm(x3_in - x1)
assert r23**2 + r31**2 < r12**2, 'right panel must be inside sphere'
panel(axes[1], x3_in, attractive=True,
      title=r'(b) $\vec{x}_3$ inside: $r_{23}^2 + r_{31}^2 < r_{12}^2$')

out = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
fig.savefig(os.path.join(out, 'fig_N3_antipodal.pdf'),
            dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(out, 'fig_N3_antipodal.png'),
            dpi=300, bbox_inches='tight')
print(f"Saved fig_N3_antipodal.pdf / .png to {out}")
