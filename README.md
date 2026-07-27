# Attractive statistical forces and Pauli crystal formation in trapped Fermi gases

Code for reproducing the numerical results in the paper.

## Files

### Main text figures
- `vtotal_minimize.py` — Minimize V_total(X) for N fermions in a 2D harmonic trap. Computes pairwise statistical forces (attractive/repulsive) and shell structure.
- `strongest_bond.py` — Identify the dominant pairwise force on each particle.
- `melting_analysis.py` — Generate the two-panel N=55 temperature scan (main Fig. 4), the complementary force-sum/pair-count SI figure, and the N=6 null control. The cached N=55 scan avoids repeating the 300-seed optimization.
- `phase_diagram.py` — Generate the shaded force-dominance diagram and interpolated T_c(N) boundary (main Fig. 5) from `phase_diagram_cache.npz`.

### Supplemental Material figures
- `sm_1body.py` — One-body density for N=6 and N=55.
- `sm_density.py` — Conditional N-body density (-ln|Psi_0|^2).
- `sm_multiN.py` — Force lines and strongest bonds for closed-shell N=3 to 55.
- `sm_shell_radii_and_histogram.py` — Shell radii comparison (V_total vs |Psi_0|) and distance-dependent attractive fraction.
- `sm_temperature.py` — Temperature evolution of forces and strongest bonds.
- `sm_structural_transition.py` — Structural order parameter r_min and representative configurations for N=55.

### Verification

The `verification/` directory contains the three independent checks and their raw JSON outputs supporting Supplementary Note 7: canonical-force validation, self-stress non-uniqueness, all-repulsive re-decomposition certificates, and the refined N=55 feasibility check. See `verification/README.md` for conventions and benchmark values.

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib

## Usage

```bash
# Main text figures
python vtotal_minimize.py 2 6      # Fig. 1 left: N=6, beta=2
python vtotal_minimize.py 2 55     # Fig. 2 top: N=55, beta=2
python strongest_bond.py 6 2       # Fig. 1 right
python strongest_bond.py 55 2      # Fig. 2 bottom
python melting_analysis.py 6       # N=6 SI null control
python melting_analysis.py 55      # Main Fig. 4 + N=55 SI panels
python phase_diagram.py            # Main Fig. 5

# SM figures
python sm_1body.py                 # SM Fig. 1
python sm_density.py               # SM Fig. 2
python sm_multiN.py                # SM Figs. 3-4
python sm_shell_radii_and_histogram.py  # SM Figs. 5-6
python sm_temperature.py           # SM Figs. 7-8
python sm_structural_transition.py # SM Figs. 9-10

# SI Note 7 verification
python verification/selfstress_check.py
python verification/certificate_check.py
python verification/refine_N55.py
```

Figures are written to `figures/` in a standalone clone. In the manuscript worktree, the scripts detect `Manuscript/Nature_Comm/` and write directly to its `figures/` directory.

## Units

All calculations use natural units: hbar = m = omega = 1.
Lengths in units of a_0 = sqrt(hbar / m omega).
Temperature parameterized by varphi = omega * beta * hbar.
