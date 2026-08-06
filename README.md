# Exact classical dual of an ideal Fermi gas

Reproducibility code for the Nature Communications manuscript. The work maps the configurational statistics of \(N\) identical fermions exactly to a classical ensemble governed by a collective statistical potential. Pauli-crystal geometry, the canonical force decomposition, and its temperature dependence are applications of that exact dual.

## Main-text figures

- `main_fig_N3_schematic.py` — geometric sign criterion for the canonical \(N=3\) force decomposition (Fig. 1).
- `main_fig_N6_combined.py` — one-body density, conditional density, effective-potential minimum, and dominant forces for \(N=6\) (Fig. 2).
- `vtotal_minimize.py` — minimization of \(V_{\rm total}(X)\), shell structure, and all canonical pair forces; supplies the \(N=55\) force panel in Fig. 3.
- `strongest_bond.py` — strongest canonical pair force on each particle; supplies the \(N=55\) dominant-force panel in Fig. 3.
- `melting_analysis.py` — two-panel \(N=55\) temperature scan with shared structural-regime shading (Fig. 4), complementary SI diagnostics, and the \(N=6\) null control.
- `phase_diagram.py` — force-dominance diagram and interpolated \(T_{\rm c}(N)\) boundary (Fig. 5), generated from `phase_diagram_cache.npz`.

## Supplementary figures and diagnostics

- `sm_1body.py` — one-body densities.
- `sm_density.py` — conditional \(N\)-body densities, \(-\ln|\Psi_0|^2\).
- `sm_multiN.py` — force networks and strongest bonds across closed shells.
- `sm_rmin_multiN.py` — innermost-shell radius across particle number.
- `sm_shell_radii_and_histogram.py` — shell-radius comparison and distance-resolved attractive fraction.
- `sm_shell_radii_vs_temp.py` — shell radii versus temperature.
- `sm_temperature.py` — temperature evolution of canonical forces and strongest bonds.
- `sm_structural_transition.py` — \(N=55\) structural order parameter and representative configurations.

## Ensemble Monte Carlo

The `mc_ensemble/` directory contains the thermal-ensemble Monte Carlo supporting
Methods ("Ensemble Monte Carlo") and Supplementary Note 8: a validated
Metropolis + replica-exchange sampler of the exact positive-weight dual, the raw chains
for the N=55 production runs and the N=6 null control, and the scripts generating the
SI figures (ensemble force crossover at k_B T_x/hbar-omega = 0.66(1); unimodal radial
marginals across the structural level crossings). See `mc_ensemble/README.md`.

## Verification

The `verification/` directory contains three independent checks and their raw JSON outputs supporting Supplementary Note 7:

1. validation of the canonical force formula;
2. self-stress non-uniqueness of unconstrained pair decompositions;
3. constructive all-repulsive re-decomposition certificates, including the refined \(N=55\) feasibility calculation.

See `verification/README.md` for conventions, limitations, and benchmark values. In particular, the first-pass \(N=55\) inequality-form infeasibility result is a numerical artifact and must not be used; the equality-form constructive certificate is the relevant result.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

## Usage

```bash
# Main-text figures
python main_fig_N3_schematic.py
python main_fig_N6_combined.py
python vtotal_minimize.py 2 55
python strongest_bond.py 55 2
python melting_analysis.py 55
python phase_diagram.py

# Supplementary controls and figures
python melting_analysis.py 6
python sm_1body.py
python sm_density.py
python sm_multiN.py
python sm_rmin_multiN.py
python sm_shell_radii_and_histogram.py
python sm_shell_radii_vs_temp.py
python sm_temperature.py
python sm_structural_transition.py

# Supplementary Note 7 verification
python verification/selfstress_check.py
python verification/certificate_check.py
python verification/refine_N55.py
```

Figures are written to `figures/` in a standalone clone. In the manuscript worktree, the scripts detect `Manuscript/Nature_Comm/` and write directly to its `figures/` directory.

## Units

All calculations use \(\hbar=m=\omega=1\). Length is measured in \(a_0=\sqrt{\hbar/(m\omega)}\), and the inverse-temperature parameter is \(\varphi=\beta\hbar\omega\).
