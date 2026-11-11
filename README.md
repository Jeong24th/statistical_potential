# Exact equilibrium configurational mapping of an ideal Fermi gas

Reproducibility code for the accompanying statistical-potential manuscript. The mapping of equilibrium configurational statistics is exact for ideal fermions in any external potential when expressed through the exact one-particle thermal kernel. It preserves the equilibrium configurational distribution and partition function; it does not assert equivalence of classical trajectories and quantum time evolution. The calculations here use harmonic confinement, whose kernel is known in closed form and gives the canonical pair-force decomposition after the one-body trap factors are separated. A kernel approximation gives an approximate mapping.

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

## Supplementary Note 7 checks

Supplementary Note 7 has four sections: definition and nonuniqueness;
characterization by continuation across dimension; minimum norm and cluster separability;
and the collinear limit and one-dimensional log-gas.

- The retained N = 55 canonical/minimum-norm comparison (Section 3) is reproduced by
  `verification/low_temperature/review_checks.py self-stress`.
- The retained N = 4 lifting table (Table S1, Section 3) is reproduced by
  `verification/revision_followup/lifting_checks.py`.

See [verification/README.md](verification/README.md) for conventions and limitations.

## Low-temperature precision audit

[`verification/low_temperature/`](verification/low_temperature/README.md) provides
the low-temperature numerical audit: freshly constructed 50- and
80-decimal-digit kernels, temperature-resolved condition numbers, signed pair
forces, cancellation diagnostics, and local high-precision refinement. Saved
coordinates and benchmark results allow a direct fixed-coordinate reproduction;
separate commands repeat the original multistart searches. The lowest-temperature
largest force remains attractive after a 1.46% correction, while some weak bond
signs change during coordinate refinement. See that directory for the pinned
Python 3.12 numerical environment, execution commands, and scope limitations.
The follow-up checks add nine low-temperature points for N = 28, 36, 45,
the N = 55 self-stress comparison, and the analytic polynomial inverse limit
evaluated at four zero-temperature mode candidates. The smaller-shell repulsion
claim is restricted to the sampled finite-temperature range; the limit does not
support a universal threshold at N = 55.

## Supplementary calculations

[`verification/revision_followup/`](verification/revision_followup/README.md) contains
the calculations supporting the manuscript: the distance between the finite-temperature
mode and the zero-temperature mode for N = 55 as a function of temperature (full-configuration
comparison after optimal rotation and particle assignment), the N = 3 obtuse-triangle
probability versus temperature (a parameter-free prediction for single-shot experiments),
the robustness of the shot-by-shot bond statistics against position noise, temperature
misassignment and a missed atom, evaluated on stored Monte Carlo configurations, and the
dimensional-lifting checks of the canonical pair decomposition (the retained N = 4 table
and additional N = 55 diagnostics). The stored robustness shots are available from the
authors; the repository supplies their SHA-256 checksums, generation scripts and summary
results, not those three shot files.

## Requirements

- Python 3.12 for the tested precision-audit environment (pinned dependencies in
  `verification/low_temperature/requirements.txt`). The legacy plotting scripts
  use the packages below.
- NumPy
- SciPy
- Matplotlib
- The plotting scripts use `text.usetex=True` and require a working LaTeX
  installation with the cm-super fonts; `verification/low_temperature/` does
  not require TeX.

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

# Current Supplementary Note 7 numerical comparisons
python verification/low_temperature/review_checks.py self-stress
python verification/revision_followup/lifting_checks.py
```

Figures are written to `figures/` next to each script.

## Units

All calculations use \(\hbar=m=\omega=1\). Length is measured in \(a_0=\sqrt{\hbar/(m\omega)}\), and the inverse-temperature parameter is \(\varphi=\beta\hbar\omega\).
