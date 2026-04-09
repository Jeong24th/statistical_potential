# Statistical Potential for Identical Fermions: Emergent Attraction and Pauli Crystal Formation

Code for reproducing the numerical results in the paper.

## Files

### Main text figures
- `vtotal_minimize.py` — Minimize V_total(X) for N fermions in a 2D harmonic trap. Computes pairwise statistical forces (attractive/repulsive) and shell structure. Generates main text Figs. 1 (left) and 2 (top).
- `strongest_bond.py` — Identify the dominant pairwise force on each particle. Generates main text Figs. 1 (right) and 2 (bottom).
- `melting_analysis.py` — Temperature dependence of force magnitudes and attractive/repulsive ratio. Generates main text Fig. 3.
- `phase_diagram.py` — Map the globally strongest force type (ATT/REP) in the (N, T) plane. Generates main text Fig. 5.

### Supplemental Material figures
- `sm_1body.py` — One-body density for N=6 and N=55. Generates SM Fig. 1.
- `sm_density.py` — Conditional N-body density (-ln|Psi_0|^2). Generates SM Fig. 2.
- `sm_multiN.py` — Force lines and strongest bonds for closed-shell N=3 to 55. Generates SM Figs. 3-4.
- `sm_shell_radii_and_histogram.py` — Shell radii comparison (V_total vs |Psi_0|) and distance-dependent attractive fraction. Generates SM Figs. 5-6.
- `sm_temperature.py` — Temperature evolution of forces and strongest bonds. Generates SM Figs. 7-8.

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
python melting_analysis.py         # Fig. 3
python phase_diagram.py            # Fig. 5

# SM figures
python sm_1body.py                 # SM Fig. 1
python sm_density.py               # SM Fig. 2
python sm_multiN.py                # SM Figs. 3-4
python sm_shell_radii_and_histogram.py  # SM Figs. 5-6
python sm_temperature.py           # SM Figs. 7-8
```

## Units

All calculations use natural units: hbar = m = omega = 1.
Lengths in units of a_0 = sqrt(hbar / m omega).
Temperature parameterized by varphi = omega * beta * hbar.
