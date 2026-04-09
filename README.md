# Emergent Statistical Attraction and Pauli Crystal Formation

Code for reproducing the numerical results in the paper.

## Files

- `vtotal_minimize.py` — Find the minimum of V_total(X) for N fermions in a 2D harmonic trap. Computes pairwise statistical forces (attractive/repulsive), strongest bonds, and shell structure. Generates main text Figs. 1, 2, and 4.
- `strongest_bond.py` — Identify the dominant pairwise force on each particle. Generates main text Figs. 1 (right) and 2 (bottom).
- `melting_analysis.py` — Temperature dependence of force magnitudes and attractive/repulsive ratio. Generates main text Fig. 3.
- `phase_diagram.py` — Map the globally strongest force type (ATT/REP) in the (N, T) plane. Generates main text Fig. 4.
- `sm_density.py` — Conditional ground-state density and 1-body density. Generates SM Figs. 1-2.
- `sm_multiN.py` — Force lines and strongest bonds for N=3 to 55. Generates SM Figs. 3-5.
- `sm_temperature.py` — Temperature evolution of forces. Generates SM Figs. 6-7.
- `sm_distance_histogram.py` — Distance-dependent attractive fraction. Generates SM Fig. 8.

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib

## Usage

```bash
# Main text figures
python vtotal_minimize.py 2 6      # Fig. 1: N=6, beta=2
python vtotal_minimize.py 2 55     # Fig. 2: N=55, beta=2
python strongest_bond.py 6 2       # Fig. 1 right panel
python strongest_bond.py 55 2      # Fig. 2 bottom panel
python melting_analysis.py 6       # Fig. 3 top
python melting_analysis.py 55      # Fig. 3 bottom
python phase_diagram.py            # Fig. 4

# SM figures
python sm_density.py 6             # SM Fig. 1 (left)
python sm_density.py 55            # SM Fig. 1 (right), SM Fig. 2
python sm_multiN.py                # SM Figs. 3-5
python sm_temperature.py           # SM Figs. 6-7
python sm_distance_histogram.py    # SM Fig. 8
```

## Units

All calculations use natural units: hbar = m = omega = 1.
Lengths in units of a_0 = sqrt(hbar / m omega).
Temperature parameterized by varphi = omega * beta * hbar.
