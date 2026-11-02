# Revision follow-up calculations (November 2026)

Scripts and results behind the additions made in the revised manuscript
(Supplementary Notes 3, 7 and 10). All units are oscillator units
(`hbar = m = omega = 1`, lengths in `a0`), as in the rest of the repository.

| Script | What it computes | Output |
|---|---|---|
| `mode_vs_phi.py` | Distance between the `V_total` minimum at inverse temperature `phi` and the zero-temperature mode (maximum of `|Psi_0|^2`) for `N = 55`, as the root-mean-square displacement of the full configuration after optimal rotation, reflection and particle assignment (Hungarian algorithm). Inputs: `../low_temperature/data/configurations.json` and the `flat_limit['55']` positions in `../low_temperature/data/review_reference.json`. | `mode_vs_phi.json`, `mode_vs_phi.png` |
| `n3_obtuse.py` | Probability that three trapped fermions form an obtuse triangle (equivalently, that one canonical pair force is attractive) as a function of temperature, by importance sampling of the exact weight `exp(-beta_phi V_total)` with `8e6` samples per temperature; also with Gaussian position noise added before classification. | `n3_obtuse.json`, `n3_obtuse.png` |
| `robustness_mc.py` | Replica-exchange Monte Carlo for `N = 55` on the 25-temperature ladder of `mc_ensemble/`, storing thinned configurations. Usage: `python robustness_mc.py <run_id> <random|crystal|mixed> <seed> [burn] [sweeps] [thin]`. | `robust_<run_id>.npz` |
| `robustness_analyze.py` | Recomputes the probability that the strongest canonical bond of a shot is attractive after perturbing the stored configurations: Gaussian position noise, misassigned temperature, one missed atom. | `robustness.json`, `robustness.png` |
| `lifting_checks.py` | Dimensional-lifting checks of Supplementary Note 7: the `N = 4` lifting table (unique three-dimensional decompositions from 80-digit forces, converging to the planar canonical coefficients), the `N = 55`, `phi = 2` lift of the refined configuration into 54 dimensions (coefficient deviation at 50 and 80 digits for `eta/eps^8 = 1, 0.3, 0.1, 0.03`), and the sign-ratio scan along the same lift family (`eta/eps^8` up to 300) plus a second random complement, showing that the sign pattern far from the plane is not universal. Input: `../low_temperature/data/configurations.json` (`phi_2_refined`). `--fast` skips the 80-digit `N = 55` evaluations. | `lifting_checks.json` |

The Monte Carlo runs used for the manuscript were `A random 11`, `B crystal 22`, `C mixed 33`
with `burn = 3000`, `sweeps = 6000`, `thin = 10` (600 stored configurations per temperature and run).
The `.npz` files are not stored in the repository (about 13 MB each); `robustness.json` contains
the pooled results.

Requirements: the pinned environment of `../low_temperature/requirements.txt` (NumPy, SciPy) plus
matplotlib for the figures. Run each script from any working directory; paths are resolved
relative to the script location.
