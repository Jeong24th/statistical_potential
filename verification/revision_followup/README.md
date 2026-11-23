# Supplementary calculations

Scripts and results supporting Supplementary Notes 3, 7, 8 and 10, together with
additional lifting diagnostics (not shown in the SM). All units are oscillator units
(`hbar = m = omega = 1`, lengths in `a0`), as in the rest of the repository.

| Script | What it computes | Output |
|---|---|---|
| `mode_vs_phi.py` | Distance between the `V_total` minimum at inverse temperature `phi` and the zero-temperature mode (maximum of `|Psi_0|^2`) for `N = 55`, as the root-mean-square displacement of the full configuration after optimal rotation, reflection and particle assignment (Hungarian algorithm). Inputs: `../low_temperature/data/configurations.json` and the `flat_limit['55']` positions in `../low_temperature/data/review_reference.json`. | `mode_vs_phi.json`, `mode_vs_phi.png` |
| `n3_obtuse.py` | Probability that three trapped fermions form an obtuse triangle (equivalently, that one canonical pair force is attractive) as a function of temperature, by importance sampling of the exact weight `exp(-beta_phi V_total)` with `8e6` samples per temperature; also with Gaussian position noise added before classification. | `n3_obtuse.json`, `n3_obtuse.png` |
| `robustness_mc.py` | Replica-exchange Monte Carlo for `N = 55` on the 25-temperature ladder of `mc_ensemble/`, storing thinned configurations. Usage: `python robustness_mc.py <run_id> <random|crystal|mixed> <seed> [burn] [sweeps] [thin]`. | `robust_<run_id>.npz` |
| `robustness_analyze.py` | Recomputes the probability that the strongest canonical bond of a shot is attractive after perturbing the stored configurations: Gaussian position noise, misassigned temperature, one missed atom. | `robustness.json`, `robustness.png` |
| `lifting_checks.py` | The retained `N = 4` lifting table of Supplementary Note 7 (Table S2, Section 3) (unique three-dimensional decompositions from 80-digit forces, converging to the planar canonical coefficients); additional N = 55 diagnostics (not shown in the SM): the `N = 55`, `phi = 2` lift of the refined configuration into 54 dimensions (coefficient deviation at 50 and 80 digits for `eta/eps^8 = 1, 0.3, 0.1, 0.03`), and the sign-ratio scan along the same lift family (`eta/eps^8` up to 300) plus a second random complement, showing that the sign pattern far from the plane is not universal. Input: `../low_temperature/data/configurations.json` (`phi_2_refined`). `--fast` skips the 80-digit `N = 55` evaluations. | `lifting_checks.json` |
| `mc_sampling_diagnostics.py` | Sampling diagnostics of the unperturbed stored shots: pooled `P_att(T)` compared with `robustness.json`, integrated autocorrelation time, effective sample size and standard error of `P_att` per temperature, run-to-run spread, the crossover `T_x`, and a 95% block-bootstrap interval for `T_x` (2000 replicates, block lengths 10, 20, 50). Usage: `python mc_sampling_diagnostics.py --shots <dir with robust_{A,B,C}.npz>`. | `mc_sampling_diagnostics.json` |
| `mc_sampling_diagnostics.py --long-runs` | Pooled crossing and individual crossings from the archived `mc_ensemble/out/N55_run{A,B,C}.npz` chains. Run `python mc_sampling_diagnostics.py --long-runs` to reproduce the pooled estimate 0.661 and its 95% block-bootstrap interval 0.653–0.671. | `long_run_crossing.json` |
| `shot_precision_check.py` | Binary64 versus 50-digit classification of the first and the worst-conditioned stored shot of each run at `k_B T/(hbar omega) = 0.4`, evaluated at the true temperature and at a temperature underestimated by 20%: condition number, identity and type of the strongest pair, number of weak-pair sign mismatches, largest coefficient error. The strongest pair and its type are the reproducible outcome; at condition numbers of order 1e16 the mismatch counts and small-coefficient errors depend on the BLAS/LAPACK build. Usage: `python shot_precision_check.py --shots <dir>`. | `shot_precision_check.json` |

For both datasets, the bootstrap resamples circular blocks independently within each run,
using the same block indices across all temperatures of that run. The 2.5th and 97.5th
percentiles of the pooled crossings give the interval for each block length (10, 20 or
50 stored samples, 2000 replicates each, NumPy random seed 834753).

The three robustness runs used for Table S4 were `A random 11`, `B crystal 22`, `C mixed 33`
with `burn = 3000`, `sweeps = 6000`, `thin = 10` (600 stored configurations per temperature and run).
The `.npz` files are not stored in the repository (about 13 MB each); `robustness.json` contains
the pooled results. The stored shots used for `robustness.json`, `mc_sampling_diagnostics.json` and
`shot_precision_check.json` are available from the authors and are identified by their SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `robust_A.npz` | `eb7e6cfd397d06e0948ca88a2b3f4c2b874597471f13ad51db5a3dec547fa58e` |
| `robust_B.npz` | `ef1fab36d736aee63dfb3fa20108c8ed83dfc94bec995c71afed949b29709029` |
| `robust_C.npz` | `8a153f28f287876caef3fc4abd0d8202ded475add11be56fbee3a4e70e9572af` |

Requirements: the pinned environment of `../low_temperature/requirements.txt` (NumPy, SciPy, mpmath) plus
matplotlib for the figures. Run each script from any working directory; paths are resolved
relative to the script location.
