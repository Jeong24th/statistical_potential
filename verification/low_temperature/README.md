# Low-temperature precision audit and polynomial-limit checks

Reproducibility package for the low-temperature numerical audit. The arithmetic kernels are the same as those used for the
2026-11-02 audit. No manuscript directory, external data download, TeX installation,
or plotting package is needed.

## Install and run

The tested environment is Python 3.12 with NumPy 2.4.4, SciPy 1.17.1, and
mpmath 1.3.0. From the repository root:

```bash
python -m pip install -r verification/low_temperature/requirements.txt

# Quick check at the most ill-conditioned refined configuration.
python verification/low_temperature/reproduce.py --cases phi_3_refined

# Recompute all 35 distinct configurations from the supplied coordinates.
python verification/low_temperature/reproduce.py --workers 4

# Re-run the local coordinate refinement at phi = 3, starting from its
# supplied unrefined Fig. 4 configuration (50-digit objective and gradient).
python verification/low_temperature/refine.py 3 --output verification/low_temperature/results/refinement
```

The quick check takes seconds and the full fixed-coordinate audit takes minutes
on a typical desktop; actual times depend on the machine. Scripts also run from
other working directories. `--output PATH` selects a separate output directory;
the default is `verification/low_temperature/results/`. Original caches and
bundled reference data are not overwritten. Existing generated results at the
chosen output path can be replaced by a rerun.

`reproduce.py` exits nonzero if the high-precision reference comparison fails.
It checks every signed pair against the reference arrays, the largest force,
dominance ratio, condition number, 50/80-digit convergence, inverse residual,
and stationarity of the three refined configurations. Binary64 rounding errors
are reported rather than required to match a particular BLAS implementation.
Use `--group main`, `--group phase`, or `--group refined` for one subset, or
`--cases phi_2 phi_2p5 phi_3` for selected configurations.

## Data and outputs

- `data/configurations.json`: the exact binary64 input coordinates and inverse
  temperatures, represented by round-tripping decimal numbers, with provenance.
- `data/reference_results.json`: original audit summaries and refinement records.
- `data/reference_forces.npz`: all 1,485 signed 80-digit pair evaluations per
  configuration, rounded to binary64 for portable regression comparisons.
- `data/manifest.json`: dataset membership, input/source SHA-256 hashes, precision
  settings, and the numerical environment. Python source checks allow Git's
  LF/CRLF normalization; JSON inputs are checked byte-for-byte.
- Generated `*_high_precision.json`: freshly computed pair forces, gradients,
  condition number, and potential with 65 significant digits retained.
- Generated `*_pairs.npz`: input coordinates, zero-based pair indices, signed
  binary64 and high-precision forces, and total-potential gradients. High-precision
  values in this NPZ are rounded to binary64; use the decimal JSON for more digits.
- Generated `reproduced_results.json` and `temperature_audit.csv`: the complete
  precision diagnostics and pass/fail reference comparisons. Group membership
  maps each result to the SM condition-number, refinement, and Fig. 5 tables.

The main group has 22 configurations: all 19 Fig. 4 temperatures, plus three
temperatures unique to Fig. 5. The phase group has 13 independently searched
N = 55 configurations, of which three are already in the main group. The refined
group adds three cases. Thus 22 + 13 - 3 + 3 = 35 distinct configurations.

The original caches did **not** save coordinates. The supplied scan coordinates
were reconstructed using the original deterministic multistart searches; they
are not represented as the original unsaved figure coordinates. Comparing an
old cache with a reconstructed/refined result includes coordinate changes.
Comparing arithmetic precision at one supplied configuration does not.

## Optional full search reconstruction

This is slower and is unnecessary for reproducing the fixed-coordinate precision
tables. The original numerical functions are loaded from the repository scripts
without executing their top-level plotting code. The original algorithms and
seeds are retained; no optimization is silently replaced by a different method.

```bash
# Original Fig. 4 search: 300 generic + 15 structured seeds at 19 temperatures;
# 1,000 generic seeds for the three additional Fig. 5 temperatures.
python verification/low_temperature/run_scan.py --workers 4

# Independent original Fig. 5 search: 1,000 generic seeds at all 13 N=55 points.
# Run run_scan.py first, using the same output directory.
python verification/low_temperature/run_phase_scan.py --workers 4

# Refine a newly reconstructed configuration instead of the bundled one.
python verification/low_temperature/refine.py 3 --source reconstructed
```

Both searches accept `--betas` to restrict the original grid and `--output` to
select a generated-data directory. For example, use `--betas 2 --workers 1` on
both searches for a single-temperature check. A full repeat can select slightly
different configurations across optimizer/BLAS versions. Input-coordinate
reproduction is the appropriate direct comparison of arithmetic precision.

## Definitions and interpretation

Units are hbar = m = omega = 1, with lengths in a0 and phi = 1/T. The kernel and
signed canonical pair magnitude are

```text
K_ij = exp[-r_ij^2 / (2 sinh(phi))]
f_ij = 2 r_ij K_ij (K^-1)_ij / sinh(phi)^2
```

Positive f means attraction. This is the specified distance-derivative
decomposition; its signs are not invariant under arbitrary pair re-decomposition.

Each kernel is constructed anew from the same exact binary64 coordinate and phi
inputs using 50 and 80 decimal digits. The code does not merely invert an already
rounded binary64 kernel at higher precision. The spectral condition number is
lambda_max/lambda_min, calculated by the 80-digit symmetric eigensolver.

`refine.py` keeps coordinates in binary64 while evaluating the potential and
gradient at 50 digits. A high-precision reference potential is subtracted before
returning the objective to L-BFGS-B. Final forces and stationarity are checked at
80 digits. This is a local-stationarity check, not a global-minimum certificate.

Headline results:

| T | Refined condition number | Refined largest force | Change from original Fig. 4 cache |
|---|---:|---:|---:|
| 0.5 | 2.99656e8 | 3.424255e5, attractive | +0.00130% |
| 0.4 | 3.32602e10 | 1.160109e7, attractive | -0.01382% |
| 1/3 | 3.57606e12 | 3.798663e8, attractive | -1.45922% |

At fixed coordinates, the original audit found no binary64/80-digit pair-sign
flips. Across the scans and refinements, the maximum 50/80-digit pair discrepancy
was 2.34e-37. Coordinate refinement at T = 1/3 changes 10 of 1,485 signs, all
relatively weak bonds; the largest affected final force is 0.404% of the maximum.
This is distinct from a fixed-coordinate precision comparison. The strongest
force remains attractive. All sampled N = 55 Fig. 5 dominance labels survive.

Binary64 pair-force sums are unreliable at the lowest temperature: at the
unrefined Fig. 4 T = 1/3 configuration the maximum net-force error is about
1.23e4, whereas the actual maximum net statistical force is about 0.615.
The large canonical pair magnitudes therefore require cancellation-aware
interpretation. The row-wise gradient contraction and upper-triangle pair
assembly have different rounding behavior; both diagnostics are recorded.

These original N = 55 checks alone do not validate other particle numbers, the
full interpolated boundary, a T -> 0 sign pattern, or the Monte Carlo ensemble.
The additional finite-temperature cases and analytic limit are treated separately below.
Original plotting scripts/caches remain historical inputs; this package does
not silently replace their plotted numbers or regenerate manuscript figures.

## Follow-up: self-stress, smaller closed shells, and the flat-kernel limit

The self-stress comparison below supports Section 3 of the condensed
Supplementary Note 7. Its N = 4 lifting table is reproduced separately by
`../revision_followup/lifting_checks.py`. The additional N = 55 lifting
sensitivity output retained there is no longer a table in Note 7; the
current Table S3 belongs to the conditioning audit in Note 9.

`review_checks.py` extends the original N = 55 audit. The supplied
`data/review_reference.json` contains the nine reconstructed configurations for
N = 28, 36, 45 at phi = 2.5, 2.7, 3, four separately optimized zero-temperature
mode candidates (N = 28, 36, 45, 55), and their reference summaries. These inputs
are additional to the original 35-case dataset; they do not replace it.

```bash
# Refined N55 canonical/minimum-norm comparison, using the high-precision net force.
python verification/low_temperature/review_checks.py self-stress

# Saved-coordinate checks; nonzero exit if the reference comparison fails.
python verification/low_temperature/review_checks.py closed-shells --saved-only --workers 4
python verification/low_temperature/review_checks.py flat-limit --saved-only --workers 4

# Repeat the original 1000-seed searches, or the 300-seed ground-state searches.
# Use the same output directory: the finite-temperature results supply warm starts.
python verification/low_temperature/review_checks.py closed-shells --workers 4
python verification/low_temperature/review_checks.py flat-limit --workers 4
```

Generated search coordinates are reused if present in the selected output
directory. Choose a fresh `--output PATH` to repeat optimization from scratch.
The default ground-state calculation uses 300 generic seeds and one warm start
when the corresponding finite-temperature result is available. N = 55 uses its
bundled refined phi = 3 warm start. Local searches are not global certificates;
results of fresh searches may differ from the saved candidates.

For a closed shell N = (q+1)(q+2)/2, let P contain all monomials through degree q
evaluated at the coordinates, and W be diagonal with a! b! on degree q and zero
below it. At a fixed nonsingular P, with s = 1/sinh(phi) = 2 epsilon^2,

    lim s^q K^{-1} = L = P^{-T} W P^{-1}
    lim epsilon^(2q) K^{-1} = 2^(-q) L
    lim g_ij / sinh(phi)^(q-2) = 2 r_ij L_ij.

The exact Gaussian feature expansion is infinite. The truncated polynomial
matrix controls the leading term; it is not an exact finite factorization.
For q >= 2, L annihilates constants and both coordinate columns, so the leading
pair field has exactly zero net force (self-stress). This explains the large
cancelling components without interpreting them as large resultant forces.
The script evaluates the inverse limit at s = 0.1, 0.01, 0.001 using 80 digits.

All nine finite-temperature points remain repulsion-dominated and have no
binary64/80-digit pair-sign differences. These coordinates have not undergone
high-precision refinement; their stationarity residuals remain in the summaries.
The separate zero-temperature candidates give leading dominance ratios
1.00239, 0.85701, 1.12604, 1.09121 for N = 28, 36, 45, 55. Thus smaller closed
shells are not universally repulsion-dominated at all temperatures. The limit
depends on the geometry; no global-mode certificate or finite-temperature branch
continuation is claimed, and the small N = 28 excess is not a shell theorem.

At the refined N = 55 configurations, the minimum-norm squared fraction of the
canonical norm is 3.96561e-13 at phi = 2 and 5.84094e-20 at phi = 3. The
self-stress fraction is one minus this small number. The minimum-norm net force
is obtained from the high-precision gradient, not from rounded large pair sums.
Distance derivatives refer to the chosen potential extension; redundant
distances do not permit an independent physical stretch of every single bond.
