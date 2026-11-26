# Figure 5 verification

This calculation checks which canonical pair force has the largest magnitude at the sampled configurations of the two-dimensional isotropic harmonic trap. The table reports the ratio of the largest attractive magnitude to the largest repulsive magnitude. A ratio above one means that the strongest pair is attractive. The numerator is zero when no pair is attractive. Length and force are in units of the oscillator length and the oscillator energy divided by that length, respectively.

## Findings

The original searches reproduced all 150 archived binary64 ratios bit for bit in the recorded environment. The strongest-pair classifications survived the extended-precision evaluations and the 52 selected coordinate refinements. The open shells had 69 attractive and 66 repulsive classifications. All 15 small closed-shell controls were repulsive. The largest final gradient norm among the 52 refined configurations was below 1.32e-9 in oscillator units.

At the closest approach to equal force magnitudes, for 23 particles at inverse temperature 3, the ratio was 0.9959770 before coordinate refinement and 0.9959809 afterward. For 11 particles at inverse temperature 2.7, the screen of nearby search endpoints found candidates of both signs. All six representative configurations became attraction-dominated after refinement, with a ratio of 1.076676. At the reconstructed 54-particle configuration at inverse temperature 3, an 80-digit eigensystem calculation gave a condition number of 5.16e12. The corresponding 55-particle value was 3.58e12. The eigenvalues are recorded in `precision/conditioning.json`.

The former figure omitted 29 archived attractive samples on nine open-shell rows because it followed only attractive intervals connected to the coldest sample. Five of those samples lie in the three cold columns. The revised PDF displays a symbol at each of the 702 sampled points. The plotted symbols agree with the grid at all 169 attractive and 533 repulsive samples. Only the sampled configurations have been classified.

## Data and methods

- `figure5_grid.csv` contains all 702 sampled particle-number and temperature combinations. It records the archive ratio, the ratio used for the revised plot, its source, and the classification shown by the previous drawing. A symbol represents a sampled configuration. Opposite symbols at adjacent temperatures bracket a change of sign.
- `search/coordinates/` contains one file for each of the 150 new searches, with the selected coordinates, winning seed, original optimizer status and binary64 force ratio. The search used the original 1,000 initial configurations and selection rule. The 135 open-shell points and 15 small closed-shell controls use inverse temperatures 2.5, 2.7 and 3 in oscillator units.
- `precision/summary.csv` compares the archived ratios with 50- and 80-digit results at the reconstructed coordinates and with the final selected ratios. `precision/results/` gives the forces of largest magnitude, stationarity residuals, conditioning diagnostics, and locally refined coordinates where applicable. A refined result uses a 50-digit objective and analytic gradient, coordinates stored in binary64, and a final 80-digit evaluation.
- Local refinement was triggered by a ratio within 0.05 of one, a change of classification, a reported optimizer failure or an initial 80-digit maximum particle-gradient norm above 0.0001. The target residual at a refined configuration was 1e-8. The endpoint screen compares alternative low-energy candidates from the same searches. Its results are in `precision/near_energy_screen.csv` and `precision/competing/`. These searches locate low-lying minima. They do not prove global optimality.
- The preceding closed-shell checks come from Supplementary Note 9. They include the three coldest temperatures at particle numbers 28, 36 and 45, and the full 13-temperature row at 55. Combined with the new calculations they cover all 162 sampled points in the three coldest columns.
- `independent/` checks selected results with a Cholesky solve and finite differences of the log determinant. `figure/figure_validation.json` records the number of individual marks checked directly in the output PDF.

## Reproduction

From the repository root, install the numerical dependencies and the plotting tools and draw the figure.

```bash
python -m pip install -r verification/figure5/requirements.txt
python phase_diagram.py --audit figures/figure_validation.json
```

The command checks the 702 table entries against the original binary64 cache, plots their verified ratios, counts the vector marks in the PDF, and writes `figures/fig_phase_diagram.pdf` and `figures/fig_phase_diagram.png`. The reference PDF and its original audit are in `verification/figure5/figure/`. The PDF creation time varies between runs. The pointwise sign digest in the audit is independent of the timestamp.

The grid has 530 binary64 values and 172 extended-precision values. The latter comprise all 162 cold samples plus the other ten temperatures on the 55-particle row. At the 150 new cold points, both 50- and 80-decimal-digit evaluations were made at the reconstructed coordinates. At 52 of those points coordinates were also locally refined before the final 80-digit evaluation. The [earlier precision audit](../low_temperature/README.md) supplies the nine remaining cold points for 28, 36 and 45 particles and the three for 55, with a 50/80-digit comparison. The original searches and software fingerprints in `search/*_run_metadata.json` refer to [the original numerical code revision](https://github.com/Jeong24th/statistical_potential/tree/663ab3f30e395f53bc4f99e0bfa3374d444a9dff).

For the following commands, change to `verification/figure5/`. The first two commands replay the binary64 force calculation on 150 stored winners. The original ratios and classifications are saved for comparison. For ill-conditioned kernels the binary64 replay tolerance scales with the measured condition number. BLAS implementations may give different rounded ratios. The next two commands regenerate the summary and the 702-point grid from the individual precision records and the earlier closed-shell checks.

```bash
python search/reconstruct.py --group open --verify
python search/reconstruct.py --group small_closed --verify
python precision/summarize.py --output scratch/summary.csv
python assemble_grid.py --summary scratch/summary.csv --output scratch/figure5_grid.csv
```

To repeat the 50/80-digit evaluations on all 150 stored winners and their triggered refinements, specify an empty output directory. The independent check uses a 90-digit Cholesky calculation and finite differences on selected coordinates. The extremal-eigenvalue check is `python precision/conditioning.py`. To repeat the other twelve cold samples, use the closed-shell and phase-group commands in the earlier precision audit.

```bash
python precision/verify.py --output scratch/replay_results --workers 2
python independent/check_pair_forces.py --output scratch/independent_checks.json
python ../low_temperature/review_checks.py closed-shells --saved-only
python ../low_temperature/reproduce.py --group phase
```

To repeat the 150 independent 1,000-start searches, use a separate working copy with empty `search/coordinates/` and `precision/results/` directories. Search checkpoints default to `scratch/search/`. Set `STATPOT_SEARCH_SCRATCH` to select a different location. Both the search and the near-energy screen must use the same checkpoint directory.

```bash
python search/reconstruct.py --group open --workers 4
python search/reconstruct.py --group small_closed --workers 4
python precision/verify.py --workers 2
python precision/screen_basins.py
python precision/compare_basins.py
python precision/summarize.py
python assemble_grid.py
```
