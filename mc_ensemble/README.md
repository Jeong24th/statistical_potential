# Thermal-ensemble Monte Carlo (SI Note 8, Methods "Ensemble Monte Carlo")

Direct sampling of the exact configurational distribution exp(-beta_phi V_total) of the
classical dual (positive weight; no sign problem), using single-particle Metropolis moves
(step sizes adapted during burn-in only) plus replica exchange across the temperature
ladder with the temperature-dependent swap rule described in Methods.

## Files

- `mc_ensemble.py` — sampler (`run`), exact-N=2 validation (`validate`), pooled analysis
  with autocorrelation/ESS/split-R-hat diagnostics, Wilson intervals, and KDE modality
  tests (`analyze`).
- `make_si_figs.py` — produces the SI figures `fig_SM_mc_crossover.pdf` and
  `fig_SM_mc_marginals.pdf` from the run outputs, and prints the per-run crossover
  temperatures.
- `out/N55_run{A,B,C}.npz` — three independent N=55 replica-exchange runs (mixed, random,
  crystalline initializations; 25 temperatures, 3x10^4 measurement sweeps, thinning 10).
- `out/N6_run{A,B}.npz` — N=6 null control (random and crystalline initializations).
- `out/validate_val_T{0.5,1.0}.npz` — N=2 exact-distribution validation chains.
- `out/N55_summary.json`, `out/N6_summary.json` — per-temperature diagnostics and
  observables (acceptance, tau, ESS, R-hat, bond statistics, modality tests).

## Reproduce

    python mc_ensemble.py validate --out out/validate.npz
    python mc_ensemble.py run --N 55 --temps "0.40,...,2.00" --burn 5000 --sweeps 30000 \
        --thin 10 --init mixed --seed 1001 --run-id N55-A --out out/N55_runA.npz
    python mc_ensemble.py analyze --pattern "out/N55_run*.npz" --out out/N55
    python make_si_figs.py

Headline results: the probability that the strongest canonical bond of a single sampled
configuration is attractive crosses one half at k_B T_x / (hbar omega) = 0.66(1) for N=55
(per-run 0.660/0.660/0.666), stays below 0.07 for N=6 at all temperatures, and the radial
marginals remain unimodal and nearly temperature-independent across the structural level
crossings (mode-level transitions).
