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
- `out/N55_run{A,B,C}.npz` — three independent N=55 replica-exchange runs (gas-like, crystalline and
  mixed initializations; 25 temperatures, 3x10^4 measurement sweeps, thinning 10).
- `out/N6_run{A,B}.npz` — N=6 null control (random and crystalline initializations).
- `out/validate_val_T{0.5,1.0}.npz` — N=2 exact-distribution validation chains.
- `out/N55_summary.json`, `out/N6_summary.json` — per-temperature diagnostics and
  observables (acceptance, tau, ESS, R-hat, bond statistics, modality tests).

## Reproduce

From this directory, run the validation and analyze the archived chains:

    python mc_ensemble.py validate --out out/validate.npz
    python mc_ensemble.py analyze --pattern "out/N55_run*.npz" --out out/N55
    python ../verification/revision_followup/mc_sampling_diagnostics.py --long-runs
    python make_si_figs.py

To generate an additional N=55 chain on the same 25-temperature grid:

    python mc_ensemble.py run --N 55 --temps "0.40,0.44,0.48,0.52,0.55,0.58,0.60,0.62,0.64,0.67,0.70,0.75,0.80,0.90,1.00,1.15,1.30,1.40,1.45,1.50,1.55,1.60,1.70,1.85,2.00" --burn 5000 --sweeps 30000 --thin 10 --init mixed --seed 1001 --run-id N55-example --out out/example_N55.npz

For N=55, the pooled probability that a shot has an attractive strongest canonical
bond crosses one half at \(k_{\rm B}T_\times/(\hbar\omega)=0.661\)
(95% block-bootstrap interval 0.653–0.671). The three long
runs give 0.6597, 0.6595 and 0.6660, with an rms spread of 0.003 about their mean.
The unperturbed pooled crossing of 0.639 in Table S4 comes from three separate
[robustness runs](../verification/revision_followup/README.md).
For N=6 the probability stays below 0.07 at all sampled temperatures. The radial
marginals remain unimodal and nearly temperature-independent across the structural
level crossings.
