# Verification of SI Note 7 (canonicality of the pairwise decomposition)

**Date:** 2026-07-27. Scripts run with the paper's conventions (ℏ=m=ω=1, φ=β, β_φ=sinh φ/φ·β, ω_φ=1/cosh(φ/2); attractive ⟺ K⁻¹ᵃᵇK_ab > 0).

## Scripts

| Script | Purpose |
|---|---|
| `selfstress_check.py` | (A) analytic pair-sum vs numerical gradient; (B) self-stress dimension vs (N−2)(N−3)/2; (C) explicit N=4 sign-flip counterexample; (D) reproduce paper minima; (E) first-pass LP (inequality/null-space form) |
| `certificate_check.py` | Equality-form LP over closed shells N=3–55 (φ=2) and N=55 (φ=1, 0.5), with dual-certificate machinery |
| `refine_N55.py` | Least-squares refinement of the borderline N=45/55 all-repulsive solutions; absolute-margin cross-check |
| `results.json`, `certificate_results.json` | Raw outputs |

## Results

1. **Implementation validated.** Canonical pairwise sum reproduces −∇V_stat to relative 10⁻⁹–10⁻¹⁰ (N=4, 6).
2. **Self-stress dimension confirmed.** Nullity of the central-decomposition map equals (N−2)(N−3)/2 exactly for N=3…8. **N=3: 0 (decomposition strictly unique). N≥4: >0 (non-unique).**
3. **Explicit N=4 counterexample.** Self-stress t·μ_aμ_b from the affine dependence (Σμ_a=0, Σμ_a x_a=0): chosen bond coefficient −94.70 → +94.70 (sign flipped) with max change of any total force 1.4×10⁻¹³. → The PRE manuscript's unconditional "geometrically unique" is false for N≥4; SI Note 7's "canonical / representation-level unique" is the correct statement.
4. **Paper minima reproduced.** N=55, φ=2: shells 3+9+9+17+17, radii vs Table 1 RMSD 2.2×10⁻⁴ a₀; strongest bond attractive, |F|=3.424×10⁵ ℏω/a₀ (paper: 3.4×10⁵). N=6: 5 attractive / 10 repulsive pairs (paper: 5/10), strongest repulsive.
5. **All-repulsive pointwise re-decompositions exist at every studied minimum** — including the N=55 crystal (φ=2): explicit u > 0 with A(−u)=F, uniform absolute margin t = 7.6×10⁻³ ℏω/a₀², equality residual 2.7×10⁻¹⁰ → 6.8×10⁻¹² after refinement, positivity robust. Same for N=3–45 (φ=2, margins 10⁻²–10⁰... scaled margins shrink monotonically with N) and N=55 at φ=0.5, 1.
   - **Note:** the first-pass inequality-form LP (`selfstress_check.py` part E) reported N=55 as infeasible (δ=−1.3×10⁻⁴, status 4). This was a numerical artifact of the null-space-basis formulation; the constructive equality-form solution with machine-precision residuals settles feasibility beyond doubt. Trust `certificate_check.py`/`refine_N55.py` for part E.
6. **Consequence (now in SI Note 7).** For N≥4, even the *existence* of attraction is representation-dependent at any single configuration — so declaring the canonical scheme is *necessary*, not merely prudent. The N=3 antipodal criterion is unconditional.
7. **Scheme-level rigidity (new argument, now in SI Note 7).** Any central, Newtonian decomposition *scheme* that is continuous in the configuration and has decaying pair forces at infinite separation must reduce, on well-separated triples, to the unique N=3 decomposition — which is attractive inside the antipodal ball. Hence no admissible scheme is everywhere repulsive: attraction in the many-body statistical force is scheme-independent at the scheme level, even though it is not at the single-configuration level.
