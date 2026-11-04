# Verification of the canonical pair decomposition

## Current Supplementary Note 7 (November 2026)

The condensed note has four sections: (1) definition and nonuniqueness,
(2) characterization by continuation across dimension, (3) minimum norm and
cluster separability, and (4) the collinear limit and one-dimensional log-gas.
The planar freedom is described by self-stress; the dimension count is in the
main text after Eq. (5).

The canonical prescription is singled out among continuous, distance-based
assignments that reproduce the same determinant-potential force in every
dimension up to N - 1. Continuity and distance dependence within the plane alone
do not give uniqueness. For N = 3, the central decomposition is unique on
non-collinear configurations; collinear values require the continuation condition.

The retained numerical comparisons are:

| Current result | Reproduction |
|---|---|
| N = 55 canonical/minimum-norm comparison, Section 3 | `low_temperature/review_checks.py self-stress` |
| N = 4 lifting table, Table S2 in Section 3 | `revision_followup/lifting_checks.py` |

The latter script additionally retains the N = 55 lifting-sensitivity and
sign-ratio checks removed when Note 7 was condensed. They are additional
diagnostics, not a current SM table or section. The historical JSON key
`N55.table_S3` is retained for compatibility; current Table S3 is the
low-temperature conditioning table of Note 9.

The cluster-separability statement is limited to the following: in D >= 2,
any cluster-separable central decomposition must contain attraction for some
configurations, because an isolated obtuse non-collinear triple has a unique
decomposition. It does not make the signs of individual N >= 4 bonds
independent of the decomposition.

## Earlier checks (July 2026)

The scripts and raw outputs directly in this directory are preserved as
numerical background. They predate the current Note 7 organization.

Conventions: hbar = m = omega = 1, phi = beta, beta_phi = sinh(phi),
omega_phi = 1/cosh(phi/2); positive canonical coefficients mean attraction.
Some legacy feasibility calculations use coefficients c_ab multiplying
(x_b - x_a), whereas the current minimum-norm comparison minimizes the sum
of squared signed magnitudes f_ab = c_ab r_ab. These norms are different.

| Script | Purpose |
|---|---|
| `selfstress_check.py` | Analytic pair-sum versus numerical gradient; generic planar self-stress dimension; an N = 4 sign-flip example; mode searches; first-pass inequality/null-space LP |
| `certificate_check.py` | Equality-form LP over closed shells N = 3-55 at phi = 2 and N = 55 at phi = 1, 0.5 |
| `refine_N55.py` | Least-squares refinement of the N = 45/55 all-repulsive solutions and an absolute-margin check |
| `results.json`, `certificate_results.json` | Historical raw outputs |

1. The canonical pair sum agrees with the numerical gradient to relative
   1e-9 to 1e-10 for the tested N = 4, 6 configurations.
2. The generic planar nullity (N - 2)(N - 3)/2 was checked for N = 3-8.
   Uniqueness for N = 3 assumes a non-collinear configuration.
3. In the N = 4 example, a self-stress changes a chosen coefficient from
   -94.70 to +94.70 while changing the net force by at most 1.4e-13.
4. The N = 55, phi = 2 mode search gives shells 3+9+9+17+17 and a largest
   attractive force of about 3.424e5. The later high-precision comparison
   uses the refined coordinates in `low_temperature/`.
5. All-repulsive pointwise decompositions were found at the studied minima.
   The N = 55, phi = 2 equality-form solution has absolute margin 7.6e-3
   and equality residual 2.7e-10, reduced to 6.8e-12 by refinement.
   These examples do not prove the existence of an all-repulsive
   decomposition at every configuration.
6. The first-pass N = 55 inequality-form LP reported infeasibility
   (delta = -1.3e-4, status 4). This was a numerical artifact of the
   null-space formulation; use the later equality-form certificate and
   refinement when interpreting that case.

For current precision protocols, reference data and commands, see
[low_temperature/README.md](low_temperature/README.md) and
[revision_followup/README.md](revision_followup/README.md).
