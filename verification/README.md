# Verification of the canonical pair decomposition

## Supplementary Note 7

The five sections cover (1) the definition and sign convention, (2) self-stresses
and the count of free parameters, (3) characterization by continuation across
dimension, (4) the minimum-norm comparison and cluster separability, and
(5) the collinear limit and one-dimensional log-gas.
The planar freedom is described by self-stress; the dimension count is given in the
main text after Eq. (5) and in Supplementary Note 7, Section 2.

The canonical prescription is singled out among continuous, distance-based
assignments that reproduce the same determinant-potential force in every
dimension up to N - 1. Continuity and distance dependence within the plane alone
do not give uniqueness. For N = 3, the central decomposition is unique on
non-collinear configurations; collinear values require the continuation condition.

The retained numerical comparisons are:

| Current result | Reproduction |
|---|---|
| N = 55 canonical/minimum-norm comparison, Table S3 in Section 4 | `low_temperature/review_checks.py self-stress` |
| N = 4 lifting table, Table S2 in Section 3 | `revision_followup/lifting_checks.py` |

The cluster-separability statement is limited to the following: in D >= 2,
any cluster-separable central decomposition must contain attraction for some
configurations, because an isolated obtuse non-collinear triple has a unique
decomposition. It does not make the signs of individual N >= 4 bonds
independent of the decomposition.

For current precision protocols, reference data and commands, see
[low_temperature/README.md](low_temperature/README.md) and
[revision_followup/README.md](revision_followup/README.md).
