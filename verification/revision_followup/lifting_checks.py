"""Dimensional-lifting checks for Supplementary Note 7 (canonical pair forces).

Reproduces, from the public data only:

  (a) Table S2: the N=4 lifting check at phi=2 (planar canonical and minimum-norm
      coefficients; unique three-dimensional decompositions for eta=0.5, 0.1, 0.01, 0.001,
      obtained by a QR solve of the lifted force equations with 80-digit forces);
  (b) Table S3: the N=55, phi=2 lift of the refined configuration into 54 dimensions,
      relative coefficient deviation for eta/eps^8 = 1, 0.3, 0.1, 0.03 at 50 and 80 digits;
  (c) the sign-ratio scan quoted in Note 7, Section 6: the ratio R of the largest attractive
      to the largest repulsive pair force along the same lift family for
      eta/eps^8 = 1, 3, 10, 30, 100, 300 (binary64, with 50/80-digit confirmation of the
      cases with R > 1) and for a second random complement (seed 27) at eta/eps^8 = 300.

Units hbar = m = omega = 1, lengths in a0; K_ij = exp(-r_ij^2 / (2 sinh phi)),
c_ij = 2 (K^-1)_ij K_ij / sinh(phi)^2 (positive = attraction), f_ij = c_ij r_ij.
The lifts are algebraic probes of the kernel, not trap equilibria.

Run:  python lifting_checks.py            (writes lifting_checks.json next to this file)
      python lifting_checks.py --fast     (skips the 80-digit N=55 evaluations)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import mpmath as mp

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / 'low_temperature' / 'data' / 'configurations.json'


def pairs_of(n):
    return list(zip(*np.triu_indices(n, 1)))


def forces_double(pos, phi):
    """Canonical coefficients c, magnitudes f, distances r, net forces F (binary64)."""
    b = np.sinh(phi)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * b))
    C = 2 * np.linalg.inv(K) * K / b ** 2
    iu = np.triu_indices(len(pos), 1)
    r = np.sqrt(d2[iu])
    c = C[iu]
    F = np.einsum('ij,ijd->id', C, pos[None, :, :] - pos[:, None, :])
    return c, c * r, r, F, np.linalg.cond(K)


def forces_high(pos, phi, dps, differentiate=False):
    """Canonical coefficients at dps decimal digits; optionally the net force by
    independent numerical differentiation of -ln det K / sinh(phi)."""
    mp.mp.dps = dps
    n, dim = pos.shape
    p = [[mp.mpf(float(t)) for t in row] for row in pos]
    b = mp.sinh(mp.mpf(float(phi)))

    def kernel(v):
        return mp.matrix([[mp.exp(-sum((v[i][d] - v[j][d]) ** 2 for d in range(dim)) / (2 * b))
                           for j in range(n)] for i in range(n)])

    K = kernel(p)
    inv = K ** -1
    pr = pairs_of(n)
    c = [2 * inv[int(i), int(j)] * K[int(i), int(j)] / b ** 2 for i, j in pr]
    r = [mp.sqrt(sum((p[i][d] - p[j][d]) ** 2 for d in range(dim))) for i, j in pr]
    f = [u * v for u, v in zip(c, r)]
    F = [mp.fsum(2 * inv[i, j] * K[i, j] / b ** 2 * (p[j][d] - p[i][d]) for j in range(n))
         for i in range(n) for d in range(dim)]
    err = None
    if differentiate:
        def potential(*flat):
            v = [list(flat[dim * i:dim * (i + 1)]) for i in range(n)]
            return -mp.log(mp.det(kernel(v))) / b
        flat = tuple(t for row in p for t in row)
        indep = []
        for k in range(len(flat)):
            order = [0] * len(flat)
            order[k] = 1
            indep.append(-mp.diff(potential, flat, tuple(order)))
        err = float(max(abs(a - b2) for a, b2 in zip(F, indep)))
        F = indep
    return dict(c=c, f=f, r=r, F=F, differentiation_error=err)


def equilibrium_matrix(pos, high=False):
    n, dim = pos.shape
    pr = pairs_of(n)
    A = mp.zeros(n * dim, len(pr))
    for k, (i, j) in enumerate(pr):
        delta = [mp.mpf(float(pos[j, d])) - mp.mpf(float(pos[i, d])) for d in range(dim)]
        rr = mp.sqrt(mp.fsum(t * t for t in delta))
        for d in range(dim):
            A[i * dim + d, k] = delta[d] / rr
            A[j * dim + d, k] = -delta[d] / rr
    return A if high else np.array(A.tolist(), float)


def check_n4():
    pos = np.array([[.9, .1], [-.8, .5], [.2, -1.], [-.3, .9]])
    z = np.array([[.3], [-.7], [.2], [1.1]])
    h = forces_high(pos, 2, 80, differentiate=True)
    A = equilibrium_matrix(pos)
    fmin = np.linalg.lstsq(A, np.array(h['F'], float), rcond=None)[0]
    out = dict(positions=pos.tolist(), lift=z.tolist(), phi=2,
               planar_canonical_coefficients=[float(t) for t in h['c']],
               planar_minimum_norm_coefficients=(fmin / np.array(h['r'], float)).tolist(),
               planar_rank=int(np.linalg.matrix_rank(A)),
               independent_gradient_error=h['differentiation_error'], lifts=[])
    for eta in (.5, .1, .01, .001):
        p = np.column_stack((pos, eta * z))
        hp = forces_high(p, 2, 80, differentiate=True)
        B = equilibrium_matrix(p, high=True)
        sol, _ = mp.qr_solve(B, mp.matrix(hp['F']))
        rel = max(abs(a - b) for a, b in zip(sol, hp['f'])) / max(map(abs, hp['f']))
        dev = max(abs(a - b) for a, b in zip(hp['c'], h['c'])) / max(map(abs, h['c']))
        out['lifts'].append(dict(eta=eta, rank=int(np.linalg.matrix_rank(equilibrium_matrix(p))),
                                 unique_coefficients=[float(f / r) for f, r in zip(sol, hp['r'])],
                                 unique_vs_canonical_relative=float(rel),
                                 planar_coefficient_deviation=float(dev),
                                 independent_gradient_error=hp['differentiation_error']))
        assert rel < mp.mpf('1e-60'), rel
    return out


def sign_ratio(c, r):
    f = np.asarray(c, float) * np.asarray(r, float)
    return float(f[f > 0].max() / (-f[f < 0]).max())


def check_n55(fast):
    cfg = json.loads(CONFIG.read_text())
    pos = np.array(cfg['phi_2_refined']['positions'])
    n = len(pos)
    eps8 = float((2 * mp.sinh(2)) ** -4)
    c0 = forces_double(pos, 2)[0]
    out = dict(phi=2, coordinate_source='configurations.json: phi_2_refined', eps8=eps8,
               complement='numpy.random.default_rng(seed).normal(size=(55,52))/sqrt(52)')

    # (b) Table S3: deviation of the lifted coefficients from the planar ones, seed 1.
    z1 = np.random.default_rng(1).normal(size=(n, 52)) / np.sqrt(52)
    h0 = forces_high(pos, 2, 50)
    rows = []
    for ratio in (1., .3, .1, .03):
        eta = ratio * eps8
        p = np.column_stack((pos, eta * z1))
        h50 = forces_high(p, 2, 50)
        row = dict(eta_over_eps8=ratio, eta=eta)
        ref = h50
        if not fast:
            h80 = forces_high(p, 2, 80)
            conv = max(abs(a - b) for a, b in zip(h50['c'], h80['c'])) / max(map(abs, h80['c']))
            row['precision_50_80_over_max'] = float(conv)
            ref = h80
        dev = max(abs(a - b) for a, b in zip(ref['c'], h0['c'])) / max(map(abs, h0['c']))
        row['relative_coefficient_deviation'] = float(dev)
        rows.append(row)
        print('table S3', row, flush=True)
    out['table_S3'] = rows

    # (c) sign-ratio scan along the same lift family, and a second complement.
    scan = []
    for ratio in (1, 3, 10, 30, 100, 300):
        eta = ratio * eps8
        p = np.column_stack((pos, eta * z1))
        c, f, r, F, cond = forces_double(p, 2)
        row = dict(seed=1, eta_over_eps8=ratio, eta=eta, R=sign_ratio(c, r), condition=cond)
        if row['R'] > 1:
            h50 = forces_high(p, 2, 50)
            row['R_50'] = sign_ratio(h50['c'], h50['r'])
            if not fast:
                h80 = forces_high(p, 2, 80)
                row['R_80'] = sign_ratio(h80['c'], h80['r'])
        scan.append(row)
        print('scan', row, flush=True)
    z27 = np.random.default_rng(27).normal(size=(n, 52)) / np.sqrt(52)
    p = np.column_stack((pos, 300 * eps8 * z27))
    c, f, r, F, cond = forces_double(p, 2)
    row = dict(seed=27, eta_over_eps8=300, eta=300 * eps8, R=sign_ratio(c, r), condition=cond)
    h50 = forces_high(p, 2, 50)
    row['R_50'] = sign_ratio(h50['c'], h50['r'])
    if not fast:
        h80 = forces_high(p, 2, 80)
        row['R_80'] = sign_ratio(h80['c'], h80['r'])
    scan.append(row)
    print('scan', row, flush=True)
    out['sign_ratio_scan'] = scan
    out['planar_R'] = sign_ratio(c0, forces_double(pos, 2)[2])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fast', action='store_true', help='skip the 80-digit N=55 evaluations')
    ap.add_argument('--output', type=Path, default=HERE / 'lifting_checks.json')
    args = ap.parse_args()
    result = dict(N4=check_n4())
    print('N4 done', flush=True)
    result['N55'] = check_n55(args.fast)
    args.output.write_text(json.dumps(result, indent=2))
    print('written', args.output)
