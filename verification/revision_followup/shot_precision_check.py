"""Precision check of the binary64 bond classification on cold stored shots (Supplementary Note 10).

For each of the three stored robustness runs (robust_{A,B,C}.npz from robustness_mc.py) the
script takes, at the coldest ladder temperature k_B T/(hbar omega) = 0.4, the first stored shot
and the shot with the largest binary64 spectral condition number of K, and classifies each of
them at the true temperature and at a temperature underestimated by 20% (T_assigned = 0.32).
Each classification is done in binary64 and, independently, at 50 decimal digits (mpmath),
and the script reports: condition number, whether the strongest canonical pair and its type
(attractive/repulsive) agree, the number of the 1485 pair signs that differ, and the largest
coefficient difference relative to the largest coefficient.

The reproducible outcome is the identity and type of the strongest pair. At condition numbers
of order 1e16 the binary64 values of the small coefficients, and hence the number of weak-pair
sign mismatches, depend on the BLAS/LAPACK implementation and should be read as orders of
magnitude only; the reference file was produced with NumPy's bundled OpenBLAS on Windows.

Units hbar = m = omega = 1. Usage:
    python shot_precision_check.py --shots <directory with robust_{A,B,C}.npz>
Reference output: shot_precision_check.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import mpmath as mp

HERE = Path(__file__).resolve().parent
N = 55
IU = np.triu_indices(N, 1)


def coefficients_double(X, T):
    d2 = np.sum((X[:, None] - X[None, :]) ** 2, -1)
    K = np.exp(-d2 / (2 * np.sinh(1 / T)))
    g = (np.linalg.inv(K) * K * np.sqrt(d2))[IU]
    return K, g


def coefficients_high(X, T, dps=50):
    with mp.workdps(dps):
        xx = [[mp.mpf(float(v)) for v in row] for row in X]
        b = mp.sinh(1 / mp.mpf(str(T)))
        dd = [[mp.fsum((xx[i][k] - xx[j][k]) ** 2 for k in range(2)) for j in range(N)] for i in range(N)]
        K = mp.matrix([[mp.exp(-d / (2 * b)) for d in row] for row in dd])
        Ki = K ** -1
        return np.array([float(Ki[i, j] * K[i, j] * mp.sqrt(dd[i][j])) for i in range(N) for j in range(i + 1, N)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shots', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=HERE / 'shot_precision_check.json')
    ap.add_argument('--dps', type=int, default=50)
    args = ap.parse_args()
    out = dict(dps=args.dps, numpy=np.__version__, cases=[], condition_scan=[])
    for name in 'ABC':
        d = np.load(args.shots / f'robust_{name}.npz')
        assert abs(float(d['temps'][0]) - 0.4) < 1e-12
        P = d['pos'][0]                                     # shots at T = 0.4
        d2 = np.sum((P[:, :, None, :] - P[:, None, :, :]) ** 2, -1)
        for T_eval in (0.4, 0.32):
            ev = np.linalg.eigvalsh(np.exp(-d2 / (2 * np.sinh(1 / T_eval))))
            cond = ev[:, -1] / ev[:, 0]
            out['condition_scan'].append(dict(run=name, T_true=0.4, T_assigned=T_eval,
                                              condition_quantiles=np.quantile(cond, [0, .5, .9, .99, 1]).tolist(),
                                              nonpositive_smallest_eigenvalue=int((ev[:, 0] <= 0).sum())))
            for label, idx in (('first', 0), ('worst-conditioned', int(np.argmax(cond)))):
                K, g = coefficients_double(P[idx], T_eval)
                gh = coefficients_high(P[idx], T_eval, args.dps)
                k, kh = int(np.argmax(np.abs(g))), int(np.argmax(np.abs(gh)))
                row = dict(run=name, shot=label, sample_index=idx, T_true=0.4, T_assigned=T_eval,
                           condition64=float(cond[idx]),
                           strongest64=[int(IU[0][k]), int(IU[1][k])], strongest_high=[int(IU[0][kh]), int(IU[1][kh])],
                           attractive64=bool(g[k] > 0), attractive_high=bool(gh[kh] > 0),
                           same_strongest_pair=bool(k == kh), same_type=bool((g[k] > 0) == (gh[kh] > 0)),
                           pair_sign_mismatches=int(np.sum((g > 0) != (gh > 0))),
                           max_coefficient_error_over_largest=float(np.max(np.abs(g - gh)) / np.max(np.abs(gh))))
                out['cases'].append(row)
                print(row, flush=True)
    out['all_same_strongest_pair'] = all(c['same_strongest_pair'] for c in out['cases'])
    out['all_same_type'] = all(c['same_type'] for c in out['cases'])
    out['max_sign_mismatches'] = max(c['pair_sign_mismatches'] for c in out['cases'])
    out['max_coefficient_error_over_largest'] = max(c['max_coefficient_error_over_largest'] for c in out['cases'])
    args.output.write_text(json.dumps(out, indent=2))
    print('same strongest pair in all cases:', out['all_same_strongest_pair'], '| same type:', out['all_same_type'],
          '| max mismatches:', out['max_sign_mismatches'], '| max coefficient error:', out['max_coefficient_error_over_largest'])


if __name__ == '__main__':
    main()
