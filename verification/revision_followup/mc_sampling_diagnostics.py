"""Sampling diagnostics of the stored N=55 robustness runs (Supplementary Note 10).

For the unperturbed stored configurations of the three replica-exchange runs
(robust_A.npz, robust_B.npz, robust_C.npz, produced by robustness_mc.py) this script
  * reclassifies every shot (strongest canonical bond attractive or not) and compares the
    pooled P_att(T) with robustness.json,
  * estimates the integrated autocorrelation time of the attraction indicator per run and
    temperature (initial positive sequence estimator) and the resulting effective sample
    size and standard error of P_att, and the run-to-run spread of P_att (standard deviation
    across the three runs, as quoted in Note 10, and the range),
  * locates the crossover T_x where the pooled P_att crosses one half (linear interpolation),
  * gives a 95% block-bootstrap interval for T_x (2000 replicates, block lengths 10, 20, 50,
    the same blocks used at every temperature).

Units hbar = m = omega = 1. Usage:
    python mc_sampling_diagnostics.py --shots <directory with robust_{A,B,C}.npz>
The stored shots (about 13 MB each) are not kept in the repository; their SHA-256 hashes
are listed in README.md. Reference output: mc_sampling_diagnostics.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 55
IU = np.triu_indices(N, 1)


def strongest_attractive(X, T):
    """True if the strongest canonical bond of each shot in X (n,55,2) is attractive."""
    d2 = np.sum((X[:, :, None, :] - X[:, None, :, :]) ** 2, -1)
    K = np.exp(-d2 / (2 * np.sinh(1 / T)))
    g = (np.linalg.inv(K) * K * np.sqrt(d2))[:, IU[0], IU[1]]
    k = np.argmax(np.abs(g), axis=1)
    return g[np.arange(len(X)), k] > 0


def tau_int(x):
    """Integrated autocorrelation time, initial-positive-sequence estimator."""
    y = x - x.mean()
    ac = np.correlate(y, y, 'full')[len(y) - 1:]
    if ac[0] == 0:
        return 1.0
    ac = ac / ac[0]
    total = 0.0
    for i in range(1, len(ac) - 1, 2):
        pair = ac[i] + ac[i + 1]
        if pair <= 0:
            break
        total += pair
    return max(1.0, 1 + 2 * total)


def crossing(T, p):
    for i in range(len(p) - 1):
        if (p[i] - .5) * (p[i + 1] - .5) <= 0 and p[i] != p[i + 1]:
            return float(T[i] + (.5 - p[i]) * (T[i + 1] - T[i]) / (p[i + 1] - p[i]))
    return float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shots', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=HERE / 'mc_sampling_diagnostics.json')
    ap.add_argument('--replicates', type=int, default=2000)
    args = ap.parse_args()

    runs = []
    temps = None
    for name in 'ABC':
        d = np.load(args.shots / f'robust_{name}.npz')
        if temps is None:
            temps = d['temps']
        assert np.allclose(d['temps'], temps)
        runs.append(np.array([strongest_attractive(X, T) for T, X in zip(temps, d['pos'])], dtype=float))
    S = np.array(runs)                      # (runs, temps, shots)
    n = S.shape[2]
    pooled = S.mean(axis=(0, 2))
    out = dict(temps=temps.tolist(), shots_per_run=int(n), pooled_P_att=pooled.tolist(), diagnostics=[])
    for i, T in enumerate(temps):
        taus = [tau_int(c[i]) for c in S]
        ess = float(sum(n / t for t in taus))
        p = float(S[:, i].mean())
        out['diagnostics'].append(dict(T=float(T), tau_int=[float(t) for t in taus], ess=ess,
                                       se=float(np.sqrt(p * (1 - p) / ess)),
                                       run_spread_sd=float(S[:, i].mean(axis=1).std(ddof=1)),
                                       run_spread_range=float(np.ptp(S[:, i].mean(axis=1)))))
    ref = HERE / 'robustness.json'
    if ref.exists():
        saved = json.loads(ref.read_text())['variants']['none' if 'none' in json.loads(ref.read_text())['variants'] else 'base']['P_att']
        out['max_difference_from_robustness_json'] = float(np.max(np.abs(np.array(saved) - pooled)))
    out['crossing'] = crossing(temps, pooled)
    rng = np.random.default_rng(834753)
    out['block_bootstrap'] = []
    for L in (10, 20, 50):
        xs = []
        for _ in range(args.replicates):
            sample = []
            for j in range(S.shape[0]):
                starts = rng.integers(0, n, size=n // L)
                ids = ((starts[:, None] + np.arange(L)) % n).ravel()
                sample.append(S[j][:, ids].mean(axis=1))
            xs.append(crossing(temps, np.mean(sample, axis=0)))
        xs = np.array(xs)
        out['block_bootstrap'].append(dict(block_length=L, replicates=len(xs),
                                           crossing_ci95=np.nanquantile(xs, [.025, .975]).tolist(),
                                           no_crossing=int(np.isnan(xs).sum())))
    out['ess_min'] = min(v['ess'] for v in out['diagnostics'])
    out['se_max'] = max(v['se'] for v in out['diagnostics'])
    out['run_spread_sd_max'] = max(v['run_spread_sd'] for v in out['diagnostics'])   # 'run-to-run spread' of Note 10
    out['run_spread_range_max'] = max(v['run_spread_range'] for v in out['diagnostics'])
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ('diagnostics', 'pooled_P_att')}, indent=2))


if __name__ == '__main__':
    main()
