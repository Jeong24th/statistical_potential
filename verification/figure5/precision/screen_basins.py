"""Look for alternative dominance labels among nearly degenerate search endpoints.

For each reconstructed winner, screen all 1,000 endpoints retained by the
search, taking those within max(1e-5, 10 times the observed binary64 energy
error of the winner) of the lowest binary64 energy (hbar omega units).
Classify each such endpoint in binary64. Report extremes, and the
lowest-energy endpoint on the opposite side of R=1, if any. The energy window
is a numerical diagnostic, not a bound on missing or misordered minima.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from verify import HERE, np

SCRATCH = Path(os.environ.get('STATPOT_SEARCH_SCRATCH',
                              HERE.parent / 'scratch/search')).resolve()


def ratio64(positions, phi):
    bp = np.sinh(phi) / phi * phi
    pairs = np.triu_indices(len(positions), 1)
    diff = positions[:, None] - positions[None, :]
    d2 = np.sum(diff * diff, axis=2)
    kernel = np.exp(-d2 / (2 * bp))
    # Common positive dimensional prefactors drop out of this ratio.
    force = (np.linalg.inv(kernel) * kernel)[pairs] * np.sqrt(d2[pairs])
    return max(0., float(max(force))) / -float(min(force))


def screen(record, candidates):
    data = json.loads(record.read_text(encoding='utf8'))
    phi = data['phi']
    with np.load(candidates, allow_pickle=False) as inputs:
        energy, pos = inputs['objective'], inputs['positions']
    if len(energy) != 1000:
        raise ValueError(f'Expected full 1,000-endpoint checkpoint at {candidates}')
    winner = int(energy.argmin())
    if winner != data['winner_seed']:
        raise ValueError(f'Selected seed differs from saved winner: {candidates}')
    energy_error = abs(energy[winner]-float(data['fixed']['energy80']))
    window = max(1e-5, 10*energy_error)
    near = np.flatnonzero(energy <= energy[winner] + window)
    ratios = np.array([ratio64(pos[ii], phi) for ii in near])
    selected = data['refinement'] or data['fixed']
    att = selected['dps80']['ratio'] > 1
    opposite = near[(ratios > 1) != att]
    alt_seed = int(opposite[energy[opposite].argmin()]) if len(opposite) else None
    return dict(N=data['N'], phi=phi, winner_seed=winner, window=window,
                candidate_count=len(near), energy_error_winner=energy_error,
                min_ratio=float(ratios.min()), max_ratio=float(ratios.max()),
                min_ratio_seed=int(near[ratios.argmin()]), max_ratio_seed=int(near[ratios.argmax()]),
                opposite_count=len(opposite), opposite_lowest_seed=alt_seed,
                opposite_energy_gap=energy[alt_seed]-energy[winner] if alt_seed is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoints', type=Path, default=SCRATCH)
    parser.add_argument('--results', type=Path, default=HERE/'results')
    parser.add_argument('--output', type=Path, default=HERE/'near_energy_screen.csv')
    args = parser.parse_args()
    rows = []
    for record in sorted(args.results.glob('n*_phi*.json')):
        endpoints = args.endpoints / (record.stem+'.npz')
        rows.append(screen(record, endpoints))
    rows.sort(key=lambda r:(r['N'], r['phi']))
    scratch = args.output.with_suffix('.tmp')
    with scratch.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scratch.replace(args.output)
    print(f'Screened {len(rows)} cases and {sum(r["candidate_count"] for r in rows)} near-energy endpoints.')
    for row in rows:
        if row['opposite_count']:
            print('Follow up', row)


if __name__ == '__main__':
    main()
