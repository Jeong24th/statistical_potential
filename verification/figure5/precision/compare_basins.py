"""Refine competing search endpoints with opposite dominance in the energy screen.

Include the original winner and the lowest-energy representative of each
opposite-sign endpoint geometry. Distinguish geometries by the RMS difference
of sorted lists of pair distances (>0.002 oscillator lengths). Also include
the most extreme opposite-sign endpoint. Compare at 80 digits after applying
the same 50-digit local optimization to every candidate.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from screen_basins import SCRATCH, ratio64
from verify import HERE, VERIFICATION, modules, mp, np, refine_locally
from scipy.spatial.distance import pdist


def compare(screen, results, endpoints, output):
    n = int(screen['N'])
    phi = float(screen['phi'])
    tag = f'n{n}_phi{phi:.1f}'
    result = json.loads((results/(tag+'.json')).read_text(encoding='utf8'))
    winner = int(result['winner_seed'])
    with np.load(endpoints/(tag+'.npz')) as data:
        energies, positions = data['objective'], data['positions']
    sign = (result['refinement'] or result['fixed'])['dps80']['ratio'] > 1
    near = np.flatnonzero(energies <= energies[winner] + float(screen['window']))
    opposite = [int(i) for i in near if (ratio64(positions[i], phi)>1) != sign]
    selected = [winner]
    representatives = []
    for i in sorted(opposite, key=lambda j:energies[j]):
        distances = np.sort(pdist(positions[i]))
        if all(np.sqrt(np.mean((distances-other)**2))>0.002 for other in representatives):
            selected.append(i)
            representatives.append(distances)
    extreme = int(screen['min_ratio_seed'] if sign else screen['max_ratio_seed'])
    if extreme not in selected:
        selected.append(extreme)
    audit, refine = modules(n, VERIFICATION)
    rows = []
    for seed in selected:
        if seed == winner and result['refinement'] is not None:
            precise = result['refinement']
        else:
            precise = refine_locally(audit, refine, positions[seed], phi)
        rows.append({'seed':seed, 'initial_ratio64':ratio64(positions[seed], phi),
                     'initial_energy64':float(energies[seed]), 'refined':precise})
    ordered = sorted(rows, key=lambda item:mp.mpf(item['refined']['energy80']))
    ground = mp.mpf(ordered[0]['refined']['energy80'])
    selected_sign = ordered[0]['refined']['dps80']['ratio'] > 1
    for row in ordered:
        row['energy_gap80'] = float(mp.mpf(row['refined']['energy80'])-ground)
    gaps = [row['energy_gap80'] for row in ordered
            if (row['refined']['dps80']['ratio']>1) != selected_sign]
    summary = dict(N=n, phi=phi, original_winner=winner,
                   near_energy_count=int(screen['candidate_count']),
                   opposite_initial_count=len(opposite),
                   opposite_geometry_count=len(representatives),
                   lowest_refined_seed=ordered[0]['seed'],
                   lowest_refined_ratio80=ordered[0]['refined']['dps80']['ratio'],
                   lowest_opposite_energy_gap80=min(gaps) if gaps else None,
                   candidates=ordered)
    path=output/(tag+'.json')
    scratch=path.with_suffix('.tmp')
    scratch.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    scratch.replace(path)
    print(tag, 'winning seed', summary['lowest_refined_seed'], 'winning R',
          summary['lowest_refined_ratio80'], 'opposite energy gap',
          summary['lowest_opposite_energy_gap80'], flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=HERE/'near_energy_screen.csv')
    parser.add_argument('--results',type=Path,default=HERE/'results')
    parser.add_argument('--endpoints',type=Path,default=SCRATCH)
    parser.add_argument('--output',type=Path,default=HERE/'competing')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with args.screen.open(encoding='utf8') as handle:
        for record in csv.DictReader(handle):
            if int(record['opposite_count']):
                compare(record,args.results,args.endpoints,args.output)
