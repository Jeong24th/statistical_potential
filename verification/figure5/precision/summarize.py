"""Collect three-temperature Fig. 5 precision comparisons in one CSV."""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = {3, 6, 10, 15, 21}
PRIOR = {28, 36, 45, 55}
EXPECTED = {(n, phi) for n in range(2, 56) if n not in PRIOR
            for phi in (2.5, 2.7, 3.)}


def summarize(results, output, partial):
    rows = []
    for file in results.glob('n*_phi*.json'):
        item = json.loads(file.read_text(encoding='utf8'))
        fixed = item['fixed']
        refined = item['refinement']
        chosen_seed = item['winner_seed']
        opposite_gap = None
        candidate_file = results.parent / 'competing' / file.name
        triggers = item['refinement_triggers'].copy()
        if candidate_file.is_file():
            competition = json.loads(candidate_file.read_text(encoding='utf8'))
            chosen_seed = competition['lowest_refined_seed']
            opposite_gap = competition['lowest_opposite_energy_gap80']
            chosen = next(c for c in competition['candidates'] if c['seed'] == chosen_seed)
            refined = chosen['refined']
            triggers.append('near-energy opposite-sign candidate')
        terminal = refined or fixed
        source = 'locally refined' if refined else 'fixed coordinates'
        if chosen_seed != item['winner_seed']:
            source = 're-ranked refined candidates'
        rows.append(dict(
            N=item['N'], phi=item['phi'],
            shell='closed-control' if item['N'] in CONTROL else 'open',
            archived_ratio=item['archived_ratio'],
            reconstructed_ratio64=item['reconstructed_ratio'],
            fixed_ratio50=fixed['dps50']['ratio'], fixed_ratio80=fixed['dps80']['ratio'],
            refined_ratio80=refined['dps80']['ratio'] if refined else None,
            final_ratio=terminal['dps80']['ratio'],
            source=source, refinement_trigger='; '.join(triggers),
            winner_seed=item['winner_seed'], selected_seed=chosen_seed,
            opposite_basin_gap80=opposite_gap, n_seeds=item['n_seeds'],
            winner_success=item['winner_success'],
            fixed_gradient80=fixed['gradient_max80'], final_gradient80=terminal['gradient_max80'],
            cond2_fixed_binary64=fixed['cond2_binary64'],
            fixed_50_80_error_over_Fmax=fixed['max_pair_difference_50_80_over_Fmax'],
            final_50_80_error_over_Fmax=terminal['max_pair_difference_50_80_over_Fmax'],
            fixed_binary64_80_error_over_Fmax=fixed['max_pair_difference_binary64_80_over_Fmax'],
            fixed_weak_pair_sign_changes=fixed['pair_sign_mismatches_binary64_80'],
            final_stationarity_passed=refined['meets_stationarity_tolerance'] if refined else None,
            source_file='results/'+file.name,
            selected_coordinates_source=(
                'competing/'+file.name if candidate_file.is_file() else
                'results/'+file.name if refined else '../search/coordinates/'+file.stem+'.npz')))
    rows.sort(key=lambda r: (r['N'], r['phi']))
    cases = {(r['N'], r['phi']) for r in rows}
    if cases != EXPECTED and not partial:
        raise ValueError(f'Expected 150 cases. Missing: {sorted(EXPECTED-cases)}; extra: {sorted(cases-EXPECTED)}')
    if not rows:
        raise ValueError('No completed results found.')
    if any(row['n_seeds'] != 1000 for row in rows):
        raise ValueError('An incomplete search was supplied.')
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_suffix('.tmp')
    with scratch.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scratch.replace(output)
    print(f'{len(rows)} cases, {sum(r["refined_ratio80"] is not None for r in rows)} locally refined case winners; {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results')
    parser.add_argument('--output', type=Path, default=HERE/'summary.csv')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    summarize(args.results, args.output, args.partial)
