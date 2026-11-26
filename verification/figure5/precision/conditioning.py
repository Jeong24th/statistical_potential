"""Recheck the largest fixed-coordinate binary64 condition estimate at 80 digits."""
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from verify import HERE, VERIFICATION, modules, mp, np


def eigenvalues(audit, positions, phi, coordinate_source):
    high = audit.high_eval(np.asarray(positions, dtype=float), phi, 80, eigen=True)
    return dict(N=len(positions), phi=phi, digits=80, coordinate_source=coordinate_source,
                lambda_min=mp.nstr(high['lmin'],65), lambda_max=mp.nstr(high['lmax'],65),
                cond2=mp.nstr(high['cond'],65))


if __name__ == '__main__':
    with (HERE/'summary.csv').open(encoding='utf8') as handle:
        largest = max(csv.DictReader(handle), key=lambda r:float(r['cond2_fixed_binary64']))
    n, phi = int(largest['N']), float(largest['phi'])
    tag = f'n{n}_phi{phi:.1f}'
    xfile = HERE.parent/'search/coordinates'/f'{tag}.npz'
    record = json.loads((HERE/'results'/f'{tag}.json').read_text(encoding='utf8'))
    audit, _ = modules(n, VERIFICATION)
    with np.load(xfile) as inputs:
        fixed = eigenvalues(audit, inputs['x'], phi, '../search/coordinates/'+xfile.name+'#x')
    fixed['coordinate_file_sha256'] = record['coordinate_file_sha256']
    study = dict(selection='Largest binary64 condition estimate at the reconstructed winners of the 150 new cases.',
                 fixed=fixed)
    if record['refinement']:
        study['same_case_after_refinement'] = eigenvalues(
            audit, record['refinement']['positions'], phi,
            f'results/{tag}.json#refinement.positions')
    reference = json.loads((VERIFICATION/'data/reference_results.json').read_text(encoding='utf8'))
    entry = reference['phase_phi_3']
    study['previous_Fig5_N55'] = dict(N=55, phi=3, digits=80,
        source='verification/low_temperature/data/reference_results.json#phase_phi_3',
        lambda_min=entry['lambda_min'], lambda_max=entry['lambda_max'], cond2=entry['cond2_high'])
    (HERE/'conditioning.json').write_text(json.dumps(study, indent=2)+'\n', encoding='utf8')
    print(f'N={n}, phi={phi}: cond2={fixed["cond2"]}; previous N=55 {entry["cond2_high"]}')
