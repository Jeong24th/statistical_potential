"""Check the cold Fig. 5 configurations of the open and small closed shells in oscillator units.

For each completed search file, reconstruct the force matrix at the same
binary64 coordinates in binary64 and with 50 and 80 decimal digits. Refine
the coordinates locally when the original ratio or either rebuilt ratio lies
within 0.05 of unity, the dominance changes, the winning optimizer failed, or
the 80-digit equilibrium gradient exceeds 1e-4. Refinement uses a 50-digit
objective and gradient with binary64 coordinates. This does not prove global
optimality. Each JSON output is atomic and bound to the source coordinate file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get('STATPOT_CODE', HERE.parents[2])).resolve()
VERIFICATION = CODE / 'verification/low_temperature'
CLOSED = {3, 6, 10, 15, 21, 28, 36, 45, 55}
OPEN = sorted(set(range(2, 56)) - CLOSED)
TEMPERATURES = (2.5, 2.7, 3.0)
NEAR = 0.05
GRADIENT_TRIGGER = 1e-4
REFINED_TOLERANCE = 1e-8


def modules(n, verification):
    if str(verification) not in sys.path:
        sys.path.insert(0, str(verification))
    import audit
    import refine
    audit.N = refine.N = n
    audit.PAIRS = np.triu_indices(n, 1)
    return audit, refine


def pair_summary(forces):
    # Individual signed pair magnitudes are positive for attraction.
    attractive = max(0, max(forces))
    repulsive = max(0, -min(forces))
    if not repulsive:
        raise ArithmeticError('No repulsive pair force at a Fig. 5 candidate.')
    return {'max_attractive': float(attractive), 'max_repulsive': float(repulsive),
            'ratio': float(attractive / repulsive),
            'dominance': 'attractive' if attractive > repulsive else 'repulsive',
            'n_attractive': sum(bool(f > 0) for f in forces)}


def check_fixed(audit, positions, phi):
    double = audit.double_eval(positions, phi)
    low = audit.high_eval(positions, phi, 50, eigen=False)
    high = audit.high_eval(positions, phi, 80, eigen=False)
    force = high['f']
    scale = max(abs(f) for f in force)
    agreement = max(abs(a-b) for a, b in zip(low['f'], force)) / scale
    if agreement > mp.mpf('1e-30') or high['inverse_residual'] > mp.mpf('1e-55'):
        raise ArithmeticError('Extended-precision force or inverse did not converge.')
    gradient = np.array(high['grad'], dtype=float).reshape(-1, 2)
    fixed = {
        'binary64': pair_summary(double['f']), 'dps50': pair_summary(low['f']),
        'dps80': pair_summary(force),
        'gradient_max80': float(np.linalg.norm(gradient, axis=1).max()),
        'energy80': mp.nstr(high['V'], 65), 'cond2_binary64': double['cond'],
        'inverse_residual80': float(high['inverse_residual']),
        'max_pair_difference_50_80_over_Fmax': float(agreement),
        'max_pair_difference_binary64_80_over_Fmax': float(
            max(abs(float(d)-h) for d, h in zip(double['f'], force)) / scale),
        'pair_sign_mismatches_binary64_80': sum(
            bool((d > 0) != (h > 0)) for d, h in zip(double['f'], force)),
    }
    if len(positions) == 2:
        if fixed['dps80']['n_attractive'] != 0 or fixed['dps80']['ratio'] != 0:
            raise ArithmeticError('The analytic two-particle repulsion check failed.')
    return fixed


def refine_locally(audit, refine, positions, phi):
    current = positions.copy()
    attempts = []
    for _ in range(2):
        objective = refine.AccuratePotential(phi, current)
        solution = minimize(objective.evaluate, current.ravel(), jac=True,
                            method='L-BFGS-B',
                            options={'maxiter': 600, 'maxls': 40,
                                     'ftol': 1e-23, 'gtol': 1e-9})
        current = solution.x.reshape(-1, 2)
        residual = float(np.linalg.norm(solution.jac.reshape(-1, 2), axis=1).max())
        attempts.append({'success': bool(solution.success), 'message': str(solution.message),
                         'iterations': int(solution.nit), 'evaluations': int(solution.nfev),
                         'gradient_max50': residual})
        if residual <= REFINED_TOLERANCE:
            break
    result = check_fixed(audit, current, phi)
    result.update(positions=current.tolist(), attempts=attempts,
                  max_particle_displacement=float(np.linalg.norm(
                      current - positions, axis=1).max()),
                  meets_stationarity_tolerance=result['gradient_max80'] <= REFINED_TOLERANCE)
    return result


def check(source, output, verification):
    n_str, phi_str = source.stem.split('_phi')
    n, phi = int(n_str[1:]), float(phi_str)
    audit, refine = modules(n, verification)
    start = time.monotonic()
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as inputs:
        positions = np.asarray(inputs['x'], dtype=float)
        archived = float(inputs['archived_ratio'])
        reproduced = float(inputs['reconstructed_ratio'])
        seed = int(inputs['winner_seed'])
        starts = int(inputs['n_seeds'])
        success = bool(inputs['winner_success']) if 'winner_success' in inputs else None
    if positions.shape != (n, 2) or not np.isfinite(positions).all():
        raise ValueError(f'Invalid coordinates: {source}')
    fixed = check_fixed(audit, positions, phi)
    ratio = fixed['dps80']['ratio']
    binary = fixed['binary64']['ratio']
    if abs(binary-reproduced) > 1e-5 * max(1, binary):
        raise ArithmeticError(f'Binary64 ratio disagrees with search at {source.name}.')
    triggers = []
    if min(abs(x-1) for x in (archived, reproduced, ratio)) <= NEAR:
        triggers.append('ratio near 1')
    if len({x > 1 for x in (archived, reproduced, ratio)}) > 1:
        triggers.append('different dominance labels')
    if success is False:
        triggers.append('original optimizer reported failure')
    if fixed['gradient_max80'] > GRADIENT_TRIGGER:
        triggers.append('large initial gradient')
    row = dict(N=n, phi=phi, source=source.name, coordinate_file_sha256=fingerprint,
               n_seeds=starts, winner_seed=seed, winner_success=success,
               archived_ratio=archived, reconstructed_ratio=reproduced, fixed=fixed,
               refinement_triggers=triggers, refinement=None)
    if triggers:
        refined = refine_locally(audit, refine, positions, phi)
        refined['energy_change80'] = float(mp.mpf(refined['energy80'])-mp.mpf(fixed['energy80']))
        row['refinement'] = refined
    row['elapsed_seconds'] = round(time.monotonic()-start, 3)
    destination = output / (source.stem + '.json')
    scratch = destination.with_suffix(f'.{os.getpid()}.tmp')
    scratch.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n', encoding='utf8')
    scratch.replace(destination)
    final = row['refinement'] or row['fixed']
    return f'{source.stem}  archive={archived:.8f}  fixed={ratio:.8f}  final={final["dps80"]["ratio"]:.8f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinates', type=Path, default=HERE.parent/'search/coordinates')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--verification', type=Path, default=VERIFICATION)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--numbers', type=int, nargs='+', default=sorted(OPEN + [3, 6, 10, 15, 21]))
    parser.add_argument('--phis', type=float, nargs='+', default=TEMPERATURES)
    parser.add_argument('--watch', action='store_true',
                        help='Process arriving atomic coordinate files until all requested cases exist.')
    args = parser.parse_args()
    if not (args.verification/'audit.py').is_file():
        parser.error('Point --verification at verification/low_temperature in the manuscript code.')
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.coordinates / f'n{n}_phi{phi:.1f}.npz'
               for n in args.numbers for phi in args.phis]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            pending = []
            waiting = 0
            for source in targets:
                if not source.is_file():
                    if not args.watch:
                        raise FileNotFoundError(source)
                    waiting += 1
                    continue
                previous = args.output / (source.stem+'.json')
                if previous.exists():
                    saved = json.loads(previous.read_text(encoding='utf8'))
                    if saved['coordinate_file_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest():
                        continue
                pending.append(source)
            jobs = [pool.submit(check, p, args.output, args.verification) for p in pending]
            for job in as_completed(jobs):
                print(job.result(), flush=True)
            if not args.watch or waiting == 0:
                return
            if not pending:
                time.sleep(5)


if __name__ == '__main__':
    main()
