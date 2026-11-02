"""Recompute the N=55 precision checks from supplied, immutable coordinates.

Run from any working directory. The default visits all 35 distinct saved cases:
22 union-grid cases + 10 additional Fig.5 configurations + 3 refined cases.
"""
from audit import *
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


def check_case(name):
    pos,beta,origin=load_saved_case(name)
    row=audit_one(pos,beta,name,origin)
    ref=json.loads((ROOT/'data/reference_results.json').read_text(encoding='utf-8'))[name]
    # The binary64 discrepancy itself is platform dependent. Validate the
    # high-precision reference and convergence, not BLAS-specific roundoff.
    errors=[]
    for key in ['Fmax','ratio']:
        if not np.isclose(row['high'][key],ref['high'][key],rtol=1e-12,atol=0):
            errors.append(f'{key} differs from the high-precision reference')
    if not np.isclose(row['cond2_high'],ref['cond2_high'],rtol=1e-12,atol=0):
        errors.append('condition number differs from reference')
    with np.load(ROOT/'data/reference_forces.npz',allow_pickle=False) as f:
        expected=f[name]
    with np.load(OUT/f'{name}_pairs.npz',allow_pickle=False) as f:
        actual=f['f_high']
    if not np.allclose(actual,expected,rtol=1e-12,atol=0):
        errors.append('signed pair forces differ from reference')
    if np.any(np.signbit(actual)!=np.signbit(expected)):
        errors.append('high-precision sign mismatch with reference')
    if row['precision_50_80_max_relative']>=1e-35:
        errors.append('50/80-digit force convergence failed')
    if row['inverse_residual_80']>=1e-65:
        errors.append('80-digit inverse residual failed')
    if name.endswith('_refined') and row['gradient_max_high']>=3e-9:
        errors.append('refined configuration stationarity failed')
    row['reference_check']={'passed':not errors,'errors':errors}
    (OUT/f'{name}.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def write_tables(rows,manifest):
    (OUT/'reproduced_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'temperature_audit.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['case','groups','beta','T','condition_number_2','Fmax_80','ratio_80','n_att_80',
              'Fmax_relative_error_64_80','worst_pair_relative_error_64_80','sign_mismatches_64_80',
              'precision50_80_relative','net_pair_sum_error','gradient_max_80','reference_passed']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in sorted(rows,key=lambda r:(r['beta'],r['name'])):
            w.writerow(dict(case=r['name'],groups=';'.join(k for k,v in manifest['groups'].items() if r['name'] in v),
                beta=r['beta'],T=r['T'],condition_number_2=r['cond2_high'],Fmax_80=r['high']['Fmax'],
                ratio_80=r['high']['ratio'],n_att_80=r['high']['n_att'],
                Fmax_relative_error_64_80=r['Fmax_rel_error'],worst_pair_relative_error_64_80=r['pair_max_relative_error'],
                sign_mismatches_64_80=r['sign_mismatches'],precision50_80_relative=r['precision_50_80_max_relative'],
                net_pair_sum_error=r['pair_sum_net_force_error_max'],gradient_max_80=r['gradient_max_high'],
                reference_passed=r['reference_check']['passed']))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',nargs='+',help='Case IDs from data/configurations.json; default: all')
    ap.add_argument('--group',choices=['main','phase','refined'],help='Select one documented dataset')
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    if args.cases and args.group:ap.error('Use either --cases or --group')
    manifest=verify_inputs()
    names=list(dict.fromkeys(x for group in manifest['groups'].values() for x in group))
    selected=args.cases or (manifest['groups'][args.group] if args.group else names)
    if not set(selected)<=set(names):ap.error('Unknown case ID')
    if len(set(selected))!=len(selected):ap.error('Duplicate case IDs would overwrite outputs')
    OUT=set_output(args.output)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(check_case,n) for n in selected]):
            rows.append(future.result())
    rows.sort(key=lambda r:(r['beta'],r['name']))
    write_tables(rows,manifest)
    failures=[r['name'] for r in rows if not r['reference_check']['passed']]
    print(f'Completed {len(rows)} cases; reference failures: {failures}; output: {OUT}',flush=True)
    if failures:raise SystemExit(1)
