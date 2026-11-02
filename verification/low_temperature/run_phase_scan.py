"""Audit the original Fig.5 N=55 search separately (1000 seeds at every point)."""
from audit import *
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_phase(beta):
    base=f'phi_{beta:.12g}'.replace('.','p')
    oldrow=json.loads((OUT/f'{base}.json').read_text())
    if oldrow['n_seeds']==1000:return oldrow
    name='phase_'+base
    path=OUT/f'{name}_optimization.npz'
    if not path.exists():
        records=[];start=time.time()
        def tracked(*args,**kwargs):
            res=minimize(*args,**kwargs)
            records.append((res.fun,res.x.reshape(N,2),res.success))
            return res
        tree=ast.parse((CODE/'phase_diagram.py').read_text(encoding='utf-8'))
        env=dict(np=np,minimize=tracked,N_SEEDS=1000)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[]),'phase_diagram.py','exec'),env)
        env['analyze_pair']((N,beta))
        V,pos,_=min(records,key=lambda r:r[0])
        np.savez(path,positions=pos,beta=beta,V=V,candidate_V=[r[0] for r in records],
                 candidate_positions=[r[1] for r in records],success=[r[2] for r in records])
        print(f'PHASE optimized beta={beta:g}: {time.time()-start:.1f}s',flush=True)
    with np.load(path) as c:pos=c['positions']
    row=audit_one(pos,beta,name,'Reconstructed original Fig.5 search: 1000 generic seeds.')
    row['original_Fig5_ratio']=oldrow['original_Fig5_ratio']
    row['n_seeds']=1000
    (OUT/f'{name}.json').write_text(json.dumps(row,indent=2))
    return row


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--betas',type=float,nargs='+',help='Optional subset of the original temperature grid')
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args()
    if args.workers<1:ap.error('--workers must be positive')
    verify_inputs()
    OUT=set_output(args.output)
    with np.load(CODE/'phase_diagram_cache.npz') as a:betas=a['beta_values'].tolist()
    if args.betas is not None:
        if not set(args.betas)<=set(betas):ap.error('--betas must be in the original grid')
        betas=args.betas
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_phase,b):b for b in betas}
        for job in as_completed(jobs):
            row=job.result();rows.append(row)
            (OUT/'phase_scan_results.json').write_text(json.dumps(sorted(rows,key=lambda r:r['beta']),indent=2))
            print(f'PHASE DONE {len(rows)}/{len(betas)} beta={row["beta"]:g} ratio80={row["high"]["ratio"]:.9g} '
                  f'cached={row["original_Fig5_ratio"]:.9g} flips={row["sign_mismatches"]}',flush=True)
