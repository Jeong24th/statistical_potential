"""Reconstruct original deterministic searches; audit every temperature.

Run: python run_scan.py
Original Fig.4: 300 generic + 15 structured seeds, unchanged original functions.
Three temperatures unique to Fig.5: original 1000 generic-seed search.
Saved summaries cannot uniquely recover the original unsaved coordinates.
"""
from audit import *
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(beta):
    name = f'phi_{beta:.12g}'.replace('.','p')
    path = OUT / f'{name}_optimization.npz'
    with np.load(CODE/'melting_analysis_cache_N55.npz') as c:
        found = np.flatnonzero(np.abs(c['betas']-beta)<1e-13)
        old = {k:c[k][found[0]].item() for k in c.files if k!='betas'} if len(found) else None
    if not path.exists():
        started=time.time()
        records=[]
        def tracked(*args,**kwargs):
            res=minimize(*args,**kwargs)
            records.append((res.fun,res.x.reshape(N,2),res.success,res.nit,res.nfev))
            return res
        if old is not None:
            env=original_functions()
            env['minimize']=tracked
            pos,V,*_=env['find_minimum'](beta)
            origin='Reconstructed original Fig.4 search: 300 generic + 15 structured seeds.'
        else:
            tree=ast.parse((CODE/'phase_diagram.py').read_text(encoding='utf-8'))
            env=dict(np=np,minimize=tracked,N_SEEDS=1000)
            nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'phase_diagram.py','exec'),env)
            env['analyze_pair']((N,beta))
            best=min(records,key=lambda r:r[0]); V,pos=best[:2]
            origin='Reconstructed original Fig.5 search: 1000 generic seeds.'
        np.savez(path,positions=pos,beta=beta,V=V,candidate_V=[r[0] for r in records],
                 candidate_positions=[r[1] for r in records],success=[r[2] for r in records],
                 nit=[r[3] for r in records],nfev=[r[4] for r in records],provenance=origin,
                 optimization_seconds=time.time()-started)
        print(f'Optimized beta={beta:g}, {len(records)} seeds, V={V:.12g}, {time.time()-started:.1f}s',flush=True)
    with np.load(path) as c:
        pos=c['positions']; origin=str(c['provenance'])
        nseeds=len(c['candidate_V'])
    if (OUT/f'{name}.json').exists():
        row=json.loads((OUT/f'{name}.json').read_text())
    else:
        row=audit_one(pos,beta,name,origin)
    row['n_seeds']=nseeds
    row['original_Fig4_cache']=old
    if old:
        row['cache_Fmax_relative_difference']=row['high']['Fmax']/old['F_max']-1
        row['cache_V_difference']=row['V_high']-old['Vmin']
        row['cache_strongest_agrees']=(row['high']['strongest']=='attractive')==old['F_max_is_att']
    with np.load(CODE/'phase_diagram_cache.npz') as c:
        j=np.flatnonzero(np.abs(c['beta_values']-beta)<1e-13)
        if len(j):
            ratio=float(c['ratio'][list(c['N_values']).index(N),j[0]])
            row['original_Fig5_ratio']=ratio
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
    with np.load(CODE/'melting_analysis_cache_N55.npz') as a, np.load(CODE/'phase_diagram_cache.npz') as b:
        betas=sorted(set(a['betas'].tolist()+b['beta_values'].tolist()))
    if args.betas is not None:
        if not set(args.betas)<=set(betas):ap.error('--betas must be in the original grid')
        betas=args.betas
    rows=[]
    # Four numerical worker processes; each BLAS library is restricted to one thread.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_one,b):b for b in betas}
        for job in as_completed(jobs):
            row=job.result(); rows.append(row)
            (OUT/'scan_results.json').write_text(json.dumps(sorted(rows,key=lambda r:r['beta']),indent=2))
            print(f'DONE {len(rows)}/{len(betas)} beta={row["beta"]:g} cond={row["cond2_high"]:.3e} '
                  f'Fmaxerr={row["Fmax_rel_error"]:.3e} signflips={row["sign_mismatches"]}',flush=True)
    metadata=dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),N=N,
                  formula='Kij=exp(-rij^2/(2*sinh(beta))); f_ij=2*rij*Kij*(K^-1)ij/sinh(beta)^2; f>0 attraction',
                  precisions_decimal=[50,80],beta_values=betas,
                  sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [CODE/'melting_analysis.py',CODE/'phase_diagram.py',CODE/'melting_analysis_cache_N55.npz',CODE/'phase_diagram_cache.npz']},
                  numpy=np.__version__,mpmath=mp.__version__)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
