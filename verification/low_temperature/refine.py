"""Local coordinate refinement with 50-digit objective and analytic gradient.

L-BFGS-B stores coordinates in binary64, but K, inverse, potential, and gradient
are evaluated in 50-digit arithmetic on every step. A high-precision constant
is subtracted before returning the potential to avoid objective cancellation.
This checks local stationarity; it does not certify a global minimum.
"""
from audit import *


class AccuratePotential:
    def __init__(self,beta,pos):
        self.beta=mp.mpf(float(beta))
        self.reference=None
        self.cached_key=None
        self.calls=0
        self.start=time.time()
        self.evaluate(pos.ravel())

    def evaluate(self,v):
        key=v.tobytes()
        if key==self.cached_key:return self.cached
        mp.mp.dps=50
        p=[[mp.mpf(float(v[2*i+k])) for k in range(2)] for i in range(N)]
        bp=mp.sinh(self.beta); wp2=1/mp.cosh(self.beta/2)**2
        K=mp.matrix(N)
        for i in range(N):
            K[i,i]=1
            for j in range(i):
                d2=sum((p[i][k]-p[j][k])**2 for k in range(2))
                K[i,j]=K[j,i]=mp.exp(-d2/(2*bp))
        L=mp.cholesky(K)
        val=wp2/2*sum(t*t for row in p for t in row)-2*sum(mp.log(L[i,i]) for i in range(N))/bp
        inv=K**-1
        grad=[[wp2*p[i][k] for k in range(2)] for i in range(N)]
        for i in range(N):
            for j in range(i):
                c=2/bp**2*inv[i,j]*K[i,j]
                for k in range(2):
                    term=c*(p[i][k]-p[j][k])
                    grad[i][k]+=term; grad[j][k]-=term
        g=np.array([[float(t) for t in row] for row in grad]).ravel()
        if self.reference is None:self.reference=val
        result=(float(val-self.reference),g)
        self.calls+=1
        if self.calls%20==0:
            print(f'beta={float(self.beta):g} call={self.calls} dV={result[0]:.6e} '
                  f'grad_max={np.max(np.abs(g)):.3e} elapsed={time.time()-self.start:.1f}s',flush=True)
        self.cached_key=key;self.cached=result
        return result


def refine_one(beta, source="saved"):
    base=f'phi_{beta:.12g}'.replace('.','p')
    if source=='saved':
        pos,_,_=load_saved_case(base)
        old=json.loads((ROOT/'data/reference_results.json').read_text())[base]
    else:
        with np.load(OUT/f'{base}_optimization.npz') as a:pos=a['positions'].copy()
        old=json.loads((OUT/f'{base}.json').read_text())
    fun=AccuratePotential(beta,pos)
    res=minimize(fun.evaluate,pos.ravel(),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=500,maxls=40,ftol=1e-22,gtol=2e-9))
    target=f'{base}_refined'
    np.savez(OUT/f'{target}_optimization.npz',positions=res.x.reshape(N,2),
             beta=beta,success=res.success,message=res.message,nit=res.nit,nfev=res.nfev,
             initial_positions=pos,energy_change=res.fun)
    row=audit_one(res.x.reshape(N,2),beta,target,
                  'Local refinement of reconstructed Fig.4 configuration; objective and gradient at 50 decimal digits, binary64 coordinate storage.')
    row['refinement']=dict(success=bool(res.success),message=res.message,nit=int(res.nit),nfev=int(res.nfev),
        energy_change=float(res.fun),max_coordinate_displacement=float(np.max(np.linalg.norm(pos-res.x.reshape(N,2),axis=1))),
        Fmax_change_relative=row['high']['Fmax']/old['high']['Fmax']-1,
        Fmax_relative_to_original_cache=row['high']['Fmax']/old['original_Fig4_cache']['F_max']-1,
        previous_gradient_max=old['gradient_max_high'])
    before=high_eval(pos,beta,80,eigen=False)['f']
    after=np.load(OUT/f'{target}_pairs.npz')['f_high']
    before=np.array([float(f) for f in before])
    mask=np.signbit(before)!=np.signbit(after)
    row['refinement']['pair_sign_changes']=int(np.sum(mask))
    row['refinement']['largest_changed_pair_over_Fmax']=float(np.max(np.abs(after[mask]))/np.max(np.abs(after))) if mask.any() else 0.
    (OUT/f'{target}.json').write_text(json.dumps(row,indent=2))
    return row


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('beta',type=float,choices=[2.0,2.5,3.0])
    ap.add_argument('--source',choices=['saved','reconstructed'],default='saved')
    ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args()
    verify_inputs()
    OUT=set_output(args.output)
    refine_one(args.beta,args.source)
