"""N=55 numerical audit. All calculations use a0=hbar=m=omega=1.

Requires numpy, scipy and mpmath. Run reproduce.py for the saved-coordinate audit.
The original scan saved summaries but not coordinates: regenerated configurations
are explicitly distinguished from the original, unsaved figure coordinates.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import ast
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
CODE = ROOT.parents[1]
OUT = Path(os.environ.get('STATPOT_AUDIT_OUTPUT', str(ROOT / 'results'))).resolve()
N = 55
PAIRS = np.triu_indices(N, 1)


def original_functions():
    src = CODE / 'melting_analysis.py'
    tree = ast.parse(src.read_text(encoding='utf-8'))
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef)]
    env = dict(np=np, minimize=minimize, N=N)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(src), 'exec'), env)
    return env


def double_eval(pos, beta):
    # Preserve the operation order used by melting_analysis.py.
    bp = np.sinh(beta) / beta * beta
    wp = 1 / np.cosh(beta / 2)
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    K = np.exp(-d2 / (2 * bp))
    inv = np.linalg.inv(K)
    f = (2 / bp) * np.sqrt(d2[PAIRS]) * (inv * K)[PAIRS] / bp
    grad = wp**2 * pos + 2 / bp**2 * np.einsum('ab,abj->aj', inv*K, diff)
    sign, logdet = np.linalg.slogdet(K)
    return dict(f=f, K=K, inv=inv, grad=grad, cond=float(np.linalg.cond(K)),
                V=float(0.5*wp**2*np.sum(pos**2)-logdet/bp), det_sign=float(sign))


def high_eval(pos, beta, dps=50, eigen=True):
    """Build K afresh at high precision from exact binary64 input coordinates.

    beta is also held at its exact input binary64 value; this isolates evaluation
    error from any change in configuration or parameter. f>0 means attraction.
    """
    mp.mp.dps = dps
    p = [[mp.mpf(float(t)) for t in row] for row in pos]
    b = mp.mpf(float(beta))
    bp = mp.sinh(b)
    wp2 = 1 / mp.cosh(b/2)**2
    d2 = [[sum((p[i][k]-p[j][k])**2 for k in range(2))
           for j in range(N)] for i in range(N)]
    K = mp.matrix([[mp.exp(-d2[i][j]/(2*bp)) for j in range(N)] for i in range(N)])
    inv = K**-1
    C = mp.matrix([[2/bp**2*inv[i,j]*K[i,j] for j in range(N)] for i in range(N)])
    f = [C[int(i),int(j)]*mp.sqrt(d2[i][j]) for i,j in zip(*PAIRS)]
    grad = [wp2*p[i][k] + mp.fsum(C[i,j]*(p[i][k]-p[j][k]) for j in range(N))
            for i in range(N) for k in range(2)]
    # Cholesky both verifies positive definiteness and gives stable logdet.
    L = mp.cholesky(K)
    logdet = 2*mp.fsum(mp.log(L[i,i]) for i in range(N))
    V = wp2/2*sum(x*x for row in p for x in row)-logdet/bp
    residual = mp.mnorm(K*inv-mp.eye(N), 'inf')
    result = dict(f=f, grad=grad, V=V, inverse_residual=residual)
    if eigen:
        eig = mp.eigsy(K, eigvals_only=True)
        result.update(lmin=eig[0], lmax=eig[N-1], cond=eig[N-1]/eig[0])
    return result


def stats(f):
    f = np.asarray(f, dtype=float)
    a = float(np.max(f))
    r = float(-np.min(f))
    imax = int(np.argmax(np.abs(f)))
    return dict(Fmax=float(abs(f[imax])), max_att=a, max_rep=r, ratio=a/r,
                strongest='attractive' if f[imax]>0 else 'repulsive',
                strongest_pair=[int(PAIRS[0][imax]),int(PAIRS[1][imax])],
                n_att=int(np.sum(f>0)), n_rep=int(np.sum(f<0)))


def audit_one(pos, beta, name, provenance):
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    d = double_eval(pos, beta)
    h50 = high_eval(pos, beta, 50, eigen=False)
    h80 = high_eval(pos, beta, 80, eigen=True)
    f80 = np.array([float(t) for t in h80['f']])
    err = np.abs(d['f']-f80)
    scale = np.max(np.abs(f80))
    rel = err/np.abs(f80)
    conv = [abs(x-y)/abs(y) for x,y in zip(h50['f'],h80['f'])]
    grad80 = np.array([float(t) for t in h80['grad']]).reshape(N,2)
    mask = np.signbit(d['f']) != np.signbit(f80)
    row = dict(name=name, beta=float(beta), T=float(1/beta), provenance=provenance,
        double=stats(d['f']), high=stats(f80), cond2_double=d['cond'],
        cond2_high=float(h80['cond']), lambda_min=float(h80['lmin']),
        lambda_max=float(h80['lmax']), kappa_eps=float(h80['cond'])*np.finfo(float).eps,
        Fmax_rel_error=abs(stats(d['f'])['Fmax']/stats(f80)['Fmax']-1),
        pair_max_error_over_Fmax=float(np.max(err)/scale),
        pair_max_relative_error=float(np.max(rel)), pair_median_relative_error=float(np.median(rel)),
        sign_mismatches=int(np.sum(mask)),
        largest_sign_mismatch_over_Fmax=float(np.max(np.abs(f80[mask]))/scale) if np.any(mask) else 0.,
        precision_50_80_max_relative=float(max(conv)),
        inverse_residual_80=float(h80['inverse_residual']),
        gradient_max_double=float(np.max(np.linalg.norm(d['grad'],axis=1))),
        gradient_max_high=float(np.max(np.linalg.norm(grad80,axis=1))),
        gradient_error_max=float(np.max(np.linalg.norm(d['grad']-grad80,axis=1))),
        V_double=d['V'], V_high=float(h80['V']),
        elapsed_seconds=time.time()-start)
    dr = pos[PAIRS[1]]-pos[PAIRS[0]]
    pair_vectors = d['f'][:,None]*dr/np.linalg.norm(dr,axis=1)[:,None]
    net = np.zeros_like(pos)
    np.add.at(net,PAIRS[0],pair_vectors)
    np.add.at(net,PAIRS[1],-pair_vectors)
    trap = pos/np.cosh(beta/2)**2
    exact_net = trap-grad80
    row.update(pair_sum_net_force_error_max=float(np.max(np.linalg.norm(net-exact_net,axis=1))),
               pair_sum_gradient_double_max=float(np.max(np.linalg.norm(trap-net,axis=1))),
               net_force_high_max=float(np.max(np.linalg.norm(exact_net,axis=1))))
    np.savez(OUT/f'{name}_pairs.npz',positions=pos,beta=beta,pair_i=PAIRS[0],pair_j=PAIRS[1],
             f_double=d['f'],f_high=f80,grad_double=d['grad'],grad_high=grad80)
    # Preserve more than binary64 digits for independent reuse.
    (OUT/f'{name}_high_precision.json').write_text(json.dumps(dict(
        dps=80, f=[mp.nstr(t,65) for t in h80['f']],
        grad=[mp.nstr(t,65) for t in h80['grad']],
        cond=mp.nstr(h80['cond'],65), V=mp.nstr(h80['V'],65)),indent=2))
    (OUT/f'{name}.json').write_text(json.dumps(row,indent=2))
    print(f'{name}: kappa={row["cond2_high"]:.6e}, Fmax={row["high"]["Fmax"]:.6e}, flips={row["sign_mismatches"]}',flush=True)
    return row



def set_output(path):
    """Set a generated-output directory without permitting input-data overwrites."""
    global OUT
    target=Path(path).resolve()
    data=(ROOT/'data').resolve()
    if target==data or data in target.parents or target in (ROOT, CODE):
        raise ValueError('Choose a separate generated-output directory.')
    OUT=target
    os.environ['STATPOT_AUDIT_OUTPUT']=str(OUT)
    OUT.mkdir(parents=True,exist_ok=True)
    return OUT


def load_saved_case(name):
    cases=json.loads((ROOT/'data/configurations.json').read_text(encoding='utf-8'))
    case=cases[name]
    pos=np.array(case['positions'],dtype=float)
    if pos.shape!=(N,2) or not np.all(np.isfinite(pos)):
        raise ValueError('Invalid saved N=55 planar configuration.')
    return pos,float(case['beta']),case['provenance']


def verify_inputs():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['data_sha256'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Input-data checksum mismatch: {name}')
    for name,expected in manifest['source_sha256'].items():
        # Git checkouts may normalize LF/CRLF; compare normalized text if necessary.
        content=(CODE/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            if not name.endswith('.py') or hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()!=manifest['source_lf_sha256'][name]:
                raise ValueError(f'Original source/cache checksum mismatch: {name}')
    return manifest
