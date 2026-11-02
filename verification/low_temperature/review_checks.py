"""Self-stress, other closed shells, and the Gaussian flat-kernel limit.

Units a0=hbar=m=omega=1. Every optimization is a multistart local search,
not a certificate of the global mode. See README for the analytic assumptions.
"""
from audit import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import math


def phase_environment(tracked=minimize):
    source = CODE / 'phase_diagram.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    env = dict(np=np, minimize=tracked, N_SEEDS=1000)
    exec(compile(ast.Module(body=[n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]),
        str(source), 'exec'), env)
    return env


def configure(n, output):
    import audit
    audit.N = n
    audit.PAIRS = np.triu_indices(n, 1)
    audit.set_output(output)
    return audit


def closed_shell_case(n, beta, output, saved_only=False):
    audit = configure(n, output)
    name = f'N{n}_phi_{beta:g}'.replace('.', 'p')
    target = audit.OUT / (name + '_optimization.npz')
    if saved_only:
        reference = json.loads((ROOT/'data/review_reference.json').read_text())
        pos = np.array(reference['closed_shells'][name]['positions'])
    elif not target.exists():
        records = []
        def tracked(*args, **kwargs):
            res = minimize(*args, **kwargs)
            records.append((res.fun, res.x.reshape(n, 2), res.success))
            return res
        phase_environment(tracked)['analyze_pair']((n, beta))
        val, pos, _ = min(records, key=lambda r: r[0])
        np.savez(target, positions=pos, beta=beta, V=val,
                 candidate_V=[r[0] for r in records],
                 candidate_positions=[r[1] for r in records],
                 success=[r[2] for r in records])
    if not saved_only:
        with np.load(target) as saved:
            pos = saved['positions']
    row = audit.audit_one(pos, beta, name,
        'Original Fig.5 optimizer and all 1000 original generic seeds; regenerated coordinates.')
    with np.load(CODE / 'phase_diagram_cache.npz') as cached:
        ni = cached['N_values'].tolist().index(n)
        bi = np.argmin(abs(cached['beta_values'] - beta))
        key = 'ratio' if 'ratio' in cached.files else 'ratios'
        row['original_ratio'] = float(cached[key][ni, bi])
    row.update(N=n, n_seeds=1000)
    if saved_only:
        expected = reference['closed_shells'][name]['summary']
        for key in ('Fmax', 'ratio'):
            assert abs(row['high'][key]/expected['high'][key]-1) < 1e-12, key
        assert abs(row['cond2_high']/expected['cond2_high']-1) < 1e-12
        assert row['precision_50_80_max_relative'] < 1e-35
        row['saved_reference_passed'] = True
    (audit.OUT / (name + '.json')).write_text(json.dumps(row, indent=2))
    return row


def equilibrium_matrix(pos):
    n = len(pos)
    ii, jj = np.triu_indices(n, 1)
    delta = pos[jj] - pos[ii]
    rr = np.linalg.norm(delta, axis=1)
    ee = delta / rr[:, None]
    A = np.zeros((2*n, len(ii)))
    for k, (i, j) in enumerate(zip(ii, jj)):
        A[2*i:2*i+2, k] = ee[k]
        A[2*j:2*j+2, k] = -ee[k]
    return A, rr


def self_stress(output):
    audit = configure(55, output)
    data = json.loads((ROOT / 'data/configurations.json').read_text())
    rows = []
    for name in ('phi_2_refined', 'phi_3_refined'):
        pos = np.array(data[name]['positions'])
        beta = data[name]['beta']
        h = audit.high_eval(pos, beta, 80, eigen=False)
        kappa = 1 / mp.cosh(mp.mpf(beta)/2)**2
        # Do not sum rounded enormous pair forces to obtain the small net force.
        F = np.array([float(kappa*mp.mpf(float(x))-g)
                      for x, g in zip(pos.ravel(), h['grad'])])
        A, rr = equilibrium_matrix(pos)
        fmin, _, rank, _ = np.linalg.lstsq(A, F, rcond=None)
        can2 = mp.fsum(f*f for f in h['f'])
        min2 = mp.fsum(mp.mpf(float(f))**2 for f in fmin)
        z_dot_fmin = mp.fsum((f-mp.mpf(float(v)))*mp.mpf(float(v))
                             for f, v in zip(h['f'], fmin))
        row = dict(name=name, beta=beta, rank=int(rank), nullity=A.shape[1]-int(rank),
            norm_canonical=float(mp.sqrt(can2)), norm_minimum=float(mp.sqrt(min2)),
            norm_ratio=float(mp.sqrt(min2/can2)), minimum_norm_squared_fraction=float(min2/can2),
            self_stress_fraction='1 - ' + mp.nstr(min2/can2, 16),
            min_pair_max=float(max(fmin)),
            stationarity=float(max(mp.sqrt(h['grad'][2*i]**2+h['grad'][2*i+1]**2) for i in range(55))),
            minimum_vs_stationary_max=float(max(abs(fmin+float(kappa)*rr/55))),
            net_force_residual=float(max(abs(A@fmin-F))),
            relative_orthogonality=float(abs(z_dot_fmin)/mp.sqrt(can2*min2)))
        rows.append(row)
    (audit.OUT / 'self_stress.json').write_text(json.dumps(rows, indent=2))
    return rows


def oscillator_values(x, q):
    v = np.empty((len(x), q+2))
    v[:, 0] = np.pi**(-0.25)*np.exp(-x*x/2)
    v[:, 1] = np.sqrt(2)*x*v[:, 0]
    for k in range(1, q+1):
        v[:, k+1] = np.sqrt(2/(k+1))*x*v[:, k]-np.sqrt(k/(k+1))*v[:, k-1]
    d = np.empty((len(x), q+1))
    for k in range(q+1):
        d[:, k] = (np.sqrt(k/2)*v[:, k-1] if k else 0)-np.sqrt((k+1)/2)*v[:, k+1]
    return v, d


def ground_objective(vec, q):
    pos = vec.reshape(-1, 2)
    states = [(a, k-a) for k in range(q+1) for a in range(k+1)]
    aa, bb = np.array(states).T
    vx, dx = oscillator_values(pos[:, 0], q)
    vy, dy = oscillator_values(pos[:, 1], q)
    M = vx[:, aa]*vy[:, bb]
    sign, ld = np.linalg.slogdet(M)
    if sign == 0:
        return 1e100, np.zeros_like(vec)
    invT = np.linalg.inv(M).T
    grad = -2*np.column_stack((np.sum(invT*dx[:, aa]*vy[:, bb], axis=1),
                              np.sum(invT*vx[:, aa]*dy[:, bb], axis=1)))
    return -2*ld, grad.ravel()


def flat_limit_case(n, output, seeds=300, saved_only=False):
    audit = configure(n, output)
    q = (math.isqrt(8*n+1)-3)//2
    assert (q+1)*(q+2)//2 == n
    dest = audit.OUT / f'ground_N{n}.npz'
    if saved_only:
        reference = json.loads((ROOT/'data/review_reference.json').read_text())['flat_limit'][str(n)]
        pos = np.array(reference['positions'])
        obj, grad = ground_objective(pos.ravel(), q)
        gradient = float(max(np.linalg.norm(grad.reshape(n,2), axis=1)))
    elif not dest.exists():
        guesses = phase_environment()['_seeds'](n, seeds)
        if n == 55:
            data = json.loads((ROOT / 'data/configurations.json').read_text())
            guesses.append(np.array(data['phi_3_refined']['positions']).ravel())
        else:
            warm = audit.OUT / f'N{n}_phi_3_optimization.npz'
            if warm.exists():
                with np.load(warm) as data:
                    guesses.append(data['positions'].ravel())
        records = [minimize(ground_objective, guess, args=(q,), jac=True,
            method='L-BFGS-B', options=dict(maxiter=30000, ftol=1e-15, gtol=1e-10))
            for guess in guesses]
        best = min(records, key=lambda r: r.fun)
        # A final restart reduces stopping at the preceding line-search tolerance.
        polished = minimize(ground_objective, best.x, args=(q,), jac=True,
            method='L-BFGS-B', options=dict(maxiter=3000, ftol=1e-15, gtol=1e-10))
        np.savez(dest, positions=polished.x.reshape(n, 2), objective=polished.fun,
                 gradient=polished.jac.reshape(n, 2), candidate_V=[r.fun for r in records],
                 candidate_positions=[r.x.reshape(n, 2) for r in records],
                 n_seeds=seeds, n_warm_starts=len(guesses)-seeds)
    if not saved_only:
        with np.load(dest) as data:
            pos = data['positions']
            gradient = float(max(np.linalg.norm(data['gradient'], axis=1)))
            obj = float(data['objective'])
    mp.mp.dps = 80
    states = [(a, k-a) for k in range(q+1) for a in range(k+1)]
    p = [[mp.mpf(float(x)) for x in row] for row in pos]
    P = mp.matrix([[x**a*y**b for a,b in states] for x,y in p])
    B = P**-1
    W = mp.diag([mp.factorial(a)*mp.factorial(b) if a+b == q else 0 for a,b in states])
    L = B.T*W*B
    ii, jj = np.triu_indices(n, 1)
    lead = [2*mp.sqrt(sum((p[int(i)][d]-p[int(j)][d])**2 for d in (0,1)))*L[int(i),int(j)]
            for i,j in zip(ii,jj)]
    annihilation = max(abs(mp.fsum(L[i,j]*v[j] for j in range(n)))
        for i in range(n) for v in ([mp.mpf(1)]*n, [p[j][0] for j in range(n)], [p[j][1] for j in range(n)]))
    errors = []
    for s in (mp.mpf('0.1'), mp.mpf('0.01'), mp.mpf('0.001')):
        K = mp.matrix([[mp.exp(-s/2*sum((p[i][d]-p[j][d])**2 for d in (0,1)))
                       for j in range(n)] for i in range(n)])
        scaled = s**q*(K**-1)
        errors.append(dict(s=float(s), relative_inverse_limit_error=float(mp.mnorm(scaled-L, 'inf')/mp.mnorm(L, 'inf'))))
    row = dict(N=n, q=q, objective=obj, gradient_max=gradient,
        n_generic_seeds=seeds, coordinate_method='Normalized oscillator Slater determinant, multistart local search',
        leading_rank=q+1, leading_force=audit.stats(lead),
        polynomial_design_inverse_residual=float(mp.mnorm(P*B-mp.eye(n), 'inf')),
        leading_self_stress_residual=float(annihilation), limit_checks=errors,
        positions=pos.tolist())
    if saved_only:
        assert abs(row['leading_force']['ratio']/reference['leading_force']['ratio']-1) < 1e-12
        assert row['polynomial_design_inverse_residual'] < 1e-65
        assert row['leading_self_stress_residual'] < 1e-65
        assert gradient < 1e-6
        row['saved_reference_passed'] = True
    (audit.OUT / f'flat_limit_N{n}.json').write_text(json.dumps(row, indent=2))
    np.savez(audit.OUT / f'flat_limit_N{n}.npz', positions=pos, leading_pairs=np.array(lead, dtype=float))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['self-stress', 'closed-shells', 'flat-limit'])
    parser.add_argument('--output', type=Path, default=ROOT/'results/review')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, default=300)
    parser.add_argument('--saved-only', action='store_true',
        help='Use bundled follow-up coordinates, skip multistart searches, and check reference values.')
    args = parser.parse_args()
    configure(55, args.output).verify_inputs()
    if args.mode == 'self-stress':
        print(json.dumps(self_stress(args.output), indent=2))
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = ([pool.submit(closed_shell_case, n, b, args.output, args.saved_only)
                     for n in (28,36,45) for b in (2.5,2.7,3.)]
                    if args.mode == 'closed-shells' else
                    [pool.submit(flat_limit_case, n, args.output, args.seeds, args.saved_only) for n in (28,36,45,55)])
            for job in as_completed(jobs):
                row = job.result()
                rows.append(row)
                (args.output/(args.mode+'.json')).write_text(json.dumps(sorted(rows, key=lambda r: (r['N'],r.get('beta',0))), indent=2))
                print(f"DONE {args.mode}: N={row['N']} beta={row.get('beta', 'limit')}", flush=True)
