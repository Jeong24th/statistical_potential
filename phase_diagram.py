"""Plot Fig. 5 and retain the original binary64 multistart search functions.

Running this script plots the verified force ratios at every sampled minimum.
compute_grid() loads or generates the original 1,000-start ratio cache.
"""
import os
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / 'figures'

# ── Scan parameters ───────────────────────────────────────────
N_values = list(range(2, 56))
beta_values = [0.5, 4.0/7.0, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7,
               2.0, 2.3, 2.5, 2.7, 3.0]
N_SEEDS = 1000
MAX_WORKERS = max(1, int(cpu_count() * 0.7))

CACHE_FILE = SCRIPT_DIR / 'phase_diagram_cache.npz'

# ── V_total + analytic gradient (per-process) ────────────────
class VtotalCached:
    def __init__(self, N, bp, wp):
        self.N = N; self.bp = bp; self.wp2 = wp * wp
        self.coeff = 2.0 / (bp * bp); self.inv_2s2 = 1.0 / (2.0 * bp)
        self._ck = None; self._cv = None
    def _c(self, v):
        k = v.tobytes()
        if self._ck == k: return self._cv
        N = self.N; pos = v.reshape(N, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        K = np.exp(-self.inv_2s2 * d2)
        sign, logdet = np.linalg.slogdet(K)
        if sign <= 0 or not np.isfinite(logdet):
            r = (1e100, np.zeros_like(v)); self._cv = r; self._ck = k; return r
        Kinv = np.linalg.inv(K); S = Kinv * K
        val = 0.5 * self.wp2 * np.sum(pos * pos) - logdet / self.bp
        grad = self.wp2 * pos + self.coeff * np.einsum('ab,abj->aj', S, diff)
        r = (val, grad.ravel()); self._cv = r; self._ck = k; return r
    def fun(self, v): return self._c(v)[0]
    def jac(self, v): return self._c(v)[1]

def _seeds(N, n_seeds):
    cfgs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        x0 = np.zeros((N, 2)); idx = 0
        ms = int(np.ceil(np.sqrt(2 * N))); r = 0.0
        for s in range(ms + 1):
            ni = s + 1
            if idx + ni > N: ni = N - idx
            if ni <= 0: break
            if s == 0: x0[idx] = [0, 0]; idx += 1; r = 0.7
            else:
                r += 0.55 + rng.randn() * 0.03
                for k in range(ni):
                    a = 2*np.pi*k/ni + rng.randn()*0.05 + seed*0.3
                    x0[idx] = [r*np.cos(a), r*np.sin(a)]; idx += 1
            if idx >= N: break
        cfgs.append(x0.flatten())
    return cfgs

def analyze_pair(args):
    """Compute ratio = max|F_att| / max|F_rep| for a single (N, beta)."""
    N, beta = args
    phi_p = beta
    beta_phi = np.sinh(phi_p) / phi_p * beta
    omega_phi = 1.0 / np.cosh(phi_p / 2.0)
    sigma2 = beta_phi
    obj = VtotalCached(N, beta_phi, omega_phi)
    best_f, best_x = np.inf, None
    for cfg in _seeds(N, N_SEEDS):
        res = minimize(obj.fun, cfg, jac=obj.jac, method='L-BFGS-B',
                       options={'maxiter': 30000, 'ftol': 1e-15})
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x.reshape(N, 2)
    pc = best_x
    d2_pc = np.sum((pc[:, None, :] - pc[None, :, :])**2, axis=2)
    K = np.exp(-d2_pc / (2.0 * sigma2))
    Kinv = np.linalg.inv(K)
    fmax_att, fmax_rep = 0.0, 0.0
    for a in range(N):
        for b in range(a+1, N):
            coeff = Kinv[a, b] * K[a, b]
            f = (2.0 / sigma2) * (pc[b]-pc[a]) * coeff / beta_phi
            mag = np.linalg.norm(f)
            dr = pc[b] - pc[a]
            dot = np.dot(f, dr / np.linalg.norm(dr))
            if dot > 0:
                fmax_att = max(fmax_att, mag)
            else:
                fmax_rep = max(fmax_rep, mag)
    ratio = fmax_att / fmax_rep if fmax_rep > 0 else 999.0
    return (N, beta, ratio)

# ── Compute (or load from cache) ──────────────────────────────
def compute_grid():
    cached = None
    if os.path.exists(CACHE_FILE):
        cached = np.load(CACHE_FILE)
        cN = list(cached['N_values'])
        cB = list(cached['beta_values'])
        cS = int(cached['n_seeds']) if 'n_seeds' in cached.files else -1
        if cN == N_values and cB == beta_values and cS == N_SEEDS:
            return cached['ratio']
        print(f"Cache parameters differ (cached n_seeds={cS}, current={N_SEEDS}) - recomputing.", flush=True)

    pairs = [(N, b) for N in N_values for b in beta_values]
    print(f"Phase diagram: {len(pairs)} (N, beta) pairs, {N_SEEDS} seeds each", flush=True)
    print(f"Workers: {MAX_WORKERS}", flush=True)
    t0 = time.time()

    ratio = np.zeros((len(N_values), len(beta_values)))
    done = 0
    with Pool(processes=MAX_WORKERS) as pool:
        for N, beta, r in pool.imap_unordered(analyze_pair, pairs, chunksize=1):
            i = N_values.index(N); j = beta_values.index(beta)
            ratio[i, j] = r
            done += 1
            if done % 10 == 0 or done == len(pairs):
                elapsed = time.time() - t0
                rem = elapsed * (len(pairs) - done) / max(done, 1)
                print(f"  {done}/{len(pairs)}  elapsed={elapsed/60:.1f}m  ETA={rem/60:.1f}m",
                      flush=True)

    np.savez(CACHE_FILE,
             N_values=np.array(N_values),
             beta_values=np.array(beta_values),
             n_seeds=np.array(N_SEEDS),
             ratio=ratio)
    print(f"Saved cache to {CACHE_FILE}", flush=True)
    return ratio

if __name__ == '__main__':
    from verification.figure5.figure.plot_figure5 import main

    main()
