"""Replica-exchange Monte Carlo for N=55 (same model, ladder and sampler as mc_ensemble/mc_ensemble.py)
that STORES thinned configurations, for the protocol-robustness analysis of Supplementary Note 8.

usage: python robustness_mc.py <run_id> <init: random|crystal|mixed> <seed> [burn=3000] [sweeps=6000] [thin=10]
output: robust_<run_id>.npz in this directory (pos: (n_temps, n_kept, 55, 2); temps; lw)
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("mc_ensemble", ROOT / "mc_ensemble" / "mc_ensemble.py")
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)

run_id, init, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
burn = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
sweeps = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
thin = int(sys.argv[6]) if len(sys.argv) > 6 else 10
temps = [0.4, 0.44, 0.48, 0.52, 0.55, 0.58, 0.6, 0.62, 0.64, 0.67, 0.7, 0.75, 0.8, 0.9, 1.0,
         1.15, 1.3, 1.4, 1.45, 1.5, 1.55, 1.6, 1.7, 1.85, 2.0]
S = mc.PTSampler(55, temps, seed, init=init)
t0 = time.time()
for s in range(burn):
    S.sweep(adapt=True)
    S.swap_round(s % 2)
    if s % 250 == 0:
        print(f"[{run_id}] burn {s}/{burn} t={time.time()-t0:.0f}s", flush=True)
S.acc[:] = 0; S.prop[:] = 0; S.swap_acc[:] = 0; S.swap_try[:] = 0
nkeep = sweeps // thin
pos = np.zeros((S.M, nkeep, 55, 2)); lw = np.zeros((S.M, nkeep)); kept = 0


def save(partial):
    np.savez_compressed(HERE / f"robust_{run_id}.npz", temps=np.array(temps), pos=pos[:, :kept], lw=lw[:, :kept],
                        acc=S.acc / np.maximum(S.prop, 1), swap=S.swap_acc / np.maximum(S.swap_try, 1),
                        partial=partial, burn=burn, sweeps=sweeps, thin=thin, init=init, seed=seed)


for s in range(sweeps):
    S.sweep(adapt=False)
    S.swap_round(s % 2)
    if s % thin == 0 and kept < nkeep:
        for i in range(S.M):
            pos[i, kept] = S.pos[i]; lw[i, kept] = S.L[i]
        kept += 1
    if s % 250 == 0:
        print(f"[{run_id}] sample {s}/{sweeps} t={time.time()-t0:.0f}s", flush=True)
        save(True)
save(False)
print(f"[{run_id}] DONE kept={kept} t={time.time()-t0:.0f}s badlogdet={S.badlogdet}", flush=True)
