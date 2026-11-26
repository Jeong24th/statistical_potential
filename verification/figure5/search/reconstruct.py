"""Repeat the Fig. 5 search at the three coldest points of the open and small closed shells.

The original analyze_pair chooses the lowest
objective among all 1,000 starts, in seed order. Save that configuration and
retain all endpoints in the scratch checkpoints for competing-minimum checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from zipfile import BadZipFile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.optimize import OptimizeResult, minimize

HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("STATPOT_CODE", HERE.parents[2])).resolve()
SCRATCH = Path(os.environ.get("STATPOT_SEARCH_SCRATCH", HERE.parent / "scratch/search")).resolve()
sys.path.insert(0, str(CODE / "verification" / "low_temperature"))
from review_checks import phase_environment


def save_arrays(path, **values):
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def reconstruct(case):
    n, phi, archived_ratio = case
    tag = f"n{n}_phi{phi:.1f}"
    target = HERE / "coordinates" / (tag + ".npz")
    checkpoint = SCRATCH / (tag + ".npz")
    chunks = SCRATCH / "chunks" / tag
    chunks.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with np.load(target) as data:
            assert int(data["n_seeds"]) == 1000 and data["x"].shape == (n, 2)
        return tag
    elapsed = 0.0
    values, coordinates, success = [], [], []
    # Older checkpoints used whole-run snapshots. A complete .part may remain
    # when a synced Windows folder refused to replace the preceding snapshot.
    for snapshot in [checkpoint,checkpoint.with_suffix(".part")]:
        if snapshot.exists():
            try:
                with np.load(snapshot) as d:
                    scores,pos,ok = d["objective"],d["positions"],d["success"]
                    if len(scores)>len(values) and pos.shape==(len(scores),n,2):
                        values,coordinates,success = list(scores),list(pos),list(ok)
                        elapsed=float(d["elapsed"])
            except (OSError,ValueError,EOFError,BadZipFile):
                pass
    for chunk in sorted(chunks.glob("*.npz")):
        with np.load(chunk) as d:
            first=int(d["first"])
            if first==len(values):
                values.extend(d["objective"])
                coordinates.extend(d["positions"])
                success.extend(d["success"])
                elapsed=float(d["elapsed"])
    loaded = len(values)
    index = 0
    start = time.monotonic()

    def tracked(*args, **kwargs):
        nonlocal index
        seed = index
        index += 1
        if seed < loaded:
            return OptimizeResult(fun=values[seed], x=coordinates[seed].ravel(),
                                  success=success[seed])
        candidate = minimize(*args, **kwargs)
        values.append(float(candidate.fun))
        coordinates.append(candidate.x.reshape(n, 2).copy())
        success.append(bool(candidate.success))
        if index % 100 == 0:
            first=max(loaded,index-100)
            save_arrays(chunks/f"{index:04d}.npz", first=first,
                        objective=values[first:], positions=coordinates[first:],
                        success=success[first:], elapsed=elapsed+time.monotonic()-start)
        return candidate

    pn, ph, ratio = phase_environment(tracked)["analyze_pair"]((n, phi))
    assert pn == n and ph == phi and index == len(values) == 1000
    winner = int(np.argmin(values))
    assert np.isfinite(values[winner]) and np.isfinite(ratio)
    # All coordinates are fully written before the public winner file is exposed.
    # The chunk files remain intact if the combined file write is interrupted.
    np.savez(checkpoint, objective=values,positions=coordinates,success=success,
             elapsed=elapsed+time.monotonic()-start)
    save_arrays(target, x=coordinates[winner], winner_seed=winner, n_seeds=index,
                N=n, phi=phi, objective=values[winner], reconstructed_ratio=ratio,
                archived_ratio=archived_ratio, successful=int(np.sum(success)),
                winner_success=success[winner], elapsed=elapsed+time.monotonic()-start)
    return tag


def grid(group):
    with np.load(CODE / "phase_diagram_cache.npz") as d:
        assert int(d["n_seeds"]) == 1000
        ns, betas, ratios = d["N_values"], d["beta_values"], d["ratio"]
        closed = {q * (q+1) // 2 for q in range(1,11)}
        chosen={int(n) for n in ns if n not in closed} if group=="open" else {3,6,10,15,21}
        cases = [(int(n),float(phi),float(ratios[i,j]))
                 for i,n in enumerate(ns) if n in chosen
                 for j,phi in enumerate(betas) if phi in (2.5,2.7,3.0)]
    assert len(cases) == (135 if group=="open" else 15)
    # Handle the closest ratios first so the more expensive refinements
    # can overlap the remaining Fig. 5 reconstructions.
    return sorted(cases,key=lambda c:(0 if abs(c[2]-1)<0.05 or c[0] in (4,7)
                                      else 1,c[0],-c[1]))


def verify_saved(cases):
    """Replay the original force calculation on every saved minimum; no search."""
    matched, same_type, max_replay_error = 0, 0, 0.0
    for n, phi, archived in cases:
        tag = f"n{n}_phi{phi:.1f}"
        with np.load(HERE/"coordinates"/(tag+".npz")) as data:
            pos=data["x"]
            winner=int(data["winner_seed"])
            value=float(data["objective"])
            ratio=float(data["reconstructed_ratio"])
            assert pos.shape==(n,2) and np.isfinite(pos).all()
            assert 0<=winner<1000 and int(data["n_seeds"])==1000
            assert int(data["N"])==n and float(data["phi"])==phi
            assert float(data["archived_ratio"])==archived
        # Reuse analyze_pair's pair loop, bypassing only the optimizer and seeds.
        def fixed(*args, **kwargs):
            return OptimizeResult(fun=value,x=pos.ravel())
        env=phase_environment(fixed)
        env["_seeds"]=lambda n,count:[pos.ravel()]
        _,_,replayed=env["analyze_pair"]((n,phi))
        error=abs(replayed-ratio)
        # Different BLAS implementations round the inverse of a cold, poorly
        # conditioned kernel differently. Check the binary64 error on its
        # conditioning scale; the 50/80-digit comparison is run separately.
        separation = pos[:, None, :] - pos[None, :, :]
        distance2 = np.sum(separation * separation, axis=2)
        kernel = np.exp(-distance2 / (2 * (np.sinh(phi) / phi * phi)))
        condition = np.linalg.cond(kernel)
        tolerance = max(1e-14, np.finfo(float).eps * condition * max(1., ratio))
        if (not np.isfinite(condition) or not np.isfinite(replayed)
                or (replayed > 1) != (ratio > 1) or error > tolerance):
            raise ValueError(f"Binary64 ratio replay failed at {tag}: {replayed}, {ratio}")
        max_replay_error=max(error,max_replay_error)
        matched+=int(ratio==archived)
        same_type+=int((ratio>1)==(archived>1))
        checkpoint=SCRATCH/(tag+".npz")
        if checkpoint.exists():
            with np.load(checkpoint) as d:
                assert d["positions"].shape==(1000,n,2)
                scores=d["objective"]
                assert int(np.argmin(scores))==winner and scores[winner]==value
                assert np.array_equal(d["positions"][winner],pos)
    print(json.dumps(dict(points=len(cases),exact_archived_ratios=matched,
                          archived_labels_reproduced=same_type,
                          max_ratio_replay_error=max_replay_error)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--group",choices=("open","small_closed"),default="open")
    parser.add_argument("--verify",action="store_true",help="Check cached ratios and saved winners without a new search.")
    args=parser.parse_args()
    cases=grid(args.group)
    if args.verify:
        verify_saved(cases)
        sys.exit(0)
    (HERE/"coordinates").mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True,exist_ok=True)
    files=["phase_diagram.py","phase_diagram_cache.npz",
           "verification/low_temperature/review_checks.py"]
    meta=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
              scipy=scipy.__version__,seeds_per_point=1000,workers=args.workers,
              source_sha256={f:hashlib.sha256((CODE/f).read_bytes()).hexdigest() for f in files})
    (HERE/(args.group+"_run_metadata.json")).write_text(json.dumps(meta,indent=2),encoding="utf8")
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(reconstruct,c):c for c in cases}
        for i,job in enumerate(as_completed(futures),1):
            print(f"{i}/{len(cases)} finished {job.result()} ({time.monotonic()-start:.0f} s)",flush=True)
