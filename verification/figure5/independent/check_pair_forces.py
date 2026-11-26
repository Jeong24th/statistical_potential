"""Independent all-pair Cholesky solves and direct finite differences of V_stat.

Coordinates are those selected by the production verification. All computations
here use oscillator units and do not import its kernel or force functions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
PRECISION = HERE.parent / "precision"
COORDINATES = HERE.parent / "search" / "coordinates"
SAMPLE = {(n, t) for n in (2, 4, 7, 23, 54) for t in (2.5, 2.7, 3.)} | {(11, 2.7)}


def evaluate(xy, phi, reference):
    """Positive dV_stat/dr pulls a pair together. Only its two kernel entries vary."""
    mp.mp.dps = 90
    n = len(xy)
    coords = [[mp.mpf(float(v)) for v in row] for row in xy]
    lengths = {(i, j): mp.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
               for i in range(n) for j in range(i)}
    b = mp.sinh(mp.mpf(float(phi)))
    kernel = mp.eye(n)
    for (i, j), r in lengths.items():
        kernel[i, j] = kernel[j, i] = mp.exp(-r*r / (2*b))
    lower = mp.cholesky(kernel)
    # One forward substitution per unit right-hand side. A dot product of
    # columns of L^-1 yields the corresponding element of K^-1.
    inverse_lower = mp.matrix(n)
    for j in range(n):
        inverse_lower[j, j] = 1 / lower[j, j]
        for i in range(j+1, n):
            inverse_lower[i, j] = -mp.fdot((lower[i, k], inverse_lower[k, j])
                                           for k in range(j, i)) / lower[i, i]
    derivatives = {}
    for (i, j), r in lengths.items():
        overlap = mp.fdot((inverse_lower[k, i], inverse_lower[k, j]) for k in range(i, n))
        derivatives[i, j] = (2 / b**2) * overlap * kernel[i, j] * r

    pairs = [min(derivatives, key=derivatives.get)]
    f_rep = -derivatives[pairs[0]]
    positive = [(f, pair) for pair, f in derivatives.items() if f > 0]
    f_att = mp.mpf(0)
    if positive:
        f_att, best = max(positive)
        pairs.append(best)
    ratio = f_att / f_rep
    fd_checks = []
    for i, j in pairs:
        r = lengths[i, j]
        h = max(r, 1) * mp.mpf("1e-28")
        values = []
        for delta in (-h, h):
            perturbed = kernel.copy()
            perturbed[i, j] = perturbed[j, i] = mp.exp(-(r+delta)**2/(2*b))
            # Direct LU determinant, without solving for an inverse or using an
            # analytic derivative. Holding all other distances fixed isolates
            # the distance derivative that defines the canonical pair force.
            values.append(-mp.log(mp.det(perturbed))/b)
        fd = (values[1] - values[0])/(2*h)
        analytic = derivatives[i, j]
        rel_error = abs(fd - analytic)/abs(analytic)
        if rel_error > mp.mpf("1e-22") or mp.sign(fd) != mp.sign(analytic):
            raise ValueError(f"Distance derivative check failed, N={n}, phi={phi}, pair={(i,j)}.")
        fd_checks.append(dict(pair_one_based=[j+1, i+1], force=float(analytic),
                              derivative=float(fd), relative_difference=float(rel_error)))
    measured = (float(f_att), float(f_rep), float(ratio))
    expected = tuple(reference[k] for k in ("max_attractive", "max_repulsive", "ratio"))
    if (not np.allclose(measured, expected, rtol=5e-15, atol=0)
            or len(positive) != reference["n_attractive"]
            or bool(ratio > 1) != (reference["dominance"] == "attractive")):
        raise ValueError(f"Production discrepancy N={n}, phi={phi}: {measured}, {expected}")
    largest_error = max((abs(a-b)/abs(b) if b else 0) for a,b in zip(measured, expected))
    return dict(ratio=float(ratio), max_attractive=float(f_att), max_repulsive=float(f_rep),
                n_attractive=len(positive), n_repulsive=sum(f < 0 for f in derivatives.values()),
                largest_relative_error_vs_recorded_numbers=largest_error,
                direct_distance_derivatives=fd_checks)


def targeted_cases():
    selected = set(SAMPLE)
    for path in (PRECISION/"results").glob("n*_phi*.json"):
        item = json.loads(path.read_text())
        terminal = item["refinement"] or item["fixed"]
        alt = PRECISION/"competing"/path.name
        if alt.is_file():
            terminal = json.loads(alt.read_text())["candidates"][0]["refined"]
        if (item["archived_ratio"] > 1) != (terminal["dps80"]["ratio"] > 1):
            selected.add((item["N"], item["phi"]))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true",
                        help="Check the available subset while other searches are running.")
    parser.add_argument("--output", type=Path, default=HERE/"pair_force_checks.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.output.is_file():
        for item in json.loads(args.output.read_text())["checks"]:
            previous[item["N"],item["phi"],item["seed"],item["coordinates_sha256"]] = item
    checks = []
    missing = []
    for n, phi in targeted_cases():
        tag = f"n{n}_phi{phi:.1f}"
        path = PRECISION/"results"/(tag + ".json")
        if not path.exists():
            missing.append(tag)
            continue
        item = json.loads(path.read_text())
        alt = PRECISION/"competing"/path.name
        if alt.exists():
            finalists = json.loads(alt.read_text())["candidates"]
            candidates = [(row["seed"], row["refined"]["positions"], row["refined"]["dps80"])
                          for row in finalists]
        elif item["refinement"] is not None:
            r = item["refinement"]
            candidates = [(item["winner_seed"], r["positions"], r["dps80"])]
        else:
            with np.load(COORDINATES/(tag + ".npz")) as saved:
                positions = saved["x"]
            candidates = [(item["winner_seed"], positions, item["fixed"]["dps80"])]
        for rank, (seed, coords, reference) in enumerate(candidates, 1):
            fingerprint = hashlib.sha256(np.asarray(coords, dtype="<f8").tobytes()).hexdigest()
            key = n, phi, seed, fingerprint
            row = previous.get(key)
            if row is None:
                row = dict(N=n, phi=phi, seed=seed, coordinates_sha256=fingerprint,
                           **evaluate(coords, phi, reference))
                print(tag, "seed", seed, "R", row["ratio"], flush=True)
            row["candidate_rank"] = rank
            checks.append(row)
    if missing and not args.partial:
        raise FileNotFoundError(f"Missing requested results: {missing}")
    output = dict(decimal_digits=90, relative_distance_step=1e-28,
                  force_unit="hbar omega / a0", pending=missing, checks=checks)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, indent=2)+"\n", encoding="utf8")
    tmp.replace(args.output)
    print(f"{len(checks)} coordinate sets checked; pending: {missing}", flush=True)


if __name__ == "__main__":
    main()
