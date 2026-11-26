"""Assemble the Fig. 5 plotting grid from archived and verified force ratios."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COLD = (2.5, 2.7, 3.0)
PREVIOUS_SHELLS = (28, 36, 45, 55)


def previous_checks(code):
    data = code / "verification" / "low_temperature" / "data"
    closed = json.loads((data / "review_reference.json").read_text())["closed_shells"]
    full = json.loads((data / "reference_results.json").read_text())
    verified = {}
    for case in closed.values():
        row = case["summary"]
        assert row["n_seeds"] == 1000
        verified[row["N"], row["beta"]] = (
            row["high"]["ratio"], row["original_ratio"], "Supplementary Note 9")
    # Three temperatures only occurred in the Fig. 5 search. Their file names
    # lack the phase prefix, so identify every input by its search provenance.
    for row in full.values():
        if row["provenance"].startswith("Reconstructed original Fig.5 search"):
            assert row["n_seeds"] == 1000
            verified[55, row["beta"]] = (
                row["high"]["ratio"], row["original_Fig5_ratio"], "Supplementary Note 9")
    assert len(verified) == 22
    return verified


def assemble(code, summary, destination):
    with np.load(code / "phase_diagram_cache.npz") as cache:
        numbers, phis, ratios = (cache[k] for k in ("N_values", "beta_values", "ratio"))
        assert int(cache["n_seeds"]) == 1000 and ratios.shape == (54, 13)
    checked = previous_checks(code)
    prior_keys = set(checked)
    with summary.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["N"]), float(row["phi"])
            assert key not in checked
            checked[key] = (float(row["final_ratio"]), float(row["archived_ratio"]),
                            row["source"])
    expected_new = {(int(n), p) for n in numbers if n not in PREVIOUS_SHELLS for p in COLD}
    assert checked.keys() - prior_keys == expected_new
    assert {(int(n), p) for n in numbers for p in COLD} <= checked.keys()

    rows = []
    for i, number in enumerate(numbers):
        # The former figure followed only the contiguous attractive interval
        # starting at the coldest sampled temperature (largest phi).
        old_pink = np.logical_and.accumulate((ratios[i] > 1)[::-1])[::-1]
        for j, phi in enumerate(phis):
            key = int(number), float(phi)
            archived = float(ratios[i, j])
            value, old, source = checked.get(key, (archived, archived, "archived binary64"))
            assert np.isfinite(value) and value >= 0
            assert np.isclose(archived, old, atol=1e-14, rtol=1e-13), key
            rows.append(dict(N=key[0], phi=key[1], reduced_temperature=1/key[1],
                             ratio=value, archived_ratio=archived,
                             dominance="attractive" if value > 1 else "repulsive",
                             previous_figure="attractive" if old_pink[j] else "repulsive",
                             source=source))
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=Path(os.environ.get(
        "STATPOT_CODE", HERE.parents[1])).resolve())
    parser.add_argument("--summary", type=Path, default=HERE / "precision" / "summary.csv")
    parser.add_argument("--output", type=Path, default=HERE / "figure5_grid.csv")
    args = parser.parse_args()
    assembled = assemble(args.code, args.summary, args.output)
    print(f"{len(assembled)} grid points written to {args.output}")
