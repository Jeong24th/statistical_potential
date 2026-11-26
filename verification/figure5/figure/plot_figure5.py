"""Plot the strongest canonical pair force at every sampled minimum in Fig. 5.

The published grid supplies the ratio, sign and original cache value at each sample.
The plot contains no interpolation between unlike signs. After saving,
read the individual vector marks back from the PDF and check all 702 points.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


CLOSED = (3, 6, 10, 15, 21, 28, 36, 45, 55)
COLD = (2.5, 2.7, 3.0)
PHI = np.array([0.5, 4 / 7, 0.7, 0.8, 1.0, 1.2, 1.5, 1.7,
                2.0, 2.3, 2.5, 2.7, 3.0])
ATTRACTION = "#9d2326"
REPULSION = "#497895"
TIE = "#604776"
# (marker, edge, face, size in points). Grayscale encodes the sign by shape
# and fill, independently of the color. A third mark preserves exact ties.
SYMBOL = {
    1: ("o", ATTRACTION, ATTRACTION, 3.8),
    -1: ("s", REPULSION, "white", 3.0),
    0: ("D", TIE, TIE, 3.3),
}
LABEL = {1: "Attractive", -1: "Repulsive", 0: "Equal magnitudes"}


def read_grid(cache, grid):
    with np.load(cache) as archive:
        ns = archive["N_values"]
        phi = archive["beta_values"]
        ratios = archive["ratio"].astype(float)
    if (not np.array_equal(ns, np.arange(2, 56))
            or phi.shape != PHI.shape
            or not np.allclose(np.sort(phi), PHI, rtol=0, atol=1e-12)
            or ratios.shape != (len(ns), len(phi))):
        raise ValueError("Expected Fig. 5 grid with N=2 through 55 and 13 temperatures.")
    seen = set()
    with grid.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            n, value = int(row["N"]), float(row["phi"])
            matches = np.flatnonzero(np.isclose(phi, value, rtol=0, atol=1e-12))
            if n not in ns or len(matches) != 1:
                raise ValueError(f"Unexpected grid point N={n}, phi={value}.")
            col = int(matches[0])
            if (n, col) in seen or ratios[n - 2, col] != float(row["archived_ratio"]):
                raise ValueError(f"Repeated point or different archive value at N={n}, phi={value}.")
            ratio = float(row["ratio"])
            sign = "attractive" if ratio > 1 else "repulsive" if ratio < 1 else "equal"
            if row["dominance"] != sign:
                raise ValueError(f"Ratio and sign differ at N={n}, phi={value}.")
            ratios[n - 2, col] = ratio
            seen.add((n, col))
    if len(seen) != ratios.size:
        raise ValueError(f"Expected {ratios.size} plotted ratios, found {len(seen)}.")
    if not np.isfinite(ratios).all() or np.any(ratios < 0):
        raise ValueError("A force-magnitude ratio must be finite and nonnegative.")
    order = np.argsort(1.0 / phi)
    return ns, phi[order], ratios[:, order], len(seen)


def draw_figure(ns, phi, ratios, pdf, png):
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 10, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "text.usetex": False,
    })
    figure, ax = plt.subplots(figsize=(5.60, 4.76))
    figure.subplots_adjust(left=.105, right=.975, bottom=.115, top=.916)
    x = 1.0 / phi
    ax.set_xlim(x[0] - .035, x[-1] + .071)
    ax.set_ylim(1.05, 55.95)
    ax.set_xticks(np.arange(.4, 2.01, .2))
    ax.set_yticks(np.arange(5, 56, 5))
    ax.set_xlabel(r"$k_{\mathrm{B}}T/(\hbar\omega)$", fontsize=12)
    ax.set_ylabel(r"$N$", fontsize=12, rotation=0, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=.65, pad=3, labelsize=10)
    for n in CLOSED:
        ax.hlines(n, x[0], x[-1] + .038, colors="#969ba0", linewidth=.45,
                  linestyles=(0, (1.5, 2.2)), zorder=0)
    ax.plot([x[-1] + .048] * len(CLOSED), CLOSED, linestyle="none", marker="D",
            markerfacecolor="white", markeredgecolor="#393939", markersize=4.1,
            markeredgewidth=.8)

    signs = np.sign(ratios - 1)
    # Connect adjacent sampled attractive points only as a visual guide along a row.
    for n, row in zip(ns, signs):
        for i in np.flatnonzero((row[:-1] == 1) & (row[1:] == 1)):
            ax.hlines(n, x[i], x[i + 1], colors="#e4b2b1", linewidth=1.45, zorder=1)
    handles = []
    for sign, (shape, edge, face, size) in SYMBOL.items():
        i, j = np.nonzero(signs == sign)
        if len(i) == 0:
            continue
        ax.scatter(x[j], ns[i], marker=shape, s=size**2,
                   facecolors=face, edgecolors=edge, linewidths=.65, zorder=3)
        handles.append(Line2D([], [], marker=shape, markerfacecolor=face,
                              markeredgecolor=edge, markeredgewidth=.8,
                              markersize=5.0, linestyle="none", label=LABEL[sign]))
    handles.append(Line2D([], [], marker="D", markerfacecolor="white",
                          markeredgecolor="#393939", markeredgewidth=.8,
                          markersize=4.8, linestyle="none", label="Closed shell"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.025),
              ncol=len(handles), frameon=False, fontsize=10, handlelength=1.0,
              handletextpad=.36, columnspacing=1.6, borderaxespad=0)
    figure.savefig(pdf, metadata={"Title": "Strongest canonical pair force at sampled minima"})

    if png:
        png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, dpi=300)

    # Store the nominal positions in PDF points. Matplotlib measures coordinates
    # up from the bottom of the page; MuPDF measures down from the top.
    xy = np.array([(temperature, n) for n in ns for temperature in x])
    points = ax.transData.transform(xy) * (72 / figure.dpi)
    points[:, 1] = figure.get_figheight() * 72 - points[:, 1]
    plt.close(figure)
    return points


def same_color(observed, expected):
    return observed is not None and np.allclose(observed, to_rgb(expected),
                                                rtol=0, atol=.003)


def check_marks(pdf, ns, phi, ratios, points):
    expected = np.sign(ratios.ravel() - 1).astype(int)
    seen = {}
    with fitz.open(pdf) as document:
        if len(document) != 1:
            raise ValueError("Expected a one-page figure.")
        for path in document[0].get_drawings():
            for sign, (_, edge, face, _) in SYMBOL.items():
                if not (same_color(path["color"], edge) and same_color(path["fill"], face)):
                    continue
                rect = path["rect"]
                center = np.array([(rect.x0 + rect.x1)/2, (rect.y0 + rect.y1)/2])
                distances = np.linalg.norm(points - center, axis=1)
                k = int(np.argmin(distances))
                if distances[k] > .12:  # skip the legend above the plotted grid
                    if center[1] < points[:, 1].min() - 3:
                        continue
                    raise ValueError(f"Unexpected mark in plot area at {center}.")
                if k in seen or sign != expected[k]:
                    raise ValueError(f"Wrong or repeated mark at grid index {k}.")
                seen[k] = sign
    if len(seen) != expected.size:
        missing = sorted(set(range(expected.size)) - seen.keys())
        raise ValueError(f"Missing drawn marks at grid indices {missing}.")
    drawn = np.array([seen[k] for k in range(expected.size)]).reshape(ratios.shape)
    cold = np.broadcast_to(np.isin(phi, COLD), drawn.shape)
    cold_open = cold & ~np.isin(ns, CLOSED)[:, None]
    return {
        "drawn_samples_verified": len(seen),
        "attractive": int(np.sum(drawn == 1)),
        "repulsive": int(np.sum(drawn == -1)),
        "equal": int(np.sum(drawn == 0)),
        "cold_samples_verified": int(cold.sum()),
        "cold_attractive": int(np.sum(drawn[cold] == 1)),
        "cold_repulsive": int(np.sum(drawn[cold] == -1)),
        "cold_open_shell_samples_verified": int(cold_open.sum()),
        "cold_open_shell_attractive": int(np.sum(drawn[cold_open] == 1)),
        "cold_open_shell_repulsive": int(np.sum(drawn[cold_open] == -1)),
        "cold_open_shell_equal": int(np.sum(drawn[cold_open] == 0)),
        "sign_digest_sha256_N_then_increasing_T":
            hashlib.sha256(drawn.astype(np.int8).tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    bundle = Path(__file__).resolve().parents[1]
    root = bundle.parents[1]
    parser.add_argument("--cache", type=Path, default=root/"phase_diagram_cache.npz")
    parser.add_argument("--grid", type=Path, default=bundle/"figure5_grid.csv")
    parser.add_argument("--pdf", type=Path, default=root/"figures/fig_phase_diagram.pdf")
    parser.add_argument("--png", type=Path, default=root/"figures/fig_phase_diagram.png")
    parser.add_argument("--audit", type=Path, help="Optional JSON record of vector-mark checks")
    args = parser.parse_args()
    ns, phi, ratios, count = read_grid(args.cache, args.grid)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    points = draw_figure(ns, phi, ratios, args.pdf, args.png)
    summary = check_marks(args.pdf, ns, phi, ratios, points)
    summary["replacement_rows"] = count
    summary["cache_sha256"] = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    summary["overrides_sha256"] = hashlib.sha256(args.grid.read_bytes()).hexdigest()
    summary["pdf_sha256"] = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    record = json.dumps(summary, indent=2)
    print(record)
    if args.audit:
        args.audit.write_text(record + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
