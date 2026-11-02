"""Distance between the V_total mode at inverse temperature phi and the zero-temperature mode
(maximum of |Psi_0|^2) for N=55, comparing full configurations after optimal rotation,
reflection and particle assignment (Supplementary Note 3 of the revised manuscript).

Inputs (from verification/low_temperature/data):
  configurations.json  -- reconstructed/refined V_total minima at 22 temperatures
  review_reference.json -- zero-temperature mode candidate (flat_limit['55']['positions'])
Outputs (this directory): mode_vs_phi.json, mode_vs_phi.png
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "low_temperature" / "data"
cfg = json.load(open(DATA / "configurations.json"))
ground = np.array(json.load(open(DATA / "review_reference.json"))["flat_limit"]["55"]["positions"])
SHELLS = [(0, 3), (3, 12), (12, 21), (21, 38), (38, 55)]


def shell_means(X):
    r = np.sort(np.linalg.norm(X, axis=1))
    return np.array([r[a:b].mean() for a, b in SHELLS])


def assign_cost(Z, Y):
    C = ((Z[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(C)
    return C[r, c].sum(), c


def align(X, Y, nth=1440):
    """min over rotations about the trap centre, reflection and permutation of
    sum_a |R x_a - y_{pi(a)}|^2 ; returns (rmsd, max displacement)."""
    best = (np.inf, None)
    for refl in (1, -1):
        Xr = X.copy(); Xr[:, 1] *= refl
        for th in np.linspace(0, 2 * np.pi, nth, endpoint=False):
            c, s = np.cos(th), np.sin(th)
            cost, cidx = assign_cost(Xr @ np.array([[c, -s], [s, c]]).T, Y)
            if cost < best[0]:
                best = (cost, (refl, cidx))
    refl, cidx = best[1]
    Xr = X.copy(); Xr[:, 1] *= refl
    for _ in range(100):                      # Kabsch refinement at fixed assignment
        U, S, Vt = np.linalg.svd(Xr.T @ Y[cidx])
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, d]) @ U.T
        Z = Xr @ R.T
        cost, c2 = assign_cost(Z, Y)
        if np.array_equal(c2, cidx):
            break
        cidx = c2
    disp = np.linalg.norm(Z - Y[cidx], axis=1)
    return float(np.sqrt(cost / len(X))), float(disp.max())


rows = []
for key, v in cfg.items():
    if not key.startswith("phi_") or key.endswith("refined"):
        continue
    use = f"{key}_refined" if f"{key}_refined" in cfg else key
    X = np.array(cfg[use]["positions"])
    rmsd, dmax = align(X, ground)
    sm = shell_means(X) - shell_means(ground)
    r = np.sort(np.linalg.norm(X, axis=1))
    regime = "G" if r[0] < 0.1 else ("I" if r[2] > 0.6 else "P")
    rows.append(dict(phi=float(v["beta"]), source=use, regime=regime, rmsd_full=rmsd, max_disp=dmax,
                     rmsd_shell=float(np.sqrt((sm ** 2).mean())), r_min=float(r[0]), r3=float(r[2])))
    print(f"phi={rows[-1]['phi']:<6} {regime} rmsd_full={rmsd:.5f} max={dmax:.5f} rmsd_shell={rows[-1]['rmsd_shell']:.5f}", flush=True)
rows.sort(key=lambda d: d["phi"])
json.dump(dict(ground_shell_means=shell_means(ground).tolist(), rows=rows), open(HERE / "mode_vs_phi.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(4.6, 3.1))
col = dict(G="#4477AA", I="#228833", P="#CC3311")
for reg, lab in [("G", "G (one central particle)"), ("I", "I (two innermost)"), ("P", "P (triangle)")]:
    pts = [d for d in rows if d["regime"] == reg]
    ax.semilogy([d["phi"] for d in pts], [d["rmsd_full"] for d in pts], "o", color=col[reg], ms=5, label=lab)
pts = [d for d in rows if d["regime"] == "P"]
ax.semilogy([d["phi"] for d in pts], [d["rmsd_shell"] for d in pts], "s", mfc="none", color="k", ms=5, label="shell-averaged radii (P)")
ax.set_xlabel(r"$\varphi=\hbar\omega/k_{\rm B}T$"); ax.set_ylabel(r"distance to $T=0$ mode  [$a_0$]")
ax.legend(fontsize=7.5, frameon=False); fig.tight_layout()
fig.savefig(HERE / "mode_vs_phi.png", dpi=150); fig.savefig(HERE / "mode_vs_phi.pdf")
print("done")
