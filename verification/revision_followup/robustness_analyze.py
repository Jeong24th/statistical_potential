"""Protocol-robustness analysis on stored N=55 Monte Carlo configurations (robust_*.npz in this
directory, produced by robustness_mc.py).  Supplementary Note 10 of the revised manuscript.

Perturbations applied to every stored shot before classification:
  * Gaussian position noise of standard deviation sigma (in a0) on every coordinate,
  * classifier evaluated at a misassigned temperature T*f,
  * one randomly chosen atom removed (N=54), with and without noise.
Observables: P_att(T) = probability that the strongest canonical bond of a shot is attractive,
the mean attractive-bond fraction, the retention of the strongest-bond identity, and the
temperature at which the pooled P_att crosses 1/2.
Outputs: robustness.json, robustness.png
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def params(T):
    phi = 1.0 / T
    return np.sinh(phi), 1.0 / np.cosh(phi / 2.0)


def classify(pos, T):
    """(strongest bond attractive?, attractive fraction, strongest pair)"""
    bphi, _ = params(T)
    N = len(pos)
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * bphi))
    C = np.linalg.inv(K) * K
    iu = np.triu_indices(N, 1)
    c = C[iu]
    mags = np.abs(c) * np.sqrt(d2[iu])
    k = int(np.argmax(mags))
    return c[k] > 0, float(np.mean(c > 0)), (int(iu[0][k]), int(iu[1][k]))


runs = sorted(glob.glob(str(HERE / "robust_*.npz")))
data = [np.load(r) for r in runs]
temps = data[0]["temps"]
assert all(np.allclose(d["temps"], temps) for d in data)
print("runs:", [Path(r).name for r in runs], "samples/run:", [d["pos"].shape[1] for d in data], "partial:", [bool(d["partial"]) for d in data])

SIG = [0.01, 0.03, 0.1, 0.3]
TF = [0.9, 1.1, 0.8, 1.2]
variants = ["base"] + [f"noise{s:g}" for s in SIG] + [f"Tx{f:g}" for f in TF] + ["missed1", "missed1_noise0.03"]
res = {v: np.zeros((len(temps), len(data))) for v in variants}
fatt = {v: np.zeros((len(temps), len(data))) for v in variants}
keep = {f"noise{s:g}": np.zeros((len(temps), len(data))) for s in SIG}
for it, T in enumerate(temps):
    for ir, d in enumerate(data):
        P = d["pos"][it]
        base = [classify(p, T) for p in P]
        res["base"][it, ir] = np.mean([b[0] for b in base]); fatt["base"][it, ir] = np.mean([b[1] for b in base])
        for s in SIG:
            out = [classify(p + rng.normal(scale=s, size=p.shape), T) for p in P]
            res[f"noise{s:g}"][it, ir] = np.mean([o[0] for o in out]); fatt[f"noise{s:g}"][it, ir] = np.mean([o[1] for o in out])
            keep[f"noise{s:g}"][it, ir] = np.mean([o[2] == b[2] for o, b in zip(out, base)])
        for f in TF:
            out = [classify(p, T * f) for p in P]
            res[f"Tx{f:g}"][it, ir] = np.mean([o[0] for o in out]); fatt[f"Tx{f:g}"][it, ir] = np.mean([o[1] for o in out])
        drop = rng.integers(55, size=len(P))
        out = [classify(np.delete(p, k, axis=0), T) for p, k in zip(P, drop)]
        res["missed1"][it, ir] = np.mean([o[0] for o in out]); fatt["missed1"][it, ir] = np.mean([o[1] for o in out])
        out = [classify(np.delete(p, k, axis=0) + rng.normal(scale=0.03, size=(54, 2)), T) for p, k in zip(P, drop)]
        res["missed1_noise0.03"][it, ir] = np.mean([o[0] for o in out]); fatt["missed1_noise0.03"][it, ir] = np.mean([o[1] for o in out])
    print(f"T={T:.2f} base={res['base'][it].mean():.3f} noise0.03={res['noise0.03'][it].mean():.3f} noise0.1={res['noise0.1'][it].mean():.3f} "
          f"noise0.3={res['noise0.3'][it].mean():.3f} Tx0.9={res['Tx0.9'][it].mean():.3f} Tx1.1={res['Tx1.1'][it].mean():.3f} missed={res['missed1'][it].mean():.3f}", flush=True)


def crossing(T, p):
    """temperature at which P crosses 1/2 (linear interpolation on the ascending ladder)"""
    for i in range(len(T) - 1):
        if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
            return float(T[i] + (0.5 - p[i]) * (T[i + 1] - T[i]) / (p[i + 1] - p[i]))
    return float("nan")


summary = {}
for v in variants:
    pooled = res[v].mean(1)
    summary[v] = dict(P_att=pooled.tolist(), run_spread=(res[v].std(1, ddof=1) if res[v].shape[1] > 1 else np.zeros_like(pooled)).tolist(),
                      fatt=fatt[v].mean(1).tolist(), T_cross=crossing(temps, pooled),
                      T_cross_runs=[crossing(temps, res[v][:, r]) for r in range(res[v].shape[1])])
    if v in keep:
        summary[v]["identity_retained"] = keep[v].mean(1).tolist()
    print(f"{v:18s} T_cross={summary[v]['T_cross']:.3f}  P(T={temps[0]})={pooled[0]:.3f}  P(T={temps[-1]})={pooled[-1]:.4f}", flush=True)
json.dump(dict(temps=temps.tolist(), n_runs=len(data), samples_per_run=[int(d["pos"].shape[1]) for d in data], variants=summary),
          open(HERE / "robustness.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
lab = {"base": "unperturbed", "noise0.01": r"$\sigma=0.01\,a_0$", "noise0.03": r"$\sigma=0.03\,a_0$", "noise0.1": r"$\sigma=0.1\,a_0$",
       "noise0.3": r"$\sigma=0.3\,a_0$", "Tx0.9": r"$T$ underestimated by 10%", "Tx1.1": r"$T$ overestimated by 10%", "missed1": "one atom missed"}
for v, st in [("base", "k-o"), ("noise0.01", "-"), ("noise0.03", "-"), ("noise0.1", "-"), ("noise0.3", "-")]:
    ax[0].plot(temps, summary[v]["P_att"], st, ms=3, label=lab[v])
for v, st in [("base", "k-o"), ("Tx0.9", "--"), ("Tx1.1", "--"), ("missed1", ":")]:
    ax[1].plot(temps, summary[v]["P_att"], st, ms=3, label=lab[v])
for a in ax:
    a.axhline(0.5, color="gray", lw=0.8, ls=":"); a.set_xlabel(r"$k_{\rm B}T/\hbar\omega$"); a.set_ylabel("P(strongest bond attractive)")
    a.legend(fontsize=7, frameon=False)
ax[0].set_title("position noise", fontsize=9); ax[1].set_title("temperature error, missed atom", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "robustness.png", dpi=150); fig.savefig(HERE / "robustness.pdf")
print("done")
