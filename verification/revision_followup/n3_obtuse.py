"""N=3 fermions in a 2D isotropic harmonic trap: probability that the three particles form an
obtuse triangle (= probability that one canonical pair force is attractive) versus temperature
(Supplementary Note 10 of the revised manuscript).

The exact weight exp(-beta_phi V_total) is sampled by importance sampling from the Gaussian trap
factor with weight det K.  Units hbar = m = omega = 1, lengths in a0.
Outputs (this directory): n3_obtuse.json, n3_obtuse.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)
NS, CHUNK = 8_000_000, 1_000_000


def params(T):
    phi = 1.0 / T
    return np.sinh(phi), 1.0 / np.cosh(phi / 2.0)


def sample(T, n):
    """points (n,3,2) drawn from the Gaussian trap factor, and weights det K (A^2 at T=0, 1 at T=inf)."""
    if T == 0.0:
        X = rng.normal(scale=np.sqrt(0.5), size=(n, 3, 2))      # |Psi0|^2 ~ A^2 exp(-sum r^2)
        A = 0.5 * np.abs((X[:, 1, 0] - X[:, 0, 0]) * (X[:, 2, 1] - X[:, 0, 1]) - (X[:, 1, 1] - X[:, 0, 1]) * (X[:, 2, 0] - X[:, 0, 0]))
        return X, A ** 2
    if np.isinf(T):
        return rng.normal(scale=1.0, size=(n, 3, 2)), np.ones(n)
    bphi, wphi = params(T)
    X = rng.normal(scale=np.sqrt(1.0 / (bphi * wphi ** 2)), size=(n, 3, 2))
    d12 = ((X[:, 0] - X[:, 1]) ** 2).sum(1); d13 = ((X[:, 0] - X[:, 2]) ** 2).sum(1); d23 = ((X[:, 1] - X[:, 2]) ** 2).sum(1)
    k12, k13, k23 = np.exp(-d12 / (2 * bphi)), np.exp(-d13 / (2 * bphi)), np.exp(-d23 / (2 * bphi))
    return X, 1 + 2 * k12 * k13 * k23 - k12 ** 2 - k13 ** 2 - k23 ** 2


def obtuse_flags(X):
    d12 = ((X[:, 0] - X[:, 1]) ** 2).sum(1); d13 = ((X[:, 0] - X[:, 2]) ** 2).sum(1); d23 = ((X[:, 1] - X[:, 2]) ** 2).sum(1)
    at3 = d12 > d13 + d23          # obtuse angle at particle 3  <=> pair (1,2) attractive
    return (d23 > d12 + d13) | (d13 > d12 + d23) | at3, at3


temps = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
noise_levels = [0.0, 0.05, 0.1, 0.2]
rows = []
for T in temps:
    num = {s: 0.0 for s in noise_levels}; cross = {s: 0.0 for s in noise_levels}
    den = den2 = num3 = 0.0; n_tot = 0
    for _ in range(NS // CHUNK):
        X, w = sample(T, CHUNK)
        for s in noise_levels:
            ob, _ = obtuse_flags(X + rng.normal(scale=s, size=X.shape) if s > 0 else X)
            num[s] += (w * ob).sum(); cross[s] += (w * w * ob).sum()
        num3 += (w * obtuse_flags(X)[1]).sum()
        den += w.sum(); den2 += (w * w).sum(); n_tot += CHUNK
    row = dict(T=("0" if T == 0 else ("inf" if np.isinf(T) else T)))
    for s in noise_levels:
        p = num[s] / den
        var = (cross[s] * (1 - 2 * p) + p * p * den2) / n_tot        # E[w^2 (ob - p)^2]
        key = "P_obtuse" if s == 0 else f"P_obtuse_noise{s:g}"
        row[key] = float(p); row[key + "_se"] = float(np.sqrt(var / n_tot) / (den / n_tot))
    row["P_pair12_attractive"] = float(num3 / den); row["ess"] = float(den ** 2 / den2)
    rows.append(row)
    print(f"T={row['T']!s:5} P_obtuse={row['P_obtuse']:.5f} +- {row['P_obtuse_se']:.5f}  noise0.1 -> {row['P_obtuse_noise0.1']:.5f}  ESS={row['ess']:.3g}", flush=True)
json.dump(rows, open(HERE / "n3_obtuse.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(4.4, 3.0))
fin = [r for r in rows if r["T"] not in ("0", "inf")]
ax.plot([r["T"] for r in fin], [r["P_obtuse"] for r in fin], "o-", color="#CC3311", ms=4, label="exact dual")
ax.plot([r["T"] for r in fin], [r["P_obtuse_noise0.1"] for r in fin], "s--", color="#4477AA", ms=3.5, label=r"with position noise $\sigma=0.1\,a_0$")
ax.axhline(0.75, color="k", ls=":", lw=1, label="independent Gaussian points: 3/4")
ax.axhline(rows[0]["P_obtuse"], color="#CC3311", ls="-.", lw=1, label=r"$T=0$ limit")
ax.set_xscale("log"); ax.set_xlabel(r"$k_{\rm B}T/\hbar\omega$"); ax.set_ylabel("obtuse-triangle probability")
ax.set_ylim(0.3, 0.8); ax.legend(fontsize=7, frameon=False, loc="lower right"); fig.tight_layout()
fig.savefig(HERE / "n3_obtuse.png", dpi=150); fig.savefig(HERE / "n3_obtuse.pdf")
print("done")
