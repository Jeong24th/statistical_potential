"""SI figures from the ensemble-MC runs + per-run crossover reproducibility."""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.size": 10,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
})

OUTDIR = r"..\Nature_Comm\figures"


def crossover(Ts, Ps):
    for i in range(len(Ts) - 1):
        if (Ps[i] - 0.5) * (Ps[i + 1] - 0.5) < 0:
            t1, t2, p1, p2 = Ts[i], Ts[i + 1], Ps[i], Ps[i + 1]
            return t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)
    return None


runs55 = [np.load(f) for f in sorted(glob.glob("out/N55_run*.npz"))]
runs6 = [np.load(f) for f in sorted(glob.glob("out/N6_run*.npz"))]
T55 = runs55[0]["temps"]
T6 = runs6[0]["temps"]

# per-run and pooled P(strongest attractive)
P_per_run = np.array([[np.mean(r["obs"][i, :, 4] > 0) for i in range(len(T55))]
                      for r in runs55])
P_pool = P_per_run.mean(axis=0)
Tc_runs = [crossover(T55, P_per_run[k]) for k in range(len(runs55))]
Tc_pool = crossover(T55, P_pool)
print("per-run Tc:", [f"{t:.4f}" for t in Tc_runs])
print(f"pooled Tc = {Tc_pool:.4f},  run spread (max-min) = "
      f"{max(Tc_runs)-min(Tc_runs):.4f},  std = {np.std(Tc_runs):.4f}")

S = json.load(open("out/N55_summary.json"))
err = np.array([r["p_strong_att_err"] for r in S["per_T"]])

P6 = np.array([[np.mean(r["obs"][i, :, 4] > 0) for i in range(len(T6))]
               for r in runs6]).mean(axis=0)
fatt55 = np.array([np.concatenate([r["obs"][i, :, 3] for r in runs55]).mean()
                   for i in range(len(T55))])
fatt6 = np.array([np.concatenate([r["obs"][i, :, 3] for r in runs6]).mean()
                  for i in range(len(T6))])

# ---------------- figure 1: ensemble crossover ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for k in range(len(runs55)):
    ax[0].plot(T55, P_per_run[k], "o", ms=2.5, color="#CC0000", alpha=0.35)
ax[0].errorbar(T55, P_pool, yerr=err, color="#CC0000", lw=1.4, marker="o",
               ms=3.5, capsize=2, label=r"$N=55$")
ax[0].plot(T6, P6, "s-", color="#2255CC", ms=3.5, lw=1.2, label=r"$N=6$")
ax[0].axhline(0.5, ls="--", lw=0.8, c="gray")
ax[0].axvline(Tc_pool, ls=":", lw=1.0, c="k")
ax[0].annotate(rf"$k_{{\rm B}}T_\times/\hbar\omega={Tc_pool:.2f}$",
               xy=(Tc_pool, 0.52), xytext=(0.95, 0.62), fontsize=9,
               arrowprops=dict(arrowstyle="->", lw=0.7))
ax[0].set_xlabel(r"$k_{\rm B}T/\hbar\omega$")
ax[0].set_ylabel(r"$P(\mathrm{strongest\ bond\ attractive})$")
ax[0].set_xlim(0.3, 2.05)
ax[0].legend(frameon=False, fontsize=9)
ax[0].text(0.02, 0.93, "(a)", transform=ax[0].transAxes)

ax[1].plot(T55, fatt55, "o-", color="#CC0000", ms=3.5, lw=1.2, label=r"$N=55$")
ax[1].plot(T6, fatt6, "s-", color="#2255CC", ms=3.5, lw=1.2, label=r"$N=6$")
ax[1].set_xlabel(r"$k_{\rm B}T/\hbar\omega$")
ax[1].set_ylabel(r"$\langle$attractive-bond fraction$\rangle$")
ax[1].set_xlim(0.3, 2.05)
ax[1].set_ylim(0.3, 0.55)
ax[1].legend(frameon=False, fontsize=9)
ax[1].text(0.02, 0.93, "(b)", transform=ax[1].transAxes)
fig.tight_layout()
fig.savefig(OUTDIR + r"\fig_SM_mc_crossover.pdf", bbox_inches="tight")
print("wrote fig_SM_mc_crossover.pdf")

# ---------------- figure 2: radial marginals (unimodal, nearly T-independent) --
def kde(x, npts=300, bw_scale=1.0):
    x = np.asarray(x, float)
    grid = np.linspace(0, max(1.2, x.max() * 1.05), npts)
    h = bw_scale * 1.06 * np.std(x) * len(x) ** (-0.2)
    d = np.zeros(npts)
    for k in range(0, len(x), 20000):
        xs = x[k:k + 20000]
        d += np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / h) ** 2).sum(axis=1)
    return grid, d / (len(x) * h * np.sqrt(2 * np.pi))


show_T = [0.40, 0.62, 1.00, 1.50, 2.00]
colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(show_T)))
fig2, ax2 = plt.subplots(1, 2, figsize=(7.0, 2.9))
for col, Tv in zip(colors, show_T):
    i = int(np.argmin(np.abs(T55 - Tv)))
    r1 = np.concatenate([r["obs"][i, :, 0] for r in runs55])
    r3 = np.concatenate([r["obs"][i, :, 2] for r in runs55])
    g, d = kde(r1)
    ax2[0].plot(g, d, color=col, lw=1.2, label=rf"$T={T55[i]:.2f}$")
    g, d = kde(r3)
    ax2[1].plot(g, d, color=col, lw=1.2)
# mode-level (argmin) markers
for x, lab in ((0.0, "G"), (0.30, "I"), (0.43, "P")):
    ax2[0].axvline(x, ls=":", lw=0.8, c="gray")
    ax2[0].text(x + 0.01, ax2[0].get_ylim()[1] * 0.92, lab, fontsize=8, c="gray")
for x, lab in ((0.43, "P"), (0.70, "I/G")):
    ax2[1].axvline(x, ls=":", lw=0.8, c="gray")
    ax2[1].text(x + 0.01, ax2[1].get_ylim()[1] * 0.92, lab, fontsize=8, c="gray")
ax2[0].set_xlabel(r"$r_{\min}/a_0$")
ax2[0].set_ylabel(r"probability density")
ax2[0].set_xlim(0, 0.9)
ax2[0].legend(frameon=False, fontsize=8)
ax2[0].text(0.02, 0.93, "(a)", transform=ax2[0].transAxes)
ax2[1].set_xlabel(r"$r_{3}/a_0$ (third-smallest radius)")
ax2[1].set_xlim(0.3, 1.15)
ax2[1].text(0.02, 0.93, "(b)", transform=ax2[1].transAxes)
fig2.tight_layout()
fig2.savefig(OUTDIR + r"\fig_SM_mc_marginals.pdf", bbox_inches="tight")
print("wrote fig_SM_mc_marginals.pdf")
