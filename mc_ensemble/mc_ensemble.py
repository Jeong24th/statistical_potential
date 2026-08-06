"""
Thermal-ensemble Monte Carlo for the exact classical dual of trapped ideal fermions.

Samples the exact configurational distribution
    P(X) ~ exp(-beta_phi V_total(X)),
    V_total = (1/2) omega_phi^2 |X|^2 + V_stat(beta_phi, X),
    V_stat  = -(1/beta_phi) ln det K,   K_ab = exp(-|x_a-x_b|^2 / (2 beta_phi)),
in units hbar = m = omega = 1, with T given in units of hbar*omega/k_B:
    phi = 1/T,  beta_phi = sinh(phi),  omega_phi = 1/cosh(phi/2).
Log-weight used by the sampler (constants dropped):
    L(X) = ln det K(X) - (beta_phi * omega_phi^2 / 2) * sum_a |x_a|^2 .

Method: single-particle Metropolis moves (adaptive step during burn-in only)
+ optional parallel tempering (replica exchange) across the temperature ladder.
Swap acceptance between temperature slots i, j holding configs Xi, Xj:
    ln A = L_i(Xj) + L_j(Xi) - L_i(Xi) - L_j(Xj),
where L_i uses the (beta_phi, omega_phi) of slot i (the potential is T-dependent).

Observables per (thinned) sample:
    r1 <= r2 <= r3 : three smallest radii |x_a| (structural order parameters)
    fatt           : fraction of pairs with canonical coefficient c_ab > 0
                     (attractive), c_ab ~ (K^-1)_ab K_ab
    ssign          : sign of the canonical coefficient of the strongest bond
                     (strength |c_ab| * r_ab), +1 attractive / -1 repulsive
    fmax           : magnitude (2/beta_phi^2) |c_ab| r_ab of the strongest bond
    lw             : log-weight L(X)

Subcommands:
    run       one (PT) run over a temperature ladder -> .npz
    validate  N=2 exact-distribution check + exact all-repulsive check
    analyze   pool runs: autocorrelation/ESS/R-hat, histograms, basin fractions,
              attractive-bond statistics, crossover temperature -> json + png

Conventions cross-checked against PRL_PRE_v1/CODE/vtotal_minimize.py and
Nature_Comm/verification/*.py (attractive <=> (K^-1)_ab K_ab > 0).
"""

import argparse
import glob
import json
import time

import numpy as np


# ----------------------------------------------------------------------
#  model
# ----------------------------------------------------------------------
def params(T):
    phi = 1.0 / T
    return np.sinh(phi), 1.0 / np.cosh(phi / 2.0)


def build_K(pos, bphi):
    d2 = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * bphi))


def logweight(pos, bphi, wphi):
    K = build_K(pos, bphi)
    sign, ld = np.linalg.slogdet(K)
    if sign <= 0 or not np.isfinite(ld):
        return -np.inf
    return ld - 0.5 * bphi * wphi ** 2 * np.sum(pos * pos)


def measure(pos, bphi):
    N = pos.shape[0]
    r = np.sort(np.linalg.norm(pos, axis=1))
    r1 = r[0]
    r2 = r[1] if N > 1 else np.nan
    r3 = r[2] if N > 2 else np.nan
    K = build_K(pos, bphi)
    Kinv = np.linalg.inv(K)
    C = Kinv * K
    iu = np.triu_indices(N, 1)
    c = C[iu]
    d = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2))[iu]
    fatt = float(np.mean(c > 0.0))
    mags = np.abs(c) * d
    k = int(np.argmax(mags))
    ssign = 1.0 if c[k] > 0 else -1.0
    fmax = float((2.0 / bphi ** 2) * mags[k])
    return r1, r2, r3, fatt, ssign, fmax


# ----------------------------------------------------------------------
#  initial configurations
# ----------------------------------------------------------------------
SHELLS = {
    55: [(3, 0.429), (9, 1.123), (9, 1.742), (17, 2.436), (17, 3.332)],
    6: [(1, 0.0), (5, 1.27)],
}


def crystal_seed(N, rng):
    if N in SHELLS:
        shells = SHELLS[N]
    else:  # triangular-number rings fallback
        shells, left, k, r = [], N, 1, 0.0
        while left > 0:
            n = min(k, left)
            shells.append((n, r))
            left -= n
            k += 1
            r += 0.7
    pts = []
    for n_s, r_s in shells:
        off = rng.uniform(0, 2 * np.pi)
        for i in range(n_s):
            ang = 2 * np.pi * i / n_s + off
            pts.append([r_s * np.cos(ang), r_s * np.sin(ang)])
    return np.array(pts) + rng.normal(scale=0.05, size=(N, 2))


def random_seed_cfg(N, rng):
    scale = max(1.0, 0.45 * np.sqrt(N))
    return rng.normal(scale=scale, size=(N, 2))


# ----------------------------------------------------------------------
#  PT sampler
# ----------------------------------------------------------------------
class PTSampler:
    def __init__(self, N, temps, seed, init="random"):
        self.N = N
        self.temps = np.asarray(sorted(temps))          # ascending T
        self.M = len(self.temps)
        self.rng = np.random.default_rng(seed)
        self.bphi = np.empty(self.M)
        self.wphi = np.empty(self.M)
        for i, T in enumerate(self.temps):
            self.bphi[i], self.wphi[i] = params(T)
        self.pos = []
        for i, T in enumerate(self.temps):
            if init == "crystal" or (init == "mixed" and T < 0.7):
                p0 = crystal_seed(N, self.rng)
            else:
                p0 = random_seed_cfg(N, self.rng)
            self.pos.append(p0)
        self.L = np.array([logweight(self.pos[i], self.bphi[i], self.wphi[i])
                           for i in range(self.M)])
        if not np.all(np.isfinite(self.L)):
            raise RuntimeError("non-finite initial log-weight")
        self.sigma = 0.25 * np.sqrt(np.maximum(self.temps, 0.2))
        self.acc = np.zeros(self.M)
        self.prop = np.zeros(self.M)
        self.swap_acc = np.zeros(max(self.M - 1, 1))
        self.swap_try = np.zeros(max(self.M - 1, 1))
        self.badlogdet = 0

    def sweep(self, adapt=False):
        for i in range(self.M):
            pos, bphi, wphi = self.pos[i], self.bphi[i], self.wphi[i]
            for _ in range(self.N):
                a = self.rng.integers(self.N)
                old = pos[a].copy()
                pos[a] = old + self.sigma[i] * self.rng.normal(size=2)
                Lnew = logweight(pos, bphi, wphi)
                if Lnew == -np.inf:
                    self.badlogdet += 1
                dL = Lnew - self.L[i]
                self.prop[i] += 1
                if np.log(self.rng.random()) < dL:
                    self.L[i] = Lnew
                    self.acc[i] += 1
                else:
                    pos[a] = old
            if adapt and self.prop[i] > 0 and self.prop[i] % (5 * self.N) == 0:
                rate = self.acc[i] / self.prop[i]
                self.sigma[i] *= np.exp(0.25 * (rate - 0.35))
                self.sigma[i] = min(max(self.sigma[i], 1e-4), 5.0)

    def swap_round(self, parity):
        for i in range(parity, self.M - 1, 2):
            j = i + 1
            Lij = logweight(self.pos[j], self.bphi[i], self.wphi[i])
            Lji = logweight(self.pos[i], self.bphi[j], self.wphi[j])
            lnA = Lij + Lji - self.L[i] - self.L[j]
            self.swap_try[i] += 1
            if np.log(self.rng.random()) < lnA:
                self.pos[i], self.pos[j] = self.pos[j], self.pos[i]
                self.L[i], self.L[j] = Lij, Lji
                self.swap_acc[i] += 1

    def run(self, burn, sweeps, thin, swap_every, out, run_id, checkpoint_every=4000):
        t0 = time.time()
        for s in range(burn):
            self.sweep(adapt=True)
            if swap_every and s % swap_every == 0:
                self.swap_round(s // swap_every % 2)
            if s % 1000 == 0:
                print(f"[{run_id}] burn {s}/{burn}  t={time.time()-t0:.0f}s", flush=True)
        # freeze step sizes; reset counters for clean sampling statistics
        self.acc[:] = 0
        self.prop[:] = 0
        self.swap_acc[:] = 0
        self.swap_try[:] = 0
        nkeep = sweeps // thin
        obs = np.full((self.M, nkeep, 7), np.nan)
        kept = 0
        for s in range(sweeps):
            self.sweep(adapt=False)
            if swap_every and s % swap_every == 0:
                self.swap_round(s // swap_every % 2)
            if s % thin == 0 and kept < nkeep:
                for i in range(self.M):
                    r1, r2, r3, fatt, ssign, fmax = measure(self.pos[i], self.bphi[i])
                    obs[i, kept] = (r1, r2, r3, fatt, ssign, fmax, self.L[i])
                kept += 1
            if s % 1000 == 0:
                print(f"[{run_id}] sample {s}/{sweeps}  t={time.time()-t0:.0f}s", flush=True)
            if s % checkpoint_every == 0 and s > 0:
                self._save(out, run_id, obs[:, :kept], burn, sweeps, thin, partial=True)
        self._save(out, run_id, obs[:, :kept], burn, sweeps, thin, partial=False)
        print(f"[{run_id}] DONE in {time.time()-t0:.0f}s; kept {kept} samples "
              f"x {self.M} temps; bad-logdet events: {self.badlogdet}", flush=True)

    def _save(self, out, run_id, obs, burn, sweeps, thin, partial):
        np.savez_compressed(
            out,
            run_id=run_id, N=self.N, temps=self.temps, obs=obs,
            burn=burn, sweeps=sweeps, thin=thin,
            sigma=self.sigma,
            acc_rate=np.divide(self.acc, np.maximum(self.prop, 1)),
            swap_rate=np.divide(self.swap_acc, np.maximum(self.swap_try, 1)),
            badlogdet=self.badlogdet, partial=partial,
        )


# ----------------------------------------------------------------------
#  statistics utilities
# ----------------------------------------------------------------------
def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time (Sokal windowing), in units of samples."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or np.std(x) == 0:
        return 1.0
    x = x - x.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf) - 1.0
    for M in range(1, n):
        if M >= c * taus[M]:
            return max(taus[M], 1.0)
    return max(taus[-1], 1.0)


def split_rhat(chains):
    """Gelman-Rubin split-R-hat; chains: list of 1D arrays (equal length not required)."""
    halves = []
    for ch in chains:
        ch = np.asarray(ch, float)
        m = len(ch) // 2
        if m >= 4:
            halves.append(ch[:m])
            halves.append(ch[m:2 * m])
    if len(halves) < 2:
        return np.nan
    n = min(len(h) for h in halves)
    arr = np.array([h[:n] for h in halves])
    W = arr.var(axis=1, ddof=1).mean()
    B = n * arr.mean(axis=1).var(ddof=1)
    if W <= 0:
        return 1.0
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def wilson_interval(p, n, z=1.96):
    """95% Wilson score interval for a proportion with effective sample size n."""
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float(max(centre - half, 0.0)), float(min(centre + half, 1.0))


def _kde(x, bw_scale=1.0, npts=400):
    """Gaussian KDE on a padded grid; Silverman bandwidth times bw_scale."""
    x = np.asarray(x, float)
    lo, hi = float(x.min()), float(x.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, npts)
    h = bw_scale * 1.06 * np.std(x) * max(len(x), 2) ** (-0.2)
    if h <= 0:
        return grid, np.zeros(npts)
    d = np.zeros(npts)
    step = 20000
    for k in range(0, len(x), step):
        xs = x[k:k + step]
        d += np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / h) ** 2).sum(axis=1)
    d /= len(x) * h * np.sqrt(2.0 * np.pi)
    return grid, d


def _sig_modes(grid, d, height_frac=0.05, trough_ratio=0.85):
    """Mode positions: local maxima above height_frac*max, separated by troughs
    at most trough_ratio times the smaller neighbouring peak."""
    if d.max() <= 0:
        return []
    pk = [i for i in range(1, len(d) - 1)
          if d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] > height_frac * d.max()]
    modes = []
    for i in pk:
        if not modes:
            modes.append(i)
            continue
        j = modes[-1]
        trough = d[min(i, j):max(i, j) + 1].min()
        if trough <= trough_ratio * min(d[i], d[j]):
            modes.append(i)
        elif d[i] > d[j]:
            modes[-1] = i
    return [float(grid[i]) for i in modes]


def modality(pooled, chains, bw_scales=(0.8, 1.0, 1.4)):
    """KDE modality: stable across bandwidths AND reproducible across runs."""
    per_bw = []
    for b in bw_scales:
        g, d = _kde(pooled, b)
        per_bw.append(_sig_modes(g, d))
    per_run = []
    for c in chains:
        if len(c) >= 100:
            g, d = _kde(c, 1.0)
            per_run.append(_sig_modes(g, d))
    return {
        "modes_by_bw": per_bw,
        "n_modes": len(per_bw[1]) if len(per_bw) > 1 else len(per_bw[0]),
        "bimodal_stable": bool(all(len(m) >= 2 for m in per_bw)),
        "bimodal_reproducible": bool(per_run and all(len(m) >= 2 for m in per_run)),
        "modes_per_run": per_run,
    }


# ----------------------------------------------------------------------
#  N=2 exact validation
# ----------------------------------------------------------------------
def exact_pair_pdf(T, s_grid):
    """PDF of s=|x1-x2|/sqrt(2) for N=2 (2D harmonic trap), exact dual."""
    bphi, wphi = params(T)
    w = s_grid * np.exp(-0.5 * bphi * wphi ** 2 * s_grid ** 2) * \
        (1.0 - np.exp(-2.0 * s_grid ** 2 / bphi))
    Z = np.trapezoid(w, s_grid)
    return w / Z


def validate(args):
    print("=" * 70)
    print("VALIDATION 1: N=2 exact separation distribution vs MC")
    for T in (0.5, 1.0):
        temps = [T]
        smp = PTSampler(2, temps, seed=args.seed, init="random")
        smp.run(burn=2000, sweeps=40000, thin=4, swap_every=0,
                out=args.out.replace(".npz", f"_val_T{T}.npz"), run_id=f"val-T{T}")
        d = np.load(args.out.replace(".npz", f"_val_T{T}.npz"))
        obs = d["obs"][0]
        # s = |x1-x2|/sqrt(2); reconstruct from r1,r2? not possible -> re-measure:
        # instead rerun quickly capturing separations directly
        rng = np.random.default_rng(args.seed + 1)
        bphi, wphi = params(T)
        pos = random_seed_cfg(2, rng)
        L = logweight(pos, bphi, wphi)
        sig = 0.5 * np.sqrt(T)
        seps = []
        for s in range(46000):
            for _ in range(2):
                a = rng.integers(2)
                old = pos[a].copy()
                pos[a] = old + sig * rng.normal(size=2)
                Ln = logweight(pos, bphi, wphi)
                if np.log(rng.random()) < Ln - L:
                    L = Ln
                else:
                    pos[a] = old
            if s >= 6000 and s % 4 == 0:
                seps.append(np.linalg.norm(pos[0] - pos[1]) / np.sqrt(2.0))
        seps = np.array(seps)
        grid = np.linspace(1e-4, max(6.0, seps.max() * 1.2), 2000)
        pdf = exact_pair_pdf(T, grid)
        cdf = np.cumsum(pdf) * (grid[1] - grid[0])
        cdf /= cdf[-1]
        emp = np.searchsorted(np.sort(seps), grid) / len(seps)
        ks = np.max(np.abs(emp - cdf))
        tau = autocorr_time(seps)
        ess = len(seps) / tau           # autocorr_time returns tau = 1+2*sum(rho): ESS = n/tau
        ks_crit = 1.36 / np.sqrt(ess)   # ~95% band using ESS
        ok = ks < ks_crit
        print(f"  T={T}: KS={ks:.4f}  (95% threshold ~{ks_crit:.4f}, ESS={ess:.0f})"
              f"  -> {'PASS' if ok else 'FAIL'}")
        # exact all-repulsive check for N=2
        fatt_mean = np.nanmean(obs[:, 3])
        print(f"  T={T}: N=2 attractive-pair fraction = {fatt_mean:.6f} (exact: 0)"
              f"  -> {'PASS' if fatt_mean == 0 else 'FAIL'}")
    print("=" * 70)


# ----------------------------------------------------------------------
#  analysis
# ----------------------------------------------------------------------
def classify_phase(r1, r3):
    """G: particle at center; P: 3-particle inner triangle; I: otherwise."""
    if r1 < 0.15:
        return "G"
    if r3 < 0.55:
        return "P"
    return "I"


def analyze(args):
    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files match {args.pattern}")
    runs = [np.load(f) for f in files]
    temps = runs[0]["temps"]
    N = int(runs[0]["N"])
    for r in runs:
        assert np.allclose(r["temps"], temps), "temperature grids differ"
    M = len(temps)
    summary = {"N": N, "files": files, "temps": temps.tolist(), "per_T": []}

    for i in range(M):
        row = {"T": float(temps[i])}
        r1_chains = [r["obs"][i, :, 0] for r in runs]
        r3_chains = [r["obs"][i, :, 2] for r in runs]
        fatt_chains = [r["obs"][i, :, 3] for r in runs]
        sign_chains = [r["obs"][i, :, 4] for r in runs]
        r1_all = np.concatenate(r1_chains)
        r3_all = np.concatenate(r3_chains)
        fatt_all = np.concatenate(fatt_chains)
        sign_all = np.concatenate(sign_chains)

        taus = [autocorr_time(c) for c in r1_chains]
        tau = float(np.mean(taus))
        # autocorr_time returns tau = 1 + 2*sum(rho); in this convention ESS = n/tau
        ess = float(sum(len(c) / t for c, t in zip(r1_chains, taus)))
        row["tau_r1"] = tau
        row["ess_r1"] = ess
        row["rhat_r1"] = split_rhat(r1_chains)
        row["rhat_fatt"] = split_rhat(fatt_chains)
        row["acc_rate"] = float(np.mean([r["acc_rate"][i] for r in runs]))

        row["r1_mean"] = float(r1_all.mean())
        row["r3_mean"] = float(np.nanmean(r3_all))
        row["fatt_mean"] = float(fatt_all.mean())
        row["fatt_per_run"] = [float(np.mean(c)) for c in fatt_chains]
        p_att = float(np.mean(sign_all > 0))
        row["p_strong_att"] = p_att
        row["p_strong_att_per_run"] = [float(np.mean(c > 0)) for c in sign_chains]
        # Wilson 95% interval on the sign chain's own ESS (graceful at p=0 and p=1)
        sign_taus = [autocorr_time(c) for c in sign_chains]
        row["ssign_constant"] = bool(all(np.std(c) == 0 for c in sign_chains))
        ess_sign = float(sum(len(c) / t for c, t in zip(sign_chains, sign_taus)))
        row["ess_ssign"] = ess_sign
        lo, hi = wilson_interval(p_att, max(ess_sign, 4.0))
        row["p_strong_att_ci95"] = [lo, hi]
        row["p_strong_att_err"] = float(0.5 * (hi - lo))

        if N > 2:
            phases = [classify_phase(a, b) for a, b in zip(r1_all, r3_all)]
            for ph in ("G", "I", "P"):
                row[f"frac_{ph}"] = float(np.mean([p == ph for p in phases]))
            per_run_fr = []
            for c1, c3 in zip(r1_chains, r3_chains):
                phs = [classify_phase(a, b) for a, b in zip(c1, c3)]
                per_run_fr.append({ph: float(np.mean([p == ph for p in phs]))
                                   for ph in ("G", "I", "P")})
            row["phase_frac_per_run"] = per_run_fr
            # NOTE: frac_G/I/P are threshold-crossing fractions of instantaneous
            # radii (descriptive only) -- NOT basin occupancies at finite T.
            # Bimodality verdicts come from KDE modality tests on the continuous
            # order parameters: bandwidth sweep = peak stability; per-run KDE
            # agreement = inter-chain reproducibility.
            row["modality_r1"] = modality(r1_all, r1_chains)
            r3_ok = r3_all[~np.isnan(r3_all)]
            r3c_ok = [c[~np.isnan(c)] for c in r3_chains]
            row["modality_r3"] = modality(r3_ok, r3c_ok)
        summary["per_T"].append(row)

    # crossover temperature: P(strongest attractive) = 0.5
    Ts = np.array([r["T"] for r in summary["per_T"]])
    Ps = np.array([r["p_strong_att"] for r in summary["per_T"]])
    Tc = None
    for i in range(len(Ts) - 1):
        if (Ps[i] - 0.5) * (Ps[i + 1] - 0.5) < 0:
            t1, t2, p1, p2 = Ts[i], Ts[i + 1], Ps[i], Ps[i + 1]
            Tc = float(t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1))
            break
    summary["T_crossover_ensemble"] = Tc
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_T"}, indent=1))
    print(f"{'T':>6} {'acc':>5} {'tau':>7} {'ESS':>7} {'Rhat':>6} "
          f"{'r1':>6} {'r3':>6} {'fatt':>6} {'Patt':>6} {'G/I/P':>17}")
    for r in summary["per_T"]:
        gip = (f"{r.get('frac_G', 0):.2f}/{r.get('frac_I', 0):.2f}/"
               f"{r.get('frac_P', 0):.2f}" if N > 2 else "-")
        print(f"{r['T']:>6.3f} {r['acc_rate']:>5.2f} {r['tau_r1']:>7.1f} "
              f"{r['ess_r1']:>7.0f} {r['rhat_r1']:>6.3f} {r['r1_mean']:>6.3f} "
              f"{r['r3_mean']:>6.3f} {r['fatt_mean']:>6.3f} "
              f"{r['p_strong_att']:>6.3f} {gip:>17}")

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # r1 (and r3) histograms
        ncol = 4
        nrow = int(np.ceil(M / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow))
        axes = np.atleast_2d(axes)
        for i in range(M):
            ax = axes[i // ncol][i % ncol]
            for r in runs:
                ax.hist(r["obs"][i, :, 0], bins=60, range=(0, 1.2),
                        histtype="step", density=True)
            ax.set_title(f"T={temps[i]:.3f}", fontsize=9)
            ax.set_xlabel("r_min")
        for j in range(M, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.tight_layout()
        fig.savefig(args.out + "_rmin_hist.png", dpi=140)
        if N > 2:
            fig2, axes2 = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow))
            axes2 = np.atleast_2d(axes2)
            for i in range(M):
                ax = axes2[i // ncol][i % ncol]
                for r in runs:
                    ax.hist(r["obs"][i, :, 2], bins=60, range=(0.2, 1.2),
                            histtype="step", density=True)
                ax.set_title(f"T={temps[i]:.3f}", fontsize=9)
                ax.set_xlabel("r3")
            for j in range(M, nrow * ncol):
                axes2[j // ncol][j % ncol].axis("off")
            fig2.tight_layout()
            fig2.savefig(args.out + "_r3_hist.png", dpi=140)
        fig3, ax3 = plt.subplots(1, 2, figsize=(9, 3.4))
        ax3[0].errorbar(Ts, Ps, yerr=[r["p_strong_att_err"] for r in summary["per_T"]],
                        marker="o")
        ax3[0].axhline(0.5, ls="--", c="gray")
        if Tc:
            ax3[0].axvline(Tc, ls=":", c="crimson")
        ax3[0].set_xlabel("T"); ax3[0].set_ylabel("P(strongest bond attractive)")
        ax3[1].plot(Ts, [r["fatt_mean"] for r in summary["per_T"]], marker="s")
        ax3[1].set_xlabel("T"); ax3[1].set_ylabel("attractive-bond fraction")
        fig3.tight_layout()
        fig3.savefig(args.out + "_bonds.png", dpi=140)
        print("figures written:", args.out + "_*.png")
    except Exception as e:  # matplotlib optional
        print("figure generation skipped:", e)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--N", type=int, required=True)
    r.add_argument("--temps", type=str, required=True)
    r.add_argument("--burn", type=int, default=5000)
    r.add_argument("--sweeps", type=int, default=30000)
    r.add_argument("--thin", type=int, default=10)
    r.add_argument("--swap-every", type=int, default=2)
    r.add_argument("--init", choices=["random", "crystal", "mixed"], default="mixed")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--run-id", type=str, required=True)
    r.add_argument("--out", type=str, required=True)

    v = sub.add_parser("validate")
    v.add_argument("--seed", type=int, default=7)
    v.add_argument("--out", type=str, default="out/validate.npz")

    a = sub.add_parser("analyze")
    a.add_argument("--pattern", type=str, required=True)
    a.add_argument("--out", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        temps = [float(t) for t in args.temps.split(",")]
        smp = PTSampler(args.N, temps, seed=args.seed, init=args.init)
        smp.run(args.burn, args.sweeps, args.thin, args.swap_every,
                args.out, args.run_id)
    elif args.cmd == "validate":
        validate(args)
    elif args.cmd == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
