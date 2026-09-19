"""Langevin dynamics of the pure spherical p-spin model at p=4 (even p:
the regime of Theorems 1-3).

Produces:
  fig_energy_relaxation.pdf : energy density e(t) for several beta (p=4)
  fig_aging.pdf             : two-time correlation C(t_w,t_w+t), aging (p=4)
  fig_arrhenius.pdf         : escape time vs beta for N = 8, 12, 16 with
                              bootstrap CIs on the Arrhenius slopes and the
                              asymptotic reference N(E_0-E_1); censored
                              medians marked as lower bounds.

Conventions as in pspin_langevin.py: E[H(s)H(t)] = N R(s,t)^p on the
radius-sqrt(N) sphere; Langevin generator Delta - beta <grad H, grad .>.
Run stages separately:  python3 pspin_p4.py relax|aging|escape
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style, seq_colors, raw_plus_smooth, OKABE_ITO
from pspin_langevin import theory_thresholds

apply_style()
FIGDIR = "../figures"
P = 4
EINF, E0, E1 = theory_thresholds(P)


# ---------------------------------------------------------------- model
def make_tensor4(N, seed):
    """Symmetrized rank-4 coupling tensor (p=4)."""
    r = np.random.default_rng(seed)
    J = r.standard_normal((N,) * 4)
    Jsym = np.zeros_like(J)
    from itertools import permutations
    for perm in permutations(range(4)):
        Jsym += J.transpose(perm)
    Jsym /= 24.0
    return Jsym


def energy_grad4(J, s, N):
    """H and Euclidean gradient, H = N^{-3/2} sum J_ijkl s_i s_j s_k s_l."""
    A = (J.reshape(N ** 3, N) @ s).reshape(N, N, N)
    B = (A.reshape(N ** 2, N) @ s).reshape(N, N)
    Bs = B @ s
    H = (s @ Bs) / N ** 1.5
    grad = 4.0 * Bs / N ** 1.5
    return H, grad


def project(s, N):
    return s * np.sqrt(N) / np.linalg.norm(s)


def langevin_step(J, s, beta, dt, N, r):
    H, g = energy_grad4(J, s, N)
    gt = g - (g @ s) * s / N
    noise = r.standard_normal(N)
    noise -= (noise @ s) * s / N
    return project(s - beta * gt * dt + np.sqrt(2.0 * dt) * noise, N), H


def gradient_descent4(J, s0, N, n_steps=4000, lr=0.05):
    s = s0.copy()
    for _ in range(n_steps):
        H, g = energy_grad4(J, s, N)
        gt = g - (g @ s) * s / N
        s = project(s - lr * gt, N)
    H, _ = energy_grad4(J, s, N)
    return s, H / N


# ------------------------------------------------------- relaxation
def relaxation():
    N = 60
    J = make_tensor4(N, seed=1)
    dt = 0.005
    n_steps = 30000
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    rng = np.random.default_rng(0)
    s0 = project(rng.standard_normal(N), N)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(betas), cmap="viridis")
    for i, beta in enumerate(betas):
        r = np.random.default_rng(100 + i)
        s = s0.copy()
        ts, es = [], []
        for k in range(n_steps):
            s, H = langevin_step(J, s, beta, dt, N, r)
            if k % 30 == 0:
                ts.append(k * dt)
                es.append(H / N)
        raw_plus_smooth(ax, ts[1:], es[1:], colors[i], rf"$\beta={beta:g}$")
        print(f"relax beta={beta:g} final e={es[-1]:+.4f}", flush=True)
    for val, lsty in [(-EINF, ":"), (-E1, "--"), (-E0, "-.")]:
        ax.axhline(val, color="0.15", ls=lsty, lw=0.9, zorder=1)
    ax.annotate(r"$-E_\infty$", xy=(0.99, -EINF + 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=9, color="0.15")
    ax.annotate(r"$-E_0\approx-E_1$", xy=(0.99, -E0 - 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=9, color="0.15")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"energy density $H_{N,p}(\sigma_t)/N$")
    ax.set_ylim(-1.92, 0.05)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    fig.savefig(f"{FIGDIR}/fig_energy_relaxation.pdf")
    plt.close(fig)
    print("saved fig_energy_relaxation.pdf (p=4)", flush=True)


# ----------------------------------------------------------- aging
def aging():
    N = 60
    J = make_tensor4(N, seed=1)
    dt = 0.005
    beta = 4.0
    rng = np.random.default_rng(0)
    s0 = project(rng.standard_normal(N), N)
    waits = [250, 1000, 4000, 16000]
    confs = {}
    r = np.random.default_rng(7)
    s = s0.copy()
    for k in range(max(waits) + 1):
        if k in waits:
            confs[k] = s.copy()
        s, _ = langevin_step(J, s, beta, dt, N, r)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(waits), cmap="plasma", hi=0.8)
    for i, tw in enumerate(waits):
        stw = confs[tw]
        r2 = np.random.default_rng(1000 + tw)
        s = stw.copy()
        Cs, ts = [], []
        for k in range(12000):
            if k % 40 == 0:
                Cs.append((s @ stw) / N)
                ts.append(k * dt)
            s, _ = langevin_step(J, s, beta, dt, N, r2)
        raw_plus_smooth(ax, ts[1:], Cs[1:], colors[i],
                        rf"$t_w={tw * dt:g}$")
        print(f"aging tw={tw} done", flush=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"lag $t$")
    ax.set_ylabel(r"two-time overlap $C(t_w,t_w+t)$")
    ax.legend(loc="lower left", title=r"age of the system",
              title_fontsize=8.5)
    fig.savefig(f"{FIGDIR}/fig_aging.pdf")
    plt.close(fig)
    print("saved fig_aging.pdf (p=4)", flush=True)


# ------------------------ multi-disorder relaxation / noise-avg aging
def relax_multi():
    """Relaxation with three disorder samples per temperature: median
    trace (EMA-smoothed) with min-max band."""
    from plot_style import ema
    N = 60
    dt = 0.005
    n_steps = 30000
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    jseeds = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(betas), cmap="viridis")
    for i, beta in enumerate(betas):
        traces = []
        for js in jseeds:
            J = make_tensor4(N, seed=js)
            r = np.random.default_rng(100 + 10 * i + js)
            s = project(r.standard_normal(N), N)
            es, ts = [], []
            for k in range(n_steps):
                s, H = langevin_step(J, s, beta, dt, N, r)
                if k % 30 == 0:
                    ts.append(k * dt)
                    es.append(H / N)
            traces.append(es)
            print(f"relax beta={beta:g} J-seed={js} "
                  f"final e={es[-1]:+.4f}", flush=True)
        traces = np.array(traces)
        ts = np.array(ts)
        med = np.median(traces, axis=0)
        ax.fill_between(ts[1:], traces.min(axis=0)[1:],
                        traces.max(axis=0)[1:], color=colors[i],
                        alpha=0.12, linewidth=0, zorder=2)
        ax.plot(ts[1:], ema(med, 0.12)[1:], color=colors[i], lw=1.8,
                label=rf"$\beta={beta:g}$", zorder=3)
        np.save(f"relax_p4_beta{beta:g}.npy", traces)
    for val, lsty in [(-EINF, ":"), (-E1, "--"), (-E0, "-.")]:
        ax.axhline(val, color="0.15", ls=lsty, lw=0.9, zorder=1)
    ax.annotate(r"$-E_\infty$", xy=(0.99, -EINF + 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=9, color="0.15")
    ax.annotate(r"$-E_0\approx-E_1$", xy=(0.99, -E0 - 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=9, color="0.15")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"energy density $H_{N,p}(\sigma_t)/N$")
    ax.set_ylim(-1.92, 0.05)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    fig.savefig(f"{FIGDIR}/fig_energy_relaxation.pdf")
    plt.close(fig)
    print("saved fig_energy_relaxation.pdf (p=4, 3 disorder samples)",
          flush=True)


def aging_multi():
    """Aging with three independent thermal histories (same disorder),
    averaged per waiting time.  beta = 2.5 sits moderately below the
    p=4 dynamical transition (beta_d = 1.837), where aging is visible
    on simulation time scales; deep in the frozen phase (beta = 4) the
    system is nearly arrested and C barely decays over the window."""
    from plot_style import ema
    N = 60
    J = make_tensor4(N, seed=1)
    dt = 0.005
    beta = 2.5
    waits = [250, 1000, 4000, 16000]
    n_follow = 24000
    noise_seeds = [7, 8, 9]
    curves = {tw: [] for tw in waits}
    for ns in noise_seeds:
        rng0 = np.random.default_rng(ns)
        s = project(rng0.standard_normal(N), N)
        confs = {}
        r = np.random.default_rng(100 + ns)
        for k in range(max(waits) + 1):
            if k in waits:
                confs[k] = s.copy()
            s, _ = langevin_step(J, s, beta, dt, N, r)
        for tw in waits:
            stw = confs[tw]
            r2 = np.random.default_rng(1000 + 13 * ns + tw)
            s2 = stw.copy()
            Cs, ts = [], []
            for k in range(n_follow):
                if k % 40 == 0:
                    Cs.append((s2 @ stw) / N)
                    ts.append(k * dt)
                s2, _ = langevin_step(J, s2, beta, dt, N, r2)
            curves[tw].append(Cs)
        print(f"aging noise seed {ns} done", flush=True)
    ts = np.array(ts)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(waits), cmap="plasma", hi=0.8)
    for i, tw in enumerate(waits):
        arr = np.array(curves[tw])
        ax.fill_between(ts[1:], arr.min(axis=0)[1:], arr.max(axis=0)[1:],
                        color=colors[i], alpha=0.15, linewidth=0,
                        zorder=2)
        ax.plot(ts[1:], ema(arr.mean(axis=0), 0.2)[1:], color=colors[i],
                lw=1.8, label=rf"$t_w={tw * dt:g}$", zorder=3)
        np.save(f"aging_p4_tw{tw}.npy", arr)
    ax.set_xscale("log")
    ax.set_xlabel(r"lag $t$")
    ax.set_ylabel(r"two-time overlap $C(t_w,t_w+t)$")
    ax.legend(loc="lower left", title=r"age of the system",
              title_fontsize=8.5)
    fig.savefig(f"{FIGDIR}/fig_aging.pdf")
    plt.close(fig)
    print("saved fig_aging.pdf (p=4, 3 thermal histories)", flush=True)


# ------------------------------------------------- escape / N-sweep
def escape():
    Ns = [8, 12, 16]
    n_samples = 15
    n_starts = 10                      # GD starts; keep the deepest minimum
    betas = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    dt = 0.01
    max_steps = 300000
    t_max = max_steps * dt

    results = {}
    for N in Ns:
        taus = np.full((len(betas), n_samples), t_max)
        cens = np.zeros((len(betas), n_samples), dtype=bool)
        anti = np.full((len(betas), n_samples), t_max)
        anti_cens = np.zeros((len(betas), n_samples), dtype=bool)
        for smp in range(n_samples):
            J = make_tensor4(N, seed=300 + smp)
            # deepest of n_starts GD minima: closer to the deep wells of
            # the theorems than a single generic quench
            best_s, best_e = None, np.inf
            for st in range(n_starts):
                s_start = project(np.random.default_rng(50 + 97 * smp + st)
                                  .standard_normal(N), N)
                smin, emin = gradient_descent4(J, s_start, N)
                if emin < best_e:
                    best_s, best_e = smin, emin
            smin = best_s
            for bi, beta in enumerate(betas):
                r = np.random.default_rng(int(beta * 1000) + 7919 * smp)
                s = smin.copy()
                tau = None
                ta = None
                for k in range(max_steps):
                    s, _ = langevin_step(J, s, beta, dt, N, r)
                    R = (s @ smin) / N
                    if tau is None and R < 0.3:
                        tau = k * dt
                    if ta is None and R < -0.5:
                        ta = k * dt
                        break          # reached the antipodal hemisphere
                    if tau is not None and k * dt > 20.0 * tau and ta is None:
                        break          # exit found; cap the transit search
                taus[bi, smp] = tau if tau is not None else t_max
                cens[bi, smp] = tau is None
                anti[bi, smp] = ta if ta is not None else t_max
                anti_cens[bi, smp] = ta is None
            print(f"N={N} sample {smp} done (e_min={best_e:+.4f})",
                  flush=True)
        results[N] = (taus, cens, anti, anti_cens)
        np.savez(f"escape_p4_N{N}.npz", betas=betas, taus=taus, cens=cens,
                 anti=anti, anti_cens=anti_cens, t_max=t_max)
    escape_analyze(results, betas, n_samples, max_steps * dt)


def escape_plot():
    """Fits and figure from the saved escape_p4_N*.npz files."""
    Ns = [8, 12, 16]
    results = {}
    betas = t_max = n_samples = None
    for N in Ns:
        d = np.load(f"escape_p4_N{N}.npz")
        results[N] = (d["taus"], d["cens"], d["anti"], d["anti_cens"])
        betas = d["betas"]
        t_max = float(d["t_max"])
        n_samples = d["taus"].shape[1]
    escape_analyze(results, betas, n_samples, t_max)


def escape_analyze(results, betas, n_samples, t_max):
    Ns = sorted(results.keys())
    # -------- fits with bootstrap CIs, censoring-aware
    def fit_slope(betas_ok, med_ok):
        c = np.polyfit(betas_ok, np.log(med_ok), 1)
        return c

    slopes, slope_ci = {}, {}
    for N in Ns:
        taus, cens, _, _ = results[N]
        frac_cens = cens.mean(axis=1)
        ok = frac_cens < 0.4                     # median well-defined
        med = np.median(taus, axis=1)
        coef = fit_slope(betas[ok], med[ok])
        boots = []
        rb = np.random.default_rng(0)
        for _ in range(1000):
            idx = rb.integers(0, n_samples, n_samples)
            mb = np.median(taus[:, idx], axis=1)
            okb = ok & (mb < 0.9 * t_max)
            if okb.sum() >= 3:
                boots.append(np.polyfit(betas[okb], np.log(mb[okb]), 1)[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        slopes[N] = coef
        slope_ci[N] = (lo, hi)
        print(f"N={N}: slope={coef[0]:.3f}  95% CI [{lo:.3f},{hi:.3f}]  "
              f"censored fractions {frac_cens.round(2)}", flush=True)
        _, _, anti, anti_cens = results[N]
        print(f"N={N}: antipodal-transit censored fractions "
              f"{anti_cens.mean(axis=1).round(2)}", flush=True)

    # ------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, N in enumerate(Ns):
        taus, cens, _, _ = results[N]
        med = np.median(taus, axis=1)
        q1 = np.percentile(taus, 25, axis=1)
        q3 = np.percentile(taus, 75, axis=1)
        c = OKABE_ITO[i]
        okp = cens.mean(axis=1) < 0.4
        ax.errorbar(betas[okp], med[okp],
                    yerr=[med[okp] - q1[okp], q3[okp] - med[okp]],
                    fmt="o", ms=5, color=c, ecolor=c, elinewidth=1.0,
                    capsize=2.5, zorder=4, label=rf"$N={N}$")
        if (~okp).any():
            ax.semilogy(betas[~okp], med[~okp], "^", ms=7, color=c,
                        markerfacecolor="none", zorder=4)
        coef = slopes[N]
        bb = np.linspace(betas.min(), betas.max(), 50)
        ax.semilogy(bb, np.exp(np.polyval(coef, bb)), "-", color=c,
                    lw=1.4, alpha=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"escape time $\tau$")
    ax.legend(loc="upper left",
              title="$p=4$; open triangles: censored", title_fontsize=8)

    # inset: fitted slope vs N with CI + asymptotic reference
    axi = ax.inset_axes([0.62, 0.10, 0.35, 0.38])
    Ns_arr = np.array(Ns, dtype=float)
    sl = np.array([slopes[N][0] for N in Ns])
    err = np.array([[slopes[N][0] - slope_ci[N][0] for N in Ns],
                    [slope_ci[N][1] - slopes[N][0] for N in Ns]])
    axi.errorbar(Ns_arr, sl, yerr=err, fmt="o-", ms=4, lw=1.2,
                 color=OKABE_ITO[3], capsize=2.5)
    axi.plot(Ns_arr, Ns_arr * (E0 - E1), "--", color="0.4", lw=1.0)
    axi.annotate(r"$N(E_0-E_1)$", xy=(0.05, 0.05),
                 xycoords="axes fraction", fontsize=7.5, color="0.35")
    axi.set_xlabel(r"$N$", fontsize=8, labelpad=1)
    axi.set_ylabel(r"slope $c(N)$", fontsize=8, labelpad=1)
    axi.tick_params(labelsize=7)
    fig.savefig(f"{FIGDIR}/fig_arrhenius.pdf")
    plt.close(fig)
    np.savez("escape_p4_summary.npz",
             Ns=Ns_arr, slopes=sl, ci=err, ref=Ns_arr * (E0 - E1))
    print("saved fig_arrhenius.pdf (p=4 N-sweep)", flush=True)


# ---------------------------------------------------- step-size check
def dtcheck():
    """Escape times at N=12, beta=3.0 with halved step size, to check
    discretization bias against the dt=0.01 medians of escape()."""
    N = 12
    beta = 3.0
    n_samples = 15
    n_starts = 10
    out = {}
    for dt, max_steps in [(0.01, 300000), (0.005, 600000)]:
        taus = []
        for smp in range(n_samples):
            J = make_tensor4(N, seed=300 + smp)
            best_s, best_e = None, np.inf
            for st in range(n_starts):
                s_start = project(np.random.default_rng(50 + 97 * smp + st)
                                  .standard_normal(N), N)
                smin, emin = gradient_descent4(J, s_start, N)
                if emin < best_e:
                    best_s, best_e = smin, emin
            r = np.random.default_rng(int(beta * 1000) + 7919 * smp
                                      + int(dt * 1e4))
            s = best_s.copy()
            tau = max_steps * dt
            for k in range(max_steps):
                s, _ = langevin_step(J, s, beta, dt, N, r)
                if (s @ best_s) / N < 0.3:
                    tau = k * dt
                    break
            taus.append(tau)
        taus = np.array(taus)
        out[dt] = taus
        print(f"dt={dt}: median tau={np.median(taus):.2f}  "
              f"IQR=[{np.percentile(taus,25):.2f},"
              f"{np.percentile(taus,75):.2f}]", flush=True)
    np.savez("dtcheck_p4.npz", **{f"dt{k}": v for k, v in out.items()})


if __name__ == "__main__":
    print(f"p={P}: E_inf={EINF:.5f} E_0={E0:.5f} E_1={E1:.5f} "
          f"E0-E1={E0 - E1:.4e}", flush=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("relax", "all"):
        relaxation()
    if stage in ("aging", "all"):
        aging()
    if stage in ("escape", "all"):
        escape()
    if stage == "relax_multi":
        relax_multi()
    if stage == "aging_multi":
        aging_multi()
    if stage == "dtcheck":
        dtcheck()
    if stage == "escape_plot":
        escape_plot()
