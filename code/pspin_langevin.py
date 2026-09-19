"""Langevin dynamics of the pure spherical p-spin model (p=3).

Produces three figures for the ICLR paper:
  fig_energy_relaxation.pdf : energy density e(t) for several beta, with
                              -E_0, -E_1, -E_inf reference lines
  fig_aging.pdf             : two-time correlation C(t_w, t_w+t), aging
  fig_arrhenius.pdf         : escape time from a deep local minimum vs beta
                              (semilog, Arrhenius fit)

Conventions follow the companion paper: H(sigma) has covariance
E[H(s)H(t)] = N R(s,t)^p on the radius-sqrt(N) sphere,
H(sigma) = N^{-(p-1)/2} sum_{ijk} J_{ijk} s_i s_j s_k,  J iid N(0,1).
Langevin generator: Delta - beta <grad H, grad .>.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style, seq_colors, raw_plus_smooth, OKABE_ITO

apply_style()
rng = np.random.default_rng(0)
FIGDIR = "../figures"

# ---------------------------------------------------------------- theory
def theory_thresholds(p):
    """E_inf, E_0, E_k thresholds of ABC2013 for the pure p-spin model."""
    Einf = 2.0 * np.sqrt((p - 1.0) / p)

    def I1(z):
        # rate function I_{1,p}(-z) for z >= Einf (ABC2013, App. A)
        if z < Einf:
            return 0.0
        return (2.0 / (Einf * Einf)) * quad(
            lambda x: np.sqrt(x * x - Einf * Einf), Einf, z)[0]

    def theta(k, z):
        # Theta_{k,p}(-z) for z >= Einf (ABC2013, Thm 2.5/2.8)
        return 0.5 * np.log(p - 1.0) \
            - (p - 2.0) * z * z / (4.0 * (p - 1.0)) \
            - (k + 1.0) * I1(z)

    E0 = brentq(lambda z: theta(0, z), Einf + 1e-9, 2.0 * Einf)
    E1 = brentq(lambda z: theta(1, z), Einf + 1e-9, 2.0 * Einf)
    return Einf, E0, E1


# ---------------------------------------------------------------- model
def make_tensor(N, seed):
    """Symmetrized coupling tensor; contraction with s^3 equals that of the
    raw iid tensor, and the gradient becomes a single tensordot."""
    r = np.random.default_rng(seed)
    J = r.standard_normal((N, N, N))
    Jsym = (J + J.transpose(0, 2, 1) + J.transpose(1, 0, 2)
            + J.transpose(1, 2, 0) + J.transpose(2, 0, 1)
            + J.transpose(2, 1, 0)) / 6.0
    return Jsym


def energy_grad(J, s, N):
    """H and Euclidean gradient for p=3, H = N^{-1} sum J_ijk s_i s_j s_k."""
    A = J @ s                       # (N,N)
    As = A @ s                      # (N,)
    H = (s @ As) / N
    grad = 3.0 * As / N
    return H, grad


def project(s, N):
    return s * np.sqrt(N) / np.linalg.norm(s)


def langevin(J, s0, beta, dt, n_steps, N, seed, record_every=1,
             record_conf_at=None):
    """Euler discretization of  ds = -beta grad_sphere H dt + sqrt(2) dB."""
    r = np.random.default_rng(seed)
    s = s0.copy()
    energies, times, confs = [], [], {}
    for t in range(n_steps):
        H, g = energy_grad(J, s, N)
        if t % record_every == 0:
            energies.append(H / N)
            times.append(t * dt)
        if record_conf_at is not None and t in record_conf_at:
            confs[t] = s.copy()
        gt = g - (g @ s) * s / N            # tangential gradient
        noise = r.standard_normal(N)
        noise -= (noise @ s) * s / N        # tangential noise
        s = s - beta * gt * dt + np.sqrt(2.0 * dt) * noise
        s = project(s, N)
    return np.array(times), np.array(energies), s, confs


def gradient_descent(J, s0, N, n_steps=4000, lr=0.05):
    s = s0.copy()
    for _ in range(n_steps):
        H, g = energy_grad(J, s, N)
        gt = g - (g @ s) * s / N
        s = project(s - lr * gt, N)
    H, _ = energy_grad(J, s, N)
    return s, H / N


# ------------------------------------------------------- experiment 1+2
def relaxation_and_aging(p_thresholds):
    Einf, E0, E1 = p_thresholds
    N = 150
    J = make_tensor(N, seed=1)
    dt = 0.005
    n_steps = 40000
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    s0 = project(rng.standard_normal(N), N)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(betas), cmap="viridis")
    for i, beta in enumerate(betas):
        t, e, _, _ = langevin(J, s0, beta, dt, n_steps, N,
                              seed=100 + i, record_every=40)
        raw_plus_smooth(ax, t[1:], e[1:], colors[i],
                        rf"$\beta={beta:g}$")
    for val, lsty in [(-Einf, ":"), (-E1, "--"), (-E0, "-.")]:
        ax.axhline(val, color="0.15", ls=lsty, lw=0.9, zorder=1)
    ax.annotate(r"$-E_\infty$", xy=(0.99, -Einf + 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=9, color="0.15")
    ax.annotate(r"$-E_0\approx-E_1$", xy=(0.99, -E0 - 0.012),
                xycoords=ax.get_yaxis_transform(), ha="right",
                va="top", fontsize=9, color="0.15")
    ax.set_xscale("log")
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"energy density $H_{N,p}(\sigma_t)/N$")
    ax.set_ylim(-1.78, 0.05)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    fig.savefig(f"{FIGDIR}/fig_energy_relaxation.pdf")
    plt.close(fig)
    print("saved fig_energy_relaxation.pdf")

    # aging: two-time correlation at beta = 4
    beta = 4.0
    waits = [250, 1000, 4000, 16000]
    record_at = set(waits)
    t, e, _, confs = langevin(J, s0, beta, dt, 40000, N, seed=7,
                              record_every=40, record_conf_at=record_at)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    colors = seq_colors(len(waits), cmap="plasma", hi=0.8)
    for i, tw in enumerate(waits):
        stw = confs[tw]
        # continue dynamics from s(tw) and correlate
        r2 = np.random.default_rng(1000 + tw)
        s = stw.copy()
        Cs, ts = [], []
        n_follow = 12000
        for k in range(n_follow):
            if k % 40 == 0:
                Cs.append((s @ stw) / N)
                ts.append(k * dt)
            H, g = energy_grad(J, s, N)
            gt = g - (g @ s) * s / N
            noise = r2.standard_normal(N)
            noise -= (noise @ s) * s / N
            s = project(s - beta * gt * dt + np.sqrt(2 * dt) * noise, N)
        raw_plus_smooth(ax, ts[1:], Cs[1:], colors[i],
                        rf"$t_w={tw*dt:g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"lag $t$")
    ax.set_ylabel(r"two-time overlap $C(t_w,t_w+t)$")
    ax.legend(loc="lower left", title=r"age of the system",
              title_fontsize=8.5)
    fig.savefig(f"{FIGDIR}/fig_aging.pdf")
    plt.close(fig)
    print("saved fig_aging.pdf")


# --------------------------------------------------------- experiment 3
def arrhenius():
    """Escape time from a deep local minimum vs beta, small N."""
    N = 12
    n_samples = 6
    betas = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    dt = 0.01
    max_steps = 400000
    med_taus = []
    all_taus = []
    for beta in betas:
        taus = []
        for smp in range(n_samples):
            J = make_tensor(N, seed=300 + smp)
            s_start = project(np.random.default_rng(50 + smp)
                              .standard_normal(N), N)
            smin, emin = gradient_descent(J, s_start, N)
            r = np.random.default_rng(int(beta * 1000) + smp)
            s = smin.copy()
            tau = max_steps
            for k in range(max_steps):
                H, g = energy_grad(J, s, N)
                gt = g - (g @ s) * s / N
                noise = r.standard_normal(N)
                noise -= (noise @ s) * s / N
                s = project(s - beta * gt * dt
                            + np.sqrt(2 * dt) * noise, N)
                if (s @ smin) / N < 0.3:          # left the well
                    tau = k
                    break
            taus.append(tau * dt)
        all_taus.append(taus)
        med_taus.append(np.median(taus))
        print(f"beta={beta:4.1f}  median escape time={med_taus[-1]:9.2f}")
    med_taus = np.array(med_taus)
    all_taus = np.array(all_taus)            # (n_beta, n_samples)

    # Arrhenius fit on the clearly activated regime (drop saturated points)
    ok = med_taus < 0.9 * max_steps * dt
    coef = np.polyfit(betas[ok], np.log(med_taus[ok]), 1)
    slope = coef[0]

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    rjit = np.random.default_rng(0)
    for i, beta in enumerate(betas):
        jitter = rjit.uniform(-0.05, 0.05, n_samples)
        ax.semilogy(beta + jitter, all_taus[i], "o", ms=3,
                    color=OKABE_ITO[0], alpha=0.30, zorder=2,
                    markeredgewidth=0)
    q1 = np.percentile(all_taus, 25, axis=1)
    q3 = np.percentile(all_taus, 75, axis=1)
    ax.errorbar(betas, med_taus,
                yerr=[med_taus - q1, q3 - med_taus],
                fmt="o", ms=6, color=OKABE_ITO[0], ecolor=OKABE_ITO[0],
                elinewidth=1.1, capsize=3, zorder=4,
                label="median escape time (IQR)")
    bb = np.linspace(betas.min(), betas.max(), 50)
    ax.semilogy(bb, np.exp(np.polyval(coef, bb)), "-",
                color=OKABE_ITO[3], lw=1.8, zorder=3,
                label=rf"Arrhenius fit $\tau\propto e^{{{slope:.2f}\,\beta}}$")
    ax.set_xlabel(r"inverse temperature $\beta$")
    ax.set_ylabel(r"escape time $\tau$")
    ax.legend(loc="upper left")
    fig.savefig(f"{FIGDIR}/fig_arrhenius.pdf")
    plt.close(fig)
    print(f"saved fig_arrhenius.pdf   (fitted slope {slope:.3f})")
    return slope


if __name__ == "__main__":
    th = theory_thresholds(3)
    print("p=3 thresholds: E_inf=%.5f  E_0=%.5f  E_1=%.5f" % th)
    relaxation_and_aging(th)
    slope = arrhenius()
    np.savetxt("results_summary.txt",
               [th[0], th[1], th[2], slope],
               header="Einf E0 E1 arrhenius_slope")
