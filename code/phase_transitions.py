"""Phase-transition figures for the ICLR paper.

Produces:
  fig_arrest.pdf          : (left) plateau energy vs temperature with the
                            equilibrium curve -1/(2T), the threshold -E_inf,
                            and the exact dynamical temperature T_d;
                            (right) long-time overlap of two noise-clones
                            (ergodicity-breaking order parameter).
  fig_landscape3d.pdf     : an actual p-spin landscape (p=3, N=3) rendered
                            on the sphere.
  fig_mechanism_phase.pdf : FORMAL comparison of the two explicit leading
                            terms beta*b and Lambda_EK of Theorem 3 in the
                            (b, beta) plane.  Not the branch boundary of
                            the theorem: the proved bound compares beta*b
                            with Lambda_EK + C_{p,b}/sqrt(beta) with
                            C_{p,b} unspecified, and holds only for
                            b > b_* in (E_2, E_1) with b_* not certified.
                            The region b <= E_2 is masked accordingly.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm

from plot_style import apply_style, seq_colors, OKABE_ITO
from pspin_langevin import (theory_thresholds, make_tensor, energy_grad,
                            project)

apply_style()
FIGDIR = "../figures"
P = 3
EINF, E0, E1 = theory_thresholds(P)
# exact dynamical temperature of the pure spherical p-spin model
TD = np.sqrt(P * (P - 2) ** (P - 2) / (2.0 * (P - 1) ** (P - 1)))


# --------------------------------------------- quench dynamics sweep
def quench_run(J, N, beta, dt, n_steps, seed, record_every=20):
    """One Langevin trajectory from a random start; returns the recorded
    energy-density trace."""
    r = np.random.default_rng(seed)
    s = project(r.standard_normal(N), N)
    es = []
    for k in range(n_steps):
        H, g = energy_grad(J, s, N)
        if k % record_every == 0:
            es.append(H / N)
        gt = g - (g @ s) * s / N
        noise = r.standard_normal(N)
        noise -= (noise @ s) * s / N
        s = project(s - beta * gt * dt + np.sqrt(2 * dt) * noise, N)
    return np.array(es)


def arrest_figure():
    from plot_style import ema
    N = 64
    dt = 0.005
    n_steps = 20000
    record_every = 20
    t_max = n_steps * dt
    temps = np.array([0.20, 0.28, 0.36, 0.45, 0.55, 0.61, 0.68, 0.75,
                      0.85, 1.00, 1.20, 1.50, 1.85, 2.30])
    seeds = [1, 2, 3]
    eps_eq = 0.05                       # equilibration tolerance
    e_mean, e_std, tau_mean, tau_capped = [], [], [], []
    for T in temps:
        beta = 1.0 / T
        es, taus = [], []
        for sd in seeds:
            J = make_tensor(N, seed=500 + sd)
            trace = quench_run(J, N, beta, dt, n_steps,
                               seed=int(T * 1e3) + sd)
            es.append(np.mean(trace[int(0.75 * len(trace)):]))
            sm = ema(trace, 0.05)
            target = -1.0 / T + eps_eq  # annealed equilibrium energy -1/T
            hit = np.nonzero(sm <= target)[0]
            taus.append(hit[0] * record_every * dt if len(hit)
                        else np.inf)
        e_mean.append(np.mean(es)); e_std.append(np.std(es))
        finite = [t for t in taus if np.isfinite(t)]
        capped = len(finite) < len(taus)
        tau_mean.append(np.mean(finite) if finite else t_max)
        tau_capped.append(capped)
        print(f"T={T:5.2f}  e_plateau={e_mean[-1]:+.4f}"
              f"  tau_eq={tau_mean[-1]:8.2f}{' (cap)' if capped else ''}")
    e_mean, e_std = np.array(e_mean), np.array(e_std)
    tau_mean = np.array(tau_mean)
    tau_capped = np.array(tau_capped)
    np.savez("arrest_data.npz", temps=temps, e_mean=e_mean, e_std=e_std,
             tau_mean=tau_mean, tau_capped=tau_capped)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    TT = np.linspace(temps.min(), temps.max(), 300)
    eq = np.maximum(-1.0 / TT, -E0)
    ax1.plot(TT, eq, "--", color="0.45", lw=1.2,
             label=r"equilibrium $e_{\mathrm{eq}}=\max(-1/T,\,-E_0)$",
             zorder=2)
    ax1.axhline(-EINF, color="0.15", ls=":", lw=1.0, zorder=1)
    ax1.annotate(r"threshold $-E_\infty$", xy=(0.97, -EINF + 0.02),
                 xycoords=ax1.get_yaxis_transform(), ha="right",
                 va="bottom", fontsize=9, color="0.15")
    ax1.axvline(TD, color=OKABE_ITO[3], ls="-.", lw=1.2, zorder=1)
    ax1.annotate(r"$T_d=1/E_\infty$", xy=(TD + 0.02, 0.03),
                 xycoords=ax1.get_xaxis_transform(), ha="left",
                 va="bottom", fontsize=9.5, color=OKABE_ITO[3])
    ax1.errorbar(temps, e_mean, yerr=e_std, fmt="o", ms=5,
                 color=OKABE_ITO[0], ecolor=OKABE_ITO[0], elinewidth=1.0,
                 capsize=2.5, zorder=4,
                 label=rf"plateau energy at $t={t_max:g}$ (sim.)")
    ax1.fill_between(TT[TT <= TD], -1.85, 0.0, color=OKABE_ITO[3],
                     alpha=0.06, zorder=0)
    ax1.set_xlabel(r"temperature $T=1/\beta$")
    ax1.set_ylabel(r"energy density")
    ax1.set_ylim(-1.85, -0.15)
    ax1.legend(loc="upper left", fontsize=8)

    ax2.axvline(TD, color=OKABE_ITO[3], ls="-.", lw=1.2, zorder=1)
    ax2.annotate(r"$T_d$", xy=(TD + 0.02, 0.95),
                 xycoords=ax2.get_xaxis_transform(), ha="left",
                 va="top", fontsize=10, color=OKABE_ITO[3])
    ok = ~tau_capped
    ax2.semilogy(temps[ok], tau_mean[ok], "o-", ms=5, lw=1.3,
                 color=OKABE_ITO[2], zorder=4,
                 label=r"equilibration time $\tau_{\mathrm{eq}}$")
    ax2.semilogy(temps[~ok], np.full((~ok).sum(), t_max), "^", ms=7,
                 color=OKABE_ITO[2], markerfacecolor="none", zorder=4,
                 label=rf"never equilibrates ($\tau>{t_max:g}$)")
    ax2.fill_between(TT[TT <= TD], 1, 3 * t_max, color=OKABE_ITO[3],
                     alpha=0.06, zorder=0)
    ax2.set_ylim(1, 3 * t_max)
    ax2.annotate("critical slowing down;\nfalls out of equilibrium\n"
                 r"below $T_d$", xy=(0.36, 0.55),
                 xycoords="axes fraction", fontsize=9, color="0.25",
                 ha="center")
    ax2.set_xlabel(r"temperature $T=1/\beta$")
    ax2.set_ylabel(r"time to reach $e_{\mathrm{eq}}+0.05$")
    ax2.legend(loc="lower left", fontsize=8)

    fig.savefig(f"{FIGDIR}/fig_arrest.pdf")
    plt.close(fig)
    print("saved fig_arrest.pdf")


# ---------------------------------------------------- landscape on S^2
def landscape_figure():
    """An actual pure p-spin landscape on the sphere for even p=4:
    H(-sigma)=H(sigma), so every well has an exact antipodal twin.  The
    two panels show the sphere from opposite directions."""
    N = 3
    r = np.random.default_rng(7)
    J4 = r.standard_normal((N, N, N, N))
    th = np.linspace(0, np.pi, 500)
    ph = np.linspace(0, 2 * np.pi, 500)
    TH, PH = np.meshgrid(th, ph)
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    S = np.sqrt(N) * np.stack([X, Y, Z], axis=-1)     # (..., 3)
    # symmetrize implicitly through the full contraction (p = 4)
    Hval = np.einsum('ijkl,...i,...j,...k,...l->...', J4, S, S, S, S) \
        / N ** 1.5
    print(f"landscape: H/N in [{Hval.min():.2f}, {Hval.max():.2f}]")

    fig = plt.figure(figsize=(7.6, 3.6))
    fig.set_layout_engine('none')
    norm = plt.Normalize(Hval.min(), Hval.max())
    cmap = plt.get_cmap("RdBu_r")
    ls = LightSource(azdeg=315, altdeg=42)
    rgb = ls.shade(norm(Hval), cmap=cmap, blend_mode="soft", vert_exag=0)
    for k, (azim, title) in enumerate([(-55, r"view of $\sigma$"),
                                       (125, r"antipodal view of $-\sigma$")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1,
                        antialiased=False, linewidth=0, shade=False)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.view_init(elev=14, azim=azim)
        ax.set_title(title, fontsize=10, pad=0)
        ax.dist = 7.2
    fig.subplots_adjust(left=0.0, right=0.88, top=0.98, bottom=0.02,
                        wspace=0.0)
    cax = fig.add_axes([0.90, 0.16, 0.022, 0.66])
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(m, cax=cax)
    cb.set_label(r"$H_{N,p}(\sigma)/N$", fontsize=10)
    cb.outline.set_visible(False)
    fig.savefig(f"{FIGDIR}/fig_landscape3d.png", dpi=300)
    plt.close(fig)
    print("saved fig_landscape3d.png")


# ---------------- Theorem-3 formal leading-term comparison (p = 4)
def mechanism_phase_figure():
    """Formal comparison of the two explicit leading terms beta*b and
    Lambda_EK_{p,beta}(b) of the LOWER BOUND in Theorem 3, at even p = 4.
    This is NOT the branch boundary of the theorem: the proved bound
    compares beta*b against Lambda_EK + C_{p,b}/sqrt(beta) with C_{p,b}
    unspecified, and it holds only for b > b_*(p) in (E_2, E_1), with
    b_* not numerically certified.  The region b <= E_2 (outside even
    the necessary condition) is masked."""
    p = 4
    EINF4, E04, E14 = theory_thresholds(p)

    def I1(z):
        if z <= EINF4:
            return 0.0
        return (2.0 / EINF4 ** 2) * quad(
            lambda x: np.sqrt(x * x - EINF4 ** 2), EINF4, z)[0]

    def theta1(z):          # Theta_{1,p}(-z)
        return 0.5 * np.log(p - 1) - (p - 2) * z * z / (4 * (p - 1)) \
            - 2.0 * I1(z)

    def omega(t):           # int log|t-x| rho_sc(x) dx, semicircle on [-2,2]
        a = abs(t)
        if a <= 2:
            return t * t / 4 - 0.5
        s = np.sqrt(a * a - 4)
        return t * t / 4 - 0.5 - (a * s / 4 - np.log((a + s) / 2))

    def Dp(z):              # half-determinant statistic D_p(-z)
        t = -z * np.sqrt(p / (p - 1))
        return np.log(np.sqrt(p * (p - 1))) + omega(t)

    def xi(z):              # Xi^EK_{1,p}(-z)
        return theta1(z) - 0.5 * Dp(z)

    def lam(beta, b):       # Lambda^EK_{p,beta}(b)
        zz = np.linspace(b, E14, 220)
        vals = beta * zz + np.array([xi(z) for z in zz])
        return -0.5 * np.log(beta * np.e) + vals.max()

    # E_2(p): root of Theta_{2,p}(-z) = 0.  Theorem 3 requires
    # b > b_*(p) in (E_2, E_1); b <= E_2 violates even the necessary
    # condition and is masked below.
    E24 = brentq(lambda z: 0.5 * np.log(p - 1)
                 - (p - 2) * z * z / (4 * (p - 1)) - 3.0 * I1(z),
                 EINF4 + 1e-9, E04)
    print(f"E_2({p}) = {E24:.5f}")

    bs = np.linspace(EINF4 + 1e-4, E14 - 1e-5, 90)
    betas = np.logspace(np.log10(3.0), np.log10(3000.0), 110)
    GAP = np.zeros((len(betas), len(bs)))
    for i, beta in enumerate(betas):
        for j, b in enumerate(bs):
            GAP[i, j] = lam(beta, b) - beta * b
    inside = bs > E24
    print("grid computed:",
          f"Lambda-term larger on {np.mean(GAP[:, inside] > 0) * 100:.1f}%"
          " of the certified-necessary region b > E_2")

    fig, ax = plt.subplots(figsize=(5.1, 3.5))
    GAPm = np.ma.masked_array(
        GAP, mask=np.broadcast_to(~inside, GAP.shape).copy())
    vmax = np.abs(GAPm).max()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    pc = ax.pcolormesh(bs, betas, GAPm, cmap="RdBu_r", norm=norm,
                       shading="auto", rasterized=True)
    ax.contour(bs[inside], betas, GAP[:, inside], levels=[0.0],
               colors="k", linewidths=1.6)
    # masked region: outside the necessary condition b > E_2
    ax.axvspan(EINF4, E24, color="0.88", zorder=3)
    ax.axvspan(EINF4, E24, facecolor="none", edgecolor="0.55",
               hatch="///", lw=0, zorder=4)
    ax.axvline(E24, color="0.25", lw=1.0, zorder=5)
    ax.annotate("outside the domain\nof Theorem 3\n($b\\leq E_2$)",
                xy=(0.5 * (EINF4 + E24), 0.50),
                xycoords=("data", "axes fraction"), fontsize=8,
                ha="center", color="0.25", zorder=6, rotation=90)
    ax.set_yscale("log")
    ax.set_xlabel(r"energy level $b$")
    ax.set_ylabel(r"inverse temperature $\beta$")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(r"$\Lambda^{\mathrm{EK}}_{p,\beta}(b)-\beta b$"
                 "  (leading terms only)")
    cb.outline.set_visible(False)
    ax.annotate("aggregate leading term\nlarger "
                r"($\Lambda^{\mathrm{EK}}>\beta b$)",
                xy=(0.44, 0.93), xytext=(0.62, 0.70),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=9, ha="center", color="0.1",
                arrowprops=dict(arrowstyle="->", color="0.1", lw=0.9))
    ax.annotate("energetic leading term\nlarger ($\\beta b$)",
                xy=(0.76, 0.16), xycoords="axes fraction", fontsize=9,
                ha="center", color="0.1")
    for val, name in [(EINF4, r"$E_\infty$"), (E24, r"$E_2$"),
                      (E14, r"$E_1$")]:
        ax.annotate(name, xy=(val, 1.015), xycoords=("data",
                    "axes fraction"), ha="center", fontsize=9)
    fig.savefig(f"{FIGDIR}/fig_mechanism_phase.pdf", dpi=300)
    plt.close(fig)
    print("saved fig_mechanism_phase.pdf")


if __name__ == "__main__":
    print(f"p={P}: E_inf={EINF:.5f} E_0={E0:.5f} E_1={E1:.5f} "
          f"T_d={TD:.4f}")
    landscape_figure()
    mechanism_phase_figure()
    arrest_figure()
