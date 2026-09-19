"""Direct enumeration of critical points of the pure spherical p-spin
model (p = 4) at small N, and measurement of the aggregate index-one
Eyring--Kramers weight

    W_{N,beta}(I) = sum_{z index-one, H(z)/N in I}
                    e^{-beta H(z)} alpha_z / sqrt(|det Q_z|),

the object bounded by the weighted Kac--Rice pressure of Theorem 3
(Xi^EK_{1,p} = Theta_{1,p} - D_p/2).  This is the experiment that tests
the paper's new mathematical quantity directly.

Produces:
  fig_saddle_sum.pdf   : (a) binned index-one counts vs Theta_{1,p};
                         (b) binned weights vs Xi^EK_{1,p};
                         (c) share of the largest single saddle in the
                             total weight, vs beta, for each N.
  saddle_data_N*.npz   : enumerated critical points per (N, sample).
  saddle_summary.txt   : counts, discovery diagnostics, dominance table.

Conventions as in pspin_langevin.py / pspin_p4.py:
E[H(s)H(t)] = N R(s,t)^p on the radius-sqrt(N) sphere; the Riemannian
Hessian at a critical point is V^T (Hess_euc H - (pH/N) I) V on the
tangent space (V an orthonormal tangent basis), whose law is the
shifted GOE c_N G_{N-1} - p u I of the appendix.

Run:  python3 saddle_enumeration.py run    (enumeration, saves npz)
      python3 saddle_enumeration.py plot   (figure + summary from npz)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

from plot_style import apply_style, OKABE_ITO
from pspin_langevin import theory_thresholds
from pspin_p4 import make_tensor4, energy_grad4, project

apply_style()
FIGDIR = "../figures"
P = 4
EINF, E0, E1 = theory_thresholds(P)

NS = [8, 10, 12]
N_SAMPLES = 8            # disorder samples per N
N_STARTS = 4000          # Newton starts per sample
SEED0 = 3000


# ------------------------------------------------------ theory curves
def I1(z):
    """GOE outlier rate I_{1,p}(z) for z >= E_inf (else 0)."""
    if z <= EINF:
        return 0.0
    return (2.0 / EINF ** 2) * quad(
        lambda x: np.sqrt(x * x - EINF ** 2), EINF, z)[0]


def theta_k(z, k):
    """Complexity Theta_{k,p}(-z) of index-k critical points."""
    return 0.5 * np.log(P - 1) - (P - 2) * z * z / (4 * (P - 1)) \
        - (k + 1) * I1(z)


def omega(t):
    """Semicircle log potential Omega(t)."""
    a = abs(t)
    if a <= 2:
        return t * t / 4 - 0.5
    s = np.sqrt(a * a - 4)
    return t * t / 4 - 0.5 - (a * s / 4 - np.log((a + s) / 2))


def Dp(z):
    """Determinant rate D_p(-z)."""
    t = -z * np.sqrt(P / (P - 1))
    return 0.5 * np.log(P * (P - 1)) + omega(t)


def xi_ek(z):
    """Xi^EK_{1,p}(-z) = Theta_1 - D_p/2."""
    return theta_k(z, 1) - 0.5 * Dp(z)


def threshold_E2():
    """E_2(p): root of Theta_{2,p}(-z) = 0."""
    return brentq(lambda z: theta_k(z, 2), EINF + 1e-9, E0)


# ------------------------------------------------- critical point search
def hessian_euc(J, s, N):
    """Euclidean Hessian 12 * J[.,.,s,s] / N^{3/2}."""
    A = (J.reshape(N ** 3, N) @ s).reshape(N, N, N)
    B = (A.reshape(N ** 2, N) @ s).reshape(N, N)
    return 12.0 * B / N ** 1.5


def newton_cp(J, s0, N, tol=1e-11, maxit=100):
    """Damped Newton for the KKT system grad H = mu * s, |s|^2 = N.
    Returns (s, converged)."""
    s = project(s0, N)
    mu = 0.0
    H, g = energy_grad4(J, s, N)
    mu = (g @ s) / N
    for _ in range(maxit):
        F = np.concatenate([g - mu * s, [((s @ s) - N) / 2.0]])
        res = np.linalg.norm(F)
        if res < tol * np.sqrt(N):
            return s, True
        Hs = hessian_euc(J, s, N)
        M = np.zeros((N + 1, N + 1))
        M[:N, :N] = Hs - mu * np.eye(N)
        M[:N, N] = -s
        M[N, :N] = s
        try:
            step = np.linalg.solve(M, -F)
        except np.linalg.LinAlgError:
            return s, False
        # backtracking on the KKT residual
        t = 1.0
        for _ in range(30):
            s_new = s + t * step[:N]
            mu_new = mu + t * step[N]
            H, g_new = energy_grad4(J, s_new, N)
            F_new = np.concatenate([g_new - mu_new * s_new,
                                    [((s_new @ s_new) - N) / 2.0]])
            if np.linalg.norm(F_new) < (1 - 0.25 * t) * res:
                break
            t *= 0.5
        else:
            return s, False
        s, mu, g = s_new, mu_new, g_new
    return s, False


def tangent_spectrum(J, s, N):
    """Eigenvalues of the Riemannian Hessian at a critical point."""
    H, g = energy_grad4(J, s, N)
    mu = (g @ s) / N                       # = pH/N at a critical point
    Hs = hessian_euc(J, s, N) - mu * np.eye(N)
    # orthonormal tangent basis via QR of the projector onto s-perp
    Q, _ = np.linalg.qr(
        np.eye(N) - np.outer(s, s) / N)
    # drop the (numerically zero) column aligned with s
    V = Q[:, np.argsort(np.abs(Q.T @ s))[: N - 1]]
    V, _ = np.linalg.qr(V)
    lam = np.linalg.eigvalsh(V.T @ Hs @ V)
    return H, lam


def enumerate_sample(N, seed):
    """Enumerate critical points of one disorder sample.  Returns dict
    with energies (density), index, alpha, logdet, discovery curve."""
    r = np.random.default_rng(seed)
    J = make_tensor4(N, seed)
    found = []                             # canonical configurations
    energies, indices, alphas, logdets = [], [], [], []
    discovery = []                         # distinct count after each start
    for k in range(N_STARTS):
        s0 = r.standard_normal(N)
        mode = k % 4
        if mode in (0, 1):                 # descend toward low energy
            s = project(s0, N)
            for _ in range(60 if mode == 0 else 15):
                H, g = energy_grad4(J, s, N)
                gt = g - (g @ s) * s / N
                s = project(s - 0.02 * gt, N)
            s0 = s
        elif mode == 2:                    # ascend (antipodal via evenness)
            s = project(s0, N)
            for _ in range(15):
                H, g = energy_grad4(J, s, N)
                gt = g - (g @ s) * s / N
                s = project(s + 0.02 * gt, N)
            s0 = s
        s, ok = newton_cp(J, s0, N)
        if ok:
            # canonical sign
            if s[np.argmax(np.abs(s))] < 0:
                s = -s
            is_new = all(
                min(np.linalg.norm(s - z), np.linalg.norm(s + z))
                > 1e-5 * np.sqrt(N) for z in found)
            if is_new:
                found.append(s.copy())
                H, lam = tangent_spectrum(J, s, N)
                energies.append(H / N)
                indices.append(int(np.sum(lam < 0)))
                neg = lam[lam < 0]
                alphas.append(np.abs(neg).max() if len(neg) else np.nan)
                logdets.append(np.sum(np.log(np.abs(lam))))
        discovery.append(len(found))
    return dict(energies=np.array(energies),
                indices=np.array(indices),
                alphas=np.array(alphas),
                logdets=np.array(logdets),
                discovery=np.array(discovery))


def run():
    for N in NS:
        results = []
        for s_idx in range(N_SAMPLES):
            seed = SEED0 + 100 * N + s_idx
            out = enumerate_sample(N, seed)
            results.append(out)
            n1 = np.sum(out["indices"] == 1)
            print(f"N={N} sample={s_idx}: {len(out['energies'])} distinct "
                  f"pairs, {n1} index-one; last-quarter discovery "
                  f"{out['discovery'][-1] - out['discovery'][3 * N_STARTS // 4]}"
                  f" new", flush=True)
        np.savez(f"saddle_data_N{N}.npz",
                 energies=np.concatenate(
                     [r["energies"] for r in results]),
                 indices=np.concatenate([r["indices"] for r in results]),
                 alphas=np.concatenate([r["alphas"] for r in results]),
                 logdets=np.concatenate([r["logdets"] for r in results]),
                 sample_id=np.concatenate(
                     [np.full(len(r["energies"]), i)
                      for i, r in enumerate(results)]),
                 discovery=np.stack([r["discovery"] for r in results]))
        print(f"saved saddle_data_N{N}.npz")


# ---------------------------------------------------------------- figure
def load(N):
    d = np.load(f"saddle_data_N{N}.npz")
    return d


def binned_rates(N, bins):
    """Disorder-averaged binned counts and weights (factor 2: each stored
    configuration represents an antipodal pair, both critical points)."""
    d = load(N)
    e, idx = d["energies"], d["indices"]
    al, ld = d["alphas"], d["logdets"]
    one = idx == 1
    cnt, _ = np.histogram(e[one], bins=bins)
    w = np.zeros(len(bins) - 1)
    for j in range(len(bins) - 1):
        m = one & (e >= bins[j]) & (e < bins[j + 1])
        if m.any():
            w[j] = np.sum(al[m] * np.exp(-0.5 * ld[m]))
    cnt = 2.0 * cnt / N_SAMPLES
    w = 2.0 * w / N_SAMPLES
    return cnt, w


def dominance(N, betas):
    """Share of the largest single index-one term in W_{N,beta}(all)."""
    d = load(N)
    one = d["indices"] == 1
    e = d["energies"][one] * N              # extensive energies
    al, ld = d["alphas"][one], d["logdets"][one]
    sid = d["sample_id"][one]
    shares, ubins = [], []
    for beta in betas:
        sh, ub = [], []
        for i in range(N_SAMPLES):
            m = sid == i
            if not m.any():
                continue
            logw = -beta * e[m] + np.log(al[m]) - 0.5 * ld[m]
            logw -= logw.max()
            wgt = np.exp(logw)
            sh.append(wgt.max() / wgt.sum())
            ub.append(e[m][np.argmax(logw)] / N)
        shares.append((np.mean(sh), np.std(sh)))
        ubins.append(np.mean(ub))
    return np.array(shares), np.array(ubins)


def plot():
    E2 = threshold_E2()
    print(f"E_2({P}) = {E2:.5f}   (E_inf={EINF:.5f}, E_1={E1:.5f}, "
          f"E_0={E0:.5f})")
    bins = np.arange(-1.85, 0.01, 0.115)
    mids = 0.5 * (bins[:-1] + bins[1:])
    zz = np.linspace(EINF - 0.55, 1.9, 300)
    th1 = np.array([theta_k(z, 1) for z in zz])
    xi = np.array([xi_ek(z) for z in zz])

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(10.6, 3.3))

    for c, N in zip(OKABE_ITO, NS):
        cnt, w = binned_rates(N, bins)
        okc = cnt > 0
        axA.plot(mids[okc], np.log(cnt[okc]) / N, "o", ms=4.5, color=c,
                 label=rf"$N={N}$")
        okw = w > 0
        axB.plot(mids[okw], np.log(w[okw]) / N, "s", ms=4.5, color=c,
                 label=rf"$N={N}$")
    axA.plot(-zz, th1, "-", color="0.2", lw=1.4,
             label=r"$\Theta_{1,p}(u)$ (theory)")
    axB.plot(-zz, xi, "-", color="0.2", lw=1.4,
             label=r"$\Xi^{\mathrm{EK}}_{1,p}(u)$ (theory)")
    for ax, ylab in [(axA, r"$\frac{1}{N}\log(\mathrm{count})$"),
                     (axB, r"$\frac{1}{N}\log(\mathrm{weight})$")]:
        for val, name in [(-E1, r"$-E_1$"), (-EINF, r"$-E_\infty$")]:
            ax.axvline(val, color="0.6", ls=":", lw=0.9)
            ax.annotate(name, xy=(val, 1.01),
                        xycoords=("data", "axes fraction"),
                        ha="center", fontsize=8.5, color="0.35")
        ax.set_xlabel(r"energy density $u$")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7.5, loc="lower left")
    axA.set_title("index-one counts", fontsize=9.5)
    axB.set_title(r"aggregate weight $\alpha/\sqrt{|\det Q|}$",
                  fontsize=9.5)

    betas = np.array([0.5, 1, 2, 4, 8, 16])
    for c, N in zip(OKABE_ITO, NS):
        sh, ub = dominance(N, betas)
        axC.errorbar(betas, sh[:, 0], yerr=sh[:, 1], fmt="o-", ms=4,
                     lw=1.2, color=c, capsize=2, label=rf"$N={N}$")
    axC.set_xscale("log")
    axC.set_ylim(0, 1.05)
    axC.set_xlabel(r"inverse temperature $\beta$")
    axC.set_ylabel("largest-saddle share of "
                   r"$\mathcal{W}_{N,\beta}$")
    axC.axhline(1.0, color="0.7", ls=":", lw=0.8)
    axC.legend(fontsize=7.5, loc="lower right")
    axC.set_title("single-channel dominance", fontsize=9.5)

    fig.savefig(f"{FIGDIR}/fig_saddle_sum.pdf")
    plt.close(fig)
    print("saved fig_saddle_sum.pdf")

    # ------------------------------------------------------- summary
    with open("saddle_summary.txt", "w") as f:
        f.write(f"E_inf={EINF:.5f} E_1={E1:.5f} E_0={E0:.5f} "
                f"E_2={E2:.5f}\n")
        for N in NS:
            d = load(N)
            e, idx = d["energies"], d["indices"]
            disc = d["discovery"]
            tail = disc[:, -1] - disc[:, 3 * N_STARTS // 4]
            f.write(f"\nN={N}: {len(e)} distinct antipodal pairs over "
                    f"{N_SAMPLES} samples\n")
            f.write(f"  minima: {np.sum(idx == 0)}, index-1: "
                    f"{np.sum(idx == 1)}, index>=2: {np.sum(idx >= 2)}\n")
            f.write(f"  deepest minimum u = {e[idx == 0].min():.4f} "
                    f"(band [-E_0,-E_inf] = [{-E0:.4f},{-EINF:.4f}])\n")
            if np.sum(idx == 1):
                f.write(f"  lowest index-1 u = {e[idx == 1].min():.4f} "
                        f"(threshold -E_1 = {-E1:.4f})\n")
            f.write(f"  discovery: new pairs in last quarter of "
                    f"{N_STARTS} starts, per sample: {tail.tolist()}\n")
            betas = np.array([0.5, 1, 2, 4, 8, 16])
            sh, ub = dominance(N, betas)
            for b, (m, s), u in zip(betas, sh, ub):
                f.write(f"  beta={b:5.1f}: largest-saddle share "
                        f"{m:.3f}+-{s:.3f}, dominant u = {u:+.3f}\n")
    print("saved saddle_summary.txt")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "run":
        run()
    elif stage == "plot":
        plot()
    else:
        raise SystemExit("usage: saddle_enumeration.py run|plot")
