# Spectral Gap Bounds for Langevin Dynamics in Mixed Spherical Spin Glasses

[![arXiv](https://img.shields.io/badge/arXiv-2609.15157-b31b1b.svg)](https://arxiv.org/abs/2609.15157)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Simulation code and data accompanying the paper

> **Spectral Gap Bounds for Langevin Dynamics in Mixed Spherical Spin Glasses**
> Masoud Badiei Khuzani
> [arXiv:2609.15157](https://arxiv.org/abs/2609.15157)

The paper proves rigorous *lower* bounds on the relaxation time
1/&gamma;<sub>N,&beta;</sub> of spherical Langevin dynamics for mixed even
spin glasses with mixture &xi;(x) = &Sigma;<sub>p&ge;4</sub>
&gamma;<sub>p</sub>&sup2;x<sup>p</sup>, under a 1RSB-type standing assumption
of strict threshold separation E&#8320;(&xi;) &gt; E&#8321;(&xi;) &gt;
E&#8322;(&xi;). The main result is an aggregate Eyring&ndash;Kramers bound in
which *all* index-one saddles below a cutoff contribute a
determinant-weighted sum of escape channels; the pure spherical
*p*-spin model with even *p* &ge; 4 is recovered as a corollary. The
numerical experiments in Section&nbsp;9 of the paper probe the pure case
*p* = 4, where every rate function has a closed form.

## Repository layout

```
code/            self-contained numpy scripts (fixed random seeds) + raw outputs
figures/         figures exactly as they appear in the paper
```

All scripts write their figures to `../figures` and their data files to the
current directory, so run them from inside `code/`.

## Scripts

| Script | What it does | Paper figures |
|---|---|---|
| `pspin_p4.py` | Langevin dynamics for the pure *p* = 4 model: threshold energy relaxation (N = 60), two-time aging C(t<sub>w</sub>, t<sub>w</sub>+t), activated escape times for N &isin; {8, 12, 16} with censoring-aware medians and bootstrap confidence intervals, step-size check | `fig_energy_relaxation.pdf`, `fig_aging.pdf`, `fig_arrhenius.pdf` |
| `saddle_enumeration.py` | Exhaustive critical-point enumeration for *p* = 4, N &isin; {8, 10, 12} (damped Newton on the KKT system from 4,000 starts per disorder sample), exact Riemannian Hessian index, and direct measurement of the determinant-weighted saddle sum W<sub>N,&beta;</sub> and its largest-channel share | `fig_saddle_sum.pdf` |
| `phase_transitions.py` | Dynamical arrest sweep (plateau energy and equilibration time vs. temperature, *p* = 3), branch-dominance diagram of the two explicit leading terms of the bound, and a 3-D landscape visualization | `fig_arrest.pdf`, `fig_mechanism_phase.pdf`, `fig_landscape3d.png` |
| `pspin_langevin.py` | Pure *p* = 3 reference implementation; computes the thresholds E<sub>&infin;</sub>, E&#8320;, E&#8321; by quadrature and root finding and provides the coupling-tensor and dynamics utilities imported by the other scripts | &mdash; |
| `plot_style.py` | Shared publication plot style (fonts, Okabe&ndash;Ito palette, helpers) | &mdash; |

The theoretical thresholds used throughout (computed in the scripts by
quadrature and root finding, cf. eq. (thresholds) in the paper), at *p* = 4:

```
E_inf = 1.73205    E_2 = 1.77648    E_1 = 1.78330    E_0 = 1.79409
```

## Data files

The raw outputs used to produce the figures are committed alongside the
scripts, so every figure can be regenerated without re-running the (long)
simulations:

- `saddle_data_N{8,10,12}.npz`, `saddle_summary.txt` — per-sample critical-point
  data and diagnostics of the saddle enumeration, including the completeness
  diagnostic (new distinct antipodal pairs discovered in the final quarter of
  the 4,000 Newton starts, per sample);
- `relax_p4_beta*.npy`, `aging_p4_tw*.npy` — relaxation traces and two-time
  overlap curves;
- `escape_p4_N*.npz`, `escape_p4_summary.npz`, `dtcheck_p4.npz` — escape-time
  samples with censoring flags, Arrhenius fits, and the step-size check;
- `arrest_data.npz` — arrest-sweep plateau energies and equilibration times;
- `results_summary.txt` — *p* = 3 thresholds and Arrhenius slope.

## Reproducing the figures

Requirements: Python &ge; 3.9 with `numpy`, `scipy`, `matplotlib`
(`pip install -r requirements.txt`).

From `code/`, to re-plot from the committed data (fast):

```bash
python pspin_p4.py relax_multi     # fig_energy_relaxation.pdf
python pspin_p4.py aging_multi     # fig_aging.pdf
python pspin_p4.py escape_plot     # fig_arrhenius.pdf
python saddle_enumeration.py plot  # fig_saddle_sum.pdf
python phase_transitions.py        # fig_arrest.pdf, fig_mechanism_phase.pdf, fig_landscape3d.png
```

To re-run the simulations from scratch (hours of CPU time; seeds are fixed):

```bash
python pspin_p4.py all             # relaxation + aging + escape sweeps
python saddle_enumeration.py run   # saddle enumeration, N = 8, 10, 12
python phase_transitions.py        # arrest sweep + phase diagram + landscape
```

## Citation

```bibtex
@article{badieikhuzani2026spectral,
  title   = {Spectral Gap Bounds for {L}angevin Dynamics in Mixed Spherical Spin Glasses},
  author  = {Badiei Khuzani, Masoud},
  journal = {arXiv preprint arXiv:2609.15157},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.15157}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).
