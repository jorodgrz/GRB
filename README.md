# GRB Classification from Compact Binary Mergers

[![tests](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml/badge.svg)](https://github.com/jorodgrz/GRBproject/actions/workflows/pytest.yml)

Population-level predictions for merger-driven short and long GRBs. The pipeline applies the Gottlieb et al. (2023, 2024) classification frameworks to COMPAS binary population synthesis (Broekgaarden et al. 2021, the full 20-model grid).

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

## Data

```bash
python tools/download_compas_data.py --confirm             # full 20-model grid, ~45 GB (canonical)
python tools/download_compas_data.py --tier 1 --confirm    # optional lightweight subset (A, F, G, J, K)
```

Files land in `data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`. BNS catalogues from [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS from [Zenodo 5178777](https://zenodo.org/records/5178777). The observational comparison in [comparison.ipynb](comparison.ipynb) reads `data/rastinejad_2024.csv` (Rastinejad et al. 2024 component decomposition).

The downloader chains `tools/embed_model_metadata.py`, which writes `model` and `ns_max` as HDF5 root attributes. Loaders validate these against `expected_model` / `expected_ns_max` and fail loudly on filename or metadata drift.

## Layout

| File | Purpose |
|---|---|
| [grb_main.ipynb](grb_main.ipynb) | Main figures, Sections 1 to 14 with sub-sections 4b TNG-resolution sweep, 7b / 8c LVK GWTC-5.0 local-rate comparison, 7c / 8d channel x class decomposition, 14 full 20-model grid scan |
| [comparison.ipynb](comparison.ipynb) | BH-engine vs HMNS-engine prediction against the Rastinejad et al. (2024) sample, using Gottlieb et al. (2025) Eq. 11 |
| [pipeline_walkthrough.ipynb](pipeline_walkthrough.ipynb) | Runnable COMPAS-to-classification walkthrough: load, NS-mass remap, classify, then a section-by-section figure tour |
| [grb_physics.py](src/grb_physics.py) | Remnant mass, ejecta, EOS, Gottlieb thresholds, ISCO |
| [grb_classify.py](src/grb_classify.py) | BNS, BHNS, unified grid, formation channels, observed-merger classifier |
| [grb_rates.py](src/grb_rates.py) | Levina+ 2026 MSSFR, cosmic integration, BH-spin marginalization, beaming, detected rates |
| [grb_io.py](src/grb_io.py) | COMPAS HDF5 loading, STROOPWAFEL weights, metadata validation |
| [grb_offsets.py](src/grb_offsets.py) | Hernquist orbits, projected offset CDFs |
| [grb_plot_style.py](src/grb_plot_style.py) | Project palette and ApJ rcParams (`apply_apj_rcparams`) |

The six `grb_*.py` modules live in `src/` but import as flat modules (`import grb_physics`): `src/` is placed on the path via `pyproject.toml` (`pythonpath = ["src"]`), `tests/conftest.py`, and the first import cell of each notebook.

`plots/` is tracked; `data/`, `COMPAS/`, `papers/`, `demos/` are not.

## Classification

**BNS, Gottlieb (2024) four-class hybrid** (`classify_bns_2024`). $M_\mathrm{TOV} = 2.2\,M_\odot$, $M_\mathrm{thresh} = 1.27\,M_\mathrm{TOV}$ (Gottlieb 2023 fiducial; Bauswein et al. 2013 give an EOS-dependent band $k \in [1.30, 1.70]$), mass ratio $q \equiv M_1 / M_2$, $q_\mathrm{thresh} = 1.2$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk mass** (`classify_bhns`). $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$. Class names follow the Gottlieb 2024 hybrid; disk-mass thresholds are Gottlieb 2023 Sec. 4 / Fig. 6.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Faint lbGRB (BHNS) | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| lbGRB + red KN (BHNS disk) | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

All disk-mass-based GRB rates are upper bounds: 100 percent jet launching above threshold (Gottlieb 2023). The BNS rates carry an optional jet-breakout correction (`breakout_fraction_bns_eos` then `apply_bns_jet_breakout`, Section 10 of `grb_main.ipynb`): the Pais et al. (2025) breakout criterion is evaluated per system, EOS-marginalized over the four `EOS_MODELS`, and remnant-conditioned (HMNS classes carry the long post-collapse launch delay and a disk-wind obstruction). A breakout-corrected rate is no longer a pure upper bound; the uncorrected rate still is.

## Assumptions

- **Cosmology**: Planck 2015, matching COMPAS `FastCosmicIntegration`. $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$.
- **SN engine**: Fryer et al. (2012) delayed mechanism (Broekgaarden+ 2021 Table 1; Model I is the rapid variation and is not in the 5-model suite). Both engines share a $\sim 1.7\,M_\odot$ NS-mass deficit from the Fryer 2012 Eq. 12-13 baryonic-to-gravitational conversion (Broekgaarden+ 2021 footnote 3); a global Alsing, Silva and Berti (2018) double-Gaussian remap closes it (Mandel and Muller 2020).
- **LVK anchor**: LVK (2026) GWTC-5.0 90 percent CRs (BNS 5.1 to 154.7 Gpc$^{-3}$ yr$^{-1}$ at $z=0$, NSBH 6.7 to 32.8 Gpc$^{-3}$ yr$^{-1}$ at $z=0$) pinned in `grb_rates.LVK_GWTC5_LOCAL_RATES` / `LVK_GWTC5_PER_MODEL_RATES`. GWTC-5.0 added no new BNS or NSBH, so the NS-side bands tightened relative to GWTC-4; the Model A BHNS rate now sits above the NSBH band.
- **MSSFR**: Levina et al. (2026) Azzalini skew log-normal best-fit to IllustrisTNG TNG100-1 (`MSSFR_PARAMS_LEVINA26_TNG100`, `SFR_PARAMS_LEVINA26_TNG100`); TNG50-1 and TNG300-1 are exposed for the Section 4b resolution sweep.

## Testing

```bash
make ci         # lint + typecheck + smoke; what CI runs on every push and PR
make smoke      # fast subset (no data/, no compas), under ~15 s
make coverage   # 70 percent coverage floor on grb_*.py over unit + anchors
make test       # full suite; data-bound tests auto-skip if data/ is empty
```

See [tests/README.md](tests/README.md) for the per-folder, per-section layout and the marker reference.

## References

- LVK (2026), [arXiv:2605.27226](https://arxiv.org/abs/2605.27226) (GWTC-5.0 population properties; intrinsic local merger rates used as the BNS / NSBH / BBH anchor)
- Abac et al. (2025), [arXiv:2508.18083](https://arxiv.org/abs/2508.18083) (GWTC-4 population properties; NSBH minimum BH mass, inherited by GWTC-5.0)
- Abbott et al. (2019), [arXiv:1805.11579](https://arxiv.org/abs/1805.11579), PRX 9, 011001 (GW170817 properties)
- Abbott et al. (2020), [arXiv:2001.01761](https://arxiv.org/abs/2001.01761), ApJL 892, L3 (GW190425)
- Abbott et al. (2023), [arXiv:2111.03634](https://arxiv.org/abs/2111.03634), PRX 13, 011048 (GWTC-3 population properties; the rate paper, not the GWTC-3 catalog arXiv:2111.03606)
- Alsing, Silva, and Berti (2018), [arXiv:1709.07889](https://arxiv.org/abs/1709.07889)
- Antoniadis et al. (2016), [arXiv:1605.01665](https://arxiv.org/abs/1605.01665)
- Asplund, Grevesse, Sauval, and Scott (2009), [arXiv:0909.0948](https://arxiv.org/abs/0909.0948), ARA&A 47, 481 ($Z_\odot = 0.0142$)
- Bailyn, Jain, Coppi, and Orosz (1998), [arXiv:astro-ph/9708032](https://arxiv.org/abs/astro-ph/9708032), ApJ 499, 367 (lower BH mass gap)
- Bardeen, Press, and Teukolsky (1972), [ADS 1972ApJ...178..347B](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B)
- Bauswein, Baumgarte, and Janka (2013), [arXiv:1302.6530](https://arxiv.org/abs/1302.6530)
- Bauswein et al. (2020), [arXiv:2004.00846](https://arxiv.org/abs/2004.00846)
- Bavera et al. (2020), [arXiv:1906.12257](https://arxiv.org/abs/1906.12257)
- Beniamini, Nava, and Piran (2016), [arXiv:1606.00311](https://arxiv.org/abs/1606.00311) (radiative-efficiency anchor for `GOTTLIEB25_F_RANGE`)
- Beniamini and Nakar (2019), [arXiv:1808.05076](https://arxiv.org/abs/1808.05076)
- Berger (2014), [arXiv:1311.2603](https://arxiv.org/abs/1311.2603), ARA&A 52, 43
- Binney and Tremaine (2008), *Galactic Dynamics*, 2nd ed., Princeton University Press
- Biscoveanu, Landry, and Vitale (2022), [arXiv:2207.01601](https://arxiv.org/abs/2207.01601) (NSBHPop population model)
- Bloom, Sigurdsson, and Pols (1999), [arXiv:astro-ph/9904338](https://arxiv.org/abs/astro-ph/9904338), MNRAS 305, 763
- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608)
- Colombo et al. (2022), [arXiv:2204.07592](https://arxiv.org/abs/2204.07592)
- Crameri, Shephard, and Heron (2020), Nat. Commun. 11, 5444
- Dietrich and Ujevic (2017), [arXiv:1612.03665](https://arxiv.org/abs/1612.03665)
- Farr et al. (2011), [arXiv:1011.1459](https://arxiv.org/abs/1011.1459), ApJ 741, 103 (lower BH mass gap)
- Finn and Chernoff (1993), [ADS 1993PhRvD..47.2198F](https://ui.adsabs.harvard.edu/abs/1993PhRvD..47.2198F)
- Fong and Berger (2013), [arXiv:1307.0819](https://arxiv.org/abs/1307.0819)
- Fong et al. (2015), [arXiv:1509.02922](https://arxiv.org/abs/1509.02922)
- Fong et al. (2022), [arXiv:2206.01763](https://arxiv.org/abs/2206.01763) (sGRB hosts I/II)
- Foucart, Hinderer, and Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011)
- Foucart et al. (2019), [arXiv:1903.09166](https://arxiv.org/abs/1903.09166)
- Fragos et al. (2010), [arXiv:1001.1107](https://arxiv.org/abs/1001.1107)
- Fryer et al. (2012), [arXiv:1110.1726](https://arxiv.org/abs/1110.1726)
- Fujibayashi et al. (2018), [arXiv:1710.07579](https://arxiv.org/abs/1710.07579), ApJ 860, 64
- Fuller and Ma (2019), [arXiv:1905.08793](https://arxiv.org/abs/1905.08793)
- Gerosa et al. (2018), [arXiv:1808.02491](https://arxiv.org/abs/1808.02491)
- Ghirlanda et al. (2016), [arXiv:1607.07875](https://arxiv.org/abs/1607.07875), A&A 594, A84
- Goldstein et al. (2017), [arXiv:1710.05446](https://arxiv.org/abs/1710.05446), ApJL 848, L14
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038)
- Gottlieb et al. (2024, 2025), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657)
- Hernquist (1990), [ADS 1990ApJ...356..359H](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Hurley, Tout, and Pols (2002), [arXiv:astro-ph/0201220](https://arxiv.org/abs/astro-ph/0201220), MNRAS 329, 897 (BSE; $\zeta_\mathrm{ad}$)
- Kasen et al. (2017), [arXiv:1710.05463](https://arxiv.org/abs/1710.05463), Nature 551, 80
- Kawaguchi et al. (2015), [arXiv:1601.07711](https://arxiv.org/abs/1601.07711)
- Kish (1965), *Survey Sampling*, Wiley (effective sample size $n_\mathrm{eff} = (\sum w)^2 / \sum w^2$)
- Koppel, Bovard, and Rezzolla (2019), [arXiv:1901.09977](https://arxiv.org/abs/1901.09977), ApJL 872, L16
- Kroupa (2001), [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005)
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728)
- Lattimer and Prakash (2001), [arXiv:astro-ph/0002232](https://arxiv.org/abs/astro-ph/0002232), ApJ 550, 426
- Levan et al. (2024), [arXiv:2307.02098](https://arxiv.org/abs/2307.02098), Nature 626, 737
- Levina et al. (2026), [arXiv:2601.20202](https://arxiv.org/abs/2601.20202)
- Lippuner et al. (2017), [arXiv:1703.06216](https://arxiv.org/abs/1703.06216), MNRAS 472, 904
- Madau and Dickinson (2014), [arXiv:1403.0007](https://arxiv.org/abs/1403.0007)
- Mandel and Muller (2020), [arXiv:2006.08360](https://arxiv.org/abs/2006.08360)
- Margalit and Metzger (2017), [arXiv:1710.05938](https://arxiv.org/abs/1710.05938), ApJL 850, L19
- Metzger (2019), [arXiv:1910.01617](https://arxiv.org/abs/1910.01617), Living Rev. Rel. 23, 1
- Mooley et al. (2018), [arXiv:1806.09693](https://arxiv.org/abs/1806.09693), Nature 561, 355
- Neijssel et al. (2019), [arXiv:1906.08136](https://arxiv.org/abs/1906.08136)
- Pais, Piran, Kiuchi, and Shibata (2025), [arXiv:2407.19002](https://arxiv.org/abs/2407.19002) (BNS jet breakout)
- Patton and Sukhbold (2020), [arXiv:2005.03055](https://arxiv.org/abs/2005.03055), MNRAS 499, 2803
- Planck Collaboration / Ade et al. (2016), [arXiv:1502.01589](https://arxiv.org/abs/1502.01589), A&A 594, A13
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981)
- Radice et al. (2018), [arXiv:1809.11163](https://arxiv.org/abs/1809.11163)
- Rastinejad et al. (2022), [arXiv:2204.10864](https://arxiv.org/abs/2204.10864), Nature 612, 223 (GRB 211211A)
- Rastinejad et al. (2024), [arXiv:2306.14947](https://arxiv.org/abs/2306.14947)
- Read et al. (2009), [arXiv:0812.2163](https://arxiv.org/abs/0812.2163)
- Salpeter (1955), [ADS 1955ApJ...121..161S](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S)
- Silverman (1986), *Density Estimation for Statistics and Data Analysis*, Chapman & Hall (KDE reflective-boundary trick, Sec. 2.10)
- Talbot and Thrane (2018), [arXiv:1801.02699](https://arxiv.org/abs/1801.02699), ApJ 856, 173 (cumulative-fraction contour levels)
- van Son et al. (2022), [arXiv:2110.01634](https://arxiv.org/abs/2110.01634), ApJ 931, 17 (post-CE separation and delay-time response to $\alpha_\mathrm{CE}$; their Fig. 14, panels H and I)
- Villasenor et al. (2005), [ADS 2005Natur.437..855V](https://ui.adsabs.harvard.edu/abs/2005Natur.437..855V) (GRB 050709)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878)
- Webbink (1984), [ADS 1984ApJ...277..355W](https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W) (CE $\alpha$-formalism)
- Xu and Li (2010), [arXiv:1004.4957](https://arxiv.org/abs/1004.4957), ApJ 716, 114 ($\lambda_\mathrm{CE}$)

## License

MIT. See [LICENSE](LICENSE).
