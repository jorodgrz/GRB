# GRB Classification from Compact Binary Mergers

[![tests](https://github.com/jorodgrz/GRB/actions/workflows/pytest.yml/badge.svg)](https://github.com/jorodgrz/GRB/actions/workflows/pytest.yml)

Population-level predictions for merger-driven short and long GRBs. The pipeline applies the Gottlieb et al. (2023, 2024) classification frameworks to COMPAS binary population synthesis (Broekgaarden et al. 2021, the full 20-model grid), then follows each class through cosmic rate integration (on the IllustrisTNG TNG100-1 metallicity-dependent star formation history), beaming, GW and short-GRB rate comparisons, and host-galaxy offsets.

## Setup

```bash
conda env create -f environment.yml
conda activate grb-env
python -m ipykernel install --user --name grb-env --display-name "GRB (grb-env)"
```

`environment.yml` (conda-forge, Python 3.10) is the only dependency manifest. It also installs `compas_python_utils` by pip from upstream COMPAS at a pinned commit, since that package is on neither PyPI nor conda-forge.

## Data

```bash
python tools/download_compas_data.py --confirm             # full 20-model grid (canonical)
python tools/download_compas_data.py --tier 1 --confirm    # optional lightweight subset (A, F, G, J, K)
python tools/download_compas_data.py --dry-run             # print the plan and download size only
```

The full grid is 18 zip archives per population, about 45 GiB compressed (7.3 GiB BNS, 38.0 GiB BHNS) and about 100 GiB once extracted. Files land in `data/COMPASCompactOutput_<KIND>_<SUFFIX>.h5`. BNS catalogues come from [Zenodo 5189849](https://zenodo.org/records/5189849), BHNS from [Zenodo 5178777](https://zenodo.org/records/5178777). Other flags: `--kind {BNS,BHNS}`, `--models <SUFFIX ...>`, `--verify` (re-check cached zips against the Zenodo MD5), `--keep-zips`, `--force`.

The downloader chains `tools/embed_model_metadata.py`, which writes `model`, `ns_max`, `mssfr_mu0` and `zenodo_id` as HDF5 root attributes. Loaders validate these against `expected_model` / `expected_ns_max` and fail loudly on filename or metadata drift.

**Model letters.** Filenames and code use the Broekgaarden et al. (2021, paper I) suffixes; the Zenodo release and the figure labels in Sections 9 and 11 use the Broekgaarden et al. (2022, paper II) letters (`grb_io.BROEKGAARDEN22_PAPER_II_LETTERS`). A to E agree in both conventions, which is why the mismatch is easy to miss.

| File suffix | Paper II | Variation | $M_\mathrm{NS,max}$ |
|---|---|---|---|
| A | A | Fiducial | 2.5 |
| B, C, D | B, C, D | Mass-transfer efficiency $\beta$ = 0.25, 0.5, 0.75 | 2.5 |
| E | E | Unstable case BB mass transfer | 2.5 |
| EH | F | Unstable case BB plus optimistic CE | 2.5 |
| alpha0p1, F, G, alpha10 | G, H, I, J | $\alpha_\mathrm{CE}$ = 0.1, 0.5, 2.0, 10 | 2.5 |
| H | K | Optimistic CE (HG donors survive) | 2.5 |
| I | L | Rapid SN remnant prescription | 2.5 |
| J | M | Maximum NS mass 2.0 $M_\odot$ | 2.0 |
| K | N | Maximum NS mass 3.0 $M_\odot$ | 3.0 |
| L | O | No (pulsational) PISN | 2.5 |
| M, N | P, Q | Core-collapse kick $\sigma$ = 100, 30 km/s | 2.5 |
| O | R | No BH natal kick | 2.5 |
| fWR0p1, fWR5 | S, T | Wolf-Rayet wind multiplier 0.1, 5 | 2.5 |

**Star formation history.** The fiducial metallicity-dependent star formation rate is the IllustrisTNG $\mathcal{S}(Z,z)$ table distributed with Levina et al. (2026). Place `SFRMetallicityFromGasWithMetalsTNG{50,100,300}-1.hdf5` in `data/` (`grb_io.TNG_SFRD_PATHS`); TNG100-1 drives every rate, and the other two boxes feed the Sections 4b, 4d and 7e resolution comparisons.

**Observational inputs.** Host-offset catalogues (Fong et al. 2022; Skobe et al. 2026), GW events and literature rates live as constants in `grb_offsets`, `grb_io` and `grb_rates`, each pinned to its source in `tests/anchors/`. Nothing in `data/` is tracked.

## Layout

| File | Purpose |
|---|---|
| [grb_main.ipynb](grb_main.ipynb) | Main research notebook, one section per figure (map below) |
| [grb_physics.py](src/grb_physics.py) | Cosmology pin, Gottlieb thresholds, NS radius and EOS table, ISCO, Foucart (2018) remnant mass, Kruger and Foucart (2020) disk and dynamical ejecta, Pais et al. jet-breakout helpers, cold-stability cut |
| [grb_classify.py](src/grb_classify.py) | BNS four-class and BHNS three-class classifiers, unified mass-plane grid, Broekgaarden formation channels I to V, channel x class contingency |
| [grb_rates.py](src/grb_rates.py) | IllustrisTNG TNG100-1 S(Z,z) table (fiducial) and the Levina et al. (2026) analytic fits, cosmic integration around COMPAS `FastCosmicIntegration`, GW detection rates, BH-spin marginalization, beaming, jet breakout, metallicity efficiency, literature rate anchors |
| [grb_io.py](src/grb_io.py) | COMPAS HDF5 loaders, STROOPWAFEL-weighted subsampling, metadata validation, model registry, observed GW events |
| [grb_offsets.py](src/grb_offsets.py) | Composite Hernquist stellar plus NFW halo host potential, Jeans pre-kick velocities, velocity-Verlet orbits, projected offset CDFs, weighted Anderson-Darling tests, observed offset catalogues |
| [grb_plot_style.py](src/grb_plot_style.py) | Project palette, ApJ rcParams (`apply_apj_rcparams`), weighted histogram helpers |
| [tools/](tools/) | `download_compas_data.py` (Zenodo downloader) and `embed_model_metadata.py` (HDF5 root attributes) |
| [tests/](tests/) | `unit/`, `anchors/` (literature values), `sections/` (one file per notebook section), `integration/` (needs `data/`, COMPAS or notebook execution) |

The six `grb_*.py` modules live in `src/` but import as flat modules (`import grb_physics`): `src/` is placed on the path via `pyproject.toml` (`pythonpath = ["src"]`), `tests/conftest.py`, and the first import cell of each notebook.

`plots/` is tracked: figures as PDF and PNG, plus the Section 6c, 8c and 11 caches (`.npz`, `.csv`). The Section 6d and 7c per-model rate caches are gitignored and rebuilt on demand (about four minutes each). `data/`, `COMPAS/`, `papers/`, `demos/`, `council/` and `TNG_S(Z,z)/` are not tracked.

### `grb_main.ipynb` sections

An unnumbered setup cell loads Model A, applies the cold-stability cut and sets the palette; every section after it is self-contained.

| Section | Content |
|---|---|
| 1 | Mass plane, BNS and BHNS, with class boundaries |
| 3, 3b | Delay-time distributions per class (per dex in $t_\mathrm{delay}$), and split by formation channel ($\mathrm{d}p/\mathrm{d}t$) |
| 3c | sbGRB + blue KN delay-time distribution across the 20-model grid against published short-GRB DTD fits |
| 4, 4b, 4c | Cosmic-integration setup; TNG50 / TNG100 / TNG300 resolution sweep on the Levina et al. fits; predicted LVK detection counts |
| 4d, 4e | TNG $\mathcal{S}(Z,z)$ tables against the Levina et al. (2026) fits: rates, and the TNG100-1 map |
| 5 | Formation efficiency $\eta(Z)$ per class |
| 6, 6b | BNS $\mathcal{R}(z)$ per class, and split by formation channel |
| 6c | Beamed sbGRB + blue KN rate against Swift/BAT short GRBs, Du et al. (2025), Pracchia and Salafia (2026) and LVK GWTC-5.0 |
| 6d | BNS $\mathcal{R}(z)$ per class across the 20-model grid |
| 7, 7b, 7c | BHNS $\mathcal{R}(z)$ with BH-spin sweep; split by formation channel; across the 20-model grid |
| 7d, 7e | Merger-rate-weighted BNS $M_\mathrm{tot}$ and BHNS $M_\mathrm{BH}$ distributions by class; their redshift evolution across the three TNG boxes |
| 8, 8a, 8b, 8c | Projected host offsets against BRIGHT and kilonova-selected samples; free-drift check; delay-time host routing (table only); free drift across the 20-model grid |
| 9 | Summary table for paper-I Models A, F, G, J, K (displayed as A, H, I, M, N) |
| 11 | Full 20-model grid scan: file validation and cached scan (11.0, 11.0b); BNS and BHNS class versus channel composition with $\mathcal{R}(z=0)$ against LVK (11.1, 11.1b); HMNS split-factor sweep (11.7, table) |
| 12 | Classification scheme decision tree |

There are no Sections 2, 10 or 11.2 to 11.6.

## Classification

**BNS, Gottlieb (2024) four-class hybrid** (`classify_bns_2024`). $M_\mathrm{TOV} = 2.2\,M_\odot$ (Raaijmakers et al. 2021). $M_\mathrm{thresh} = k\,M_\mathrm{TOV}$ with $k = 1.27$, so $M_\mathrm{thresh} \approx 2.79\,M_\odot$; this is a fiducial chosen so that $M_\mathrm{thresh} \approx 2.8\,M_\odot$ (Gottlieb 2023), and it sits inside the $k \in [1.259, 1.514]$ spanned by the 23-EOS base sample of Bauswein et al. (2020) Table I. Mass ratio $q \equiv M_1 / M_2 \geq 1$, $q_\mathrm{thresh} = 1.2$. The long-lived / short-lived HMNS split at $1.2\,M_\mathrm{TOV} \approx 2.64\,M_\odot$ is a code heuristic (motivated by Margalit and Metzger 2017), not a Gottlieb (2024) number; it is the `hmns_factor` kwarg and Section 11.7 sweeps it. For EOS sweeps pass `k_thresh` so both thresholds move with $M_\mathrm{TOV}$.

| Class | Condition |
|---|---|
| sbGRB + blue KN | $M_\mathrm{tot} < 1.2\,M_\mathrm{TOV}$ |
| lbGRB + red KN (HMNS) | $1.2\,M_\mathrm{TOV} \leq M_\mathrm{tot} < M_\mathrm{thresh}$ |
| lbGRB + red KN (disk) | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q \geq q_\mathrm{thresh}$ |
| Faint lbGRB | $M_\mathrm{tot} \geq M_\mathrm{thresh}$, $q < q_\mathrm{thresh}$ |

**BHNS, disk mass** (`classify_bhns`). $M_\mathrm{disk} = M_\mathrm{rem}^\mathrm{Foucart\,2018} - M_\mathrm{dyn}^\mathrm{KF\,2020}$ at fiducial BH spin $\chi_\mathrm{BH} = 0.5$ (`A_BH_FID`). Class names follow the Gottlieb 2024 hybrid; disk-mass thresholds are Gottlieb 2023 Sec. 4 / Fig. 6. The Foucart et al. (2018) fit is calibrated for $Q = M_\mathrm{BH}/M_\mathrm{NS} \in [1, 7]$; on Model A after the cold-stability cut 23.8 percent of the BHNS weight has $Q > 7$, so those disk masses and class assignments are extrapolations.

| Class | Condition |
|---|---|
| No GRB | $M_\mathrm{disk} < 0.01\,M_\odot$ |
| Faint lbGRB (BHNS) | $0.01 \leq M_\mathrm{disk} < 0.1\,M_\odot$ |
| lbGRB + red KN (BHNS disk) | $M_\mathrm{disk} \geq 0.1\,M_\odot$ |

All disk-mass-based GRB rates are upper bounds: 100 percent jet launching above threshold (Gottlieb 2023). `grb_rates` also provides a BNS jet-breakout correction (`breakout_fraction_bns_eos`, then `apply_bns_jet_breakout`, then `beamed_rate`): the Pais et al. breakout criterion is evaluated per system, EOS-marginalized over the four `EOS_MODELS`, and remnant-conditioned (HMNS classes carry the long post-collapse launch delay and a disk-wind obstruction). No figure in `grb_main.ipynb` applies it, so every plotted rate is the uncorrected upper bound; the Section 6c caption explains why: the median breakout fraction it returns for the sbGRB + blue KN class is 0. A breakout-corrected rate is no longer a pure upper bound. BHNS rates for the two GRB classes carry a separate population-level misalignment factor of 0.5 (`apply_bhns_misalignment`; Fragos et al. 2010, Kawaguchi et al. 2015).

## Assumptions

- **Cosmology**: Planck 2015, matching COMPAS `FastCosmicIntegration`. $H_0 = 67.74$ km/s/Mpc, $\Omega_m = 0.3089$, $\Omega_\Lambda = 0.6911$.
- **SN engine**: Fryer et al. (2012) delayed mechanism in every model except I (paper-II L), the rapid variation (Broekgaarden+ 2021 Table 1). The delayed engine carries a $\sim 1.7\,M_\odot$ NS-mass deficit (Broekgaarden+ 2021 footnote 3), with a narrow genuinely-empty void at $[1.70, 1.76)\,M_\odot$. The rapid engine does not share the void (Model I has 812 BNS neutron-star components in 774 systems in that interval; Model A has none), so the void belongs to the delayed remnant prescription rather than to the baryonic-to-gravitational conversion.
- **NS masses**: the fiducial keeps the raw COMPAS masses and applies only a cold-stability cut $m_i \le M_\mathrm{TOV}$ (`cold_stable_ns_mask`), applied once in the notebook's setup cell to every BNS and BHNS sample. The Alsing, Silva and Berti (2018) double-Gaussian remap is kept in `grb_physics` for anyone who wants that sensitivity, but no notebook cell applies it: it inflates the weight across the deficit rather than recovering it. With no remap, nothing on the science path draws a random number.
- **Rate normalisation**: one star-forming mass serves both populations and all 20 models, $M_\mathrm{SF} = 53 \times 10^6 \times 79.11\,M_\odot = 4.19\times10^9\,M_\odot$ (`simulation_star_forming_mass`), a property of the COMPAS sampling configuration rather than a calibration. Metallicity binning follows COMPAS `ClassMSSFR` (`dPdlogZ_binned_to_grid`), so the top grid point absorbs star formation above $Z = 0.03$, which under the TNG100-1 MSSFR is 28.5 percent of the $z = 0$ star formation (Section 4 prints it). That extrapolation applies the $Z = 0.03$ yield to every higher metallicity.
- **MSSFR**: the IllustrisTNG TNG100-1 S(Z,z) table as distributed by Levina et al. (2026) (`data/SFRMetallicityFromGasWithMetalsTNG100-1.hdf5`, read by `grb_io.load_tng_sfrd` and `grb_rates.tng_sfr_and_dPdlogZ`). The Levina et al. (2026) Table 1 analytic fits (Azzalini skew log-normal plus Madau and Dickinson 2014 SFR; `MSSFR_PARAMS_LEVINA26_TNG100`, `SFR_PARAMS_LEVINA26_TNG100`, and the TNG50-1 / TNG300-1 sets) are kept for the Section 4b resolution sweep and the Section 4d table-versus-fit comparison.
- **Rates**: intrinsic, comoving, source frame, in Gpc$^{-3}$ yr$^{-1}$ unless suffixed `_obs`. Compare with observed short-GRB rates only after `beamed_rate`; compare with GW rates directly. Local rates $\mathcal{R}(z=0)$ are quoted from unsmoothed arrays, since the default display smoothing (reflective boundary at $z = 0$) lifts the local value by roughly 30 percent.
- **LVK anchor**: LVK (2026) GWTC-5.0 90 percent CRs (BNS 5.1 to 154.7 Gpc$^{-3}$ yr$^{-1}$ at $z=0$, NSBH 6.7 to 32.8 Gpc$^{-3}$ yr$^{-1}$ at $z=0$) pinned in `grb_rates.LVK_GWTC5_LOCAL_RATES` / `LVK_GWTC5_PER_MODEL_RATES`. GWTC-5.0 added no new BNS or NSBH, so the NS-side bands tightened relative to GWTC-4; the Model A BHNS rate (73.0, intrinsic) sits above the NSBH band.

## Testing

```bash
make ci         # lint + typecheck + smoke; the local pre-PR gate
make smoke      # no slow, data-bound or COMPAS-bound tests
make coverage   # unit + anchors with a 70 percent coverage floor on grb_*.py
make test       # full suite; data-bound tests auto-skip if data/ is empty
make clean      # remove __pycache__, pytest, ruff, mypy and coverage artifacts
```

GitHub Actions ([.github/workflows/pytest.yml](.github/workflows/pytest.yml)) runs two jobs on pushes to `main` and on pull requests into `main`: lint (`ruff format --check`, `ruff check`, `mypy`) and unit + anchor tests with the 70 percent coverage floor. A notebook-execution job and the full suite run on manual dispatch only, and skip their data-bound tests because CI has no `data/`.

See [tests/README.md](tests/README.md) for the per-folder, per-section layout and the marker reference.

## References

Every entry below is cited in `src/`, `grb_main.ipynb` or a test. arXiv identifiers were checked against the arXiv API on 2026-10-03.

- LVK (2026), [arXiv:2605.27226](https://arxiv.org/abs/2605.27226) (GWTC-5.0 population properties; intrinsic local merger rates used as the BNS / NSBH / BBH anchor)
- Abac et al. (2025), [arXiv:2508.18083](https://arxiv.org/abs/2508.18083), ApJL 1005, L51, 2026 (GWTC-4.0 population properties; NSBH minimum BH mass, inherited by GWTC-5.0)
- Abbott et al. (2019a), [arXiv:1811.12907](https://arxiv.org/abs/1811.12907), PRX 9, 031040 (GWTC-1; source of the GW170817 component-mass anchor, Table III low-spin)
- Abbott et al. (2019b), [arXiv:1805.11579](https://arxiv.org/abs/1805.11579), PRX 9, 011001 (GW170817 properties; superseded as the mass anchor by GWTC-1)
- Abbott et al. (2020), [arXiv:2001.01761](https://arxiv.org/abs/2001.01761), ApJL 892, L3 (GW190425)
- Abbott et al. (2021), [arXiv:2106.15163](https://arxiv.org/abs/2106.15163), ApJL 915, L5 (GW200105 and GW200115; NSBH anchors, Table 2)
- Alsing, Silva, and Berti (2018), [arXiv:1709.07889](https://arxiv.org/abs/1709.07889), MNRAS 478, 1377
- Asplund, Grevesse, Sauval, and Scott (2009), [arXiv:0909.0948](https://arxiv.org/abs/0909.0948), ARA&A 47, 481 ($Z_\odot = 0.0142$)
- Bailyn, Jain, Coppi, and Orosz (1998), [arXiv:astro-ph/9708032](https://arxiv.org/abs/astro-ph/9708032), ApJ 499, 367 (lower BH mass gap)
- Bardeen, Press, and Teukolsky (1972), [ADS 1972ApJ...178..347B](https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B)
- Bauswein, Baumgarte, and Janka (2013), [arXiv:1307.5191](https://arxiv.org/abs/1307.5191), PRL 111, 131101 (prompt-collapse threshold)
- Bauswein et al. (2020), [arXiv:2004.00846](https://arxiv.org/abs/2004.00846), PRL 125, 141103 (Table I: $M_\mathrm{thres}$ per EOS and the $k$ range)
- Bavera et al. (2020), [arXiv:1906.12257](https://arxiv.org/abs/1906.12257), A&A 635, A97
- Behroozi, Wechsler, and Conroy (2013), [arXiv:1207.6105](https://arxiv.org/abs/1207.6105), ApJ 770, 57 (stellar-to-halo mass relation)
- Binney and Tremaine (2008), *Galactic Dynamics*, 2nd ed., Princeton University Press
- Biscoveanu, Landry, and Vitale (2022), [arXiv:2207.01568](https://arxiv.org/abs/2207.01568), MNRAS 518, 5298 (NSBHPop population model)
- Bloom, Sigurdsson, and Pols (1999), [arXiv:astro-ph/9805222](https://arxiv.org/abs/astro-ph/9805222), MNRAS 305, 763
- Broekgaarden et al. (2021), [arXiv:2103.02608](https://arxiv.org/abs/2103.02608), MNRAS 508, 5028 (paper I; the COMPAS model grid)
- Broekgaarden et al. (2022), [arXiv:2112.05763](https://arxiv.org/abs/2112.05763), MNRAS 516, 5737 (paper II; the model letters used on Zenodo and in figure labels)
- Colombo et al. (2022), [arXiv:2204.07592](https://arxiv.org/abs/2204.07592), ApJ 937, 79
- Crameri, Shephard, and Heron (2020), Nat. Commun. 11, 5444
- Dietrich and Ujevic (2017), [arXiv:1612.03665](https://arxiv.org/abs/1612.03665)
- Du et al. (2025), ApJL 987, L13 (Swift/BAT, Fermi/GBM and Konus-Wind short-GRB local rates)
- Dutton and Maccio (2014), [arXiv:1402.7073](https://arxiv.org/abs/1402.7073), MNRAS 441, 3359 (halo concentration)
- Farr et al. (2011), [arXiv:1011.1459](https://arxiv.org/abs/1011.1459), ApJ 741, 103 (lower BH mass gap)
- Finn and Chernoff (1993), [ADS 1993PhRvD..47.2198F](https://ui.adsabs.harvard.edu/abs/1993PhRvD..47.2198F)
- Fong and Berger (2013), [arXiv:1307.0819](https://arxiv.org/abs/1307.0819), ApJ 776, 18
- Fong et al. (2015), [arXiv:1509.02922](https://arxiv.org/abs/1509.02922), ApJ 815, 102
- Fong et al. (2022), [arXiv:2206.01763](https://arxiv.org/abs/2206.01763), ApJ 940, 56 (BRIGHT short-GRB host galaxies I; offset catalogue)
- Foucart, Hinderer, and Nissanke (2018), [arXiv:1807.00011](https://arxiv.org/abs/1807.00011), PRD 98, 081501
- Foucart et al. (2019), [arXiv:1903.09166](https://arxiv.org/abs/1903.09166), PRD 99, 103025
- Fragos et al. (2010), [arXiv:1001.1107](https://arxiv.org/abs/1001.1107), ApJL 719, L79
- Fryer et al. (2012), [arXiv:1110.1726](https://arxiv.org/abs/1110.1726), ApJ 749, 91
- Fuller and Ma (2019), [arXiv:1907.03714](https://arxiv.org/abs/1907.03714), ApJL 881, L1
- Gehrels (1986), ApJ 303, 336
- Gerosa et al. (2018), [arXiv:1808.02491](https://arxiv.org/abs/1808.02491), PRD 98, 084036
- Ghirlanda et al. (2016), [arXiv:1607.07875](https://arxiv.org/abs/1607.07875), A&A 594, A84
- Gottlieb et al. (2023), [arXiv:2309.00038](https://arxiv.org/abs/2309.00038), ApJL 958, L33
- Gottlieb et al. (2025), [arXiv:2411.13657](https://arxiv.org/abs/2411.13657), ApJ 984, 77 (posted 2024; the code cites it as Gottlieb 2024 for the four-class hybrid and Gottlieb 2025 for the HMNS disk-wind fraction and lifetime)
- Hernquist (1990), [ADS 1990ApJ...356..359H](https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H)
- Hurley, Tout, and Pols (2002), [arXiv:astro-ph/0201220](https://arxiv.org/abs/astro-ph/0201220), MNRAS 329, 897 (BSE; $\zeta_\mathrm{ad}$)
- Kawaguchi et al. (2015), [arXiv:1506.05473](https://arxiv.org/abs/1506.05473), PRD 92, 024014 (BH spin misalignment)
- Kish (1965), *Survey Sampling*, Wiley (effective sample size $n_\mathrm{eff} = (\sum w)^2 / \sum w^2$)
- Koppel, Bovard, and Rezzolla (2019), [arXiv:1901.09977](https://arxiv.org/abs/1901.09977), ApJL 872, L16
- Kroupa (2001), [arXiv:astro-ph/0009005](https://arxiv.org/abs/astro-ph/0009005), MNRAS 322, 231
- Kruger and Foucart (2020), [arXiv:2002.07728](https://arxiv.org/abs/2002.07728), PRD 101, 103002
- Lattimer and Prakash (2001), [arXiv:astro-ph/0002232](https://arxiv.org/abs/astro-ph/0002232), ApJ 550, 426
- Levan et al. (2024), [arXiv:2307.02098](https://arxiv.org/abs/2307.02098), Nature 626, 737
- Levina et al. (2026), [arXiv:2601.20202](https://arxiv.org/abs/2601.20202)
- Madau and Dickinson (2014), [arXiv:1403.0007](https://arxiv.org/abs/1403.0007), ARA&A 52, 415
- Madau and Fragos (2017), [arXiv:1606.07887](https://arxiv.org/abs/1606.07887), ApJ 840, 39
- Mandel and Muller (2020), [arXiv:2006.08360](https://arxiv.org/abs/2006.08360), MNRAS 499, 3214
- Mandel et al. (2026), [arXiv:2512.12018](https://arxiv.org/abs/2512.12018) (maximum host offsets of BNS mergers)
- Margalit and Metzger (2017), [arXiv:1710.05938](https://arxiv.org/abs/1710.05938), ApJL 850, L19
- Navarro, Frenk, and White (1997), [arXiv:astro-ph/9611107](https://arxiv.org/abs/astro-ph/9611107), ApJ 490, 493
- Neijssel et al. (2019), [arXiv:1906.08136](https://arxiv.org/abs/1906.08136), MNRAS 490, 3740
- Pais, Piran, Kiuchi, and Shibata (2024), [arXiv:2407.19002](https://arxiv.org/abs/2407.19002), ApJ 976, 35 (BNS jet breakout; cited as Pais et al. 2025 in the code)
- Patton and Sukhbold (2020), [arXiv:2005.03055](https://arxiv.org/abs/2005.03055), MNRAS 499, 2803
- Planck Collaboration / Ade et al. (2016), [arXiv:1502.01589](https://arxiv.org/abs/1502.01589), A&A 594, A13
- Pracchia and Salafia (2026), [arXiv:2601.03861](https://arxiv.org/abs/2601.03861), A&A 710, A267 (short-GRB delay times and $\mathcal{R}(z)$)
- Raaijmakers et al. (2021), [arXiv:2105.06981](https://arxiv.org/abs/2105.06981), ApJL 918, L29
- Radice et al. (2018), [arXiv:1809.11163](https://arxiv.org/abs/1809.11163), ApJL 869, L35 (viscous-dynamical ejecta; not a source for the disk-wind fraction)
- Rastinejad et al. (2022), [arXiv:2204.10864](https://arxiv.org/abs/2204.10864), Nature 612, 223 (GRB 211211A)
- Rastinejad et al. (2025), [arXiv:2409.02158](https://arxiv.org/abs/2409.02158), ApJ 979, 190 (uniform kilonova modelling)
- Read et al. (2009), [arXiv:0812.2163](https://arxiv.org/abs/0812.2163), PRD 79, 124032
- Salpeter (1955), [ADS 1955ApJ...121..161S](https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S)
- Scholz and Stephens (1987), JASA 82, 918 (k-sample Anderson-Darling)
- Skobe et al. (2026), [arXiv:2604.06586](https://arxiv.org/abs/2604.06586), ApJ 1009, 50 (kilonova-selected host offsets)
- Wanderman and Piran (2015), [arXiv:1405.5878](https://arxiv.org/abs/1405.5878), MNRAS 448, 3026
- Webbink (1984), [ADS 1984ApJ...277..355W](https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W) (CE $\alpha$-formalism)
- Xu and Li (2010), [arXiv:1004.4957](https://arxiv.org/abs/1004.4957), ApJ 716, 114 ($\lambda_\mathrm{CE}$)

## License

MIT. See [LICENSE](LICENSE).
