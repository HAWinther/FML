# Cosmology fitting (FML/CosmologyFitting)

`./mcmc` samples the posterior of any cosmological or model parameter given background or CMB data. It uses an
adaptive Metropolis–Hastings sampler with several chains in parallel (OpenMP, or MPI), and the
chains are written in GetDist format. With `sampler = minimize` it finds the best fit instead, optionally with a
profile likelihood and the Hessian covariance (see "Best fits and profile likelihoods" below).

The theory is the Einstein–Boltzmann solver of `FML/Cosmology` (see `../Cosmology/EinsteinBoltzmannSolver/README.md`
for its parameters), the samplers are in `FML/MCMC` (namespace `FML::MCMC`: `Posterior`, `Sampler`, `Minimizer`),
and this folder holds the cosmology-specific parts (namespace `FML::COSMOLOGY::FITTING`): `Theory` (what the
likelihoods need from a parameter point, computed only as far as needed), `Likelihoods`, `SampledParameterMap`
(a sampled point → the solver's `RunSettings`, with the aliases), the program `./mcmc` (`Main.cpp`), the parameter
files (`params/`) and the data (`external/`). Build and run from this folder (GSL as for the solver; `FML_DIR` as
there):

```
make                                               # ./mcmc  (make clean && make USE_MPI=true for MPI)
make test                                          # tests/test_fitting (the samplers: make run in ../MCMC/tests)
python3 python/fetch_sn_data.py                    # Pantheon+ and DES-SN5YR into external/sn/ (87 MB, once)
./mcmc params/mcmc_lcdm.ini                        # flat LCDM: DESI DR2 + Planck distance priors + Pantheon+
./mcmc params/mcmc_w0wa.ini                        # w0-wa dark energy, same data
./mcmc params/mcmc_desi_bbn.ini                    # LCDM from DESI DR2 BAO + BBN alone
python3 python/plot_chains.py chains/lcdm_desi_planckdp_pantheonplus   # triangle plot and 68% limits (getdist)
```

The parameter file uses the format of `./cmb`. Any `./cmb` parameter fixes that value (for example `model = w0wa`
or `m_nu = 0.06`), and `name=value` arguments after the file replace its entries (including `sample_*`). The
MCMC-specific entries are:

| Entry | Meaning |
|---|---|
| `likelihoods = desi_dr2, planck_dp, pantheonplus` | Data: supernovae `pantheonplus` (Pantheon+, 1590 light curves with z > 0.01, STAT+SYS covariance), `desy5` (DES-SN5YR, 1829), `union3` (Union3, 22 bins), each with the absolute magnitude marginalized analytically (as Cobaya; `sn`, `sn_free_amplitude`: the binned JLA data of the course), `desi_dr2`, `desi_dr1` (BAO with full covariance), `boss_dr12` (BOSS DR12 consensus BAO + full shape: D_M, H and fσ_8 at z = 0.38, 0.51, 0.61), `boss_dr12_bao` (its BAO only), `eboss_dr16_lrg`, `eboss_dr16_qso` (SDSS DR16 BAO + RSD of the LRGs, including BOSS DR12 at z = 0.38, 0.51, and of the quasars; do not combine `eboss_dr16_lrg` with `boss_dr12`), `desi_dr1_shapefit` (DESI DR1 full shape, ShapeFit compression + BAO; not with `desi_dr1`), `kids1000_cosebis` (KiDS-1000 cosmic shear, COSEBIs; needs `nonlinear`, see below), `planck_dp` (Planck 2018 R, l_A, ω_b of Chen, Huang & Wang 2019, LCDM-type early universe), `bbn` (ω_b = 0.02218 ± 0.00055); CMB: `planck_lite_TTTEEE`, `planck_lite_TT` (+ `_lowl`), `planck_lensing`, `planck_lensing_marged`, `cmb_mock` (below) |
| `sample_<name> = start, min, max, step [, label]` | Sampled parameter with a flat prior on [min, max]. Valid names are any `./cmb` parameter, a model parameter such as `w0`, a nuisance parameter (`A_planck`, `kids_A_IA`, `kids_dz1`..`kids_dz5`), or the aliases `H0`, `omega_b`, `omega_cdm`, `Omega_m`, `logA` = ln(10¹⁰A_s), `tau_reio` (sets z_reion), `theta_MC_100` (sets h), and for Brans–Dicke `log10_bd_omega` and `inv_bd_omega` = 1/ω (0 = GR) |
| `gauss_<name> = mean, sigma` | Additional Gaussian prior on a sampled parameter |
| `mcmc_chains`, `mcmc_burn_in`, `mcmc_adapt_every`, `mcmc_min_samples`, `mcmc_max_steps`, `mcmc_Rminus1`, `mcmc_check_every`, `mcmc_seed`, `mcmc_threads`, `mcmc_output` | Sampler settings (defaults 4, 2000, 100, 4000, 100000, 0.01, 1000, 1, one thread per chain (1 with CMB likelihoods), `chains/run`) |
| `mcmc_resume = true` | Continue an interrupted run from its checkpoints (`<root>_<i>.checkpoint`, written with the chains after every round of `mcmc_check_every` steps; `mcmc_checkpoint = false` turns them off). Works with any number of MPI processes |
| `mcmc_oversample_fast = 4` | Fast-only updates per step of the nuisance parameters (`A_planck`) |
| `mcmc_covmat = file` | Initial proposal covariance (GetDist `.covmat`, e.g. from an earlier run); parameters missing from it get their step² |
| `sampler = mcmc` | `mcmc` (default), `minimize` (best fit, below) or `hessian` (the next entry) |
| `mcmc_hessian = file` | Instead of sampling: covariance from the numerical Hessian of −ln posterior at the start point (2d² + 1 evaluations, spread over MPI processes), written as a `.covmat`. The steps start at the proposal widths and are first adapted to 1.5 conditional σ in each direction (2d evaluations per pass, ≤ 4 passes; `hessian_adapt_steps = false` keeps the proposal widths) |
| `minimize_starts`, `minimize_max_evals`, `minimize_tol`, `minimize_scatter` | Minimizer settings (defaults 4 starts, 3000 slow evaluations per start, 0.01 in −2 ln P, starts after the first drawn from start + 1 × N(0, covmat)) |
| `minimize_profile = name, min, max, n` | After the best fit: profile −2 ln P in a sampled parameter at n values, the others minimized |
| `minimize_hessian = true` | After the best fit: covariance from the Hessian there, written to `<root>.covmat` (a proposal for the MCMC) |
| `mcmc_sound_horizon = recombination` or `fit` | How r_d and z_* are computed. `recombination` uses our RECFAST history (about 0.08 s per point). `fit` uses Aubourg et al. for r_d and Hu–Sugiyama for z_*; it needs only the background (10 ms per point) but biases l_A by about 1 |

How the sampler works:
- During the burn-in it adapts the proposal covariance and scale (Haario et al. 2001) towards 25% acceptance.
- At the end of the burn-in it freezes the covariance pooled from all chains.
- It stops once every chain has `mcmc_min_samples` samples and R−1 < `mcmc_Rminus1` (Gelman–Rubin).
- A point whose model cannot be built (an unphysical parameter or a failed shooting) is rejected.
- Fast/slow (Lewis 2013, as CosmoMC): likelihood nuisance parameters are fast. The theory of the last two slow
  points is cached, and each step is followed by `mcmc_oversample_fast` updates that move only the fast parameters
  (the proposal L z with z_slow = 0, L the Cholesky factor, i.e. their conditional covariance), with their own scale.
- Checkpoints: after every round the chains and the full state of each chain (position, proposal, scales,
  counters, burn-in history, random generator) are written; a resumed run continues exactly as an uninterrupted one.

The theory is evaluated only as far as the likelihoods need. Supernovae need only the background. BAO and the
distance priors also need the recombination history, which is computed without reionization: z_* (τ = 1) and the
drag epoch, as in CAMB and CLASS. The distance-prior quantities are computed with the conventions of Chen et al.,
because the published values depend on them.

The outputs are:
- `<root>_<i>.txt`: chain i, as weight, −ln posterior, parameters, derived parameters;
- `<root>.paramnames`, `.ranges`, `.covmat`;
- `<root>.summary.txt`: means, standard deviations, 68% limits and the best fit.

With several likelihoods the χ² = −2 ln L of each is also a derived parameter, `chi2_<likelihood>`.

The derived parameters are H0, ω_b, ω_c, Ω_m, Ω_DE and the age, plus r_d, h·r_d, z_* and 100θ_* when the
recombination history is computed, and 100θ_MC, τ, z_re, ln(10¹⁰A_s), 10⁹A_s e^(−2τ), σ_8 and
S_8 = σ_8 (Ω_m/0.3)^½ with CMB likelihoods (σ_8 and S_8 also with fσ_8 data).

### Redshift-space distortions (fσ_8)
The BOSS DR12 and eBOSS DR16 likelihoods measure fσ_8 as defined in CAMB (and used by CosmoMC and Cobaya):
fσ_8 = σ_8,vd²/σ_8 of the CDM + baryons, with σ_8,vd² = ∫ dk/k Δ_cb²(k) f(k) W²(k R_8) and the linear growth rate
f(k) = dln Δ_cb/dln a (`PowerSpectrum::get_fsigma8`; equal to f σ_8 for scale-independent growth, and correct for
scale-dependent growth such as massive neutrinos or modified gravity). The redshifts are collected from the
likelihoods (`TheoryOptions::growth_z`). With CMB likelihoods the values come from the same perturbation run; without
them only the matter power spectrum is computed (no C_ℓ; ~5 s per point with `precision = fast` on 2 cores). Against
CAMB 2.0.5 (Planck cosmology, one massive neutrino) fσ_8 agrees to 4e-5 at the default precision and 3e-4 with
`precision = fast` for z = 0–1.5, and σ_8 of the CDM + baryons to 2e-4 (`tests/test_fitting`). For Brans–Dicke with
ω = 10⁷ it equals ΛCDM to 4e-7. At the Planck 2018 cosmology (`preset = planck2018`, m_ν = 0.06 eV) χ² = 6.7
(`boss_dr12`, 9 values), 4.5 (`boss_dr12_bao`, 6), 5.2 (`eboss_dr16_lrg`, 9) and 10.0 (`eboss_dr16_qso`, 3).

`desi_dr1_shapefit` is DESI's DR1 full-shape likelihood in the ShapeFit compression (DESI 2024 V, arXiv:2411.12021),
as DESI's Cobaya likelihood `desi_shapefit_bao_all` (github.com/cosmodesi/desi-kp-cosmological-likelihoods): for BGS,
LRG1–3, ELG2 and QSO the BAO scalings q_iso, q_ap relative to the DESI fiducial cosmology, df = f√A_p over its fiducial
value and dm = m − m_fid, and q_∥, q_⊥ for Lyman-α (26 values; the means and covariances of the DESI likelihood files,
converted to `external/bao/desi_dr1_shapefit_*.txt`). The theory follows that likelihood: f = σ_8,θθ/σ_8,cb,
A_p = P_cb(k_p)/s³ and m = dln P_nw/dln k at k_p = 0.03/s h/Mpc with s = h r_d/(h r_d)_fid, and P_nw from cosmoprimo's
`peakaverage` BAO filter (ported: the envelopes of the extrema of P/P_EH,nw at the fiducial positions rescaled by
s). The fiducial (Planck 2018 ΛCDM of `cosmoprimo.fiducial.DESI`) is computed once, with the precision of the run.
Against DESI's likelihood with cosmoprimo 1.0.0 and CAMB 2.0.5 (`tests/test_fitting`, `precision = fast`): the q to
1e-6, df to 3e-4 and dm to 7e-4 (σ(dm) = 0.05–0.13) for two cosmologies away from the fiducial (one with
m_ν = 0.12 eV), χ² to 0.04 (36.61 against 36.65, 30.70 against 30.66); at the fiducial χ² = 32.805 (26 values).
~6 s per point with `precision = fast` on 2 cores.

### Brans–Dicke

`model = jbd` with `bd_omega` sampled directly or through `log10_bd_omega` (flat in log ω) or `inv_bd_omega` = 1/ω
(0 = GR, ω = 10¹²); ω is then a derived parameter. `bd_GeffG_today` (G_eff/G today, Joudaki et al.'s G_matter/G) can
be sampled as any model parameter. `params/mcmc_jbd_planck_boss_pantheonplus.ini` is the CMB + SN + BOSS fit of
Joudaki, Ferreira, Lima & Winther (2022, arXiv:2010.15278) with plik_lite + lensing, Pantheon+ and BOSS DR12 (BAO +
fσ_8), the neutrino mass free and `nonlinear = jbd_boost`; its header lists the DESI ShapeFit and KiDS-1000 variants.
~22 s per slow point on 2 cores.

### Cosmic shear (KiDS-1000)

`kids1000_cosebis` is the KiDS-1000 COSEBIs likelihood of Asgari et al. (2021, arXiv:2007.15633) with the data of the
KiDS release (`external/kids1000/`, made by `python/convert_kids1000.py`): E_n, n = 1–5, of the 15 pairs of the 5
tomographic bins, θ = 0.5′–300′, the release's covariance (which includes the shear-calibration uncertainty) and SOM
n(z). The theory is that of the release's CosmoSIS pipeline: Limber C_ℓ of shear and NLA intrinsic alignments
(`FML/WeakLensing/CosmicShear.h`; C1ρ_crit = 5·10⁻¹⁴ρ_crit = 0.01388, D(z) from the linear P at large scales) on the
nonlinear total-matter P(k, z) of the run (`Theory::pk_nl`, 400 k from 10⁻⁵ to 100 h/Mpc at z = 0–6.1 every 0.05),
COSEBIs from C_ℓ for ℓ = 2–10⁵, and the photo-z shifts applied as CosmoSIS does (cubic interpolation of the shifted
histograms, then a cubic spline). Nuisance parameters `kids_A_IA` (0 if not sampled) and `kids_dz1`–`kids_dz5` (their
prior mean if not sampled); the correlated Gaussian prior of the shifts is part of ln L. Baryonic feedback as KiDS:
`nonlinear = hmcode2016`, `hmcode2016_eta0_from_A = true` and `sample_hmcode2016_A` (η_0 = 0.98 − 0.12 A); any other
nonlinear method works (e.g. `jbd_boost` with `jbd_boost_gr = hmcode2016`, which passes A on to the GR universe).
Modified lensing enters through `Theory::lensing_Sigma(z)`, the quasi-static Σ = G_light/G_* of the Weyl potential
(Brans–Dicke g/φ(a), μ–Σ, cubic Galileon, f(R); 1 otherwise, an error for Horndeski): for Brans–Dicke with ω = 50
it differs from 1 by up to 2.3% and equals that of the full perturbations to 2e-7; with ω = 10⁷ (and `jbd_boost`)
χ² at the KiDS maximum posterior is that of ΛCDM. `params/mcmc_kids1000_lcdm.ini` has KiDS' priors (with
ln 10¹⁰A_s in place of S_8).

Against the KiDS chain (CosmoSIS/kcap, CAMB HMcode-2016; `tests/test_fitting`, `precision = fast`): at the maximum
posterior χ² = 82.19 against 82.25 and σ_8 = 0.83803 against 0.83797. At 14 chain points (8 posterior draws and the
extremes in A_IA, S_8, A and ω_c) |Δχ²| ≤ 0.14 except at Ω_m = 0.50 and 0.55 (0.25, 0.69; at the edge of the
posterior). The δ_z prior equals the chain's prior of its uncorrelated shift parameters. ~12 s per slow point on 2
cores (6 s for P(k), 6 s for the HMcode table); A_IA and the shifts are fast (0.06 s).

### Best fits and profile likelihoods

```
./mcmc params/mcmc_w0wa.ini sampler=minimize                                  # best fit -> chains/..._w0wa....minimum.txt
./mcmc params/mcmc_w0wa.ini sampler=minimize minimize_profile=w0,-1.05,-0.65,9 minimize_hessian=true
```

`sampler = minimize` maximizes the posterior with the same parameter file (`FML/MCMC/Minimizer.h`):
- **Method.** Nelder–Mead simplex (adaptive coefficients, Gao & Han 2012) in coordinates whitened by the covmat
  (`mcmc_covmat`, else the steps), so the first simplex has the size of the posterior in every direction. It needs
  no derivatives and tolerates the small numerical noise of the Boltzmann code. Once the simplex has shrunk to a
  spread < `minimize_tol` in −2 ln P it is restarted around the best point, until a restart gains < `minimize_tol`.
- **Fast/slow.** The simplex runs over the slow parameters; at each slow point an inner simplex minimizes the fast
  ones (`A_planck`) on the cached theory, so the cost is counted in theory evaluations of the slow parameters.
- **Starts.** `minimize_starts` independent minimizations, the first from the start values, the others scattered
  by the covmat; they run in parallel over MPI processes and OpenMP threads like the chains. Their spread in
  −2 ln P shows whether the minimum is global.
- **Objective.** −2 ln posterior: Gaussian priors (`gauss_*`, e.g. the τ and A_planck priors) count as data, flat
  priors only bound the parameters. `<root>.minimum.txt` lists −2 ln L, −2 ln P, all parameters and the χ² of each
  likelihood.
- **Profiles.** `minimize_profile` fixes one parameter at n values and minimizes the others from the best fit (in
  parallel), written to `<root>.profile_<name>.txt` (value, −2 ln L, −2 ln P, all parameters). Profiles are
  independent of priors and volume effects, unlike marginalized posteriors (e.g. for EDE or isocurvature
  fractions near zero).
- **Hessian.** `minimize_hessian = true` computes the Hessian at the best fit with adapted steps and writes its
  inverse to `<root>.covmat`, a proposal for a following MCMC.

| Check | Result |
|---|---|
| `MCMC/tests/test_minimizer`: correlated 4D Gaussian with a Gaussian prior, Rosenbrock, maximum on a prior boundary, two peaks with several starts | maximum to < 0.005σ, −2 ln P to 1e-4; (1, 1) to 2e-5; boundary to 1e-9; global peak found |
| Fast/slow on a Gaussian (2 slow + 2 fast) | maximum to 0.002σ with 73 slow evaluations (and ~4500 cheap fast-only calls) |
| Profile and Hessian of a Gaussian | ((v − μ)/σ_marg)² to 1e-4; covariance exact (1e-14) |
| Planck plik_lite TTTEEE + low-ℓ TT + τ prior, ΛCDM, `precision = fast` (146 slow evaluations, 31 min on 2 cores) | ω_b = 0.022383, ω_c = 0.12013, n_s = 0.9665, ln(10¹⁰A_s) = 3.049, τ = 0.056, H0 = 67.36, −2 ln L = 589.97; Planck 2018 best fit (TT,TE,EE+lowE+lensing, full likelihoods): 0.022383, 0.12011, 0.9661, 3.045, 0.054, 67.32 (our τ prior replaces lowE) |
| ... + `planck_lensing` (`params/mcmc_planck_lite_lensing_lcdm.ini`; 158 slow evaluations, 30 min) | ω_b = 0.022397, ω_c = 0.11982, n_s = 0.9670, ln(10¹⁰A_s) = 3.049, τ = 0.056, H0 = 67.49, σ_8 = 0.8129; χ² = 589.97 (plik_lite) + 8.88 (lensing, 9 bandpowers). Planck 2018 best fit (TT,TE,EE+lowE+lensing, full likelihoods): 0.022383, 0.12011, 0.9661, 3.045, 0.054, 67.32, σ_8 = 0.8120 |
| Mock TT, TE, EE, BB with f_cdi = 0.2, c_ad_cdi = 0.2, r = 0.05 (`params/mcmc_cmb_mock_iso_r.ini`, 10 μK′, ℓ ≤ 2000, f_sky = 0.4; 3 parameters, 30 s per point) | best fit f_cdi = 0.1997, c_ad_cdi = 0.2021, r = 0.0499 from (0.1, 0, 0.02), −2 ln L = 0.005 (110 evaluations); 2 starts agree to 0.004; Hessian σ(f_cdi) = 0.037, σ(c_ad_cdi) = 0.037 (correlation −0.49), σ(r) = 0.0052; a short MCMC from that covmat runs |
| Mock TT, TE, EE with 10% of the dark matter in an axion of 10⁻²⁶ eV (`params/mcmc_cmb_mock_axion.ini`, `model = axion`, 10 μK′, ℓ ≤ 2000, f_sky = 0.4; 2 parameters, 10 s per point) | best fit ax_fraction = 0.1006, ax_log10_m = −25.997 from (0.05, −25.5), −2 ln L = 0.0015 (52 evaluations); the second start reaches 0.007; Hessian σ(ax_fraction) = 0.016, σ(ax_log10_m) = 0.096 (correlation 0.91) |
| w0wa, DESI DR2 + distance priors + Pantheon+ (2 min on 2 cores incl. the profile) | best fit −2 ln L = 1423.017 (MCMC best sample 1423.07), w0 = −0.854, wa = −0.522; 4 starts agree to 0.004; Δχ² = −7.66 against the ΛCDM minimum (1430.675); profile Δχ²(w0 = −1) = 7.5; Hessian σ(w0) = 0.060, σ(wa) = 0.24 (MCMC 0.054, 0.22) |

### CMB fits

```
./mcmc params/mcmc_planck_lite_lcdm.ini                         # Planck 2018 plik_lite TTTEEE + low-ell TT, LCDM
./mcmc params/mcmc_planck_lite_lensing_lcdm.ini                 # ... + Planck 2018 lensing
make clean && make USE_MPI=true && mpirun -np 4 ./mcmc params/mcmc_planck_lite_lcdm.ini   # one chain per MPI process
(cd ../Cosmology/EinsteinBoltzmannSolver && ./cmb params/planck2018.ini m_nu=0.06 precision=fast \
   output_dir=../../CosmologyFitting/data/cmb_mock)          # fiducial for the mock data
./mcmc params/mcmc_cmb_mock_lcdm.ini                            # mock TT/TE/EE, must recover the fiducial
./mcmc params/mcmc_planck_lite_lcdm.ini sampler=minimize minimize_starts=1   # Planck best fit, ~30 min on 2 cores
```

- **Data.** `planck_lite_TTTEEE` is the Planck 2018 high-ℓ plik_lite likelihood (foreground-marginalized,
  binned TT 30 ≤ ℓ ≤ 2508, TE and EE 30 ≤ ℓ ≤ 1996; 613 bins with their covariance; Planck 2018 V) with the
  calibration A_planck (prior 1 ± 0.0025). `_lowl` adds the low-ℓ TT (2 ≤ ℓ ≤ 29) compressed into two Gaussian
  bins (Prince & Dunkley 2019). Data in `external/planck2018_lite/` (from planck-lite-py). The low-ℓ EE
  likelihood is not public in a simple form, so τ gets the Gaussian prior of Planck's lowE alone,
  τ = 0.0506 ± 0.0086. `planck_lensing` is the Planck 2018 lensing likelihood (Planck 2018 VIII): 9 bandpowers of
  [L(L+1)]² C_L^φφ/2π for 8 ≤ L ≤ 400 (conservative range, T+P reconstruction) with their covariance and the
  linear correction for the dependence of the N0/N1 biases on the CMB and lensing spectra (CMBlikes format, as
  Cobaya's `planck_2018_lensing.native`; data in `external/planck2018_lensing/`); it shares `A_planck` with
  plik_lite. `planck_lensing_marged` is the CMB-marginalized variant, for use without CMB power-spectrum data.
  `cmb_mock` is the exact Gaussian likelihood of TT, TE, EE for a fiducial spectrum with
  white noise and a beam (options `cmb_mock_file`, `_lmax`, `_fsky`, `_noise`, `_beam`), for tests and forecasts;
  `cmb_mock_bb = true` adds the lensed BB (e.g. `sample_r` with mock data made with `r > 0`). With `r > 0` (fixed or
  sampled) the theory includes the tensor modes (~10 s more per point).
- **Theory.** The full `./cmb` pipeline with the precision of the parameter file; use `precision = fast`
  (~15 s per point on 2 cores; at the Planck best fit Δχ² = 1.2 for plik_lite TTTEEE and 1.1 for TT
  relative to the default precision, 0.01 for lensing). With CMB likelihoods the chains of one process are advanced
  one after the other, and each evaluation uses all OpenMP threads.
- **Parameters.** Sample `omega_b, omega_cdm, theta_MC_100, tau_reio, logA, n_s` (+ `A_planck`) as Planck does:
  θ_MC (a cheap proxy of the acoustic angle, Hu–Sugiyama z_* and r_s from the background) removes the
  H0–ω_m degeneracy from the proposal. θ_MC is CosmoMC's (CAMB `cosmomc_theta`: Hu–Sugiyama z_*, baryon loading
  R = 3·10⁴ a ω_b), so the Planck values apply: 1.04090 at the Planck 2018 TT,TE,EE+lowE mean (Planck: 1.04090);
  100θ_* (τ = 1) is derived and agrees with Planck (1.04103 vs 1.04110 ± 0.00031).
- **Proposal.** `params/mcmc_planck_lite_lcdm.covmat` comes from `mcmc_hessian` at the Planck best fit (99
  evaluations, 25 min on 2 cores), so the burn-in can be short (600 steps). Its errors are already Planck's:
  σ(ω_b) = 0.000145, σ(ω_c) = 0.0013, σ(100θ_MC) = 0.00031, σ(τ) = 0.0085, σ(ln 10¹⁰A_s) = 0.017,
  σ(n_s) = 0.0041 (Planck 2018 TT,TE,EE+lowE: 0.00015, 0.0014, 0.00031, 0.0073, 0.016, 0.0044).
- **MPI.** `make USE_MPI=true` (needs `mpicxx`; macOS `brew install open-mpi`; `make clean` first) builds `./mcmc` with MPI: chain i runs on
  process i mod N, and the burn-in covariance, R−1 and the output are combined over the processes. Use one
  process per chain (`mcmc_chains` ≥ N) and `OMP_NUM_THREADS` = cores per process, e.g. 4 processes × 8 threads.
  With the same seed it gives the same chains as without MPI (up to rounding in the last digits).
- **Cost.** A converged Planck ΛCDM run (4 chains, R−1 < 0.02) needs roughly 3000–5000 steps per chain after the
  burn-in, i.e. ~15,000–20,000 evaluations: about a day with 4 MPI processes of 8 cores each, or several days on a
  laptop.

Validation:
- `../MCMC/tests` checks the sampler (correlated Gaussian, flat and Gaussian priors, rejected regions, fast/slow,
  checkpoints) and the minimizer, profiles and Hessian (above).
- `tests/test_fitting` checks the distances, r_d against CLASS (1e-4), and the distance priors at the Planck best fit
  (R and l_A within 0.3σ).
- With DESI DR2 + BBN (`params/mcmc_desi_bbn.ini`, 4 × 4000 samples, about 25 min on 2 cores) we get
  Ω_m = 0.2975 ± 0.0085, h·r_d = 101.53 ± 0.72 Mpc and H0 = 68.53 ± 0.60, the same as the DESI DR2 analysis.
- The plik_lite implementation reproduces planck-lite-py (which agrees with Cobaya's plik_lite) to 1e-13 in ln L
  for all four variants. The lensing likelihood reproduces Cobaya's `planck_2018_lensing.native` and `.CMBMarged`
  to 1e-14 (tests/test_fitting); at the Planck 2018 cosmology χ² = 8.7 for its 9 bandpowers, and `precision = fast`
  changes it by 0.008. At the Planck best fit with `precision = fast`, our 100θ_* = 1.04103 and σ_8 lie within
  1σ of Planck, and χ² = 592 for 615 bins.
- The supernova likelihoods agree with an independent Python implementation (numpy, CLASS distances) to 1e-5 in
  χ²: at the Planck 2018 cosmology χ² = 1413.5 (Pantheon+, 1590), 1655.4 (DES-SN5YR, 1829), 29.9 (Union3, 22).
- DESI DR2 + Planck distance priors + Pantheon+ (`params/mcmc_lcdm.ini`, `mcmc_w0wa.ini`; 15 and 25 min on 2
  cores) gives, in ΛCDM, H0 = 68.35 ± 0.28 and Ω_m = 0.3016 ± 0.0036; in w0wa, w0 = −0.848 ± 0.054,
  wa = −0.55 ± 0.22 and Ω_m = 0.311 ± 0.006, with Δχ² = −7.6 relative to ΛCDM. This is close to the DESI DR2
  result with the full CMB likelihoods and Pantheon+ (w0 ≈ −0.84, wa ≈ −0.6; arXiv:2503.14738).

## Files

| File | Contents |
|---|---|
| `Theory.h/.cpp` | Distances, r_d, z_*, distance-prior quantities, θ_MC, θ_*, lensed CMB spectra, τ, σ_8, σ_8(z) and fσ_8(z), linear P_cb(k, z), nonlinear P(k, z) and D(z) (each computed only when needed); nuisance parameters |
| `Likelihoods.h/.cpp` | SN (Pantheon+, DES-SN5YR, Union3, JLA bins), BAO (DESI DR1/DR2), BAO + RSD (BOSS DR12, eBOSS DR16), DESI DR1 ShapeFit, KiDS-1000 COSEBIs, Planck distance priors, BBN, Planck 2018 plik_lite and lensing, mock CMB |
| `SampledParameterMap.h/.cpp` | A sampled point → `RunSettings`, with the aliases (H0, omega_b, omega_cdm, Omega_m, logA, tau_reio, theta_MC_100) |
| `Main.cpp` | `./mcmc`: MCMC, best fits, profiles and Hessians from a parameter file |
| `params/` | `mcmc_*.ini` and the Planck proposal covmat |
| `python/plot_chains.py` | Triangle plots and marginalized limits of `./mcmc` chains (GetDist) |
| `python/fetch_sn_data.py` | Downloads Pantheon+ and DES-SN5YR into `external/sn/` |
| `python/convert_kids1000.py` | Converts the KiDS-1000 data release into `external/kids1000/` |
| `external/` | DESI DR1/DR2 BAO, DESI DR1 ShapeFit, BOSS DR12 and eBOSS DR16 BAO + RSD (`bao/`), Planck 2018 plik_lite (`planck2018_lite/`) and lensing (`planck2018_lensing/`), KiDS-1000 COSEBIs (`kids1000/`), supernovae (`sn/`: Union3; Pantheon+ and DES-SN5YR via the script), binned JLA supernovae |
| `chains/` | Summaries and plots of the runs quoted above |
| `tests/` | `test_fitting`: theory and likelihoods |

## Possible next steps

ACT DR6 lensing and the full (clik) Planck likelihoods, fast–slow dragging in the sampler, DES-Dovekie supernovae,
and a converged Planck LCDM run compared with Planck.
