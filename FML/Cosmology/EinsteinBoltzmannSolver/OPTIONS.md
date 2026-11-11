# Options: nonlinear power, tensor and isocurvature modes, precision

The options of the solver beyond the standard ΛCDM run. Every parameter is listed with its description in
[params/default.ini](params/default.ini) (`./cmb --write-defaults`).

## Nonlinear power and helium

`nonlinear = halofit` adds the Halofit P_NL(k) (Takahashi et al. 2012, as CLASS's
`halofit` without its massive-neutrino correction (Bird et al. 2012): with m_nu > 0 up to ~2% lower than
CLASS, and the run warns) as a 4th column of `matter_power*.txt` and uses it for the lensing
potential (the Weyl potential of the Limber part scaled by √(P_NL/P_L)), so the lensed spectra include nonlinear
lensing. Against CLASS: P_NL to < 1e-3 at z = 0 and 1 for k ≤ 1 h/Mpc (`python/compare_class_halofit.py`, with
`pert_k_max = 1`), C_ℓ^φφ to 0.1% up to L = 2500, lensed TT 0.3%, EE 0.6% and BB 1% for ℓ ≤ 1500 (BB near ℓ_max needs
`lensing_delta_ell` ≳ 1500). `Yp_bbn = true` takes Y_p from standard BBN for the run's ω_b and N_eff (PArthENoPE 3.0
table, the CLASS default; 2e-5 from CLASS, which uses ΔN = N_eff − 3.046 with this table).

`nonlinear = halofit_bird` adds the updated Bird massive-neutrino correction; `nonlinear = hmcode2020`
selects gravity-only HMcode-2020 with cold-matter variances, neutrinos and BAO damping. Both use the same
nonlinear output and lensing paths. HMcode-2020 supports `lcdm` and smooth `w0wa` backgrounds (with curvature)
and rejects other models. `nonlinear = hmcode2016` selects HMcode-2016 (as CAMB `mead2016`, same backgrounds;
the general halo model in `FML/HaloModel`). The original Halofit option remains unchanged. See
[nonlinear methods, library API and validation](../Spectra/NONLINEAR.md) for units, limits and reference tests.

`nonlinear = react` adds the official ReACT reaction multiplied by an HMcode-2020 matched GR pseudo spectrum.
It requires an enabled optional backend (`USE_REACT=true`) and supports flat `lcdm`, `fr` with n=1, `w0wa`,
and normal-branch `dgp` with `dgp_lcdm_background=1`, including massive neutrinos. Reactions are restricted to
z<=2.5; above `react_z_max` the solver uses the matched pseudo baseline with R=1 and reports this approximation.
See [ReACT setup, conventions, reference tests and limits](../../ReACT/README.md).

`nonlinear = ee2` uses the EuclidEmulator2 boost (Euclid Collaboration: Knabenhans et al. 2021; `FML/Emulator`) for
flat `lcdm` and `w0wa` within its parameter ranges (an error outside them), tabulated at `ee2_n_z` redshifts up to
z = 10 (linear above) on its k-range (0.0087–9.4 h/Mpc; held above, 1 below). Its N_eff is 3.046.

`nonlinear = jbd_boost` (with `model = jbd`) is the prescription of Sletmoen & Winther (2024, arXiv:2403.03786; their
`bd.py`): P_BD(k) = P_BD^lin(k) · P_GR(k)/P_GR^lin(k) with the GR (ΛCDM) universe of the same physical densities
(ω_b, ω_c), n_s and σ₈ today and h_GR = h_BD √φ_ini, compared at the same k in h/Mpc of each universe; P_GR/P_GR^lin
from `jbd_boost_gr` (`ee2` as the paper, or `hmcode2020`, `hmcode2016`, `halofit`, `halofit_bird`). In the paper it is
1% accurate against N-body simulations for ω ≥ 100, z ≤ 3, k ≤ 1 h/Mpc (2% to 5 h/Mpc). Flat, with
`bd_density_parameter_definition = 0` (hi_class's, as the paper). The GR universe is solved too: with `ee2` only for
σ₈ (k ≤ 0.5/Mpc, ~6 s), with the other methods on the k-range of the run (a second perturbation solve). Against
`bd.py` (hi_class + EuclidEmulator2; ω = 100, ω_m = 0.15, ω_b = 0.02, h = 0.7, A_s = 2e-9, n_s = 1): h_GR to 1e-7,
A_s,GR to 2e-4 and the boost to 2e-4 at z = 0–3 for 0.009 ≤ k ≤ 9 h/Mpc (`tests/test_brans_dicke`; FML's emulator
agrees with `ee2.exe` to 1e-5). For ω → ∞ it is `ee2` for ΛCDM.

`nonlinear = axionhmcode` (with `model = axion`) uses the mixed cold/ultralight-axion halo model of
[AxionHMcode](../../AxionHMcode/README.md) (Vogt et al. 2022; the Dome et al. 2024 calibration, or the basic model with
`axionhmcode_basic = true`) on the solver's P_cb and P_axion, extrapolated to 1e-5 – 1e3 h/Mpc (P_cb with the
Eisenstein & Hu shape, P_axion/P_cb held beyond the k-range: set `pert_k_max` past the axion's Jeans scale). P_NL/P_L
of the total (of P_cb when the axion is not in P_m, m < 1e-25 eV) is tabulated at `axionhmcode_n_z` redshifts up to
`axionhmcode_z_max` (~1.5 s each) and 1e-3 – 100 h/Mpc; linear outside. Flat, no massive neutrinos. The Dome
calibration covers z = 1–8, axion fractions of the matter 0.01–0.3 and m = 10^-24.5 eV; elsewhere it is an
extrapolation (`axionhmcode_allow_extrapolation = false` makes it an error; the table always starts at z = 0), and far
from the pivot mass (10⁻²⁶ eV) it can fail (the run then suggests the basic model). Check (`tests/test_axion`): the
table equals AxionHMcode's total/linear total for the same input; for m = 10⁻²² eV (CDM-like) P_NL/P_L is within ~5%
of HMcode-2020 for ΛCDM up to k = 0.6 h/Mpc.

## Tensor modes

`r = 0.05` (tensor-to-scalar ratio at `k_pivot`) adds gravitational waves; the tilt is
n_t = −r/8 (2 − r/8 − n_s) (CLASS's `scc`) unless `n_t_consistency = false` and `n_t = ...`. The tensor TT, EE, TE
are added to the scalar spectra, BB is the tensor one, and all are lensed together. `cells_tensor.txt` holds the
tensor contributions alone (ℓ ≤ `tensor_ell_max`, default 600). The equations are those of CLASS (Pitrou et al.
hierarchy, flat), with the friction (2 + α_M)ℋh′ and the stress divided by M*² from the theory of gravity
(Horndeski α_M, Brans–Dicke φ, f(R) 1 + f_R; GR otherwise). Massive neutrinos are evolved with the massless
hierarchy and weighted by 3 p_ncdm in the anisotropic stress (equal to ρ_ncdm while they are relativistic; CLASS's
`tensor_method = massless` approximation). About 10 s on 2 cores (`[Tensors]` timing); settings `tensor_*` in
`params/default.ini`. Flat universes only.

| Check (tests/test_tensors, python/compare_class_tensors.py) | Result |
|---|---|
| Against CLASS, r = 0.1, unlensed tensor TT, EE, BB | 0.1–0.5% for ℓ ≤ 300; EE, BB 0.5% and TT 3% for 300–600 (where tensor TT is < 10⁻³ of the scalar) |
| Lensed total BB (scalar + tensor) against CLASS | 0.1% at ℓ = 50–300, 0.7% at ℓ = 10 |
| Neutrino free streaming (modes entering in the radiation era) | amplitude × 0.8026 (Weinberg 2004: 0.8026) |
| Matter era without anisotropic stress | h = 3 j₁(kη)/(kη)/√6 to 0.2% |
| Horndeski c_M = 0.5 | subhorizon h ∝ 1/(a M*) (WKB) to 1e-3 |

## Isocurvature modes

`f_cdi`, `f_bi`, `f_nid`, `f_niv` (CDM, baryon, neutrino density, neutrino velocity)
switch on isocurvature modes with P_II = f² A_s (k/k_pivot)^(n_I − 1) (`n_cdi`, ...) and correlation
P_RI = c √(P_RR P_II) (`c_ad_cdi`, ...; CLASS's parameters and sign convention, Σc² < 1; the isocurvature modes are
uncorrelated with each other). The correlation matrix is Cholesky-factored into independent components, each a
perturbation solve with a mix of adiabatic and isocurvature initial conditions (CLASS's superhorizon series,
Bucher, Moodley & Turok 2000), and the C_ℓ, C_ℓ^φφ and P(k) are summed over components; each active mode adds one
perturbation solve and line-of-sight pass. GR only (`model = lcdm`, `w0wa`, the dark-energy models; in
`dark_scattering`, `ide` and `coupled_quintessence` the dark energy starts adiabatic in the isocurvature modes); `python/params.py` passes the
modes to CLASS (`ic = ad,cdi,...`) so `python/compare_class_spectra.py` compares them directly (`--precise` for NIV).
Example: `params/isocurvature.ini`.

`f_axi` (`n_axi`, `c_ad_axi`) is the density isocurvature of the ultralight axion of `model = axion` (axionCAMB's
axion isocurvature): the frozen field starts with δρ_ax/ρ_ax = S and the rest unperturbed. Or `ax_log10_Hinf`, the
Hubble rate of inflation in GeV, which sets it from the field fluctuation δφ = H_inf/2π: f_axi² A_s = H_inf²/(π²φ_i²)
(φ_i the initial field in reduced Planck masses), n_axi = 1 − r/8 with r = 2H_inf²/(π²A_s M_pl²), uncorrelated.
Against axionCAMB (m = 10⁻²⁵ eV, 10% of the dark matter, `ax_switch_mH = 3`): the transfer functions of the mode to
3e-3, the C_ℓ to 3e-3 in shape; axionCAMB's amplitude is larger by 1/h² (its `phiinit` is h times the field).

| Check (tests/test_isocurvature) | Result |
|---|---|
| Pure modes against CLASS (fine sampling, `l_max_scalars` = 1500), 10 ≤ ℓ ≤ 600 | CDI, NID < 5e-4 in TT and EE; NIV ~1e-3 (CLASS's NIV at ℓ = 2 is not converged) |
| f = 1, n = 1.1, c_ad = 0.3 with the adiabatic mode, `precision = fast` | TT, EE, TE < 3e-3 (CDI, NIV); NID with massive neutrinos at the adiabatic level |
| Decomposition, initial entropy, independence of x_start | exact; S_c = −1; < 1e-6 (CDI), < 1e-3 (NID, NIV) |
| CDI in `dark_scattering`/`ide` (zero coupling) and `coupled_quintessence` (β = 0) | ds = ide (identical); cq = ΛCDM to 2e-6 |

## Fast precision preset for parameter fits

`precision = fast` (in a file or on the command line) sets the
precision parameters to values tuned for CMB likelihoods: lensed TT, EE, TE to ℓ = 2500 in ~13–16 s instead of
~70 s (2 cores), with Δχ² ≈ 0.2 relative to a high-precision run for Planck-like noise (f_sky = 0.6,
33 μK·arcmin, 7′ beam; the defaults give 0.02). The main saving is a log-linear k-grid (`pert_dk_max`: the CMB
sources oscillate in k with period ~2π/r_s, so above k ≈ 0.03/Mpc a linear spacing of 0.004/Mpc is enough;
112 modes instead of 400); the rest are a coarser stored x-grid, `lmax_nu = 16`, `ode_abserr = 1e-6` and a
coarser line-of-sight grid. Explicit precision parameters override the preset. `python/precision_scan.py`
computes Δχ² of any run against a reference (see its header for the reference settings used).

| Setting (Planck 2018 ΛCDM) | Time | Δχ² (Planck noise) | Δχ² (cosmic variance) |
|---|---|---|---|
| default | 70 s | 0.02 | 5.4 |
| `precision = fast` | 13–16 s | 0.22 | 5.8 |

## Comparing the models

Models beyond ΛCDM: `./cmb params/musigma.ini`, or `model=<name>` plus its parameters on the command line. To
compare all models with ΛCDM (H(z), C_ℓ^TT, C_ℓ^φφ, P(k); ~70 s per model on 2 cores):

```
for m in "lcdm" "musigma model=musigma mu0=0.3 Sigma0=0.1" "ndgp model=dgp dgp_lcdm_background=1" \
         "sdgp model=dgp Omega_rc=0 dgp_branch=1" "quintessence model=quintessence quint_lambda=1" \
         "jbd model=jbd bd_omega=100" "fr model=fr fR0=1e-5 pert_k_max=1" \
         "coupled_quintessence model=coupled_quintessence" "dark_scattering model=dark_scattering" \
         "cubic_galileon model=cubic_galileon" "horndeski model=horndeski hd_cK=1 hd_cB=0.6 hd_cM=0.3" \
         "ede model=ede ede_fEDE=0.12 ede_log10zc=3.56 ede_theta_i=2.83" "ide model=ide"; do
  set -- $m; name=$1; shift
  ./cmb lensing=true transfer_ells= output_dir=data/models/$name "$@" > data/models/$name.log; done
python3 python/plot_models.py                # data/models/models.png
```

## The k-range of the line of sight

The line-of-sight integration runs over the perturbation k-range unless `los_k_max` is set. When `pert_k_max` is
raised only for P(k) (f(R) above: 1/Mpc), `los_k_max = 0.3` keeps the C_ℓ cost of the default run (as in
`params/fr.ini`). Do not set it much lower than the default for the C_ℓ: the check k_max r_* > ℓ is only a minimum
(ℓ ≤ 500 needs 0.1/Mpc for 2% in TT, and 0.045/Mpc gives 47%).
