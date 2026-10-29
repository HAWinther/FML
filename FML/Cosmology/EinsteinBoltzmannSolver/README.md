# Einstein–Boltzmann solver (FML/Cosmology)

A C++17 Einstein–Boltzmann code for the CMB and large-scale structure. It computes the background expansion,
recombination, linear perturbations and line-of-sight power spectra of ΛCDM, its main extensions and several
dark-energy and modified-gravity models, and is validated against CLASS module by module.

The solver is split into the folders of `FML/Cosmology/`: `Background`
(parameters, models = species + gravity, background, massive neutrinos, BBN), `Models` (the dark-energy and
modified-gravity models), `Recombination`, `Perturbations` (scalar, isocurvature and tensor modes), `Spectra`
(line-of-sight C_ℓ, lensing, Halofit, two-point functions) and this folder (the parameter files and the programs).
Everything is in namespace `FML::COSMOLOGY`. Parameter fits use it through `FML/CosmologyFitting` (likelihoods,
`./mcmc`) and the samplers of `FML/MCMC`. The original teaching solver is in `Cosmology/Legacy`.


```
CosmologyParameters ─► BackgroundCosmology ─► RecombinationHistory ─► Perturbations ─► PowerSpectrum ─► Lensing, TwoPoint
                              ▲                                              ▲
                            Model = species + gravity (registry: lcdm, musigma, dgp, quintessence, jbd, fr, coupled_quintessence, dark_scattering, cubic_galileon, horndeski, ede, ide)
```

Conventions: SI units internally, time variable `x = ln a`, Newtonian gauge (Dodelson / Callin conventions).

## Features

| Area | What is included |
|---|---|
| Models | A model is a list of species and a theory of gravity, selected by name in the parameter file (`model = ...`, see the table below) |
| Background | Photons, baryons, CDM, massless and one massive neutrino species, curvature (open/closed), Λ |
| Recombination | RECFAST 1.5 (default: effective three-level H and He I atoms with the fudge factors and Gaussian K_H correction, triplet He, evolved baryon temperature) or the course model (Saha → Peebles; `recombination = peebles`); tanh reionization (CAMB/CLASS `reio_camb`), τ, visibility function, sound horizon; Y_p fixed or from BBN (`Yp_bbn`, PArthENoPE table as CLASS) |
| Perturbations | CDM, baryons, photon temperature and polarization hierarchies, massless neutrinos, massive neutrinos (momentum-resolved Ma–Bertschinger hierarchy + CLASS fluid approximation), tight coupling, curvature; OpenMP over k |
| Spectra | C_ℓ^TT, EE, TE by line of sight (hyperspherical Bessel functions when curved, with CLASS's flat approximation), P(k) of total matter and CDM+baryons |
| Tensor modes | Gravitational waves (`r`, `n_t` or the consistency relation): photon temperature and polarization and neutrino hierarchies, anisotropic stress, tight coupling, modified-gravity friction (α_M, M*² of Horndeski, Brans–Dicke, f(R)); TT, EE, TE, BB by line of sight, added to the scalar spectra and lensed with them; flat |
| Isocurvature modes | CDM, baryon, neutrino density and neutrino velocity isocurvature (`f_*`, `n_*`, `c_ad_*` as CLASS), each correlated with the adiabatic mode; CLASS initial conditions; one extra perturbation solve per mode; GR only |
| Nonlinear | Halofit (Takahashi et al. 2012) P_NL(k, z) (`nonlinear = halofit`), used for nonlinear CMB lensing |
| Lensing | C_ℓ^φφ (LOS + Limber), lensed TT, EE, TE, BB (full-sky correlation-function method) |
| Two-point functions | CMB C(θ), ξ±, ξ_X with beams; matter ξ(r), σ_R, σ₈; angular C_ℓ and w(θ) of a projected tracer (exact + Limber) |
| Fitting | MCMC, best fits and profile likelihoods with background, Planck CMB and lensing data: `FML/CosmologyFitting` (samplers in `FML/MCMC`) |

| Model (`model =`) | Parameters | Physics |
|---|---|---|
| `lcdm` | – | ΛCDM in GR (with curvature, massive neutrinos) |
| `w0wa` | `w0`, `wa` | Dark energy with w(a) = w₀ + w_a(1 − a), smooth (no perturbations, so gauge-inconsistent for w ≠ −1: 2e-3 in P(k) at k = 0.001/Mpc); for background data. A wCDM fluid with perturbations: `dark_scattering` with `ds_xi = 0` |
| `musigma` | `mu0`, `Sigma0` | Planck/DES μ(a) = 1 + μ₀ Ω_DE(a)/Ω_DE,0 (growth) and Σ(a) (lensing); ΛCDM background; flat |
| `dgp` | `Omega_rc`, `dgp_branch`, `dgp_lcdm_background` | DGP braneworld: modified Friedmann equation (normal branch + Λ, or self-accelerating), quasi-static μ = 1 + 1/(3β), Σ = 1; optionally ΛCDM expansion; flat |
| `quintessence` | `quint_lambda` | Scalar field with V ∝ e^(−λφ), thawing from rest, with its perturbations; λ = 0 is ΛCDM |
| `jbd` | `bd_omega` | Jordan–Brans–Dicke + Λ: scalar-field background and perturbations; φ₀ gives the Newtonian G; flat |
| `coupled_quintessence` | `cq_lambda`, `cq_beta` | Quintessence (V ∝ e^(−λφ)) coupled to CDM (m ∝ e^(−βφ)): energy exchange, fifth force G(1+2β²), drag |
| `dark_scattering` | `ds_w`, `ds_xi`, `ds_cs2` | wCDM with a dark-energy fluid (perturbations, ĉ_s²) and elastic DM–DE scattering (momentum exchange, Simpson 2010); ξ = 0 is wCDM |
| `cubic_galileon` | – | Cubic Galileon on its tracker (no Λ): E² = [ρ + √(ρ² + 4Ω_φ)]/2, w₀ ≈ −1.19, with the full (braided) scalar perturbations; flat |
| `horndeski` | `hd_parametrization`, `hd_cK`, `hd_cB`, `hd_cM`, `hd_M2_ini`, `hd_w0`, `hd_wa`, `hd_stability_check` | Horndeski in the α parameterization of hi_class (α_T = 0): α_X = c_X f(a) with f = Ω_DE(a), a or 1, M*² from α_M; ΛCDM or w₀w_a expansion; full scalar perturbations; ghost, gradient and stiffness (c_s² ≤ 1e5) checks; α_K = α_B = 0 with α_M ≠ 0 is rejected; flat |
| `ede` | `ede_fEDE`, `ede_log10zc`, `ede_theta_i`, `ede_n` | Axion-like early dark energy V = m²f²(1 − cos φ/f)ⁿ (n > 2) + Λ: m and f found from the peak fraction f_EDE at z_c; full Klein–Gordon background and perturbations through the oscillations |
| `ide` | `ide_xi`, `ide_w`, `ide_cs2`, `ide_stability_check` | Interacting dark energy: fluid (w, ĉ_s²) exchanging energy with CDM, Q = ξHρ_DE along the CDM velocity; analytic background; doom-factor stability check |
| `fr` | `fR0`, `fR_n` | Hu–Sawicky f(R): ΛCDM background (to O(f_R0)), quasi-static μ(k,a) = (1+4Q)/((1+3Q)(1+f_R)), Σ = 1/(1+f_R), Q = (k/a)² f_RR/(1+f_R) = (k/a)²λ_C²/3; flat |

Agreement with CLASS (same physics, CLASS with RECFAST): background 1e-7, X_e 3e-4, z_* and r_s 1e-6, TT 5e-4
rms for ℓ = 30–1000 (3e-3 for 1000–2000), EE 3e-3 rms, P(k) 0.1%, lensed/unlensed TT ratio 1e-3, effect of the
neutrino mass 5e-4, σ₈ 4e-5. (The Peebles model of the course gives TT 0.6–0.8% and EE 1.4% rms.) Details
and the physics are in `report/paper.pdf`.

## Build

Requires a C++17 compiler and GSL (`brew install gsl` / `apt install libgsl-dev`). The Makefiles use FML's
`Makefile.config`; put the GSL location for your machine in `FML/Makefile.local` (e.g. `LOCAL_PREFIX = /opt/homebrew`).
Run the programs from this folder (the parameter files are in `params/`, outputs go to `data/`).

```
make                       # the solver ./cmb
make milestones benchmark  # ./milestones (report data) and ./benchmark
make test                  # build and run all unit tests (tests/test_*.cpp)
make USE_OMP=false         # without OpenMP
make USE_DEBUG=true        # with the DEBUG_OPTIONS of Makefile.config
```

Data files shipped with FML (the BBN table in `Cosmology/Background/external/`) are found through the FML folder
compiled into the programs; set the environment variable `FML_DIR` if FML is moved after building. FML compiles
with `-march=native` by default: results then differ from a build without it at the 1e-4 level (fused
multiply-adds), within all test tolerances.

## Running

```
./cmb                                     # fiducial model, outputs in data/
./cmb params/planck2018.ini               # parameters from a file
./cmb m_nu=0.06 lensing=true output_dir=data/test     # command-line values (override the file)
./cmb --write-defaults my.ini             # complete parameter file with all parameters and their defaults
```

Every cosmological, precision and output parameter can be set in a parameter file (`name = value`, `#` comments)
or as `name=value` on the command line. `params/default.ini` lists all of them with descriptions; it is generated
from the code (`./cmb --write-defaults`), so it is always complete. Unknown names and malformed values are errors.
Units: k in 1/Mpc, angles in degrees, x = ln a. `preset = planck2018` loads the Planck 2018 parameters first: the TT,TE,EE+lowE+lensing+BAO means with massless
neutrinos (as in the references and tests); `preset = planck2018_baseline` is Planck's baseline (TT,TE,EE+lowE+lensing
means, one massive neutrino of 0.06 eV; 100 θ_s = 1.04184 against 1.04033).
Each run writes `<output_dir>/params_used.ini` (re-readable) and `params.txt` (read by the Python scripts).

| Parameter file | Run |
|---|---|
| `params/default.ini` | All parameters at their defaults (the fiducial model) |
| `params/planck2018.ini` | Planck 2018 (preset `planck2018`, massless neutrinos), lensed spectra to ℓ = 2500 |
| `params/curvature.ini` | Curved universe (`OmegaK=...`), as in the report |
| `params/neutrinos.ini` | One massive neutrino with lensing (`m_nu=...`), as in the report |
| `params/twopoint.ini` | All two-point functions (perturbations to k = 1/Mpc), as in the report |
| `params/isocurvature.ini` | Planck 2018 with a correlated CDM isocurvature mode (f = 0.3, c_ad = −0.2) |
| `params/tensors.ini` | Planck 2018 with tensor modes, r = 0.05: lensed BB with the primordial B modes |
| `params/musigma.ini`, `dgp.ini`, `quintessence.ini`, `jbd.ini`, `fr.ini`, `coupled_quintessence.ini`, `dark_scattering.ini`, `cubic_galileon.ini`, `horndeski.ini`, `ede.ini`, `ide.ini` | The models beyond ΛCDM, with their parameters |

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

Outputs (in `output_dir`): `background.txt`, `recombination.txt`, `perturbations_k<k>.txt`, `cells.txt`
(D_ℓ^TT, EE, TE in μK²), `cells_lensed.txt`, `matter_power.txt` (k [h/Mpc], P_m, P_cb), `transfer.txt`, and
optionally `cmb_correlation_*.txt`, `matter_correlation.txt`, `angular_cl_z<z>.txt`, `angular_w_z<z>.txt`.

Run time of the fiducial model: ~45 s on 2 cores, dominated by the perturbations. `./benchmark` measures the
time of each stage for the main options (lensing +4 s, massive neutrinos +7 s, curvature +15–28 s, two-point
functions +3 s).

**Nonlinear power and helium.** `nonlinear = halofit` adds the Halofit P_NL(k) (Takahashi et al. 2012, as CLASS's
`halofit` without its massive-neutrino correction (Bird et al. 2012): with m_nu > 0 up to ~2% lower than
CLASS, and the run warns) as a 4th column of `matter_power*.txt` and uses it for the lensing
potential (the Weyl potential of the Limber part scaled by √(P_NL/P_L)), so the lensed spectra include nonlinear
lensing. Against CLASS: P_NL to < 1e-3 at z = 0 and 1 for k ≤ 1 h/Mpc (`python/compare_class_halofit.py`, with
`pert_k_max = 1`), C_ℓ^φφ to 0.1% up to L = 2500, lensed TT 0.3%, EE 0.6% and BB 1% for ℓ ≤ 1500 (BB near ℓ_max needs
`lensing_delta_ell` ≳ 1500). `Yp_bbn = true` takes Y_p from standard BBN for the run's ω_b and N_eff (PArthENoPE 3.0
table, the CLASS default; 2e-5 from CLASS, which uses ΔN = N_eff − 3.046 with this table).

**Tensor modes.** `r = 0.05` (tensor-to-scalar ratio at `k_pivot`) adds gravitational waves; the tilt is
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

**Isocurvature modes.** `f_cdi`, `f_bi`, `f_nid`, `f_niv` (CDM, baryon, neutrino density, neutrino velocity)
switch on isocurvature modes with P_II = f² A_s (k/k_pivot)^(n_I − 1) (`n_cdi`, ...) and correlation
P_RI = c √(P_RR P_II) (`c_ad_cdi`, ...; CLASS's parameters and sign convention, Σc² < 1; the isocurvature modes are
uncorrelated with each other). The correlation matrix is Cholesky-factored into independent components, each a
perturbation solve with a mix of adiabatic and isocurvature initial conditions (CLASS's superhorizon series,
Bucher, Moodley & Turok 2000), and the C_ℓ, C_ℓ^φφ and P(k) are summed over components; each active mode adds one
perturbation solve and line-of-sight pass. GR only (`model = lcdm`, `w0wa`, the dark-energy models; in
`dark_scattering`, `ide` and `coupled_quintessence` the dark energy starts adiabatic in the isocurvature modes); `python/params.py` passes the
modes to CLASS (`ic = ad,cdi,...`) so `python/compare_class_spectra.py` compares them directly (`--precise` for NIV).
Example: `params/isocurvature.ini`.

| Check (tests/test_isocurvature) | Result |
|---|---|
| Pure modes against CLASS (fine sampling, `l_max_scalars` = 1500), 10 ≤ ℓ ≤ 600 | CDI, NID < 5e-4 in TT and EE; NIV ~1e-3 (CLASS's NIV at ℓ = 2 is not converged) |
| f = 1, n = 1.1, c_ad = 0.3 with the adiabatic mode, `precision = fast` | TT, EE, TE < 3e-3 (CDI, NIV); NID with massive neutrinos at the adiabatic level |
| Decomposition, initial entropy, independence of x_start | exact; S_c = −1; < 1e-6 (CDI), < 1e-3 (NID, NIV) |
| CDI in `dark_scattering`/`ide` (zero coupling) and `coupled_quintessence` (β = 0) | ds = ide (identical); cq = ΛCDM to 2e-6 |

**Fast precision preset for parameter fits.** `precision = fast` (in a file or on the command line) sets the
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

## Code layout

| File (in `FML/Cosmology/`) | Contents |
|---|---|
| `Common.h` | Solver aliases and compatibility names for constants, numerical methods, splines, ODE solvers and timing. Implementations live in shared `FML/Units`, `Math`, `FileUtils`, `Timing` and `Spline` modules. |
| `Background/CosmologyParameters.h/.cpp` | Parameters with derived Ω's; `fiducial()`, `Planck2018()`, `Planck2018Baseline()` |
| `EinsteinBoltzmannSolver/ParameterFile.h/.cpp` | Parameter files: one table of all parameters (name, description, unit), reading with overrides and checks, writing |
| `Background/Species.h` | Interface of a species: background (ρ, p) and perturbations (variables, equations, moments, outputs) |
| `Background/Gravity.h` | Interface of a theory of gravity: E²(x) from the species, Ψ and Φ' from their moments |
| `Background/Model.h/.cpp` | `Model` (species + gravity) and the registry of built-in models with their parameters |
| `Models/` | `StandardSpecies` (baryons, photons, CDM, massless and massive neutrinos, Λ, CPL), `GeneralRelativity`, `MuSigma`, `DGP`, `Quintessence`, `BransDicke`, `HuSawicky`, `CoupledQuintessence`, `DarkScattering`, `CubicGalileon`, `HorndeskiAlpha`, `ScalarField` (shared by `Quintessence` and `EarlyDarkEnergy`), `InteractingDarkEnergy` |
| `Background/BackgroundCosmology.h/.cpp` | H, Hp and derivatives from the model, Ω_i(x), η, t, distances, curvature geometry `sinK`, `cotK` |
| `Background/MassiveNeutrinos.h/.cpp` | Fermi–Dirac momentum integrals and Gauss–Laguerre momentum grid |
| `Recombination/RecombinationHistory.h/.cpp` | X_e, T_b, τ, visibility function, sound horizon, reionization |
| `Perturbations/Perturbations.h/.cpp` | Photon–baryon core (tight coupling, polarization), the species' blocks, segmentation of the ODE, source functions; 2D splines in (x, k) |
| `Spectra/HypersphericalBessel.h/.cpp` | Hyperspherical Bessel functions Φ^ν_ℓ(χ) (open, closed, flat) |
| `Spectra/PowerSpectrum.h/.cpp` | Line-of-sight integration, C_ℓ, C_ℓ^φφ, lensed spectra, P(k), transfer functions |
| `Spectra/PowerSpectrumTwoPoint.cpp` | Two-point functions of `PowerSpectrum`: CMB C(θ), ξ(r), σ₈, projected C_ℓ and w(θ) |
| `Perturbations/Tensors.h/.cpp` | Tensor modes: gravitational-wave equation with photon and neutrino hierarchies, sources, line of sight, tensor C_ℓ |
| `Perturbations/Isocurvature.h/.cpp` | Isocurvature modes: correlation matrix → independent components (initial conditions in `Perturbations::isocurvature_values`) |
| `Spectra/Halofit.h/.cpp` | Nonlinear matter power spectrum (Takahashi et al. 2012) |
| `Background/BBN.h/.cpp` | Primordial helium from the BBN table (`external/bbn/`) |
| `Spectra/Lensing.h/.cpp` | Lensing of the CMB spectra (Challinor & Lewis 2005), Wigner d-functions |
| `Spectra/TwoPoint.h/.cpp` | Two-point numerics: Legendre/Wigner sums, Filon transform for ξ(r), σ_R, redshift windows |
| `EinsteinBoltzmannSolver/Main.cpp` | `./cmb`: runs the pipeline from a parameter file |
| `EinsteinBoltzmannSolver/tests/` | One unit-test program per module and model (~650 checks): analytic limits, conservation laws, convergence, internal consistency; `test_regression` compares key outputs of several configurations and all models with stored values (`tests/reference/`) |
| `EinsteinBoltzmannSolver/Milestones.cpp` | `./milestones`: all models and numbers for the report (`report/data/`) |
| `EinsteinBoltzmannSolver/Benchmark.cpp` | `./benchmark [config ...]`: run time per stage (`data/benchmark.txt`) |
| `EinsteinBoltzmannSolver/python/params.py` | Reads `params.txt` and builds the matching CLASS input |
| `EinsteinBoltzmannSolver/python/derive_brans_dicke.py`, `derive_cubic_galileon.py` | Symbolic derivations (sympy) of the Brans–Dicke and cubic Galileon equations |
| `EinsteinBoltzmannSolver/python/derive_horndeski.py`, `derive_horndeski_alpha.py` | Horndeski (F(φ)R/2 + G2 − G3□φ) in Newtonian gauge, then the equations in terms of α_K, α_B, α_M (run in this order) |
| `EinsteinBoltzmannSolver/python/compare_hiclass.py` | A `horndeski` run against hi_class (its `class` executable) |
| `EinsteinBoltzmannSolver/python/compare_class_ede.py` | An `ede` run against CLASS_EDE (its `class` executable) |
| `EinsteinBoltzmannSolver/python/compare_class_wcdm.py` | wCDM (`dark_scattering` with ξ = 0) against CLASS's fluid dark energy |
| `EinsteinBoltzmannSolver/python/plot_models.py` | Ratios of H(z), C_ℓ^TT, C_ℓ^φφ and P(k) of all models to ΛCDM |
| `EinsteinBoltzmannSolver/python/plot_*.py` | Quick-look plots of the outputs in `data/` (CLASS overlay if `classy` is installed) |
| `EinsteinBoltzmannSolver/python/compare_class_*.py` | Comparisons with CLASS (`pip install classy`): background, recombination, perturbations, spectra (CLASS with RECFAST; `--hyrec` for HyRec), lensing, neutrinos, two-point functions |
| `external/`, `../Background/external/` | Planck 2018 TT/EE/TE spectra, SDSS DR7 LRG and WMAP+ACT P(k), binned JLA supernovae (for the plots and the report); the BBN helium table (`Cosmology/Background/external/bbn/`). The likelihood data are in `FML/CosmologyFitting/external/` |

### Models

A model (`Background/Model.h`) is a list of species and a theory of gravity:

* A **species** (`Background/Species.h`) gives ρ(x), p(x) and the first two derivatives of ρ for the background and,
  if it clusters, a `SpeciesPerturbations` object: its variables (possibly in several approximation schemes,
  e.g. the massive-neutrino hierarchy and fluid), initial conditions, equations, and the quantities to store
  (`Perturbations::get(name, x, k)`). It talks to the metric only through its moments δρ, δp, (ρ+p)θ/(ck) and
  (ρ+p)σ, and receives Φ, Ψ and Φ'. Interacting species (coupled quintessence, dark scattering) are one
  perturbation block with all their variables. Photons and baryons are the fixed core of `Perturbations` (Thomson
  coupling, tight coupling, polarization and the line-of-sight sources).
* The **gravity** (`Background/Gravity.h`) gives E²(x) = (H/H0)² from the species densities, and owns its own
  perturbation variables (GR: Φ; Brans–Dicke, cubic Galileon and Horndeski: Φ and two scalar-field variables), from which, with the summed moments, it gives Φ, Ψ
  and Φ'. `GeneralRelativity` implements the Einstein equations; `MuSigmaGravity` the μ–Σ parameterization
  (evolving the GR potential of the comoving density with an equation derived from energy-momentum
  conservation, so that μ = Σ = 1 is GR exactly), which `DGPGravity` (own μ(a) and background) and
  `HuSawickyGravity` (scale-dependent μ(k,a)) reuse.
* The **registry** in `Background/Model.cpp` lists the built-in models with their parameters and defaults. A run
  selects one with `model = <name>`; its parameters are then valid in the parameter file (and a parameter of
  another model is an error). `./cmb --write-defaults` lists them.

To add a model: write its species and/or gravity in `Cosmology/Models/`, and add an entry to the registry.

Validation of the models (each test program prints what it checks): the GR/ΛCDM limit of every model;
μ–Σ and DGP potentials against the algebraic relations with the comoving density, and the growth against an
independent solution of the growth equation with μ(a) (to < 1e-4); the quintessence background against an
independent integration in cosmic time, and the Einstein constraints that are not evolved (Poisson and
momentum, with the field's density and momentum) to 1e-7; the Brans–Dicke background equations, the
matter-era solution φ ∝ a^(1/(1+ω)), the unused momentum constraint (5e-8), and the growth with
G_eff = G φ₀/φ; f(R): f_R(a), the Compton wavelength and the limits of μ, the potentials at every k, and
the scale-dependent growth at each k (to 2e-4); coupled quintessence: an independent cosmic-time integration
with the coupling, the Einstein constraints (which hold only if the exchange cancels) and the CDM growth with
fifth force and drag; dark scattering: the constraints with momentum exchange, the drag in the growth, and
for ξ = 0 the wCDM/ΛCDM ratios against CLASS's fluid dark energy (`python/compare_class_wcdm.py`: P(k) 5e-5,
C_ℓ^TT 2e-4 for ℓ ≤ 800); cubic Galileon: the tracker (E⁴ − ρE² = Ω_φ, the pressure from the field equations,
w = −2/(1+Ω_φ)), the unused momentum constraint with all braiding terms (3e-8) and the sub-horizon growth
against the quasi-static G_eff (7e-5); Horndeski (α's): all α = 0 is ΛCDM, the unused momentum constraint with
braiding, running Planck mass and kineticity (3e-8), the stability conditions, and three theories rebuilt from
their α's reproduce the dedicated implementations: the cubic Galileon (α_K = 6Ω_DE, α_B = 2Ω_DE; 5e-9),
quintessence (α_K = 3(ρ+p)_φ/H²; 2e-11) and Brans–Dicke on its own expansion (M*² = φ, α_M = −α_B,
α_K = ωα_M²; 9e-9, and the field perturbation). Against hi_class (`python/compare_hiclass.py`, propto_omega with
ΛCDM and w₀w_a expansion, propto_scale): the ratios to ΛCDM agree to 1e-4 in P(k) and 8e-4 in C_ℓ^TT.
Early dark energy: the shooting reproduces f_EDE and z_c (1e-10), the stored field satisfies the Klein–Gordon
equation, the oscillation-averaged dilution a^(−6n/(n+1)), the unused Einstein constraints with the field (5e-6),
and f_EDE → 0 is ΛCDM; against CLASS_EDE (`python/compare_class_ede.py`) the potential parameters agree (f to
1e-3) and the ratios to ΛCDM agree to 3e-5 in P(k) for k ≤ 0.05 h/Mpc (7e-4 at 0.1) and 1e-3 in C_ℓ^TT.
Interacting dark energy: ρ_c against an independent integration, the constraints with energy exchange (2e-6;
they test that the exchanged energy and momentum cancel), ξ = 0 against the separate wCDM implementation of
`dark_scattering` (identical), and the rejection of unstable (d > 0) and negative-ρ_c models.
`test_regression` stores key outputs of all models.

## Report

`report/paper.pdf` describes the physics, numerics and validation of every module (course milestones I–IV, without
the MCMC fits; these use the course's Peebles recombination) and the extensions (curvature, lensing, massive
neutrinos, two-point functions, RECFAST, and the model library with its validation). To reproduce it
(the Python scripts need `classy`; `make_figures.py --no-class` skips the CLASS comparison):

```
make milestones && ./milestones                 # report/data/ (~2 min on 2 cores)
python3 report/make_figures.py            # milestone figures and numbers
for K in -0.2 -0.05 0 0.05 0.2; do
  ./cmb params/curvature.ini OmegaK=$K output_dir=data/curvature_$K > data/log_curv_$K.txt; done
python3 report/make_curvature_figure.py
for m in 0 0.06 0.15 0.3; do
  ./cmb params/neutrinos.ini m_nu=$m output_dir=data/neutrinos_$m > data/log_nu_$m.txt; done
python3 report/make_neutrino_figure.py
./cmb params/twopoint.ini > data/log_twopoint.txt
python3 report/make_twopoint_figure.py
./cmb transfer_ells= output_dir=data/rf/fid
./cmb transfer_ells= output_dir=data/rf/fid_peebles recombination=peebles
./cmb reionization=false pert_n_k=2 ell_max=20 output_dir=data/rf/norei
./cmb reionization=false recombination=peebles pert_n_k=2 ell_max=20 output_dir=data/rf/norei_peebles
python3 report/make_recfast_figure.py
# all models into data/models/ (see "Running" above; f(R) with pert_k_max=1)
python3 report/make_models_figure.py
python3 report/fill_numbers.py             # numbers into the new sections (only where @@key@@ placeholders remain)
cd report && pdflatex paper && pdflatex paper     # REVTeX 4.2
```

## Possible next steps

HMcode and modified-gravity nonlinear corrections (Halofit is for LCDM/wCDM; no massive-neutrino correction), Horndeski
with alpha_T != 0 or from G_i(phi, X), f(R) beyond the quasi-static approximation, tensor modes in curved universes,
several massive neutrino species, speed-ups beyond `precision = fast` (CLASS-like radiation-streaming and
ultra-relativistic fluid approximations, analytic Jacobians for the stiff solver).
