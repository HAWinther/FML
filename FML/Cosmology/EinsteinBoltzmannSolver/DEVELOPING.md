# Developing the Einstein–Boltzmann solver

## Code layout

| File (in `FML/Cosmology/`) | Contents |
|---|---|
| `Common.h` | Solver aliases and compatibility names for constants, numerical methods, splines, ODE solvers and timing. Implementations live in shared `FML/Units`, `Math`, `FileUtils`, `Timing` and `Spline` modules. |
| `Background/CosmologyParameters.h/.cpp` | Parameters with derived Ω's; `fiducial()`, `Planck2018()`, `Planck2018Baseline()` |
| `EinsteinBoltzmannSolver/ParameterFile.h/.cpp` | Parameter files: one table of all parameters (name, description, unit), reading with overrides and checks, writing |
| `Background/` (headers, `README.md`, `tests/`) | The background module (`BACKGROUND`): species (with the Fermi–Dirac integrals of massive neutrinos), field equations, `ComponentBackground` and the background models; `Presets.h`: the background registry (parameters with defaults and descriptions) |
| `Background/BackgroundCosmology.h/.cpp` | The solver's interface to the background (SI units): H, Hp and derivatives, Ω_i(x), η, t, distances, curvature geometry `sinK`, `cotK`, from the `ComponentBackground` of the model; the `PerturbationModel` |
| `Perturbations/` (headers, `README.md`, `tests/`) | The perturbation module (`PERTURBATIONS`): species perturbations, perturbed field equations, the `PerturbationModel` of every model; `Presets.h`: the model registry (built on the background registry: the models the solver offers, their parameters with defaults and descriptions; parameter file, validation, `--write-defaults`) |
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
| `EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.{h,cpp}` | the whole solver in one class: settings, `solve()` (all stages or up to one), accessors, `output()` |
| `EinsteinBoltzmannSolver/Main.cpp` | `./cmb`: reads a parameter file and runs an `EinsteinBoltzmannSolver` |
| `EinsteinBoltzmannSolver/tests/` | One unit-test program per module and model (~650 checks): analytic limits, conservation laws, convergence, internal consistency; `test_regression` compares key outputs of several configurations and all models with stored values (`tests/reference/`) |
| `EinsteinBoltzmannSolver/Milestones.cpp` | `./milestones`: all models and numbers for the report (`report/data/`) |
| `EinsteinBoltzmannSolver/Benchmark.cpp` | `./benchmark [config ...]`: run time per stage (`data/benchmark.txt`) |
| `EinsteinBoltzmannSolver/python/params.py` | Reads `params.txt` and builds the matching CLASS input |
| `EinsteinBoltzmannSolver/python/derive_brans_dicke.py`, `derive_cubic_galileon.py` | Symbolic derivations (sympy) of the Brans–Dicke and cubic Galileon equations |
| `EinsteinBoltzmannSolver/python/derive_horndeski.py`, `derive_horndeski_alpha.py` | Horndeski (F(φ)R/2 + G2 − G3□φ) in Newtonian gauge, then the equations in terms of α_K, α_B, α_M (run in this order) |
| `EinsteinBoltzmannSolver/python/compare_hiclass.py` | A `horndeski` run against hi_class (its `class` executable) |
| `EinsteinBoltzmannSolver/python/compare_class_ede.py` | An `ede` run against CLASS_EDE (its `class` executable) |
| `EinsteinBoltzmannSolver/python/compare_axioncamb.py` | An `axion` run against axionCAMB (its `camb` executable) |
| `EinsteinBoltzmannSolver/python/compare_class_wcdm.py` | wCDM (`dark_scattering` with ξ = 0) against CLASS's fluid dark energy |
| `EinsteinBoltzmannSolver/python/plot_models.py` | Ratios of H(z), C_ℓ^TT, C_ℓ^φφ and P(k) of all models to ΛCDM |
| `EinsteinBoltzmannSolver/python/plot_*.py` | Quick-look plots of the outputs in `data/` (CLASS overlay if `classy` is installed) |
| `EinsteinBoltzmannSolver/python/compare_class_*.py` | Comparisons with CLASS (`pip install classy`): background, recombination, perturbations, spectra (CLASS with RECFAST; `--hyrec` for HyRec), lensing, neutrinos, two-point functions |
| `external/`, `../Background/external/` | Planck 2018 TT/EE/TE spectra, SDSS DR7 LRG and WMAP+ACT P(k), binned JLA supernovae (for the plots and the report); the BBN helium table (`Cosmology/Background/external/bbn/`). The likelihood data are in `FML/CosmologyFitting/external/` |

## Models: species and field equations

A model is a background (`Cosmology/Background`: species and field equations) and its linear perturbations
(`Cosmology/Perturbations`: a `PerturbationModel` on that background):

* A **species perturbation** (`Perturbations/SpeciesPerturbation.h`) has its variables (possibly in several
  approximation schemes, e.g. the massive-neutrino hierarchy and fluid, or a scalar field and the fluid of its
  oscillations), initial conditions, equations, and the quantities to store (`Perturbations::get(name, x, k)`). It
  talks to the metric only through its moments δρ, δp, (ρ+p)θ/(ck) and (ρ+p)σ, and receives Φ, Ψ and Φ'. Interacting
  species (coupled quintessence, dark scattering, interacting dark energy) are one perturbation block with all their
  variables. Photons and baryons are the fixed core of `Perturbations` (Thomson coupling, tight coupling,
  polarization and the line-of-sight sources).
* The **perturbed field equations** (`Perturbations/PerturbedFieldEquation.h`) own their own perturbation
  variables (GR: Φ; Brans–Dicke, cubic Galileon and Horndeski: Φ and two scalar-field variables), from which, with
  the summed moments, they give Φ, Ψ and Φ'. `GeneralRelativity` implements the Einstein equations; `MuSigma` the
  μ–Σ parameterization (evolving the GR potential of the comoving density with an equation derived from
  energy-momentum conservation, so that μ = Σ = 1 is GR exactly), which `DGP` (μ(a) from its background) and
  `HuSawicky` (scale-dependent μ(k,a)) reuse.
* The **registries**: `Background/Presets.h` (the backgrounds, with their parameters) and `Perturbations/Presets.h`
  (the full models: each entry is built on the background entry of the same name and adds the parameters that only
  concern the perturbations) build a model by name with its parameters (`make_background`, `make_perturbation_model`);
  the model registry is the list of models the solver offers, `PERTURBATIONS::model_parameters(name)` their parameters
  with defaults and descriptions. A run selects one with `model = <name>` (every name of the registry, also
  `quintessence_inverse_power_law`, `quintessence_pngb`, `quintessence_hilltop` and `ede_zc`, with all their
  parameters, e.g. `quint_phi_ini`, `quint_dphidlna_ini`, `bd_GeffG_today`, `bd_density_parameter_definition`); its
  parameters are then valid in the parameter file (and a parameter of another model is an error). `./cmb
  --write-defaults` lists the models, `params_used.ini` the parameters of the run's model.

To add a model: its species and/or field equations in `Cosmology/Background` and `Cosmology/Perturbations` (a
`BackgroundModel_<name>.h` and a `PerturbationModel_<name>.h` with their registry entries, the parameters with
descriptions). A custom model can also be run without a registry entry: `BackgroundCosmology(params, pmodel)` with any
`PerturbationModel` (e.g. `PERTURBATIONS::PerturbationModel_Assembled` on a `ComponentBackground`;
`tests/test_models.cpp`, `tests/test_horndeski.cpp`, `tests/test_background.cpp`). The perturbation data types
(`ModeAtX`, `Moments`, `MetricAtX`, `IsoValues`, `InitialData`, `TensorCoefficients`) are those of
`Perturbations/PerturbationTypes.h`.

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
Ultralight axions (`test_axion`): the shooting reproduces Ω_ax (1e-9) for m = 10⁻³³–10⁻¹⁸ eV, the frozen start and
the w = 0 fluid after the switch, the unused Einstein constraints with the axion before and after the switch (5e-6),
and the limits (a tiny fraction is ΛCDM, a heavy axion is CDM); against axionCAMB with its switch at m = 3H
(`ax_switch_mH = 3`; `python/compare_axioncamb.py`): P_cb/P_cb(ΛCDM) to 7e-4 for k ≤ 1 h/Mpc (m = 10⁻²⁵ and 10⁻²⁷ eV,
10% of the dark matter; 3e-3 for 10⁻³⁰, 10⁻³² eV), an all-axion dark matter through the Jeans cutoff to 1e-3,
C_ℓ^TT/C_ℓ^TT(ΛCDM) to 6e-5, the axion isocurvature transfer functions to 3e-3. With the default switch (m/H ~ 125,
which converges: switches at m/H = 10–300 agree to 0.5% at the cutoff) axionCAMB's m = 3H approximation shows: up to
1.3% in P_cb (10%), 17% near the cutoff of an all-axion dark matter, 4-5% for the lightest axions (whose a⁻³ from
m = 3H misses the rolling field). Two axionCAMB issues turned up: without massive neutrinos its axion background has
no massless neutrinos (a_osc 12% early; patched for the comparison), and its axion isocurvature amplitude is 1/h²
too large (`phiinit` is h times the field).
Interacting dark energy: ρ_c against an independent integration, the constraints with energy exchange (2e-6;
they test that the exchanged energy and momentum cancel), ξ = 0 against the separate wCDM implementation of
`dark_scattering` (identical), and the rejection of unstable (d > 0) and negative-ρ_c models.
`test_regression` stores key outputs of all models.

## The report

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
# all models into data/models/ (see "Comparing the models" in OPTIONS.md; f(R) with pert_k_max=1)
python3 report/make_models_figure.py
python3 report/fill_numbers.py             # numbers into the new sections (only where @@key@@ placeholders remain)
cd report && pdflatex paper && pdflatex paper     # REVTeX 4.2
```

## Possible next steps

HMcode baryonic feedback and ReACT support for additional gravity models, Horndeski
with alpha_T != 0 or from G_i(phi, X), f(R) beyond the quasi-static approximation, tensor modes in curved universes,
several massive neutrino species, speed-ups beyond `precision = fast` (CLASS-like radiation-streaming and
ultra-relativistic fluid approximations, analytic Jacobians for the stiff solver).
