# Einstein–Boltzmann solver

A C++17 Einstein–Boltzmann code for the CMB and large-scale structure, part of [FML](../../../README.md). It computes
the background expansion, recombination and the linear perturbations, and from them the CMB and matter power spectra,
for ΛCDM and its extensions and for 12 dark-energy and modified-gravity models. It agrees with CLASS module by
module, typically to 1e-4–2e-3.

You can use it in four ways:

- as a program: `./cmb params/planck2018.ini`
- as a library: [`EinsteinBoltzmannSolver.h`](EinsteinBoltzmannSolver.h)
- for the initial conditions of [SimulationCode](../../SimulationCode/INITIAL_CONDITIONS.md#transfer-function-input),
  which runs it at start-up (`ic_type_of_input = internal`)
- as the theory code of the likelihoods and samplers in [CosmologyFitting](../../CosmologyFitting/README.md)

## Quick start

You need a C++17 compiler and GSL (`brew install gsl` / `apt install libgsl-dev`). OpenMP is optional. Put the GSL
location for your machine in `FML/Makefile.local` (e.g. `LOCAL_PREFIX = /opt/homebrew`). Run the programs from this
folder: the parameter files are in `params/` and the output goes to `data/`.

```
make                                                 # the solver ./cmb (USE_OMP=false: without OpenMP)
./cmb                                                # the fiducial model
./cmb params/planck2018.ini                          # parameters from a file
./cmb m_nu=0.06 lensing=true output_dir=data/test    # command-line values (override the file)
./cmb --write-defaults my.ini                        # a complete parameter file with every parameter and its default
make test                                            # build and run all the unit tests (tests/test_*.cpp)
```

The fiducial model takes ~45 s on 2 cores, most of it in the perturbations. `precision = fast` cuts the lensed CMB
spectra to ℓ = 2500 from ~70 s to 13–16 s, for parameter fits (see [OPTIONS.md](OPTIONS.md#fast-precision-preset-for-parameter-fits)).
`make milestones benchmark` builds `./milestones` (the data of the report) and `./benchmark`, which times each stage.
The benchmark gives these extra costs over the fiducial run: lensing +4 s, massive neutrinos +7 s, curvature +15–28 s
and two-point functions +3 s.

FML compiles with `-march=native` by default. This changes the results at the 1e-4 level compared with a build
without it (fused multiply-adds), within all the test tolerances. The programs find FML's data files, such as the
BBN table in `Cosmology/Background/external/`, through the FML folder compiled into them. If you move FML after
building, set the environment variable `FML_DIR`.

## What it computes

| Area | What is included |
|---|---|
| Models | A model is a list of species and a theory of gravity, selected by name in the parameter file (`model = ...`, see the table below) |
| Background | Photons, baryons, CDM, massless and one massive neutrino species, curvature (open/closed), Λ |
| Recombination | RECFAST 1.5 (default: effective three-level H and He I atoms with the fudge factors and Gaussian K_H correction, triplet He, evolved baryon temperature) or the course model (Saha → Peebles; `recombination = peebles`); tanh reionization (CAMB/CLASS `reio_camb`), τ, visibility function, sound horizon; Y_p fixed or from BBN (`Yp_bbn`, PArthENoPE table as CLASS) |
| Perturbations | CDM, baryons, photon temperature and polarization hierarchies, massless neutrinos, massive neutrinos (momentum-resolved Ma–Bertschinger hierarchy + CLASS fluid approximation), tight coupling, curvature; OpenMP over k |
| Spectra | C_ℓ^TT, EE, TE by line of sight (hyperspherical Bessel functions when curved, with CLASS's flat approximation), P(k) of total matter and CDM+baryons; primordial P_R = A_s (k/k_pivot)^(n_s − 1 + α_s/2 ln(k/k_pivot)) with optional running `alpha_s` (as CLASS; adiabatic mode only, C_ℓ ratios to CLASS 5e-4) |
| Tensor modes | Gravitational waves (`r`, `n_t` or the consistency relation): photon temperature and polarization and neutrino hierarchies, anisotropic stress, tight coupling, modified-gravity friction (α_M, M*² of Horndeski, Brans–Dicke, f(R)); TT, EE, TE, BB by line of sight, added to the scalar spectra and lensed with them; flat |
| Isocurvature modes | CDM, baryon, neutrino density and neutrino velocity isocurvature (`f_*`, `n_*`, `c_ad_*` as CLASS), each correlated with the adiabatic mode; CLASS initial conditions; one extra perturbation solve per mode; GR only |
| Nonlinear | Halofit, Halofit+Bird, HMcode-2016, HMcode-2020 and optional ReACT reactions (`halofit`, `halofit_bird`, `hmcode2016`, `hmcode2020`, `react`), used for nonlinear CMB lensing |
| Lensing | C_ℓ^φφ (LOS + Limber), lensed TT, EE, TE, BB (full-sky correlation-function method) |
| Two-point functions | CMB C(θ), ξ±, ξ_X with beams; matter ξ(r), σ_R, σ₈; angular C_ℓ and w(θ) of a projected tracer (exact + Limber) |
| Fitting | MCMC, best fits and profile likelihoods with background, Planck CMB and lensing data: `FML/CosmologyFitting` (samplers in `FML/MCMC`) |

## Models

A model is a set of species and a theory of gravity, selected with `model = <name>`. Each model's parameters are in
its parameter file in `params/` and in `./cmb --write-defaults`.

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
| `ede` | `ede_fEDE`, `ede_log10zc`, `ede_theta_i`, `ede_n` | Axion-like early dark energy V = m²f²(1 − cos φ/f)ⁿ (n > 1) + Λ: m and f found from the peak fraction f_EDE at z_c; full Klein–Gordon background and perturbations through the oscillations, until the field oscillates so fast (m/H > ~125; for n ≳ 2.5 not before today) that background and perturbations switch to the cycle-averaged fluid (w = (n−1)/(n+1), effective sound speed) |
| `ide` | `ide_xi`, `ide_w`, `ide_cs2`, `ide_stability_check` | Interacting dark energy: fluid (w, ĉ_s²) exchanging energy with CDM, Q = ξHρ_DE along the CDM velocity; analytic background; doom-factor stability check |
| `fr` | `fR0`, `fR_n` | Hu–Sawicky f(R): ΛCDM background (to O(f_R0)), quasi-static μ(k,a) = (1+4Q)/((1+3Q)(1+f_R)), Σ = 1/(1+f_R), Q = (k/a)² f_RR/(1+f_R) = (k/a)²λ_C²/3; flat |

## Accuracy

Agreement with CLASS (same physics, CLASS with RECFAST): background 1e-7, X_e 3e-4, z_* and r_s 1e-6, TT 5e-4
rms for ℓ = 30–1000 (3e-4 for 1000–2000), EE 2e-3 rms, P(k) 0.1%, lensed/unlensed TT ratio 1e-3, effect of the
neutrino mass 5e-4, σ₈ 4e-5. (The Peebles model of the course gives TT 0.6–0.8% and EE 1.4% rms.) Details
and the physics are in `report/paper.pdf`.
The measured accuracy of the nonlinear spectra, tensor and isocurvature modes and the precision preset is in
[OPTIONS.md](OPTIONS.md), and the validation of each model is in [DEVELOPING.md](DEVELOPING.md#models-species-and-field-equations).

## Parameters

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
| `params/neutrinos.ini` | One massive neutrino with lensing (`m_nu=...`), as in the report (`deg_ncdm=3`: three degenerate species of mass `m_nu` each, solved as one; `m_nu_list=0.0086, 0.05`: species of different masses, each its own hierarchy) |
| `params/twopoint.ini` | All two-point functions (perturbations to k = 1/Mpc), as in the report |
| `params/isocurvature.ini` | Planck 2018 with a correlated CDM isocurvature mode (f = 0.3, c_ad = −0.2) |
| `params/tensors.ini` | Planck 2018 with tensor modes, r = 0.05: lensed BB with the primordial B modes |
| `params/musigma.ini`, `dgp.ini`, `quintessence.ini`, `jbd.ini`, `fr.ini`, `coupled_quintessence.ini`, `dark_scattering.ini`, `cubic_galileon.ini`, `horndeski.ini`, `ede.ini`, `ide.ini` | The models beyond ΛCDM, with their parameters |

## Output

Outputs (in `output_dir`): `background.txt`, `recombination.txt`, `perturbations_k<k>.txt`, `cells.txt`
(D_ℓ^TT, EE, TE in μK²), `cells_lensed.txt`, `matter_power.txt` (k [h/Mpc], P_m, P_cb), `transfer.txt`, and
optionally `cmb_correlation_*.txt`, `matter_correlation.txt`, `angular_cl_z<z>.txt`, `angular_w_z<z>.txt`.

## Using the solver from code

```cpp
#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
using namespace FML::COSMOLOGY;

auto ebs = EinsteinBoltzmannSolver::from_file("params/planck2018.ini", {"lensing=true"});
ebs.solve();                                    // or solve(EinsteinBoltzmannSolver::Stage::Perturbations)
double sigma8 = ebs.power_spectrum().get_sigma8();
const auto & bg = ebs.background();             // recombination(), perturbations(), tensors() likewise
ebs.output();                                   // the files enabled in the settings, in output_dir
```

## Documentation

- [OPTIONS.md](OPTIONS.md): nonlinear P(k) (Halofit, HMcode, ReACT) and helium from BBN, tensor and isocurvature
  modes, the fast precision preset, comparing all the models, the k-range of the line of sight.
- [DEVELOPING.md](DEVELOPING.md): the code layout, how a model is built from species and field equations, adding a
  model, the validation of the models, reproducing the report, possible next steps.
- [params/default.ini](params/default.ini): every parameter with its description.
- `report/paper.pdf`: the physics, numerics and validation of every module.
- The background and perturbation modules: [Background](../Background/README.md), [Perturbations](../Perturbations/README.md).
- [Nonlinear methods](../Spectra/NONLINEAR.md) and [ReACT](../../ReACT/README.md).
