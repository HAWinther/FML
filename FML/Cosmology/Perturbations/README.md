# Cosmology/Perturbations

The linear perturbations of the models of `Cosmology/Background`, built on top of it (the background module does not depend
on this one). In the namespace `FML::COSMOLOGY::PERTURBATIONS`, next to the solver's driver (`Perturbations.h/.cpp`, `Tensors`,
`Isocurvature`), which runs on it for every model. The types and conventions are the solver's (Newtonian gauge, SI units,
x = ln a, densities times H0^2 a^2 / rho_c0). Header-only; needs the Background module (and GSL for the momentum
quadrature).

## Structure

```
PerturbationTypes.h          ModeAtX, Moments, MetricAtX, IsoValues, InitialData, TensorCoefficients (as the solver),
                             PerturbationSettings (the solver's species options), PerturbationContext (background,
                             settings, K, s_ell, eta, cotK, mode_at_x)
SpeciesPerturbation.h        the base: one species' block of the ODE state (modes, switch points, cache, initial
                             conditions, moments, rhs, outputs, free_streaming_density)
PerturbedFieldEquation.h     the base: the metric (Phi, Psi, Phi') from the summed moments; G_eff_initial, tensor(x)
SpeciesPerturbation_<name>.h CDM, MasslessNeutrinos (Boltzmann hierarchy), MassiveNeutrinos (momentum-resolved
                             hierarchy, then the CLASS fluid approximation), ScalarField (quintessence, EDE, axion: Klein-Gordon,
                             then the effective fluid of the oscillating field), DarkEnergyFluid (DarkScattering: CDM +
                             dark-energy fluid with momentum exchange; InteractingDarkEnergy: CDM + fluid with energy
                             exchange), CoupledQuintessence (coupled CDM + field)
QuasiStaticGravity.h         the base: the quasi-static limit of a theory on a Background (any, not only a
                             ComponentBackground): mu(x, k), Sigma(x, k), the second-order (2LPT) source, k in H0/c,
                             and the screening (Screening: type potential / density / gradient, threshold, the
                             surviving fraction of the fifth force). The one definition of these, used by the MuSigma
                             field equations here, Cosmology/Growth, HybridBiasOneLoopMG (mgmodel_from_theory) and
                             SimulationCode's gravity models (f(R), DGP, cubic Galileon, JBD, mu, symmetron, (m, beta))
QuasiStaticGravity_<name>.h  MuSigma (Planck/DES), HuSawicky, DGP (either branch, H of any background), BransDicke,
                             Symmetron, MBeta (power-law (m(a), beta(a))), CubicGalileon (the tracker: the DGP form
                             with r_c(a), beta(a)), Horndeski (reduced Horndeski from the table of
                             SimulationCode/scripts/horndeski_frontend.py)
PerturbedFieldEquation_<name>.h  GeneralRelativity, BransDicke (quasi_static(): its sub-horizon limit), MuSigma (the
                             mu-Sigma equations for a QuasiStaticGravity; default Planck/DES) with DGP and HuSawicky
                             derived from it, HorndeskiAlpha (alpha parameterization:
                             AlphaFunctions, StandardAlphas, FunctionAlphas, HorndeskiAlphaBackground with the stability
                             check), CubicGalileon
PerturbationModel.h          the base: which background species get which perturbation, and the field equations;
                             StandardSpeciesPerturbations (CDM, massless and massive neutrinos, except species to skip;
                             degenerate massive species, the same mass and temperature, are one hierarchy with the
                             sum of their densities; several masses: outputs delta_ncdm_<i>, v_ncdm_<i>, and
                             Perturbations adds delta_ncdm, v_ncdm of all of them)
                             for the models to reuse; PerturbationModel_Assembled (a model from given parts)
PerturbationModel_<name>.h   LCDM, w0wa, MuSigma, DGP, HuSawicky, Horndeski, Quintessence, EDE, Axion, BransDicke,
                             CoupledQuintessence, CubicGalileon, DarkScattering, IDE, each with its make_..._perturbations
Presets.h                    the model registry (built on the background registry: name, description, the parameters
                             of the perturbations; model_parameters(name) = background's + perturbations'), and
                             make_perturbation_model(name, params, model params), make_model_background
```

Baryons and photons are not species here: they are coupled by Thomson scattering and stay in the solver's core.
Smooth species (Lambda, w0wa dark energy) get no perturbations; a species whose perturbations belong to another block
(the CDM of dark scattering, coupled quintessence and IDE) is skipped by StandardSpeciesPerturbations. A species object
is shared by all modes and threads (no per-mode state); the model keeps typed pointers to the background objects it
needs, found once at construction (nothing is looked up or cast when the equations are evaluated). The perturbation-only
parameters (mu0, Sigma0, fR0, fR_n, hd_* but hd_w0/hd_wa, ds_xi, ds_cs2, ide_cs2, ide_stability_check) are listed
(with defaults and descriptions) in the registry of Presets.h, the others in the background registry; the solver's checks and warnings come with the models (w0wa: no dark-energy
perturbations, warned once per process; dgp: Omega_Lambda < 0 warned; musigma, fr: flat, dark energy,
1 + 2 Sigma0 - mu0 >= 0; horndeski: flat, ghost / gradient / stiffness checks, D = 0 with alpha_M rejected; IDE: the
doom factor; dark scattering: w != -1, cs2 > 0, xi >= 0).

```cpp
#include <FML/Cosmology/Perturbations/Presets.h>
using namespace FML::COSMOLOGY::PERTURBATIONS;

FML::COSMOLOGY::BACKGROUND::BackgroundParameters p;
p.m_nu_eV = {0.06};
auto model = make_perturbation_model("horndeski", p, {{"hd_cK", 1.0}, {"hd_cB", 0.6}, {"hd_cM", 0.3}});
auto ctx = model->context();                         // PerturbationSettings: lmax_nu, lmax_ncdm, n_q_ncdm, ...
auto species = model->species_perturbations(ctx);    // cdm, massless neutrinos, massive neutrinos
auto gravity = model->field_equation_perturbations(ctx);
ModeAtX b = ctx.mode_at_x(-5.0, 0.1 / FML::UTILS::SI::Mpc);   // + tau', tau'', R from the recombination history
```

## The oscillating scalar field

The background replaces a scalar field that oscillates fast about the minimum of its potential (PNGB and hilltop
quintessence, early dark energy; m / H > ~125) by a fluid with the cycle-averaged w (BACKGROUND::ScalarFieldSpecies).
SpeciesPerturbation_ScalarField switches at the same x (a switch point) from the Klein-Gordon perturbations to the
effective fluid of an oscillating field (as Poulin, Smith, Grin, Karwal & Kamionkowski 2018 for V ~ phi^(2n); n = 1
is the ultralight-axion case): delta, theta with w = c_a^2 = w_osc and
c_s^2(k, a) = [2 (n-1) varpi^2 + (k/a)^2] / [2 (n+1) varpi^2 + (k/a)^2], varpi the angular frequency of the
oscillation (from the period of the actual potential at the switch density, then ~ rho^((n-1)/2n)). The switch keeps
the comoving density perturbation (so the Einstein constraints hold through it) and takes the velocity from the shift
of the oscillation phase (see the header). A field that is a fluid already at x_start starts as a fluid. The
Einstein-Boltzmann solver used to integrate the field (and its perturbations) through the oscillations: for early dark
energy with n >= 2.5 and the usual z_c the field never oscillates fast enough to switch, so nothing changes; it does
switch for n close to 2 (n = 2.2, z_c = 1e6: at z ~ 1e4) or 1 < n <= 2 (which the old solver rejected), and the
quintessence potentials with a minimum switch only after today with the default parameters. The ultralight axion
(model axion) switches like this by default, or with ax_switch_mH = s at m / H = s: s <= 4 before the first crossing,
as axionCAMB (s = 3), keeping axionCAMB's synchronous-gauge delta and (1 + w) v continuous (ModeAtX::v_cdm, the CDM
velocity, is set for the switch); s > 4 at the first crossing after m / H = s. Both background and perturbations
switch at the exact crossing (or m / H), found between the nodes of the field's grid, so the results are smooth in the
parameters.

## Tests

`tests/`: `make run` (header-only, does not link the solver). Equation by equation against the solver's old species and
gravity perturbations (`Cosmology/Models`, removed; their values for the inputs of the test are stored in
`tests/data/solver_perturbations.txt`, see the test's header for how they were made) for every model: lcdm (m_nu = 0,
0.06, 0.1 eV; Omega_k = 0, +-0.02), jbd, w0wa, musigma, dgp (normal, self-accelerating, LCDM background), fr,
horndeski (the three parameterizations), quintessence and coupled quintessence (with curvature), ede and ede_zc,
cubic_galileon, dark_scattering (locked and unlocked velocities, xi = 0), ide, at k = 1e-4, 0.05, 5 /Mpc and
x = -17, -8, -2, 0 (the standard species at all of them for lcdm, at one point for the other models), each side with a
ModeAtX from its own background: structure (names, n_vars, output names), ModeAtX, switch points,
caches, initial conditions (adiabatic and isocurvature), moments, rhs, mode switch, outputs, and the field equations'
cache, initial conditions, metric, outputs and tensor coefficients. The agreement is that of the backgrounds: 1e-8 for
all species and field equations except the massive neutrinos (2e-7: the solver's 48-node quadrature of the
pressure and pseudo-pressure was off by up to 7e-8) and early dark energy late in its oscillating phase (7e-7 at x >= -2,
the drift of the oscillation phase of two independent integrations; the field's density is < 1e-6 of the total there).
The solver's Brans-Dicke conformal time was offset by its GR initial value; the stored ModeAtX is corrected. Also: the
oscillating-fluid mode (the frequency against the Gamma-function value of a power law, c_s^2 -> w and -> 1, the switch,
a fluid from the start), the other quintessence potentials, v_phiphi and the background's d2rhodx2 against finite
differences, neutrinos off and Omega_nu = 0, the jbd density definitions, the registry (every model) and
the models' checks.

## Used by the solver

`Perturbations.cpp` (in this folder) builds the species and field equations of every model from this module
(`BackgroundCosmology::perturbation_model()`, a `PerturbationContext` with the settings of `PerturbationOptions`); its
data types are aliases of these. A custom model is a `ComponentBackground` with a `PerturbationModel` (e.g.
`PerturbationModel_Assembled`) given to `BackgroundCosmology(params, model)`.
