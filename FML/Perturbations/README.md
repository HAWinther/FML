# FML/Perturbations

The linear perturbations of the models of `FML/Background`, built on top of it (the background module does not depend
on this one). Staged here, in the namespace `FML::COSMOLOGY::PERTURBATIONS`, until the Einstein-Boltzmann solver
(`Cosmology/`) moves onto it; the types and conventions are the solver's (Newtonian gauge, SI units, x = ln a,
densities times H0^2 a^2 / rho_c0), so its driver can switch to them. Header-only; needs the Background module (and
GSL for the momentum quadrature).

## Structure

```
PerturbationTypes.h          ModeAtX, Moments, MetricAtX, IsoValues, InitialData, TensorCoefficients (as the solver),
                             PerturbationSettings (the solver's species options), PerturbationContext (background,
                             settings, K, s_ell, eta, cotK, mode_at_x)
SpeciesPerturbation.h        the base: one species' block of the ODE state (modes, switch points, cache, initial
                             conditions, moments, rhs, outputs, free_streaming_density)
PerturbedFieldEquation.h     the base: the metric (Phi, Psi, Phi') from the summed moments; G_eff_initial, tensor(x)
SpeciesPerturbation_<name>.h CDM, MasslessNeutrinos (Boltzmann hierarchy), MassiveNeutrinos (momentum-resolved
                             hierarchy, then the CLASS fluid approximation)
PerturbedFieldEquation_<name>.h  GeneralRelativity, BransDicke (flat)
PerturbationModel.h          the base: which background species get which perturbation, and the field equations;
                             StandardSpeciesPerturbations (CDM, massless and massive neutrinos) for the models to reuse
PerturbationModel_<name>.h   LCDM, BransDicke, each with its background (make_lcdm_perturbations, make_jbd_perturbations)
Presets.h                    make_perturbation_model(name, params, model params): lcdm, jbd (names and parameters of
                             the background registry)
```

Baryons and photons are not species here: they are coupled by Thomson scattering and stay in the solver's core.
Smooth species (Lambda) get no perturbations. A species object is shared by all modes and threads (no per-mode state);
the model keeps typed pointers to the background objects it needs, found once at construction.

```cpp
#include <FML/Perturbations/Presets.h>
using namespace FML::COSMOLOGY::PERTURBATIONS;

FML::COSMOLOGY::BACKGROUND::BackgroundParameters p;
p.m_nu_eV = {0.06};
auto model = make_perturbation_model("jbd", p, {{"bd_omega", 1000.0}});
auto ctx = model->context();                         // PerturbationSettings: lmax_nu, lmax_ncdm, n_q_ncdm, ...
auto species = model->species_perturbations(ctx);    // cdm, massless neutrinos, massive neutrinos
auto gravity = model->field_equation_perturbations(ctx);
ModeAtX b = ctx.mode_at_x(-5.0, 0.1 / FML::UTILS::SI::Mpc);   // + tau', tau'', R from the recombination history
```

## Tests

`tests/`: `make run` (links the Einstein-Boltzmann solver). Equation by equation against the solver's own species and
gravity perturbations (`Cosmology/Models`, built as an explicit `Model` of `BackgroundCosmology`, since the solver
itself now runs these classes for lcdm and jbd), for LCDM (m_nu = 0, 0.06, 0.1 eV; Omega_k = 0, +-0.02) and Jordan-Brans-Dicke (omega = 100,
1000), at k = 1e-4 to 5 /Mpc and x = -17 to 0, each side with a ModeAtX from its own background: structure (names,
n_vars, output names), ModeAtX, switch points, caches, initial conditions (adiabatic and isocurvature), moments, rhs,
mode switch, outputs, and the field equations' cache, initial conditions, metric, outputs and tensor coefficients. The
agreement is the backgrounds' (1e-7; 3e-7 where the massive-neutrino pressure and pseudo-pressure enter, as the
solver's 48-node quadrature is off by up to 7e-8). The solver's Brans-Dicke conformal time is offset by its GR
initial value; the test corrects for it. Also: neutrinos off and Omega_nu = 0, the jbd density definitions
(rescaled massive neutrinos), the registry and errors.

## Used by the solver

`Cosmology/Perturbations/Perturbations.cpp` builds the species and field equations of lcdm and jbd from this module
(`BackgroundCosmology::perturbation_model()`, a `PerturbationContext` with the settings of `PerturbationOptions`); its
data types are aliases of these. Not yet here: the other models (quintessence, EDE, DGP, f(R), mu-Sigma, Horndeski,
w0wa, cubic Galileon, coupled and interacting dark energy, dark scattering), which the solver still takes from
`Cosmology/Models`.
