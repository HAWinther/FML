# FML/Background

One background cosmology for all of FML (design: `claude/fml-background-design.md`). Staged here, in the namespace
`FML::COSMOLOGY::BACKGROUND`, until the Einstein-Boltzmann solver, SimulationCode and the other modules move onto
it; it then replaces `Cosmology/Background` and lives in `FML::COSMOLOGY`. Header-only; needs the Spline,
ODESolver and Math modules (GSL).

## Structure

```
Background (abstract)          E(x), dlogE/dx, Omega(component, x), parameters, range   x = ln a
 |                              + non-virtual: eta, t, chi, x_of_chi, D_M, D_A, D_L, D_H, a/z versions, info, output
 |-- ComponentBackground        species (Species.h) + field equations (FieldEquations.h)
 |-- TabulatedBackground        tables of E and Omega_i; from_class(<root>background.dat)
 '-- FunctionBackground         user functions (old callback interfaces, tests)

Species.h                  PowerLawSpecies (baryons, CDM, photons, massless nu, Lambda), CPLDarkEnergy,
                           MassiveNeutrinoSpecies, ScalarFieldSpecies (the base of the scalar-field dark energies)
FieldEquations.h           the base; FieldEquation_GeneralRelativity.h, FieldEquation_DGP.h, FieldEquation_BransDicke.h
BackgroundModel_<name>.h   one model each, its species and make_ function:
                           LCDM, w0wa, DGP, Quintessence (exponential, inverse power law, PNGB, hilltop), EDE,
                           BransDicke, CoupledQuintessence, CubicGalileon, DarkScattering (wCDM), IDE
Presets.h                  the registry and make_background(name, params, model params): lcdm, w0wa, dgp,
                           quintessence(_inverse_power_law, _pngb, _hilltop), ede, ede_zc, jbd, coupled_quintessence,
                           cubic_galileon, dark_scattering, ide, musigma, fr, horndeski
```

Background only: models that only modify the perturbations (mu-Sigma, Hu-Sawicky f(R), Horndeski alphas) have the
LCDM or w0wa background, and perturbation parameters (ds_xi, ds_cs2, ide_cs2, ...) are accepted but not used.

Units: E dimensionless, conformal time and distances in Mpc/h, cosmic time in Gyr (`t_H0`: units of 1/H0).
Radiation is always included (photons from `T_CMB`, `N_eff` neutrinos of which one per entry of `m_nu_eV` is massive).

```cpp
#include <FML/Background/Presets.h>
using namespace FML::COSMOLOGY::BACKGROUND;

BackgroundParameters p;          // h, Omega_b, Omega_cdm, Omega_k, T_CMB, N_eff, m_nu_eV
p.m_nu_eV = {0.06};
auto bg = make_lcdm(p);          // std::shared_ptr<const ComponentBackground>
double E = bg->E(std::log(0.5)), chi = bg->chi_of_z(1.0), DA = bg->D_A_of_z(1.0);
auto dgp = make_background("dgp", p, {{"Omega_rc", 0.25}, {"dgp_branch", -1}});
```

## Tests

`tests/`: `make run` (links the Einstein-Boltzmann solver, whose background is one of the references). Checks the
analytic Einstein-de Sitter and radiation-free LCDM backgrounds (flat, open, closed), the Fermi-Dirac integrals,
LCDM with massive neutrinos and curvature, w0wa and DGP against the solver (1e-9 in E and Omega_i, 3e-8 in times
and distances), a TabulatedBackground of a sampled background, and a CLASS run (the reference data of
`TransferProviders/tests`; `make run DATA=dir`, SKIP without them): parameters inferred from the file, and
`make_lcdm` with them = CLASS to 2e-7.
`test_models` and `test_scalar_fields`: every model against the solver's model of the same name and parameters
(1e-8 in E and Omega_i, 4e-8 in times and distances), the extra options against SimulationCode's JBD, quintessence
and EDE cosmologies (1e-6), and limits (LCDM limits, tracker, closure, finite-difference drho/dx).

## Not yet here

Reading SimulationCode / solver parameter files; the linear growth (`GrowthHistory`, separate from the background).
