# Cosmology/Background

One background cosmology for all of FML (design: `claude/fml-background-design.md`), in the namespace
`FML::COSMOLOGY::BACKGROUND`. The Einstein-Boltzmann solver runs on it (its `BackgroundCosmology`, in this folder, is an
SI-unit adapter of a `ComponentBackground`; `CosmologyParameters` and `BBN` are also the solver's); SimulationCode and
the other modules are to move onto it. Header-only; needs the Spline, ODESolver and Math modules (GSL).

## Structure

```
Background (abstract)          E(x), dlogE/dx, Omega(component, x), parameters, range   x = ln a
 |                              + non-virtual: eta, t, chi, x_of_chi, sinK, cotK, D_M, D_A, D_L, D_H, a/z versions,
 |                                info, output
 |-- ComponentBackground        species (Species.h) + field equations (FieldEquations.h); rho/p/drho totals
 |-- TabulatedBackground        tables of E and Omega_i; from_class(<root>background.dat)
 '-- FunctionBackground         user functions (old callback interfaces, tests)

Species.h                  PowerLawSpecies (baryons, CDM, photons, massless nu, Lambda), CPLDarkEnergy,
                           MassiveNeutrinoSpecies (rho, p, pseudo-pressure, dp/dx; FermiDiracIntegrals I_rho, I_p,
                           I_pp), ScaledSpecies (another species times a constant), ScalarFieldSpecies (the base of
                           the scalar-field dark energies)
FieldEquations.h           the base; FieldEquation_GeneralRelativity.h, FieldEquation_DGP.h, FieldEquation_BransDicke.h
BackgroundModel_<name>.h   one model each, its species and make_ function:
                           LCDM, w0wa, DGP, Quintessence (exponential, inverse power law, PNGB, hilltop), EDE,
                           Axion (ultralight-axion dark matter, its own Component::Axions), BransDicke, CoupledQuintessence, CubicGalileon, DarkScattering (wCDM), IDE,
                           Tabulated (dark energy from a table of H(z) or w(a): make_tabulated_E, make_tabulated_w;
                           not in the registry, used by SimulationCode's Tabulated cosmology)
Presets.h                  the registry (name, description, parameters with defaults and descriptions) and
                           make_background(name, params, model params): lcdm, w0wa, dgp, quintessence
                           (_inverse_power_law, _pngb, _hilltop), ede, ede_zc, axion, jbd, coupled_quintessence,
                           cubic_galileon, dark_scattering, ide, musigma, fr, symmetron, dilaton, mbeta, horndeski,
                           horndeski_table
BackgroundModel_HorndeskiTable.h   horndeski_table: a reduced Horndeski theory from the table of
                           SimulationCode/scripts/horndeski_frontend.py (BackgroundParameters::model_file): its H(a)
```

Background only: models that only modify the perturbations (mu-Sigma, Hu-Sawicky f(R), Horndeski alphas) have the
LCDM or w0wa background. The registry lists the parameters of the background only; those that only concern the
perturbations (mu0, Sigma0, fR0, fR_n, hd_* but hd_w0/hd_wa, ds_xi, ds_cs2, ide_cs2, ide_stability_check) are an error
here and belong to the model registry of `Cosmology/Perturbations/Presets.h`, which is built on this one (a model's
parameters are those of its background followed by those of its perturbations: `PERTURBATIONS::model_parameters`;
`PERTURBATIONS::make_model_background` builds the background from all of them). For the perturbations a species also gives d2rho/dx2
(`d2rhodx2`, analytic for the species here, a 5-point difference of drho/dx by default), a scalar field d2v/dphi2
(`v_phiphi`) and d2phi/dx2 (`ddphidx`), and ComponentBackground the sums `d2rho_total`, `d2rhodx2(component)`.

Units: E dimensionless, conformal time and distances in Mpc/h, cosmic time in Gyr (`t_H0`: units of 1/H0);
`sinK(chi)`, `cotK(chi)` take chi in Mpc/h (curvature `K_hmpc2()` = -Omega_k (H0/c)^2).
The linear perturbations are built on top of this module in `Cosmology/Perturbations` (this module does not depend on
it).
Radiation is always included (photons from `T_CMB`, `N_eff` neutrinos of which one per entry of `m_nu_eV` is massive).

```cpp
#include <FML/Cosmology/Background/Presets.h>
using namespace FML::COSMOLOGY::BACKGROUND;

BackgroundParameters p;          // h, Omega_b, Omega_cdm, Omega_k, T_CMB, N_eff, m_nu_eV
p.m_nu_eV = {0.06};
auto bg = make_lcdm(p);          // std::shared_ptr<const ComponentBackground>
double E = bg->E(std::log(0.5)), chi = bg->chi_of_z(1.0), DA = bg->D_A_of_z(1.0);
auto dgp = make_background("dgp", p, {{"Omega_rc", 0.25}, {"dgp_branch", -1}});
```

## Tests

`tests/`: `make run` (header-only, does not link the solver). Checks the
analytic Einstein-de Sitter and radiation-free LCDM backgrounds (flat, open, closed), the Fermi-Dirac integrals
(I_rho, I_p, I_pp against direct integration to 1e-10, and their small- and large-M limits), the massive neutrino
species (prefactor, pseudo-pressure, dp/dx against finite differences), `p_total`, `ScaledSpecies` and the
Brans-Dicke accessors, `sinK`/`cotK` (flat, open, closed, the series region and its edge),
LCDM with massive neutrinos and curvature, w0wa and DGP against the solver (1e-9 in E and Omega_i, 3e-8 in times
and distances), a TabulatedBackground of a sampled background, and a CLASS run (the reference data of
`TransferProviders/tests`; `make run DATA=dir`, SKIP without them): parameters inferred from the file, and
`make_lcdm` with them = CLASS to 2e-7.
`test_models` and `test_scalar_fields`: every model against the solver's model of the same name and parameters
(1e-8 in E and Omega_i, 4e-8 in times and distances), the extra options against SimulationCode's JBD, quintessence
and EDE cosmologies (1e-6), limits (LCDM limits, tracker, closure, finite-difference drho/dx) and the registry.
The solver references are the backgrounds of the solver's old model classes (`Cosmology/Models`, removed), stored in
`tests/data/solver_*.txt` (E, dlogE/dx, Omega_i, eta, t, chi, D_A at the x of the tests; how they were made:
`tests/SolverReference.h` and the file headers).

## Not yet here

Reading SimulationCode / solver parameter files. The linear growth is separate from the background: `Cosmology/Growth`.
