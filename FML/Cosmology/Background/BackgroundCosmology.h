#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDCOSMOLOGY_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDCOSMOLOGY_H

// =============================================================================
// Background cosmology: expansion history, conformal time, cosmic time, distances.
//
// Time variable x = ln(a). SI units (H in 1/s, eta in m, t in s).
//
// The model (Model.h) is a list of species and a theory of gravity. The gravity gives the expansion
// rate from the species densities; in GR
//   E^2(x) = (H/H0)^2 = sum_i rho_i(x)/rho_c0 + Omega_K e^{-2x}.
// The model is built from the registry (params.model, default lcdm), or passed explicitly. The named
// accessors rhoB, rhoCDM, ... sum the species with that Role (0 if there is none).
// Everything else (H, Hp and derivatives, Omega_i, eta, t, distances) follows from E^2.
//
// Usage:
//   BackgroundCosmology cosmo(params);
//   cosmo.solve();
//   cosmo.info();
//   cosmo.output("data/background.txt");
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

class MassiveNeutrinoSpecies;

class BackgroundCosmology {
  public:
    // Model selected by params.model (Models::build)
    explicit BackgroundCosmology(const CosmologyParameters& params);
    // Explicit model (e.g. assembled in a test)
    BackgroundCosmology(const CosmologyParameters& params, Model model);

    // Integrate eta(x) and t(x) on [x_start, x_end] and store them as splines
    void solve(double x_start = -20.0, double x_end = 5.0, int n_points = 5000);

    // -------------------------------------------------------------------------
    // Model
    // -------------------------------------------------------------------------
    const Model& get_model() const { return model; }
    const std::vector<std::shared_ptr<const Species>>& species() const { return model.species; }
    const Gravity& gravity() const { return *model.gravity; }

    // Sum over all species of rho_i / rho_c0 and its x-derivatives (curvature not included)
    double rho_species(double x) const;
    double drho_species(double x) const;
    double ddrho_species(double x) const;
    // Sum over the species with a given role
    double rho_of(Species::Role role, double x) const;
    double drho_of(Species::Role role, double x) const;
    double ddrho_of(Species::Role role, double x) const;
    double p_of(Species::Role role, double x) const;

    // -------------------------------------------------------------------------
    // Energy densities in units of today's critical density: rho_i(x) / rho_c0
    // -------------------------------------------------------------------------
    double rhoB(double x) const     { return rho_of(Species::Role::Baryons, x); }
    double rhoCDM(double x) const   { return rho_of(Species::Role::CDM, x); }
    double rhoGamma(double x) const { return rho_of(Species::Role::Photons, x); }
    double rhoNu(double x) const    { return rho_of(Species::Role::MasslessNeutrinos, x); }
    double rhoNcdm(double x) const  { return rho_of(Species::Role::MassiveNeutrinos, x); }
    double rhoDE(double x) const    { return rho_of(Species::Role::DarkEnergy, x); }
    double rhoK(double x) const     { return OmegaK0 * std::exp(-2.0 * x); }   // curvature, Omega_K e^{-2x}

    double drhoBdx(double x) const     { return drho_of(Species::Role::Baryons, x); }
    double drhoCDMdx(double x) const   { return drho_of(Species::Role::CDM, x); }
    double drhoGammadx(double x) const { return drho_of(Species::Role::Photons, x); }
    double drhoNudx(double x) const    { return drho_of(Species::Role::MasslessNeutrinos, x); }
    double drhoNcdmdx(double x) const  { return drho_of(Species::Role::MassiveNeutrinos, x); }
    double drhoDEdx(double x) const    { return drho_of(Species::Role::DarkEnergy, x); }
    double drhoKdx(double x) const     { return -2.0 * rhoK(x); }

    double ddrhoBddx(double x) const     { return ddrho_of(Species::Role::Baryons, x); }
    double ddrhoCDMddx(double x) const   { return ddrho_of(Species::Role::CDM, x); }
    double ddrhoGammaddx(double x) const { return ddrho_of(Species::Role::Photons, x); }
    double ddrhoNuddx(double x) const    { return ddrho_of(Species::Role::MasslessNeutrinos, x); }
    double ddrhoNcdmddx(double x) const  { return ddrho_of(Species::Role::MassiveNeutrinos, x); }
    double ddrhoDEddx(double x) const    { return ddrho_of(Species::Role::DarkEnergy, x); }
    double ddrhoKddx(double x) const     { return 4.0 * rhoK(x); }

    // Massive neutrinos (MassiveNeutrinoSpecies; all 0 without): pressure (units of rho_c0) and dp/dx,
    // w = p/rho goes from 1/3 to 0
    bool   has_massive_nu() const { return ncdm != nullptr; }
    double pNcdm(double x) const;
    double dpNcdmdx(double x) const;
    double pseudo_pNcdm(double x) const;   // pseudo-pressure, dp/dx = -(5p - pseudo_p)
    double M_ncdm_of_x(double x) const;    // m_nu c^2 a / (k_B T_ncdm(a)): mass over temperature at x
    // rho_ncdm a^4 / I_rho(M) in units of rho_c0, so that rho_ncdm = ncdm_prefactor() e^{-4x} I_rho(M)
    double ncdm_prefactor() const;

    // E^2 = (H/H0)^2 and derivatives (from the gravity)
    double E2_of_x(double x) const;
    double dE2dx_of_x(double x) const;
    double ddE2ddx_of_x(double x) const;

    // -------------------------------------------------------------------------
    // Expansion rate (analytic, available before solve())
    // -------------------------------------------------------------------------
    double H_of_x(double x) const;             // H        [1/s]
    double Hp_of_x(double x) const;            // Hp = aH  [1/s]
    double dHpdx_of_x(double x) const;         // dHp/dx   [1/s]
    double ddHpddx_of_x(double x) const;       // d^2Hp/dx^2 [1/s]

    // Density parameters Omega_i(x) = rho_i(x) / rho_crit(x)
    double OmegaB_of_x(double x) const;
    double OmegaCDM_of_x(double x) const;
    double OmegaGamma_of_x(double x) const;
    double OmegaNu_of_x(double x) const;
    double OmegaNcdm_of_x(double x) const;
    double OmegaK_of_x(double x) const;
    double OmegaDE_of_x(double x) const;
    double OmegaM_of_x(double x) const { return OmegaB_of_x(x) + OmegaCDM_of_x(x); }
    double OmegaR_of_x(double x) const { return OmegaGamma_of_x(x) + OmegaNu_of_x(x); }

    double TCMB_of_x(double x) const;          // Photon temperature [K]

    // -------------------------------------------------------------------------
    // Times and distances (require solve())
    // -------------------------------------------------------------------------
    double eta_of_x(double x) const;           // Conformal time            [m]
    double detadx_of_x(double x) const;        // c / Hp                    [m]
    double t_of_x(double x) const;             // Cosmic time               [s]
    double eta0() const;                       // Conformal time today      [m]
    double t0() const;                         // Age of the universe       [s]

    // -------------------------------------------------------------------------
    // Spatial geometry. Curvature K = -Omega_K H0^2 / c^2 [1/m^2]: K < 0 open, K > 0 closed.
    // sinK(chi) is the comoving transverse distance for comoving radial distance chi,
    //   sinK = sin(sqrt(K) chi)/sqrt(K) (closed), chi (flat), sinh(sqrt(-K) chi)/sqrt(-K) (open),
    // and cotK(chi) = d ln sinK / d chi (= 1/chi when flat). These enter the line-of-sight
    // integrals and the truncation of the Boltzmann hierarchies in a curved universe.
    // Both use series expansions for |K| chi^2 << 1, so they are smooth through K = 0.
    // -------------------------------------------------------------------------
    double get_K() const { return K; }
    double sinK(double chi) const;
    double cotK(double chi) const;

    double comoving_distance_of_x(double x) const;   // chi = eta0 - eta     [m]
    double transverse_distance_of_x(double x) const; // r(chi), curvature-corrected [m]
    double angular_distance_of_x(double x) const;    // d_A = a r            [m]
    double luminosity_distance_of_x(double x) const; // d_L = r / a          [m]

    // Characteristic epochs (x-values), found by root finding on [x_start, x_end]
    double x_matter_radiation_equality() const;
    double x_matter_DE_equality() const;
    double x_acceleration_onset() const;       // where d^2a/dt^2 = 0, i.e. dHp/dx = 0

    // -------------------------------------------------------------------------
    const CosmologyParameters& get_params() const { return params; }
    double get_H0() const { return H0; }
    double get_x_start() const { return x_start; }
    double get_x_end() const { return x_end; }
    bool   is_solved() const { return solved; }

    // Print key numbers; write a table on [x_min, x_max] (defaults: solve range)
    void info() const;
    void output(const std::string& filename, double x_min = 0.0, double x_max = 0.0,
                int n_points = 2000) const;

  private:
    CosmologyParameters params;
    Model model;
    double H0;
    double OmegaK0;
    double K;  // Spatial curvature -Omega_K H0^2 / c^2 [1/m^2]
    std::vector<std::vector<const Species*>> by_role;   // species of each Role
    const MassiveNeutrinoSpecies* ncdm = nullptr;

    // Initial values eta(x_start), t(x_start): exact solution for a universe of matter (baryons + CDM)
    // and radiation (photons, neutrinos, massive neutrinos while relativistic), accurate to
    // ~ Omega_DE/Omega_M a^3 at early times.
    double eta_initial(double x) const;
    double t_initial(double x) const;

  private:
    double x_start = 0.0, x_end = 0.0;
    bool solved = false;
    Spline1D eta_spline;
    Spline1D t_spline;

    void require_solved(const char* func) const;
};

} // namespace FML::COSMOLOGY

#endif
