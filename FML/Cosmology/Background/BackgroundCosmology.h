#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDCOSMOLOGY_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDCOSMOLOGY_H

// =============================================================================
// Background cosmology: expansion history, conformal time, cosmic time, distances.
//
// Time variable x = ln(a). SI units (H in 1/s, eta in m, t in s).
//
// This class is the solver's interface to the background; the background itself is a
// BACKGROUND::ComponentBackground (Cosmology/Background: species + field equations, one solve that tabulates
// the conformal time, the cosmic time and the comoving distance), and the perturbations of the model come
// from a PERTURBATIONS::PerturbationModel (Cosmology/Perturbations) built on the same background:
//
//   BackgroundCosmology(params)         params.model from the model registry (PERTURBATIONS::make_perturbation_model
//                                       on PERTURBATIONS::make_model_background), with params.model_params
//   BackgroundCosmology(params, pmodel) an explicit PERTURBATIONS::PerturbationModel (a custom model, e.g. a
//                                       PerturbationModel_Assembled) and its background; the cosmological parameters
//                                       of params must be those of its background. The background is not rebuilt by
//                                       solve() (its range must cover the solve range)
//
// The named accessors rhoB, rhoCDM, ... sum the species of that component (0 if there is none).
// Second derivatives (ddHpddx, ddE2ddx, ddrho...): a 5-point central difference (step 1e-3 in x) of the analytic
// first derivatives of the ComponentBackground (relative error ~1e-11 for smooth backgrounds, 1e-8 near the knots of
// the massive-neutrino tables).
//
// Usage:
//   BackgroundCosmology cosmo(params);
//   cosmo.solve();
//   cosmo.info();
//   cosmo.output("data/background.txt");
// =============================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Perturbations/PerturbationModel.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    // Tensor-mode coefficients (M*^2, alpha_M, alpha_T) of the theory of gravity, see Tensors.h
    using TensorCoefficients = PERTURBATIONS::TensorCoefficients;

    class BackgroundCosmology {
      public:
        // Model selected by params.model (see above)
        explicit BackgroundCosmology(const CosmologyParameters & params);
        // Explicit perturbation model with its background (a custom model)
        BackgroundCosmology(const CosmologyParameters & params, PERTURBATIONS::PerturbationModelPtr model);
        BackgroundCosmology(BackgroundCosmology && other) noexcept;
        BackgroundCosmology(const BackgroundCosmology &) = delete;
        BackgroundCosmology & operator=(const BackgroundCosmology &) = delete;
        BackgroundCosmology & operator=(BackgroundCosmology &&) = delete;
        ~BackgroundCosmology();

        // Make the times and distances available on [x_start, x_end] (the range checked by eta_of_x, t_of_x).
        // The ComponentBackground is built by the constructor on [-25, 5] with 200 points per unit x; it is
        // only rebuilt if [x_start, x_end] is not inside that or n_points asks for a finer table
        // (n_points / (x_end - x_start) points per unit x).
        void solve(double x_start = -20.0, double x_end = 5.0, int n_points = 5000);

        // -------------------------------------------------------------------------
        // The background and the perturbation model (Cosmology/Background, Cosmology/Perturbations)
        // -------------------------------------------------------------------------
        const BACKGROUND::ComponentBackground & background() const { return *bg; }
        const BACKGROUND::ComponentBackgroundPtr & background_ptr() const { return bg; }
        const PERTURBATIONS::PerturbationModelPtr & perturbation_model() const { return pmodel; }
        const std::string & model_name() const { return params.model; }
        // Name of the theory of gravity of the perturbations ("GR", "Brans-Dicke", ...)
        std::string gravity_name() const;
        // Tensor-mode coefficients at x (of the perturbed field equations)
        TensorCoefficients tensor_coefficients(double x) const;

        // Sum over all species of rho_i / rho_c0 and its x-derivatives (curvature not included)
        double rho_species(double x) const;
        double drho_species(double x) const;
        double ddrho_species(double x) const;
        // Sum over the species of a component (BACKGROUND::Component: Baryons, CDM, Photons, ..., Other)
        double rho_of(BACKGROUND::Component c, double x) const;
        double drho_of(BACKGROUND::Component c, double x) const;
        double ddrho_of(BACKGROUND::Component c, double x) const;
        double p_of(BACKGROUND::Component c, double x) const;

        // -------------------------------------------------------------------------
        // Energy densities in units of today's critical density: rho_i(x) / rho_c0
        // -------------------------------------------------------------------------
        double rhoB(double x) const { return rho_of(BACKGROUND::Component::Baryons, x); }
        double rhoCDM(double x) const { return rho_of(BACKGROUND::Component::CDM, x); }
        double rhoGamma(double x) const { return rho_of(BACKGROUND::Component::Photons, x); }
        double rhoNu(double x) const { return rho_of(BACKGROUND::Component::MasslessNeutrinos, x); }
        double rhoNcdm(double x) const { return rho_of(BACKGROUND::Component::MassiveNeutrinos, x); }
        double rhoDE(double x) const { return rho_of(BACKGROUND::Component::DarkEnergy, x); }
        double rhoK(double x) const { return OmegaK0 * std::exp(-2.0 * x); } // curvature, Omega_K e^{-2x}

        double drhoBdx(double x) const { return drho_of(BACKGROUND::Component::Baryons, x); }
        double drhoCDMdx(double x) const { return drho_of(BACKGROUND::Component::CDM, x); }
        double drhoGammadx(double x) const { return drho_of(BACKGROUND::Component::Photons, x); }
        double drhoNudx(double x) const { return drho_of(BACKGROUND::Component::MasslessNeutrinos, x); }
        double drhoNcdmdx(double x) const { return drho_of(BACKGROUND::Component::MassiveNeutrinos, x); }
        double drhoDEdx(double x) const { return drho_of(BACKGROUND::Component::DarkEnergy, x); }
        double drhoKdx(double x) const { return -2.0 * rhoK(x); }

        double ddrhoBddx(double x) const { return ddrho_of(BACKGROUND::Component::Baryons, x); }
        double ddrhoCDMddx(double x) const { return ddrho_of(BACKGROUND::Component::CDM, x); }
        double ddrhoGammaddx(double x) const { return ddrho_of(BACKGROUND::Component::Photons, x); }
        double ddrhoNuddx(double x) const { return ddrho_of(BACKGROUND::Component::MasslessNeutrinos, x); }
        double ddrhoNcdmddx(double x) const { return ddrho_of(BACKGROUND::Component::MassiveNeutrinos, x); }
        double ddrhoDEddx(double x) const { return ddrho_of(BACKGROUND::Component::DarkEnergy, x); }
        double ddrhoKddx(double x) const { return 4.0 * rhoK(x); }

        // Massive neutrinos (all 0 without): pressure (units of rho_c0) and dp/dx, w = p/rho goes from 1/3 to 0
        bool has_massive_nu() const { return not ncdm.empty(); }
        double pNcdm(double x) const;
        double dpNcdmdx(double x) const;
        double pseudo_pNcdm(double x) const; // pseudo-pressure, dp/dx = -(5p - pseudo_p)
        // One mass only (throw with several distinct masses):
        double M_ncdm_of_x(double x) const; // m_nu c^2 a / (k_B T_ncdm(a)): mass over temperature at x
        // rho_ncdm a^4 / I_rho(M) in units of rho_c0, so that rho_ncdm = ncdm_prefactor() e^{-4x} I_rho(M)
        double ncdm_prefactor() const;

        // E^2 = (H/H0)^2 and derivatives
        double E2_of_x(double x) const;
        double dE2dx_of_x(double x) const;
        double ddE2ddx_of_x(double x) const;

        // -------------------------------------------------------------------------
        // Expansion rate (available before solve())
        // -------------------------------------------------------------------------
        double H_of_x(double x) const;       // H        [1/s]
        double Hp_of_x(double x) const;      // Hp = aH  [1/s]
        double dHpdx_of_x(double x) const;   // dHp/dx   [1/s]
        double ddHpddx_of_x(double x) const; // d^2Hp/dx^2 [1/s]

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

        double TCMB_of_x(double x) const; // Photon temperature [K]

        // -------------------------------------------------------------------------
        // Times and distances (require solve())
        // -------------------------------------------------------------------------
        double eta_of_x(double x) const;    // Conformal time            [m]
        double detadx_of_x(double x) const; // c / Hp                    [m]
        double t_of_x(double x) const;      // Cosmic time               [s]
        double eta0() const;                // Conformal time today      [m]
        double t0() const;                  // Age of the universe       [s]

        // -------------------------------------------------------------------------
        // Spatial geometry. Curvature K = -Omega_K H0^2 / c^2 [1/m^2]: K < 0 open, K > 0 closed.
        // sinK(chi) is the comoving transverse distance for comoving radial distance chi,
        //   sinK = sin(sqrt(K) chi)/sqrt(K) (closed), chi (flat), sinh(sqrt(-K) chi)/sqrt(-K) (open),
        // and cotK(chi) = d ln sinK / d chi (= 1/chi when flat). These enter the line-of-sight
        // integrals and the truncation of the Boltzmann hierarchies in a curved universe.
        // Both use series expansions for |K| chi^2 << 1, so they are smooth through K = 0 (Background::sinK, cotK).
        // -------------------------------------------------------------------------
        double get_K() const { return K; }
        double sinK(double chi) const;
        double cotK(double chi) const;

        double comoving_distance_of_x(double x) const;   // chi = eta0 - eta     [m]
        double transverse_distance_of_x(double x) const; // r(chi), curvature-corrected [m]
        double angular_distance_of_x(double x) const;    // d_A = a r            [m]
        double luminosity_distance_of_x(double x) const; // d_L = r / a          [m]

        // Characteristic epochs (x-values), found by root finding on [x_start, x_end]
        double x_matter_radiation_equality() const; // massive neutrinos: 3 p radiation, rho - 3 p matter
        double x_matter_DE_equality() const;
        double x_acceleration_onset() const; // where d^2a/dt^2 = 0, i.e. dHp/dx = 0

        // -------------------------------------------------------------------------
        const CosmologyParameters & get_params() const { return params; }
        double get_H0() const { return H0; }
        double get_x_start() const { return x_start; }
        double get_x_end() const { return x_end; }
        bool is_solved() const { return solved; }

        // Print key numbers; write a table on [x_min, x_max] (defaults: solve range)
        void info() const;
        void output(const std::string & filename, double x_min = 0.0, double x_max = 0.0, int n_points = 2000) const;

      private:
        CosmologyParameters params;
        double H0;
        double OmegaK0;
        double K;      // Spatial curvature -Omega_K H0^2 / c^2 [1/m^2]
        double L_unit; // meters per Mpc/h of the background: c / H0 / c_over_H0_mpch

        // The background and the perturbation model
        BACKGROUND::ComponentBackgroundPtr bg;
        PERTURBATIONS::PerturbationModelPtr pmodel;
        std::unique_ptr<PERTURBATIONS::PerturbedFieldEquation> tensor_fe; // tensor coefficients of pmodel
        BACKGROUND::BackgroundRange range;
        // Massive neutrinos of the background: one entry per distinct mass, with the factor of its (rescaled,
        // degenerate) species
        std::vector<std::pair<const BACKGROUND::MassiveNeutrinoSpecies *, double>> ncdm;
        const BACKGROUND::MassiveNeutrinoSpecies * single_ncdm(const char * func) const;

        // An explicit perturbation model (its background is not rebuilt)
        bool explicit_pmodel = false;

        void build_background(); // bg (+ pmodel, tensor_fe) on range
        void find_massive_neutrinos();
        double ddlogEdx(double x) const; // d^2 ln E / dx^2

      private:
        double x_start = 0.0, x_end = 0.0;
        void require_in_range(const char * func, double x) const; // throws outside [x_start, x_end]
        bool solved = false;
        void require_solved(const char * func) const;
    };

} // namespace FML::COSMOLOGY

#endif
