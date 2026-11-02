#ifndef FML_COSMOLOGY_BACKGROUND_COSMOLOGYPARAMETERS_H
#define FML_COSMOLOGY_BACKGROUND_COSMOLOGYPARAMETERS_H

// =============================================================================
// All cosmological / model parameters in one place.
//
// The free parameters are plain data members. Derived quantities (H0, Omega_gamma,
// Omega_nu, Omega_Lambda) are member functions, so they can never get out of sync
// with the free parameters. Omega_Lambda is fixed by
//   Omega_b + Omega_cdm + Omega_gamma + Omega_nu + Omega_k + Omega_Lambda = 1.
//
// Massive neutrinos: deg_ncdm degenerate massive species of mass m_nu (eV) each (CLASS's deg_ncdm; default 1) with
// temperature T_ncdm_over_TCMB * TCMB (CLASS default 0.71611, which gives m_nu / (93.14 eV h^2) for Omega_ncdm today
// per species and contributes N_ncdm = (0.71611 / (4/11)^(1/3))^4 = 1.0132 per species to Neff). Neff is the total;
// the remaining Neff - N_ncdm species are massless. m_nu = 0 means all Neff species are massless. The degenerate
// species have the same perturbations: they are solved as one (with deg_ncdm times the density).
// Species of different masses: m_nu_list (one entry per species; 0 = massless, equal masses are degenerate) instead
// of m_nu and deg_ncdm. Each distinct mass has its own hierarchy; delta_ncdm, v_ncdm are then those of all of them.
//
// Model: the name of a model in the model registry (Cosmology/Perturbations/Presets.h, default lcdm) and its own
// parameters (model_params, name -> value; parameters that are not given take the registry default).
// =============================================================================

#include <iosfwd>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    struct CosmologyParameters {
        // --- Background ----------------------------------------------------------
        double h = 0.67;                   // H0 / (100 km/s/Mpc)
        double OmegaB = 0.05;              // Baryons today
        double OmegaCDM = 0.267;           // Cold dark matter today
        double OmegaK = 0.0;               // Curvature today (0 = flat)
        double Neff = 3.046;               // Effective number of neutrinos, massless + massive (0 = no neutrinos)
        double m_nu = 0.0;                 // Mass of each massive neutrino species [eV] (0 = all massless)
        int deg_ncdm = 1;                  // Number of degenerate massive species of mass m_nu
        std::vector<double> m_nu_list;     // Masses of the species [eV] (not empty: instead of m_nu, deg_ncdm)
        double T_ncdm_over_TCMB = 0.71611; // Temperature of the massive species today / TCMB
        double TCMB = 2.7255;              // CMB temperature today [K]

        // --- Recombination / reionization ----------------------------------------
        double Yp = 0.245;              // Primordial helium mass fraction (used unless Yp_bbn)
        bool Yp_bbn = false;            // Yp from standard BBN for omega_b and Neff (BBN.h, as CLASS's YHe = BBN)
        double helium_fraction() const; // the Yp actually used
        bool reionization = true;
        double z_reion = 8.0;       // Hydrogen (+ first He) reionization redshift
        double delta_z_reion = 0.5; // Width
        double z_He_reion = 3.5;    // Second helium reionization
        double delta_z_He_reion = 0.5;

        // --- Model (the model registry, Cosmology/Perturbations/Presets.h) ---------
        std::string model = "lcdm";
        std::map<std::string, double> model_params;

        // --- Primordial power spectrum -------------------------------------------
        double A_s = 2.1e-9;   // Amplitude at k_pivot
        double n_s = 0.965;    // Spectral index
        double k_pivot = 0.05; // Pivot scale [1/Mpc]
        // Running dn_s/dln k (CLASS's alpha_s): P_R(k) = A_s (k/k_pivot)^{n_s - 1 + alpha_s/2 ln(k/k_pivot)}.
        // Adiabatic mode only (an error with isocurvature modes).
        double alpha_s = 0.0;
        // Tensor modes: P_t(k) = r A_s (k/k_pivot)^n_t (computed when r > 0; flat universes only). With
        // n_t_consistency the tilt follows the slow-roll consistency relation n_t = -r/8 (2 - r/8 - n_s) (as CLASS'
        // n_t = scc), otherwise n_t.
        // Isocurvature modes cdi, bi, nid, niv (Isocurvature.h, as CLASS): amplitude ratio f (P_II = f^2 A_s
        // (k/k_pivot)^{n - 1}; 0 = off), tilt n and correlation c with the adiabatic mode
        std::array<double, 4> iso_f{{0.0, 0.0, 0.0, 0.0}};
        std::array<double, 4> iso_n{{1.0, 1.0, 1.0, 1.0}};
        std::array<double, 4> iso_c{{0.0, 0.0, 0.0, 0.0}};
        double r = 0.0;
        double n_t = 0.0;
        bool n_t_consistency = true;
        double tensor_tilt() const { return n_t_consistency ? -r / 8.0 * (2.0 - r / 8.0 - n_s) : n_t; }

        // --- Derived quantities --------------------------------------------------
        double H0() const;         // [1/s]
        double OmegaGamma() const; // Photons (from TCMB)
        double OmegaNu() const;    // Massless neutrinos (from Neff - N_ncdm and OmegaGamma)
        double OmegaR() const { return OmegaGamma() + OmegaNu(); } // Photons + massless neutrinos
        bool has_massive_nu() const { return not neutrino_masses().empty(); }
        /// The masses of the massive species [eV]: m_nu_list without its zeros, or deg_ncdm times m_nu
        std::vector<double> neutrino_masses() const;
        double N_ncdm() const; // Contribution of the massive species to Neff (all of them; 0 without)
        double T_ncdm() const { return T_ncdm_over_TCMB * TCMB; } // [K]
        double M_ncdm() const; // m_nu c^2 / (k_B T_ncdm), the mass in units of the temperature today (one species)
        double OmegaNcdm() const; // Massive neutrinos today, all species (Fermi-Dirac momentum integral)
        double OmegaM() const { return OmegaB + OmegaCDM; }
        double OmegaLambda() const; // Fixed by the closure relation

        // Value of a model parameter (model_params or the registry default)
        double model_param(const std::string & name) const;

        // Throws std::invalid_argument if the parameters are unphysical (or the model is unknown)
        void validate() const;

        // Write "name = value" lines (read by python/params.py for comparisons with CLASS)
        void write(const std::string & filename) const;

        void print(std::ostream & os) const;
        void print() const;

        // Named parameter sets
        static CosmologyParameters fiducial() { return CosmologyParameters{}; }
        // Planck 2018 (paper VI, Table 2) means of TT,TE,EE+lowE+lensing+BAO with z_reion of TT,TE,EE+lowE+lensing,
        // and massless neutrinos (Planck's baseline has m_nu = 0.06 eV): 100 theta_s is 1.04033 instead of
        // 1.04186. Kept for the existing references; for Planck's baseline use Planck2018Baseline().
        static CosmologyParameters Planck2018();
        // Planck 2018 baseline: means of TT,TE,EE+lowE+lensing (Table 2) with one massive neutrino of 0.06 eV
        // (N_eff = 3.046). With CLASS (RECFAST, reio_camb): 100 theta_s = 1.04184, tau = 0.0542, sigma8 = 0.8106.
        static CosmologyParameters Planck2018Baseline();
    };

} // namespace FML::COSMOLOGY

#endif
