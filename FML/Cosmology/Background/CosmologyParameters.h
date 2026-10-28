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
// Massive neutrinos: one massive species of mass m_nu (eV) with temperature T_ncdm_over_TCMB * TCMB
// (CLASS default 0.71611, which gives m_nu / (93.14 eV h^2) for Omega_ncdm today and contributes
// N_ncdm = (0.71611 / (4/11)^(1/3))^4 = 1.0132 to Neff). Neff is the total; the remaining
// Neff - N_ncdm species are massless. m_nu = 0 means all Neff species are massless.
//
// Model: the name of a model in the registry (Model.h, default lcdm) and its own parameters
// (model_params, name -> value; parameters that are not given take the registry default).
// =============================================================================

#include <iosfwd>
#include <array>
#include <map>
#include <string>

namespace FML::COSMOLOGY {

    struct CosmologyParameters {
        // --- Background ----------------------------------------------------------
        double h = 0.67;                   // H0 / (100 km/s/Mpc)
        double OmegaB = 0.05;              // Baryons today
        double OmegaCDM = 0.267;           // Cold dark matter today
        double OmegaK = 0.0;               // Curvature today (0 = flat)
        double Neff = 3.046;               // Effective number of neutrinos, massless + massive (0 = no neutrinos)
        double m_nu = 0.0;                 // Mass of the single massive neutrino species [eV] (0 = all massless)
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

        // --- Model (Model.h) -----------------------------------------------------
        std::string model = "lcdm";
        std::map<std::string, double> model_params;

        // --- Primordial power spectrum -------------------------------------------
        double A_s = 2.1e-9;   // Amplitude at k_pivot
        double n_s = 0.965;    // Spectral index
        double k_pivot = 0.05; // Pivot scale [1/Mpc]
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
        bool has_massive_nu() const { return m_nu > 0.0; }
        double N_ncdm() const; // Contribution of the massive species to Neff (0 if m_nu = 0)
        double T_ncdm() const { return T_ncdm_over_TCMB * TCMB; } // [K]
        double M_ncdm() const;    // m_nu c^2 / (k_B T_ncdm), the mass in units of the temperature today
        double OmegaNcdm() const; // Massive neutrinos today (Fermi-Dirac momentum integral)
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
        static CosmologyParameters Planck2018(); // Planck 2018 TT,TE,EE+lowE+lensing best fit
    };

} // namespace FML::COSMOLOGY

#endif
