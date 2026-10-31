#ifndef FML_BACKGROUND_BACKGROUNDPARAMETERS_H
#define FML_BACKGROUND_BACKGROUNDPARAMETERS_H

// =====================================================================================================================
// The parameters every background has, and the range in x = ln a it is valid on.
//
// Radiation is always included: photons from T_CMB and N_eff neutrinos. With massive neutrinos (m_nu_eV, one entry
// per massive species, degenerate masses repeated) each massive species has the temperature T_ncdm_over_TCMB * T_CMB
// (CLASS default 0.71611, which gives Omega_nu h^2 = m_nu / 93.14 eV) and contributes
//   N_ncdm = (T_ncdm_over_TCMB / (4/11)^(1/3))^4 = 1.0132
// to N_eff (N_eff is the total; the remaining N_eff - n_massive N_ncdm species are massless), as the Einstein-
// Boltzmann solver and CLASS do.
// =====================================================================================================================

#include <FML/Units/Units.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    struct BackgroundParameters {
        double h{0.67};                   ///< H0 / (100 km/s/Mpc)
        double Omega_b{0.049};            ///< Baryons today
        double Omega_cdm{0.265};          ///< Cold dark matter today
        double Omega_k{0.0};              ///< Curvature today (> 0 open, < 0 closed)
        double T_CMB{2.7255};             ///< CMB temperature today [K]
        double N_eff{3.046};              ///< Effective number of neutrinos, massless + massive
        std::vector<double> m_nu_eV{};    ///< Masses of the massive neutrino species [eV]
        double T_ncdm_over_TCMB{0.71611}; ///< Temperature of the massive species today / T_CMB

        /// H0 in 1/s
        double H0_SI() const { return h * FML::UTILS::SI::H0_over_h; }
        /// Photons today: (pi^2/15) (k_B T)^4 / (hbar c)^3 / (rho_c0 c^2)
        double Omega_gamma() const {
            using namespace FML::UTILS::SI;
            const double kT = k_b * T_CMB;
            const double rho_c0c2 = 3.0 * H0_SI() * H0_SI() / (8.0 * M_PI * G) * c * c;
            return M_PI * M_PI / 15.0 * std::pow(kT, 4) / std::pow(hbar * c, 3) / rho_c0c2;
        }
        /// Contribution of one massive species to N_eff
        double N_ncdm_per_species() const { return std::pow(T_ncdm_over_TCMB / std::cbrt(4.0 / 11.0), 4); }
        /// Number of massless neutrino species
        double N_massless() const { return N_eff - double(m_nu_eV.size()) * N_ncdm_per_species(); }
        /// Massless neutrinos today
        double Omega_nu_massless() const {
            return 7.0 / 8.0 * std::pow(4.0 / 11.0, 4.0 / 3.0) * N_massless() * Omega_gamma();
        }
        double sum_m_nu_eV() const {
            double sum = 0.0;
            for (double m : m_nu_eV)
                sum += m;
            return sum;
        }

        void validate() const {
            const std::string msg = "[BackgroundParameters] ";
            if (not(h > 0.0) or not std::isfinite(h))
                throw std::invalid_argument(msg + "h must be positive");
            if (not(Omega_b >= 0.0) or not(Omega_cdm >= 0.0) or not std::isfinite(Omega_k))
                throw std::invalid_argument(msg + "Omega_b and Omega_cdm must be >= 0 and Omega_k finite");
            if (not(T_CMB >= 0.0) or not(N_eff >= 0.0) or not(T_ncdm_over_TCMB > 0.0))
                throw std::invalid_argument(msg + "T_CMB, N_eff must be >= 0 and T_ncdm_over_TCMB > 0");
            for (double m : m_nu_eV)
                if (not(m > 0.0) or not std::isfinite(m))
                    throw std::invalid_argument(msg + "Neutrino masses must be positive (leave m_nu_eV empty for none)");
            if (N_massless() < -1e-10)
                throw std::invalid_argument(msg + "N_eff = " + std::to_string(N_eff) + " is smaller than the " +
                                            std::to_string(m_nu_eV.size()) + " massive species contribute");
            if (not m_nu_eV.empty() and T_CMB == 0.0)
                throw std::invalid_argument(msg + "Massive neutrinos need T_CMB > 0");
        }
    };

    /// The range in x = ln a a background is valid on (and where its times and distances are tabulated)
    struct BackgroundRange {
        double x_min{std::log(1e-10)};
        double x_max{std::log(10.0)};
        /// Points per unit x of the tables of the conformal time, cosmic time and comoving distance
        int points_per_unit_x{200};
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
