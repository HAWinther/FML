#ifndef FML_COSMOLOGY_SPECTRA_HALOFIT_H
#define FML_COSMOLOGY_SPECTRA_HALOFIT_H

// =============================================================================
// Nonlinear matter power spectrum with Halofit in the revised form of Takahashi et al. (2012, arXiv:1208.2701),
// as CLASS's "halofit" (without the massive-neutrino corrections of Bird et al. 2012, i.e. exact for m_nu = 0).
//
// At each redshift, from the linear Delta^2_L(k) = k^3 P_L/(2 pi^2):
//   sigma^2(R) = int dln k Delta^2_L e^{-k^2 R^2},   sigma(1/k_sigma) = 1,
//   n_eff = -3 - dln sigma^2/dln R,   C = -d^2 ln sigma^2/dln R^2   (at R = 1/k_sigma),
//   Delta^2_NL = Delta^2_Q + Delta^2_H,  y = k/k_sigma,
//   Delta^2_Q = Delta^2_L (1 + Delta^2_L)^beta/(1 + alpha Delta^2_L) exp(-y/4 - y^2/8),
//   Delta^2_H = a y^{3 f1} / (1 + b y^{f2} + (c f3 y)^{3 - gamma}) / (1 + mu/y + nu/y^2),
// with the coefficients a, b, c, gamma, alpha, beta, mu, nu (functions of n_eff, C, Omega_DE (1 + w)) and
// f_i = Omega_m^{-0.0307, -0.0585, 0.0743} of Takahashi et al. (their eqs. A6-A13). Omega_m, Omega_DE and w are
// those of the background at that redshift (w = p/rho of the dark-energy species; -1 if there is none).
// Valid for LCDM and wCDM; for other models it is an extrapolation.
//
// ratio(x, k) = P_NL/P_L is tabulated on a grid in x (z <= z_max) and ln k and splined; 1 at z > z_max and where
// sigma < 1 at all scales (no nonlinear scale).
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

#include <functional>
#include <vector>

namespace FML::COSMOLOGY {

    class Halofit {
      public:
        // P_lin(x, k): linear power spectrum [m^3] at k [1/m] (extrapolated beyond the perturbation k-range)
        Halofit(const BackgroundCosmology & bg,
                const std::function<double(double, double)> & P_lin,
                double z_max = 10.0,
                int n_z = 60);

        double ratio(double x, double k) const; // P_NL / P_L
        // One redshift: Delta^2_NL from Delta^2_L on the log-k grid lnk (k in 1/Mpc); returns false (and copies
        // Delta^2_L) if there is no nonlinear scale. k_sigma [1/Mpc], n_eff and C are returned for tests.
        static bool nonlinear(const std::vector<double> & lnk,
                              const std::vector<double> & D2L,
                              double Omega_m,
                              double Omega_de,
                              double w,
                              std::vector<double> & D2NL,
                              double & k_sigma,
                              double & n_eff,
                              double & C);

      private:
        double x_min;
        Spline2D log_ratio; // ln(P_NL/P_L)(x, ln k[1/Mpc])
        double lnk_min, lnk_max;
    };

} // namespace FML::COSMOLOGY

#endif
