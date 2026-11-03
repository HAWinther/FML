#ifndef FML_HALOMODEL_MASSFUNCTIONFITS_HEADER
#define FML_HALOMODEL_MASSFUNCTIONFITS_HEADER

// Fitting functions for the halo multiplicity function f(sigma), dn/dlnM = f(sigma) rho / M |dln sigma / dln M|
//   f_tinker2008_delta(sigma, z, Delta) : spherical overdensity Delta (w.r.t. the mean density), 200 <= Delta <= 3200
//   f_watson2013_fof(sigma)             : friends-of-friends with linking length b = 0.2

#include <cmath>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            /// The Tinker et al. 2008 multiplicity function for an overdensity Delta (with respect to the mean density)
            /// in 200 <= Delta <= 3200: the parameters of their Table 2 interpolated in log(Delta) and their redshift
            /// dependence (eqs. 5-8). Returns 0 outside this range
            inline double f_tinker2008_delta(double sigma, double z, double Delta) {
                static const double table_Delta[9] = {200, 300, 400, 600, 800, 1200, 1600, 2400, 3200};
                static const double table_A[9] = {0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260};
                static const double table_a[9] = {1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66};
                static const double table_b[9] = {2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41};
                static const double table_c[9] = {1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44};
                if (Delta < 200.0 * (1.0 - 1e-10) or Delta > 3200.0 * (1.0 + 1e-10))
                    return 0.0;
                int i = 0;
                while (i < 7 and Delta > table_Delta[i + 1])
                    i++;
                const double t = std::log(Delta / table_Delta[i]) / std::log(table_Delta[i + 1] / table_Delta[i]);
                auto interp = [&](const double * v) { return v[i] + t * (v[i + 1] - v[i]); };
                const double alpha = std::pow(10.0, -std::pow(0.75 / std::log10(Delta / 75.0), 1.2));
                const double A = interp(table_A) * std::pow(1.0 + z, -0.14);
                const double a = interp(table_a) * std::pow(1.0 + z, -0.06);
                const double b = interp(table_b) * std::pow(1.0 + z, -alpha);
                const double c = interp(table_c);
                return A * (std::pow(sigma / b, -a) + 1.0) * std::exp(-c / (sigma * sigma));
            }

            /// The universal FoF (b = 0.2) multiplicity function f(sigma) of Watson et al. 2013 (eq. 12) with
            /// dn/dlnM = f(sigma) rho / M |dln sigma / dln M| (Tinker et al. 2008: f_tinker2008_delta)
            inline double f_watson2013_fof(double sigma) {
                const double A = 0.282, alpha = 2.163, beta = 1.406, gamma = 1.210;
                return A * (std::pow(beta / sigma, alpha) + 1.0) * std::exp(-gamma / (sigma * sigma));
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML

#endif
