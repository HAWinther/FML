#ifndef FML_MATH_NUMERICAL_METHODS_H
#define FML_MATH_NUMERICAL_METHODS_H
#include <FML/Math/Grids.h>
#include <functional>
#include <vector>

namespace FML {
    namespace MATH {
        /// Trapezoidal integral of matching tabulated arrays; empty/single-point arrays give zero.
        double trapz(const std::vector<double> & x, const std::vector<double> & y);

        /// Fill j_l(x), l=0..lmax using stable upward/Miller recurrence. Buffer needs lmax+1 entries.
        /// This retains the cosmology solver's small-argument approximation and normalization.
        void spherical_bessel_j_array(int lmax, double x, double * values);

#ifdef USE_GSL
        /// Adaptive quadrature (GSL QAG, 21-point rule, workspace size 1000).
        /// Reports a warning and returns the estimate on non-convergence.
        double integrate(const std::function<double(double)> & f,
                         double a,
                         double b,
                         double epsabs = 0.0,
                         double epsrel = 1e-8);

        /// Brent root finder. The endpoints must bracket a root; either endpoint order is accepted.
        /// Reports a warning and returns the last estimate if max_iter is exhausted.
        double find_root(const std::function<double(double)> & f,
                         double a,
                         double b,
                         double epsabs = 1e-12,
                         double epsrel = 1e-12,
                         int max_iter = 200);

        /// Spherical Bessel j_l(x), with the cosmology solver's high-l/underflow handling.
        /// Distinct from j_ell: that older API has a more aggressive small-argument cutoff.
        double spherical_bessel_j(int ell, double x);
#endif
    } // namespace MATH
} // namespace FML
#endif
