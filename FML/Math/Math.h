#ifndef _MATH_HEADER
#define _MATH_HEADER

#ifdef USE_GSL
#include <gsl/gsl_sf_bessel.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#ifdef USE_GSL
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#endif

namespace FML {
    namespace MATH {

#ifdef USE_GSL
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
#endif
        using DVector = std::vector<double>;

        /// Python linspace. Generate a lineary spaced array
        DVector linspace(double xmin, double xmax, int num);

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
        /// Find roots \f$ f(x) = 0 \f$ using Newtons method. Requires C++17
        std::optional<double> find_root_newton(const std::function<double(double)> & function,
                                               const std::function<double(double)> & derivative,
                                               double x0,
                                               double epsilon = 1e-7);
#endif

        /// Find roots \f$ f(x) = 0 \f$ using bisection
        double find_root_bisection(const std::function<double(double)> & function,
                                   std::pair<double, double> xrange,
                                   double epsilon = 1e-7);

#ifdef USE_GSL
        /// Find roots \f$ f(x) = 0 \f$ using bisection with f given as a spline. Requires GSL.
        double find_root_bisection(const Spline & y,
                                   double y_value,
                                   std::pair<double, double> xrange = {0.0, 0.0},
                                   double epsilon = 1e-7);
#endif

        /// Hyperspherical bessel functions using recursion formula
        DVector Hyperspherical_j_ell_array(int lmax, const double nu, const double chi, double K);

        /// Spherical bessel functions using recursion formula
        DVector j_ell_array(int lmax, const double x);

        /// Spherical bessel functions from CXX or GSL with fix for very small or large arguments
        double j_ell(const int ell, const double arg);

#ifdef USE_GSL
        /// Airy function (requires GSL)
        double Airy_Ai(double z);
#endif

        /// General method for evaluating continued fraction. Gives back the result and if it converged
        std::pair<double, bool> GeneralizedLentzMethod(std::function<double(int)> & a,
                                                       std::function<double(int)> & b,
                                                       double epsilon,
                                                       int maxsteps);

#ifdef USE_GSL
        /// WKB approximation for The hyper spherical bessel functions (for a curved Universe)
        /// For a flat Universe call with nu = 1.0, chi = k*(eta0-eta) and K = 0.0. Requires GSL.
        double HyperSphericalBesselWKB(int ell, double nu, double chi, double K);
#endif
    } // namespace MATH
} // namespace FML
#endif
