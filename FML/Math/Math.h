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

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
#include <optional>
#endif

namespace FML {

    //===================================================================================
    /// This namespace deals with special math functions and general math algorithms
    /// needed in the library
    //===================================================================================
    namespace MATH {

#ifdef USE_GSL
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
        using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
#endif
        using DVector = std::vector<double>;

        /// Python linspace. Generate a linearly spaced array (num = 1 gives {xmin}, num <= 0 an empty array)
        DVector linspace(double xmin, double xmax, int num);
        
        /// Python logspace. Generate a log spaced array
        DVector logspace(double xmin, double xmax, int num, double base = 10.0);

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
        /// Find roots \f$ f(x) = 0 \f$ using Newtons method. Requires C++17
        std::optional<double> find_root_newton(const std::function<double(double)> & function,
                                               const std::function<double(double)> & derivative,
                                               double x0,
                                               double epsilon = 1e-7);
#endif

        /// Find roots \f$ f(x) = 0 \f$ using bisection. f must have different signs at the end-points (throws
        /// if not, or if f is NaN there). The accuracy epsilon is relative to the size of the range:
        /// we stop when the bracket is smaller than epsilon * |xrange.second - xrange.first| (or cannot be split
        /// further in floating point) and return the lower end of the bracket. Points with f = 0 are treated as being on the side of
        /// xlow (so a root exactly at xlow returns xlow)
        double find_root_bisection(const std::function<double(double)> & function,
                                   std::pair<double, double> xrange,
                                   double epsilon = 1e-7);

#ifdef USE_GSL
        /// Find roots \f$ f(x) = 0 \f$ using bisection with f given as a spline. Requires GSL.
        /// The range can be given in any order; {0,0} means the full range of the spline. If f has the same sign
        /// at both end-points a coarse grid is searched for a sign change.
        double find_root_bisection(const Spline & y,
                                   double y_value,
                                   std::pair<double, double> xrange = {0.0, 0.0},
                                   double epsilon = 1e-7);
#endif

        /// Hyperspherical bessel functions using recursion formula
        DVector Hyperspherical_j_ell_array(int lmax, const double nu, const double chi, double K);

        /// Spherical bessel functions using recursion formula
        DVector j_ell_array(int lmax, const double x);

        /// Spherical bessel function \f$ j_\ell(x) \f$ from CXX or GSL with fix for very small or large arguments.
        double j_ell(const int ell, const double arg);

        /// Legendre polynomials. Computes an array with \f$ P_0(\mu), P_1(\mu), ..., P_{ellmax}(\mu) \f$
        /// using a recursion relation
        std::vector<double> legendre_ell_of_mu_vector(double mu, int ell_max);

#ifdef USE_GSL
        /// Airy function \f$ {\rm Ai}(x) \f$ (requires GSL). Found by solving and splining \f$ y'' - xy = 0 \f$.
        double Airy_Ai(double z);
#endif

        // General method for evaluating continued fraction
        std::pair<double, bool> GeneralizedLentzMethod(const std::function<double(int)> & a,
                                                       const std::function<double(int)> & b,
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
