#ifndef FML_COSMOLOGY_NUMERICS_UTILS_H
#define FML_COSMOLOGY_NUMERICS_UTILS_H

// =============================================================================
// Physical constants, grid helpers, numerical utilities, and the splines and ODE solver of FML under the
// names used in the solver:
//   Spline1D = FML::INTERPOLATION::SPLINE::GSLSpline, Spline2D = ...::GSLSpline2D,
//   UniformSpline1D = ...::UniformSpline, ODESolver = FML::SOLVERS::ODESOLVER::ODESolver (+ ODEFunction),
//   ODEStepper = a GSL stepper (gsl_odeiv2_step_rk8pd, ...; the implicit ones are used with
//   ODESolver::set_finite_difference_jacobian(true))
//
// Conventions used throughout the code:
//   * SI units internally (H0 in 1/s, distances in m, times in s)
//   * Time variable x = ln(a)
// =============================================================================

#include <FML/ODESolver/ODESolver.h>
#include <FML/Units/Units.h>
#include <FML/Spline/Spline.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace FML::COSMOLOGY {

using Spline1D = FML::INTERPOLATION::SPLINE::GSLSpline;
using Spline2D = FML::INTERPOLATION::SPLINE::GSLSpline2D;
using UniformSpline1D = FML::INTERPOLATION::SPLINE::UniformSpline;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
using ODEStepper = const gsl_odeiv2_step_type*;

// Bicubic 2D splines, or bilinear when an axis has fewer than 4 points (coarse grids, e.g. a few k-modes in a test)
inline const gsl_interp2d_type* spline2d_type(size_t nx, size_t ny) {
  return nx >= 4 && ny >= 4 ? gsl_interp2d_bicubic : gsl_interp2d_bilinear;
}

// Shared compile-time SI constants, owned by FML/Units/Units.h. Existing solver code keeps its names.
namespace Constants = FML::UTILS::SI;

// -----------------------------------------------------------------------------
// Numerical utilities
// -----------------------------------------------------------------------------
namespace Utils {

  // Path of a data file shipped with FML, rel relative to the FML folder (e.g.
  // "CosmologyFitting/external/bao/desi_dr2_mean.txt"): $FML_DIR/rel if the environment variable FML_DIR is set,
  // else <FML folder>/rel with the folder compiled in by the Makefiles (-DFML_ROOT_DIR=...), else rel itself
  // (relative to the working directory)
  std::string fml_file(const std::string& rel);

  // Grids
  std::vector<double> linspace(double xmin, double xmax, int n);
  std::vector<double> logspace(double xmin, double xmax, int n);   // log-spaced between xmin and xmax (> 0)

  // Integrate f from a to b with adaptive quadrature (GSL QAG, 21-point Gauss-Kronrod)
  double integrate(const std::function<double(double)>& f, double a, double b,
                   double epsabs = 0.0, double epsrel = 1e-8);

  // Trapezoidal rule on tabulated data
  double trapz(const std::vector<double>& x, const std::vector<double>& y);

  // Root of f in [a,b] (f(a) and f(b) must have opposite signs). Brent's method.
  double find_root(const std::function<double(double)>& f, double a, double b,
                   double epsabs = 1e-12, double epsrel = 1e-12, int max_iter = 200);

  // Spherical Bessel function j_ell(z), robust also for large ell and z << ell
  double j_ell(int ell, double z);
  // j_l(x) for all l = 0..L at once (j must hold L + 1 values): upward recurrence for x > L,
  // Miller's downward recurrence normalised with sum (2l+1) j_l^2 = 1 otherwise
  void j_ell_array(int L, double x, double* j);

  // Simple wall-clock timer
  // Timer::print and the info() messages of the solvers are silenced when false (e.g. inside an MCMC)
  inline bool verbose = true;

  class Timer {
    public:
      explicit Timer(std::string name = "") : name(std::move(name)) { start(); }
      void   start() { t0 = std::chrono::steady_clock::now(); }
      double elapsed() const {  // seconds
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      }
      void print() const;
    private:
      std::string name;
      std::chrono::steady_clock::time_point t0;
  };
}

} // namespace FML::COSMOLOGY

#endif
