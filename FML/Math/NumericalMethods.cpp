#include <FML/Math/NumericalMethods.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#ifdef USE_GSL
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_bessel.h>
#endif

namespace FML { namespace MATH {
#ifdef USE_GSL
namespace {
  // GSL's default error handler calls abort(). We switch it off once at startup
  // and instead check the returned status codes ourselves (throwing on errors).
  struct GSLErrorHandlerOff {
    GSLErrorHandlerOff() { gsl_set_error_handler_off(); }
  } gsl_error_handler_off_instance;

  // Trampoline from std::function to GSL's C function pointer interface
  double gsl_function_wrapper(double x, void* params) {
    return (*static_cast<const std::function<double(double)>*>(params))(x);
  }
}

  double integrate(const std::function<double(double)>& f, double a, double b,
                   double epsabs, double epsrel) {
    const size_t limit = 1000;
    std::unique_ptr<gsl_integration_workspace, decltype(&gsl_integration_workspace_free)>
      w(gsl_integration_workspace_alloc(limit), gsl_integration_workspace_free);
    gsl_function F;
    F.function = &gsl_function_wrapper;
    F.params   = const_cast<void*>(static_cast<const void*>(&f));
    double result, error;
    int status = gsl_integration_qag(&F, a, b, epsabs, epsrel, limit, GSL_INTEG_GAUSS21,
                                     w.get(), &result, &error);
    if (status != GSL_SUCCESS)
      std::cerr << "Utils::integrate: warning: " << gsl_strerror(status)
                << " (estimated error " << error << ")\n";
    return result;
  }

  double find_root(const std::function<double(double)>& f, double a, double b,
                   double epsabs, double epsrel, int max_iter) {
    const double fa = f(a), fb = f(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa * fb > 0.0)
      throw std::runtime_error("find_root: root not bracketed (f(a)=" + std::to_string(fa) +
                               ", f(b)=" + std::to_string(fb) + ")");

    std::unique_ptr<gsl_root_fsolver, decltype(&gsl_root_fsolver_free)>
      solver(gsl_root_fsolver_alloc(gsl_root_fsolver_brent), gsl_root_fsolver_free);
    gsl_function F;
    F.function = &gsl_function_wrapper;
    F.params   = const_cast<void*>(static_cast<const void*>(&f));
    gsl_root_fsolver_set(solver.get(), &F, std::min(a, b), std::max(a, b));

    double root = 0.5 * (a + b);
    for (int iter = 0; iter < max_iter; iter++) {
      gsl_root_fsolver_iterate(solver.get());
      root = gsl_root_fsolver_root(solver.get());
      const double lo = gsl_root_fsolver_x_lower(solver.get());
      const double hi = gsl_root_fsolver_x_upper(solver.get());
      if (gsl_root_test_interval(lo, hi, epsabs, epsrel) == GSL_SUCCESS) return root;
    }
    std::cerr << "Utils::find_root: warning: max iterations reached\n";
    return root;
  }

  double spherical_bessel_j(int ell, double z) {
    if (ell < 0) throw std::invalid_argument("j_ell: ell must be >= 0");
    if (z == 0.0) return ell == 0 ? 1.0 : 0.0;
    if (z < 0.0)  return (ell % 2 == 0 ? 1.0 : -1.0) * spherical_bessel_j(ell, -z);

    // Low ell: GSL's direct routine is accurate and fast
    if (ell < 50) {
      gsl_sf_result r;
      if (gsl_sf_bessel_jl_e(ell, z, &r) == GSL_SUCCESS) return r.val;
    }

    // Below the turning point j_ell decays like an Airy function: at z = nu - t nu^{1/3}
    // the value relative to the first peak is ~ Ai(2^{1/3} t)/Ai_max, i.e. ~1e-13 for t = 10.
    // Return 0 there instead of fighting underflow (and to save time).
    const double nu = ell + 0.5;
    if (z < nu - 10.0 * std::cbrt(nu)) return 0.0;

    // General case: j_ell(z) = sqrt(pi/(2z)) J_{ell+1/2}(z), which is robust for large ell
    gsl_sf_result r;
    const int status = gsl_sf_bessel_Jnu_e(nu, z, &r);
    if (status == GSL_EUNDRFLW) return 0.0;
    if (status != GSL_SUCCESS)
      throw std::runtime_error("j_ell: GSL error " + std::string(gsl_strerror(status)) +
                               " for ell=" + std::to_string(ell) + " z=" + std::to_string(z));
    return std::sqrt(M_PI / (2.0 * z)) * r.val;
  }


#endif
  double trapz(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) throw std::invalid_argument("trapz: x and y differ in size");
    double sum = 0.0;
    for (size_t i = 1; i < x.size(); i++) sum += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
    return sum;
  }

  // j_l(x) for l = 0..L: upward recurrence for x > L (stable there), otherwise Miller's downward
  // recurrence normalised with sum_l (2l+1) j_l^2 = 1 (GSL's Steed routine underflows to NaN for x << L)
  void spherical_bessel_j_array(int L, double x, double* j) {
    if (x < 1e-10) {
      j[0] = 1.0;
      for (int l = 1; l <= L; l++) j[l] = 0.0;
      return;
    }
    const double sx = std::sin(x), cx = std::cos(x);
    if (x > L) {
      j[0] = sx / x;
      if (L >= 1) j[1] = sx / (x * x) - cx / x;
      for (int l = 1; l < L; l++) j[l + 1] = (2.0 * l + 1.0) / x * j[l] - j[l - 1];
      return;
    }
    const int N = L + 30 + static_cast<int>(std::sqrt(40.0 * (L + x)));
    std::vector<double> f(N + 2, 0.0);
    f[N + 1] = 0.0;
    f[N] = 1e-300;
    for (int l = N; l >= 1; l--) {
      f[l - 1] = (2.0 * l + 1.0) / x * f[l] - f[l + 1];
      if (std::fabs(f[l - 1]) > 1e100) for (int m = l - 1; m <= N + 1; m++) f[m] *= 1e-100;
    }
    double fmax = 0.0;
    for (int l = 0; l <= N; l++) fmax = std::max(fmax, std::fabs(f[l]));
    for (int l = 0; l <= N; l++) f[l] /= fmax;
    double norm = 0.0;
    for (int l = 0; l <= N; l++) norm += (2.0 * l + 1.0) * f[l] * f[l];
    double scale = 1.0 / std::sqrt(norm);
    const double j0 = sx / x, j1 = sx / (x * x) - cx / x;
    if ((std::fabs(j0) > std::fabs(j1) ? j0 * f[0] : j1 * f[1]) < 0.0) scale = -scale;
    for (int l = 0; l <= L; l++) j[l] = f[l] * scale;
  }
} }
