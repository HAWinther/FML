// =============================================================================
// Unit tests for the step-0 infrastructure: Utils, Spline1D/2D, ODESolver,
// CosmologyParameters. Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Common.h>
#include "TestHelpers.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace FML::COSMOLOGY;

using Test::check;
using Test::check_close;

// Independent reference for j_ell: Miller's downward recurrence normalised to j_0,
// in long double. Accurate for all z > 0.
static long double j_ell_reference(int ell, long double z) {
  if (z > ell + 1.0L) {  // upward recurrence is stable for z > ell
    long double jm = std::sin(z) / z, j = std::sin(z) / (z * z) - std::cos(z) / z;
    if (ell == 0) return jm;
    for (int l = 1; l < ell; l++) {
      const long double jp = (2 * l + 1) / z * j - jm;
      jm = j; j = jp;
    }
    return j;
  }
  const int lstart = ell + 100 + static_cast<int>(2.0L * std::sqrt(static_cast<long double>(ell) * 40.0L));
  long double jp = 0.0L, j = 1e-300L, result = 0.0L;
  for (int l = lstart; l > 0; l--) {
    const long double jm = (2 * l + 1) / z * j - jp;
    jp = j; j = jm;
    if (l - 1 == ell) result = j;
    if (std::fabs(j) > 1e250L) { j *= 1e-250L; jp *= 1e-250L; result *= 1e-250L; }
  }
  return result * (std::sin(z) / z) / j;
}

// -----------------------------------------------------------------------------

static void test_grids_and_quadrature() {
  std::cout << "Grids, quadrature, root finding\n";
  auto x = Utils::linspace(-3.0, 2.0, 11);
  check("linspace endpoints and spacing", x.front() == -3.0 && x.back() == 2.0 && std::fabs(x[1] - x[0] - 0.5) < 1e-14);
  auto k = Utils::logspace(1e-5, 1.0, 6);
  check_close("logspace middle point", k[1], 1e-4, 1e-12);

  check_close("integrate sin on [0,pi]", Utils::integrate([](double t) { return std::sin(t); }, 0.0, M_PI), 2.0, 1e-12);
  auto xs = Utils::linspace(0.0, M_PI, 2001);
  std::vector<double> ys(xs.size());
  for (size_t i = 0; i < xs.size(); i++) ys[i] = std::sin(xs[i]);
  check_close("trapz sin on [0,pi]", Utils::trapz(xs, ys), 2.0, 1e-6);
  check_close("find_root cos(x) = x",
              Utils::find_root([](double t) { return std::cos(t) - t; }, 0.0, 1.0), 0.739085133215160642, 1e-12);
}

static void test_spline1d() {
  std::cout << "Spline1D\n";
  auto x = Utils::linspace(0.0, 2.0 * M_PI, 400);
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size(); i++) y[i] = std::sin(x[i]);
  Spline1D f(x, y, "sin");

  check_close("value",      f(1.234),         std::sin(1.234),  1e-8);
  check_close("deriv_x",    f.deriv_x(1.234), std::cos(1.234),  1e-5);
  check_close("deriv_xx",   f.deriv_xx(1.234), -std::sin(1.234), 1e-3);
  check_close("integral",   f.integral(0.0, M_PI), 2.0, 1e-8);

  // Decreasing x-array (e.g. integrating tau backwards in time)
  std::vector<double> xr(x.rbegin(), x.rend()), yr(y.rbegin(), y.rend());
  Spline1D fr(xr, yr, "sin reversed");
  check_close("decreasing x input", fr(2.5), std::sin(2.5), 1e-8);

  // Copies share the data and remain valid after the original is gone
  Spline1D g;
  { Spline1D tmp(x, y, "tmp"); g = tmp; }
  check_close("copy outlives original", g(0.5), std::sin(0.5), 1e-8);

  // Out of range: clamped (and one warning on stderr)
  Spline1D h(x, y, "clamp test");
  h.set_out_of_bounds_warning(false);
  check_close("out-of-range clamps to endpoint", h(10.0), y.back(), 1e-14);

  // Thread-safe evaluation
  const int n = 100000;
  double sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < n; i++) sum += f(2.0 * M_PI * i / (n - 1.0));
  check_close("parallel evaluation (sum of sin ~ 0)", sum / n, 0.0, 1e-6);
}

static void test_uniform_spline() {
  std::cout << "UniformSpline1D\n";
  const double dx = 2.0 * M_PI / 32.0;
  const int n = 2000;
  std::vector<double> y(n);
  for (int i = 0; i < n; i++) y[i] = Utils::j_ell(100, i * dx);
  UniformSpline1D f(0.0, dx, y);
  double max_err = 0.0, jmax = 0.0;
  for (double z = 0.0; z < (n - 1) * dx; z += 0.0123) {
    const double ref = Utils::j_ell(100, z);
    max_err = std::max(max_err, std::fabs(f(z) - ref));
    jmax = std::max(jmax, std::fabs(ref));
  }
  check_close("j_100 table (32 points per 2 pi) vs direct", max_err / jmax, 0.0, 1e-5);
  check_close("value at a node is exact", f(37 * dx), y[37], 1e-15);
  check("out of range returns end-point values", f(-1.0) == y.front() && f(1e6) == y.back());
}

static void test_spline2d() {
  std::cout << "Spline2D\n";
  auto x = Utils::linspace(0.0, 3.0, 200);
  auto y = Utils::linspace(-1.0, 2.0, 150);
  std::vector<std::vector<double>> z(x.size(), std::vector<double>(y.size()));
  for (size_t i = 0; i < x.size(); i++)
    for (size_t j = 0; j < y.size(); j++) z[i][j] = std::sin(x[i]) * std::cos(y[j]);
  Spline2D f(x, y, z, "sin*cos");

  const double x0 = 1.3, y0 = 0.4;
  check_close("value",    f(x0, y0),          std::sin(x0) * std::cos(y0),  1e-7);
  check_close("deriv_x",  f.deriv_x(x0, y0),  std::cos(x0) * std::cos(y0),  1e-5);
  check_close("deriv_y",  f.deriv_y(x0, y0), -std::sin(x0) * std::sin(y0),  1e-5);
  check_close("deriv_xx", f.deriv_xx(x0, y0), -std::sin(x0) * std::cos(y0), 1e-3);
  check_close("deriv_xy", f.deriv_xy(x0, y0), -std::cos(x0) * std::sin(y0), 1e-3);

  // Decreasing y-array
  std::vector<double> yr(y.rbegin(), y.rend());
  std::vector<std::vector<double>> zr(x.size(), std::vector<double>(y.size()));
  for (size_t i = 0; i < x.size(); i++)
    for (size_t j = 0; j < y.size(); j++) zr[i][j] = z[i][y.size() - 1 - j];
  Spline2D fr(x, yr, zr, "reversed y");
  check_close("decreasing y input", fr(x0, y0), std::sin(x0) * std::cos(y0), 1e-7);
}

static void test_ode() {
  std::cout << "ODESolver\n";

  // Harmonic oscillator y'' = -y, y(0)=0, y'(0)=1  ->  y = sin(x)
  ODEFunction sho = [](double, const double* y, double* dydx) {
    dydx[0] = y[1];
    dydx[1] = -y[0];
    return GSL_SUCCESS;
  };
  ODESolver ode(1e-3, 1e-10, 1e-10);
  auto x = Utils::linspace(0.0, 10.0, 101);
  ode.solve(sho, x, {0.0, 1.0}, gsl_odeiv2_step_rk8pd);
  check_close("harmonic oscillator y(10)",  ode.get_final_data_by_component(0), std::sin(10.0), 1e-8);
  check_close("stored derivative y'(10)",   ode.get_derivative_data_by_component(0).back(), std::cos(10.0), 1e-8);

  // Backwards integration: y' = y from x=0 (y=1) to x=-5  ->  y = exp(x)
  ODEFunction growth = [](double, const double* y, double* dydx) { dydx[0] = y[0]; return GSL_SUCCESS; };
  ode.solve(growth, Utils::linspace(0.0, -5.0, 50), {1.0}, gsl_odeiv2_step_rk8pd);
  check_close("backward integration exp(-5)", ode.get_final_data_by_component(0), std::exp(-5.0), 1e-9);

  // Stiff problem: y' = -lambda (y - cos x) - sin x, y(0) = 1  ->  y = cos(x)
  const double lambda = 1e5;
  ODEFunction stiff = [lambda](double t, const double* y, double* dydx) {
    dydx[0] = -lambda * (y[0] - std::cos(t)) - std::sin(t);
    return GSL_SUCCESS;
  };
  for (auto [m, name] : {std::pair{gsl_odeiv2_step_msbdf, "MSBDF"},
                         std::pair{gsl_odeiv2_step_rk4imp, "RK4IMP"}}) {
    ODESolver s(1e-6, 1e-10, 1e-10);
    s.set_finite_difference_jacobian(true);
    s.solve(stiff, Utils::linspace(0.0, 3.0, 31), {1.0}, m);
    check_close(std::string("stiff problem, ") + name + " (FD Jacobian)",
                s.get_final_data_by_component(0), std::cos(3.0), 1e-7);
  }

  // Error reporting
  ODEFunction failing = [](double t, const double*, double* dydx) {
    dydx[0] = 1.0;
    return t > 0.5 ? GSL_EBADFUNC : GSL_SUCCESS;
  };
  bool threw = false;
  try { ode.solve(failing, {0.0, 1.0}, {0.0}, gsl_odeiv2_step_rk8pd); } catch (const std::runtime_error&) { threw = true; }
  check("failing rhs throws", threw);
}

static void test_bessel() {
  std::cout << "Spherical Bessel functions\n";
  check_close("j_0(2.5)", Utils::j_ell(0, 2.5), std::sin(2.5) / 2.5, 1e-14);
  check_close("j_1(2.5)", Utils::j_ell(1, 2.5), std::sin(2.5) / 6.25 - std::cos(2.5) / 2.5, 1e-14);
  check("j_ell(ell>0, 0) = 0 and j_0(0) = 1", Utils::j_ell(5, 0.0) == 0.0 && Utils::j_ell(0, 0.0) == 1.0);

  // Compare against independent recurrence over a wide range of (ell, z),
  // relative to the maximum of |j_ell| (the quantity that matters for integrals)
  double max_err = 0.0;
  int worst_ell = 0; double worst_z = 0.0;
  for (int ell : {2, 10, 49, 50, 51, 100, 500, 1000, 2000, 3000}) {
    double jmax = 0.0;
    std::vector<double> zs = Utils::linspace(0.01, 2.0 * ell + 500.0, 4000);
    std::vector<double> ref(zs.size());
    for (size_t i = 0; i < zs.size(); i++) {
      ref[i] = static_cast<double>(j_ell_reference(ell, zs[i]));
      jmax = std::max(jmax, std::fabs(ref[i]));
    }
    for (size_t i = 0; i < zs.size(); i++) {
      const double err = std::fabs(Utils::j_ell(ell, zs[i]) - ref[i]) / jmax;
      if (err > max_err) { max_err = err; worst_ell = ell; worst_z = zs[i]; }
    }
  }
  std::cout << "    (worst at ell = " << worst_ell << ", z = " << worst_z << ")\n";
  check_close("j_ell vs recurrence, ell in [2,3000]", max_err, 0.0, 1e-9);

  Utils::Timer t;
  double s = 0.0;
  const int n = 200000;
  for (int i = 0; i < n; i++) s += Utils::j_ell(1500, 1000.0 + 2000.0 * i / n);
  std::cout << "    " << n << " evaluations of j_1500(z): " << t.elapsed() << " s (checksum " << s << ")\n";
}

static void test_parameters() {
  std::cout << "CosmologyParameters\n";
  CosmologyParameters p;
  p.validate();
  // Omega_gamma h^2 = 2.4728e-5 (T/2.7255 K)^4
  check_close("Omega_gamma h^2", p.OmegaGamma() * p.h * p.h, 2.4728e-5, 1e-3);
  check_close("Omega_nu/Omega_gamma (Neff=3.046)", p.OmegaNu() / p.OmegaGamma(), 3.046 * 7.0 / 8.0 * std::pow(4.0 / 11.0, 4.0 / 3.0), 1e-14);
  check_close("closure relation", p.OmegaB + p.OmegaCDM + p.OmegaR() + p.OmegaK + p.OmegaLambda(), 1.0, 1e-14);
  check_close("H0 [km/s/Mpc]", p.H0() * Constants::Mpc / Constants::km, 67.0, 1e-12);

  bool threw = false;
  p.h = -1.0;
  try { p.validate(); } catch (const std::invalid_argument&) { threw = true; }
  check("invalid parameters rejected", threw);
}

int main() {
  test_grids_and_quadrature();
  test_spline1d();
  test_uniform_spline();
  test_spline2d();
  test_ode();
  test_bessel();
  test_parameters();
  return Test::summary();
}
