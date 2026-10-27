// =============================================================================
// Tests of the minimizer (Minimizer.h) and the posterior (Posterior.h). Run with `make run`.
//
//   * Correlated 4D Gaussian with a Gaussian prior: maximum of the posterior, -2 ln P, convergence flag
//   * Rosenbrock function (curved valley): minimum (1, 1)
//   * Fast/slow: inner minimization over the fast parameters, one slow evaluation per outer point
//   * Maximum outside the prior box: found on the boundary
//   * Several starts: the global one of two separated peaks
//   * Profile likelihood of a correlated Gaussian: ((v - mu)/sigma_marginal)^2
//   * Hessian: the covariance of a Gaussian (central differences are exact for a quadratic)
// =============================================================================

#include "TestHelpers.h"
#include <FML/MCMC/Minimizer.h>

#include <cmath>
#include <iostream>

using namespace FML::MCMC;

using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
  SampledParameter P(const std::string& n, double start, double lo, double hi, double prop, bool fast = false) {
    SampledParameter p;
    p.name = n;
    p.start = start;
    p.min = lo;
    p.max = hi;
    p.proposal = prop;
    p.fast = fast;
    return p;
  }
  MinimizerOptions quiet(int starts = 1) {
    MinimizerOptions o;
    o.starts = starts;
    o.verbose = false;
    o.tol = 1e-4;
    return o;
  }
  // -0.5 (x - mu)^T C^-1 (x - mu) for the 4D Gaussian below
  const std::vector<double> mu = {1.0, -2.0, 0.5, 3.0};
  const std::vector<double> sig = {0.1, 2.0, 0.3, 1.0};
  double rho(size_t i, size_t j) { return i == j ? 1.0 : (i + j == 1 ? 0.8 : (i + j == 5 ? -0.5 : 0.1)); }
  std::vector<std::vector<double>> gauss_cov() {
    std::vector<std::vector<double>> C(4, std::vector<double>(4));
    for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 4; j++) C[i][j] = rho(i, j) * sig[i] * sig[j];
    return C;
  }
  LikelihoodFunction gaussian(long* calls = nullptr) {
    std::vector<std::vector<double>> Ci;
    FML::MCMC::spd_inverse(gauss_cov(), Ci);
    return [Ci, calls](const std::vector<double>& x) {
      if (calls) (*calls)++;
      LikelihoodResult r;
      double q = 0.0;
      for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++) q += (x[i] - mu[i]) * Ci[i][j] * (x[j] - mu[j]);
      r.loglike = -0.5 * q;
      r.derived = {x[0] + x[1]};
      return r;
    };
  }
}

int main() {
  std::cout << "Posterior\n";
  {
    auto p = P("a", 0.0, -1, 1, 0.1);
    p.gaussian_prior = true;
    p.prior_mean = 0.5;
    p.prior_sigma = 0.25;
    LikelihoodFunction thrower = [](const std::vector<double>& x) -> LikelihoodResult {
      if (x[0] > 0.9) throw std::runtime_error("fails");
      LikelihoodResult r;
      r.loglike = x[0] < -0.9 ? std::nan("") : 0.0;
      return r;
    };
    Posterior post({p}, {}, {}, thrower);
    check_close("Gaussian prior", post.log_prior({0.0}), -2.0, 1e-15);
    check("outside the box: -inf", !std::isfinite(post.log_prior({1.5})));
    check("likelihood throws: invalid point", !post.evaluate({0.95}).valid());
    check("likelihood NaN: invalid point", !post.evaluate({-0.95}).valid());
    check_close("valid point: ln P = ln prior + ln L", post.evaluate({0.5}).logpost, 0.0, 1e-15);
    bool threw = false;
    try { Posterior({P("f", 0, -1, 1, 1, true), P("s", 0, -1, 1, 1)}, {}, {}, thrower); } catch (const std::exception&) { threw = true; }
    check("fast parameters must come last", threw);
  }

  std::cout << "Correlated 4D Gaussian with a Gaussian prior\n";
  {
    // the Gaussian prior N(0.7, 0.2) on x2 shifts the maximum: x = mu + C e2 (0.7 - mu2) / (C22 + s^2), the
    // posterior is again Gaussian
    auto p2 = P("c", 0.0, -10, 10, 0.5);
    p2.gaussian_prior = true;
    p2.prior_mean = 0.7;
    p2.prior_sigma = 0.2;
    Posterior post({P("a", 0.0, -10, 10, 0.5), P("b", 0.0, -20, 20, 1.0), p2, P("d", 0.0, -10, 10, 1.0)}, {"a_plus_b"},
                   {"a+b"}, gaussian());
    const auto C = gauss_cov();
    const double t = (0.7 - mu[2]) / (C[2][2] + 0.04);
    std::vector<double> xmax(4);
    for (size_t i = 0; i < 4; i++) xmax[i] = mu[i] + C[i][2] * t;
    const double m2lnP = (0.7 - mu[2]) * (0.7 - mu[2]) / (C[2][2] + 0.04);   // -2 ln P at the maximum
    Minimizer mz(post, quiet());
    const auto r = mz.run();
    std::cout << "  " << r.evals << " evaluations\n";
    double worst = 0.0;
    for (size_t i = 0; i < 4; i++) worst = std::max(worst, std::fabs(r.x[i] - xmax[i]) / sig[i]);
    check_close("maximum (|dx|/sigma)", worst, 0.0, 0.02);
    check_close("-2 ln P at the maximum", -2.0 * r.logpost, m2lnP, 2e-4);
    check_close("derived parameter at the maximum", r.derived[0], r.x[0] + r.x[1], 1e-14);
    check("converged", r.converged);
    check("< 1500 evaluations", r.evals < 1500);
  }

  std::cout << "Rosenbrock function\n";
  {
    LikelihoodFunction rb = [](const std::vector<double>& x) {
      LikelihoodResult r;
      r.loglike = -0.5 * (std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - x[0] * x[0], 2));
      return r;
    };
    Posterior post({P("x", -1.2, -5, 5, 0.5), P("y", 1.0, -5, 5, 0.5)}, {}, {}, rb);
    MinimizerOptions o = quiet();
    o.tol = 1e-8;
    const auto r = Minimizer(post, o).run();
    check_close("x = 1", r.x[0], 1.0, 2e-3);
    check_close("y = 1", r.x[1], 1.0, 4e-3);
  }

  std::cout << "Fast/slow parameters\n";
  {
    // a, b slow; c, d fast. Count the calls with a new slow point
    long calls = 0, slow = 0;
    std::vector<double> last(2, 1e300);
    auto g = gaussian();
    LikelihoodFunction like = [&](const std::vector<double>& x) {
      calls++;
      if (x[0] != last[0] || x[1] != last[1]) { slow++; last = {x[0], x[1]}; }
      return g(x);
    };
    Posterior post({P("a", 0.0, -10, 10, 0.5), P("b", 0.0, -20, 20, 1.0), P("c", 0.0, -10, 10, 0.5, true),
                    P("d", 0.0, -10, 10, 1.0, true)}, {"a_plus_b"}, {"a+b"}, like);
    MinimizerOptions o = quiet();
    o.threads = 1;
    const auto r = Minimizer(post, o).run();
    double worst = 0.0;
    for (size_t i = 0; i < 4; i++) worst = std::max(worst, std::fabs(r.x[i] - mu[i]) / sig[i]);
    std::cout << "  " << r.evals << " slow points, " << slow << " slow changes, " << calls << " calls\n";
    check_close("maximum (|dx|/sigma)", worst, 0.0, 0.03);
    check_close("-2 ln P = 0", -2.0 * r.logpost, 0.0, 2e-3);
    check("one slow change per slow point (inner minimization keeps it)", slow <= r.evals + 1);
    check("fast parameters minimized inside (calls >> slow points)", calls > 5 * r.evals);
  }

  std::cout << "Maximum outside the prior\n";
  {
    LikelihoodFunction lin = [](const std::vector<double>& x) {
      LikelihoodResult r;
      r.loglike = -0.5 * (std::pow((x[0] - 2.0) / 0.5, 2) + std::pow(x[1] - 0.3, 2));
      return r;
    };
    Posterior post({P("x", 0.0, -1, 1, 0.3), P("y", 0.0, -1, 1, 0.3)}, {}, {}, lin);
    MinimizerOptions o = quiet();
    o.tol = 1e-7;
    const auto r = Minimizer(post, o).run();
    check_close("on the boundary x = 1", r.x[0], 1.0, 1e-3);
    check_close("y = 0.3", r.x[1], 0.3, 1e-2);
  }

  std::cout << "Several starts: two peaks\n";
  {
    LikelihoodFunction two = [](const std::vector<double>& x) {
      LikelihoodResult r;
      const double a = std::exp(-0.5 * std::pow((x[0] + 3.0) / 0.5, 2)), b = 2.0 * std::exp(-0.5 * std::pow((x[0] - 3.0) / 0.5, 2));
      r.loglike = std::log(a + b + 1e-300);
      return r;
    };
    Posterior post({P("x", -3.0, -10, 10, 0.5)}, {}, {}, two);
    MinimizerOptions o = quiet(8);
    o.start_scatter = 12.0;   // starts spread over the prior
    Minimizer mz(post, o);
    const auto r = mz.run();
    check_close("first start finds the local peak", mz.start_results()[0].x[0], -3.0, 1e-2);
    check_close("best of the starts is the global peak", r.x[0], 3.0, 1e-2);
  }

  std::cout << "Profile likelihood\n";
  {
    Posterior post({P("a", 0.0, -10, 10, 0.5), P("b", 0.0, -20, 20, 1.0), P("c", 0.0, -10, 10, 0.5),
                    P("d", 0.0, -10, 10, 1.0)}, {"a_plus_b"}, {"a+b"}, gaussian());
    Minimizer mz(post, quiet());
    const auto best = mz.run();
    const std::vector<double> vals = {mu[1] - 3.0, mu[1] - 1.0, mu[1], mu[1] + 2.0};
    const auto prof = mz.profile(1, vals, best.x);
    double worst = 0.0;
    for (size_t k = 0; k < vals.size(); k++) {
      const double expect = std::pow((vals[k] - mu[1]) / sig[1], 2);   // profile of a Gaussian: marginal sigma
      worst = std::max(worst, std::fabs(-2.0 * prof[k].logpost - expect));
      worst = std::max(worst, std::fabs(prof[k].x[1] - vals[k]));
    }
    check_close("-2 ln P(b) = ((b - mu)/sigma_b)^2, b fixed", worst, 0.0, 1e-3);
  }

  std::cout << "Hessian\n";
  {
    Posterior post({P("a", 0.0, -10, 10, 0.05), P("b", 0.0, -20, 20, 0.5), P("c", 0.0, -10, 10, 0.1),
                    P("d", 0.0, -10, 10, 0.5)}, {"a_plus_b"}, {"a+b"}, gaussian());
    Minimizer mz(post, quiet());
    std::vector<double> h = {0.05, 0.5, 0.1, 0.5};
    std::vector<std::vector<double>> C;
    const bool ok = Minimizer::covariance(mz.hessian({0.9, -1.0, 0.7, 2.5}, h, false), h, C);
    const auto C0 = gauss_cov();
    double worst = 0.0;
    for (size_t i = 0; i < 4; i++)
      for (size_t j = 0; j < 4; j++) worst = std::max(worst, std::fabs(C[i][j] - C0[i][j]) / (sig[i] * sig[j]));
    check("positive definite", ok);
    check_close("covariance = C (relative to sigma_i sigma_j)", worst, 0.0, 1e-8);
    // adapted steps: 1.5 conditional sigma (1/sqrt(H_ii)) from steps 20x too large, same covariance
    std::vector<double> h2 = {1.0, 10.0, 2.0, 9.0};
    std::vector<std::vector<double>> Ci, C2;
    FML::MCMC::spd_inverse(C0, Ci);
    Minimizer::covariance(mz.hessian({0.9, -1.0, 0.7, 2.5}, h2, true), h2, C2);
    double ws = 0.0, wc = 0.0;
    for (size_t i = 0; i < 4; i++) {
      ws = std::max(ws, std::fabs(h2[i] * std::sqrt(Ci[i][i]) / 1.5 - 1.0));
      for (size_t j = 0; j < 4; j++) wc = std::max(wc, std::fabs(C2[i][j] - C0[i][j]) / (sig[i] * sig[j]));
    }
    check_close("adapted steps = 1.5 / sqrt(H_ii)", ws, 0.0, 0.4);
    check_close("... covariance = C", wc, 0.0, 1e-8);
  }

  return Test::summary();
}
