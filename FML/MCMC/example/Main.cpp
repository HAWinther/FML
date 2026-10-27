// Example of the samplers in FML/MCMC: a correlated 2D Gaussian (with a derived parameter), sampled with the
// adaptive Metropolis-Hastings sampler and maximized with the minimizer (best fit, Hessian, profile)
#include <FML/MCMC/Minimizer.h>
#include <FML/MCMC/Sampler.h>

#include <cmath>
#include <iostream>

using namespace FML::MCMC;

int main() {
  // ln L of a Gaussian with means (0, 1), sigmas (1, 2) and correlation 0.5
  const double rho = 0.5;
  LikelihoodFunction like = [=](const std::vector<double>& x) {
    const double u = x[0], v = (x[1] - 1.0) / 2.0;
    LikelihoodResult r;
    r.loglike = -0.5 * (u * u - 2.0 * rho * u * v + v * v) / (1.0 - rho * rho);
    r.derived = {x[0] + x[1]};
    return r;
  };
  SampledParameter a{"a", "a"}, b{"b", "b"};
  a.start = 0.5; a.min = -10; a.max = 10; a.proposal = 1.0;
  b.start = 0.0; b.min = -10; b.max = 10; b.proposal = 2.0;
  const Posterior post({a, b}, {"a_plus_b"}, {"a+b"}, like);

  SamplerOptions opt;
  opt.chains = 4;
  opt.burn_in = 1000;
  opt.min_samples = 5000;
  opt.check_every = 1000;
  opt.output_root = "chains/gaussian";   // GetDist chains
  Sampler mc(post, opt);
  mc.run();
  const auto m = mc.mean(), s = mc.stddev();
  for (size_t i = 0; i < mc.names().size(); i++)
    std::cout << mc.names()[i] << " = " << m[i] << " +- " << s[i] << "\n";
  std::cout << "R-1 = " << mc.Rminus1() << ", acceptance " << mc.acceptance() << "\n";

  MinimizerOptions mopt;
  mopt.verbose = false;
  Minimizer mz(post, mopt);
  const MinimizerResult best = mz.run();
  std::cout << "best fit a = " << best.x[0] << ", b = " << best.x[1] << ", -2 ln L = " << best.chi2() << "\n";
  std::vector<double> h = {1.0, 2.0};
  std::vector<std::vector<double>> C;
  Minimizer::covariance(mz.hessian(best.x, h), h, C);
  std::cout << "Hessian: sigma_a = " << std::sqrt(C[0][0]) << ", sigma_b = " << std::sqrt(C[1][1])
            << ", correlation " << C[0][1] / std::sqrt(C[0][0] * C[1][1]) << "\n";
  const auto prof = mz.profile(1, {-1.0, 1.0, 3.0}, best.x);
  for (const auto& r : prof) std::cout << "profile: b = " << r.x[1] << ", -2 ln P = " << -2.0 * r.logpost << "\n";
}
