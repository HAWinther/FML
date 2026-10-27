#ifndef FML_MCMC_POSTERIOR_H
#define FML_MCMC_POSTERIOR_H

// =============================================================================
// The posterior shared by the sampler (Sampler.h) and the minimizer (Minimizer.h): the sampled parameters with their
// priors, the derived parameters and the likelihood function.
//
//   * Priors: flat within [min, max] for every sampled parameter, optionally times a Gaussian.
//   * Fast/slow parameters: parameters marked fast (likelihood nuisance parameters, cheap to vary because the theory
//     is cached by the likelihood function) must come last; n_slow() is the number of slow ones (all parameters if
//     none or all are fast).
//   * evaluate(x): ln posterior = ln prior + ln L and the derived parameters; a point outside the prior, or where the
//     likelihood throws or is not finite, has ln posterior = -inf.
//   * GetDist headers: <root>.paramnames (derived names with a *) and <root>.ranges; covmat files in the GetDist
//     format ("# name1 name2 ..." + matrix).
// =============================================================================

#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace FML::MCMC {

struct SampledParameter {
  std::string name, label;         // label: LaTeX for GetDist (without $)
  double start = 0.0, min = 0.0, max = 0.0, proposal = 0.0;   // proposal: initial step size (1 sigma)
  bool gaussian_prior = false;
  double prior_mean = 0.0, prior_sigma = 0.0;
  bool fast = false;                // see the header comment (fast parameters must come last)
};

// The likelihood part of the posterior: ln L and the derived parameters for a point
struct LikelihoodResult {
  double loglike = -std::numeric_limits<double>::infinity();
  std::vector<double> derived;
};
using LikelihoodFunction = std::function<LikelihoodResult(const std::vector<double>&)>;

class Posterior {
  public:
    Posterior(std::vector<SampledParameter> params, std::vector<std::string> derived_names,
              std::vector<std::string> derived_labels, LikelihoodFunction like);

    struct Point {
      double logpost = -std::numeric_limits<double>::infinity();
      double loglike = -std::numeric_limits<double>::infinity();
      std::vector<double> derived;
      bool valid() const { return logpost > -std::numeric_limits<double>::infinity(); }
    };
    Point evaluate(const std::vector<double>& x) const;
    double log_prior(const std::vector<double>& x) const;   // -inf outside the box

    size_t dim() const { return par.size(); }
    size_t n_slow() const { return nslow; }
    const std::vector<SampledParameter>& params() const { return par; }
    const std::vector<std::string>& derived_names() const { return dnames; }
    const std::vector<std::string>& derived_labels() const { return dlabels; }
    const std::vector<std::string>& names() const { return all_names; }   // sampled then derived
    std::vector<double> start() const;
    std::vector<std::vector<double>> diagonal_covariance() const;        // proposal^2 on the diagonal
    const LikelihoodFunction& likelihood() const { return like; }

    void write_paramnames(const std::string& root) const;   // <root>.paramnames and <root>.ranges

  private:
    std::vector<SampledParameter> par;
    std::vector<std::string> dnames, dlabels, all_names;
    LikelihoodFunction like;
    size_t nslow = 0;
};

  // Cholesky factor L (lower) of a symmetric positive-definite matrix; returns false if not positive definite
  bool cholesky(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L);
  // Inverse of a symmetric positive-definite matrix (false if not positive definite)
  bool spd_inverse(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& inv);
  // GetDist .covmat for the parameters par: entries of parameters not in the file are zero except the diagonal
  // proposal^2; their names are returned in missing
  std::vector<std::vector<double>> read_covmat(const std::string& file, const std::vector<SampledParameter>& par,
                                               std::vector<std::string>& missing);
  void write_covmat(const std::string& file, const std::vector<SampledParameter>& par,
                    const std::vector<std::vector<double>>& C);

} // namespace FML::MCMC

#endif
