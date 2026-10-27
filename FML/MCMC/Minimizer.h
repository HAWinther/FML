#ifndef FML_MCMC_MINIMIZER_H
#define FML_MCMC_MINIMIZER_H

// =============================================================================
// Maximum of the posterior (best fit), profile likelihoods and the Hessian, for a Posterior (Posterior.h).
//
//   * Method: Nelder-Mead simplex (adaptive coefficients of Gao & Han 2012) in whitened coordinates u, x = x0 + L u
//     with L the Cholesky factor of a covariance (the proposal covmat or the diagonal proposal widths), so the
//     initial simplex has the size of the posterior in every direction. Derivative-free and robust against the
//     small numerical noise of the Boltzmann code. After the simplex has converged (spread of -2 ln P over the
//     vertices < tol) it is restarted around the best point with half the size, until a restart improves -2 ln P by
//     less than tol (at most max_restarts).
//   * Fast/slow parameters: the simplex runs over the free slow parameters only; for each slow point the fast ones
//     are minimized by an inner simplex (warm-started from the previous optimum, tolerance tol/10). As the
//     likelihood function caches the theory for the slow parameters, a slow point costs one theory evaluation.
//   * Several starts (starts): the first from the start values, the others drawn from a Gaussian around them with
//     the covariance (times start_scatter^2) within the prior. The starts are spread over MPI processes and OpenMP
//     threads (threads, as for the MCMC chains); their spread in -2 ln P shows whether the minimum is global.
//   * Profiles: profile(i, values, from) minimizes over all parameters but i at each fixed value of parameter i,
//     each from the point from (e.g. the best fit); the points are spread over the processes and threads.
//   * Hessian: hessian() computes the Hessian of -ln P by central differences (2 d^2 + 1 evaluations over the MPI
//     processes); covariance() inverts it (falls back to the diagonal 1/H_ii if it is not positive definite).
//     With adapt_steps the steps are first rescaled (2d evaluations per pass, at most 4 passes) to 1.5 times the
//     conditional width 1/sqrt(H_ii) in each direction: much larger steps (e.g. proposal widths along a
//     degeneracy) pick up the non-Gaussian shape and can give a matrix that is not positive definite, much smaller
//     ones the numerical noise. The steps used are returned in h.
//   * The objective is -2 ln posterior: the Gaussian priors (gauss_*) count as data, the flat priors only bound
//     the parameters. Points outside the prior or where the likelihood fails count as +infinity.
// =============================================================================

#include <FML/MCMC/Posterior.h>

#include <cstdint>
#include <string>
#include <vector>

namespace FML::MCMC {

struct MinimizerOptions {
  int starts = 4;               // independent minimizations
  int threads = 0;              // OpenMP threads over the starts (0: one per start, bounded by OpenMP)
  int max_evals = 3000;         // slow evaluations per minimization
  int max_restarts = 6;
  double tol = 0.01;            // convergence in -2 ln P
  double start_scatter = 1.0;   // starts 2..: start + start_scatter * N(0, covariance)
  std::uint64_t seed = 1;
  std::vector<std::vector<double>> covmat;   // d x d (empty: diagonal proposal^2)
  bool verbose = true;
  int progress_every = 0;       // > 0: print the best -2 ln P every this many slow evaluations (process 0)
};

struct MinimizerResult {
  std::vector<double> x, derived;
  double logpost = -1e300, loglike = -1e300;
  long evals = 0;                // slow evaluations (theory computations)
  bool converged = false;        // the last restart improved -2 ln P by < tol
  double chi2() const { return -2.0 * loglike; }
};

class Minimizer {
  public:
    Minimizer(const Posterior& posterior, MinimizerOptions options = MinimizerOptions());

    // All starts; returns the best and keeps every start's result
    MinimizerResult run();
    const std::vector<MinimizerResult>& start_results() const { return all; }

    // One minimization from x0 with the parameters marked in fixed kept at their values in x0
    MinimizerResult minimize(const std::vector<double>& x0, const std::vector<bool>& fixed = {}) const;

    // Profile in parameter i: minimum over the others at each value, starting from the point from
    std::vector<MinimizerResult> profile(size_t i, const std::vector<double>& values, const std::vector<double>& from) const;

    // Hessian of -ln P at x with steps h (all parameters; MPI-parallel)
    std::vector<std::vector<double>> hessian(const std::vector<double>& x, std::vector<double>& h,
                                             bool adapt_steps = true) const;
    // Its inverse; false (and the diagonal 1/H_ii, or h^2 where H_ii <= 0) if not positive definite
    static bool covariance(const std::vector<std::vector<double>>& H, const std::vector<double>& h,
                           std::vector<std::vector<double>>& C);

    // <root>.minimum.txt: -2 ln L, -2 ln P and the parameters of r (and the other starts, if any);
    // <root>.paramnames and .ranges
    void write(const std::string& root, const MinimizerResult& r) const;
    void write_profile(const std::string& file, size_t i, const std::vector<MinimizerResult>& prof) const;

  private:
    const Posterior& post;
    MinimizerOptions opt;
    std::vector<std::vector<double>> cov;
    std::vector<MinimizerResult> all;

    // results of n independent tasks, spread over the MPI processes and OpenMP threads, gathered on all processes
    template <class F> std::vector<MinimizerResult> parallel_tasks(int n, F task) const;
    std::vector<double> minus_logpost(const std::vector<std::vector<double>>& pts) const;   // over MPI processes
};

} // namespace FML::MCMC

#endif
