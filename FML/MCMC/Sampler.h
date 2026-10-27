#ifndef FML_MCMC_SAMPLER_H
#define FML_MCMC_SAMPLER_H

// =============================================================================
// Adaptive Metropolis-Hastings sampler with several chains and a Gelman-Rubin convergence test.
//
//   * Posterior (priors, likelihood, derived parameters, fast/slow split): Posterior.h.
//   * Proposal: multivariate Gaussian. During the burn-in it adapts (Haario, Saksman & Tamminen 2001): every
//     adapt_every steps the covariance is re-estimated from the chain so far, scaled by (2.38^2/d) s, and the
//     scale s is tuned towards an acceptance rate of 0.25 (Roberts & Rosenthal). At the end of the burn-in the
//     covariance of all chains together is frozen, so the sampling phase is an exact Markov chain.
//   * Initial proposal: diagonal with the given step sizes, or options.initial_covmat (e.g. the .covmat of an
//     earlier run, or one from a Fisher/Hessian estimate), which makes the burn-in much shorter.
//   * MPI (make USE_MPI=true, mpirun -np N): the chains are distributed over the processes (chain i on process
//     i mod N, N <= chains); the burn-in covariance, R-1 and the results are combined over all processes,
//     and after run() every process holds all chains. Only process 0 writes files and prints.
//   * The chains of a process advance in rounds of check_every steps, in parallel (OpenMP). After each round of the
//     sampling phase the Gelman-Rubin statistic R-1 (largest over the parameters) is computed from the
//     post-burn-in samples; the run stops when R-1 < Rminus1_stop and every chain has min_samples samples,
//     or after max_steps steps per chain.
//   * Fast/slow parameters (Lewis 2013, as CosmoMC): parameters marked fast (likelihood nuisance parameters, cheap
//     to vary because the theory is cached) come last. Each step is one update of all parameters followed by
//     oversample_fast updates of the fast ones only, with the proposal L z restricted to z_slow = 0 (L is the
//     lower-triangular Cholesky factor of the proposal covariance, so these move only the fast parameters with
//     their conditional covariance given the slow ones); the fast updates have their own scale, tuned to 25%
//     acceptance in the burn-in. One row per step is stored.
//   * Checkpoints (output_root set): after every round each chain writes <root>_<i>.txt (its samples so far) and
//     <root>_<i>.checkpoint (its full state: position, proposal, scales, counters, burn-in history, random
//     generator); with resume = true a run continues from these (any number of MPI processes).
//   * A point where the posterior cannot be evaluated (exception, NaN, outside the prior) is rejected.
//   * Output (GetDist format): <root>_<i>.txt with "weight -ln(posterior) params derived" per distinct sample
//     of chain i after the burn-in, <root>.paramnames, <root>.ranges, <root>.covmat (frozen proposal) and a
//     summary <root>.summary.txt (means, standard deviations, 68% limits, best fit, acceptance, R-1).
// =============================================================================

#include <FML/MCMC/Posterior.h>

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace FML::MCMC {

struct SamplerOptions {
  int chains = 4;
  int threads = 0;              // 0: one per chain (bounded by OpenMP's maximum)
  int burn_in = 2000;           // steps per chain with adaptation, discarded
  int adapt_every = 100;
  int check_every = 1000;       // steps per chain between convergence checks
  int min_samples = 4000;       // post-burn-in steps per chain before the run may stop
  int max_steps = 100000;       // total steps per chain (including the burn-in)
  double Rminus1_stop = 0.01;
  std::uint64_t seed = 1;
  std::string output_root;      // empty: no files
  bool verbose = true;
  std::vector<std::vector<double>> initial_covmat;   // d x d proposal covariance at the start (empty: diagonal)
  int oversample_fast = 4;      // fast-only updates per step (if there are fast parameters)
  bool checkpoint = true;       // write chains and checkpoints after every round (needs output_root)
  bool resume = false;          // continue from <output_root>_<i>.checkpoint if present for every chain
};

class Sampler {
  public:
    explicit Sampler(Posterior posterior, SamplerOptions options = SamplerOptions());
    Sampler(std::vector<SampledParameter> params, std::vector<std::string> derived_names,
         std::vector<std::string> derived_labels, LikelihoodFunction like, SamplerOptions options = SamplerOptions())
      : Sampler(Posterior(std::move(params), std::move(derived_names), std::move(derived_labels), std::move(like)),
             options) {}
    Sampler(const Sampler&) = delete;              // par refers to post
    Sampler& operator=(const Sampler&) = delete;
    void run();

    // Results (post-burn-in samples of all chains, equal weights per step)
    std::vector<double> mean() const;          // sampled then derived
    std::vector<double> stddev() const;
    std::vector<std::vector<double>> covariance() const;   // sampled parameters only
    double Rminus1() const { return R1; }
    double acceptance() const;
    std::vector<double> best_fit() const { return best; }  // sampled then derived
    double best_loglike() const { return best_ll; }
    double best_logpost() const { return best_lp; }
    long n_samples() const;                    // post-burn-in steps, all chains
    const std::vector<std::string>& names() const { return post.names(); }
    // Equal-weight samples (rows: sampled then derived) of chain i
    const std::vector<std::vector<double>>& samples(int chain) const { return chains[chain].rows; }
    double log_prior(const std::vector<double>& x) const { return post.log_prior(x); }   // -inf outside the box
    const Posterior& posterior() const { return post; }

  private:
    struct Chain {
      int index = 0;                              // global chain number (0 .. chains-1)
      std::vector<double> x, derived;
      double lp = 0.0, ll = 0.0;
      std::vector<std::vector<double>> history;   // all states (sampled params), for adaptation
      std::vector<std::vector<double>> rows;      // post-burn-in states incl. derived
      std::vector<double> rows_lp;
      long steps = 0, accepted = 0, accepted_sampling = 0, steps_sampling = 0, accepted_window = 0;
      long steps_fast = 0, accepted_fast = 0, accepted_window_fast = 0;
      double scale = 1.0, scale_fast = 1.0;
      bool frozen = false;                        // burn-in over, proposal frozen
      std::vector<std::vector<double>> L;         // Cholesky factor of the proposal covariance
      std::mt19937_64 rng;
      std::vector<double> best_x, best_derived;
      double best_lp = -std::numeric_limits<double>::infinity(), best_ll = 0.0;
    };

    Posterior post;
    const std::vector<SampledParameter>& par;    // post.params()
    SamplerOptions opt;
    std::vector<Chain> chains;
    double R1 = std::numeric_limits<double>::infinity();
    std::vector<double> best;
    double best_ll = -std::numeric_limits<double>::infinity(), best_lp = -std::numeric_limits<double>::infinity();

    void init_chain(Chain& c, int index);
    void advance(Chain& c, int n_steps, bool burn);
    void adapt(Chain& c) const;
    void set_proposal(Chain& c, const std::vector<std::vector<double>>& cov) const;
    double gelman_rubin() const;
    void write_output() const;
    void gather_chains();                         // MPI: every process gets all chains (in global order)
    bool is_root() const;
    size_t n_slow = 0;                            // parameters 0..n_slow-1 are slow, the rest fast
    bool mh_update(Chain& c, bool fast_only);     // one Metropolis-Hastings update; true if accepted
    void write_chain_file(const Chain& c, const std::string& file) const;
    void write_checkpoint(const Chain& c) const;
    bool load_checkpoint(Chain& c, int index) const;
};

} // namespace FML::MCMC

#endif
