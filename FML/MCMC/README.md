# MCMC (FML/MCMC)

Samplers for any posterior given as a likelihood function of a parameter vector, in namespace `FML::MCMC`. Nothing
here knows about cosmology: `FML/CosmologyFitting` uses it for the cosmological fits (`./mcmc`), with the
Einstein–Boltzmann solver of `FML/Cosmology` as the theory.

| File | Contents |
|---|---|
| `Posterior.h/.cpp` | `Posterior`: the sampled parameters (`SampledParameter`: start, flat prior [min, max], step, optional Gaussian prior, fast flag), the derived parameters and the likelihood function (`LikelihoodFunction`: x → ln L and derived values); `evaluate`, `log_prior`; GetDist `.paramnames`/`.ranges`; covmat files and small linear algebra (`cholesky`, `spd_inverse`, `read_covmat`, `write_covmat`) |
| `Sampler.h/.cpp` | `Sampler` (options `SamplerOptions`): adaptive Metropolis–Hastings with several chains in parallel (OpenMP threads and MPI processes), fast/slow parameters, Gelman–Rubin convergence, checkpoints and resume, GetDist output |
| `Minimizer.h/.cpp` | `Minimizer` (options `MinimizerOptions`): maximum of the posterior (Nelder–Mead with restarts, fast/slow split, several starts), profile likelihoods, Hessian covariance |
| `HMC.h` | Header-only Hamiltonian Monte Carlo for gradient-based posteriors of any dimension (`run_hmc`, `HMCOptions`, `HMCMass`), plus `integrated_autocorrelation_time` and `gelman_rubin`; see [HMC](#hmc) |
| `Parallel.h/.cpp` | The MPI operations the samplers need, on FML's MPI setup (`Global`); trivial without `USE_MPI` |
| `example/` | A correlated 2D Gaussian: sampling, best fit, Hessian and a profile |
| `tests/` | `test_sampler` (Cholesky, correlated Gaussian, priors, rejected regions, GetDist output, fast/slow, checkpoint/resume), `test_hmc` (diagonal mass adaptation over 6 decades of variance, an operator mass block coupled to parameters, reflection at bounds, Gelman–Rubin), `test_minimizer` (Gaussian with prior, Rosenbrock, prior boundary, two peaks, fast/slow, profile, Hessian) |

## Usage

```cpp
#include <FML/MCMC/Sampler.h>
#include <FML/MCMC/Minimizer.h>
using namespace FML::MCMC;

LikelihoodFunction like = [](const std::vector<double>& x) {
  LikelihoodResult r;
  r.loglike = -0.5 * (x[0] * x[0] + (x[1] - 1.0) * (x[1] - 1.0) / 4.0);
  r.derived = {x[0] + x[1]};
  return r;
};
SampledParameter a{"a", "a"}, b{"b", "b"};
a.start = 0.0; a.min = -10; a.max = 10; a.proposal = 1.0;
b.start = 0.0; b.min = -10; b.max = 10; b.proposal = 2.0;
Posterior post({a, b}, {"a_plus_b"}, {"a+b"}, like);

SamplerOptions opt;
opt.output_root = "chains/toy";   // GetDist chains chains/toy_<i>.txt, .paramnames, .ranges, .covmat, .summary.txt
Sampler mc(post, opt);
mc.run();                         // mc.mean(), mc.stddev(), mc.covariance(), mc.best_fit(), mc.Rminus1()

Minimizer mz(post);
MinimizerResult best = mz.run();  // best.x, best.logpost, best.loglike, best.derived
std::vector<double> h = {1.0, 2.0};
std::vector<std::vector<double>> C;
Minimizer::covariance(mz.hessian(best.x, h), h, C);
```

A complete program is in `example/` (`make && ./mcmc`). Link `Posterior.o Sampler.o Minimizer.o Parallel.o
Global.o` (FML modules `MCMC Global`, see `example/Makefile`).
With MPI (`USE_MPI=true`) FML initializes MPI; the chains (and minimizer starts, profile points) are spread over
the processes and every process gets all results.

## Sampler

- During the burn-in the proposal covariance and scale adapt (Haario et al. 2001) towards 25% acceptance; at the
  end of the burn-in the covariance pooled from all chains is frozen, so the sampling phase is an exact Markov chain.
- The run stops once every chain has `min_samples` samples and R−1 < `Rminus1_stop` (Gelman–Rubin), or after
  `max_steps`.
- A point where the likelihood throws or is not finite, or that is outside the prior, is rejected.
- Fast/slow (Lewis 2013, as CosmoMC): fast parameters (cheap to vary, e.g. likelihood nuisance parameters when the
  likelihood caches the rest) come last. Each step is followed by `oversample_fast` updates that move only the fast
  parameters (the proposal L z with z_slow = 0, L the Cholesky factor, i.e. their conditional covariance), with
  their own scale.
- Checkpoints: after every round the chains and the full state of each chain (position, proposal, scales,
  counters, burn-in history, random generator) are written; with `resume = true` a run continues exactly as an
  uninterrupted one (with any number of MPI processes).

## HMC

`HMC.h` is independent of `Posterior`/`Sampler`: it needs U = −ln posterior *and its gradient*, and is meant for
problems with many parameters (field-level inference, `FML/FieldLevel`), where Metropolis does not scale and where
storing every state is impossible. Header only, no FML dependencies.

```cpp
#include <FML/MCMC/HMC.h>
using namespace FML::MCMC;
HMCPotential U = [](const std::vector<double>& x, std::vector<double>& grad) {
  grad = {x[0], 4 * x[1]};
  return 0.5 * x[0] * x[0] + 2 * x[1] * x[1];
};
HMCOptions opt;           // warmup, samples, n_leapfrog, target_acceptance, lower/upper, seed, ...
opt.lower = {-10, 0};     // optional bounds (reflection)
opt.upper = {10, 5};
HMCResult r = run_hmc(U, {0.1, 0.1}, HMCMass{}, opt);
auto x0 = recorded_column(r, 0);              // post-warmup samples of parameter 0
double tau = integrated_autocorrelation_time(x0);
```

- State = [custom block | parameters]. The custom block (`HMCMass::n_custom` leading coordinates, e.g. a field)
  has a mass given as two operators, `apply_sqrt` (white noise → M^{1/2} noise) and `apply_inverse`; nothing of it
  is recorded. The parameters have a diagonal mass, adapted during warmup (Stan's windows: 15% initial buffer,
  windows of 25, 50, 100, ..., 10% final buffer; regularised variance), may have bounds and are recorded each step
  (or pass a recorder function to record anything else).
- Leapfrog with a jittered number of steps in [n/2, n]; the step size is adapted by dual averaging
  (Hoffman & Gelman 2014) towards `target_acceptance` and restarted after each mass window. Bounds are handled by
  reflection, which is exact for the diagonal-mass parameters. A non-finite potential rejects the trajectory;
  `divergences` counts trajectories with H1 − H0 > `divergence_threshold` in the sampling phase.
- One chain per call: run several with different seeds (and starts) and use `gelman_rubin` on recorded columns.

## Minimizer

- Nelder–Mead simplex (adaptive coefficients, Gao & Han 2012) in coordinates whitened by a covariance (the
  proposal covmat, else the steps), so the first simplex has the size of the posterior in every direction. It needs
  no derivatives and tolerates small numerical noise. Once the simplex spread in −2 ln P is < `tol` it restarts
  around the best point, until a restart gains < `tol`.
- Fast/slow: the simplex runs over the slow parameters; at each slow point an inner simplex minimizes the fast ones.
- `starts` independent minimizations (the first from the start values, the others scattered by the covariance),
  over MPI processes and OpenMP threads; their spread shows whether the minimum is global.
- The objective is −2 ln posterior: Gaussian priors count as data, flat priors only bound the parameters.
- `profile(i, values, from)`: minimum over the other parameters at fixed values of parameter i.
- `hessian(x, h)`: central differences of −ln P (2d² + 1 evaluations over the MPI processes), with the steps first
  adapted to 1.5 conditional σ; `covariance()` inverts it.

Tests: `cd tests && make run`.
