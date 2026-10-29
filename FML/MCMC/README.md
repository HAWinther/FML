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
- The run stops once every chain has `min_samples` samples and R−1 < `Rminus1_stop`, or after `max_steps` (then
  with a warning, and `converged()` is false). R−1 is the Gelman–Rubin statistic as computed by CosmoMC and
  Cobaya: the largest eigenvalue of W⁻¹B (W the mean of the chain covariances, B the covariance of the chain means),
  which also catches unconverged combinations of parameters. One chain has no R−1 and always runs to `max_steps`.
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
- One chain per call: run several with different seeds and starts, then use `diagnose_hmc_chains` on recorded columns. Each concurrent chain needs its own model/workspace when the potential uses mutable scratch.

## Convergence diagnostics

`Diagnostics.h` is a standalone C++17 header, with no FFTW, MPI or Python dependency.
It implements rank-normalized split-Rhat, folded rank-normalized split-Rhat, their
maximum `rhat`, and bulk/tail effective sample sizes. ESS uses multi-chain variance,
FFT autocovariances and Geyer's initial positive/monotone sequence; anticorrelation
can give ESS above the draw count (bounded by S log10(S)). Tail ESS is the minimum
of the 5% and 95% quantile-indicator ESS values, also returned separately.
Definitions follow [Stan's posterior analysis guide](https://mc-stan.org/docs/reference-manual/analysis.html)
and [Vehtari et al.](https://doi.org/10.1214/20-BA1221).

```cpp
#include <FML/MCMC/Diagnostics.h>
// chains[chain][draw]: one scalar, equal-weight post-warmup draws
const auto d = FML::MCMC::diagnose_chains(chains);
if (!d.valid()) {
    std::cerr << FML::MCMC::diagnostic_status_name(d.status) << "\n";
} else {
    std::cout << d.rhat << " " << d.ess_bulk << " " << d.ess_tail << "\n";
}
// Existing Metropolis sampler, after run(): includes sampled/derived column j.
const auto parameter = mc.diagnostics(j);
// Independent HMCResult objects; j selects a recorded column (beta is column 0 in FieldLevel).
const auto hmc_parameter = FML::MCMC::diagnose_hmc_chains(runs, j);
```

The API requires at least two independent chains with eight draws each. Warmup
must already be removed. Keep every rejected-step repeat; compressed GetDist
weights must be expanded into chronological repetitions, not treated as draws or
arbitrary importance weights. Do not concatenate independent chains or label
successive chunks of one chain as independent runs. Ragged lengths throw rather
than silently truncate. Odd-length chains drop their middle draw when split;
tail quantiles use the original pooled draws, matching the reference convention.
Ranks use average ties and Blom scores `(rank - 3/8)/(S + 1/4)`.

`status` distinguishes insufficient chains/draws, nonfinite input, an original
chain frozen at one value, and undefined numerical estimates. Initial invalid
results have NaN metrics; a numerical-failure result may retain defined component
metrics but `valid()` is false. `valid()` means estimates are defined, **not**
that chains converged. `draws_used` counts retained split draws. A constant
folded trace carries no scale information and contributes folded Rhat=1; an
original constant chain is instead invalid. If a tail indicator has no pooled
variation its ESS is undefined, not the sample count. These explicit degeneracy
policies are intentionally more conservative than ArviZ's constant-ESS shortcut.

Run several dispersed chains (four is a useful default) and examine scientifically
relevant parameters and field summaries, not just one nuisance parameter. Rhat
below 1.01 and bulk/tail ESS above roughly 100 per chain are useful screening
criteria, not guarantees of posterior accuracy. Inspect trace behavior and HMC
divergences as well. The two-chain FieldLevel unit check uses looser bounds as a
regression smoke check; it does not certify joint-field convergence or coverage.

`Sampler::Rminus1()` and its automatic stopping threshold are the CosmoMC/Cobaya
multivariate Gelman–Rubin statistic (largest eigenvalue of W⁻¹B, see above). Modern diagnostics are explicit post-run calculations over
retained samples; they do not change transitions, adaptation, checkpoints or the
stopping statistic. The legacy stopping code now fails closed on zero/nonfinite
within-chain variance. The legacy HMC helpers also fail closed: short, frozen or
nonfinite scalar series have infinite IAT (implied ESS=0); degenerate legacy
Gelman-Rubin returns infinity. New code should use the modern multi-chain API.
This work does not implement sample streaming or online moments (backlog #16).

Native tests include checked-in numerical fixtures generated with ArviZ 0.21.0,
so normal builds/CI need no Python packages. Optional reference regeneration and
broader checks:

```sh
c++ -std=c++17 -O2 -I/path/to/FML tests/diagnostics_driver.cpp -o /tmp/diagnostics_driver
# In a separate Python environment with arviz==0.21.0 and numpy:
python tests/check_diagnostics_reference.py --driver /tmp/diagnostics_driver
# Add --write-header tests/DiagnosticsReference.h only to deliberately regenerate fixtures.
```

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
