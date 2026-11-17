# MAP optimization

`Framework.h` retains the original `map(posterior, white, parameters, vary, iterations,
tolerance)` interface and its numerical behavior. Include `MAP.h` for the separate
options overload:

```cpp
#include <FML/FieldLevel/MAP.h>
FF::MAPOptions options;
options.algorithm = FF::MAPAlgorithm::LBFGS;
options.vary_parameters = false;
options.max_iterations = 128;
options.tolerance = 2e-5;
options.history = 5;
auto report = FF::map(posterior, initial_white, parameters, options);
const auto & fit = report.fit;
// Inspect fit.status, fit.gradient_norm and report.trace before claiming convergence.
```

`MAPAlgorithm::GradientDescent` uses the same field metric/scalar scaling as the
legacy optimizer, with value-only rejected trials and no L-BFGS history. Defaults
are L-BFGS, joint parameters, 200 iterations, tolerance 1e-6, history 5, 40 line-search
trials, Armijo coefficient 1e-4 and shrink factor .5. History must be positive for
L-BFGS; zero is permitted for options-based gradient descent.

## Target, coordinates and distributed ownership

`BasicPosterior::value(white,p,workspace)` evaluates the same normalized likelihood,
white prior and scalar priors as `evaluate`, without calling the forward-model
pullback. Unsupported or numerically unrepresentable states return infinity;
programming/model errors still propagate. It uses the existing likelihood
`evaluate` interface: custom/survey likelihoods may still calculate cotangents and
observation transposes. The expensive forward-model adjoint is omitted. Accepted
trials are reevaluated with a fresh prediction and gradient; no implicit cache
reuse is assumed.

The optimizer works in white-field coordinates and scaled scalar coordinates
`p_j / Parameter::scale()`. Fields and their dot products are distributed; scalar
coordinates/gradients are replicated and added after field reductions, exactly
once. When parameters are fixed they never enter the search direction or history
updates. The field base inverse Hessian uses the model's existing metric with
exponent -1; the scalar base metric is identity in scaled coordinates. L-BFGS scales
the base metric using `s.y / (y.H_base.y)`.

Built-in slab/pencil models collectively validate all options. Custom distributed
models must expose `controls(const std::vector<double>&)` for replicated optimizer
metadata, in addition to the existing `Execution` state/reduction/control policy.
Serial models need no additional control hook. Bulk scratch/history/trace storage
is allocated with collective failure propagation through `Execution::maximum`.
Forward/metric/likelihood callbacks retain their existing collective-safety contract.

## Safeguards and memory

Armijo backtracking tests values first. L-BFGS skips nonfinite, nonpositive or nearly
orthogonal curvature pairs, and resets history if its direction is not descent.
The base metric must be finite and positive definite. Armijo alone does not imply
Wolfe curvature conditions or a global convergence guarantee. Piecewise CIC models
can remain sensitive to assignment faces; status and gradients remain explicit.

`history=m` retains at most 2m field vectors and 2m scalar vectors per rank. Vector
payload is reported in `peak_history_bytes`; container overhead, temporary vectors,
forward/adjoint workspaces and trace storage are additional. The trace stores scalar
diagnostics once per accepted iterate, including iteration 0. It does not retain
field snapshots. `peak_history_pairs`, value/gradient evaluation counts, skipped
pairs and history resets are reported separately.

The convergence norm matches legacy MAP: global maximum absolute raw field-gradient
component, plus maximum scaled scalar-gradient component when parameters vary.
It depends on the mesh normalization. The trace also records the full field-gradient
L2 norm. Optimizer iteration limits and failed line searches never count as convergence.
No posterior/likelihood/checkpoint numerical law is changed by choosing an optimizer.

See [the frozen-data comparison](validation/studies/pm_optimizer_validation/README.md) and
[measured results](validation/studies/pm_optimizer_validation/RESULTS.md).

## Stable objective-change comparisons

Set `options.stable_objective_changes=true` to compare a directly computed target
change with `-armijo*step*slope`. The default is false, preserving prior options-based
runs and legacy comparisons. Explicit requests require a likelihood with a local
`difference(trial_modes, trial_log_noise, base_modes, base_log_noise)` method
returning a value convertible to long double. Unsupported likelihoods reject the
request; survey likelihoods do not gain this interface automatically.

`GaussianFieldLikelihood::difference` includes half-complex multiplicities, the
noise variance and its parameter-dependent normalization. Small log-noise changes
use `expm1`; larger changes avoid overflowing variance ratios. `BasicPosterior`
adds the stable white-prior change and scalar-prior changes, with scalar priors
added once after the distributed field/likelihood reduction. Gaussian and uniform
scalar priors retain their existing meaning. Ordinary `value`/`evaluate` and their
numerical target values are unchanged.

The lower-level API is:

```cpp
auto reference = posterior.value_reference(base_white, base_parameters, workspace);
auto trial = posterior.value_change(candidate_white, candidate_parameters, workspace,
                                    base_white, reference);
// trial.value is the ordinary target; trial.change is trial minus base.
```

The reference owns local predicted modes and a small parameter vector, not a copy
of the base white field. Supply the unchanged consistent `base_white` for every
call using that reference. The posterior/model must outlive it, and callbacks,
likelihood and prior definitions must remain deterministic and fixed. Foreign or
empty references reject collectively. Inputs and parameter support retain the
usual forward-model contract.

The optimizer makes one extra reference prediction per attempted iteration and
reports `reference_evaluations`. Accepted points still recompute a fresh gradient
and require finite values/gradients; they are not retested using the rounded
full-value inequality that this mode replaces. `MAPIteration::accepted_change`
and `armijo_bound` audit accepted trials. Rounded absolute trace values can move
by roundoff; stable comparisons do not promise strictly monotone rounded values.

Stable arithmetic removes lost objective changes, but forward predictions remain
double precision and CIC support boundaries remain nonsmooth. The unchanged
stopping criterion can still fail. See [the bounded rerun](validation/studies/pm_stable_armijo_validation/RESULTS.md).
