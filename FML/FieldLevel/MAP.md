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

See [the frozen-data comparison](pm_optimizer_validation/README.md) and
[measured results](pm_optimizer_validation/RESULTS.md).
