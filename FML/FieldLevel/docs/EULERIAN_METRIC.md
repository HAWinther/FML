# Posterior-aware Eulerian field metric (FL-7)

Eulerian Gaussian MAP/HMC now obtains its field metric from the posterior's
actual noise policy, Gaussian covariance volume and retained-mode mask. On an
active initial mode the zero-field diagonal precision is

```
Q(k) = 1 + [b_delta - b_laplacian k^2]^2 P_L(k; theta) / P_noise.
```

`P_L` includes the spectrum/amplitude parameters at the analysis epoch. The
quadratic density and tidal operators have no linear response at zero field.
Only the intersection of the initial band, matter band and actual likelihood
mask contributes; DC, excluded Nyquist planes and other prior-only directions
have precision one. If a Gaussian likelihood uses a covariance volume different
from the model's physical volume, its variance normalization is included.
Supported mode weights are the model's half-complex multiplicities or zero,
with Hermitian symmetry on self-conjugate FFT planes. Asymmetric masks are
rejected before they could give inconsistent square-root/inverse operators.
This is a positive diagonal **linearization**, not the nonlinear posterior Hessian.

```cpp
auto posterior = FF::fixed_noise_posterior(model, observed_modes, noise_power);
auto work = model.workspace();
auto frozen = model.initial_parameters();
auto inverse_applied = posterior.apply_field_metric(input, frozen, work, -1.);
```

Use the posterior interface for Eulerian metric calls. It supplies the noise and
likelihood context; no dummy noise coordinate is added to the physical scalars.
The low-level model method `apply_gaussian_field_metric` requires that context
explicitly. The former unit-noise Eulerian `model.apply_field_metric` entry point
is removed. Existing non-Eulerian forwards use the generic dispatch's historical
metric path. Other likelihood families need their own metric adapter.

## Freezing, caching and numerical behavior

HMC keeps the metric parameters/noise at the saved initial scalar snapshot for
the entire run, even when the current sampled scalars change. MAP uses its current
scalar point. No position-dependent HMC metric or unaccounted kinetic derivative
is introduced. The Metropolis-corrected objective, data and target gradients are
unchanged; a pre-fix/corrected probe checks their serialized bytes exactly.

Metric applications transform the **sampled mesh directly**, including compact
grids. Product-grid lifting/restriction is unnecessary. Active-mode log precisions
are cached by scalar snapshot, effective noise/covariance and local likelihood
weights, and reused for square-root/inverse/power applications. Log-sum-exp avoids
overflow in intermediate response-squared/power ratios; unrepresentable metric
powers are rejected instead of introducing a zero/infinite mass. The FFT multiplier
is applied directly, avoiding subtraction of nearly equal identity corrections.

Cache invalidation is collective: a likelihood-mask change on only some owners
also rebuilds ranks with no active modes, keeping FFT collective order consistent.
Callback failures leave the cache invalid on every rank and can be retried. Metric
scratch leaves the latest forward/reverse tape and its scalar snapshot intact.
The zero exponent is exact identity; other powers preserve null modes to FFT
roundoff. The sampler state/layout and compact whitened prior are unchanged.

The [Eulerian checkpoint identity](EULERIAN_RESTART.md) now includes this metric
contract. **Pre-FL-7 Eulerian checkpoints are rejected** by the corrected reader;
resume them with their original executable or start a fresh chain. Existing
linear/LPT/PM/survey checkpoint identities and default metric behavior are retained.
Fresh-process corrected Eulerian runs reproduce uninterrupted trajectories exactly.

## Validation and exploration probe

[Recorded results](validation/studies/eulerian_metric/RESULTS.md) cover analytic
mode precisions for several noise levels and metric powers, an independently
finite-differenced zero-field posterior Hessian, covariance volume and masks,
explicit noise-index policies, cache reuse/recovery, metric symmetry/positivity,
MAP descent, forward-tape preservation and shared/compact slabs/pencils on
1/2/4 ranks. Existing compact and checkpoint regressions also run.

A separate frozen field-only probe compares genuinely pre-fix and corrected
executables at identical data, initial seeds, integrator settings and draw budget:
four chains, 16 warmup and 128 retained draws each, eight leapfrog steps at 0.15.
Six fixed active cosine/sine projections show bulk ESS rising from 4.5--4.9 to
171--261; maximum rank Rhat falls from 4.38 to 1.023. Both versions have zero
sampling divergences. The old traces are plainly unmixed, so this is a short-run
diagnostic comparison, not a calibrated asymptotic efficiency multiplier.

The corrected probe still does not pass every Rhat<1.01/bulk-ESS>400 production
gate. It samples fields with fixed physical scalars, not the marginalized sigma8
posterior, and it does not validate all field coordinates or coverage. No draws
were added, gates relaxed or step size retuned. Longer joint field/scalar mixing,
real-catalog application and posterior/refinement validation remain subsequent
work. FL-7's normalization/response defect is fixed; full scientific mixing is open.

```sh
make -C FML/FieldLevel/tests eulerian-metric USE_MPI=false
make -C FML/FieldLevel/tests eulerian-metric USE_MPI=true NP_LIST='1 2 4'
```
