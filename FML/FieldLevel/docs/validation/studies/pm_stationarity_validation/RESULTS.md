> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Stalled PM reconstruction: diagnosis

16 November 2026. [Protocol and reproduction](README.md).

**This bounded case points to CIC boundary proximity plus finite-precision
objective comparisons. It does not show an adjoint/cache discrepancy in the
probed directions. No convergence fix is applied.**

The unchanged L-BFGS solve reproduces the committed failure at iteration 83:
U=-1049.6324599303684, raw gradient 1.1603772163e-4, field-gradient L2=.00339709550.
Captured projected coefficients and gradients match the committed result bitwise.
The full distributed state is retained, allowing probes without repeating the solve.

## Cache and derivative evidence

Fresh workspaces, reused workspaces after all perturbations, and replay budgets
0/1/4/8 give **zero measured value, field-gradient and scalar-gradient differences**
at the stalled state and initial-field control. Public value-only and full target
values also agree exactly at both base points. The independently assembled raw
value matches the public value on every measured perturbation. Every observer
trajectory agrees bitwise with the default trajectory.

All three directions pass the preceding scaled derivative criteria on measured
stencil-preserving pairs. All h values, including crossings and roundoff failures,
remain in the plots/CSV. Illustrative stalled-point one-sided slopes:

| Direction | h | Adjoint | Left slope | Right slope | Changed axes, left/right |
| --- | ---: | ---: | ---: | ---: | --- |
| Negative gradient |1e-9 |-.0033970955 |-.0033896032 |-.0033991547 |0 / 0 |
| Failed search, normalized |1e-9 |-.0026678058 |-.0026670888 |-.0026662801 |0 / 0 |
| Random active |1e-8 |-.0001249809 |-.0001249642 |-.0001250581 |0 / 0 |

At h=1e-8 along the negative gradient, the opposite side changes two stencil
axes and its slope becomes+.0076511. The unchanging side remains close to the
adjoint. At extremely small h the double-precision forward differences become
noisy; those values are retained rather than treated as derivative failures.

*Archived figure: One-sided derivative errors (`pm_stationarity_validation/figures/derivatives.png`). See [archive guide](../../../ARCHIVE.md).*

## Why the line search rejects

The nearest face is only 3.1902214204e-10 force-grid cells away, at final node 16
(a=1), Lagrangian particle 22978 (grid coordinate 22,14,2), axis 2. This is a
perturbation-induced CIC support change at a corresponding node, not a count of
ordinary particle motion between time steps. All17 force/deposition nodes are checked.

All40 ordinary public-value comparisons reject, reproducing the failed search.
Larger trials change CIC stencils and increase the objective. A much smaller
stencil-preserving descent window exists in the algebraically stable difference:

| At alpha=3.814697265625e-6 | Objective change |
| --- | ---: |
| Adjoint linear prediction | -2.05026375e-12 |
| Stable Gaussian/white-prior difference | -2.04350461e-12 |
| Ordinary public-value subtraction | +1.00044417e-11 |

No stencil changes at that candidate. Stable differences identify 11 Armijo-passing
samples overall, including a stencil-preserving interval. The smallest samples
approach forward roundoff and are not a high-precision gravity reference. The
stronger example above agrees with the linear prediction within about 0.3%.
Raw target evaluation/accumulation and subtraction obscure this small decrease.
There is no measured base-point value/full normalization offset.

*Archived figure: Failed search, stencil changes and nearest face (`pm_stationarity_validation/figures/line_search.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Directional stencil changes (`pm_stationarity_validation/figures/boundaries.png`). See [archive guide](../../../ARCHIVE.md).*

The stable difference is an independent algebraic evaluation of the same
fixed-noise Gaussian target using represented forward predictions. Local
long-double accumulation has 64 mantissa bits on this build; prediction FFTs and
the global reduction remain double precision. It changes neither solver nor target
in this study. It demonstrates a numerically hidden descent window, not that a
modified optimizer will necessarily reach the gradient tolerance.

## Engineering checks

Fifteen observer/adjoint/observation regression invocations pass in serial,
MPI 1/2/4 slabs/pencils and ASan/UBSan (leak detection disabled). Tests check all
nodes, bitwise output preservation, collective one-rank callback failure, workspace
invalidation/recovery and periodic endpoint canonicalization. Snapshot smoke tests
load a serial state across serial, slab/pencil1/2/4 and sanitizer builds. Missing
metadata agreement, duplicate IDs and inconsistent rank paths reject collectively.
Eleven legacy numerical output files remain bitwise unchanged after the shared
validation helper extraction. See [engineering.json (archived)](../../../ARCHIVE.md).

One exploratory pass assembled raw values independently; the final pass additionally
uses the exact public value path and confirms they match. Original logs/data remain
in review artifacts. No solver, adjoint or force/assignment law was modified, and
no full-library CI run is claimed.

## Next step and limits

Implement and test stable objective-change comparisons for Armijo on this frozen
state, then repeat the complete reconstruction with unchanged tolerance. Keep CIC
nonsmoothness and force-precision limits explicit. Further assignment changes
should follow evidence from that test.

One realization, fixed nuisance parameters and three directional probes do not
verify every adjoint component or establish posterior calibration. Broader
seed/amplitude/refinement checks remain deferred. Production restart, performance
claims and survey/tracer/RSD/cosmological extensions are outside this diagnostic.
