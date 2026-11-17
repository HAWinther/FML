> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Iteration-98 stable-Armijo endpoint

16 November 2026. [Protocol](README.md), [previous full fit](../pm_stable_armijo_validation/RESULTS.md).

## Result

The capture reproduces the stable fit's projected field and gradient **bitwise**,
with full-state fresh value/gradient errors exactly zero. It ends at iteration 98,
U=-1049.6324599322945 and raw gradient 1.3902008166620516e-4 versus the unchanged
2e-5 tolerance. Status remains **LineSearchFailed**. This diagnostic changes no
solver, assignment, force, gradient or stopping rule.

The closest corresponding trajectory coordinate is **7.46736e-13 grid cells**
from a CIC face, at final node 16, particle 22978, axis 2. This is a distance in
mesh coordinates, not a physical length or a guaranteed crossing threshold for
every perturbation direction.

*Archived figure: Endpoint diagnosis (`pm_stable_stationarity_validation/figures/endpoint.png`). See [archive guide](../../../ARCHIVE.md).*

## One-sided derivatives

For the normalized negative-gradient direction at h=1e-7:

| Quantity | Measured value |
| --- | ---: |
| Adjoint slope | -0.00425904582 |
| Left slope (negative perturbation, 0 changed stencil axes) | -0.00426080224 |
| Right slope (positive perturbation, 2 changed stencil axes) | +0.00802672575 |

At h=1e-6..1e-8 this directional asymmetry persists. The branch-preserving left
slope agrees closely with the adjoint, while positive movement along the nominal
negative-gradient direction crosses CIC branches and increases the target. These
are finite one-sided measurements near a face, not exact limiting derivatives.
The smallest perturbations become dominated by double-precision forward changes.
Stable target arithmetic cannot restore information lost in the predictions.

The existing central-difference gates require both signs to preserve the stencil
in h=1e-10..1e-5. **Negative-gradient and random directions have no eligible pairs
at this endpoint**, so both gates remain failed/unresolved. The failed-search
direction has five eligible pairs and passes (best scaled error 7.62e-8). All
three initial-state derivative gates pass. A lack of eligible pairs does not by
itself demonstrate a faulty adjoint. All scales and stencil changes are retained.

*Archived figure: Derivative checks (`pm_stable_stationarity_validation/figures/derivatives.png`). See [archive guide](../../../ARCHIVE.md).*

## Failed search and roundoff

All 40 reconstructed L-BFGS trial steps have positive stable changes; none meets
Armijo. The smallest measured change is about 7.29e-15, where forward roundoff
limits interpretation. The public stable-change API and independent algebraic
assembly agree (maximum absolute discrepancy 5.55e-17, at larger changes) and
make identical acceptance decisions. Their ordinary returned values agree exactly.

Ordinary subtraction would accept **17/40** steps, including tiny apparent
improvements where the stable change is positive. Thus simply reverting to
ordinary comparisons could admit roundoff-driven progress. The original unit
trial is reconstructed exactly; at very small alpha the reconstructed direction
can differ from the optimizer's internal arithmetic by subtraction rounding.

*Archived figure: Search and stencils (`pm_stable_stationarity_validation/figures/line_search.png`). See [archive guide](../../../ARCHIVE.md).*

## Replay and engineering

Fresh/reused workspaces, value/full targets and checkpoint budgets 0/1/4/8 give
**zero** value, field-gradient and scalar-gradient discrepancies at both tested
states. Observed/default trajectories are bitwise identical. No cache or replay
fault was found in these checks.

Short capture runs pass in serial, four-rank pencil and ASan/UBSan builds, with
stable comparisons both enabled and disabled. Old/default numerical capture
files are bitwise unchanged. Invalid flags and mismatched MPI flags reject.
The full capture, trajectory probe and public stable-API replay run on four
pencil ranks. Full-library CI and a new slab scientific run were not performed.

`summary.json` retains **passed=false** for the two unresolved derivative gates;
**consistency_passed=true**. No threshold or optimizer status was changed. This
is one frozen realization and does not establish local optimality in every
direction, physical calibration or convergence.

## Suggested next implementation

The evidence points to CIC branch regularity as a practical obstruction here,
with forward roundoff limiting very small steps. Keep stable comparisons as an
opt-in safeguard. A useful next experiment is an **opt-in TSC assignment with
matching deposit/gather derivatives and adjoint**, covering the relevant force
and reported-density paths. Preserve CIC defaults, validate derivatives across
TSC knots, then compare matched-model reconstructions at the same tolerance.
This would change the discretized forward model; it has not been implemented or
claimed converged here. Broader seed/amplitude/refinement remains deferred.
