> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Stable Armijo: frozen-state check and reconstruction

16 November 2026. [Protocol](README.md), [API](../../../MAP.md).

The stable comparison accepts the diagnosed descent at the frozen state. Ordinary value comparisons still reject all 40 samples; stable comparisons accept 11, reproducing the independent diagnostic. The returned ordinary values agree with the existing value API. Forces, assignment, gradient and tolerance remain unchanged.

*Archived figure: Frozen search (`pm_stable_armijo_validation/figures/frozen.png`). See [archive guide](../../../ARCHIVE.md).*

## Full reconstruction

| Method | Iterations/status | Raw gradient | Predictions | Pullbacks | Reference predictions | MAP seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| L-BFGS ordinary | 83 / LineSearchFailed | 0.000116038 | 1395 | 84 | 0 | 482.30 |
| L-BFGS stable | 98 / LineSearchFailed | 0.00013902 | 792 | 99 | 99 | 297.75 |
| GD ordinary | 24 / MaximumIterations | 0.00219002 | 168 | 25 | 0 | 65.56 |
| GD stable | 24 / MaximumIterations | 0.00219002 | 192 | 25 | 24 | 69.25 |

Stable L-BFGS progresses past the old recorded objective, reaching U=-1049.6324599322945, but it still ends with **LineSearchFailed** above the unchanged 2e-5 gradient tolerance. Its coefficient error is 0.269554, correlation 0.964336, and residual 0.788084 noise SD. The identified false rejection is removed; this does not establish a converged reconstruction.

Value trials drop from 1311 to 594. The new prediction count includes 99 baseline-reference predictions. Wall time is indicative under uncontrolled host activity. Stable GD retains the 24-step coefficients bitwise, with reference-prediction overhead. All accepted stable changes satisfy the change-form Armijo inequality; rounded absolute objectives can move by roundoff.

*Archived figure: Convergence and work (`pm_stable_armijo_validation/figures/comparison.png`). See [archive guide](../../../ARCHIVE.md).*
*Archived figure: Reconstruction (`pm_stable_armijo_validation/figures/stable-lbfgs-initial.png`). See [archive guide](../../../ARCHIVE.md).*

The summary deliberately retains the failed scientific convergence criterion (`passed=false`). Consistency and recovery checks pass. No optimizer status or tolerance was relaxed. The remaining line failure needs further study of CIC nonsmoothness and forward precision at the new endpoint; it is not claimed solved by this change.

## Engineering

Eighteen regression/optimizer invocations pass across serial, MPI1/2/4 slabs/pencils and ASan/UBSan, with bounds checks and leak detection disabled. Tests include an analytic coupled quadratic with a constant 1e24 target term, tiny/large noise changes, variance/normalization, empty MPI ownership, Gaussian/uniform scalar priors, antisymmetry, foreign references, unsupported likelihoods, rank-inconsistent options and prediction accounting. The huge-constant test reaches the known field optimum using stable comparisons.

Twelve frozen-state/input/default-output engineering runs pass: serial, slab/pencil1/2/4 and sanitizer snapshot replays; 11 legacy numerical CSV files remain bitwise unchanged with the new feature disabled. Legacy-API misuse and inconsistent rank flags reject. The combined validation Makefile builds both the frozen probe and reconstruction executable. Full-library CI was not run.

Local likelihood/white-prior changes use long-double arithmetic and a double global reduction; scalar priors are added once afterward. The reference stores modes/parameters and uses the caller-owned consistent base white field. Existing value/evaluate methods, HMC target values, checkpoint identities and default MAP behavior are unchanged. Stable requests for unsupported likelihoods fail explicitly.

## Scope and next step

One fixed-parameter realization and finite reference. Broader seed/amplitude/refinement and posterior calibration remain deferred. No new assignment kernel, force law, survey likelihood difference, cosmological derivative or production scaling is claimed. Keep stable comparisons available as an opt-in numerical safeguard. Inspect the new failed endpoint before deciding whether a smoother assignment or another stopping rule is warranted.
