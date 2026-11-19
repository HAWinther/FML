# Conditional HMC integration and metric probes

All declared local probes completed: **96 saved states, five cases, eight paired
momentum trials, 3840 primary field trajectories**. Derivative/forward/native
controls: **PASS**. The original eight-dataset pilot and checkpoints are unchanged.
These are conditional numerical-stability probes, not new posterior chains.

The fixed source states are draws 0,1024,2047 from every original chain. All scalar
parameters remain fixed during each trajectory. The cases use equal integration
time 1.2: baseline 0.15/8 steps, half-step 0.075/16, quarter-step 0.0375/32; two
more cases use the current scalar parameters in the linearized mass at 0.15/8 and
0.075/16. The original mass uses each ensemble's initial scalar parameters.

## Main results

Dataset 204 reproduces local integration failures: baseline **15/96 divergent
trajectories**, mean proposal acceptance **0.572**. Half-step and quarter-step give
**0/96** divergences with acceptance **0.966/0.993**. The half-step case has zero
divergences across all 768 primary trajectories in that variant. This is evidence
for a safer local integration setting on these states, not a global guarantee.

Simply updating the zero-field linear metric to current scalar parameters makes
dataset 204 worse: **36/96** divergences and acceptance **0.422**. Adding the half
step to that metric yields zero divergences and acceptance 0.918, still below the
original-metric half-step case. It also reduces dataset 203's acceptance at the
original step from 0.986 to 0.658. This candidate is not selected as an improvement.

| Dataset | Baseline acceptance | Half step | Quarter step | Current-scalar metric | Current metric + half step |
|---|---:|---:|---:|---:|---:|
| 201 | 0.907 | 0.977 | 0.994 | 0.978 | 0.995 |
| 202 | 0.982 | 0.996 | 0.999 | 0.986 | 0.996 |
| 203 | 0.986 | 0.997 | 0.999 | 0.658 | 0.969 |
| 204 | 0.572 | 0.966 | 0.993 | 0.422 | 0.918 |
| 205 | 0.983 | 0.996 | 0.999 | 0.960 | 0.990 |
| 206 | 0.950 | 0.987 | 0.997 | 0.891 | 0.982 |
| 207 | 0.767 | 0.964 | 0.991 | 0.974 | 0.994 |
| 208 | 0.982 | 0.996 | 0.999 | 0.981 | 0.995 |

The linear metric is proportional to
`1 + alpha^2 [b_delta - b_laplacian k^2]^2 P_ref/P_noise` on supported modes.
It omits nonlinear density/tidal response. For one problematic dataset-204 frame,
the current-scalar metric has only 0.117–0.223 times the original mode precision;
all eight new probes diverge there while the original metric has none. This is
consistent with dropping useful protection while nonlinear curvature remains.
It does not independently identify the full nonlinear Hessian or a universal mass.

No dataset-207 divergence is reproduced in the 96 baseline probes there, despite
81 cold divergences in its original 8192 cold updates. The local sample therefore
must not be treated as a reconstruction or frequency estimate for all historical
failures. Likewise, stable local trajectories cannot establish that datasets 203,
205 or 206 mix well; their original failures remain intact.

## Controls and cost limits

All 96 source predictions/quadratics reproduce. Three field-direction and five
scalar finite-difference checks per state pass; maximum best scaled discrepancy
is 6.25e-7. Bound-limited scalar steps are local checks. The instrumented kernel
reconstructs Hamiltonian error and matches actual acceptance/divergence decisions.
Trial zero for every case/state exactly matches the ordinary field sampler:
**480 exact native comparisons**, excluded from primary summaries. A separate
stiff-Gaussian fixture has 96 exact matches and distinguishes unstable integration
from small-step/matched-mass stabilization.

Half and quarter steps cost roughly two and four times the leapfrog force calls.
The plotted accepted full-white jump per gradient includes prior-only directions;
it is not an active-field ESS or full-chain efficiency estimate. Correlated saved
states, eight momenta and post-hoc diagnostic design do not supply a population
coverage or mixing result. No original sampler defaults, priors or chain lengths
were changed, and no inputs or failed cases were replaced.

## Reproduction and next experiment

[Maintained driver and controls](../../../../validation/galaxy/stability/README.md),
[fixed plan](../../../../validation/galaxy/stability/plan.json), FROZEN.json and
summary.json preserve all settings and outcomes. Raw inputs, per-trajectory CSVs,
native logs and the inspected figure remain at
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-stability/`.
The figure is campaign/stability-overview.png.

Next: a separately frozen actual-chain comparison using the half-step integrator,
with original failures preserved and unchanged diagnostics. Do not replace the
mass with the current-scalar linear metric alone. Further metric work should
capture nonlinear curvature; exchange-ladder and field/scalar mixing failures
need separate diagnosis. No new full-chain calibration claim is established here.
