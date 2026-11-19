# Next stage after the completed calibration pilot

## Current decision: sampler campaigns frozen (19 November 2026)

Further incremental sampler comparisons are frozen. The experimental kernels and
all failed outcomes below remain as historical evidence; their old “next” items
are superseded by this decision. Prioritize the [real Box-1 application](../../../docs/beyond2pt/BOX1_RESULTS.md):
observed catalog modes are prepared, but the exact matching phases and verified
reference spectrum/background assignment are still required for a fixed-IC fit.
No new sampler chains were run for this application step. A later joint-sampler
redesign should state one structural hypothesis, budget and decision criterion.

The frozen eight-dataset pilot completed all budgets: only 2/8 pass every gate.
See [all outcomes](../../../docs/validation/studies/calibration_pilot/RESULTS.md).
No chains, truths or failed results were replaced or extended.

[Conditional numerical probes](../../../docs/validation/studies/stability_probe/RESULTS.md)
on 96 saved states show that 0.075/16 steps remove observed local integration
failures compared with 0.15/8. Updating the zero-field linear mass to current
scalars alone makes the worst dataset less stable. All native/derivative controls
pass, but local probes do not validate full-chain mixing or calibration.

Next: freeze an actual-chain half-step comparison at equal integration time and
report force cost, all twelve diagnostics, transport and divergences. Retain the
original pilot. Nonlinear-curvature-aware mass design and exchange/mixing problems
remain separate; a new mass must preserve the conditional target and avoid an
uncorrected field-dependent kinetic measure. Full nonlinear calibration,
larger-target/real-catalog validation and production MPI replicas remain open.

## Completed paired comparison and dataset-203 audit

The [fixed four-chain 256/512 comparison](../../../docs/validation/studies/integrator_comparison/RESULTS.md) is complete, including actual gradient and wall cost, all twelve diagnostics and exact original-prefix replay. The [separate dataset-203 audit](../../../docs/validation/studies/mixing_audit/RESULTS.md) finds weak cold-end transport and curved/weak directions; shrinking the integration step alone does not address these.

Next experiment: preserve the cold target and compare a separately frozen ladder with additional levels between beta=1/0.5 and beta=0.5/0.25 (e.g. geometric midpoints). Declare equal cost or explicit work-normalized budgets and retain all coordinate, divergence and refreshed-state transport gates. Freeze and test new layout identities before sampling. This has not been run; better exchange alone would not establish convergence.

Use the stable half-step field configuration on both sides of the proposed ladder comparison, holding its mass and trajectory time fixed, so numerical divergence is controlled while isolating exchange changes. Keep production defaults and the original calibration pilot intact.

## Completed cold-end ladder comparison (19 November 2026)

[Both four-chain 256/512 cohorts](../../../docs/validation/studies/ladder_comparison/RESULTS.md) complete with zero divergences. The refined thirteen-replica ladder raises minimum pooled exchange acceptance from 0.0461 to 0.2827 and qualified returns from 2 to 9 for 20% more primary field gradients. Both still fail all twelve coordinate gates; one refined ensemble has no complete return. Provisional ESS/cost improves across all six field projections, but sigma8 and several bias coefficients worsen.

Next: separately freeze a field/scalar-coupling or preconditioning comparison on the difficult target, using the refined ladder as a transport reference and the stable half-step update. The current scalar kernel already has joint amplitude/bias proposals. Validate the new move/metric independently before sampling; retain all diagnostic gates and measured work. Broader prior-predictive calibration, larger fields and real-catalog application remain separate milestones.

## Completed nonlinear-metric comparison (19 November 2026)

[Local Gauss-Newton operator/trajectory controls](../../../docs/validation/studies/gauss_newton_probe/RESULTS.md) pass, but the [fresh four-chain 256/512 comparison](../../../docs/validation/studies/metric_comparison/RESULTS.md) fails all twelve coordinate and transport gates. Sigma8 bulk ESS improves 2.66 times at equal gradient work; all six field-projection bulk ESS values worsen. Zero divergences in both. Do not adopt the nonlinear metric as the new reference. Preserve the half-step/refined-ladder/original-mass reference and all experimental failures.

Next proposed increment: exact initial-support amplitude/field interweaving, leaving prior-only coordinates unchanged and using the global active-rank Jacobian. The current full-white rescaling is correct; this is an efficiency proposal. Use the model's exact initial filter rather than a fitted numerical rank, and validate reversibility, prediction invariance, analytic marginals and distributed ownership before any fixed-budget chain test. Robust joint calibration remains open.

## Completed exact-support interweaving (19 November 2026)

[The new kernel and frozen four-chain comparison](../../../docs/validation/studies/support_interweave/RESULTS.md) preserve the full prior and pass analytic/backend/native controls. Cold coupled acceptance improves 0.6581 to 0.8153; all five scalar ESS/gradient estimates improve (sigma8 2.97 times), and the transport gate passes with returns 6,2,1,4. All six white and post-hoc physical-mode ESS estimates worsen; all twelve original coordinate gates still fail. Retain the exact-support move as a validated optional coupling kernel, without a production/default or convergence claim.

Next proposed increment: full-target joint field/bias mode jumps, e.g. symmetric differences from a frozen donor pool, restricted to the exact initial subspace with physical scalar bounds rejected. First prove symmetry/detailed balance on multimodal analytic references and verify model/native identities, then freeze a bounded comparison. The support move improves scalar coupling but preserves physical initial modes; physical-mode trapping remains unresolved. Broader calibration is not ready.

### Frozen-donor joint field/scalar jumps (19 November 2026)

[The fixed-pool joint kernel](../../../docs/validation/studies/donor_jumps/RESULTS.md)
passes 30 independent native controls, frozen model feasibility, native identity/restart
and real four-chain runner checks. Fresh fixed 256/512 comparison retains the exact-support
rescaling, original field metric and target. Candidate failures: 12/12
original coordinates and 6/6 additional predeclared physical
initial-mode checks; transport False; divergences 0.
Extra donor value evaluations are explicitly counted. All failures and source records remain.
No independent calibration or production-catalog claim follows from this selected-data study.
