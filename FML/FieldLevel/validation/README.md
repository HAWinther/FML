# Maintained validation drivers

| Directory | Purpose |
|---|---|
| [pm/hmc](pm/hmc/README.md) | Small HMC/quadrature references and the test-suite smoke check |
| [pm/joint](pm/joint/README.md) | TSC joint-MAP initialization used by the posterior driver |
| [pm/posterior](pm/posterior/README.md) | Active-field/amplitude/noise sampling and independent-mock campaigns |
| [survey](survey/README.md) | Cartesian Gaussian/Poisson calibration and portability checks |
| common | Shared PM problem declarations; internal to validation |

Keep run outputs outside this directory. Historical investigations and full raw
runs are described in [the validation index](../docs/VALIDATION.md) and preserved
in [the external archive](../docs/ARCHIVE.md). These bounded drivers do not
constitute production cosmological calibration.

## Periodic galaxy amplitude project

[galaxy/fixed_ic](galaxy/fixed_ic/README.md) contains the bounded synthetic
fixed-IC amplitude/bias pilot and its frozen protocol. It keeps all chain/plot
outputs outside the library. The [first result](../docs/validation/studies/galaxy_fixed_ic/RESULTS.md)
retains its failed mixing gate; this is not yet a galaxy-catalog analysis.

The [joint-proposal follow-up](../docs/validation/studies/galaxy_fixed_ic_joint/RESULTS.md)
passes the fixed pilot gates using `galaxy/fixed_ic/plan_joint.json` and the same
frozen data. Both campaigns are retained separately.

`galaxy/numerics/` contains a frozen, resumable synthetic numerical/resource
screen. Keep its raw modes, logs and plots outside the library. See
[the guide](../docs/NUMERICAL_SCREEN.md) for results and sampling-readiness limits.

[galaxy/joint](galaxy/joint/README.md) declares the checkpointed four-chain joint
initial-field/amplitude/four-bias pilot. Its [recorded result](../docs/validation/studies/joint_pilot/RESULTS.md)
retains the original fixed budget and every failed gate.

[galaxy/blocked](galaxy/blocked/README.md) compares alternating field HMC and
cached scalar updates against the joint pilot at the same target/draw budget.
Its [recorded result](../docs/validation/studies/blocked_pilot/RESULTS.md) preserves
all gates and failures; the original joint campaign stays separate.

### Amplitude/field rescaling (18 November 2026)

[galaxy/interweave](galaxy/interweave/README.md) adds a separately declared amplitude/full-field MH block to the unchanged blocked sampler. [Results](../docs/validation/studies/interweave_pilot/RESULTS.md) retain the frozen gate **FAIL** and full comparison cost.

### Geometry audit (18 November 2026)

[galaxy/geometry](galaxy/geometry/README.md) diagnoses the separated interweaving chain using deterministic paths, exact bounded-bias profiles and endpoint finite differences. [Recorded results](../docs/validation/studies/geometry_audit/RESULTS.md) motivate a separate likelihood-tempering connectivity test.

### Replica-exchange connectivity (18 November 2026)

[galaxy/tempering](galaxy/tempering/README.md) implements a serial likelihood-only replica-exchange connectivity test with complete ensemble checkpoints. [Results](../docs/validation/studies/tempering_connectivity/RESULTS.md) retain the frozen gate **FAIL**, exchange counts and labeled transport; no cold-posterior convergence claim.

### Configurable prior-bridge refinement (18 November 2026)

[galaxy/tempering](galaxy/tempering/README.md) supports explicit runtime ladders and repeated exchange passes. [The separate eleven-level bridge result](../docs/validation/studies/tempering_bridge/RESULTS.md) retains connectivity gate **PASS** and the original failed six-level benchmark.

- [Four-cold-chain tempering posterior campaign](../docs/validation/studies/tempering_posterior/RESULTS.md): frozen gate **FAIL**, 2/11 monitored coordinate gates fail; exact checkpoint/replay and pause recovery pass.

- [Four-cold-chain tempering posterior campaign](../docs/validation/studies/tempering_long/RESULTS.md): frozen gate **PASS**, 0/11 monitored coordinate gates fail; exact checkpoint/replay and pause recovery pass.

- [Posterior numerical refinement](../docs/validation/studies/posterior_refinement/RESULTS.md): **PASS**, six settings on all 8192 cold states, exact source replay and correlation-aware importance-sensitivity checks.

- [Independent fixed-truth mocks](../docs/validation/studies/independent_mocks/RESULTS.md): two fresh four-chain budgets complete; computational gate **PASS**, descriptive recovery only; no coverage calibration claim.
