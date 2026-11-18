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
