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
