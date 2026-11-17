> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Diagnosing the stalled PM reconstruction (#52)

This is a read-only diagnostic of the committed L-BFGS failure, not a new solver,
assignment scheme or broader seed/amplitude/refinement campaign. Same compensated
N32/M16/16-step fit, fixed M16 CIC observation window, fixed nuisance parameters,
and frozen N128/M64 data as the [optimizer comparison](../pm_optimizer_validation/README.md).

## Exact state and search direction

The capture executable observes predictions and returned likelihood cotangents
around the unchanged public optimizer. At each full gradient it copies the accepted
white field and adds the existing white prior to the copied cotangent. It retains
the first subsequent unit-step trial. The failed search direction is reconstructed
as accepted-white minus unit-trial; the first trial reproduces exactly. The
reconstructed direction can differ from the original direction by the rounding
of that subtraction, so extremely small replayed steps are diagnostic rather than
a claim of bitwise replay of every internal trial.

Capture replays the original 83-iteration failure; projected coefficients/gradients
are checked against the committed result. Full compact fields are saved by global
cell ID in rank CSV shards, with parameters, target metadata and projected fields.
The probe reads these bounded validation files on any supported decomposition;
this is not a production checkpoint or changed-layout inference restart API.
Inputs are bounded to N,M=16..64, even meshes and 2..64 steps; the science case is
N32/M16. Shard shape, finite values, complete/unique IDs and metadata are validated
collectively. All ranks need access to the files.

## Derivative and stencil probes

At the stalled point and the original independent small seed-300 starting field,
probe the normalized negative gradient, normalized failed search direction and
seed-11 random active direction. Fourteen h values from 1e-3 to 1e-13 include extra
points around 1e-9. Retain both one-sided slopes and the central difference.
The same earlier FD gates apply to measured same-stencil pairs in 1e-10..1e-5:
best scaled central error <2e-5 and at least two errors <1e-4. All other scales remain
in the report, including changed stencils and roundoff-dominated differences.

`ParticleMeshForwardT::evolve_observed` calls a local read-only callback at every
position/force node, before deposition. All ranks call it in the same order;
callbacks must not call collectives. Callback exceptions propagate collectively.
Use the node index with `settings().scale_factors`; `Result::a` is final-output
metadata, not a time stamp for intermediate callback states. Positions are current;
momentum can be staggered and density/diagnostics are not completed node outputs.
The default
`evolve()` uses a compile-time empty observer, with no added communication or
arithmetic. No force/adjoint law or checkpoint identity changes.

Compare corresponding particle/node/axis CIC base indices with the unperturbed
trajectory. This measures perturbation-induced support changes, not normal
physical crossings between different time nodes. Canonicalize periodic cell M
to cell 0: a wrapped position can round to exactly 1. Track the closest face with
signed unwrapping around the baseline coordinate. Observed and default trajectories
must agree bitwise in each probe.

For fixed noise/parameters, compute stable target differences algebraically:

```text
Delta U = sum_q [white0(q)*Delta white(q) + .5*Delta white(q)^2]
        + sum_k weight(k)/variance * [Re(conj(pred0-data)*Delta pred)
                                      + .5*|Delta pred|^2].
```

Noise normalization and scalar priors cancel. Local accumulation uses the native
long-double type, followed by a double MPI reduction. Forward predictions remain
double precision; this is not a high-precision gravity solution. Record the public
`posterior.value` subtraction as used by the actual line search, plus an
independently assembled raw value for comparison. Probe all 40 original alpha
values, with the adjoint linear prediction and Armijo bound. Raw acceptance includes
rounding of the full-value Armijo RHS; stable acceptance uses the difference directly.

Fresh/reused workspaces and adjoint checkpoint budgets0/1/4/8 test cache/replay
sensitivity. Preserve all diagnostics regardless of the outcome.

## Reproduce

```sh
make -C FML/FieldLevel/pm_stationarity_validation USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_stationarity_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --phase probe --np 4 --mpirun 'mpirun --oversubscribe'
```

The default probe phase uses the retained full snapshot, avoiding another optimizer
replay. `--phase all` captures a new state first; reproduce the original capture
with four pencil ranks and one OpenMP thread. Compiler/FFT/backend changes can
alter a nonsmooth optimizer trajectory, so the published-state probe is the more
direct comparison. Use fresh outputs, suitable compiler/FFTW/launcher overrides,
and separate builds when changing flags. Each subprocess is bounded to 900s by
default and cleaned up as a process group on timeout/interruption. NumPy and
Matplotlib are required; the runner sets writable cache directories.

Published measurements can be reanalyzed without native jobs:

```sh
python3 FML/FieldLevel/pm_stationarity_validation/analyze.py --input FML/FieldLevel/pm_stationarity_validation --output /tmp/fml-stationarity-report
```

See [RESULTS.md](RESULTS.md). Broader physical calibration, joint-parameter
convergence and production scaling are outside this diagnostic.

## Stable-comparison follow-up

The shared capture accepts an optional final `STABLE_CHANGE` flag (0 or 1,
default 0). Stable mode skips the baseline reference prediction before retaining
the first actual trial. The runner's `--stable-changes` selects the
[iteration-98 study](../pm_stable_stationarity_validation/README.md) and its
retained snapshot/reference. The analyzer accepts `--reference` and
`--line-expectation measure` to measure a new endpoint without requiring the
original endpoint's hidden-descent outcome. All original default gates remain.
