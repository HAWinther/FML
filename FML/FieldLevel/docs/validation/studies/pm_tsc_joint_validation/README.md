> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Joint TSC reconstruction (#52)

A single bounded joint MAP of the initial white field, log power amplitude A,
and log noise power N. Use the frozen matched TSC N128/M64/32-step data from the
[fixed-parameter experiment](../pm_tsc_validation/RESULTS.md), with the same
N32/M16/16-step fit, compensated TSC, M16 CIC observation window, physical
initial modes, seed-300/.05 start and fixed flat cosmology. The public posterior
and options MAP API are unchanged: history-5 stable-change L-BFGS, cap 128,
`vary_parameters=true`, numerical tolerance 2e-5.

The joint stopping norm is the maximum of the raw field gradient and scalar
`abs(g_j)*Parameter::scale()`. Gaussian scalar scales are .5. The numerical
tolerance is unchanged, but now applies to both field and scalar coordinates.
The driver reports each norm separately and checks a fresh full endpoint replay.

## Priors and interpretation

The Gaussian means and sigmas are A=0±.5 and N=log(1e-4)±.5. This validation
explicitly widens the amplitude support from [-1,1] to **[-4,4]**; noise support
remains [-16,-4]. This avoids clipping the possible interior joint mode. It is a
declared change to the numerical/prior domain, not an automatic production change.
No sampler, library target or optimizer is changed by this study.

A joint density mode in whitened coordinates need not recover the injected scalar
parameters. The physical initial density scales as exp(A/2) times the white field.
At fixed physical density, increasing A can reduce the white-field prior cost.
The normalized likelihood and priors give the exact gradient identities

```text
g_A = .5 * (dot(s, g_s) - dot(s,s)) + A/.5^2
g_N = .5 * (D - chi2) + (N - N0)/.5^2
```

Here `g_s` includes the white prior, D is the global full-Hermitian weighted mode
count and chi2 uses the fitted noise variance. These are tested independently at
truth, start and endpoint. At an exactly stationary field and amplitude,
`A=dot(s,s)/8`: an increased amplitude is expected for this joint mode. It is not
an unbiased amplitude estimate. Marginalized parameter inference needs posterior
sampling/integration and calibration, which remain outside this bounded test.
Recovery plots therefore use the **physical density with the fitted amplitude**,
not the raw fitted white coefficients alone.

## Run and reproduce

```sh
make -C FML/FieldLevel/pm_tsc_joint_validation USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_tsc_joint_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --np 4 --mpirun 'mpirun --oversubscribe'
```

Use appropriate FFTW/compiler/launcher overrides and separate builds. NumPy and
Matplotlib are needed for analysis. Each child has a 900s default deadline with
process-group cleanup; `--phase cases` and `--phase analyze` separate native work
and reporting. The native CLI is `pm_tsc_joint PREFIX DATA N M STEPS ITERATIONS
[VARY] [FD]`; VARY and FD default to 1. Meshes are bounded even 16..64, time steps
2..64, and optimizer iterations 0..200. VARY=0 supports fixed-target compatibility
checks. Every replicated input must agree across MPI ranks.

The parameter path is recorded at full-gradient evaluations; optimizer counters
select the accepted points, including after any rejected gradient evaluations.
No full field is copied for this trace. Out-of-support proposals are counted
separately: they use a value request without a prediction. Actual forward counts
include accepted gradients and reference predictions. Central differences use
stable changes at start/endpoint in white, scalar and combined directions, over
seven scales. Retain the existing refined gates: best error <2e-5, at least two
errors <1e-4 in the final four scales. All native statuses/failures are retained.

```sh
python3 FML/FieldLevel/pm_tsc_joint_validation/analyze.py --input FML/FieldLevel/pm_tsc_joint_validation --output /tmp/fml-tsc-joint-report
```

See [RESULTS.md](RESULTS.md). This is one finite reference and realization, not a
seed/amplitude/refinement campaign, survey extension, posterior coverage test or
evidence for general joint parameter recovery.
