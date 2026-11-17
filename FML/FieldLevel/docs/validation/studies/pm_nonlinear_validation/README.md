> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Many-mode PM accuracy and refined-data reconstruction

This explicit #52 validation campaign expands the earlier 18-mode experiment to
122 real initial Fourier directions (integer squared wavenumber 1..9). The observed
band has 256 real directions (squared wavenumber 1..16). Box length is 1, flat
Omega_m=0.3, ZA initialization at a=0.2, Gaussian noise power 1e-4 and no bias/RSD.
It uses production PM predictions, adjoints, posterior and MAP without changing
the numerical force law or sampler.

Three fixtures separate a change in strength from a change in realization:

| Fixture | Initial white coefficients | P | Final a |
| --- | --- | --- | --- |
| weak91 | seed 91 | 0.0005 | 0.6 |
| strong91 | seed 91 | 0.002 | 1 |
| strong173 | seed 173 | 0.002 | 1 |

Initial power is extrapolated to a=1. These amplitudes and final times are
stress-test controls, not a claim of a validated shell-crossing regime. The actual
final mesh-density RMS is recorded. Fourier coefficients are matched physically
across N, with canonical wavevector-keyed counter RNG. These are new
realizations: the RNG key/stream convention differs from the earlier 18-mode
study, so seed 91 does not preserve that study's low-mode coefficients. A basis is generated one
pair at a time; no active-modes times global-cells basis array is retained.

Every fixture generates one noisy dataset at N=64/M=32/32 steps, then reuses it
for all comparisons and fits. All reference mocks use noise seed 92; the three
fixtures are not independent noise realizations. This is a finite refined numerical reference, not
independent external physics or an exact continuum solution. Data files include
target identity (seed, power, final time, bands, noise); incompatible data reject.

## Declared checks

- Matched physical initial Fourier coefficients within 2e-12; finite supported
  outputs and 124 projected derivatives (122 field and two scalars).
- Particle refinement N=32/48/64 at M=16/16 steps, force comparison M=32/64 at
  N=64/32 steps, time comparison 32/64 steps at N=64/M=32.
- Retain previous finest-correction thresholds: density <5%, projected posterior
  gradient <15%, discrepancy <0.1 noise SD per observed degree. References are
  finite and separate for each family; their own gradients use the same data.
  M64 has only one particle per force cell at N64; further particle refinement
  there is needed before calling it a continuum reference.
- Field and mixed field/scalar directional finite differences at N=32/M=16/16
  steps for weak91 and strong91. Best last-four scaled error <2e-5 and at least
  two of those <1e-4. Multiple refined steps must agree.
- Fixed-parameter reconstruction from refined data: weak91 at N=16/32, strong91
  at N=32 (all M=16/16 steps), with 12 MAP iterations and raw cell gradient
  tolerance 2e-5. Report optimizer status, projected gradient, initial coefficient
  error/field correlation and data residual. A lower objective alone does not establish
  convergence. Accepted recovery requires relative initial coefficient error <0.8
  of the zero-field baseline and initial-field correlation >0.7. These are bounded
  optimization diagnostics, not posterior coverage.

The two original weak-field 80-iteration attempts did not finish within 455
seconds and were interrupted; a briefly queued strong-field attempt was also stopped. Their logs/partial outputs are retained in the local review artifact.
Only the work budget was reduced to 12; convergence/recovery thresholds are
unchanged. This budget revision is a runtime decision, not evidence of convergence.

All fits start from seed-300 active coefficients scaled by .05. The same full-grid
production optimizer is used; absent physical modes start at zero and remain prior
null. Raw grid-gradient tolerances have resolution-dependent meaning, so the
projected 122-dimensional gradient norm is also reported. Initial-field correlation
is the normalized coefficient dot product, equal to the zero-mean physical-field
correlation in this orthonormal basis. Coefficient RMSE is
relative to the zero-field baseline; no noisy truth is used to initialize fits.

## Reproduce

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=false
python3 FML/FieldLevel/pm_nonlinear_validation/run.py --bin-dir /path/to/binaries --output /path/to/fresh/OUT --jobs 2
```

Use normal FML compiler/FFTW overrides and separate builds when changing flags.
NumPy and Matplotlib are needed. The campaign is outside routine CI. Phases
`cases`, `fits`, `analyze` support separate runs. Analysis always saves measured
failures and plots, returning nonzero when declared numerical criteria fail.
Use `RESULTS.md` for measured outcomes and actual backend/runtime scope.

Native interface:

```text
pm_nonlinear PREFIX N M STEPS SEED POWER FINAL_A DATA_PATH FD FIT_ITERATIONS [FORWARD_ONLY] [FORCE_WINDOW] [CIC_REFERENCE] [OPTIMIZER] [HISTORY]
```

By default, N/M are even 16..64; steps 2..64; P in (0,.01]; final a in (.2,1]; FD is 0/1;
MAP budget 0..200. `DATA_PATH=-` generates data. The common Fourier export uses
16x16x9 half-complex indexing with interleaved real/imaginary components. This
bounded replicated output is for validation, not scalable production storage.
MPI controls and output paths are checked collectively before model construction.

A separate [particle-sampled force-reference study](../pm_force_reference_validation/README.md)
uses `FORWARD_ONLY=1`, permitting N<=128 with FD=FIT_ITERATIONS=0 and external
compatible data, or `DATA_PATH=-` to generate a new mock without an adjoint. It
computes forward predictions and a value-only posterior
without requesting a pullback. It emits `.forward_meta.csv` and no gradient file;
the original default-mode output schema and N<=64 bound are unchanged. This
extension does not change or overwrite the original measurements above.

`FORCE_WINDOW=0` retains the original numerical law; `1` selects the opt-in
CIC-compensated force. It requires the preceding forward-mode argument (0 for
full gradients). Every run writes a `.window.csv` sidecar; old output schemas
stay unchanged. The [separate window campaign](../pm_window_validation/README.md)
reports new measurements without overwriting this historical default-law study.

### Optional fixed observation window

The final optional CLI argument after `FORWARD_ONLY` and `FORCE_WINDOW` is
`CIC_REFERENCE` (zero by default). Positive values select the fixed coarser CIC
window; data headers must explicitly match `observation_cic,REFERENCE`, and
`.observation.csv` records it. Legacy raw data and output bytes remain unchanged
when the reference is zero. See [the response study](../pm_observation_validation/README.md).

### Refined-data compensated reconstruction

[The fixed-observation reconstruction study](../pm_reconstruction_validation/README.md)
generates a compensated N128/M64 mock in forward-only mode and fits it on a
coarser discretization with the same declared observation window. Fits also emit
`.fit_start_modes.csv`, the starting prediction before optimization, for residual
comparisons. Existing numerical outputs and the public MAP algorithm are unchanged.

### Optional optimizer comparison

Arguments after CIC_REFERENCE are OPTIMIZER (0 legacy, 1 value-only GD, 2 L-BFGS)
and HISTORY (1..32, default5). Existing numerical files remain unchanged under
the default. Fits also write `.fit_optimizer.csv` with MAP-only actual call/time
and history/RSS diagnostics; options-based methods write `.fit_trace.csv`. See
[the comparison protocol](../pm_optimizer_validation/README.md).

Shared private validation helpers now live in `Validation.h`, used also by the
[stalled-state diagnostic](../pm_stationarity_validation/README.md). The default
native numerical outputs remain bitwise unchanged.

The optional final STABLE_CHANGE argument (0/1, default 0) enables stable objective
changes for OPTIMIZER=1/2. Extra reference-prediction counts and accepted-change
traces are written to `.fit_change.csv` and `.fit_change_trace.csv`. See
[stable Armijo validation](../pm_stable_armijo_validation/README.md).

The final optional argument after `STABLE_CHANGE` is `ASSIGNMENT` (2=CIC default,
3=TSC). `FORCE_WINDOW=2` selects matching TSC compensation; nonzero compensation
must match the assignment. TSC writes an additional `.assignment.csv` and declares
`assignment_tsc` in its data identity. The [TSC study](../pm_tsc_validation/README.md)
generates and fits matched TSC mocks. Existing CIC calls and numerical schemas
remain unchanged.
