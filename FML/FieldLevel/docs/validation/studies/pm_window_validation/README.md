> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Explicit PM CIC force-window experiment (#52)

`PMSettings::force_window` now selects `PMForceWindow::None` (unchanged default)
or the opt-in `CICCompensated` convention. For integer wavevector n and force mesh
M, one CIC window is `W_M(n)=product_i sinc(pi*n_i/M)^2`. Compensation multiplies
the continuous spectral **force** by `1/W_M^2`, accounting for deposition and
force gathering. DC and each derivative's component Nyquist handling are unchanged.
The real even multiplier is included in the reverse force operator too.

This is an explicit numerical model choice. It does not remove aliasing or
particle sampling errors and can amplify short-scale particle/grid noise. No
extra smoothing or high-k cap is silently introduced. It remains opt-in and
requires accuracy/mixing validation for the intended application.

## Separate force and observation windows

The reported density remains the raw CIC mesh density. It is **not deconvolved**.
Consequently changing M changes both the dynamics and the final density window.
The campaign reports two comparisons:

- Raw observed-band density, retaining the prior <5% and <0.1-noise-SD thresholds.
- A descriptive common-window diagnostic: multiply each prediction by
  `W_32/W_M`, smoothing the fine-grid output to the common M32 window. This is
  postprocessing for a controlled density comparison, not a new likelihood,
  posterior target, or an alias-free particle Fourier reference.

The raw score/gradient comparison still uses the actual raw-density likelihood.
Common-window density agreement cannot establish posterior-gradient accuracy.

## Fixed protocol

Use the same strong91 realization, P=.002, 122 initial real directions, 256
observed real degrees, box length 1, Omega_m=.3 and a=.2→1, with the frozen M32
mock/noise from the previous [particle-sampled study](../pm_force_reference_validation/README.md).
For both windows, forward runs use N128/M32/32 steps, N96/M64/32 steps,
N128/M64/32 steps and N128/M64/64 steps. Compare force grids, N96→128 particles
at M64, and time refinement. Recompute both raw projected posterior gradients
at N64/M32 and N64/M64 with 32 steps. These are finite references.

A full-posterior field and mixed field/scalar finite-difference sweep at
N32/M16/16 steps checks the new option. Require best last-four scaled error <2e-5
and at least two last-four errors <1e-4, unchanged from prior campaigns.
All physical initial coefficients must match within 2e-12. Window metadata,
finite supported outputs and conservation are checked. All prior raw accuracy
criteria remain visible (<5% density, <15% projected gradient, <.1 noise SD).

Independent NumPy CIC/full-DFT/gather and native tiny-kick probes use the original
18-mode linear fixture in `linear.initial.csv`, at amplitudes 1e-3/1e-4 and native
kick increments 1e-8/1e-9. Require native/NumPy statistic differences and
shrinking-kick changes <2e-5. At dense N64/M16 and N64/M32, require
compensation to move the linear response closer to 1 and within 1%. The initial compensated probe's coarse 1e-8 kick is underresolved at sparse
N8/M64 (native/NumPy statistic difference 2.29e-4). Retain that failure and add
1e-10: compensated code-agreement acceptance uses 1e-9 and 1e-10, requiring both
to pass the unchanged 2e-5 criterion and their response difference to pass it.
Uncompensated probes retain their original 1e-8/1e-9 acceptance. All raw rows,
coarse-step failures and refinement checks are reported separately. This is
measurement-step refinement, not a force-law change or tolerance relaxation.

The continuum longitudinal response is 1; evaluate
this separately from implementation agreement, especially for sparse loads.

Six forward/gradient runs per window family plus the directional sweep are
bounded explicitly in the runner (eight forward, four full-gradient, one FD run).
No MAP or new posterior chains are run. This is one stronger realization and a
small linear-force control, not general coverage or a proof of continuum accuracy.

## Reproduce

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=true USE_PENCIL_GRID=true
make -C FML/FieldLevel/pm_resolution_validation pm_force_probe USE_MPI=false
python3 FML/FieldLevel/pm_window_validation/run.py --bin-dir /path/to/mpi/binaries --probe-dir /path/to/serial/probe --output /path/to/fresh/OUT --np 4 --mpirun "mpirun --oversubscribe"
```

Use clean/separate builds and suitable compiler/library/launcher overrides.
The explicit campaign uses OMP1, a 300-second per-process budget and process-group
cleanup on timeout/keyboard interruption. Phases `cases`, `probes`, `analyze`
are separate; analysis writes reports and returns nonzero for retained failures.
NumPy and Matplotlib are required. This is outside routine CI.

The shared driver now takes `[FORWARD_ONLY] [FORCE_WINDOW]` after its original
arguments. The optional window flag is 0/1, replicated and validated. A
`.window.csv` sidecar identifies every run; old output schemas remain unchanged.
The force probe and independent NumPy probe also accept an explicit window flag.

Checkpoint identity retains the original law-tag-1 bytes for `None`. The new
compensated convention has law tag 2 and explicit window metadata. Cross-window
restart rejects even at a zero state where saved values/forces coincide. This
preserves old default targets without silently interpreting them as compensated.
See [RESULTS.md](RESULTS.md) for actual measurements, validation and remaining work.
