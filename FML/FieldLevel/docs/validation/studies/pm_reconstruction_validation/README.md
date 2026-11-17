> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Refined-data reconstruction with a fixed PM observation window (#52)

Generate one new strong91 mock using compensated forces at N128/M64/32 steps,
then fit it at N32/M16/16 steps. Both models observe the same M16 CIC window;
noise is added after the response. This is the reconstruction task: broader
seed/amplitude/refinement checks remain explicitly deferred.

Box 1, Omega_m=.3, a=.2→1, P=.002 extrapolated to a=1, 122 initial real directions,
256 observed real degrees, noise power 1e-4 and seed 92. Fixed nuisance parameters.
Initialize from independent seed-300 active coefficients scaled by .05, not truth.
Use the unchanged public full-grid MAP optimizer, gradient tolerance 2e-5.

Fits with budgets 1,4,12,24 each restart from the identical initial state. They
are deterministic iteration-budget endpoints, not a recorded per-line-search
trajectory. Report objective, raw cell max gradient and projected active-gradient
norm, accepted iteration count/status, relative coefficient error against a
zero-field baseline, physical initial-field correlation and residual noise units.
Recovery thresholds remain error <.8 and correlation >.7 from the preceding study.
Convergence requires status Converged; recovery alone is not convergence. Retain
failed criteria and save reports before returning nonzero.

The reference is a finite, better-sampled discretization, not independent physics
or a continuum truth. This is one fixed-parameter reconstruction, not posterior
coverage, uncertainty calibration or a global MAP proof. The noise realization
is frozen across fit budgets. No optimizer conditioning is changed here.

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_reconstruction_validation/run.py --bin-dir /path/to/binaries --output /fresh/OUT --np 4 --mpirun 'mpirun --oversubscribe'
```

Separate/clean builds when changing flags. Override compiler, FFTW and launcher
for your environment. NumPy/Matplotlib required. Each child is bounded to 600s by
default with process-group cleanup on timeout/interruption; exact commands and
runtimes are recorded. `--phase cases` and `--phase analyze` can run separately.
This explicit study is outside CI. See RESULTS.md for measured scope and outcomes.

The initial N64/M32 fit needed 168 seconds for one accepted iteration. Its result
and the interrupted four-iteration attempt are retained in review artifacts.
The bounded campaign therefore uses N32/M16 (eight particles per force cell),
with a new genuinely generated M16-window mock, keeping all criteria unchanged.

## Reanalyze the retained measurements

The study directory retains `science/` exports and `plan.json`, so the
report can be regenerated without rerunning native trajectories:

```sh
python3 FML/FieldLevel/pm_reconstruction_validation/analyze.py --input FML/FieldLevel/pm_reconstruction_validation --output /tmp/fml-pm-reconstruction-report
```

Exit code 1 retains a failed declared criterion; inspect `summary.json` rather
than interpreting it as missing output. The measured summary reproduces exactly
from the retained inputs.
