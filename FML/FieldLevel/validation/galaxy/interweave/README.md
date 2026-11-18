# Amplitude/field interweaving pilot

This is a separate serial validation campaign adding one block to the preceding
[blocked sampler](../blocked/README.md). It preserves the target, initial fields
and scalars, field/scalar kernels and their component seeds, four chains, 256
discarded warmup sweeps, 1024 retained draws and eleven-coordinate convergence
gates. The [protocol](plan.json) is frozen before execution. Earlier primary and
restart-replay draws are not pooled. Stop at the fixed budget even on failure.

After each unchanged field-HMC/cached-scalar sweep, four amplitude/field proposals
use a reflected Gaussian amplitude increment of width 0.10 on [0.5,1.5]. Biases
stay fixed; all stored white coordinates rescale together:

```
r = alpha / alpha_new
white_new = r * white
log_acceptance = U_old - U_new + N * log(r)
```

U is the full existing posterior, including actual field/scalar priors and
likelihood. The reflection kernel is symmetric in physical alpha. The reversible
extended map `(alpha, white, alpha_new) -> (alpha_new, r*white, alpha)` has absolute
Jacobian `r^N`, giving the positive `N*log(r)` term. This is a proposal correction,
not a new posterior prior/penalty. A nonsymmetric amplitude proposal would need
its reverse/forward density ratio too. Rejections retain the current state.
Numerically unrepresentable candidate targets reject; invalid current states fail.

All 64 stored real coordinates rescale, including DC, Nyquist and other prior-only
directions. Therefore N=64, not the count of active or observed Fourier modes.
The independent tests include unused coordinates to detect a missing Jacobian
contribution. Active-only rescaling is not implemented. A future distributed
version must use the global dimension and collective validation/reductions.

For amplitude-only LPT, alpha*white determines the physical initial density, so
this direction should preserve forward predictions. Shared/compact 1LPT/2LPT
fixtures check that property. The implementation still evaluates full U at each
candidate; it does not assume that numerical likelihood differences vanish.
The physical target and normalized white prior are unchanged. Field mass remains
frozen at the chain's original scalar snapshot. Existing production APIs are not
changed; the pure serial proposal helper lives in this validation directory.

Tests use analytic Gaussian marginals integrated independently in alpha, with
uniform/truncated-Gaussian amplitude priors, one/three white dimensions, observed
and unobserved likelihoods, and unused Gaussian directions. Exact conditional
Gaussian field refreshes make those toy tests ergodic; a rescaling-only chain would
keep alpha*white fixed and cannot alone sample the full joint posterior. Reciprocal
transforms, signed Jacobian, reflection, malformed proposals and nonfinite targets
are also checked. The cosmology preflight retains the full-posterior/cache and
bias-gradient checks from the blocked driver.

Build/run from the repository root with normal compiler/FFTW flags:

```sh
make -C FML/FieldLevel/validation/galaxy/interweave interweave_pilot interweave_diagnostics interweave_test USE_MPI=false
FML/FieldLevel/validation/galaxy/interweave/interweave_test
python3 FML/FieldLevel/validation/galaxy/interweave/run.py /absolute/new/output
python3 FML/FieldLevel/validation/galaxy/interweave/analyze.py /absolute/new/output \
  --diagnostics /absolute/path/to/interweave_diagnostics --baseline /absolute/blocked/output
```

Use `run.py --exe PATH` for an external build. Two serial chains run concurrently
by default. The versioned checksum checkpoint format adds rescaling counters and
a new target/plan/chain identity; older pilot checkpoints are rejected. New RNG
streams use `930000 + 100000*chain + outer_index`, independent of the unchanged
field/scalar component streams. Four attempts consume each stream entirely within
an outer sweep, so no partial stream crosses a saved boundary. Rerunning the same
command resumes from committed boundaries. Indexed draws replay deterministically;
summary.json counts committed draws. The planned independent 64-draw continuation
is excluded from diagnostics. Ordinary process restart is checked, not power-loss
durability or production MPI interweaving support.

Analysis uses the shared joint-pilot monitor definitions/library diagnostics.
`joint.*` derived filenames are local to this new campaign. Comparison validates
that physical inputs, original component settings and gates match the blocked
baseline. Added full-posterior evaluations cost time: report measured main-chunk
cost rather than claiming a matched-compute experiment. Unmixed-chain ESS/time
ratios are descriptive, not asymptotic speedup factors. Scalar quantiles remain
provisional if any gates fail; residual bands exclude fresh observational noise.
A small same-model pilot does not validate coverage, physical refinement, all
field coordinates or real catalogs. Keep outputs/checkpoints/figures outside the
library; retain only the durable study report under docs/validation/studies/.

On runner restart, a zero-transition native resume first refreshes summary.json
from the authoritative CURRENT payload. This handles interruption after checkpoint
publication but before the summary write, before deciding the next chunk/replay
boundary. A dedicated stale-summary fixture verifies repair without new transitions
or extra draw records. The original uninterrupted campaign does not enter this
restart-only path; its sampler/native binary is unchanged.
