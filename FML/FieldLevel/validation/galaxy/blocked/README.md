# Blocked field/cached-scalar pilot

This maintained experiment compares a blocked sampler against the preceding
[joint pilot](../joint/README.md). It uses the identical synthetic Gaussian tracer
posterior, dispersed initial states, four chains, 256 discarded warmup sweeps and
1024 retained draws per chain. The [protocol](plan.json) was frozen before running.
It is a validation prototype using existing library kernels, not a new production
sampler/checkpoint API. Exact catalog inputs and physical convergence remain open.

Each outer sweep first applies one field-only HMC transition conditional on the
current scalars: eight leapfrog steps at size 0.15, no jitter or adaptation. The
mass uses the corrected Gaussian metric frozen at that chain's initial scalars.
The scalar block then applies four existing FixedIC joint amplitude/bias
transitions conditional on the new field, each with eight conditional bias-slice
sweeps and a reflected amplitude random walk of width 0.07. Cached sufficient
statistics evaluate the actual fixed-noise Fourier likelihood, not a surrogate.
The basis is rebuilt when the field or amplitude changes. Cheap bias updates
reuse it without another LPT prediction. Uniform physical priors stay unchanged.

Each conditional kernel preserves its own conditional target: Metropolis-corrected
HMC for the field, convex conditional slice updates for each bias and physical
amplitude/bias MH with the normalized proposal-density ratio. Their composition
preserves the joint target; the whole deterministic sweep need not be reversible.
There is no new Gaussian prior penalty or Jacobian in physical scalar coordinates.
The field's prior is constant during a scalar block and cancels there.

The target, primary draw count and gates match the baseline. Integration settings,
block count and warmup adaptation differ; this is not a single-setting ablation
or a matched-compute comparison. Record cost including cache construction, work
allocation and I/O. ESS-per-second ratios from unmixed chains are descriptive,
not calibrated asymptotic speedup factors. Never pool baseline/replay chains.
All eleven monitors must satisfy Rhat < 1.01 and bulk/tail ESS >= 400 with zero
sampling divergences. Stop at the declared budget even on failure.

Build and run from the repository root, adding normal compiler/FFTW settings:

```sh
make -C FML/FieldLevel/validation/galaxy/blocked blocked_pilot blocked_diagnostics USE_MPI=false
python3 FML/FieldLevel/validation/galaxy/blocked/run.py /absolute/new/output
python3 FML/FieldLevel/validation/galaxy/blocked/analyze.py /absolute/new/output \
  --diagnostics /absolute/path/to/blocked_diagnostics --baseline /absolute/joint/output
```

The runner supports an external build via `--exe PATH` and independent serial
chains (`--workers 2` by default). Analysis reuses the joint pilot's exact monitors,
library diagnostics, plot and residual conventions. The `joint.*` analysis files
inside this experiment are local derived data, never pooled with the baseline.

The validation executable accepts `DIRECTORY start|resume CHAIN [sweep_budget]`.
A new `start` with budget zero checks 18 full-posterior/cache differences and 72
bias finite differences. The zero-budget check performs no Markov transitions.
The preflight also compares four uninterrupted sweeps against two plus two in
fresh processes, and tests wrong-chain and checksum-corrupt rejection.

The serial-only checkpoint contains field, scalars, completed sweep count and
cumulative counters in a versioned, checksummed codec payload, atomically renamed
to CURRENT. Target/build/plan/chain identity is checked. Random streams are keyed
by chain, completed outer index and block: `53000+100000*chain+2*index`, plus one
for the scalar block. No partially consumed stream crosses an outer boundary.
Restart rebuilds scratch caches and uses the same fixed metric snapshot. This
format is distinct from the production HMC checkpoint format; formats are never
silently interchanged. It is not power-loss durability validation.

Checkpoints occur every 64 sweeps and at chunk completion. Indexed draw records
are atomically replaced on replay. summary.json counts committed draws; orphan
files do not count. Rerun the same runner command/executable to resume. A branch
of chain 0 at sweep 384 resumes for 64 sweeps and must reproduce retained draw
bytes and numerical summary; these duplicate draws are excluded. Keep all raw
outputs outside the library. Existing separate MPI tests cover the underlying
kernels; this serial composition does not establish MPI blocked-driver support.

The failed comparison motivates a separately tested
[amplitude/field interweaving proposal](NEXT_STEPS.md). This is a proposed next
kernel with explicit Jacobian and validation requirements, not an extension of
the recorded experiment.
