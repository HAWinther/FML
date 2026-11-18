# Bounded joint Eulerian field/amplitude/bias pilot

This is the next synthetic free-IC sampler gate after the FL-7 field-only probe.
It samples the whitened initial field together with `sigma8_ratio`, `b_delta`,
`b_delta2`, `b_K2` and `b_laplacian`, with fixed noise and spectrum shape.
The amplitude is sigma8 divided by the reference-spectrum sigma8; this fixture
does not specify an absolute reference sigma8. Priors are uniform on
[0.5,1.5], [0,4], [-4,4], [-4,4] and [-100,100], respectively.

The [frozen plan](plan.json) declares the same N4/N8/N16 PSWF target used for the
metric probe, four dispersed scalar/field starts, 256 warmup and 1024 retained
draws per chain. Step size and scalar diagonal mass adapt during warmup only;
the corrected field metric stays at each chain's initial scalar snapshot.
Trajectory lengths jitter according to the existing HMC convention. This is a
new joint campaign, not an extension or pooling of the field-only experiment.

Every monitored coordinate must have computable diagnostics, rank-normalized
split/folded Rhat < 1.01 and bulk/tail ESS >= 400; all sampling trajectories must
have zero divergences. Monitors are all five scalars and the six predeclared
cosine/sine initial-field projections. Passing these gates does not establish
all-coordinate convergence, posterior calibration, physical resolution convergence
or real-catalog fidelity. Failure is recorded without extending the draw budget,
retuning, dropping chains or relaxing gates.

Build and run from the repository root (supply normal FFTW/compiler settings):

```sh
make -C FML/FieldLevel/validation/galaxy/joint joint_pilot joint_diagnostics USE_MPI=false
python3 FML/FieldLevel/validation/galaxy/joint/run.py /absolute/output/directory
python3 FML/FieldLevel/validation/galaxy/joint/analyze.py /absolute/output/directory \
  --diagnostics /absolute/path/to/joint_diagnostics
```

Builds can be kept outside the library and passed with `run.py --exe PATH`.
The declared campaign runs independent serial chains, two at a time by default.
MPI slab/pencil forward/adjoint/metric and restart behavior are covered by the
existing separate regression suites; this campaign does not measure MPI scaling.

The runner saves checksummed native checkpoints every 64 trajectories and on
chunk completion, and records indexed JSON draws atomically. Rerun the identical
command to resume after interruption: it refuses a changed executable or plan.
An interrupted chunk can replay from its most recent checkpoint; uncommitted draw
files are overwritten. Readers use summary.json's committed count. This is ordinary
process-restart support, not a tested power-loss/filesystem durability guarantee.
At chain 0 trajectory 384 the runner forks a checkpoint, resumes both copies for
64 trajectories and checks the raw retained draw bytes and numerical summary.
Replay draws are never included in diagnostics. The original draw budget stays
unchanged. For a pause, stop the runner and its running child processes; resume
with the same command and executable.

`analyze.py` uses the library's existing diagnostic implementation, verifies all
four data targets match, and produces traces, provisional scalar distributions,
scalar correlations and binned fitted-data residual powers. The residual bands
are variation across posterior forward predictions; they do not include newly
sampled observational noise and are not an independent predictive test. The
noise-power line is a reference, not a goodness-of-fit acceptance threshold.
A short same-model mock cannot measure coverage. If gates fail, pooled quantiles
and correlations remain descriptive diagnostics, not certified posterior results.

Raw chains/checkpoints, build provenance and figures belong outside the library.
The durable report is stored under docs/validation/studies/joint_pilot/.
