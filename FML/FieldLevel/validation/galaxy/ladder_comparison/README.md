# Bounded cold-end temperature-ladder comparison

Dataset 203 was selected post hoc because the original calibration pilot had
severe field mixing failures without divergences and weak cold-end exchange.
This is a targeted sampler diagnostic, not independent mock calibration.

Both cohorts use the existing counted native tempering driver and stable field
step 0.075 with 16 leapfrog steps, integration time 1.2 and the original frozen
per-ensemble metric. The baseline has the original eleven replicas. The refined
ladder adds beta=sqrt(0.5) and sqrt(0.125), giving thirteen replicas. The positive
betas, actual prior-refresh endpoint, four exchange passes, cold target, data,
scalar/rescaling kernels, priors and all twelve diagnostic/transport/divergence
gates are otherwise fixed. Only the likelihood is tempered.

Each cohort has four fresh chains of 256 discarded warmup sweeps and 512 retained
cold draws. Equal cold-draw budgets are not equal work: twelve versus ten positive
replicas increases field work by 20%. Report actual primary-chain total/sampling gradient calls,
active cohort wall time, per-coordinate ESS/cost and all exchange/qualified-trip
diagnostics, even on failure. Failed-chain ESS and intervals remain provisional.
No extension, exclusions, pooling or retuning is permitted within this experiment.

Both cohorts reuse original mock 203's data-generation, initialization and sampler
seed domains in separate output namespaces. Streams are keyed by replica index;
after insertion the common beta values do not all receive matching streams. These
cohorts are not independent calibrations or exactly paired noise at common beta.
All four cold targets must match the original dataset byte-for-byte.

## Controls and execution

The native driver already supports arbitrary validated ladders. No production
kernel or public API changes are introduced. Native preflight compares complete
13-replica checkpoints/summaries/draws after uninterrupted and fresh-process split
runs; checks target identity against the old ladder; checks exact measured gradient
counts including warmup separation; and rejects changed beta values/count on resume
without writes. Existing integrator/native Gaussian and target controls remain in
force. All chains also perform the existing full-ensemble checkpoint replay.

```sh
python3 preflight.py /absolute/empty-controls --exe /absolute/tempering_pilot
python3 study.py prepare /absolute/empty-comparison --source /absolute/pilot/mock203 --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
python3 study.py run /absolute/comparison --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics --max-rounds 1
python3 study.py run /absolute/comparison --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics --resume
python3 study.py analyze /absolute/comparison --exe /absolute/tempering_pilot --diagnostics /absolute/calibration_diagnostics
```

Preparation hashes all original 8192 cold records, plans, summaries, checkpoints,
runtime sources, binaries and selected comparison plans before sampling. Sources
and original inputs are verified again on resume and analysis. The study lock is
inherited by cohort runners; native children retain their cohort writer lock. Each round launches at most four native
workers per cohort, eight total, and advances one checkpoint chunk per ensemble.
SIGINT/SIGTERM or a PAUSE marker drains active chunks; resume with --resume.
--max-rounds bounds a run without changing the total declared budget. A cumulative
four-hour sampling cap uses 2400-second conservative reservations. Crashed batches
retain their charge; analysis with a lost cost receipt does not certify wall-time
efficiency. Missing budget state beside existing sampling work rejects resume.

Active cohort seconds include initialization, replay and shared-hardware contention,
but exclude waiting between rounds for the other cohort. Primary gradient counts
exclude auxiliary validation replay, which is included in wall timing. No dedicated-machine
speedup is inferred. Better exchange or more trips alone does not certify mixing.
Raw chains, binaries, logs, dense plots and full hash inventories stay outside the
library. See [results](../../../docs/validation/studies/ladder_comparison/RESULTS.md).
