# Configurable likelihood-tempered connectivity pilots

This maintained serial prototype tests replica exchange for the synthetic periodic
Eulerian field/amplitude/bias problem. It is not a production sampler/MPI API or a
posterior-convergence campaign. Production model and sampler headers are unchanged.
Only beta=1 records are cold posterior states; hot and replay states are not pooled.

Two separate fixed protocols are retained:

- [Original six-level plan](plan.json): beta=[1,1/2,1/4,1/8,1/16,0], one even/odd
  exchange pass per sweep, two ensembles, 64 warmup and 256 measured sweeps. The
  original [failed benchmark](../../../docs/validation/studies/tempering_connectivity/RESULTS.md)
  is retained with its archived executable and runner.
- [Prior-bridge plan](plan_bridge.json): powers of two through 1/512, then beta=0
  (11 replicas), four alternating exchange passes per sweep, same two ensembles,
  warmup, measured budget, physical target, local kernels and numerical gates.
  Both ladder and exchange frequency change; this is not an isolated ladder or
  matched-compute comparison. No gate is relaxed if transport remains poor.

`run.py --plan PATH` selects the declared protocol and writes its explicit numeric
ladder to the output directory. The native CLI accepts
`DIRECTORY start|resume ENSEMBLE [sweep_budget [ladder_file [exchange_passes]]]`.
A ladder file contains only whitespace-separated beta values. Values must strictly
decrease from 1 to 0, with 2–64 replicas and 1–8 exchange passes. The fixed model,
local kernel settings, seeds, 64/256 schedule and chunk size must match the compiled
fixture; the runner rejects plan edits to those controls. This is a configurable
ladder interface, not a general model-parameter-file interpreter.

## Targets and local moves

The family is `exp[-U_prior-beta*L]`: priors and observed data stay unchanged.
At positive beta, fixed noise power 1000/beta gives beta times the original
likelihood quadratic; its Gaussian normalization differs only by a replica-specific
constant. This equivalence requires fixed noise/shared modes. It cannot be assumed
for inferred noise parameters or another likelihood family. beta=1 is the original
N4/N8/N16 posterior. All replicas see the same original noise-1000 mock data.

Each positive-beta slot uses the previous field-HMC/cached-scalar/rescaling sweep,
with matching noise in its bias cache and frozen field metric. The original scalar
reference and counters belong to the temperature slot. Momenta refresh per move.
At beta=0, refresh standard-Gaussian white coordinates and all bounded-uniform
scalars directly from the proper prior, then evaluate original L for exchanges.
Unsupported scalar prior families reject explicitly. No infinite-noise posterior
or flattened white prior is constructed.

After local updates, pass p attempts disjoint neighbors with parity=(outer+p)%2:

```
log_accept_swap = (beta_i-beta_j) * (L_i-L_j)
```

Exchange complete physical fields/scalars, original likelihood values and walker
labels. Temperature-slot metrics/counters stay fixed; scratch caches rebuild before
the next update. Repeated passes use the same sweep-keyed RNG stream. Each pass
preserves the product tempered target. The native test checks analytic correlated
Gaussian field/scalar means, variances and covariance at all six legacy levels
and all eleven bridge levels under repeated exchanges.

## Transport and gates

Every edge must exchange at least once per ensemble, pooled edge acceptance must
be >=0.05, each ensemble must complete a measured cold→prior→cold round trip, and
all measured positive-beta field divergence counts must be zero. Visits before
the measured window do not count. These gates are connectivity diagnostics, not
posterior Rhat/ESS, all-field convergence, evidence or coverage success criteria.

Repeated exchanges could bounce at the prior endpoint without a prior refresh.
For multiple passes, phase 2 is credited only when that walker actually refreshes
in the beta=0 local update. A return to cold after that refresh completes a trip.
The trace records the refreshed label and every pass's events/labels. Analysis
independently reconstructs this stricter counting. For one pass, every completed
prior visit already includes a refresh before departure; original bookkeeping and
serialized output remain byte-compatible. Labels track stochastic walkers, not
an unchanged physical configuration.

## Running and restart

Build with normal compiler/FFTW settings from the repository root:

```sh
make -C FML/FieldLevel/validation/galaxy/tempering tempering_pilot tempering_test USE_MPI=false
FML/FieldLevel/validation/galaxy/tempering/tempering_test
python3 FML/FieldLevel/validation/galaxy/tempering/run.py /absolute/new/bridge-output \
  --plan FML/FieldLevel/validation/galaxy/tempering/plan_bridge.json
python3 FML/FieldLevel/validation/galaxy/tempering/analyze.py /absolute/new/bridge-output \
  --baseline /absolute/interweave/campaign --previous /absolute/six-level/campaign
```

Use `--exe PATH` for an external build. Two serial ensembles run concurrently by
default. A zero-budget new native start verifies all positive-beta target/cache
contexts and actual beta=0 prior moments without Markov transitions. The archive
holds original six-level sources/binary: its benchmark is never continued by the
new campaign. The generalized native executable does separately reproduce retained
six-level continuation byte-for-byte, including draws, from an old saved prefix.

The codec saves all replica states/likelihoods/labels, local and exchange counters,
phases and completed trips every 16 sweeps and at chunk end. Allocation/read bounds
and seed separation reflect the configured size. Canonical beta values and pass
count join the target fingerprint for nonlegacy layouts. A changed ladder or
schedule rejects the saved state before continuation, even if replica count matches.
The original six-level/one-pass identity and wire format remain compatible.

The runner fingerprints the binary and selected plan, checks its generated ladder
file on resume, and refreshes summaries from authoritative CURRENT state with
zero transitions before scheduling work. Native resume rechecks all original
likelihoods. An independent copy at sweep 96 resumes for 16 sweeps and must match
the complete ensemble payload, summary and cold records; duplicate replay draws
are excluded. Indexed record publication overwrites any uncommitted orphan on
replay. This is process-restart support, not power-loss durability testing or a
production MPI ensemble checkpoint implementation.

## Analysis and scope

Analysis verifies the physical data target against earlier pilots, cold likelihoods
against predictions, all pass-level label moves, aggregate counters and refreshed-
prior transport. It supports legacy and configurable layouts and reproduces the
original six-level benchmark statistics on a separate copy. Plots show exchanges,
walker transport, cold amplitudes and energy overlap. Original source/binary/plan
hashes, raw data and plots stay outside the library. Only durable reports belong
under docs/validation/studies/.

The exploratory prior-weight ESS calculation uses reconstructed pre-swap prior
likelihoods. These are finite-sample overlap diagnostics, not posterior weights,
exchange-rate predictions or modified gates. No failed ensemble is dropped.
A longer four-cold-chain posterior campaign must be declared separately after
measured connectivity. Physical refinement, calibration, catalog inputs and
production MPI replica layout remain future stages.

## Four-cold-chain validation

`plan_posterior.json` freezes four ensembles, 256 discarded warmup sweeps and
1024 retained cold draws per ensemble. The eleven-replica ladder, four exchange
passes, physical target and local kernels are unchanged from the bridge pilot.
Five scalar and six fixed field projections must each satisfy rank-normalized
split/folded Rhat < 1.01 and bulk/tail ESS >= 400, in addition to connectivity and
zero measured positive-temperature field divergences. Only cold draws enter
posterior diagnostics. Stop at the frozen budget even if these gates fail.

The optional eighth native argument is a whitespace settings file containing
`warmup samples ensembles seed_layout`. Warmup must be even, total sweeps at most
1,000,000, and ensembles between one and four. Layout 1 preserves the original
64/256 two-ensemble trajectories and checkpoint identities. New schedules use
layout 2: `0x5200000000000000 | ensemble<<30 | replica<<24 | sweep<<4 | purpose`.
This is an injective seed-tuple encoding within the supported bounds. Purposes
0/1/2 select field/scalar/rescaling kernels; prior refresh uses 0/1. Exchanges
reserve replica 63, purpose 3. Initial conditions retain seed `4103+ensemble`.
All four scalar starts, schedule and seed layout enter checkpoint identity.
The historical native `connectivity_only` marker indicates that native summaries
alone make no posterior-convergence claim; `posterior.py` applies the cold gates.

Build `tempering_diagnostics` as well as the existing targets, then run:

```sh
python3 run.py /absolute/output/campaign --exe /absolute/build/tempering_pilot \
  --plan plan_posterior.json --workers 4
python3 analyze.py /absolute/output/campaign --baseline /absolute/joint/campaign
python3 posterior.py /absolute/output/campaign \
  --diagnostics /absolute/build/tempering_diagnostics
```

The runner checkpoints complete ensembles every 16 sweeps. Send SIGINT/SIGTERM
to the Python runner, or create `campaign/PAUSE`, to stop after active chunks
finish. Children have separate process groups so a terminal interrupt does not
cut their checkpoint publication short. Rerun the same command with `--resume`
to clear the task-owned marker and continue; the plan and executable must match
exactly. `--max-chunks 1` provides a bounded pause test per ensemble. A paused
campaign has `paused.json`, not a new `completed.json`; use `CURRENT` as the
state authority and regenerate stale summaries before resuming. A hard kill can
repeat indexed outputs after the last complete checkpoint; retained indices are
overwritten atomically, not appended. Replay at warmup+32 to warmup+48 compares
complete checkpoint bytes, counters and cold records; replay draws are excluded.
The runner requires schedules in chunks of 16 and at least 48 retained draws.

`summary.json` records transport, `posterior-summary.json` records all cold gates.
Descriptive intervals and fitted-data residuals remain provisional when any gate
fails. These small synthetic checks establish neither independent-mock coverage,
numerical-refinement stability nor real-catalog fidelity.

The runner holds an exclusive OS writer lock (`RUN.lock`) for the whole campaign,
including native children and replay. A concurrent runner rejects before writing
checkpoints. The lock releases automatically on process exit; retain the file.

## Separately declared continuation

`plan_long.json` fixes 2048 retained draws per chain, using the completed 1024-draw
prefixes exactly once. It is a continuation on the same synthetic target, not an
independent replication. The previous 1024-draw result remains FAIL and immutable.
No target, priors, numerical settings, seeds, metric, local kernels, ladder,
exchange schedule or convergence thresholds change. Stop at 2048 even on failure.

```sh
python3 extend.py /absolute/new/campaign --source /absolute/old/campaign \
  --exe /absolute/build/tempering_pilot --plan plan_long.json
python3 run.py /absolute/new/campaign --exe /absolute/build/tempering_pilot \
  --plan plan_long.json --workers 4
```

The importer locks both trees, checks the source plan and completed schedule,
requires matching controls, copies indexed cold records and records source/binary
identities plus checkpoint/target/prefix hashes. Native `extend` reads a completed
checkpoint under its original sample bound and identity, verifies every replica's
likelihood, then publishes the same state/counters under the new schedule identity
in an empty destination. It requires zero sweep budget and tuple seed layout 2;
its final arguments are `SOURCE OLD_SAMPLES`. Ordinary resume still rejects a
changed schedule. An incomplete import cannot start the runner. If an import is
interrupted, use a new empty destination; source records remain untouched.

`replay_start` declares the first retained index for independent 16-sweep replay;
it defaults to 32. The longer plan uses 1056, beyond the copied prefix, with replay
at 1056..1071. Checkpoint and pause semantics are unchanged. Diagnostics accept
`PREFIX [DRAWS]`, retaining the original default 1024; `posterior.py` obtains the
exact count from the frozen plan. The transport report separates newly executed
native wall time from prefix-inclusive cost. Previously examined prefixes and
replay duplicates never become extra independent chains or duplicate samples.

## Independent fixed-truth mocks

`plan_mocks.json` predeclares two new datasets, each four fresh chains with
256 discarded warmup and 2048 retained draws. `plan_mock1.json` and
`plan_mock2.json` keep the validated kernels, priors, bands, numerical settings,
11-level ladder, four exchange passes and all original convergence gates.
No original posterior prefixes are reused. The two-mock study uses eight workers.

Layout 3 settings append independent-mock metadata:

```text
256 2048 4 3
FML_MOCK_1 1 5101 5102 5103
```

The four fields are mock ID, truth seed, noise seed and initial-white seed base.
IDs are 1..65535, data/initial seeds are positive 32-bit integers with no overlaps
within a mock. The study controller checks cross-mock overlaps and prevents reuse
of the original 4101..4106 seeds. The full 64-bit sampler key is
`0x5300000000000000 | mock_id<<32 | ensemble<<30 | replica<<24 | sweep<<4 | purpose`.
Purpose/slot conventions match layout 2. Metadata is fingerprinted in checkpoints
and recorded in targets/summaries. Layouts 1/2 retain their original bytes and
trajectories. Layout 3 uses fresh start/resume; the prefix importer supports layout
2 only. `tempering_mock_test` verifies 608256 distinct keys for both complete budgets.

```sh
make tempering_pilot tempering_diagnostics tempering_mock_test
./tempering_mock_test
python3 run_mocks.py /absolute/new-study --exe /absolute/build/tempering_pilot
# After both fixed budgets complete, analyze each mock independently:
python3 analyze.py /absolute/new-study/mock1
python3 posterior.py /absolute/new-study/mock1 --diagnostics /absolute/build/tempering_diagnostics
python3 analyze.py /absolute/new-study/mock2
python3 posterior.py /absolute/new-study/mock2 --diagnostics /absolute/build/tempering_diagnostics
python3 recover_mocks.py /absolute/new-study
```

`run_mocks.py` validates/locks the study and runs both per-mock controllers. A
study SIGINT/SIGTERM creates per-mock pause markers and drains active native chunks.
Rerun with `--resume`; completed batches/replay remain intact. Per-campaign writer
locks are inherited by native children, protecting against an orphaned child
writing after its parent exits. Dataset metadata and executable/plan hashes must
match on restart. Failed or successful mocks are preserved without substitutions,
extensions or cross-mock pooling.

Truth is fixed in physical scalar coordinates, while the initial field and
observational noise vary independently. Central 50/90/95% interval inclusion,
truth CDF fractions, offsets and field-projection recovery are descriptive. A miss
is not an automatic failure, and two mocks are not a coverage determination or
prior-predictive SBC. Failed-mixing recovery summaries remain provisional. Fitted
data residuals are not independent predictive checks. Computational convergence
and recovery/calibration claims are explicitly separated in recovery-summary.json.

## Opt-in prior-predictive truths

[Prior-predictive controls](PRIOR_PREDICTIVE.md) draw scalar truths from the fitted
bounded-uniform priors with independent white/noise/initial/sampler domains. The
new FML_MOCK_2 metadata and exact prior/truth enter checkpoint identity. Existing
fixed-truth formats remain compatible. Native prior/noise and analytic SBC
controls are separate from the long nonlinear calibration campaign.
