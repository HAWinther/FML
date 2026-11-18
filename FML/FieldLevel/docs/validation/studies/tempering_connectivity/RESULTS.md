# Likelihood-tempered replica-exchange connectivity pilot

Recorded 18 November 2026. **Declared connectivity gate: FAIL.**
There is **no posterior-convergence or sigma8-inference success claim**.
[Maintained driver/target argument](../../../../validation/galaxy/tempering/README.md);
[frozen plan](../../../../validation/galaxy/tempering/plan.json);
[machine-readable report](summary.json).

## Target, ladder and budget

Two independent serial ensembles use beta=[1,1/2,1/4,1/8,1/16,0], each with
64 discarded warmup sweeps and 256 measured sweeps. Each positive-beta slot uses
the preceding field-HMC/cached-scalar/rescaling kernels with fixed noise power
1000/beta and a temperature-slot frozen metric reference. Only the likelihood is
tempered: all original priors and observed data remain unchanged. beta=1 is the
same N4/N8/N16 synthetic target as the earlier pilots. beta=0 refreshes the proper
Gaussian-white/bounded-uniform-scalar prior directly, then evaluates original L.
No infinite-noise posterior is constructed. Noise is fixed; normalization changes
are replica-specific constants, not extra state-dependent target terms.

Alternate disjoint even/odd neighbor exchanges accept with
`log A=(beta_i-beta_j)*(L_i-L_j)`. Fields, scalars, original likelihood and labels
move together. Temperature-slot metric references/counters remain in place and
scratch caches rebuild. Momenta refresh per local transition. Component seeds,
ladder, swap parity, full ensemble and transport counters are checkpointed or
fingerprinted. This is a serial validation prototype, not a production MPI ensemble
API. Physical model and production sampler headers remain unchanged.

The plan was frozen before sampling. No ladder change, retained-budget extension,
retuning, excluded ensemble or hot/replay pooling was made. Exactly 512 measured
cold records enter connectivity analysis. These are not a four-chain posterior
mixing or coverage campaign. Raw exchange rates are correlated empirical outcomes,
not independent-binomial estimates.

## Preflight and restart verification

The 11 dedicated checks pass: swap-ratio sign/reciprocity, malformed ladders,
invalid inputs before mutation, label transport and independent correlated-Gaussian
marginals at all six betas, including scalar/field covariance and the beta=0 prior.
Maximum analytic-reference mean/variance/covariance discrepancy is 0.00994167 for
100000 measured toy sweeps; this finite-sample check is not exact moment equality.

Five positive-beta full-target differences and fifteen bias-cache comparisons
match the intended prior+beta*likelihood target; maximum scaled error
1.19326e-14. The actual native beta=0
refresh passes moment checks on 8192 prior draws. Its field mean/second moment are
0.00328387 and
1.00786. All physical scalar-prior
checks pass. Full ensemble execution for five sweeps equals two plus three across
fresh processes in raw checkpoint and summary bytes. Wrong ensemble, checksum
corruption and older single-replica formats reject without advancing CURRENT.

During the pilot, a copy at sweep 96 resumes for 16 sweeps. All six final states,
likelihoods, labels, counters, transport phases, summaries and sixteen cold record
files replay byte-for-byte. Duplicate replay records are excluded. Native resume
also rechecks every saved original likelihood. The runner repairs summaries from
authoritative native state with zero-transition resumes before scheduling chunks
or replay continuation. This is process-restart validation, not power-loss testing.

## Observed connectivity

Every edge must exchange at least once in each ensemble, every pooled edge rate
must be >=0.05, each ensemble must complete at least one measured labeled
cold→prior→cold round trip, and positive-beta measured divergence counts must be
zero. Visits before the measured window do not count. Walker labels persist through
stochastic local updates, including independent prior refresh; they do not identify
an unchanged physical configuration.

| beta edge | Ensemble 1 accepted/attempted | Ensemble 2 accepted/attempted | Pooled rate |
|---|---:|---:|---:|
| 1 ↔ 0.5 | 23/128 | 13/128 | 0.1406 |
| 0.5 ↔ 0.25 | 18/128 | 12/128 | 0.1172 |
| 0.25 ↔ 0.125 | 17/128 | 14/128 | 0.1211 |
| 0.125 ↔ 0.0625 | 13/128 | 18/128 | 0.1211 |
| 0.0625 ↔ 0 | 1/128 | 2/128 | 0.0117 |

Measured round trips by ensemble: **0, 0**.
Per-ensemble gate details: [{"ensemble": 0, "each_edge_exchanged": true, "roundtrips": 0, "roundtrip_gate": false, "divergence_gate": true}, {"ensemble": 1, "each_edge_exchanged": true, "roundtrips": 0, "roundtrip_gate": false, "divergence_gate": true}].
Measured positive-beta field divergence counts:
[0, 0, 0, 0, 0], [0, 0, 0, 0, 0].

Analysis independently reconstructs disjoint label moves, edge attempts/accepts,
transport phases and round-trip counters from all measured records. Cold likelihood
quadratics reconstructed from saved predictions agree to
6.13888e-16 scaled error. Both serialized
data targets match the original interweaving campaign exactly. No sigma8 Rhat/ESS
gate is applied or claimed for this short connectivity experiment.

Main-chunk per-process wall times sum to 686.8 s
with two concurrent ensembles on this host. This includes all six replicas,
initialization, target/cache work, exchange likelihood evaluations and output/I/O;
preflight and duplicate replay are excluded. It is not a production hardware or
MPI-scaling forecast, and cannot be compared as an efficiency multiplier against
unmixed non-tempered runs with different work schedules.

## Consequence and limits

The declared short connectivity gates fail. Preserve this ladder benchmark and diagnose low-overlap edges/transport before a separately declared ladder-refinement test. Do not proceed as though the cold posterior has converged.

Prior path barriers motivated this ladder but did not guarantee adjacent energy
overlap. Empirical exchange rates, measured-window round trips and cold traces are
short diagnostics, not evidence, configuration masses, all-field mixing, physical
refinement or real-catalog inference. A revised ladder would be a new explicit
tuning experiment and cannot silently alter the retained records here.

Raw native builds, source/binary/plan fingerprints, commands, logs, ensemble states,
cold/replay records and plots remain outside the library under workspace
`outputs/fieldlevel-tempering-pilot/`. Figures under `campaign/`: `connectivity.png`
and `energy-overlap.png`. All earlier pilots remain unchanged. Item 52 is ongoing.

## Exploratory prior-end overlap diagnosis

Undoing the recorded disjoint accepted swaps recovers the independently refreshed
pre-swap beta=0 likelihood quadratic on every measured sweep: 512 prior draws.
This follow-up uses saved records only. Median prior L is 734.9; the 5–95% range
is 248.9–3344.1. Exponential weights exp(-beta*L) give the following descriptive
prior-sample ESS fractions:

| Candidate beta | Prior weight ESS / 512 |
|---:|---:|
| 0.0625 | 0.0087 |
| 0.03125 | 0.0537 |
| 0.015625 | 0.1561 |
| 0.0078125 | 0.2903 |
| 0.00390625 | 0.4464 |
| 0.001953125 | 0.6195 |
| 0.0009765625 | 0.7804 |
| 0 | 1.0000 |

These are noisy finite-sample overlap diagnostics, not cold posterior estimates,
exchange-rate predictions or new connectivity gates. No cold records are reweighted.
The original gate stays FAIL. The prior gap appears much larger than two additional
rungs alone would reliably resolve. [The next-stage proposal](../../../../validation/galaxy/tempering/NEXT_STEPS.md)
calls for explicit configurable-ladder tuning, with any bridge tested separately.

A deliberately stale ensemble summary is also repaired from native checkpoint
state by zero-transition resumes; full checkpoint bytes and record counts remain
unchanged. Completed-run replay adds no transition; changed binaries reject.
