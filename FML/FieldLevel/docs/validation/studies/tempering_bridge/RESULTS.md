# Configurable prior-bridge replica connectivity refinement

Recorded 18 November 2026. **Declared connectivity gate: PASS.**
No posterior-convergence, marginalized-sigma8 or coverage success claim.
[Runtime ladder interface and kernels](../../../../validation/galaxy/tempering/README.md);
[separate fixed bridge plan](../../../../validation/galaxy/tempering/plan_bridge.json);
[machine-readable results](summary.json);
[original six-level benchmark](../tempering_connectivity/RESULTS.md).

## Change and fixed comparison

The original six-level experiment is retained unchanged. This separate refinement
uses beta=1,1/2,...,1/512,0: eleven replicas, ten positive-temperature slots and
one exact prior refresh. Four alternating disjoint exchange passes follow each
unchanged local update sweep. The same two ensembles each discard 64 warmup sweeps
and retain 256 outer sweeps. Physical data, cold target, priors, local field/scalar/
rescaling kernels and slot-based component seed rules are unchanged.

Both ladder size and exchange frequency change. This is a connectivity refinement,
not an isolated ladder ablation or matched-compute speed test. Twice as many
positive-temperature replicas run, and cheap exchanges occur more frequently.
All original edge/round-trip/divergence thresholds remain; no post-run tuning,
extra sweeps, excluded ensemble or hot/replay pooling is performed.

## Configurability and backward compatibility

Native CLI accepts an explicit whitespace-separated ladder and exchange-pass count,
with 2–64 replicas and 1–8 passes. The runner selects a fingerprinted JSON plan,
generates/checks its ladder file, and rejects unsupported changes to the compiled
physical fixture/local kernels. Replica/edge arrays, target contexts, codec bounds,
label checks and expected counter schedules use the configured layout. Canonical
beta values and pass count enter the target identity for nonlegacy layouts.

The original six-level/one-pass identity and wire format are preserved. The new
executable resumes a copy of a real original retained-window checkpoint and
continues five sweeps byte-for-byte against the archived old executable: full state,
summary and five new cold draw records agree. Generalized analysis also reproduces
the original critical statistics exactly on a separate copy. Original binary,
runner, source, raw records and failed gate remain archived and unchanged.
A changed ladder or schedule rejects the checkpoint before continuation.

## Repeated exchange and meaningful transport

Pass p uses parity=(outer_index+p)%2 with the same outer-keyed swap RNG. Each
Metropolis exchange preserves the product tempered target, moving complete physical
states/likelihoods/labels while leaving slot metrics and counters in place.
Every pass records its accepted edges and resulting labels. Edge counts aggregate
all passes; more attempts alone cannot be interpreted as improved net transport.

Repeated passes could bounce off beta=0 before any local refresh there. For this
multi-pass layout a round trip requires an actual beta=0 prior refresh after a
measured cold visit, followed by a measured return to cold. The phase is marked at
the prior refresh and cold visits are observed after every pass. Unrefreshed bounces
are excluded. This preserves the meaning implicit in the old one-pass completed
round trips; it is not a relaxed gate. Walker labels do not identify an unchanged
physical configuration. Only final beta=1 records count as cold inference draws.

## Numerical and restart validation

All 25 kernel checks pass: layout/pass bounds, parity/counter arithmetic, malformed
inputs, old transport, refreshed-prior transport, and independent correlated
Gaussian mean/variance/covariance references under both six-level single passes
and eleven-level repeated passes. Maximum moment discrepancies are
0.00994167 and 0.00780722, within the declared finite-sample 0.02
check; these are not exact moment equalities.

All ten positive-beta target contexts and thirty bias-cache/full-target comparisons
pass, worst scaled discrepancy 1.19326e-14. The actual
beta=0 refresh passes 8192 white/scalar prior moment checks. Full eleven-state
execution for five sweeps equals two plus three across fresh processes in raw
checkpoint and summary bytes. Wrong ensemble, corruption and malformed input reject.

During the campaign, a copy at sweep 96 resumes for 16 sweeps. Complete ensemble
states, likelihoods, counters, qualification phases, summaries and all sixteen cold
draw records including pass traces match byte-for-byte. Duplicate replay draws
are excluded. Native resumes recheck saved original likelihoods. Indexed output
and authoritative summary refresh retain ordinary process-restart semantics;
this is not power-loss durability or production MPI ensemble checkpointing.

## Measured connectivity

Every edge must exchange at least once per ensemble, every pooled edge rate must
be >=0.05, each ensemble must complete at least one qualified cold-prior-cold trip,
and measured positive-beta field divergence counts must be zero.

| beta edge | Ensemble 1 accepted/attempted | Ensemble 2 accepted/attempted | Pooled rate |
|---|---:|---:|---:|
| 1 ↔ 0.5 | 57/512 | 44/512 | 0.0986 |
| 0.5 ↔ 0.25 | 86/512 | 65/512 | 0.1475 |
| 0.25 ↔ 0.125 | 48/512 | 79/512 | 0.1240 |
| 0.125 ↔ 0.0625 | 87/512 | 57/512 | 0.1406 |
| 0.0625 ↔ 0.03125 | 137/512 | 132/512 | 0.2627 |
| 0.03125 ↔ 0.015625 | 309/512 | 321/512 | 0.6152 |
| 0.015625 ↔ 0.0078125 | 341/512 | 356/512 | 0.6807 |
| 0.0078125 ↔ 0.00390625 | 390/512 | 368/512 | 0.7402 |
| 0.00390625 ↔ 0.00195312 | 406/512 | 398/512 | 0.7852 |
| 0.00195312 ↔ 0 | 264/512 | 283/512 | 0.5342 |

Refresh-qualified round trips per ensemble: **1, 1**.
Gate details: [{"ensemble": 0, "each_edge_exchanged": true, "roundtrips": 1, "roundtrip_gate": true, "divergence_gate": true}, {"ensemble": 1, "each_edge_exchanged": true, "roundtrips": 1, "roundtrip_gate": true, "divergence_gate": true}].
Measured field divergences:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0].

The prior-end pooled rate is 0.5342, compared with
0.0117 for the original direct 1/16→0
link. Independent analysis reconstructs every pass's label movement, all aggregate
counts and actual-prior-refresh qualification from the records. Cold quadratic
likelihoods reconstructed from predictions agree to
4.88958e-16 scaled error. Data targets match
the original interweaving fixture serialized bytes exactly.

Summed main-chunk process-wall time is 1435.9 s,
versus 686.8 s for six levels
(2.091 times). Both use two concurrent serial
ensembles. Timing includes all replica work, initialization and I/O and excludes
preflight/duplicate replay. Host load and two simultaneous algorithm changes
prevent an isolated speedup or production hardware/scaling claim.

## Consequence

The declared short connectivity gates pass. Next is a separately frozen four-cold-chain posterior validation with the existing physical scalar/field convergence gates, exact ensemble restart and full replica cost. Transport success alone does not certify posterior mixing.

This two-ensemble short pilot has no posterior Rhat/ESS pass criterion. It does
not establish all-field mixing, calibrated amplitude intervals, physical refinement,
external LEFTfield agreement or real-catalog inference. Beta=0 is mathematically
optional for replica exchange, but this protocol deliberately retains the original
prior-end transport goal. Raw prior-weight ESS diagnostics remain exploratory;
no cold records are reweighted or failed ensemble discarded.

Build/run commands, original/native/source/plan hashes, primary and duplicate
records, complete states and figures remain outside the library at workspace
`outputs/fieldlevel-tempering-bridge/`. Original six-level sources are archived in
`before/`. Plots under `campaign/`: `connectivity.png`, `energy-overlap.png`.
Production model/sampler APIs and MPI ensemble support are unchanged. Item 52
remains ongoing.

Additional guards reject a changed beta value with unchanged replica count, or
a changed pass schedule, without changing CURRENT. A stale-summary fixture is
repaired from the complete native state using zero transitions; completed output
counts/checkpoint bytes remain unchanged. The runner rejects a modified generated
ladder, a different selected plan or changed executable before sampling.
