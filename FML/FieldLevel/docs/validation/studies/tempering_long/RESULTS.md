# Longer four-cold-chain posterior validation

Frozen posterior gate: **PASS**; 0/11 monitored coordinates fail.
Connectivity gate: **PASS**.
Amplitude ratio: Rhat=1.00543, bulk ESS=842.726,
tail ESS=1527.97. This is sigma8/reference sigma8, not absolute sigma8.

Four independent ensembles each have 256 discarded warmup sweeps and
2048 retained cold draws on the unchanged synthetic N4/N8/N16 2LPT target.
Eleven replicas, four exchange passes and all local kernels match the bridge
pilot. Priors, data, spectrum shape, fixed noise and numerical settings are
unchanged. The first 1024 draws per chain are copied from the completed campaign,
counted once; only the remaining 1024 draws per chain were executed here. The
longer budget was declared after examining the previous failed diagnostics, so
this is not an independent replication. The earlier FAIL result remains intact.
Runtime schedule/ensemble controls and tuple seed layout 2 enter
complete-ensemble checkpoint identity. No excluded chains, gate changes,
retuning or extension of the frozen budget occurred. Replay is excluded.

## Frozen coordinate gates

Each row requires status ok, rank-normalized split/folded Rhat < 1.01,
bulk ESS >= 400 and tail ESS >= 400. Only beta=1 records enter these diagnostics.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Gate |
|---|---:|---:|---:|---|
| sigma8_ratio | 1.00543 | 842.726 | 1527.97 | PASS |
| b_delta | 1.00407 | 852.885 | 1922.39 | PASS |
| b_delta2 | 1.00826 | 728.542 | 1537.26 | PASS |
| b_K2 | 1.00332 | 1989.61 | 2634.46 | PASS |
| b_laplacian | 1.00072 | 1925.31 | 5766.01 | PASS |
| cos100 | 1.00168 | 1628.85 | 2752.7 | PASS |
| sin100 | 1.00323 | 1166.64 | 2106.78 | PASS |
| cos110 | 1.00337 | 1430.03 | 2114.2 | PASS |
| sin110 | 1.00163 | 2052.46 | 3473.31 | PASS |
| cos001 | 1.0016 | 1187.59 | 2285.38 | PASS |
| sin001 | 1.00315 | 1048.75 | 1345.58 | PASS |

## Transport, correctness and cost

Actual-prior-refresh-qualified cold/prior/cold round trips by ensemble: [24, 17, 14, 18].
Measured positive-beta field divergences by ensemble: [0, 0, 0, 0].
Pooled neighbor exchange rates: [0.1278076171875, 0.1287841796875, 0.122314453125, 0.129638671875, 0.31231689453125, 0.59112548828125, 0.69793701171875, 0.7418212890625, 0.78668212890625, 0.5430908203125].
All serialized targets match the original synthetic fixture; maximum scaled
cold quadratic reconstruction error is 8.17764e-16.

Preflight passed 25 exchange-kernel checks and exact old/new executable
continuation on a real 11-replica checkpoint. All four extended fixture chains
match fresh runs byte-for-byte (320 indexed cold records plus complete replica
checkpoints/summaries). Post-import pause/restart and terminal-boundary replay
pass. Original eleven diagnostics remain identical under variable draw counts;
legacy six/eleven checkpoint bytes and invalid-input guards pass. Dynamic-count
analysis and plotting pass on the 80-draw fixture. Every copied prefix record
remains byte-identical to its source, and all four source checkpoints retain
their original hashes.
The actual campaign replay at retained draws 1056..1071 matches complete checkpoint,
summary and all cold records exactly. Final fresh-process reload preserves all
four final checkpoints and summaries, including transport phases and trip counts.

Newly executed main native process wall time summed across workers:
6035.5 s. Prefix-inclusive summed process wall
time is 14588.6 s, including the original
warmup and first 1024 retained draws. These are process wall times, not CPU time.
Elapsed wall time of this continuation, including import and analysis: 1614.5 s.
Preflight/replay are excluded from the main-process cost. No matched-compute
sampler speedup or independent-replication claim is made.

## Interpretation and artifacts

These eleven monitored coordinates meet the declared bounded mixing gates on this target. The next separate gate is numerical-refinement posterior stability.

This is one same-model synthetic target, six field projections rather than all
coordinates, and fitted-data residuals rather than independent prediction.
Independent-mock calibration, real-catalog inference, larger grids and production
MPI replica layouts remain separate milestones.

[Runtime/restart instructions](../../../../validation/galaxy/tempering/README.md),
[frozen plan](../../../../validation/galaxy/tempering/plan_long.json),
FROZEN.json and summary.json preserve controls, identities and full results.
Raw builds, records, replay, plots and review artifacts remain outside the library:
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-tempering-long/`.
Plots: campaign/posterior-overview.png, traces.png, scalar-posterior.png,
residuals-diagnostics.png, connectivity.png and energy-overlap.png.
