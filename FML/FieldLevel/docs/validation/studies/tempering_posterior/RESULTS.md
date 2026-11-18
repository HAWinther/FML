# Four-cold-chain posterior validation

Frozen posterior gate: **FAIL**; 2/11 monitored coordinates fail.
Connectivity gate: **PASS**.
Amplitude ratio: Rhat=1.00689, bulk ESS=465.646,
tail ESS=798.786. This is sigma8/reference sigma8, not absolute sigma8.

Four independent ensembles each completed 256 discarded warmup sweeps and
1024 retained cold draws on the unchanged synthetic N4/N8/N16 2LPT target.
Eleven replicas, four exchange passes and all local kernels match the bridge
pilot. Priors, data, spectrum shape, fixed noise and numerical settings are
unchanged. Runtime schedule/ensemble controls and tuple seed layout 2 enter
complete-ensemble checkpoint identity. No excluded chains, gate changes,
retuning or extension of the frozen budget occurred. Replay is excluded.

## Frozen coordinate gates

Each row requires status ok, rank-normalized split/folded Rhat < 1.01,
bulk ESS >= 400 and tail ESS >= 400. Only beta=1 records enter these diagnostics.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Gate |
|---|---:|---:|---:|---|
| sigma8_ratio | 1.00689 | 465.646 | 798.786 | PASS |
| b_delta | 1.00505 | 451.739 | 941.075 | PASS |
| b_delta2 | 1.01194 | 401.387 | 885.186 | FAIL |
| b_K2 | 1.00684 | 856.976 | 1032.31 | PASS |
| b_laplacian | 1.00642 | 881.741 | 2925.9 | PASS |
| cos100 | 1.00313 | 950.471 | 1238.61 | PASS |
| sin100 | 1.00329 | 655.677 | 1004.55 | PASS |
| cos110 | 1.00309 | 759.297 | 814.73 | PASS |
| sin110 | 1.00178 | 1081.96 | 1792.54 | PASS |
| cos001 | 1.00371 | 585.439 | 1101.02 | PASS |
| sin001 | 1.01214 | 566.593 | 1085.71 | FAIL |

## Transport, correctness and cost

Actual-prior-refresh-qualified cold/prior/cold round trips by ensemble: [11, 4, 6, 7].
Measured positive-beta field divergences by ensemble: [0, 0, 0, 0].
Pooled neighbor exchange rates: [0.1353759765625, 0.1295166015625, 0.1180419921875, 0.1302490234375, 0.31005859375, 0.5947265625, 0.700927734375, 0.7420654296875, 0.7874755859375, 0.5438232421875].
All serialized targets match the original synthetic fixture; maximum scaled
cold quadratic reconstruction error is 7.28089e-16.

Preflight passed 25 exchange-kernel checks; native retained continuation agrees
byte-for-byte with both original six-level and eleven-level executables. All four
new starts replay exactly across warmup/retained boundaries. Malformed or changed
controls reject. Bounded pause, SIGTERM pause and stale-summary recovery reproduce
uninterrupted full checkpoints and 192 indexed cold records. A concurrent writer
rejects without changing checkpoint bytes; released locks permit continuation.
The actual campaign replay at retained draws 32..47 matches complete checkpoint,
summary and all cold records exactly. Final fresh-process reload preserves all
four final checkpoints and summaries, including transport phases and trip counts.

Summed main native process wall time: 8553.1 s
(four workers; not CPU time). Overall elapsed wall time, including the controlled
pause and analysis: 2231.4 s. Replay/preflight are excluded from the summed
main-process figure. Compared with the short two-ensemble bridge, this campaign
uses eight times the positive-beta local sweep work; no matched-compute claim.

## Interpretation and artifacts

Nine of eleven coordinate gates pass, including the amplitude ratio. Every bulk and tail ESS exceeds 400. The only failures are Rhat for b_delta2 (1.01194) and sin001 (1.01214), against the frozen <1.01 requirement. The overall frozen posterior gate therefore remains FAIL; descriptive intervals and fitted-data residuals are provisional.

The next bounded experiment should be a separately declared longer run with the same sampler to resolve these two remaining Rhat failures. Keep this failed-budget result intact and freeze the new budget before sampling. The present result does not justify sampler retuning by itself. After all mixing gates pass, numerical-refinement posterior stability remains the next scientific gate.

This is one same-model synthetic target, six field projections rather than all
coordinates, and fitted-data residuals rather than independent prediction.
Independent-mock calibration, real-catalog inference, larger grids and production
MPI replica layouts remain separate milestones.

[Runtime/restart instructions](../../../../validation/galaxy/tempering/README.md),
[frozen plan](../../../../validation/galaxy/tempering/plan_posterior.json),
FROZEN.json and summary.json preserve controls, identities and full results.
Raw builds, records, replay, plots and review artifacts remain outside the library:
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-tempering-posterior/`.
Plots: campaign/posterior-overview.png, traces.png, scalar-posterior.png,
residuals-diagnostics.png, connectivity.png and energy-overlap.png.
