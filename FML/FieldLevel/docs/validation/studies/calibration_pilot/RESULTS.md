# Completed eight-dataset prior-predictive pilot

All eight declared datasets completed four chains with 256 warmup +2048 retained
cold draws each: 32 chains and 65,536 cold records. Sampling took 28,922.54 seconds
(8 h 2 min), within the declared cumulative 16-hour sampling cap. All analysis
completed. Computational gate: **FAIL**, with **2/8 datasets passing**. No truth,
seed, gate, budget or failed dataset was changed or replaced.

| Dataset | Overall | Failed coordinates | Positive-beta divergences | Cold divergences | Minimum pooled edge rate | Qualified round trips by ensemble |
|---|---|---:|---:|---:|---:|---|
| 201 | PASS | 0/12 | 0 | 0 | 0.1071 | 8,6,7,4 |
| 202 | PASS | 0/12 | 0 | 0 | 0.1212 | 29,14,15,24 |
| 203 | FAIL | 11/12 | 0 | 0 | 0.0514 | 6,0,6,5 |
| 204 | FAIL | 12/12 | 5735 | 2278 | 0.0938 | 0,5,7,3 |
| 205 | FAIL | 8/12 | 0 | 0 | 0.1540 | 53,40,61,57 |
| 206 | FAIL | 1/12 | 0 | 0 | 0.0474 | 3,1,2,3 |
| 207 | FAIL | 0/12 | 135 | 81 | 0.1360 | 8,10,15,13 |
| 208 | FAIL | 1/12 | 0 | 0 | 0.1428 | 28,11,25,28 |

Every one of twelve coordinates requires Rhat<1.01 and bulk/tail ESS>=400.
Existing transport and zero-divergence requirements apply separately. Dataset
207 passes all coordinate diagnostics but fails the divergence gate. Dataset 208
misses only the cos110 bulk-ESS gate (388.85). These remain failures; no tolerance
was changed or chain extended until pass.

Failures have different signatures. Dataset 203 has severely separated field
projections (maximum Rhat 1.265, minimum bulk ESS 11.78), no divergences and one
ensemble with zero qualified round trips. Dataset 205 fails eight coordinates
despite many round trips and no divergences. Dataset 206 has one coordinate fail
and minimum pooled exchange acceptance 0.04736, below 0.05. A single integration
change cannot be assumed to solve every failure.

Dataset 204 records 5735 divergent positive-beta field updates, including 2278
cold updates. Dataset 207 records 135, including 81 cold updates. The other six
have zero. Counts exclude outer warmup and aggregate all four ensembles; each
completed dataset has 8192 cold HMC field-update attempts. Divergent proposals
are rejected, but repeated failures undermine reliable finite-chain exploration.

summary.json retains every diagnostic, rank, credible interval and inclusion/miss.
Ranks and recovery summaries from failed datasets remain provisional. All outcomes
are retained in the overview histogram; no passing-subset calibration claim is
made. Eight datasets and four independent endpoints each do not establish precise
coverage even if every monitored mixing gate were to pass.

Raw checkpoints, indexed records, native replays, logs and plots remain at
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-calibration/pilot/`.
The figure is calibration-overview.png. This result supersedes the unstarted status
in the earlier runner-control record; that earlier freeze history is preserved.

Next diagnostic: [conditional integration/metric probes](../stability_probe/RESULTS.md).
The initial model remains small and same-model. Larger-target, real-catalog and
production MPI replica validation are still open.
