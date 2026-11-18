# Bounded calibration runner and likelihood diagnostics

Engineering/control validation: **PASS**. A separately frozen eight-dataset nonlinear
pilot is prepared and **not started**. No statistical calibration result follows
from these controls.

## Implemented behavior

The study runner freezes independent scalar/white/noise/initial/analysis seeds,
source identities, numerical plans and both executables before generating data.
Sequential datasets use four independent chains each and the existing sampler.
A native diagnostic adapter checks the original five scalars and six field
projections **plus likelihood Q**, with unchanged Rhat<1.01 and bulk/tail ESS>=400
gates. Existing transport/divergence gates are also required.

Analysis retains every declared dataset, including unstarted/partial progress,
failed diagnostics, every interval miss and provisional rank. It reconstructs all
cold Q values independently from stored predictions and data, and validates truth
Q/noise generation against the independent FFT reference. Only the declared final
endpoint from each of four independent chains enters ranks (L=4). Separate analysis
RNG controls randomize ties. No hot/replay pooling, rank-uniformity test, passing-
subset calibration or calibrated-recovery claim is made.

## Actual controls and failures retained

- Two fresh datasets, IDs 301/302, each complete four 48-sweep chains with no warmup.
  The study pauses after the first chunk, reports all incomplete outcomes and
  resumes exactly to its declared fixed budget. Every twelve-coordinate diagnostic
  is present, including Q, and all twelve gates fail for each short dataset. This
  is expected for a smoke with only 192 total records per dataset and ESS>=400.
- Likelihood Q Rhat is 1.3554/1.2711; bulk ESS is 10.27/12.76. These failures remain
  in the report. Their ranks and all interval summaries are marked provisional.
  No scientific mixing result is claimed. All 384 cold Q values reconstruct with
  scaled error below 6.7e-16, and both native full-ensemble replay checks pass.
- A third independent fixed 4x48 control (ID 401) receives an actual parent SIGTERM
  during native work. All chains drain to sixteen-sweep checkpoints; the study
  ledger acknowledges the elapsed batch and clears its pending reservation.
  Concurrent analysis is rejected by the study lock. Resume completes the original
  budget with exact native replay; no seeds or draw budgets change.
- Focused native diagnostic fixtures pass all twelve IID coordinates, then fail
  **only Q** when its four chains have discrepant means. A constant-Q fixture also
  fails Q while the original eleven coordinates pass. Thus adding Q materially
  affects the computational decision rather than merely adding a plot.
- Focused Python controls verify exact full-study key enumeration, malformed-plan
  rejection, immutable mock plans, retained interval misses, nonfinite diagnostic
  serialization and independent FFT/direct-space field projections.

## Runtime and restart controls

The cumulative sampling-time budget is durable across invocations. A batch reserves
2400 seconds before launch and refunds unused time after measured completion. A
crash/unacknowledged batch retains its entire reservation. Focused controls verify
that crash charges survive reopening the ledger, caps reject further batches,
overruns remain charged and missing ledgers beside native work reject restart.
Idle graceful-pause time is not charged. This is a checkpoint-boundary soft cap;
a running batch may drain past its reservation. Offline analysis is separate.

The executed smoke sources are retained under external frozen-control-source/.
The final implementation additionally hardens missing-ledger rejection and freezes
the native runner's reference plan.json; focused guards pass after those additions.
The figure's dataset labels were corrected without changing statistics. The final
long-study manifest was prepared again before any nonlinear data were generated.
No production truth, seed, prior, schedule, gate or time cap was changed.

## Frozen nonlinear pilot

PILOT_STUDY.json and PILOT_FROZEN.json preserve:

- Eight datasets, IDs 201–208, with independent prior-predictive truths and four
  chains each; 256 warmup +2048 retained draws, endpoint index 2047.
- Existing eleven temperatures, four exchange passes, physical priors, model,
  cutoffs and noise power. Native kernels and their checkpoint format are unchanged.
- Exactly 2,433,024 sampler/exchange keys and 56 generation keys in disjoint domains;
  deterministic enumeration digest is retained in the frozen manifest.
- A cumulative 57,600-second (16-hour) sampling budget, four workers, and independent
  tie-analysis seed 930501. A full 2400-second reservation must fit before launch;
  capped partial datasets stay in the report. No automatic extension-until-pass.

Preparation makes no model/data draws. The final unstarted analysis lists all
**eight incomplete** outcomes and makes no calibration claim. The full pilot is
not started, so no runtime-to-completion or calibrated-recovery result is implied.
Eight datasets with five possible endpoint ranks remain a low-power computational
pilot even if its full monitored gates eventually pass.

## Artifacts and next action

[Driver and exact method](../../../../validation/galaxy/calibration/README.md),
[declared study](../../../../validation/galaxy/calibration/plan.json), summary.json,
PILOT_FROZEN.json and PILOT_STUDY.json retain controls and all outcomes. Raw native
records, binaries, source snapshots, scripted checks and plots remain at:
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-calibration/`.
The control plot is smoke/calibration-overview.png. START.sh, PAUSE.sh, RESUME.sh
and ANALYZE.sh address the unstarted frozen pilot/ directory.

Next: launch that frozen pilot, retain every failed/incomplete outcome and assess
all twelve quantities. Same-model calibration, larger-target validation, exact
challenge inputs/real-catalog science and production MPI replica orchestration
remain separate milestones. No new MPI execution evidence is added here.
