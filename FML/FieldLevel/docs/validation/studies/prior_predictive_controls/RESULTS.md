# Prior-predictive truth-generation controls

Control validation: **PASS**. Opt-in prior-predictive scalar/field/noise truths,
identity checks and runner support are implemented. The long nonlinear calibration
campaign is **not started**; this does not establish nonlinear mixing or coverage.

## New behavior

`FML_MOCK_2` adds a scalar-truth seed and prior-predictive mode to layout-3 mock
metadata. Five physical scalars are drawn from the exact model's bounded uniforms;
all 64 real white coordinates keep their standard Gaussian law. Truth white,
scalar truths, noise, starting fields and sampling have separate RNG domains.
Generation mode, metadata, exact scalar-prior definition and generated truths
enter restart identity. Unsupported scalar priors reject explicitly.

All four ensembles generate identical data independently. Starts and frozen
metrics use the existing four truth-independent configurations. No posterior
kernel, physical prior, noise, numerical cutoff or old campaign outcome changes.
The new target payload exports an auditable prediction and likelihood quadratic
at truth, raw white/noise fields and full integer mode coordinates.

## Independent generation and noise evidence

- 1024 complete sets of scalar, white and noise draws match a standalone Python
  Philox reference exactly on this platform (maximum scaled error 0). The reference
  also passes the official zero-key/zero-counter known-answer vector.
- All five uniform CDF and cross-coordinate screens pass. Maximum empirical CDF
  distance is 0.0472. Gaussian spatial covariance screens and independent white/noise
  cross-moment screens pass. Maximum unit Fourier-noise power discrepancy is 0.0835.
  These are fixed-fixture law checks, not a randomness theorem.
- Six actual nonlinear data-generation targets verify independent scalar, white,
  noise, starting-field and mock-ID controls. Changing starting-field seeds leaves
  the truth/data unchanged. The actual normalized observation noise is reproduced
  with an independent orthonormal FFT (maximum error below 2.8e-17); all six
  likelihood quadratics at truth match the independently reconstructed residuals.
- Actual positive-beta targets and cached bias likelihoods pass their existing
  consistency checks on the generated datasets. Beta=0 prior-refresh checks pass.

## Restart and legacy evidence

The key checker enumerates 3,041,336 distinct generation/sampler/exchange keys
covering the two legacy mocks and eight candidate four-chain 256/2048 budgets.
Candidate IDs 101–108 exercise the layout; the control smoke uses ID 101 and is
not pooled into any scientific campaign. Reserve fresh IDs (e.g. 201–208), declare
all truth/noise/analysis seeds and re-enumerate that exact manifest before SBC.

All changed generation seeds, mock ID and switching back to fixed-truth mode
reject resume without changing the original fixture checkpoint, summary or target.
Seven malformed runner-plan cases reject before launching native batches. Native
metadata checks reject malformed scalar seeds, collisions and reserved mock IDs.

Genuinely executed old/new start+resume trajectories match byte-for-byte for
all three legacy seed layouts: checkpoint, summary, target and indexed draws.
The three actual completed original/mock-1/mock-2 checkpoints also load/save
exactly using copies outside the source campaigns.

A separate four-chain smoke completes exactly 64 sweeps/retained draws per chain,
no warmup, with the existing eleven replicas and four exchange passes. It pauses
after one checkpoint chunk, resumes, and reproduces the full ensemble and sixteen
indexed cold records exactly. All four final fresh-process reloads are exact;
zero measured positive-beta divergences occur. Independently rebuilt cold
likelihood quadratics match all 256 retained records. This short smoke cannot
satisfy the proposed full-budget ESS gate and has no mixing/calibration claim.

## Calibration-analysis controls

A separate normal-normal reference uses 4096 synthetic datasets and four
independent exact posterior endpoints each. Correct parameter and likelihood
ranks pass the declared fixture screen. A deliberately data-ignoring prior sampler
also passes parameter-rank checks but is detected by the likelihood quantity:
its five rank-bin counts are [2104,1121,584,237,50]. Exact likelihood-rank counts
are [834,798,815,809,840]. This validates the rationale for a data-dependent test
quantity without claiming anything about nonlinear FML posterior calibration.

`endpoint_ranks` randomizes ties including both possible extreme ranks and rejects
invalid shapes/nonfinite values. Endpoint independence and stationarity are
caller responsibilities; correlated MCMC records must not be passed as if they
were independent posterior draws. Analytic fixture thresholds are controls, not
an eight-dataset nonlinear uniformity decision rule.

## Reproduction and remaining scope

[Method and commands](../../../../validation/galaxy/tempering/PRIOR_PREDICTIVE.md),
plan-smoke.json, FROZEN.json and summary.json retain the controls and all outcomes.
Raw binaries, draws, old/new trajectories, logs and the scripted preflight remain
outside the library at
`/Users/hansw/Documents/Codex/2026-10-28/i-w/outputs/fieldlevel-prior-predictive/`.
A first preflight attempt used a malformed test setting (a broad string replacement
made white/noise seeds equal); the native validator rejected it. That attempt is
retained under preflight-attempt1. No scientific dataset was dropped or replaced.

Next: implement the bounded prior-predictive campaign orchestration and analysis,
including likelihood-Q convergence diagnostics and prior ranks. Freeze exact
truth/noise/initial/analysis seeds, fresh sampler IDs, budgets and a wall-time cap
before generating those data. Preserve failed mixing and every rank/interval miss.
See [proposed calibration design](../../../../validation/galaxy/prior_sensitivity/CALIBRATION_PLAN.md).
Production MPI replica orchestration, larger-target validation and real-catalog
science remain open. This change adds no new MPI execution evidence.
