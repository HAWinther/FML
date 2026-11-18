# Bounded joint Eulerian field/amplitude/bias pilot

Recorded 18 November 2026. **Frozen-budget mixing gate: FAIL.**
[Machine-readable report](summary.json); [driver and restart instructions](../../../../validation/galaxy/joint/README.md);
[frozen protocol](../../../../validation/galaxy/joint/plan.json).

## Target and budget

The joint pilot uses the same synthetic N4/N8/N16, L=100, PSWF epsilon=1e-8
Gaussian tracer target as the FL-7 field-only comparison: P_ref=20000 exp(-5k),
noise power 1000, spherical initial cutoff 1.7 kf, cubic matter/likelihood cutoffs
1.8/1.6 kf and fixed shape/cosmology. It samples 64 stored whitened coordinates
(including prior-only directions) and all five physical scalars. The amplitude
is **sigma8/reference sigma8**, not an absolute sigma8 determination.

Four dispersed scalar and independent white-field starts use the predeclared
seeds. Every chain completes 256 warmup trajectories plus 1024 retained draws.
HMC uses up to 12 jittered leapfrog steps, initial step size 0.05, warmup-only
step-size and scalar diagonal-mass adaptation, target acceptance 0.9 and the
corrected field metric frozen at that chain's initial scalar snapshot. Priors
are uniform on [0.5,1.5], [0,4], [-4,4], [-4,4], [-100,100].

The protocol was saved before building/running. No budget extension, post-run
retuning, chain exclusion or convergence-gate change was made. The field-only
probe and restart replay draws are not pooled into these four chains.

## Joint mixing results

Each of the eleven declared coordinates must have computable rank-normalized
split/folded Rhat < 1.01, bulk ESS >= 400 and tail ESS >= 400. Sampling divergences
must be zero in every chain. Diagnostics use FML::MCMC::diagnose_chains.

| Coordinate | Rhat | Bulk ESS | Tail ESS | Gate |
|---|---:|---:|---:|---|
| sigma8_ratio | 1.31713 | 10.4 | 78.2 | FAIL |
| b_delta | 2.09179 | 5.3 | 20.4 | FAIL |
| b_delta2 | 2.87602 | 4.6 | 26.0 | FAIL |
| b_K2 | 2.05962 | 5.4 | 30.7 | FAIL |
| b_laplacian | 1.91010 | 5.6 | 37.6 | FAIL |
| cos100 | 2.19387 | 5.1 | 32.1 | FAIL |
| sin100 | 2.26804 | 5.0 | 34.3 | FAIL |
| cos110 | 2.50076 | 4.8 | 23.0 | FAIL |
| sin110 | 1.95870 | 5.6 | 41.2 | FAIL |
| cos001 | 1.93468 | 5.6 | 25.5 | FAIL |
| sin001 | 2.58708 | 4.8 | 13.9 | FAIL |

Failed coordinates: sigma8_ratio, b_delta, b_delta2, b_K2, b_laplacian, cos100, sin100, cos110, sin110, cos001, sin001.
Sampling divergence counts: 0, 0, 0, 0.
Mean acceptance probabilities: 0.9718, 0.9603, 0.9439, 0.9818.
Adapted step sizes: 0.054479, 0.10546, 0.14741, 0.024437.
Acceptance and absence of divergences alone do not establish convergence.

| Scalar | Generating value | Descriptive 5%, 50%, 95% |
|---|---:|---|
| sigma8_ratio | 1 | 1.013, 1.3499, 1.4882 |
| b_delta | 1.6 | 0.0036758, 0.12867, 1.5394 |
| b_delta2 | 0.3 | -2.4556, -0.022004, 3.888 |
| b_K2 | -0.2 | -3.3147, -0.59964, 3.7942 |
| b_laplacian | 7 | -91.412, 79.613, 99.504 |

These pooled quantiles are **descriptive, provisional when gates fail**, not
certified posterior intervals or a coverage test. A single generating value
inside/outside an interval would not establish calibration. All chain-specific
scalar distributions, correlations, traces and selected field projections are
retained for diagnosing amplitude/bias/field coupling.

## Restart, numerical scope and plots

Each chain ran in native-checkpointed chunks of 64 trajectories. At chain 0
trajectory 384, an independent process resumed a copy for 64 trajectories. All
64 resulting retained JSON draw records and numerical summaries matched the
primary continuation; replay draws were excluded. The runner also rejects a
changed executable before resuming. All four serialized data targets match.
Exactly 4096 primary draws are analyzed. Existing separate tests establish
slab/pencil metric, adjoint and restart behavior; this pilot uses serial chains
two at a time and makes no MPI-scaling claim.

The fitted-data residual figure bins physical Fourier modes with the model's
half-complex multiplicities. It shows residual power of the posterior mean and
5–95% variation across forward predictions, normalized by the generating noise.
These bands exclude fresh observation noise; they are not a replicated-data
posterior predictive test or an independent goodness-of-fit gate.

Raw checkpoints, commands, source/binary/plan provenance, diagnostic CSVs and
figures are retained outside the library in workspace
`outputs/fieldlevel-joint-pilot/`. The main chunks sum to 525.1 seconds of
per-process wall time on this host with two concurrent serial chains; this is
not a production timing or hardware forecast. Figures: `traces.png`,
`scalar-posterior.png`, `residuals-diagnostics.png`, `joint-summary.png` under `campaign/`.

The traces show chains lingering in separated quadratic-bias/field configurations,
with some scalar marginals accumulating near prior boundaries. This demonstrates
poor joint exploration; it does not by itself distinguish multimodality from
strong correlations or inadequate adaptation/trajectory settings. The plotted
distributions therefore cannot be used to infer an amplitude shift.

## Consequence

The declared joint mixing gate fails. Keep this campaign as a failed benchmark; diagnose the scalar/field coupling and compare a separately declared sampler improvement before increasing resolution or claiming a marginalized amplitude result.

This is same-model synthetic validation. It does not establish full-field
convergence, repeated-mock coverage, resolution independence, external LEFTfield
agreement or a real-catalog result. Exact catalog IC/spectrum inputs and initial
cubic-filter conventions remain open.
