# Periodic framework stages 1–3: initial validation

Recorded 4 November 2026. This is the initial periodic milestone of backlog #52,
not a certification of survey inference, nonlinear coverage or distributed adjoints.

The 53-check framework suite passes under Apple Clang, GCC 12, serial ASan/UBSan,
and OpenMPI/Clang17/OpenMP with one rank, including NO_AUTO_FML_SETUP. MPI two/four
rank runs deliberately check early rejection, not distributed field predictions.
The previous 104-check serial safety suite (105 with manual MPI setup), multi-rank
rejection checks and the existing 37-check BAO/gradient/HMC suite remain passing.
The general example exercises joint BAO/field/bias/noise inference, writes streamed
white-field samples and returns an initial/matter/velocity reconstruction. Shared
Makefiles build the new targets. No Python or GSL is required for these checks.

## Deterministic checks

- Central directional finite differences and independent forward/reverse dot
  products verify the white-field adjoint for linear, Zel'dovich and 2LPT models.
- Parameter derivatives are checked for BAO dilation, b1, b2, b_s2,
  b_laplacian and log noise power, with distinct initial/bias/likelihood cutoffs
  and plane-parallel RSD. The comparison tolerance is 2e-6 relative to
  max(1,|analytic|,|numeric|); no compiler-specific gradient relaxation is used.
- Linear field-only MAP agrees with the exact conditional mean to 1e-10 maximum
  cell error. Exact conditional and direct field-marginal likelihoods agree.
- The Gaussian log-determinant/noise derivative is checked analytically at zero
  residual. A lost-likelihood negative control checks that actual data reduce
  field covariance; a high-noise limit returns the unit prior covariance.
- The generic BAO adapter preserves the existing 2LPT/bias/RSD prediction within
  1e-11 in the tested retained modes, including the explicit G2-to-s² convention.
- Reconstructions return initial density, band-limited real-space matter density,
  Lagrangian displacements and velocity samples. Matter/velocity outputs are
  independent of tracer bias; Zel'dovich velocities obey their supplied factor.
- Invalid dimensions, stale/foreign workspaces, unsupported odd grids,
  non-Hermitian data and response, prior support and unsupported MPI are checked.

The separate seven-check resolution/compatibility suite also verifies generic
amplitude/tilt spectrum derivatives, explicit likelihood composition and rejection
of improper uniform priors and duplicate overrides. A bounded-uniform alpha prior reproduces the original
BAO posterior differences to 1e-9, including its prior rather than only predictions.

A separate fixed-physical-field mesh check uses axis and mixed low-frequency
cosines with the same spectrum normalization on 8³/16³ particle meshes and
16³/32³ density meshes. With distinct bias cutoff, RSD, quadratic/tidal and
higher-derivative bias, the relative tracer-mode change is 2.36561e-5.
This is a controlled numerical convergence case, not a production kmax prescription.
The first LPT backend explicitly restricts the initial band below half Nyquist so
its quadratic products are resolved; it does not claim high-cutoff padding adjoints.

## Bayesian checks

At fixed parameters, field HMC is compared with the exact linear conditional mean
and covariance of a real Fourier projection; the callback streams the samples.
This conditional Gaussian regression has no divergent proposals in the tested runs.

Joint field/bias/noise inference uses a separate 100-by-160 numerical quadrature of
the exact field-marginalized two-parameter posterior. Two dispersed HMC chains use
600 warmup steps, 2000 samples each, 24 jittered leapfrog steps and target acceptance
0.95. Mean comparisons use five Monte Carlo standard errors from reference widths
and measured bulk ESS; width comparisons also use bulk/tail ESS. Usable Rhat/ESS
and a low divergent-proposal rate are required separately.

| Quantity | Exact-marginal quadrature | Clang joint HMC | GCC joint HMC |
|---|---:|---:|---:|
| Mean b1 | 1.8186097 | 1.7730484 | 1.8609336 |
| Mean log noise power | 6.5806867 | 6.6889680 | 6.5663862 |
| b1 split-Rhat | — | 1.00582 | 1.00542 |
| log-noise split-Rhat | — | 1.00615 | 1.00781 |
| b1 bulk ESS | — | 412 | 338 |
| log-noise bulk ESS | — | 278 | 241 |
| Divergent proposals, chain 1 / chain 2 | — | 2 / 1 | 2 / 0 |

Standard-library normal-draw implementations differ between Clang/libc++ and
GCC/libstdc++, so sampled trajectories/moments are not expected to be identical.
The original single-chain fixed numerical tolerance was replaced by the dispersed
chain/Monte-Carlo checks above, rather than declaring an arbitrary moment offset
acceptable. Deterministic gradients and exact linear calculations pass both builds.

**Sampler limitation:** the broad log-noise prior creates a narrow-noise geometry.
Rare divergent proposals remain despite high acceptance and usable Rhat/ESS; they
are not described as zero or suppressed in reports. The regression threshold is
under 0.5% per chain in this toy, together with correct mean/width checks. That is
not a production acceptance rule. Investigate divergences and tails, tune or
reparameterize, and perform field-summary and posterior-predictive validation
before interpreting production results. The short nonlinear joint-inference run
is an API/finite-sampling smoke check, not nonlinear coverage calibration.

## Remaining scope

The initial periodic stages 1–3 milestone is delivered. Full extraction of separate
evolution/tracer kernels, high-cutoff padded adjoints, generalized noise/count
likelihoods and survey response, self-consistent cosmological growth derivatives,
MPI prediction/adjoint/HMC state, PM/COLA and production nonlinear/model-mismatch
coverage remain. Workspace ownership is explicit, but some shared-kernel scratch
still allocates: FL-6's measured allocation/reuse work is not marked complete.

## Distributed linear stage 4a (12 November 2026)

A separate `test_framework_mpi` suite now exercises **actual distributed linear
inference** on one, two and four OpenMPI ranks. The existing `test_framework` and
BAO safety suites still test deliberate rejection by the original single-rank
backends. These are distinct entry points, not contradictory MPI claims.

The new suite checks local ownership, an independent direct DFT with global N³
normalization, scalar/field finite differences, a forward/reverse dot product,
nonidentity Fourier response, exact MAP, and fixed-parameter field HMC against an
exact real cosine-projection mean/variance. It includes collective rejection of a
one-rank malformed field, cross-rank Hermitian violation, inconsistent config,
parameters or HMC controls, missing derivatives, and a one-rank callback exception.
Unrepresentable spectrum proposals reject with infinite potential and cleared
gradients. Nonlinear evolution is explicitly rejected by this backend.

Apple Clang serial and ASan/UBSan pass 21 checks. OpenMPI/Clang17 passes 21, 46 and
92 total checks on 1/2/4 ranks (the helper sums per-rank checks; one-rank-only serial
comparisons and multi-rank-only consistency checks explain differing counts).
OpenMPI/OpenMP with two threads per rank and `NO_AUTO_FML_SETUP` also passes on
two ranks. The existing 53-check framework, seven-check resolution, 37-check BAO,
104-check safety and 23-check standalone HMC regressions pass after the shared
posterior/sampler changes.

Across the 1/2/4-rank runs, deterministic potential, field-gradient projection,
scalar gradients and exact log marginal differ by at most 1.14e-13. The fixed-field
HMC final projection differs by at most 5.52e-13; the short joint chain's final
scalar values differ by at most 1.79e-9. Floating-point reduction order can affect
longer trajectories, so rank-count bitwise identity is not promised.

For the fixed-parameter HMC projection, the exact mean/variance are
0.897204666 / 0.688142439; 1600 draws after 200 warmup steps give mean/variance
0.907132029 / 0.670479694 and approximately 410 effective samples. Mean acceptance
is 0.906878 with zero divergences. The short joint field/amplitude/tilt/bias/noise
run has mean acceptance 0.646138 and zero divergences; it checks synchronized
execution only and does **not** establish joint posterior calibration.

Fields/momenta and retained mode arrays remain local; only metadata, scalar
posterior information and sampler controls are replicated. This is not a measured
strong-scaling claim. Frequent safety collectives and scratch allocations remain.
Distributed LPT/deposition adjoints, nonlinear coverage and checkpoint/restart are
still open, with survey likelihoods deferred until after the prioritized MPI work.

## Distributed 1LPT stage 4b (12 November 2026)

Historical stage-4b results; stage 4c below extends the same backend to 2LPT.

`DistributedLPTModel` adds actual distributed Zel'dovich predictions, smoothed
Lagrangian bias/RSD adjoints, sparse particle/mesh routing and joint MAP/HMC
execution. `Evolution::LPT2` is deliberately rejected until its separate distributed
source/adjoint increment is implemented. The original single-rank model stays intact.

The new `test_framework_lpt_mpi` suite passes 20 serial/one-rank checks and 36/72
total helper checks on two/four OpenMPI ranks. Two ranks with two OpenMP threads
per rank and `NO_AUTO_FML_SETUP` pass 36 checks; serial ASan/UBSan passes 20.
The shared Makefile's `framework-lpt-mpi` target passes on two ranks. Rebuilt
stage-4a MPI linear tests pass 46 checks on two ranks, and the unchanged periodic
framework regression passes all 53 checks. This increment does not alter the
shared HMC engine or the original BAO/LPT numerical kernels.

At mesh 8, density mesh 16 and kernel width 6, the checks cover:

- Field and all seven scalar derivatives (amplitude, tilt, b1, b2, b_s2,
  b_laplacian, log noise), with separate initial/bias/likelihood cutoffs, RSD and
  a nonidentity Fourier response. Directional and component finite differences
  use 2e-6 relative to max(1,|analytic|,|numeric|).
- A separate forward/reverse dot product through both the particle-node transport
  and initial/density Fourier remapping; serial predictions agree within 1e-12,
  field gradients within 1e-11 and scalar gradients within 1e-9.
- A strong-displacement case (white field multiplied by 80), exercising periodic
  wraps and rank crossings without assuming neighboring-slab particle motion.
- An independent sparse scatter/reverse test against an analytic mesh cotangent,
  including colliding/distant node deposits and empty-request ranks. Missing
  forward state and invalid uint64 node keys reject collectively before exchange.
- Nonlinear field MAP lowers the global objective; a deliberately short joint HMC
  chain moves with synchronized scalar states and no divergences. It is an
  execution smoke check, not a nonlinear posterior-moment/coverage test.
- One-rank malformed fields fail collectively; nonfinite proposals reject before
  converting particle coordinates to integer mesh indices.

Optional test output records each retained mode and owned field-gradient cell.
Comparing those arrays across 1/2/4 ranks gives maximum absolute differences
1.63e-17 for predictions and 4.45e-16 for field gradients. The ordinary posterior
potential differs by at most 5.69e-14; the strong-displacement potential is about
1.61e6 and differs by 1.17e-9 (approximately 7.3e-16 relative). Scalar-gradient
and short joint-trajectory comparisons also agree within floating-point rounding.
Rank-count bitwise identity for arbitrary longer chains is not promised.

The updated MPI example runs in linear or `1lpt` mode. Its two-rank 1LPT field MAP
has maximum gradient 1.85e-7; the short field HMC example has acceptance 0.736987
and zero divergences. These toy outcomes are not production convergence criteria.

The FFT fields remain local. Sparse maps are rebuilt per prediction and exchange
is through checked MPI-int layouts, with explicit overflow rejection rather than
large-count transport. Concentrated particles can concentrate incoming requests.
No strong-scaling, balanced peak-memory or new OpenMP particle-loop speedup claim
is made. Wider numerical convergence, nonlinear inference calibration, 2LPT,
checkpoint/restart and reconstruction output bundles remain open.

## Distributed 1LPT/2LPT stage 4c (12 November 2026)

The distributed LPT backend now supports both Zel'dovich and EdS-factorized 2LPT.
The combined suite runs the original stage-4b checks for each order, including
serial prediction/posterior/gradient comparisons, all seven scalar derivatives,
field directional differences and independent forward/reverse dot products,
large periodic displacements, sparse node communication, MAP and short joint-HMC
execution. Non-LPT evolution remains explicitly rejected.

Apple Clang serial and OpenMPI/Clang17 with one rank pass 44 checks. Actual two/four
rank runs pass 80/160 total helper checks; those totals sum per-rank checks and
exclude one-rank-only serial comparisons. Serial ASan/UBSan passes 44 checks.
Two ranks with two OpenMP threads each and `NO_AUTO_FML_SETUP` pass 80 checks.
The existing linear MPI suite passes 46 checks on two ranks, and the original
periodic framework suite passes 53. The shared Makefile target also runs the
combined LPT suite on two ranks. The shared HMC engine and original serial kernels
are unchanged by this increment.

Additional independent checks:

- Two orthogonal initial plane waves have analytic S2 = A B cos(kx) cos(kz).
  A direct sum over the corresponding exact 2LPT particle positions verifies
  the 3/7 coefficient/sign and `1+2f` second-order RSD factor. Input cutoff .08
  excludes the mixed source wave while the full-grid evolution retains it.
  Maximum retained-mode error is 8.11e-7 against a 2e-5 absolute finite-assignment
  tolerance; the actual second-order prediction change is 0.02859. This direct
  particle reference is independent of the implemented mesh source/adjoint.
- A single axis plane wave gives identical 1LPT and 2LPT predictions to 1e-12,
  checking vanishing second-order source on distributed slabs.
- Assignment widths 4 and 8 pass separate 2LPT field-adjoint finite differences;
  the main cases use width 6. Mesh 8 / density mesh 16 and separate cutoffs are
  used throughout these bounded regressions, not a production convergence study.

Full prediction and field-gradient arrays are recorded separately for each order
and compared across 1/2/4 ranks. For 2LPT, maximum absolute differences are
1.50e-17 for predictions and 4.45e-16 for field gradients. The 1LPT comparison
remains within 1.63e-17 / 4.45e-16. Scalar gradients and short joint trajectories
also agree within floating-point rounding. This does not promise bitwise identity
for arbitrary longer HMC chains.

The example now accepts `2lpt` as well as `1lpt` and the default linear mode.
On two ranks, the 2LPT field MAP gradient norm is 1.88e-7; its short field HMC
example has acceptance 0.666303 and zero divergences. The toy joint-HMC checks
and examples verify execution and movement, not nonlinear posterior calibration.

The source and reverse use the full distributed initial-grid derivative operator
with zero and Nyquist planes removed. Unsmoothed matter density/tensors are cached
separately from bias fields only for 2LPT (seven extra local fields), with temporary
source/reverse fields. No recomputation policy, new speedup or strong-scaling
claim is made. High-cutoff padding, general cosmological growth derivatives,
checkpoint/restart, distributed reconstruction outputs and nonlinear scientific
validation remain open; survey likelihoods are still deferred.
