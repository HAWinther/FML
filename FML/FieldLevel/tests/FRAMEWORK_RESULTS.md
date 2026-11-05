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
