# Bounded fixed-IC amplitude pilot — 17 November 2026

A synthetic Gaussian tracer field with 56 active real IC coefficients and 124
real likelihood degrees of freedom was analyzed at fixed known physical ICs.
All four Eulerian bias coefficients were sampled/marginalized with the declared
bounded uniform priors and fixed noise. This is not a discrete galaxy catalog,
challenge reproduction or coverage study. Native data, executable, protocol and
completed-chain hashes are recorded in the external run manifest.

Four chains per configuration, each with 256 warmup and 1024 retained draws,
were completed without extension or thinning. The three configurations share
physical ICs, data, filters, spectrum, noise and priors. Numerical settings are
particle/product mesh N, deposition mesh Nd and kernel width w:

| N / Nd / w | Mean alpha | SD | alpha Rhat | alpha bulk ESS | alpha tail ESS |
|---|---:|---:|---:|---:|---:|
| 12 / 24 / 6 | 0.99484 | 0.01806 | 1.00904 | 287 | 299 |
| 12 / 36 / 6 | 0.99519 | 0.01848 | 1.00902 | 331 | 466 |
| 16 / 48 / 8 | 0.99395 | 0.01865 | 1.00940 | 270 | 419 |

The known amplitude is 1. All results are **preliminary**: the frozen bulk-ESS
requirement of 400 is not met, including some bias coefficients. Rhat and tail
ESS alone do not establish adequate mixing. No failed configuration was removed
or extended.

Coarse/fine mean shifts are 0.048 and 0.067 finest-mesh posterior SD. The paired
length-64 batch MC errors are approximately 0.076 and 0.073 SD, respectively.
The observed shifts fall inside the 0.1 SD numerical budget, but their MC
uncertainty is too large to certify that budget. The final refinement changes
several numerical settings jointly; this is not a complete independent scan.

Native scalar-chain comparisons on serial versus pencil MPI 1/2/4 ranks agree
to a maximum absolute 5.7e-13 in this small seeded smoke test. The cache/conditional
sampler also passes a separate joint-posterior quadrature test (including the
amplitude/bias Jacobian), replay, numerical-rejection and MPI-reduction tests.
The Eulerian full-target values and all bias gradients agree with the cache.

Next: an exact joint amplitude/bias proposal using the small conditional Gaussian
likelihood geometry, followed by a **new** declared bounded campaign. Preserve
this failed-mixing pilot. Do not substitute analytic unbounded/Gaussian-prior
bias marginalization for the bounded uniform-prior target. Exact catalog inputs,
NUFFT ingestion and free-field sampling remain subsequent milestones.

See the [protocol and maintained driver](../../../../validation/galaxy/fixed_ic/README.md)
and [machine-readable outcome](summary.json). Run its analyzer to generate
posterior/traces, corner, mesh comparison, field/residual and residual-power plots.
Large chains and generated figures stay outside the library.
