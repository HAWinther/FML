# Validation record — 12 November 2026

All listed runs passed. Builds used fresh objects in scratch directories; no existing repository test binaries or objects were reused. Exact compiler/link/launcher commands, logs and hashes are in the accompanying review bundle.

| Configuration | Core | Fourier |
|---|---:|---:|
| Serial, Clang, double FFTW | 17 checks | 36 checks |
| Open MPI, 1 rank, double FFTW-MPI | 17 | 36 |
| Open MPI, 2 ranks | 18 | 40 |
| Open MPI, 4 ranks | 18 | 40 |
| Serial, single-precision FFTW | 17 | 36 |
| OpenMP, 1 and 4 threads, Clang 17 | 17 each | 36 each |
| AddressSanitizer + UndefinedBehaviorSanitizer | 17 | 36 |

The core was built and run **without FFTW/GSL**. Fourier builds require FFTW but no GSL. Standalone Makefile builds/runs passed for core-only and serial Fourier configurations. The example Makefile built and the example ran. Root-suite target wiring was checked by a Make dry run; the full library suite was not run.

Adjacent existing serial regressions were freshly built and passed: 282 power/polyspectrum checks and 23 ensemble-statistics checks. Existing raw fourth-moment production code was not modified.

## Independent numerical checks

- Exact two-state Rademacher joint cumulant, including mixed signs/amplitudes and empty sample ranks.
- Four-variable correlated Gaussian quadrature with independently computed covariance: all degree-four moments satisfy Wick's theorem.
- Streaming N−1 covariance against hand-computed powers, explicit Gaussian mode normalization, negative connected values and volume scaling.
- A 100,000-realization synthetic ensemble of independent complex Gaussian mode powers: connected covariance is consistent with the analytic zero within the stated deterministic test bounds (not an estimated coverage claim).
- Direct DFTs of analytic real fields, independent full signed-wavevector enumeration, and integer pair-convolution enumeration of closing quartets. Compared raw fourth sums, all three Wick products, mode counts, quartet counts and normalized units.
- Serial meshes 8 and 9 exercise even Nyquist exclusions and odd high modes. Multi-rank meshes 8 and 12 respect the existing mesh-width/rank divisibility requirement; odd-mesh MPI coverage is only the 1-rank run.
- Narrow low-mode unpadded estimates agree with the same selection evaluated on the padded work mesh. Broad/high-shell cases check genuine closure against modulo-mesh aliases. Empty-shell invalid outputs, zero-power shells, input preservation and length^9 scaling are checked.
- A separate 36-member exact Gaussian Fourier ensemble uses one Hermitian mode pair, `delta(x)=X cos(kx)+Y sin(kx)`, with independent unit-variance Gaussian X/Y quadrature. Exact external covariance gives zero ensemble connected moment. Same-field Wick subtraction gives the analytically predicted **−3** dimensionless integrated plug-in bias.
- MPI failures include different rank options/configuration lengths/references, nonfinite data on one input rank, and collective empty/sample errors. Tests finish without deadlock at 1/2/4 ranks.

Double Fourier-oracle comparisons use `2e-9 * (1+abs(expected))`; single-precision comparisons use `32*epsilon_float * (1+abs(expected))`. Physical normalized-output errors are checked after dividing out the known `V^3/N_quartets` conversion, so cancellation roundoff is not mistaken for a volume-independent absolute tolerance. The output conversion is also checked directly. Core analytic moment comparisons use `1e-12` scaled tolerance; Monte Carlo null bounds are looser and explicit in the test.

## Limits

These are analytic, manufactured, direct-enumeration and synthetic-ensemble tests. No production cosmological N-body ensemble, external trispectrum-code comparison, survey window, connected particle shot-noise correction, production performance forecast or uncertainty propagation is claimed. Single-precision MPI and combined MPI+OpenMP were not run. Sanitizers instrumented the new module and Global.cpp, with external FFTW libraries uninstrumented. See `../TODO.md`.
