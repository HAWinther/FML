# Arbitrary-order EdS LPT: scientific validation

Validated 12 November 2026. This stage completes the small band-limited validation
and delivery milestone of backlog 55. The generic algorithm remains parameterized
by a finite requested order; `validated_max_order=6` records the tested reference
range, not an upper implementation limit or a universal accuracy guarantee.

## Plane-parallel numerical fix

The new mesh plane-wave tests exposed a valid input rejected by the original
engine: the cubic mixed determinant summed 36 signed products separately. It
could leave a roundoff-only nonlinear source in an exactly rank-one deformation;
subsequent orders rejected the resulting incompatible source. The original header
reproduces this failure on the `(1,1,1)` fixture through order 6.

The column permutations are now paired into 2x2 minors before summing. This is
algebraically the same mixed determinant, and the cancellation-safe absolute
contribution envelope remains in use. Compatibility thresholds were not loosened.
The tested `(1,1,1)` and `(1,2,-1)` plane waves now have exactly zero nonlinear
coefficients through order 6 in float/double. A direct native contraction check
also verifies exact cancellation for the aligned rank-one test tensor.
This does not promise bitwise cancellation for every floating-point representation
of an arbitrary rank-one tensor.

The regrouping changes floating-point summation, with no extra stored fields or
changed source coefficients. No runtime or RSS improvement is claimed.

## Nonzero sixth-order reference and precision

The sixth-order fixture has three non-coplanar waves, unequal complex amplitudes
and nonzero transverse terms. Every exported Fourier coefficient through order 6
is checked against the independent sparse oracle. The oracle extracts coefficients
of the original determinant/cofactor Lagrangian equations and Cauchy invariants;
it does not call the production pair/triple contractions or use wrapped products.
The input has componentwise support K=1 and N=16, which retains the entire
sixth-order support. This is a genuine nonzero sixth-order check, not a zero-field
smoke test or a comparison between two executions of the same kernel.

Additional checks cover translations, reflections, axis permutations, negative
amplitude scaling, cache/recompute consistency and fixed-support N=16 to N=24
refinement. Equations are evaluated from computed mesh coefficients on the
physically reachable sparse modes, excluding roundoff-only forbidden modes from
that diagnostic contraction. The separate every-coefficient comparison includes
all forbidden modes, so their roundoff leakage is still checked.

| Configuration | Export cases | Worst relative coefficient comparison | Scalar-equation residual | Cauchy residual |
|---|---:|---:|---:|---:|
| serial | 14 | 1.7455e-15 | 9.6152e-16 | 8.9242e-17 |
| float | 14 | 4.2585e-07 | 3.0845e-07 | 1.8409e-08 |
| omp | 14 | 1.7455e-15 | 9.6152e-16 | 8.9242e-17 |
| float-omp | 14 | 4.2585e-07 | 3.0845e-07 | 1.8409e-08 |
| mpi | 22 | 1.7455e-15 | 9.6152e-16 | 8.9242e-17 |
| sanitize | 14 | 1.7455e-15 | 9.6152e-16 | 8.9242e-17 |

Each coefficient comparison is scaled to the largest reference coefficient at
that order, not each individual near-zero coefficient. Double tolerance is
`3e-12 * scale + 1e-18`; float is `3e-5 * scale + 1e-18` (about 250 float epsilons).
The equation residual scale is `max_abs(Psi_n) * 2pi * n^2`, using the full
computed longitudinal plus transverse displacement. These are fixture-scaled
checks, not uniform forward-error bounds for arbitrary data or large orders.
Float comparisons include quantization of the supplied decimal seed.

The MPI case covers 1/2/4 ranks, 1/2 OpenMP threads and both workspace policies.
Actual scalar size, ranks and thread counts are recorded in export metadata.
Serial and OpenMP float are validated; float MPI has not been run here and needs
matching float-MPI FFTW libraries. Empty slabs remain unsupported by FFTWGrid.

## Resolution and input-support sensitivity

For the same finite-support seed, N=16 and N=24 agree to 1.75e-15 in double and
2.93e-7 in float, using the per-order maximum-coefficient scale above. This checks
that the full-bandwidth work policy preserves the retained physical coefficients;
it does not test a cosmological continuum limit with extra input power.

A separate controlled input-support experiment adds a K=2 wave of amplitude
`.02+.005i`, then a K=3 wave `.002-.001i`, with their conjugates. Low-mode
`sum(.5^n Psi_n)` changes by:
- serial: K=1 versus K=3: 0.000445139; K=2 versus K=3: 4.37777e-06, scaled to the largest K=3 low-mode coefficient.
- float: K=1 versus K=3: 0.000445139; K=2 versus K=3: 4.37778e-06, scaled to the largest K=3 low-mode coefficient.

The smaller added tail produces a smaller difference. This illustrates input
support sensitivity for a declared sequence; K=3 is not claimed to be an
infinite-support or production-spectrum reference. Conservative intermediate
bandwidth is retained in every run. Earlier alias-sensitive high-frequency
order-1..5 checks remain active.

## Truncation, time scaling and sampled Jacobians

`example/ArbitraryOrder.cpp` reports Parseval displacement norms, growth-weighted
per-order terms, truncated-sum norms, minimum grid-sampled Jacobians and
nonpositive sample counts. It uses the public evaluator as well as explicit
per-order sums. Existing native tests compare evaluator coefficients and
`dPsi/dln(a)` against `g^n`, `n*g^n` and centered finite differences with
precision-aware steps.

The scientific runner independently reconstructs real-space gradients and
determinants with NumPy inverse transforms. Double diagnostic discrepancies are
below 1.4e-15 of the indicated column scale; float discrepancies are below 3.8e-7.
All native/NumPy nonpositive sample counts agree in the tested cases.

For the coupled fixture, sixth-order diagnostics are:

| Growth g | Weighted order-6 displacement RMS | Minimum sampled Jacobian |
|---|---:|---:|
| 0.5 | 8.476923e-09 | 0.7403426 |
| 1 | 5.425231e-07 | 0.5079632 |
| 4 | 0.002222175 | -0.2772119 |

The terms decrease across the tested orders even at g=4, while that truncated
map already has negative Jacobian samples. Decreasing terms and physical
single-stream validity are distinct tests. Finite-order decay is not a proof of
series convergence; positive sampled Jacobians do not exclude crossings between
cells or establish an infinite-order limit.

The plane-wave reference has delta amplitude `2*abs(.1+.04i)` and its exact
pre-crossing map is ZA. The continuum first crossing is at
`g=1/(2*abs(.1+.04i))`, about 4.64238. The example gives sampled minimum Jacobian
0.892305 at g=.5 and -0.292343 at g=6, identical at all retained truncation orders.
Continuing the formal plane-wave series beyond crossing is a diagnostic, not a
valid single-stream physical evolution.

The example limits meshes to 4..64 and requires `2*order < mesh` to retain all
of its K=1 input-generated support in the diagnostic fields. Its extra gradient
copies and partial sums are outside the solver's owned-memory report. The solver
budget remains an incremental owned-memory bound, not a process RSS guarantee.

## Regression, build and CI checks

Fresh C++17 builds cover serial double, serial float, OpenMP double/float,
ASan+UBSan double and Open MPI/Clang17 with OpenMP. No compiler warnings occur in
the captured builds. All configurations pass 42 native API/contract checks,
nine independent Python low-order tests and the local parallel-contract tests.
The MPI recovery/ownership suite passes 34/80/160 checks at 1/2/4 ranks, and its
24 earlier order-5 rank/thread/workspace comparisons pass with worst relative
coefficient error 9.68e-16. The table above covers the additional order-6 science
suite. Sanitizer checks use external, uninstrumented FFTW libraries.

The pre-existing library LPT suite passes all 84 checks; Fourier resampling passes
all 70. The low-order comparison driver's double-wavevector/complex-float compile
error was fixed with scalar-type casts. Roundoff perturbations, finite-difference
steps and native tolerances now respect field precision.

CI already auto-discovers `LPT/tests`. Its normal run now includes the science
suite and example verification; a separate workflow step cleans/rebuilds/runs the
module in float with OpenMP. The exact float Makefile target and both example
targets pass in a clean copied build directory using local libraries. The GitHub
Linux job itself has not been dispatched from this local work.

The test driver writes every exported coefficient and a report including scalar
size, actual rank/thread configuration, work mesh and compatibility ratio.
`test_science.py --output DIR` preserves sparse inputs, commands, raw fields,
examples, summaries and hashes; by default it cleans a temporary directory.

Remaining work: larger and realistic cosmological-field calibration, rigorous or
empirical series-convergence limits for those fields, higher-order references,
cheaper proven bandwidth policies, float MPI and empty-slab support, plus explicit
SimulationCode/COLA integration. General cosmology, decaying modes and time-Taylor
integration remain separate mathematical extensions.
