# PSWF matter deposition and position-adjoint validation

Recorded 18 November 2026. Bounded numerical checks pass.
[Machine-readable summary](summary.json); [API and numerical limits](../../../NUFFT_MATTER.md).

The N12-particle/N24-deposition Eulerian fixture compares 26 retained Fourier
modes to independent direct particle sums. Maximum absolute errors are
2.48416e-6 for default width-6 smooth assignment and 1.72081e-9 for width-9 PSWF
at epsilon=1e-8. This is one controlled fixture, not a production convergence
or performance benchmark. Different support sizes have different costs.

An independent 19-position single-particle check on an N32 fine grid, with
wavevector (5,-3,6) and epsilon=1e-10, measures maximum absolute mode error
1.29737e-10, position-gradient error against the exact Fourier derivative
1.43182e-7, and error against the discrete finite difference 1.21498e-8.
Derivatives are with respect to fractional-box coordinates. Value tolerance is
not a bound on derivative error. Separate fitted-weight derivative checks run
at epsilon=1e-4, 1e-8 and 1e-12. Compact-stencil jumps remain an approximation;
the reverse is local to a fixed stencil, not a derivative at a jump.

The Eulerian composition suite passes 63 serial checks and
107, 111, 111 MPI checks at 1, 2 and 4 ranks respectively. It runs
both assignment choices on slabs and pencils, complete field/scalar finite
differences through 2LPT and Eulerian operators, independent matter sums,
failure/replica controls and PSWF checkpoint-identity distinctions.

The broader distributed LPT suite passes 87/129/157 checks at 1/2/4 ranks for
**each** assignment choice. It covers both evolution orders, nonzero Lagrangian
bias and RSD, spectral responses, large wrapped/rank-crossing displacements,
all small pencil orientations, unequal domains, a noninteger fine/initial mesh
ratio, reconstruction, metrics and short MAP/HMC runs. Maximum PSWF relative
errors (the suite's relative-difference convention) are:

| Check | Maximum error |
| --- | --- |
| Posterior field finite difference | 2.06414e-09 |
| All scalar finite differences | 3.01908e-08 |
| Forward/adjoint dot product | 1.1832e-12 |
| Large-displacement finite difference | 1.00703e-07 |

Maximum slab/pencil comparison error is 1.03624e-14. The unchanged smooth option
also passes its original checks. Existing fresh-process checkpoint prepare/resume
and catalog suites pass at 1/2/4 ranks. These checkpoint regressions exercise the
existing supported adapters; no Eulerian production-restart driver was added.

Builds use Clang 17, double precision, FFTW and bounds checks. MPI runs use Open
MPI and two OpenMP threads per rank. Separate serial/MPI build directories and
exact commands/results are retained outside the source tree under workspace
`outputs/fieldlevel-nufft-matter/`, alongside raw logs and `accuracy.png`.

No full challenge catalog was downloaded or new amplitude posterior sampled.
No production memory, runtime, scaling, float, larger-rank or long-HMC validation
is claimed. Compact latent/product meshes, challenge-specific inputs/initial
filtering, production restart and real-data/free-IC science gates remain open.
