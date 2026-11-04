# Initial glass-IC validation — 4 November 2026

The first implementation was checked with Apple Clang in serial double precision,
OpenMPI 5.0.11 / Clang 17 / OpenMP in double and single FFT precision, and serial
AddressSanitizer plus UndefinedBehaviorSanitizer. MPI tests use 1, 2, 3 and 4 ranks
and two OpenMP threads; a four-rank run with one thread also passes. Serial FFTW is 3.3.8; the isolated OpenMPI FFTW builds are
3.3.10. No system dependencies were changed. Every test translation unit was built
from scratch with its own Global.cpp object; GSL is disabled.

The module suite passes 28 checks with one rank and 29 with multiple ranks (one
additional inconsistent-settings test). The example also runs in serial and two-rank
MPI double/float configurations. Sanitizers cover the complete test suite; an
additional long-running sanitized example was interrupted after the tests passed
and is not counted as a completed validation run. Two-rank writes read back on four
ranks exactly reproduce the original ID-keyed Poisson positions; batch size 13
forces repeated IO migration, including empty batches on non-reading ranks.

For the independent periodic Ewald comparison, 64 seeded particles are evaluated
with direct real-image and reciprocal sums, no mesh/window interpolation, and
negligible softening. Coarse/fine P3M meshes 24/48 give RMS relative vector errors
0.00550229 and 0.00523510. The split is 1.25 mesh cells, so changing mesh also changes
its physical radius; this is a check of the default family, not an asymptotic error
order. Retest force interpolation, mesh, split, softening and cutoff together for
production accuracy. A separate fixed-physical-split 48-to-72 mesh test confirms the
relaxed load remains quiet. The approximately 0.6% reference-force agreement here
is not a universal tolerance, particularly for very weak net forces.

For N=512, seed=73, mesh=48, dt=0.1, damping=2 and 400 steps:

| Quantity | Measured value |
|---|---:|
| Final/initial sum of six low-k structure factors | 1.87065e-5 |
| Final/initial force RMS | 0.00489308 |
| Mean nearest-neighbour distance / mean spacing | 0.956374 |
| Pairs closer than half mean spacing | 0 |
| Nearest-direction second moments xx, yy, zz | 0.313908, 0.355522, 0.330570 |

Structure factors use direct particle Fourier sums, not the force mesh. Pair and
nearest-direction diagnostics use independent direct periodic distances. These
small-sample statistics reject obvious clustering/Cartesian artifacts; they do
not prove isotropy or establish an asymptotic k^4 law. The run reports MaximumSteps
when both stopping tolerances are set to zero, and the zero-step/restart and loose
explicit convergence cases are checked separately. Extreme finite forces causing
residual overflow return NonFinite before a drift.

Across double-precision MPI task counts, the low-k suppression ratio agrees to
better than 1e-12 relative. Float-FFT results agree with double to 6e-6 relative in
this ratio and better than 1e-6 relative in the force-decay ratio. Initial Poisson
coordinates are exactly ID-keyed; relaxed results are tolerance-reproducible,
not promised bitwise identical. No large-scale MPI performance claim is made.

Independent CIC nodal interpolation verifies glass q, wrapped positions, velocities
and ownership after shifts. An imposed 0.01*cos(2*pi*q_x) density mode on the relaxed
glass and an 8^3 lattice recovers its expected Fourier amplitude within 1.5% and
0.1%, respectively. Existing full 3LPT including curl reduces to exact plane-parallel
Zel'dovich displacements and velocities in the manufactured test. Provider tests
check physical-volume normalization, fixed Fourier amplitudes, zero/Nyquist modes,
gauge, nonfinite power and time coverage. IO tests include truncation, duplicate
IDs and existing-prefix rejection.

Remaining production work includes larger independent seeds/volumes, shell and
angular power statistics, pair-correlation functions, relaxation/timestep/softening
convergence, arbitrary target spectra over resolved scales, external glass-generator
comparisons and clustered ghost-memory/rank scaling. Multiple-species quiet loads,
automatic SimulationCode selection and relaxation checkpoint/resume are extensions.
