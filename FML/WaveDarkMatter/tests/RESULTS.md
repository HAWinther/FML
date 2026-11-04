# Initial wave-dark-matter validation — 4 November 2026

The suite passes with Apple Clang 16 and serial FFTW 3.3.8; OpenMPI 5.0.11,
Clang 17.0.6, OpenMP and isolated FFTW 3.3.10 double/float libraries; and serial
AddressSanitizer plus UndefinedBehaviorSanitizer. MPI uses 1, 2, 3 and 4 ranks,
including uneven and empty complex FFT slabs. The shared Makefiles also build.
No system dependencies were changed. Normal tests need no Python or GSL.

There are 37 core checks on one rank, 38 with multiple ranks (one additional
inconsistent-parameter check), and 9 soliton checks. Both example paths run in
serial, two-rank double/float MPI and sanitizer builds. Two-rank checkpoints reload
on four ranks with retained wave/source/epoch and continue the same trajectory to
precision tolerances. Same-layout/precision restart is exact in the tested cases.
Payload corruption and overwrite attempts are rejected collectively.

## Independent numerical tests

Full complex transforms on meshes 2, 3 and 4 agree with explicit O(N^6) complex
DFTs, including empty ranks and independent negative/positive modes. Free-wave
dispersion, two-wave interference, Parseval kinetic energy, periodic Poisson signs,
mean removal, external-density sources and comoving acceleration have analytic tests.

`generate_reference.py` uses NumPy 1.26.4 / SciPy 1.12.0 to generate a 4^3 nonlinear
method-of-lines solution with adaptive DOP853 (rtol=2e-13, atol=2e-14). The native
split solver agrees with its committed full-complex final-field fixture. A separate
radial SciPy boundary-value solve constructs the isolated nodeless soliton with
eta=G=1, f(0)=1 and U(infinity)=0. Its chemical potential is -2.453886796609026 and
mass 3.882765275664014. This is a numerical BVP reference, not an empirical density
fit or a profile relaxed by the evolution code under test. Local degree-five
interpolation of the reference table avoids linear-interpolation Laplacian noise.
The committed fixtures keep SciPy out of ordinary C++ test execution.

For a nonlinear periodic 12^3 field evolved to t=0.4, relative L2 field errors
against a finer trajectory are:

| dt | Field error | Absolute static-energy drift |
|---|---:|---:|
| 0.1 | 1.70577e-3 | 0.162143 |
| 0.05 | 4.27566e-4 | 0.0407151 |
| 0.025 | 1.06208e-4 | 0.0101901 |

Both errors decrease by approximately four. These deliberately coarse steps expose
the order of the method; use smaller steps for accurate interacting simulations.
Forward/backward self-gravitating evolution and unitary norm conservation also pass.
Phase overflow and invalid initializers preserve the live state.

The EdS integrals of dt, dt/a and dt/a² agree with closed forms. Expanding free
waves use the integrated kinetic phase and reverse correctly. For a nonlinear
wave field over a=0.5 to 0.6, the normalized Layzer–Irvine residual decreases from
2.90608e-7 at 20 observations/steps to 7.26609e-8 at 40 in double precision. Float
rounding dominates that very fine residual; its tested 2/4-step intervals give
2.81993e-5 and 4.22140e-6. Fine-step float residuals need not improve monotonically.
The energy-balance tracker has its own quadrature error and is not an error estimator.

## Soliton resolution and periodic volume

An isolated radial soliton placed in a finite periodic Poisson problem is not an
exact periodic eigenstate. Global constant-potential shifts are removed by fitting
a single eigenvalue before measuring the stationary-field residual.

| Mesh | Box side | Density L2 drift at t=0.5 | Eigen-residual / isolated binding frequency | Initial tail fraction | Relative static-energy drift |
|---|---:|---:|---:|---:|---:|
| 32^3 | 12 | 2.11225e-3 | 2.64566e-3 | 8.40448e-9 | 5.22192e-8 |
| 64^3 | 12 | 2.11266e-3 | 2.65304e-3 | 1.51521e-11 | 5.22481e-8 |
| 80^3 | 16 | Not evolved | 1.11222e-3 | 5.87975e-15 | Not evolved |

The near-identical density drifts/eigen-residuals at 32/64, and smaller residual in
the larger box, support a finite-periodic-volume floor rather than an unresolved
stationary profile. The higher mesh suppresses near-Nyquist power. The 64^3 sampled
mass is 3.882765254016659, agreeing closely with the independent radial reference.
A larger volume, longer evolution and more general moving/interacting states still
need application-specific checks. This is not isolated-boundary solver validation.

## Small-case resource measurement

The analytic 24^3 example uses 884,736 accounted persistent bytes per solver:
three full complex buffers and two double real arrays. This excludes opaque FFTW
plan/work allocation, caller-owned outputs and runtime overhead. Forty static
self-gravitating steps perform 240 complex FFTs. Diagnostic/force transforms are
counted separately. The same script records a fresh-child process RSS measurement;
one x86_64 macOS sample was 3,469,312 bytes for the complete small example, which
holds two solvers. Timings vary (roughly a tenth of a second for the static loop
on this host); the raw logs include binary/source hashes. These are resource
observations for a small case, not distributed production benchmarks.

Remaining scope includes production volume/resolution/timestep and de Broglie
convergence, direct axion-transfer IC preparation, self-consistent live-CDM stepping,
longer soliton/collision and cosmological benchmarks, nonlinear alias control,
self-interactions and isolated/absorbing boundaries. Long-double evolution is not
in the validation matrix; checkpoint version 1 rejects long-double fields.
