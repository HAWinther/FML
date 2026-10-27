# Changelog

All notable changes to FML. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The entries below are relative to upstream commit `f9f666a` (April 2026).

## [Unreleased]

### Utilities: FileUtils, RandomGenerator, FoFBinning, MemoryLogging, Timings, ParameterMap, Matplotlibcpp
- `FileUtils::read_regular_ascii` (used by COLASolver for the input P(k) / transfer file): a comment line or a token
  like `nan` after the first data line made it loop forever (adding the same row until memory ran out), a first line
  with more columns than `ncols` wrote out of bounds and a later line with another number of columns silently shifted
  the data. Both readers now read line by line: blank lines, lines starting with #, ! or / and everything after a #
  are skipped, nan / inf / denormals are read, and a wrong number of columns or a token that is not a number gives an
  error with the line number. `loadtxt` returned an empty row for a blank line (callers then read out of bounds) and
  threw "stod" for values below 2.2e-308. `read_file_and_extract_two_columns` was declared without the filename (could
  not be called). `read_regular_ascii_subsampled` always kept the first line.
- `RandomGenerator`: `set_normal_sigma` did nothing for the GSL generator (the argument shadowed the member: rms 1
  instead of 10), and assigning a GSL generator to another (`b = a`) made them share the state (copies and `clone`
  were fine). Documented that `std::normal_distribution` (mt19937 + normal numbers) differs between standard
  libraries, and that seeds 0 and 1 are the same for GSL's ranlxd1. `GaussianRandomField::
  generate_white_noise_field_real` printed every value of the first slice (debug output).
- `FoFHalo` (FoFBinning.h): `get_velocity_dispersion()` did not compile (returned an array as a double), and the
  velocity dispersion was NaN when <v^2> - <v>^2 rounded below zero (all members with the same velocity: 981 of 3000
  components in a test); now clamped at 0.
- `MemoryLog` (COLASolver builds with it): the singleton was created without synchronisation and `add_label` searched
  the allocation map before taking the lock (data races with allocations from OpenMP threads). The limit of 100000
  allocations counted all allocations ever made, so long runs stopped logging; it now counts the allocations alive
  at the same time (the memory-vs-time history is capped separately). When the log stops, `print` still shows the
  resident set size. Getters for the memory in use, the peak, the number of allocations, labels.
- `Timings`: a second `EndTiming` without a new `StartTiming` added the time again; it now adds nothing and returns 0.
  `Timings` is copyable (the mutex made every class holding it non-copyable) and has `GetTotalTime`.
- `ParameterMap::get(name, fiducial)` returned the fiducial value when the parameter existed with another type (an
  int 5 read as double gave the fiducial value); it is now an error with readable type names. Missing includes.
- `Matplotlibcpp.h`: without numpy (`-DWITHOUT_NUMPY`, as in the Plotting example) the first call to `plot`, `xlim`
  etc. crashed (segfault: Python was called before the interpreter was started); all functions now start it first.
  `Py_SetProgramName` (deprecated from Python 3.11) is only called for older versions, and `register` is no longer
  defined away for code that includes the header.
- `tests/test_utils.cpp` (21 checks; the old code hangs in the first one).
- Regression: identical except the FoF halo positions and velocities in fof_two_species_hdf5 (1 ulp) and the SO
  velocity dispersion of halos with a single particle in R_200m (0 vs 1e-6 km/s): floating-point contraction (FMA)
  differences from the changed code layout; with -ffp-contract=off the outputs are identical.

### FFTLog fixes
- The output grid was one log-step off: the transform b[n] was labelled with r[n] = kcrc e^{-L} / k[0] ... instead
  of kcrc e^{-L + L/N} / k[0] ..., so xi(r) was evaluated at the wrong r (and normalised with the wrong (2 pi r)^-1.5).
  For P(k) = exp(-k^2) with N = 1024 over k = 1e-4 - 1e4 the error was 2.7% (ell = 0, 0.05 < r < 10); for an LCDM
  P(k) with N = 4096 xi(r = 100 Mpc/h) was 1.6% too high and the BAO peak 0.45% too small in r. Now 3e-9.
- The bias q != 0 gave wrong results (the phase of the coefficients used arg Gamma(x_+) - arg Gamma(x_-) instead of
  the sum). It now works, and `ComputeXiLM` takes the bias as a new last argument (default 0, as before): with
  q = -1 xi_2 and xi_4 are accurate to 1e-14 also at small r (with q = 0 they ring there).
- log Gamma of the coefficients was computed as log(Gamma(z)), which under/overflows for |Im z| >~ 470 (many points
  over few decades, e.g. N = 8192 over 8 decades): now the log of the Lanczos formula (and a stable reflection).
  The public `DiscreteHankelTransform` (with precomputed coefficients) was declared with a different signature than
  it was defined with and could not be linked.
- `tests/test_fftlog.cpp` (xi_ell for ell = 0, 2, 4 of exp(-k^2) against the analytic result, N = 8192, the round
  trip and precomputed coefficients; 5 checks failed before).

### Emulator (EuclidEmulator2) fixes
- A missing data file gave a segfault (`open` returns -1, `!fp` was tested; the size and `mmap` were not checked),
  more than 101 redshifts overflowed a buffer, copies of the emulator double-freed the splines and the
  accelerators, the memory mapped file and the GSL objects of `Cosmology` leaked, and the header could not be
  included in two translation units (34 multiple definitions). Now errors with a message, `EuclidEmulator` is not
  copyable (COLASolver holds it in a shared pointer), `Cosmology` copies its GSL objects, everything is freed, and
  the globals/functions are inline. The boost is unchanged (to the last digit). Outside the k-range of the emulator
  (0.0087 - 9.4 h/Mpc) a spline of the boost is constant: documented.
- COLASolver passed A_s at its own pivot (`cosmology_kpivot_mpc`) to EuclidEmulator2, which uses 0.05/Mpc: for
  kpivot = 0.002 and ns = 0.96 the non-linear boost (the P_NL column of the P(k) output) was 3-7% off at k = 0.3-3
  h/Mpc. It is now converted (A_s (0.05/kpivot)^(ns-1)); no change for the default pivot 0.05.
- `tests/test_emulator.cpp` (reference boosts, errors, copies; + test_emulator_odr.cpp).

### Spline, ODESolver, Math: numerical core fixes
- `MATH::find_root_bisection(Spline, y_value, xrange)`: a range given as (high, low) became (low, low) and always
  threw "same sign" (y = x^2 = 4 on (5,0)); a root exactly at an end-point of the spline range (e.g. `y_value =
  y(xmax)`) threw instead of returning the end-point. Both fixed; results for the normal case are unchanged.
- `MATH::find_root_bisection(function, xrange, epsilon)`: looped forever when epsilon times the range was below the
  floating point spacing at the root (epsilon = 1e-17 on [1,2], or [1e10, 1e10+1] with epsilon = 1e-7). It now stops
  when the interval cannot be split further (and after 10000 iterations). A NaN at an end-point passed the bracket
  check and gave a wrong root (3.0 instead of 1.25 for sqrt(x-1) - 0.5 on (0,3)); it now throws. The tolerance
  semantics (relative to the range) and the treatment of f = 0 (on the side of xlow, which SphericalCollapse relies
  on) are unchanged and documented, so HaloModel, SphericalCollapse and RecombinationHistory give identical results.
- `MATH::j_ell_array`: j_ell for all ell >= 1 was normalized with j_0 = sin(x)/x, which loses all precision at the
  zeros of j_0: at x = 2 pi j_4 was 5e-13 instead of 0.186, at x = 10 pi j_28 was 70 times too small. It now
  normalizes with j_1 when |j_0| < 0.01 |j_1|. The downward recursion started at lmax when 1.2x (1.6x, 5x) < lmax,
  i.e. assumed j_(lmax+1) = 0: j_10(6.862) with lmax = 10 was 12% off, j_100(84) with lmax = 100 29% off, j_5(1.19)
  with lmax = 5 1% off. It now starts at least at max(lmax, x) + 4 x^(1/3) + 20 (also in
  `Hyperspherical_j_ell_array` for K <= 0). Accuracy vs scipy is now < 1e-12 relative to the envelope for lmax = 5 to
  3000 (was up to 5e-3). The PowerSpectrum Bessel splines (keta_max = 8000, 16 samples per oscillation, ell_max =
  4000) change by at most 5e-11 of the envelope (98.7% of the values bit-identical); the EinsteinBoltzmannSolver
  example output (background, recombination, perturbations, transfer, P(k), C_ell) is byte-identical.
- `MATH::j_ell`: returned 0 for x > 14000 for all ell (|j_ell| ~ 1/x there, libstdc++ `std::sph_bessel` throws for
  such arguments); it now uses the recursion (j_10(20000) = -2.921094e-5 as scipy). Same for the GSL version (x >
  9000, ell < 500).
- `MATH::linspace`: num = 1 returned {xmax} (numpy: {xmin}), num = 0 wrote out of bounds (crash); now {xmin} and {}.
- `MATH::Airy_Ai`: outside the spline range [-50, 5] the leading order asymptotic formulas were used: 0.9% too large
  just above z = 5 (a 0.9% jump at z = 5), 3e-4 below z = -50. It now uses `gsl_sf_airy_Ai` there (in [-1e6, 100],
  the asymptotic formulas outside). Values inside [-50, 5] are unchanged. `HyperSphericalBesselWKB` changes only
  where it uses these arguments, and at the turning point (chi nu = sqrt(ell(ell+1))) it returned NaN (0 within
  ~1e-12 of it, 1% off within 1e-10): w is now kept 1e-6 away from 1 (changes < 5e-5 relative within 1e-6 of the
  turning point, where the WKB approximation is good to ~3e-4 anyway).
- `SPLINE::Spline2D::eval_deriv(x, y, 2, 1)` called a null function pointer (segfault) and invalid combinations like
  (4,-1) silently returned f_x; only (0,0), (1,0), (0,1), (2,0), (1,1), (0,2) are accepted now.
- Splines: too few points (nx = 1 read x[1] out of bounds, nx < 3 for cubic / < 4 for bicubic aborted in GSL) and
  non-monotone 2D axes (GSL abort) now give an error via `throw_error` with the spline name (the "not monotone"
  message printed the previous name). 2D splines with decreasing x and/or y arrays are now reversed like in 1D.
  Copies (copy constructor and assignment) keep the out-of-bounds warning setting. The natural boundary conditions
  (f'' = 0 at the ends: for sin on [0,10] with 101 points f' is 2e-2 off at the ends vs 8e-6 inside) and the
  clamping of x out of range also for the derivatives (deriv_x(x > xmax) = deriv_x(xmax), relied on near the ends by
  e.g. HaloModel `powerlaw_extend_logY_of_logx_spline`) are unchanged and documented.
- Splines with OpenMP: in an inactive (one-thread) parallel region nested in an active one (a library `omp parallel
  for` called from a user's parallel loop) all threads got thread number 0 and shared one GSL accelerator (data race).
  The accelerator is now picked with the thread number of the active level. Results are bit-identical.
- `ODESolver`: implicit steppers (rk1imp, rk2imp, rk4imp, bsimp, msbdf) used without a Jacobian with the
  `std::function` interface got a dummy Jacobian that never set dfdy, dfdt (uninitialized memory, valgrind); the
  function pointer interface would call a null pointer. Both now give an error. The GSL driver is freed if the ODE
  function throws (leaked before). Documented that `get_data()` is data[ix][icomponent], that the derivatives are
  computed from the ODE function, and that rkf45/rk8pd are much more efficient than the fiducial rk2 (y'' = -y on
  [0,1000]: 350000 vs 15000 function calls); the fiducial stepper is unchanged.
- Tests (`tests/test_numcore.cpp`, 26 checks with MPI / 31 without, < 1 s): the cases above against scipy/analytic
  values, plus four bad-input modes that must stop with a clear message. With the old code: 9 FAIL, then a segfault
  in `linspace(.., 0)`; the bad-input modes gave two GSL aborts, a segfault and one silent run.
- Regression: no change in test_core, test_halomodel, test_sphericalcollapse, test_survey, test_fnl and
  test_reconstruction (identical output with 1 and 2 tasks); test_pofk and test_paircount differ only in round-off
  digits (1e-16, 1e-14) that also change between runs of the old code (thread reduction order).

### RamsesUtils fixes
- `read_ramses` with `keep_only_particles_in_domain = false` returned an empty vector (the particles were written
  beyond the size of the vector, which was only reserved). `read_ramses_single` allocated the whole file and filled
  only the particles in the local domain: with MPI the rest were zero particles (id 0 at the origin). A CPU file
  without particles crashed the reader (division by zero) and a particle at exactly x = 1 was dropped on all tasks.
- The boxsize and velocities assumed Mpc = 3.08e24 cm: for newer RAMSES versions (3.0857e24, e.g. TestData/
  ExampleRamsesSnapshot) they were 0.18% too large (boxsize 1001.84 instead of 1000 Mpc/h). Both are now computed
  from unit_l, unit_t and aexp, which is right for either constant.
- The `header_XXXXX.txt` file is optional: without it the format is taken from `part_file_descriptor.txt` (newer
  RAMSES), otherwise the default/set format is used (the module's example could not read the bundled snapshot).
  `write_icdeltab` is inline (the header could not be included in two translation units); `get_npart` returns size_t.
- `tests/test_ramses.cpp` (synthetic snapshots written by the test: both Mpc constants, header file or
  part_file_descriptor.txt, an empty CPU file, a particle at x = 1, 1-4 tasks). The old code crashed on the empty
  CPU file.

### CAMBUtils fixes
- The total-matter transfer function was the matter-weighted sum with a constant OmegaMNu, wrong while the neutrinos
  are not fully non-relativistic: P_total / P_cb was 0.2% off at z = 51 and 1.9% at z = 200 (0.4 eV) compared with
  CAMB. It now uses the total column of the tables (now 6e-6). P_cb and P_nu are unchanged to the last digit (the
  total cancels there). In COLASolver this changes only the linear total-matter P(k) column and the "total"
  species transfer function of the outputs, for runs with massive neutrinos.
- Above the largest k of the tables T(k) was constant, so P(k) grew as k^ns (P(5 kmax)/P(kmax) = 4.7); it is now a
  power law with the slope at kmax. Growth rates requested outside the redshift range of the tables were
  extrapolated with the 4-point Lagrange polynomial (f_cb = 1.07 at z = 332 for tables up to z = 200); now the value
  at the end of the table (with a warning). `get_zmin_splines` returned the largest redshift. An infofile not
  ordered in redshift aborted in GSL; the files are now sorted (with a message). The header could not be included
  in two translation units (33 multiple definitions): everything is inline now.
- The massive neutrino growth rate from the 60-redshift tables (input/camb_data_*_zmax200) is 8-66% off at z >= 150
  and k >= 0.05 h/Mpc (T_nu oscillates in time faster than the table spacing resolves; fine for z <= 49): documented,
  with a warning above z = 50 (it is used for the initial velocities of neutrino particles, where T_nu/T_cb is small).
- `scripts/make_camb_transfer_tables.py` wrote the redshifts to the infofile with 3 decimals: the growth rates at
  z = 0.15-1 were 0.3-0.6% off (the 4-point Lagrange derivative used slightly wrong redshifts). Now 10 significant
  digits (the existing tables in input/ are not regenerated: they are regression inputs).
- `tests/test_camb.cpp` (P_cb and P_total/P_cb vs CAMB at z = 0, 51, 200, k > kmax, z outside the tables, an
  unordered infofile; 4 checks failed before and the unordered infofile aborted).
- Regression: the total-matter P(k) of the neutrino cases changes by 1e-6 - 1e-5 at z <= 4 and 1e-3 at z = 20, and the
  `D1_Tk` columns of the gravitymodel output at k = 10 h/Mpc for nbody_gauge (tables up to k = 2.1 h/Mpc) by 2.5e-4;
  everything else (particles, cb/cdm/baryon P(k), snapshots) is bit-identical. The SO velocity dispersion of FoF
  halos with a single particle inside R_200m (exactly 0, the round-off of v^2 - v^2) changes from 0 to 1e-6 km/s
  with the new build (no other halo property changes).

### GadgetUtils fixes
- Reading a file with several particle types into particles with `set_family` while keeping only the particles in
  the local domain (MPI) set the type with the indices of the whole file: writes past the end of the vector and wrong
  types (2000 of 5000 particles with 2 tasks, 3991 with 4). The type is now set when the particle is added.
- `write_gadget_single` (binary and HDF5) divided the mass of every type by the number of type 1 particles: with
  types 0/1/3 of 1000/3000/1000 particles the MassTable of types 0 and 3 was 3x too small (69384 and 34692 instead
  of 208152 and 104076 x 1e10 Msun/h). Now the number of particles of the type. In the binary writer particles with
  `get_family` and the default OmegaFamilyOverOmegaM got NumPart_Total = 0 and MassTable = 0 (the totals are now
  summed over the tasks as in the HDF5 writer: with families the call is collective). Particles with a family
  outside 0-5 made the writer abort on an assert (or write garbage with NDEBUG); they are now skipped.
- IDs were written as 32 bit int: an ID of 3000000000 was read back as -1294967296. The writers now use 8 byte IDs
  if some ID does not fit in an int or there are more than 2^31 particles (no MPI calls; 4 bytes otherwise, as
  before). The reader takes the ID size (4 or 8), and float or double for POS/VEL/MASS, from the block sizes, so
  files with 64 bit IDs can be read.
- The reader guessed the bytes per particle from the file size including the header: files with 288 or fewer
  particles were rejected and a file with no particles crashed (division by zero). The block sizes are now checked
  against the header and files with no particles are skipped.
- New in the reader: format 2 (SNAP) files (labelled blocks, unknown blocks are skipped) and the MASS block (types
  with MassTable = 0; stored with `set_mass` in 1e10 Msun/h, only for these particles). Files written by
  `write_gadget_species` with per-particle masses (COLASolver mass-encoded baryon mode) could not be read back.
- The endianness was only detected once: a reader used for a big-endian and then a little-endian file aborted. The
  HDF5 reader kept the number of files of the first snapshot it read (a 2-file snapshot after a 1-file one: 300 of
  600 particles, no error). `BoxSize` with 3 elements (SWIFT) is accepted by the HDF5 reader.
- Blocks larger than 2^31 - 1 bytes (about 179M particles per file) now give a clear error instead of a wrong block
  size. `GADGET_ONLY_DM` applies to the reader and the writer (`GADGET_ONLY_READ_DM`, used by the writer before,
  is the same). `write_section` takes the size as `size_t`.
- Tests: `tests/test_gadget.cpp` (all the cases above, binary and HDF5, 1-4 tasks; on the old code every part except the plain dark matter round trip fails
  or aborts). The `MassTable` check of `GadgetUtils/example/TestHDF5.cpp` expected the old (wrong) ratio.
- Output: COLASolver snapshots (binary and HDF5, one or several species, with and without masses) are bit-identical.
  Changed: the MassTable of types other than 1 and NumPart_Total for particles with `get_family`
  (`write_gadget_single`), and files with IDs >= 2^31 (8 byte IDs instead of truncated ones).

### Global: FFTW threads, sub-communicators and smaller fixes
- FFTW was initialized before `init_fml` had counted the OpenMP threads, so with `USE_FFTW_THREADS` (the default for
  COLASolver) FFTW made all its plans with 1 thread while the startup info said thread support was enabled. FML is now
  initialized first (and `init_fftw` uses the OpenMP thread count itself). 256^3 r2c + c2r 1.7x faster on 2 cores;
  COLASolver force calculation ~20% faster on 2 cores (64^3 particles, 128^3 force grid), more with more cores.
  Results unchanged.
- `init_fml` reduced the CPU-affinity check over `MPI_COMM_WORLD` instead of `FML::communicator`, so running on a
  sub-communicator (as documented in Global.h) hung.
- `assert_mpi` can be used outside namespace FML (`assert_mpi_impl` was not found).
- `SystemMemory.h` functions are inline (including the header in two files failed to link).
- `create_folder`: all tasks get the result and throw together (task 0 threw while the others waited in a barrier).
- `GatherFromTasks` (and the fallback of Min/Max/SumOverTasks) static_asserts that the type is trivially copyable (it
  sends the bytes). The `MultigridSolver` generic example's `MyType` had a user-written copy constructor and did
  not compile with this check: it is now defaulted (the grids are sent between tasks as bytes).

### COLASolver: linear growth with massive neutrinos
- The scale-dependent LPT growth factors (used by scale-dependent COLA and for the `Pcb_linear` column of the P(k)
  output) evolved the neutrinos with a fluid approximation with a free-streaming scale: for 0.4 eV the growth of
  delta_cb from z = 49 to 0 was 2% too large at k = 0.01-0.1 h/Mpc (4% in P(k)) compared with CAMB. They now use the
  neutrino transfer functions directly in the source (fnu T_nu(k,a) / T_cb(k,a)) and the CAMB growth rate at the
  start of the integration; the growth agrees with CAMB to 1e-4. (A Newtonian integration with the CAMB background,
  initial conditions and neutrino source reproduces CAMB to 1e-4, so linear neutrinos in the force are exact.)
- The initial 1LPT velocities used the growth rate at k = 0 for all k (0.998 at z = 49 for 0.4 eV, where neutrinos
  cluster) instead of f(k, zini) (0.976 for k above the free-streaming scale): +1.7% in P_cb(z = 0) on linear scales
  for runs without scale-dependent COLA. With scale-dependent growth (massive neutrinos with transfer functions, or
  modified gravity) the velocities now use f(k, zini); with scale-independent COLA the particles get the residual
  (f(k) - f(0)) Psi relative to the COLA frame.
- 64^3 particles in 256 Mpc/h, 0.4 eV linear neutrinos at linear amplitude, growth z = 49 -> 0 vs CAMB at k = 0.03
  h/Mpc: scale-dependent COLA +4.0% -> -0.1%, PM (Quinn) +1.5% -> -0.2%, scale-independent COLA +2.2% -> +0.9% with
  10 + 10 steps (0.0% with 100 + 100 steps). (Round 35 reported -2.5% / -1.9% / -0.1%: that comparison used the
  `Pcb_linear` column, which had the error above.) All methods still fall below linear theory at k > 0.1 h/Mpc (-3%
  at 0.125, -20% at 0.37), unchanged with a 2x finer force grid: particle discreteness of the 64^3 lattice.
- Regression: the neutrino cases change by -1% (PM, particle neutrinos), -3.3% to -0.7% (two species,
  scale-dependent COLA, k = 0.03-0.1) and +0.1% (scale-dependent COLA, linear neutrinos) in P(k), and the
  `D1mnu` columns of the gravitymodel output (now T_nu(k,a) / T_cb(k,zini)); f(R) and mu(a,k) cases change by < 1e-5.

### MultigridSolver: DGP error handling, symmetron tests, examples
- `DGPSolverCosmology`: a negative discriminant threw an exception inside the OpenMP loop (the program terminated
  without unwinding). It is now set to zero while iterating and the converged solution is checked: the run stops with
  a message if it is negative anywhere. For the normal branch the discriminant is > 1/9 for any rc, a and field when
  delta >= -1, so this only happens for a bad density field.
- The mean-residual option only uses the arithmetic every solver type supports (the generic example with a
  user-defined type did not compile).
- Tests (`tests/test_multigrid.cpp`): symmetron (zero field before symmetry breaking, linear theory for a small cosine
  perturbation to 5e-7 of the amplitude, screening in a halo with the field below the vacuum value) and DGP with a
  clustered field. Regression case `symmetron_exact_multigrid`. In COLA (64^3 in 200 Mpc/h, a_ssb = 0.5, L = 4
  Mpc/h) the exact symmetron P(k) agrees with the approximate screening method on large scales (+4.2% over GR at
  k = 0.07 h/Mpc) and is more screened on small scales (1.31 vs 1.40 times GR at k = 1 h/Mpc).
- All examples build and run (the `test` target in the examples Makefile referred to a file that does not exist and
  is removed).
- COLASolver symmetron: `GeffOverG` was NaN at k = 0 when a = assb exactly (0 / 0: the background field and the mass
  both vanish), which made the run fail with NaN positions when a time step landed on assb (with the exact solver or
  the screening method). It is now 1 when the background field is zero.
### HaloModel fixes
- The NFW Fourier transform was integrated with equal weights across the jump of the profile at Rvir (first order,
  0.4-2% errors that did not decrease with the accuracy setting) and tabulated on 75 x 75 nodes over 16 decades in R
  and k (up to 3.5% in P(k) at k > 10 h/Mpc); the one/two-halo integrals used 200 points in nu (1% at k = 20-100).
  Now the analytic transform for the (default) NFW profile (`use_analytic_nfw_fourier`, `NFWHaloDensityProfile`),
  the trapezoidal rule up to Rvir for other profiles, 400 x 400 nodes and 600 points in nu. HMCode P(k) vs CAMB
  (HMcode 2016) for z = 0-2: max error 0.11% for k <= 100 h/Mpc (was 1.7-2.9% at k > 10); vs the example's HMCode
  output: 0.34% (was 1.6%). Without HMCode P(k) changes by up to -3.8% at k = 10-100 (z = 1), now converged to 0.1%.
- M(R) used the physical mean density with a comoving R: halo masses were (1+z)^3 too large at z > 0 (n(M) wrong by
  factors 7 to 7500 at z = 1, also the R, Rvir and nu columns of output_nofM and the power-law c(M)); P(k) unchanged.
- cofM_multiplier was applied again on every compute_at_redshift call (cmin x1.5 per call). The values used are now
  cmin_current and cofM_0_current.
- Robustness: compute_at_redshift aborted at high z (the R where nu = 1 for neff out of range), z beyond the spherical
  collapse splines (z > 10) silently used z = 10 (now an error; `zmax_spherical_collapse`), output_pofk beyond
  [kmin_pofk, kmax_pofk] gave a wrong P(k) (now an error), read_pofk_file aborted in GSL on P(k) = 0 (now a clear
  error) and estimated the end slopes outside the data range, a HaloModel could not be copied (the timer), and the
  n(M) output header said dlogn/dlogM for dn/dlnM.
- `tests/test_halomodel.cpp` (9 checks failed before and 2 did not compile).

### SphericalCollapse fixes
- DeltaVir in curved models: the virial energy counted curvature as dark energy (OmegaDE = 1 - OmegaM(z)), -4.2%
  (z = 0) and -2.8% (z = 1) for OmegaM = 0.3, OmegaK = 0.7. The force of the smooth components is now taken from the
  background acceleration (2(1 + dlogE/dloga) + OmegaM(z) = -sum OmegaDE (1 + 3w)), which also handles several dark
  energy components and radiation in E(x).
- The initial conditions of the non-linear equation were the linear growing mode only: deltac was biased by ~0.8
  delta_ini (+2.1% at z = 10 when starting at a = 1e-3, the default of the free functions, and 2.08 at z = 100). Now
  the second-order growing mode (delta + 17/21 delta^2); the default start of the free functions is now a = 1e-5
  (as the classes). EdS deltac = 1.686470 and DeltaVir = 18 pi^2 for any collapse redshift.
- Tolerances: the absolute ODE tolerance (1e-6) was large compared with delta_ini ~ 1e-5 (delta_ini ~1% too high)
  and the bisection tolerance of delta_ini (1e-8) biased deltac by ~2e-4 and DeltaVir by ~4e-4 (with noise in
  deltac(z)). Now 1e-12 and 1e-13. LCDM deltac and DeltaVir agree with an independent integration to ~1e-6.
- Robustness: the search for the virialization time could loop forever, a failed bracket for delta_ini (no collapse
  in time) returned garbage silently, the linear solution was corrupted after the collapse (-16%), mu <= 0 was
  checked after the computation and exceptions inside the OpenMP loop aborted the program. Copies of the default
  LCDM SphericalCollapseModel referred to the original object (OmegaM of the original, dangling after it is gone).
  compute_sphericalcollapse_splines and SphericalCollapse::run_at_all_redshifts take an optional zmax (default 10).
- `tests/test_sphericalcollapse.cpp` (EdS, LCDM, open, w0wa + mu against analytic/independent results; 12 checks
  failed before and the error checks aborted the program).

### MultigridSolver: f(R) in terms of sqrt(|f_R|) (the new default)
- `FofrSolverCosmology::set_method("sqrt")` (the default) solves for u = sqrt(|f_R|) (Bose, Hellwing & Li 2017). The
  equation in a cell is solved exactly in the Gauss-Seidel sweeps: for n = 1 it is a cubic with exactly one positive
  root, otherwise it is monotone in u (safeguarded Newton, ~2-3 iterations), so f_R stays negative on every level.
  `"log"` is the original ln|f_R| with Newton steps. On a clustered 64^3 field, of 48 cases (|f_R0| = 1e-4 - 1e-6,
  n = 0.5 - 4, a = 0.1 - 1) sqrt converged in all and log gave NaN in 29 (and did not converge in 200 V-cycles in one).
  In COLA (64^3 particles, 128^3 force grid) log fails at the third force calculation for |f_R0| = 1e-6. Where both
  converge they agree to 0.1-0.5% in the fifth force (different discretizations of the kinetic term) and 0.3% in P(k);
  for n = 1 the cost is the same, otherwise ~1.2-1.5 times that of log.
- The log method computes its kinetic term directly (it built two `std::function`s per cell evaluation).
- A non-finite solution now stops the run with a message (it went on with NaN forces); the min/max of f_R printed are
  over all tasks.
- `MultiGridSolver::set_custom_update`: a pointwise solve of the equation in a cell to replace the Newton update.
- COLASolver: `gravity_model_fofr_multigrid_method` (`sqrt` or `log`). The regression case `fofr_exact_multigrid`
  changes by < 1e-3 in P(k).
- Based on a contributed patch (sqrt method, cubic solver and update hook).

### MultigridSolver: fixes and faster convergence
- Prolongation: the N-linear interpolation of the coarse-grid correction summed only one of the neighbors (`iAdd =`
  instead of `+=`) and used vertex-centered weights on the cell-centered grids. It is now cell-centered N-linear
  interpolation (weights 3/4, 1/4 per direction) with the neighbors communicated across tasks (the residual grid,
  used for the correction, now has one extra slice on each side). Poisson: the residual is reduced by 0.14 per V-cycle
  with 2+2 sweeps (was 0.33); f(R): 0.12 (was 0.15) and 0.03 with 10 sweeps (was 0.04).
- The residual was computed after every sweep on the main grid (and twice after the last one) and after each
  prolongation even when not verbose; it is now only computed when needed (verbose output, the last sweep). A V-cycle
  takes ~55-60% of the time it did, and fewer V-cycles are needed.
- `get_Gradient(level, index, order)` and `get_Gradient2`: with MPI the x-derivative used the cell x-1 for all points
  of the stencil (zero derivative); the order-2 first derivative had the wrong sign of its last weight.
- New `set_cube_ordering(true)`: the Gauss-Seidel sweeps color a cell by the parity of each coordinate (2^NDIM colors)
  so no cell in the 3^NDIM cube around a cell is updated at the same time. Needed for equations with mixed
  derivatives: with the chess-board ordering the result depended on the number of threads and tasks.
- New `set_subtract_mean_residual(true)` for equations where L(f + c) = L(f): solves L(f) = const (the constant is the
  mean of L on each level). The discretized DGP equation has no exact periodic solution (the non-linear term does
  not have exactly zero mean), so its residual stalled at ~1e-2 - 1e-6 and the exact DGP solver in COLASolver never
  reached the tolerance (1000 V-cycles per force calculation, ~12 minutes for a 64^3 grid; the force was accurate to
  ~3e-4). `DGPSolverCosmology` uses both options and now converges in ~5 V-cycles; the P(k) agrees with the approximate
  screening method to 0.3%.
- The coarsest level is swept until its residual is reduced by 10 (with MPI it has one slice per task, so with 16
  tasks the convergence factor was 0.78 per V-cycle; now 0.14 for any number of tasks).
- `MPIMultiGrid::restrict_down` is OpenMP parallel; the level check for a single task was the MPI one.
- `FofrSolverCosmology`, `DGPSolverCosmology`, `SymmetronSolverCosmology`: the density multigrid was constructed with
  the arguments shifted (`nleft` as `periodic`); same result for the values used.
- `tests/test_multigrid.cpp` (run by `make test`, 1/2/4 tasks): derivatives, Poisson convergence, DGP-type equation.
  Regression cases `fofr_exact_multigrid` and `dgp_exact_multigrid`.

### ComputePowerSpectra: one-loop hybrid bias spectra with the LPT kernels of modified gravity
- `HybridBiasOneLoopMG.h`: `oneloop_hybrid_bias_spectra_mg` gives the one-loop LPT prediction of all 15 hybrid bias
  spectra with the full scale- and time-dependent kernels (as in Aviles & Cervantes-Cota 2017; Brando, Li & Koyama
  2026) of a general mu(k,a) model (`mgmodel_mu`), Hu-Sawicki f(R) (`mgmodel_fofr`) and normal-branch DGP
  (`mgmodel_dgp`), with or without screening, or LCDM (`mgmodel_gr`), for several output times at once. The kernels
  are derived directly from the Eulerian quasi-static equations through T Psi(q) = int_k (ik/k^2) Sigma(k)
  e^(ik.(q + Psi(q))) (set partitions of the legs), which gives the frame-lagging terms and also the transverse part
  of the second order displacement that a scale-dependent force generates (it enters the third order kernel and the
  one-loop correlators). The growth ODEs are solved for every node of the loop integrals (no interpolation of the
  kernels). `mg_linear_growth` gives D(k,a).
- Tests: with EdS kernels all 15 spectra agree with `oneloop_hybrid_bias_spectra` to 1e-4; the second and third
  order kernels agree with the Eulerian SPT kernels of the same equations (mapped to LPT) to 1e-5 for f(R) with and
  without screening, DGP and a mu(k,a) model; LCDM kernels give the expected +0.9% (2nd order) and +1.9% (3rd order)
  corrections to EdS; the second order loop functions agree with mgpt to 0.1% (f(R), unscreened DGP).
  `tests/test_hybridbias.cpp` (run by `make test`) checks the EdS limit and the third order kernels against stored
  Eulerian values.
- `oneloop_hybrid_bias_spectra` (EdS): the loop spectra are no longer interpolated across the cut (a node on each side
  of it), which changes the nabla^2 delta spectra near the cut by up to 1.5e-3 (now converged to 1e-4), the
  refinement of the R-type integrals towards p = k now goes towards p = k (it went the other way), and R1, R2 at r = 1
  are no longer NaN. The other spectra change by less than 1e-4.
- COLASolver: `hybridbias_oneloop_kernels` (`EdS`, the default, or `model`: the kernels of the gravity model; f(R)
  and DGP use their screening setting, the other models G_eff(k,a) without screening, GR the LCDM kernels).
- Regression cases `scaledep_cola_fine_force_grid` (scale-dependent COLA with an offset lattice: in GR identical P(k)
  to `fine_force_grid`) and `hybridbias_zeldovich`.
### Reconstruction, Hessian and smoothing fixes
- `RSDReconstructionFourierMethod` removed the wrong RSD shift: (Psi.r)r/(1+beta) instead of beta (Psi.r)r, and
  each iteration subtracted another shift from the already shifted positions, so more iterations made it worse
  (Zeldovich test with beta = 0.5: removed / put in = 1.6 after 1 iteration, 3.1 after 3, 4.3 after 8). It now uses
  the iterative scheme of Burden et al. (2015): the shift of iteration n is beta (Psi_n(s).r)r with Psi_n from the
  density of the positions of iteration n - 1 and always applied to the observed positions s (removed / put in =
  0.96 after 3 iterations, rms residual 27% of the RSD shift; stays within 0.8-1.1 with more iterations). The
  particle count is no longer stale after particles move between tasks. Survey data (`survey_data = true`) was not
  implemented (the line of sight ignored the observer) and now aborts with a message.
- Hessian: `SymmetricTensorEigensystem` shared one set of GSL workspaces between the OpenMP threads (wrong
  eigenvalues/eigenvectors with more than one thread); `ComputeHessianWithFT` allocated N(N-1) grids (out of bounds
  in 2D, N(N+1)/2 now), and the mixed derivatives of the Nyquist modes are set to zero. The eigenvector layout is
  documented (component i of eigenvector k is `eigenvectors[i * N + k]`).
- Smoothing: the 2D tophat filter was 2(1 - cos kR)/(kR)^2 instead of 2 J1(kR)/kR (Bessel J1 from Abramowitz &
  Stegun, error < 1e-7, as `std::cyl_bessel_j` is missing in libc++). `compute_grid_PDF` added to old values when the
  output vector was reused, left out the cell(s) with the largest value, divided by zero for a constant grid and could
  index past the per-thread arrays.
- `tests/test_reconstruction.cpp` checks these against analytic results (10 checks failed before and the 2D Hessian
  crashed). COLASolver output is unchanged (regression: 1168 files identical).

### ComputePowerSpectra fixes
- Shot noise for particles with masses or delta-f weights: P(k), the multipoles and the mono-/bispectrum subtracted
  1/N, now sum m^2/(sum m)^2 (sum w^2/N^2 for delta-f weights) and for the bispectrum S2 (P1 + P2 + P3) + S3 with
  S3 = sum m^3/(sum m)^3 (new helper `particle_shotnoise`). Unchanged for equal-mass particles. Masses uniform in
  [0.2, 3]: 1/N left +20% of the shot noise in P(k); delta-f weights uniform in [-1, 1]: -200%.
- Bispectrum: the bins with fewer than (2pi)^3 ~ 248 triangles were set to zero (the threshold of 1 triangle was
  applied to the count times the measure 1/(2pi)^3). The bin counts now match a brute-force count of the triangles
  (also 14 small bins of the test). COLASolver bispectrum output changes for those bins (they had B = 0).
- NGP with interlacing on one MPI task lost the particles in the last half cell (the shifted particles were put in
  an extra slice that is only folded back with several tasks): mean density contrast -0.06 in the test.
- Multipoles from a Fourier grid: without OpenMP the mode counts were multiplied by the number of MPI tasks (summed
  over the tasks twice).
- The P(k) bin index truncated towards 0, so k up to a full bin below the first bin was put in the first bin; now -1.
- PolyspectrumBinning: the binning type passed to the constructor was not stored (always linear in the member),
  `reset()` did not clear `bincount_is_set` (a later estimate used the zeroed N123), at least 2 bins are required,
  and the reduced spectrum of an empty bin is 0 (was NaN).
- `compute_LPT_growth_factors_*` in DisplacementFields.h are inline (ComputePowerSpectrum.h could not be included in
  two translation units), the declaration and definition of the direct summation agree (`const T *`),
  `FML::power` returns 0 for a negative exponent instead of recursing forever, and `FFTWGrid<1>` (which crashed in
  the constructor) is a compile-time error.
- `tests/test_pofk.cpp` checks all of these against brute force (10 checks failed before). COLASolver output is
  unchanged (regression: 1168 files identical).

### Survey: GalaxiesToBox fixes
- `GalaxiesRandomsToBox` with `shiftPositions` added the minimum instead of subtracting it: for a survey not in the
  first octant the points ended up outside [0,1). `GalaxiesToBox` shifted (and scaled) the observer once per galaxy
  (wrong observer for more than one galaxy). The largest redshift (for the r(z) table) was found with a data race over
  the threads (a galaxy beyond the table could get a wrong distance), and the box extent and z range were per MPI task
  (different boxes on different tasks); both are now reduced over the threads and tasks. Scaling without shifting and
  a zero-extent box are now caught by an assert. `tests/test_survey.cpp` checks |x - observer| * boxsize = r(z)
  against independent values, all points in [0,1) and the same box on all tasks (5 checks failed before).

### Particles with default member initializers
- `MPIParticles::create_particle_grid` and `RamsesReader::read_ramses` took the dimension from `GetNDIM(T())` in a
  constant expression, which does not compile for a particle class with a default member initializer (e.g.
  `double mass{1.0};` next to an uninitialized `pos[3]`). They now use a local object (as PairCounting,
  ParticlesInBoxes and KDTree); no `constexpr ... GetNDIM(T())` is left in the library. `tests/test_core.cpp` checks
  `MPIParticles` with such a particle.

### PairCounting: new counting engine
- The box and survey correlation functions use a new engine (`PAIRCOUNTS::detail`): both catalogues on one grid
  (cells of ~rmax/2, at least ~2 particles per cell), positions/weights sorted by cell in flat arrays, loops over pairs
  of cells (neighbour offsets pruned by the cell-cell distance, the periodic shift per cell pair instead of per pair,
  every auto pair once), the binning compiled in (no `std::function` per pair), cells distributed over the MPI tasks
  by estimated cost (instead of slabs) and over the threads. The survey functions put every catalogue on the grid once
  (instead of once per pair type) and take the estimator as a new last argument (default "LZ"; "LS" is an alias).
  `GeneralPairCounter` (with `ParticlesInBoxes`) is unchanged.
- The counts are identical to the fixed old code (all bins, auto/cross/(r,mu)/weights; the example gives the same
  output). Speed with 2 threads, 3D: 1e6 points rmax 0.03 9.0 -> 1.9 s, rmax 0.01 1.4 -> 0.38 s; 2e5 points rmax 0.1
  6.5 -> 2.2 s; (r, mu) with 20 mu bins 3.8x faster; survey with 1e5 data and 2e6 randoms 17.9 -> 6.1 s.

### PairCounting: fixes
- Cross pairs of catalogues of very different size (e.g. data x many more randoms) missed pairs: the search in the
  finer grid started from the cell at the corner of the cell of the coarser grid. 200 x 200000 points with rmax 0.02:
  635 of 1380 pairs; 4000 data x 50 times more randoms with rmax 0.1: -1.4% in the last r bin (the opposite order
  was right). Now the search starts from the cell of the particle. Without a periodic box the search was also capped
  at half the grid (pairs with r > ~0.5 missed).
- The correlation function from the box functions used the 3D shell volume in 1D and 2D (xi of random points 4776 in
  1D and 74 in 2D) and a uniform-in-mu shell fraction in 2D; now the NDIM-dimensional volume and the right fraction.
- Auto pairs: the sign of mu depended on the order the cells are visited (N(mu > 0) - N(mu < 0) = -13% of the
  pairs). Every auto pair is now counted half at mu and half at -mu (folded |mu| counts unchanged).
- Out-of-bounds writes: a pair at r (mu) just below rmax (mumax) could get the bin index nbins (clamped now), and
  `FromAngularCorrelationToMultipoles` with one multipole. mu = 1 is included in the last bin when mumax = 1 (1D).
- The header can be included in several translation units (inline functions and variables); particles with default
  member initializers compile (also in ParticlesInBoxes and KDTree: the dimension is no longer taken from `T()` at
  compile time); the survey D1R2 counts use the type of the second randoms.
- New `tests/test_paircount.cpp` (brute force; 1/2/4 MPI tasks).

### KDTree: new implementation
- A flat kd-tree (coordinates in one array reordered by node, leaves of 8 points, split at the median of the widest
  dimension, bounding boxes; periodic searches prune with the minimum-image distance to the node boxes, so no
  periodic images are needed) behind the same interface. New: `knearest_indices` (k nearest, with squared
  distances), `neighborhood_indices_and_distances`, `size()`; all queries are const (thread safe). The internal
  node classes (`KDNode`, `KDNodePtr`, `comparer`, `sort_on_idx`) are gone.
- 3D, uniform points, against the fixed old tree (below): memory 51 instead of ~140 bytes per point; radius queries
  with ~30 neighbours 4-10 us instead of 30-58 us, with ~3000 neighbours 0.06-0.17 ms instead of 1.6-3.3 ms;
  building 1e6 / 4e6 points 0.55-0.65 / 3.0-3.7 s instead of 0.9-1.0 / 7-10 s; nearest neighbour the same
  (2-7 us). `DensityProfileBinner` does one MPI reduction instead of one per profile.
- `tests/test_kdtree.cpp` also checks the k nearest neighbours and the tree from particles.

### KDTree: fixes
- The header defined non-inline functions, so it could not be included in two translation units (28 multiple
  definitions at link time). All are now inline.
- Periodic nearest neighbour: every query searched all 2^ndim periodic images from a poor starting bound, which made
  it 50-500 times slower than without a box (0.36 / 2.0 ms per query for 1e5 / 1e6 points). Now the point itself is
  searched first and only the images that can hold a nearer point, with the best distance as bound: 3 / 5 us.
- Periodic radius queries with r > boxsize/2 silently used r = boxsize/2 (1014 of 1926 points for r = 0.7). Now all
  3^ndim images are searched and every point within the minimum-image distance is returned once. The query point is
  wrapped into the box, `neighborhood()` is periodic like `neighborhood_indices/points()`, and `periodic_wrap` can no
  longer return boxsize for tiny negative x.
- `DensityProfileBinner`: the bin index is clamped (r just below rmax could round to nbins).
- New `tests/test_kdtree.cpp` (brute force: 2D/3D, with and without a box, the edge cases above, the profile binner,
  1/2/4 MPI tasks).

### Fixed: scale-dependent COLA with an offset particle lattice
- With `ic_lattice_offset` (the fiducial half force cell when `force_nmesh > particle_Npart_1D`, introduced earlier in
  this cleanup) scale-dependent COLA interpolated its LPT potentials at the offset Lagrangian positions with CIC,
  which smooths the displacements of the COLA frame: P(k) at z = 0 too low already in linear theory, by 1.4% /
  5% / 18% / 33% at k = 0.1 / 0.2 / 0.4 / 0.6 h/Mpc (64^3, 256 Mpc/h, 2x grid; relative to scale-independent COLA
  and Quinn without COLA, which agree to 0.1%), and +8% in the baryon-CDM relative mode with linear neutrinos. Now
  the potentials are interpolated at q - offset (the grid nodes): scale-dependent COLA agrees with scale-independent
  COLA to < 0.01% in GR and with 200 Quinn steps to 0.02% for f(R) and mu(a,k) at normal amplitude. Runs with
  `force_nmesh = particle_Npart_1D` (no offset) are unchanged; no regression case had the offset with
  scale-dependent COLA (new case `mu_bz_scaledep_cola`, and `tests/check_mu_gravity.py` catches it).

### Gravity model `mu`: phenomenological mu(a,k)
- `gravity_model = "mu"` with `gravity_model_mu_parametrization` = `DE` (mu = 1 + mu0 Omega_DE(a)/Omega_DE),
  `BZ` (Bertschinger & Zukin mu(a,k)) or `table` (mu(a) or mu(a,k) from a file, e.g. from hi_class; example
  `input/mu_table_BZ_example.txt`). No screening. Growth factors with mu(a,k) (scale-dependent growth for a
  k-dependent mu).
- Validation (`tests/check_mu_gravity.py`, linear amplitude): P_mu/P_GR = (D_mu/D_GR)^2 from an independent growth
  solution to 0.05% for k <= 0.1 h/Mpc and 0.2-1.2% up to k = 0.6 h/Mpc (PM force resolution), with and without
  scale-dependent COLA; growth factors to 3e-5; tables = parametrisations to 0.001%. Regression cases `mu_de_cola`,
  `mu_bz_scaledep_cola`, `mu_table`; all other files unchanged bit for bit.

### Big Bang starts: the early-time force (resolved open issue)
- The difference between BullFrog runs from z = 1e7 and from 2LPT at z = 19 (+0.4% at k < 0.1, +3% at k = 0.5 h/Mpc
  with the interlaced TSC force on a 1x grid) comes from the interlaced force while the particles are close to the
  lattice: modes with k > ~k_Ny/8 grow too fast at z > ~19. With the NUFFT force the two starts agree (0.02% at
  k = 0.2, 0.3% at k = 0.4 h/Mpc), and with `force_early_method = nufft` for z > 10 the Big Bang start gives the
  result of the 2LPT start with the same force, at almost no extra cost. README table; the code now warns when
  starting at z > 100 without an early force. No change of defaults or results.

### Baryons: mass-encoded baryon-CDM relative mode
- `particle_baryon_relative_mode` (default `displacement`, unchanged): with `mass` both species follow the total
  CDM+baryon flow and the relative density is carried by the particle masses m_i = 1 + eps_i(q), eps_i =
  delta_cb(zini) (T_i - T_cb)(z=0) / T_cb(zini) (compensated, Hahn, Rampf & Uhlemann 2021); `mass_ini` uses the
  relative density at zini and keeps the relative velocity. A particle type with a mass is used only in these modes.
  The density assignment, the per-species P(k), FoF and SO are mass weighted; GADGET binary files get a MASS block and
  HDF5 files `PartTypeN/Masses` (MassTable 0 for those types). Library: optional mass contrast in
  `NBodyInitialConditions` / `generate_species_from_delta_cb`, `GadgetSpecies::get_mass`.
- Validation (64^3 + 64^3, 256 Mpc/h, z_ini = 49, z = 0): P(delta_c - delta_b) / linear = 1.03 / 1.04 / 0.98 at
  k = 0.031 / 0.055 / 0.10 h/Mpc with `mass` (G = 0.98 / 0.98 / 0.86 relative to its own linear run) vs 0.49 / 0.52 /
  0.49 with `displacement` and 0.87 / 0.92 / 0.86 with `mass_ini`; independent of the force grid (4x: unchanged, where
  `displacement` and `mass_ini` break down); P_cb unchanged (0.02%). `mass` is exact at z = 0 in linear theory and
  overestimates the relative mode by 1.5% / 4% / 9% at z = 1 / 3 / 10.
- Warning for `displacement` / `mass_ini` with `force_nmesh > 2 particle_Npart_1D`. New regression cases
  `two_species_mass_cola_fof_hdf5` and `two_species_mass_ini`; all other files unchanged bit for bit.

### Lightcone: potential, Doppler and gravitational redshift
- `plc_relativistic_effects` (default false): during the steps that record the lightcone the potential Phi/c^2 =
  -1.5 Omega_M (H0/c)^2 delta / (a k^2) is computed from the density used for the force (one extra FFT), interpolated
  to the particles and stored with each lightcone particle (+4 bytes per stored particle). HDF5 lightcone files get
  `Potential`, `RadialVelocity`, `CosmologicalRedshift` and `ObservedRedshift` (1 + z_obs = (1 + z)(1 + v_r/c -
  Phi/c^2)), ascii files the same as columns; with onion maps also Healpix maps of Phi/c^2 sampled at the shells'
  mid-radius (`OnionMap_phi_cb_imap<N>` + C_ell). Library-side: `OnionSlice::pixel_direction`, `set_pixel`,
  `sum_field_over_tasks`.
- Validation (512 Mpc/h, 128^3, full sky to z = 0.15): stored Phi vs an independent Poisson solve of the z = 0
  snapshot: correlation 1.0000, slope 1.008 for the last step and 1.011-1.071 for the shells, as expected from the
  decay of D/a (1.012-1.062); maps idem. Phi/c^2 rms 2.3e-5, z_obs - z rms 1.2e-3 (Doppler; gravitational 2.6e-5).
- New regression case `hdf5_lightcone_relativistic`; all other files unchanged bit for bit.

### Neutrinos: energy density in the Poisson equation
- `force_neutrino_energy_density` (default false): the massive neutrinos source the Poisson equation with their
  energy density. Linear neutrinos (`force_linear_massive_neutrinos`): f_nu <E/m>(a) delta_nu (delta_nu from the
  transfer functions is already the energy-density contrast); 0.1 eV: P_cb at z = 0 +0.022% at k = 0.01 h/Mpc, the
  linear-theory estimate. Neutrino particles: each weighted by E/m = sqrt(1 + u^2) (u = q/(a m c)), normalised by the mean <E/m>(a) of the
  unperturbed Fermi-Dirac distribution (checked against scipy to 1e-15). The source is (1 - f_nu) delta_cb +
  f_nu <E/m>(a) delta_nu, and the neutrino P(k) (every step and at the outputs, with its shot noise) is that of
  the energy-density contrast, as in CAMB/CLASS; the total P(k) at the outputs is the energy-density contrast of all
  matter. Only the energy density enters (the Poisson constraint; as in Elbers et al. 2021). Implemented with
  scoped weights w E / (m <E/m>) that are restored exactly afterwards.
- <E/m> - 1 for one 0.033 eV neutrino (sum 0.1 eV): 0.33 at z = 50, 0.067 at z = 20, 0.019 at z = 10. Effect on
  P_cb at z = 0 (1024 Mpc/h, 0.1 eV, from z = 50): +0.033% at k = 0.01 h/Mpc, +0.01% at 0.02 (linear-theory
  estimate +0.022%); on P_nu a few % at z >= 20. Comparing P_nu with CAMB at z >~ 20 is limited by the delta-f noise
  (and the white shot-noise model, also for the near-lattice CDM) in both cases.
- New regression case `neutrino_particles_energy_density`; all other files unchanged bit for bit. (The option was
  briefly called `particle_neutrino_energy_density` before it also covered the linear neutrinos.)

### Threaded density assignment without atomics, spatial sorting of particles
- `particles_to_grid` with more than one OpenMP thread: the local x-range is split into chunks of at least
  2 ORDER + 2 slices, the particles of the even chunks are added in parallel, then those of the odd chunks (no two
  threads write the same cell, no atomic adds). Deterministic: the same result for any number of threads > 1, equal
  to the one-thread result up to round-off (test in `tests/test_grid_tools.cpp`). Falls back to atomic adds for slabs
  thinner than 4 chunks. One thread: a plain loop (it was an OpenMP region with one thread). Assignment of 128^3
  particles to a 256^3 grid (this machine, 2 threads vs 1): TSC 1.5x (lattice order) / 2.1x (shuffled); with atomics
  2 threads were slower than 1. Default results unchanged except one float32 velocity in one regression file that
  changed by 1 ulp: the compiler fuses the multiply-adds of the new loop differently (with -ffp-contract=off the
  old and new code give identical results).
- `FML::PARTICLE::sort_particles_by_position` / `MPIParticles::sort_by_position(nblocks)`: stable counting sort of the
  particles by block (x-major), in place (4 bytes per particle). COLASolver `particle_sort_every_nsteps` (default 0 =
  never): sorts all species every n steps with blocks of 4^3 force cells. 128^3 COLA, 20 steps, sort every 5 steps:
  kick (force interpolation) -31 to -37%, total -4 to -6% (the FFTs dominate); P(k) identical to the printed digits.

### Performance (results unchanged bit for bit)
- COLASolver: the EuclidEmulator2 boost (written with the P(k) every step, on by default) built the emulator from
  scratch for every call, i.e. at every step (reading its data file and setting up 15 2D splines, ~0.2 s, a third of
  the run time of a 64^3 COLA run). It is now built once per cosmology (`src/EE2Boost.h`).
- `FML::INTERPOLATION::WindowFunctionTable`: the Fourier window of the density assignment from a table of the 1D
  factors (identical values, no sines and no std::function per mode); used in `bin_up_deconvolved_power_spectrum`,
  `deconvolve_window_function_fourier` and the force.
- `particles_to_grid` and `interpolate_grid_vector_to_particle_positions`: the kernel weights and cell indices are
  computed once per dimension (ORDER*N kernel evaluations per particle instead of ORDER^N*N), multiplied in the
  same order as before.
- 3D fast paths without decoding every Fourier index (no integer divisions per mode) in `bin_up_power_spectrum`,
  `bin_up_deconvolved_power_spectrum` and `compute_force_from_density_fourier` (which also computed the wavevector
  twice per mode).
- Run time, 1 thread: 64^3 particles / 128^3 force grid, 20 COLA steps: 12.9 s -> 6.8 s; 128^3 / 256^3: 96.9 s ->
  66.7 s (the FFTs are now ~60%). All 955 regression files are bit for bit identical.
- New timers `DensityAssignment+FFT` and `PofkEveryStep` in the summary at the end of a run.
- Startup warning when the OpenMP threads of a task have fewer cores to run on than threads (CPU affinity, Linux).
  With OpenMPI's default binding for 1-2 tasks all threads of a task share one core: 2 threads ran 9% slower than 1
  in the 128^3 test, and 5% faster with `--bind-to none` (README).

### Test of local primordial non-gaussianity
- `tests/test_fnl.cpp` (run by `make test`, 1/2/4 tasks): generates Phi = phi + fNL (phi^2 - <phi^2>) with
  `generate_nonlocal_gaussian_random_field_fourier` and recovers fNL = 0, +100, -100 from the reduced bispectrum
  Q = B / (P1P2 + P2P3 + P3P1) = 2 fNL, plus the same-seed difference (Q(fNL) - Q(0)) / 2 to 3%. The expectation is
  the exact binned prediction 2 fNL sum_tri (P1P2 + cyc) / N123 / (Pbar1 Pbar2 + cyc), computed in the test: for
  P ~ k^-3 binning alone raises the estimator by 3% (8 bins on 64^3) to 1.3% (16 bins), so compare to the binned
  prediction rather than to 2 fNL when measuring fNL at the percent level.

### Spherical overdensity masses
- `FML::FOF::SphericalOverdensity` (`FriendsOfFriends/SphericalOverdensity.h`): SO properties around given centers
  (shrinking sphere center, radius and mass for any number of overdensities, velocity and dispersion), using all
  particles within a search radius (copied from the neighbor tasks), independent of the number of tasks; warns about
  spheres that do not close within the search radius. `tests/test_so.cpp` (isothermal spheres, 1/2/4 tasks).
- COLASolver: `fof_so_definitions` (e.g. `{"200m", "200c", "vir"}`): SO masses around the FoF halos in the halo
  catalogue (text and HDF5, with truncation and overlap flags) and a mass function per definition with the Tinker
  et al. 2008 fit for the same Delta. Checked against an independent calculation (identical radii and particle
  counts). README: SO masses with PM forces need R_Delta of several force cells (numbers).
- `ParameterMap::get` for string lists (was declared but not instantiated).
- Regression: new case `fof_so`; `fof_two_species_hdf5` now also has SO masses.

### ComputePowerSpectra: hybrid bias spectra
- `HybridBiasSpectra.h` (namespace `FML::CORRELATIONFUNCTIONS::HYBRIDBIAS`): the Lagrangian bias basis spectra of
  Modi, Chen & White 2020 and Kokron et al. 2021. The operators 1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2> and
  nabla^2 delta_L are formed from the linear field on a grid of the size of the particle lattice (exact at the
  lattice points), each particle gets the values at its lattice point (from its ID, or a user function; fetched
  from the task that owns it with MPI) and the weighted particles give the advected fields. Returns the 15 cross
  spectra P_ab(k) (and the Poisson shot noise of each), the tracer spectrum sum_ab b_a b_b P_ab for given biases and
  the cross spectra of a tracer field with the advected operators. The steps (Lagrangian fields, values at the
  particles, advected fields, spectra) are also available separately. Optional growth factor D(k,z)/D(k,zini) and
  smoothing of the linear field.
- Tests: s^2 = (2/3) delta^2 and nabla^2 delta = -k^2 delta exactly for a plane wave, without displacements the
  advected spectra equal those of the Lagrangian fields exactly, and the result is identical on 1, 2 and 4 tasks.
  Example in `ComputePowerSpectra/example/HybridBiasExample.cpp`.
- `bin_up_cross_power_spectrum` was declared with const grids but defined with non-const ones (a call with const
  grids did not link).

### ComputePowerSpectra: Zel'dovich prediction of the hybrid bias spectra
- `HybridBiasZeldovich.h`: `zeldovich_hybrid_bias_spectra` gives the Zel'dovich (resummed 1LPT) prediction of all 15
  basis spectra (including the nabla^2 delta ones) for a linear P(k), with separate filters for the operators and the
  displacement. The angular structure of the correlators comes from projecting the products of the kernels onto
  spherical harmonics (no hand-written tensor algebra), the part linear in the correlators is transformed
  analytically and the rest is integrated over q and mu.
- Tests: without displacements the delta^2/s^2 spectra agree with direct k-space convolutions to 1e-5; the 10 spectra
  without nabla^2 agree with ZeNBu (Chen et al.) to 2e-4; spectra measured from particles moved with the Zel'dovich
  approximation (6 realizations, 128^3 in 500 Mpc/h) agree to ~1-2% (with the cut at half the lattice Nyquist
  frequency up to 0.5 h/Mpc; with the cut at the Nyquist frequency the lattice sampling affects the operators with
  power near the cut for k > ~0.4 k_Ny).
- `HybridBiasOptions::kcut`: a spherical cut of the linear field (isotropic operators, as assumed by the theory).
  `modes_in_bins`: the Fourier modes in each bin, to average a prediction over the modes of the low-k bins.
- The operator enum is now in `HybridBiasOperators.h` (shared by both headers).

### ComputePowerSpectra: one-loop LPT prediction of the hybrid bias spectra
- `HybridBiasOneLoop.h`: `oneloop_hybrid_bias_spectra` gives the one-loop CLEFT prediction (Vlah, Castorina & White
  2016; the formulation of velocileptors) of all 15 spectra. The nabla^2 delta spectra (not in velocileptors) are
  derived the same way: nabla^2 delta_L = -k^2 delta_L is a linear operator, so its one-loop terms are those of
  delta_L with the weight -k^2 on the leg of the operator (U3, A10 = <delta_1 Delta_i Delta_j> and U11, with R_n
  weighted on the external or internal leg and Q5 on a loop leg); P_d2n and P_s2n are complete at Zel'dovich order.
  No counterterms or stochastic terms (add them when fitting). The loop spectra Q_n(k) and
  R_n(k) are computed in k-space with an exact treatment of a sharp cutoff of P(k) (same filter for the operators and
  the displacement).
- Tests: agrees with velocileptors' CLEFT (same P(k) with its Gaussian cutoff) to 2.5e-4 for the 10 spectra without
  nabla^2; converged in qmax and dq to 5e-4. The nabla^2 terms reduce to velocileptors' for a weight 1 (to 1e-6), and
  the loop corrections to P_1n, P_dn, P_nn agree with the difference between particles moved with 2LPT and with the
  Zel'dovich approximation (4 realizations at small amplitude) to a few percent, as for P_1d and P_dd. In COLA at
  z = 0 they reduce the Zel'dovich deficit of P_1n, P_dn, P_nn (-12% to -18% at 0.03 < k < 0.3 h/Mpc) to +5-8%.

### ComputePowerSpectra: f(R) prediction of the hybrid bias spectra (EdS kernels)
- `HybridBiasModifiedGravity.h`: Hu-Sawicki f(R) in the simplest approximation, EdS loop kernels with the f(R) linear
  power spectrum. `fofr_linear_growth_ratio` gives D_fR(k,a)/D_GR(a) (agrees with the COLA f(R) gravity model to
  ~1e-3) and `fofr_linear_power_spectrum` turns P_GR(k,a) into P_fR(k,a) for `zeldovich_hybrid_bias_spectra` /
  `oneloop_hybrid_bias_spectra`. Missing: the scale dependence of the 2nd/3rd order growth and screening in the
  kernels (Aviles & Cervantes-Cota 2017; Brando, Li & Koyama 2026).
- Test (COLA 128^3 in 500 Mpc/h, F5 vs GR with the same IC, standard COLA): the one-loop f(R)/GR ratios follow the
  screened COLA ratios to ~1-2% for P_11, P_1d, P_dd, P_1n, P_dn, P_nn at k < 0.3 h/Mpc at z = 0 and 1 (Zel'dovich
  ratios are off by 3-6% at z = 0); the P_1s2 ratio is off by ~7% and P_d2d2, P_s2s2 drift by up to ~5% towards
  the cut.

### COLASolver: hybrid bias spectra
- `hybridbias = true` writes the hybrid bias spectra at every output (`hybridbias_zX.txt` in the snapshot folder,
  with the linear P_cb(k)). The linear field of the initial conditions is kept on a grid of the size of the particle
  lattice and scaled to the output redshift with the 1LPT growth factor of the gravity model (scale-dependent
  growth included). Options: `hybridbias_nmesh`, `_noperators`, `_density_assignment_method`, `_interlacing`,
  `_subtract_shotnoise`, `_smoothing_scale`, `_smoothing_filter`. Not with `particle_baryons` or
  `ic_random_field_type = read_particles`.
- `hybridbias_kcut_over_nyquist` (default 1: spherical cut at the lattice Nyquist frequency) and
  `hybridbias_zeldovich` (write the Zel'dovich prediction with the same P(k), cut and smoothing, averaged over the
  modes in the low-k bins; the k-values are split over the MPI tasks).
- `hybridbias_oneloop`: also write the one-loop LPT prediction (`hybridbias_oneloop_zX.txt`; the operator cut is
  used for the displacement too; not with smoothing). The predictions are computed on a table of 40 log-spaced
  k-values for the low-k bins and splined (agrees with evaluating every mode to 4e-4); ~20 s per prediction and
  output on 2 cores, independent of the number of particles. In a COLA run (64^3 in 500 Mpc/h) one loop fixes most
  of the Zel'dovich deficit in P_11, P_1delta, P_deltadelta and P_1delta2 at z = 0 (P_11: -11% to +1% at
  0.05 < k < 0.3 h/Mpc).
- Test (64^3 particles in 500 Mpc/h): P_11 agrees with P(k) of the particles to 0.1%, P_1delta = P_deltadelta =
  P_linear on large scales, identical on 1, 2 and 4 tasks; also runs with scale-dependent COLA and without COLA.

- `hybridbias` is `always_optional` (it was `optional`, which made it required with `all_parameters_must_be_in_file =
  true`, so existing parameter files without it stopped, e.g. all regression cases).

### CosmicWeb: ORIGAMI halo grouping
- `ORIGAMIHalos.h`: the halo step of ORIGAMI (Falck et al. 2012) on the Delaunay tessellation of all particles:
  halo particles (morphology 3) above `core_density_threshold` (default 200 times the mean Voronoi density) that are
  neighbors form the cores, the other halo particles join through their neighbors in rounds (each joins the halo
  of its densest neighbor that already is in one, so the result does not depend on the order or the number of
  tasks), the halo particles not connected to a core form their own groups, groups with fewer than 20 particles
  are dropped (no unbinding). Gives the halo of each particle and a catalogue (particles, core particles, mass,
  center of mass, densest particle). MPI with label exchange between the tasks.
- Checked against an independent python implementation on scipy's Delaunay (identical halos, catalogue to 5e-13)
  and identical on 1-4 tasks. Added to the CosmicWeb example (with USE_CGAL).

### ComputePowerSpectra: bispectrum fixes
- The bispectrum shot noise had the wrong sign on the 1/n^2 term (B_obs = B + (P1+P2+P3)/n + 1/n^2): for
  Poisson particles the result was 2/n^2 instead of 0. Matters for sparse tracers (nP ~ 1).
- The bin count N123 included modes with |k| = khigh but the estimator did not, which biased the bins whose upper
  edge falls on grid modes (always the last bin): for a field with delta(k) = 1 23 of 90 configurations were
  off by up to 10%, now all are 1. Both now use [klow, khigh) and skip k = 0.
- P(k) and kmean of the polyspectrum binning use the same mode weights as bin_up_power_spectrum.
- `compute_trispectrum` from particles called the bispectrum (did not compile).
- `PolyspectrumBinning(nbins, Nmesh)` had the two arguments swapped in the definition.
- `get_reduced_spectrum` read out of bounds when nbins < ORDER.

### CosmicWeb: T-web, V-web and ORIGAMI
- New module `FML/CosmicWeb` (MPI). `CosmicWeb.h`: the T-web (Hahn et al. 2007, Forero-Romero et al. 2009), the
  tidal tensor k_i k_j / k^2 delta(k) of the Gaussian smoothed density field from a grid or from particles, cells
  classified by the number of eigenvalues above lambda_threshold (void, sheet, filament, knot), with volume and mass
  fractions and optionally the eigenvalues. `VWeb` does the same for the velocity shear (e.g. from the DTFE) and
  `classify_tensor` for any symmetric tensor. Eigenvalues with Jacobi rotations (accurate to rounding also for
  degenerate ones). No GSL needed.
- `ORIGAMI.h`: ORIGAMI (Falck, Neyrinck & Szalay 2012), the morphology of each particle from the number of
  orthogonal axes of the initial lattice along which it has been crossed (x, y, z and the two diagonals in each
  coordinate plane, 4 orthogonal triplets, search a quarter of the row), linear in N^3. The lattice point comes
  from the particle ID as numbered by `create_particle_grid` (or a user function).
- Checked: T-web against analytic tidal fields (exact classification, eigenvalues to 1e-15) and a numpy
  implementation (identical classes), ORIGAMI against a brute-force python implementation (identical bitmasks and
  morphologies), both identical on 1-4 tasks. Example in `CosmicWeb/example`.

### VoidFinders: merging zones into voids (ZOBOV hierarchy, VIDE)
- `ZoneMerging.h`: `merge_zones` builds the ZOBOV void hierarchy with a union-find over the zone links in
  increasing order: when two voids join at a link the shallower one is finished (its leak is the link, its parent
  the deeper void), equivalent to jozov's growing of a void from each zone until a deeper zone is reached.
- `ZobovOptions::merging` / `VoxelOptions::merging` (and `VoxelWatershed`): `None` (every zone is a void, as in
  Revolver, the default), `Threshold` (merge through links below `link_density_threshold`, default 0.2 of the mean,
  as in VIDE) and `Full` (the full hierarchy). The catalogue has all voids of the hierarchy with `parent` (-1 for
  top-level voids) and `nzones`; `write_catalogue` writes the two as the last columns.
- Triangulation: `WatershedGeneral`/`WatershedDensity` can return the links between the basins (`WatershedLink`).
- Checked against an independent brute-force jozov implementation (members, zones, leak ratios and parents
  identical for ZOBOV and voxel zones, full and threshold), identical on 1-4 tasks; without merging the catalogues
  are unchanged. The barycenter of a merged void is only meaningful for voids smaller than half the box.

### VoidFinders: spherical underdensity void finder
- `SphericalVoidFinder.h`: spheres with integrated density contrast Delta(<R) = delta_threshold (default -0.8)
  around the low density cells of the smoothed density grid (exact particle counts, masses if the particles have
  them), optional random-walk recentering to larger spheres, then the largest non-overlapping spheres are kept
  (allowed overlap and minimum/maximum radius in units of the mean separation are options). MPI: the particles
  within the max radius of the domain are copied from the neighbors, overlaps are removed globally.
- Checked against an independent python implementation (scipy KD-tree; identical catalogue to 5e-13), planted
  empty spheres are found at R = a / (1 + Delta)^(1/3) as expected, identical on 1-4 tasks (also with recentering).
- `void_density_field` (VoxelVoidFinder.h) is the density field used for the voxel voids and the candidates.

### VoidFinders: ZOBOV and voxel void finders as in Revolver
- New module `FML/VoidFinders` (periodic boxes, MPI). The voids are the watershed zones without merging, with
  the cuts and outputs of Revolver (Nadathur et al.): center, R_eff, delta_min, delta_avg, lambda, DensRatio.
  `Void.h`: the catalogue (`Void`, `write_catalogue` in Revolver's column format, sorted by minimum density).
- `ZobovVoidFinder.h`: watershed on the Voronoi densities, center = circumcenter of the minimum and its 3 lowest
  density mutually adjacent neighbors, cuts on the minimum density (< mean) and the number of particles (>= 5).
- `VoxelVoidFinder.h`: CIC + Gaussian smoothing (grid size and smoothing scale as in Revolver, the same truncated
  kernel as scipy's gaussian_filter) and a parallel 6-neighbor watershed on the grid. `VoxelWatershed` runs the
  watershed on any density grid (e.g. the DTFE density); `void_density_field` gives the smoothed density.
- Both can find clusters (overdensities). Checked against independent python implementations of Revolver's
  algorithms (catalogues equal to 5e-12, the voxel pipeline from the particles identical) and identical on 1-4 tasks.
- Triangulation: `WatershedBasin::pos_circ` (the circumcenter above) and `quantity_leak` (ZOBOV's leak density:
  the lowest link to a neighboring basin), set by `WatershedGeneral` for group classes that have
  `set_pos_circ`/`set_leak_quantity`.

### Friends-of-friends halos
- `FriendsOfFriends` warns when groups reach the edge of the buffer copied from the neighbor tasks (they might be cut
  off and found twice); `FML::FOF::FoF_number_of_groups_reaching_buffer_edge`. With a too small buffer a halo on a
  task boundary was split in two (the example: 64 instead of 63 halos with 2 tasks), silently.
- FoF IDs in the particles (`set_fofid`) are now correct with several tasks: only for halos that are kept, and for
  particles in halos kept by the neighbor task (sent back).
- `tests/test_fof.cpp` (in `make test`, 1/2/4 tasks): groups, centers and FoF IDs compared with a serial FoF, and the
  buffer warning. The FoF example reads its test data again (header line) and uses a 10 Mpc/h buffer (it then matches
  the reference catalogue exactly with any number of tasks).
- COLASolver (`src/HaloFinding.h`): FoF with CDM and baryons as separate particles (linked together with their
  masses) and with neutrino particles (CDM+baryons only); halo masses from OmegaCB (changes the masses with massive
  neutrinos by 1 - f_nu); HDF5 halo catalogue (`fof_output_fileformat`); configurable mass-function bins
  (`fof_massfunction_nbins`, `_Mmin`, `_Mmax`); the Tinker et al. 2008 and Watson et al. 2013 mass functions in
  `massfunc_zX.txt` (two new columns). Regression case `fof_two_species_hdf5`.

### Triangulation: DTFE (Delaunay Tessellation Field Estimator) in 3D
- `FML/Triangulation/DTFE.h`: `DTFE<T>` makes the periodic MPI tessellation and computes the DTFE density at the
  particles, (D+1) m / volume of the incident tetrahedra. Fields given at the particles are interpolated linearly in
  each Delaunay tetrahedron (constant gradient per tetrahedron) to a `FFTWGrid<3>`.
- `density_to_grid` (in units of the mean density), `velocity_to_grid` (velocity, divergence, vorticity and the
  traceless shear, from `get_vel`) and `fields_to_grid` (any number of fields, optionally their gradients). The
  general `interpolate_to_grid` calls a user function with the density, values and gradients in every grid cell.
- Grid value at the grid point or averaged over the grid cell (nsub^3 samples), volume or mass weighted. Grid points
  at (ix,iy,iz)/Nmesh as in the rest of FML. Each task fills its own x-slab (Nmesh divisible by the number of tasks).
- Samples in tetrahedra that might not be correct (too small buffer) are counted, not used, and give a warning.
- Validated against analytic fields: linear fields and their gradients exact (1e-15), smooth velocity fields
  converge with particle number (gradient error ~N^-1/3 at grid points), the volume averaged density integrates to
  the total mass, identical results (1e-12) on 1-4 tasks. Example in `Triangulation/example/DTFEExample.cpp`.
- `MPIPeriodicDelaunay::cell_is_reliable` (the buffer check for one cell, used by `num_bad_cells`).

### Triangulation: bug fixes, 2D support and faster tessellation / watershed
- Watershed results no longer depend on the number of MPI tasks. Every particle is linked to its lowest
  neighbor (ties broken by position, quantity kept in double instead of float) and the labels are exchanged with
  the neighbor tasks until converged (was: at most 3 merge passes, which often did not finish). With a large enough
  buffer the groups are identical to a serial run (50k particles: 2 and 3 tasks differed in 2 and 15 groups before).
- `WatershedGeneral` copied the whole CGAL triangulation (`auto dt = ...`) and walked the basins with a recursive
  function (stack overflow for large basins); now a reference and an iterative walk. 2-6x faster.
- Insertion in random order until the triangulation is 1-sheeted, then along a space filling curve with the last
  vertex as hint; `dt.is_valid()` only with `DEBUG_TESSELATION`. Tessellation 4-5x faster (200k particles: 14.8 s ->
  3.1 s). Voronoi volumes computed from the cells around each vertex (1.6x faster, same result as CGAL to 1e-11).
- Buffer check: counts the cells around local particles that contain a guard point or whose circumsphere extends
  outside the buffer (the old check missed cases where the volumes were wrong). A warning means increase
  `buffer_fraction`; without a warning the Voronoi cells are exactly those of a full-box tessellation.
- 2D: Voronoi areas implemented and the watershed compiles and runs (the guard points wrote out of bounds in 2D).
  Neighbors are mapped to the original vertices when the triangulation is not 1-sheeted (`random_fraction = 0`),
  which gave wrong minima before; a warning is printed in that case.
- Communication: the boundary particles are selected once and reused (`communicate_boundary_values`), MPI messages
  are split so that counts never overflow `int`, the groups are merged on task 0 with one `MPI_Reduce`. Guard points
  deterministic (fixed seeds) and placed correctly for all tasks; fixed `size_t` underflow in their count.
- `WatershedBinning`: `merge` takes a const reference, `finalize` skips empty groups. Example: radius uses `1.0/3.0`.

### COLASolver: COLA with CDM and baryons as separate particles
- The species share the COLA frame of the total CDM+baryon field (LPT fields of delta_cb, also scale-dependent);
  their initial velocities are relative to it (the baryon-CDM relative velocity), evolved by the PM part. The
  CDM+baryon P(k) equals that of one species with COLA (< 0.01%, also scale-dependent COLA with linear neutrinos).
- Needs `timestep_method = Quinn` or `FastPM` with baryons: the Tassev operators get the force-free relative mode
  wrong (+30% in P of delta_c - delta_b with 20 linear steps; Quinn: exact, FastPM: -3.5%).
- Library: `NBodyInitialConditions` can store the 1LPT field of another density field (for a shared COLA frame).
- Regression cases `two_species_cola`, `two_species_scaledep_cola_nu`. README: the relative mode at normal amplitude
  is ~30% below linear on large scales in all runs (displacement representation, see README).

### COLASolver: refactoring of the particle species (no change in results)
- `src/Species.h`: one `SpeciesInfo` per particle container (name, GADGET type, mass fraction, weight in delta_cb,
  ID offset, dynamics) and `species_transfer_function`. The every-step and output P(k), linear theory, shot noise
  and GADGET output loop over the species (`NBodySimulation::for_each_species` / `for_each_cb_species`, generic
  lambdas over the different particle types).
- `src/KickDrift.h`: one kick and one drift for all species, chosen by the species dynamics (relativistic for the
  neutrinos); the place to add species-dependent dynamics.
- `src/NeutrinoParticles.h`: class owning the neutrino particles (initial conditions, delta-f weights, density).
  `src/SimulationDensity.h`: the density field as a sum of contributions (CDM+baryon particles, then the N-body
  gauge source, the neutrino particles or linear neutrinos). `InitialConditions.h`: `generate_species_from_delta_cb`
  (shared by baryons and neutrinos) and `two_species_ic_splines`. `Simulation.h` 1800 -> 1190 lines.
- One table of what the species need (`species_requirements()` in `SimulationParameters.h`) used for the validation
  and printed at the end of PARAMETERS.md.
- Library: `FML::GRID::for_each_fourier_mode`, `transform_fourier_modes`, `multiply_fourier_by_function_of_k`,
  `linear_combination_fourier` (Smoothing/SmoothingFourier.h); used in COLASolver (incl. the scale-dependent COLA
  kernels).
- All regression results unchanged bit for bit.

### COLASolver: massive neutrinos as particles (delta-f method, optional)
- `particle_neutrinos = true`: neutrino particles (`particle_neutrino_Npart_1D`, default `particle_Npart_1D`) with the
  delta-f method of Elbers et al. 2021: Fermi-Dirac thermal momenta (reproducible per particle ID), relativistic
  drift and kick, and weights w = 1 - f0(y)/f_i so that the particles only carry the perturbation. The force uses
  delta_m = (1 - f_nu) delta_cb + f_nu delta_nu. `particle_neutrino_deltaf = false` for plain particles. Quinn time
  stepping only, no COLA (README). Default off: all regression results unchanged bit for bit.
- P_nu and P_total at the outputs and every step (with the neutrino shot noise sum w^2/N^2); GADGET binary/HDF5 output
  with the neutrinos as type 3 (and `PartType3/Weights`). Can be combined with `particle_baryons`.
- Library: particles with a `get_deltaf_weight()` method are assigned with that weight and without the mean
  (`particles_to_grid`; the MPI boundary slices now take the background value into account);
  `GadgetWriter::write_gadget_species` takes a list of species with accessors (any particle types, extra HDF5
  fields).
- Checked against CAMB for 0.1 and 0.4 eV (P_nu within ~5% for k < 0.15 h/Mpc at z = 0; P_cb and P_total within
  0.3% of the run with linear neutrinos). Regression cases `neutrino_particles`, `neutrino_particles_baryons_hdf5`.
- `scripts/make_camb_transfer_tables.py` and the tables `input/camb_data_lcdm_nu0.1_zmax200`, `..._nu0.4_zmax200`
  (60 redshifts up to z = 200, 3 degenerate massive neutrinos, accurate neutrino transfer functions and CAMB
  AccuracyBoost = 3: the default CAMB accuracy gives a P_nu that is up to 16% too low below the free-streaming scale).

### Growth rates from the transfer functions
- `LinearTransferData::get_*_growth_rate` used the derivative of the spline in z, which is inaccurate near the ends of
  the table (4% at the high-z end, 2% at z = 0). Now a 4-point Lagrange derivative in log(a) through the tabulated
  redshifts: 0.01-0.1% for CDM and baryons. **Changes results** for scale-dependent growth with massive neutrinos
  (regression case `scaledep_cola_neutrinos` by up to 0.8%, D_cb(k) from z = 20 to 0 now within 0.1% of CAMB instead
  of 0.3%) and `two_species_neutrinos_hdf5` (0.5%). The N-body gauge runs are not affected (their tables extend past
  zini).

### COLASolver: CDM and baryons as separate particles (optional)
- `particle_baryons = true`: CDM and baryons in two particle containers (gravity only), IC from the CDM and baryon
  transfer functions at zini (total velocity = growing mode, relative velocity from d log(T_i/T_cb)/dlna), the
  density for the force is (OmegaCDM delta_cdm + Omegab delta_b)/OmegaCB. Default off: one species as before
  (all regression results unchanged bit for bit).
- `particle_baryon_lattice` (`same`, default, or `offset`): with the same lattice the baryon-CDM difference follows
  linear theory on all scales; offset lattices give a spurious difference for k > k_Ny/8 (README).
- P(k) of CDM, baryons and CDM+baryons at outputs and every step; GADGET binary/HDF5 output with CDM as type 1
  and baryons as type 2 (`GadgetWriter::write_gadget_species`, `GadgetHDF5Writer::write_gadget_species`).
- `NBodyInitialConditions` can take a separate 2LPT source field.
- `pofk_subtract_shotnoise` now also applies to the every-step P(k) (it had no effect there).
- Regression cases `two_species_bullfrog`, `two_species_neutrinos_hdf5`.

### COLASolver: new cosmologies (dark energy backgrounds)
- `cosmology_model = "Quintessence"`: scalar-field dark energy with exponential, inverse power law
  (Ratra-Peebles), PNGB and hilltop potentials; V0 found by shooting for the dark energy density today
  (`src/Cosmology_ScalarField.h`, `src/Cosmology_Quintessence.h`).
- `cosmology_model = "EDE"`: axion-like early dark energy parametrized by fEDE, zc, theta_ini and n (m and f by
  shooting) plus a cosmological constant (`src/Cosmology_EDE.h`).
- `cosmology_model = "Tabulated"`: the background from a file, as H(z)/H0, a CLASS background file or w(a)
  (`src/Cosmology_Tabulated.h`, example `input/class_background_w0wa.dat`).
- Checked against an independent solution (H to < 1e-4, w to < 1e-5 for a < 1); a CLASS table of w0waCDM gives
  the same simulation as `w0waCDM` (P(k) to 3e-6). Regression cases `quintessence_exponential`,
  `quintessence_pngb`, `early_dark_energy`, `tabulated_class`.

### COLASolver: growth factors in the radiation era, time steps uniform in D, outputs
- The growth equations start from the growing mode of a matter + radiation universe (D1 = 1 + 3a/(2a_eq) and the
  matching 2LPT/3LPT modes), at a < a_eq/10, instead of the Einstein-de Sitter values. Before, the tables
  contained a decaying mode when they started in the radiation era (e.g. D(z=0)/D(z=1e7) was 3x too small; the
  simulations were consistent with the table, so P(k, z = 0) only changes by < 5e-4 in the regression cases).
- `timestep_scalefactor_spacing = "growthfactor"`: steps uniform in D(a). 15-40% smaller time-stepping errors for
  BullFrog (default unchanged).
- `plc_output_hdf5file`: lightcone particles as GADGET HDF5 files. `GadgetHDF5Writer::set_periodic_wrap`.
- `pofk_every_step` (default true) to switch off the P(k) at every step; the files are now written once (they were
  rewritten at every output). These files no longer subtract shot noise by default when `pofk = false`.
- **Removed** `output_fileformat = "FML"` (raw dump of the particles; use GADGET or HDF5). The library functions
  `MPIParticles::dump_to_file/load_from_file` are kept for checkpointing.
- Regression cases `hdf5_lightcone`, `bullfrog_growthfactor_spacing`; `check_timestepping.py` also tests the
  growth factor spacing.

### COLASolver: accurate forces on a grid equal to the particle grid (experimental)
- `force_early_method` (`none`, `nufft`, `interlaced`) with `force_early_until_redshift`: use an accurate force only
  at early times, when the particles are close to the lattice and CIC is least accurate.
- `force_interlacing_order`: compute the force for several copies of the particles shifted by a fraction of a cell
  and average it at the particles (cancels the leading aliasing of the assignment and the interpolation).
- `force_deconvolve_window`: divide the force by W(k)^2. Stable together with interlacing. TSC or PCS with two
  interlaced grids and deconvolution gives close to the P(k) of a much finer grid on a 1x grid.
- `force_density_method` (`assignment`, `nufft`, `sheet`, `sheet_nufft`) and `force_interpolation_method`
  (`fiducial`, `assignment`, `nufft`): exact density and force with non-uniform FFTs, and spectral sheet
  interpolation. `force_nufft_epsilon`, `force_sheet_refinement`.
- A warning when these options are used with a force grid finer than the particle grid (they amplify the
  discreteness of the particles there).
- Starting from (almost) the Big Bang with BullFrog: `ic_LPT_order = 1` and e.g. `ic_initial_redshift = 1e7`
  (the growth factor tables now start before the initial redshift).
- New particle type `ParticleBasicQ` (stores the Lagrangian position, needed for sheet interpolation).
- README section with accuracy and cost tables. Regression cases `nufft_force`, `sheet_nufft_force`,
  `interlaced_force`, `bullfrog_bigbang`, `early_force_switch`.

### COLASolver: force grid finer than the particle grid
- The continuous gradient kernel gives large spurious forces from the particle lattice when
  `force_nmesh > particle_Npart_1D`. The fiducial gradient kernel is now `discrete_4pt` for force grids up to 3x
  the particle grid and `discrete_2pt` for finer grids (continuous for grid = particle grid).
- `ic_lattice_offset` (`fiducial`, `none`, `half_force_cell`): by default the particle lattice is offset by half a
  force-grid cell when the force grid is finer (particles on the grid nodes make CIC respond to the size of the
  displacement). `tests/check_force_grid.py`, regression case `fine_force_grid`.
- **Changes results** for runs with `force_nmesh > particle_Npart_1D` (P(k) is much more accurate). Use
  `ic_lattice_offset = "none"` and `force_gradient_kernel = "discrete_2pt"` (or `"continuous"`) for the old
  behaviour.

### COLASolver: new features
- Time-stepping methods `FastPM` and `BullFrog` (Rampf, List & Hahn 2024) in `TimeStepping.h`;
  `tests/check_timestepping.py`.
- GADGET HDF5 snapshots: read and write (`USE_HDF5`, `output_fileformat = "HDF5"`, HDF5 initial conditions are
  detected automatically), `GadgetUtils/GadgetUtilsHDF5.h`, `./nbody --features`.
- N-body gauge (radiation, massive neutrinos and GR on large scales): `force_nbody_gauge`,
  `scripts/make_nbody_gauge_transfer.py` (CLASS via classy), `src/NBodyGauge.h`, scale-dependent initial
  velocities; `tests/check_nbody_gauge.py`.
- Parameter schema (`src/ParameterSchema.h`): one table of all parameters with defaults, validation, warnings for
  unknown keys and `./nbody --parameters PARAMETERS.md` to write the reference.
- Model registry (`src/Models.h`) and smaller particle types chosen automatically (`src/Particles.h`).

### COLASolver: refactoring and tests
- `Simulation.h` split into `SimulationParameters.h`, `TimeStepping.h`, `InitialPowerSpectrum.h`,
  `InitialConditions.h`, `ExternalIC.h`; COLA state in `COLA.h`.
- Regression suite (`tests/regression`: `run.sh`, `cases.py`, `compare.py`).
- MG-PICOLA accuracy tests revived (`tests/run_accuracy_tests.sh`), `tests/check_nongaussian.py`.

### Library
- `NUFFT/NUFFT.h`: non-uniform FFTs (type 1 and 2, MPI and OpenMP), density and force helpers.
- `Interpolation/SheetInterpolation.h`: spectral sheet interpolation of the displacement field.
- `GRID::shift_grid_fourier`, `shift_grid_real`, `PARTICLE::shift_particles`,
  `INTERPOLATION::generate_interlacing_shifts` and `particles_to_fourier_grid_interlaced` (interlacing of any order).
- `tests/test_grid_tools.cpp`.
- Faster particle assignment with one thread (no atomic adds).

### Build system
- One shared `Makefile.config` included by all Makefiles, machine settings in `Makefile.local` (not tracked).
- Top-level `make examples | examples-extra | python | test | clean`; `tests/test_core.cpp`.
- `.gitignore`, `.clang-format`, typo fixes (the FFTWGrid methods `create_wisdow`, `load_wisdow`, `save_wisdow` are renamed `create_wisdom`, `load_wisdom`, `save_wisdom`).

### Fixed
- Non-gaussian initial conditions gave NaN.
- `FFTWGrid::reallocate`; integer overflow in `get_index_real/fourier` for large grids.
- Out-of-bounds write in `MPIParticles::create`; check for unwrapped particles and bounds in the task search;
  communication of more than 2 GB of particles.
- Lambda in `FUNS` (Global.h); per-thread spline accelerators.
- Uninitialized GADGET header padding; EuclidEmulator2 boost above z = 10.
- Reductions with `MPI_Allreduce`, FFTW plan cache, `MPI_Alltoall` counts, ODESolver memory leak, no global
  `using namespace` in the multigrid solver headers.
