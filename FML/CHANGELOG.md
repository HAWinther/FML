# Changelog

All notable changes to FML. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The entries below are relative to upstream commit `f9f666a` (April 2026).

## [Unreleased]

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
