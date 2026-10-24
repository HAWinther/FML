# Changelog

All notable changes to FML. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The entries below are relative to upstream commit `f9f666a` (April 2026).

## [Unreleased]

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
