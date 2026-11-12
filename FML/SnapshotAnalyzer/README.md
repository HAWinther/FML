# SnapshotAnalyzer

`SnapshotAnalyzer` is a standalone executable for post-processing Gadget snapshots (binary or HDF5). It uses FML's Gadget reader and analysis kernels, and does not include
or depend on `SimulationCode` or `NBodySimulation`.

## Build

From this directory, use the same compiler and dependency configuration as the
rest of FML:

```sh
make
```

Lua and FFTW are required. MPI and OpenMP follow the normal `Makefile.config`
options. ZOBOV and DTFE are optional and require CGAL; enable them with `make USE_CGAL=true`. Gadget HDF5 snapshots
need `make USE_HDF5=true`.

## Capabilities

All the analyses are the shared analysis code in [FML/Analysis](../Analysis/README.md). P(k), multipoles, the
bispectrum, FoF/SO halos (and relaxed FoF) and baryonification have the same parameter names, defaults, algorithms
and output files as SimulationCode, so the two give the same result for the same particles.

- Particle positions, velocities, IDs, types and masses from Gadget binary (`prefix.0, prefix.1, ...` or a single
  file `prefix`) or Gadget HDF5 snapshots (`prefix.hdf5` or `prefix.0.hdf5, ...`; per-particle `Masses` are read
  for types with MassTable = 0); `particle_types` selects the Gadget types to analyse (default: all).
- `snapshot_read_mode = "files_per_task"` (default): each MPI task reads every NTasks-th file and the particles are
  then sent to the task owning their position, so every file is read once. `"all_tasks"`: every task reads every
  file and keeps its own particles (the earlier behavior).
- The cosmology for the analyses (critical-density SO masses, redshift-space displacements) is LCDM from the
  snapshot header (`Omega0`, `OmegaLambda`, plus `cosmology_Omega_r`), or H(a)/H0 from a table
  (`cosmology_hubble_table`, columns a and H/H0, e.g. SimulationCode's `cosmology_<name>.txt`). The density parameter
  of the selected particles comes from their total mass.
- Mass-weighted P(k) (`pofk_z<z>.txt`) with interlacing, density assignment and shot-noise subtraction.
- Redshift-space multipoles (`pofk_multipole_z<z>.txt`): the velocity-to-displacement factor
  `1 / (100 a E(a) B)` comes from the cosmology.
- Bispectrum B(k1, k2, k3) (`bispectrum_z<z>.txt`, SimulationCode's parameters and columns: the bin indices, the
  mean k of the bins, B, the reduced bispectrum Q and the number of triangles, for all `bispectrum_nbins`^3 bins;
  linear bins from 0 to the Nyquist frequency). Memory and time grow as `bispectrum_nbins` grids of
  `bispectrum_nmesh`^3 and `bispectrum_nbins`^3 integrals. On a SimulationCode snapshot it equals SimulationCode's
  own bispectrum to the float32 precision of the Gadget positions (~2e-6).
- Cosmic-web classification with the T-web (Hahn et al. 2007; [FML/CosmicWeb](../CosmicWeb/README.md)): the
  eigenvalues of the tidal tensor of the Gaussian-smoothed density (`cosmicweb_smoothing_scale_mpch`) classify the
  cells as voids, sheets, filaments or knots (0-3 eigenvalues above lambda_th). `cosmicweb_z<z>.txt`: the volume and
  mass fractions of the four types for each of `cosmicweb_lambda_thresholds` (the mass fractions weight the cells by
  the unsmoothed density on the same grid).
- Hybrid bias spectra (Modi, Chen & White 2020; Kokron et al. 2021): the 15 cross spectra P_ab(k) of the Lagrangian
  bias operators 1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L, evaluated on the particle lattice
  and advected with the particles (`hybridbias_z<z>.txt`, SimulationCode's columns plus `Plinear(k)`, the linear
  field of this realization scaled to the snapshot, in the same bins). Needs a simulation started from a lattice.
  The linear field delta_L comes from
  - an initial-conditions snapshot of the same particles (`hybridbias_ic_snapshot`, e.g. SimulationCode with the
    initial redshift in `output_redshifts`): the lattice (and its offset) from the positions, delta_L = -div Psi of
    the displacement Psi = x - q, minus the second order part (3/7) sum_{i<j} (phi_ii phi_jj - phi_ij^2) with
    nabla^2 phi = -div Psi (`hybridbias_ic_lpt_correction = "2lpt"`; "none" for Zel'dovich IC). The particles are
    matched to the lattice by ID, so any ID scheme works. The IC must be early enough that every particle is within half a cell of
    its lattice point (checked: every lattice point must get exactly one particle);
  - or a file (`hybridbias_linear_field_source = "file"`, `hybridbias_linear_field_file`): a real-space grid on the
    lattice written by SimulationCode (`hybridbias_output_linear_field = true`) or by your own IC code (format and a
    numpy writer in `FML/ComputePowerSpectra/HybridBiasLinearFieldIO.h`). The lattice index of a particle is
    `id - hybridbias_id_offset` = (ix N + iy) N + iz (SimulationCode's IDs), or from `hybridbias_ic_snapshot` if
    that is also given.

  The field is scaled with D(z)/D(z_L) of the analysis cosmology (scale independent; `hybridbias_growth_ratio`
  overrides it, e.g. for scale-dependent growth set it per run or use SimulationCode's in-run spectra). Set
  `cosmology_Omega_r` to the radiation of the simulation: without it D(z)/D(z_L) is ~0.4% too large from z_L = 20.
  On a SimulationCode run (64^3 in 256 Mpc/h) the spectra from the linear field file equal the in-run ones to 2e-5 of
  sqrt(P_aa P_bb); from the IC snapshot (2LPT IC at z = 20) to < 0.3% for most pairs (0.8% for delta^2 delta^2,
  s^2 s^2) with the 2LPT correction and up to 18% without it.
- Minkowski functionals V0-V3 (volume, surface area, integrated mean curvature, Euler characteristic per volume) of
  the excursion sets delta >= delta_th of the Gaussian-smoothed density contrast ([FML/Morphology](../Morphology/README.md),
  exact counts of the voxel complex, the isotropic Crofton estimator). Thresholds in nu = delta / sigma_0
  (`minkowski_threshold_variable = "nu"`) or in delta. `minkowski_z<z>.txt`: per threshold V0-V3 in (h/Mpc)^k, the
  Gaussian random field predictions for the sigma_0 and sigma_1 of the same field (Tomita 1986; Schmalzing &
  Buchert 1997; departures measure non-Gaussianity), the Euler characteristic of the excursion set and the exact
  voxel-surface V1, V2. On a Gaussian random field the measured functionals agree with the predictions to ~1% (V0,
  V1), ~5% (V2) and ~8% (V3) of their amplitude for one 96^3 realization. The density is not deconvolved by the
  assignment window: use a smoothing scale of several cells.
- Distributed periodic Friends-of-Friends halos with optional spherical overdensity masses
  (`fof_so_definitions = {"200m", "500c", "vir"}`), overlap flags, the halo catalogue (`halos_z<z>.txt`, or `.hdf5`
  with `fof_output_fileformat = "HDF5"` and `USE_HDF5=true`) and mass functions (`massfunc_z<z>.txt`,
  `massfunc_so<def>_z<z>.txt`).
- Relaxed FoF (`fof_relaxed = true`; [FriendsOfFriends/RelaxedFoF.md](../FriendsOfFriends/RelaxedFoF.md)) instead of
  standard FoF, for low-resolution simulations (Dai et al. 2019, as FastPM's RFOF): a linking length per halo-size
  bin (longer for small halos) and rejection of fake halos by their velocity dispersion. The `fof_relaxed_*`
  parameters and defaults are SimulationCode's (FastPM's fit for a 0.3 Mpc/h mean separation and 40 FastPM steps:
  refit for other simulations); the catalogue gets the pass, bin, linking length and sigma_v / V_std of each halo
  (text columns, HDF5 group `RelaxedFoF`). The velocities must be peculiar velocities (km/s after the Gadget
  sqrt(a) convention) and the redshift that of the snapshot: r0(z) <= 0 (every halo rejected) for z > 2.2 with the
  defaults.
- Voxel watershed void catalogues from a smoothed density field, with grid, smoothing, density cut, and
  zone-merging controls (`voids_z<z>.txt`).
- Spherical underdensity voids with configurable density threshold, radius cuts, recentering, and overlap removal
  (`voids_spherical_z<z>.txt`).
- ZOBOV Voronoi-watershed void catalogues when built with CGAL (`voids_zobov_z<z>.txt`).
- DTFE density contrast, peculiar velocity (km/s), divergence and vorticity (km/s per Mpc/h) spectra when built with
  CGAL (`dtfe_spectra_z<z>.txt`); optionally the fields as one text file per MPI task
  (`dtfe_fields_z<z>.rank<NNNNN>.txt`).
- Position-only baryonification with M200c hosts around FoF halos and the FML finite-support profile transport:
  the host catalogue (`baryonification_z<z>.txt`), P(k) before and after (`pofk_baryonified_z<z>.txt`) and, with
  `baryonification_output_particles = true`, the corrected Gadget snapshot (`gadget_baryonified_z<z>.*`).

The files are written to `output_folder` with SimulationCode's names (z with three decimals). The reader currently
loads Gadget binary and HDF5 snapshots.

## Parameter file

The executable accepts a Lua file. Names in the file that are not parameters give a warning; names that were
parameters in earlier versions give an error that says what to use instead. A configuration with the defaults:

```lua
snapshot_input = "./snapshot_001/snapshot_001" -- Gadget file prefix
snapshot_format = "gadget" -- or "gadget_hdf5" (USE_HDF5=true)
snapshot_read_mode = "files_per_task" -- or "all_tasks"
particle_types = {} -- Gadget types to analyse, e.g. {1}; empty: all
snapshot_length_unit_mpch = 1.0 -- the snapshot length unit in Mpc/h (0.001 for kpc/h)
snapshot_mass_unit_msunh = 1e10 -- the snapshot mass unit in Msun/h
particle_allocation_factor = 1.25 -- room for particles moving between MPI tasks
cosmology_Omega_r = 0.0 -- radiation in the LCDM expansion history from the header
cosmology_hubble_table = "" -- or a file with columns a, H(a)/H0
output_folder = "."

pofk = true
pofk_nmesh = 256 -- must be divisible by the number of MPI tasks
pofk_interlacing = false
pofk_subtract_shotnoise = false
pofk_density_assignment_method = "CIC" -- NGP, CIC, TSC, PCS, PQS
pofk_nbins = 0 -- 0: nmesh/2 linear bins of width 2pi/B (as SimulationCode)
pofk_log_binning = false -- logarithmic bins from 2pi/B to the Nyquist k (with nbins > 0)

pofk_multipole = false
pofk_multipole_nmesh = 256
pofk_multipole_interlacing = false
pofk_multipole_subtract_shotnoise = false
pofk_multipole_ellmax = 4 -- even; P_0, P_2, ..., P_ellmax
pofk_multipole_density_assignment_method = "CIC"
pofk_multipole_nbins = 0
pofk_multipole_log_binning = false

bispectrum = false
bispectrum_nmesh = 128
bispectrum_nbins = 16 -- linear bins from 0 to the Nyquist frequency
bispectrum_density_assignment_method = "CIC"
bispectrum_interlacing = true
bispectrum_subtract_shotnoise = false

hybridbias = false
hybridbias_nmesh = 128
hybridbias_noperators = 5 -- 1, delta, delta^2, s^2, nabla^2 delta: 15 spectra
hybridbias_density_assignment_method = "CIC"
hybridbias_interlacing = true
hybridbias_subtract_shotnoise = false
hybridbias_smoothing_scale = 0.0 -- Mpc/h, of the linear field; 0: none
hybridbias_smoothing_filter = "gaussian" -- or tophat, sharpk
hybridbias_kcut_over_nyquist = 1.0 -- spherical cut of the linear field at this times the lattice Nyquist; 0: none
hybridbias_linear_field_source = "ic_snapshot" -- or "file"
hybridbias_ic_snapshot = "" -- prefix of the IC snapshot (same snapshot_format)
hybridbias_ic_lpt_correction = "2lpt" -- or "none" (Zel'dovich IC)
hybridbias_linear_field_file = "" -- e.g. SimulationCode's hybridbias_linear_field_<name>.dat
hybridbias_id_offset = 0 -- lattice index = id - offset (file source without an IC snapshot)
hybridbias_growth_ratio = 0.0 -- D(z)/D(z_L); 0: from the cosmology

cosmicweb = false
cosmicweb_nmesh = 128
cosmicweb_smoothing_scale_mpch = 2.0 -- Gaussian, exp(-k^2 R^2 / 2); 0: none
cosmicweb_lambda_thresholds = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
cosmicweb_density_assignment_method = "CIC"

minkowski = false
minkowski_nmesh = 128
minkowski_smoothing_scale_mpch = 5.0 -- Gaussian; several cells
minkowski_density_assignment_method = "CIC"
minkowski_threshold_variable = "nu" -- nu: delta_th = nu sigma_0 of the smoothed field; delta: delta_th itself
minkowski_threshold_min = -3.0
minkowski_threshold_max = 3.0
minkowski_nthresholds = 25 -- linear from min to max

fof = false
fof_nmin_per_halo = 20
fof_linking_length = 0.2 -- units of the mean particle separation
fof_nmesh_max = 0 -- 0: automatic
fof_buffer_length_mpch = 3.0 -- buffer from the neighbor tasks and SO search radius
fof_output_fileformat = "ascii" -- or "HDF5"
fof_massfunction_nbins = 30
fof_massfunction_Mmin = 1e10
fof_massfunction_Mmax = 1e16
fof_so_definitions = {} -- e.g. {"200m", "500c", "vir"}
fof_so_nmin_center = 20
fof_relaxed = false -- relaxed FoF instead (then fof_linking_length and fof_nmin_per_halo are not used)
fof_relaxed_mode = "fastpm" -- or "paper" (shorter linking lengths for the particles of rejected halos)
fof_relaxed_l1 = 0.25 -- linking length of the smallest halos: l1 - a1 / (1 + z)
fof_relaxed_a1 = 0.012
fof_relaxed_l6 = 0.24 -- of the largest halos: max(l6 - a2 / (1 + z), fof_relaxed_linking_length_min)
fof_relaxed_a2 = 0.06
fof_relaxed_linking_length_min = 0.2
fof_relaxed_bin_edges = {20, 40, 80, 160, 320} -- upper edges (particles) of the halo bins
fof_relaxed_nmin = 8
fof_relaxed_velocity_rejection = true -- reject sigma_v(3D) > r0 1100 km/s (E(z) M / 1e15 Msun/h)^(1/3)
fof_relaxed_b1 = 7.02 -- r0 = b1 - b2 ln(1 + z)
fof_relaxed_b2 = 6.025
fof_relaxed_r0_min = 0.0 -- floor of r0 (0: none)

voids = false
spherical_voids = false
zobov_voids = false -- requires a USE_CGAL=true build
void_nmesh = 0 -- Revolver default; explicit sizes must be divisible by MPI task count
void_smoothing_scale = -1.0 -- negative selects mean particle spacing
void_smoothing_method = "revolver" -- or "gaussian"
void_density_assignment_method = "CIC"
void_min_density = 1.0
void_merging = "none" -- or "threshold", "full" (also used by ZOBOV)
void_link_density_threshold = 0.2
spherical_void_delta = -0.8
spherical_void_candidate_density_cut = 0.5
spherical_void_max_radius_mean_separation = 8.0
spherical_void_min_radius_mean_separation = 2.0
spherical_void_recentering_steps = 0
spherical_void_max_overlap = 0.0
zobov_buffer_fraction = 0.5
zobov_random_fraction = 0.3
zobov_min_density_cut = 1.0
zobov_min_particles = 5

dtfe = false -- requires a USE_CGAL=true build
dtfe_nmesh = 128 -- must be divisible by MPI task count
dtfe_buffer_fraction = 0.5 -- increase if cells come out as NaN (large voids, many tasks)
dtfe_random_fraction = 0.3
dtfe_sampling = "cell_average" -- or "grid_point"
dtfe_weighting = "volume" -- or "mass"
dtfe_nsub = 2 -- samples per dimension for cell_average
dtfe_nbins = 40 -- logarithmic k-bins of the spectra
dtfe_write_fields = false

baryonification = false -- uses the pofk_* settings for P(k) before/after
baryonification_baryon_fraction = 0.157 -- Omega_b / Omega_m; illustrative, set to the intended model value
baryonification_output_particles = false
baryonification_min_halo_mass_msunh = 1.0e12
baryonification_fof_linking_length = 0.2
baryonification_fof_min_particles = 20
baryonification_fof_nmesh_max = 0
baryonification_fof_buffer_mpch = 3.0
baryonification_profile_radial_bins = 2048
baryonification_min_radius_over_r200 = 1.0e-6
baryonification_max_radius_over_r200 = 30.0
baryonification_gas_mass_scale_msunh = 1.0e14
baryonification_gas_mass_exponent = 0.3
baryonification_gas_core_radius_over_r200 = 0.1
baryonification_gas_ejection_radius_over_r200 = 4.0
baryonification_truncation_radius_over_r200 = 4.0
baryonification_stellar_mass_scale_msunh = 2.5e11
baryonification_stellar_amplitude = 0.09
baryonification_stellar_total_exponent = 0.32
baryonification_stellar_central_exponent = 0.6
baryonification_stellar_radius_over_r200 = 0.015
baryonification_relax_collisionless = true
baryonification_relaxation_amplitude = 0.3
baryonification_relaxation_exponent = 2.0
baryonification_concentration_norm = 5.0
baryonification_concentration_pivot_mass_msunh = 1.0e14
baryonification_concentration_mass_slope = -0.1
baryonification_concentration_redshift_power = -1.0
```

Baryonification needs exactly one selected Gadget type, representing the total CDM+baryon matter field. It does
not change masses, velocities, IDs or types (Gadget's stored precision still applies); the output snapshot has the
format of the input and masses in the Gadget unit 1e10 Msun/h. Extra Gadget blocks beyond POS, VEL, ID and MASS are not preserved. Existing
`gadget_baryonified_z<z>.*` files (or the input) are never overwritten. The profile defaults are illustrative and
uncalibrated; read the [Baryonification module notes](../Baryonification/README.md) for the model's
finite-support, background, overlap, and calibration limitations.

Run with `./snapshot_analyzer parameterfile.lua` (or with `mpirun`). An error on any MPI task stops all tasks.

## Tests

`make -C tests run` (options `MPIRUN`, `NP_LIST`, `USE_CGAL=true`, `USE_HDF5=true`) builds the executable and runs
`tests/check_snapshot_analyzer.py` (needs numpy; h5py for the HDF5 check) on a synthetic snapshot with three halos
and a spherical void: every analysis on 1, 2 and 4 tasks and with both read modes gives the same result, the halos,
baryonification hosts and the largest void are where they were put, the baryonified snapshot keeps IDs, velocities
and masses, HDF5 input gives the same results as binary, and errors stop all tasks with a clear message. The shared
analysis code has its own unit tests in [FML/Analysis/tests](../Analysis/tests).

## Roadmap

1. Snapshot metadata checks beyond the header (e.g. units groups, SWIFT velocity convention).
2. Configurable SO centering controls.
3. Preserve additional Gadget blocks during baryonification output.
4. Use FML/Analysis from SimulationCode in place of its own copies of the P(k), halo and baryonification code (and
   add the void finders and DTFE to its outputs).

FoF uses the algorithm of Davis et al. (1985), [doi:10.1086/163168](https://doi.org/10.1086/163168),
through FML's standalone `FriendsOfFriends` module. Relaxed FoF is that of Dai, Feng, Seljak & Singh (2019),
[arXiv:1908.05276](https://arxiv.org/abs/1908.05276), as implemented in FastPM.

Voxel void finding follows Revolver ([Nadathur et al. 2019](https://github.com/seshnadathur/Revolver)); the watershed
can also merge zones using the VIDE threshold convention ([Sutter et al. 2015](https://arxiv.org/abs/1406.1191)).
Spherical underdensity finding follows Padilla, Ceccarelli & Lambas (2005),
[doi:10.1111/j.1365-2966.2005.09500.x](https://doi.org/10.1111/j.1365-2966.2005.09500.x). ZOBOV follows
Neyrinck (2008), [arXiv:0712.3049](https://arxiv.org/abs/0712.3049).
The hybrid bias spectra follow Modi, Chen & White (2020), [arXiv:1910.07097](https://arxiv.org/abs/1910.07097),
and Kokron et al. (2021), [arXiv:2101.11014](https://arxiv.org/abs/2101.11014).
The T-web follows Hahn et al. (2007), [arXiv:astro-ph/0610280](https://arxiv.org/abs/astro-ph/0610280), and
Forero-Romero et al. (2009), [arXiv:0809.4135](https://arxiv.org/abs/0809.4135). The Gaussian-field Minkowski
functionals are those of Tomita (1986) and Schmalzing & Buchert (1997),
[arXiv:astro-ph/9508154](https://arxiv.org/abs/astro-ph/9508154).
DTFE follows Schaap & van de Weygaert (2000), [arXiv:astro-ph/0011007](https://arxiv.org/abs/astro-ph/0011007),
and Cautun & van de Weygaert (2011), [arXiv:1105.0370](https://arxiv.org/abs/1105.0370).
