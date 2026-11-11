# SnapshotAnalyzer

`SnapshotAnalyzer` is a standalone executable for post-processing Gadget binary
snapshots. It uses FML's Gadget reader and analysis kernels, and does not include
or depend on `SimulationCode` or `NBodySimulation`.

## Build

From this directory, use the same compiler and dependency configuration as the
rest of FML:

```sh
make
```

Lua and FFTW are required. MPI and OpenMP follow the normal `Makefile.config`
options. ZOBOV is optional and requires CGAL; enable it with `make USE_CGAL=true`.

## Initial capabilities

- Particle positions, velocities, IDs, families, and masses from Gadget binary
  snapshots, including multipart snapshots.
- Mass-weighted three-dimensional power spectrum, with interlacing, configurable
  density assignment, and shot-noise subtraction.
- Redshift-space power-spectrum multipoles using the existing FML multipole
  estimator. The velocity-to-displacement factor must be supplied explicitly;
  the module does not infer expansion history or assume a Gadget velocity
  convention.
- Distributed periodic Friends-of-Friends halo finding, writing a text catalog
  with particle count, mass, center, mean velocity, and velocity dispersion.
- Optional spherical-overdensity masses, radii, and particle counts around the
  FoF centers. Thresholds are expressed relative to the mean density of all
  particles included in the snapshot.
- Voxel watershed void catalogs from a smoothed CIC density field, with grid,
  smoothing, density cut, and zone-merging controls.
- Spherical underdensity void finding with configurable density threshold,
  radius cuts, recentering, and overlap removal.
- ZOBOV Voronoi-watershed void catalogs when built with CGAL.
- DTFE density, velocity components, divergence, and vorticity spectra when
  built with CGAL; optional per-MPI-rank text grids contain the corresponding
  sampled fields.
- Position-only baryonification using FoF seeds, `M200c` host masses, and
  the standalone FML finite-support profile transport; writes a separate
  Gadget snapshot and a host catalogue.

The reader currently loads Gadget's binary format. HDF5 snapshots and other
formats are planned extensions.

## Parameter file

The executable accepts a Lua file, following the parameter-file style used by
SimulationCode. A minimal power-spectrum configuration is:

```lua
snapshot_input = "./snapshot_001/snapshot_001" -- Gadget file prefix
output_file = "snapshot_analysis.txt"

pofk = true
pofk_nmesh = 256
pofk_nbins = 40
pofk_interlacing = true
pofk_subtract_shotnoise = true
pofk_density_assignment_method = "PCS" -- NGP, CIC, TSC, PCS, PQS

pofk_multipole = false
pofk_multipole_nmesh = 256
pofk_multipole_nbins = 40
pofk_multipole_ellmax = 4 -- even, non-negative; computes ell=0,2,...,ellmax
pofk_multipole_interlacing = true
pofk_multipole_subtract_shotnoise = true
pofk_multipole_density_assignment_method = "PCS"
-- Required only when pofk_multipole = true. Converts stored velocities to
-- shifts in the normalized periodic coordinates used by FML.
pofk_multipole_velocity_to_displacement = 0.0

fof = false
fof_linking_length = 0.2 -- units of mean interparticle separation
fof_nmin = 20
fof_buffer_length_mpch = 5.0
fof_nmesh_max = 512
fof_output_file = "fof_catalogue.txt"

so = false -- requires fof = true
so_overdensity = {200, 500} -- Delta times the mean particle density
so_initial_overdensity = 200 -- used to estimate the initial shrinking radius
so_search_radius_mpch = 5.0
so_nmin_center = 20

voids = false
void_nmesh = 0 -- Revolver default; explicit sizes must be divisible by MPI task count
void_smoothing_scale = -1.0 -- negative selects mean particle spacing
void_smoothing_method = "revolver" -- or "gaussian"
void_density_assignment_method = "CIC"
void_min_density = 1.0
void_merging = "none" -- or "threshold", "full"
void_link_density_threshold = 0.2
void_output_file = "void_catalogue.txt"

spherical_voids = false
spherical_void_delta = -0.8
spherical_void_candidate_density_cut = 0.5
spherical_void_max_radius_mean_separation = 8.0
spherical_void_min_radius_mean_separation = 2.0
spherical_void_recentering_steps = 0
spherical_void_max_overlap = 0.0
spherical_void_output_file = "spherical_void_catalogue.txt"

zobov_voids = false -- requires a USE_CGAL=true build
zobov_buffer_fraction = 0.5
zobov_random_fraction = 0.3
zobov_min_density_cut = 1.0
zobov_min_particles = 5
zobov_void_output_file = "zobov_void_catalogue.txt"

dtfe = false -- requires a USE_CGAL=true build
dtfe_nmesh = 128 -- must be divisible by MPI task count
dtfe_buffer_fraction = 0.5
dtfe_random_fraction = 0.3
dtfe_sampling = "cell_average" -- or "grid_point"
dtfe_weighting = "volume" -- or "mass"
dtfe_nsub = 2 -- samples per dimension for cell_average
dtfe_write_fields = false
dtfe_fields_prefix = "dtfe_fields"
dtfe_spectra_file = "dtfe_spectra.txt"

baryonification = false
baryonification_output_prefix = "snapshot_baryonified" -- writes prefix.0, prefix.1, ...
baryonification_host_catalogue = "baryonification_hosts.txt"
baryonification_min_halo_mass_msunh = 1.0e12
baryonification_overdensity_mean = 0.0 -- zero derives 200c from Gadget Omega0/OmegaLambda
baryonification_baryon_fraction = 0.157 -- illustrative; set to the intended model value
baryonification_fof_linking_length = 0.2
baryonification_fof_min_particles = 20
baryonification_fof_nmesh_max = 512
baryonification_search_buffer_mpch = 5.0
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

The profile, stellar, relaxation, and concentration parameters can also be set
with the `baryonification_*` names exposed in `AnalyzerOptions` (see
`src/SnapshotAnalyzer.h`). The input prefix and all output files must be
distinct; existing Gadget outputs or host catalogues are refused. This initial
path requires exactly one occupied Gadget family whose particles represent the
total CDM+baryon matter field. It does not intentionally change masses,
velocities, IDs, or family labels (Gadget's stored precision still applies).
Extra Gadget blocks beyond POS, VEL, ID, and MASS are not preserved in the
output file.

The default `M200c` threshold uses a matter, curvature, and Lambda
expansion history derived from the Gadget header; it omits radiation. Set
`baryonification_overdensity_mean` explicitly when that approximation is not
appropriate. The profile defaults are illustrative and uncalibrated; read the
[Baryonification module notes](../Baryonification/README.md) for the model's
finite-support, background, overlap, and calibration limitations.

Run with `./snapshot_analyzer parameterfile.lua`. Output columns are physical
`k [h/Mpc]`, `P(k) [(Mpc/h)^3]`, and the number of Fourier modes in the bin.

## Roadmap

1. Add input abstraction and HDF5/Gadget variants, with snapshot metadata checks.
2. Add species selection and configurable SO centering controls to the current
   FoF and SO catalog pipeline.
3. Add configurable particle-family support and preserve additional Gadget
   blocks during baryonification output.
4. Add more snapshot-derived statistics and later expose the same analysis API
   to SimulationCode without coupling this executable back to the solver.

Large-file I/O currently follows `GadgetReader` behavior, so every MPI rank
reads the snapshot files and retains its domain's particles. Parallel/collective
I/O is a future scalability improvement.

FoF uses the algorithm of Davis et al. (1985), [doi:10.1086/163168](https://doi.org/10.1086/163168),
through FML's standalone `FriendsOfFriends` module.

Voxel void finding follows Revolver ([Nadathur et al. 2019](https://github.com/seshnadathur/Revolver)); the watershed
can also merge zones using the VIDE threshold convention ([Sutter et al. 2015](https://arxiv.org/abs/1406.1191)).
Spherical underdensity finding follows Padilla, Ceccarelli & Lambas (2005),
[doi:10.1111/j.1365-2966.2005.09500.x](https://doi.org/10.1111/j.1365-2966.2005.09500.x). ZOBOV follows
Neyrinck (2008), [arXiv:0712.3049](https://arxiv.org/abs/0712.3049).
DTFE follows Schaap & van de Weygaert (2000), [arXiv:astro-ph/0011007](https://arxiv.org/abs/astro-ph/0011007),
and Cautun & van de Weygaert (2011), [arXiv:1105.0370](https://arxiv.org/abs/1105.0370).
