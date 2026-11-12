# Analysis

Analyses of a set of N-body particles, written so that the same code serves a standalone snapshot analysis
([SnapshotAnalyzer](../SnapshotAnalyzer/README.md)) and, later, a simulation's on-the-fly analysis (SimulationCode's
`AnalyzeOutput`). Nothing here depends on SimulationCode, on an input format or on a parameter-file reader.
Everything is in `FML::ANALYSIS` and header-only (C++17).

| Header | What |
|---|---|
| `AnalysisContext.h` | `AnalysisContext`: box size, scale factor, expansion history (`Background`), particle unit conversions |
| `PowerSpectrum.h` | P(k) and redshift-space multipoles P_ell(k) and their output files |
| `Halos.h` | FoF (or relaxed FoF, `fof_relaxed`) halos, spherical overdensity (SO) masses, overlap flags, halo catalogue (text/HDF5), mass functions |
| `Baryonification.h` | Position-only baryonification (FoF/M200c hosts + FML/Baryonification profiles) |
| `Voids.h` | Void catalogues: voxel watershed, spherical underdensities, ZOBOV (CGAL) |
| `DTFE.h` | DTFE density and velocity fields and their power spectra (CGAL) |
| `ParticleUtils.h` | `redistribute_particles`: move displaced particles to their owner task, growing storage as needed |
| `Output.h` | `write_on_root` and `throw_if_any_task_failed`: errors are agreed on by all tasks (no deadlocks) |

## The context

An analysis gets the particles (positions in box units [0,1), the FML convention) and an `AnalysisContext`:

- `boxsize` (Mpc/h) and `scale_factor`;
- `background`: E(a) = H/H0 and Omega_m0, from `Background::LCDM(Omega_m, Omega_Lambda, Omega_r)` (curvature from
  closure), `Background::from_table(file, Omega_m)` (columns a, H/H0) or any function;
- `mass_to_msunh` and `velocity_to_kms`: particle mass and velocity units to Msun/h and peculiar km/s.

From these follow the redshift-space displacement factor, `velocity_to_displacement() = v_unit / (100 a E(a) B)`,
and the density parameter of the particles, `Omega_of_particles(total_mass)`, which sets the critical-density
SO thresholds (so `200c` is right for whatever the particles represent, e.g. CDM+baryons only).

A Gadget snapshot read with FML's GadgetReader: `mass_to_msunh = 1e10`, `velocity_to_kms = 1` (the reader already
multiplies by sqrt(a)). SimulationCode's particles: `mass_to_msunh` = the mean particle mass in Msun/h (masses are in
units of it) and `velocity_to_kms = 100 B / a` (code velocities are a^2 dx/dt / (H0 B)).

## Settings

Each analysis has a settings struct with the same keys and defaults as SimulationCode where both have the
parameter (`pofk_*`, `pofk_multipole_*`, `fof_*`, `baryonification_*`), described once by
`parameters(f)`, which calls `f(name, value, default, description)` for every parameter. A reader adds
`Settings::prefix` to the name, so any parameter system can fill them (SnapshotAnalyzer's Lua reader does
`settings.parameters(...)` with `lua.read_*`; SimulationCode's `ParameterFields` can do the same). Extra keys that
SimulationCode does not have: `pofk_nbins`, `pofk_log_binning` (and the multipole equivalents) and
`fof_so_nmin_center`.

## The analyses

- `measure_power_spectrum` / `measure_power_spectrum_multipoles`: as SimulationCode's (default binning: nmesh/2
  linear bins). The multipoles take the velocity-to-displacement factor from the context.
- `find_halos`: FoF (Davis et al. 1985), then for each `fof_so_definitions` entry (`200m`, `500c`, `vir`, ...) a
  shrinking-sphere center and the outermost radius where the mean enclosed density is Delta times the reference
  (m: mean density of the particles, c: critical density, vir: Bryan & Norman 1998); halos whose SO center is inside
  a more massive halo are flagged (`so_overlap`). `write_halo_outputs` writes `halos_z<z>.txt` or `.hdf5` and
  `massfunc_z<z>.txt`, `massfunc_so<def>_z<z>.txt` (SimulationCode's names and columns; fits can be added as extra
  columns by the caller).
  With `fof_relaxed = true` the halos are found with relaxed FoF (`FriendsOfFriends/RelaxedFoF.h`, Dai et al. 2019
  as FastPM's RFOF; the redshift, E(z) and the mass and velocity units come from the context, so the velocities must
  be peculiar velocities); the catalogue then has the pass, bin, linking length and sigma_v / V_std of each halo.
- `find_voxel_voids`, `find_spherical_voids`, `find_zobov_voids` (FML/VoidFinders; settings `void_*`,
  `spherical_void_*`, `zobov_*`) and `write_void_catalogue`: `voids_z<z>.txt`, `voids_spherical_z<z>.txt`,
  `voids_zobov_z<z>.txt` (positions and radii in Mpc/h).
- `compute_dtfe_fields`, `dtfe_spectra`, `write_dtfe_spectra`, `write_dtfe_fields` (FML/Triangulation; settings
  `dtfe_*`): the density contrast, the peculiar velocity (km/s), its divergence and vorticity (km/s per Mpc/h) and
  their spectra, `dtfe_spectra_z<z>.txt`. Cells whose tetrahedra need particles beyond the buffer copied from the
  neighbor tasks come out as NaN: increase `dtfe_buffer_fraction` for large voids and many tasks.
- `baryonify`: the hosts are the resolved, non-truncated, non-overlapping M200c halos above the mass cut whose
  profile support is below half the box; the particles are displaced and redistributed. `write_baryonification_hosts`
  and `write_baryonified_power_spectrum` write `baryonification_z<z>.txt` and `pofk_baryonified_z<z>.txt`.

All functions are collective (call them on every task); an error on one task is turned into an exception on every
task.

## Using it from a simulation (sketch)

```cpp
FML::ANALYSIS::AnalysisContext context;
context.boxsize = boxsize;
context.scale_factor = a;
context.background.E_of_a = [&](double a) { return cosmo->HoverH0_of_a(a); };
context.background.Omega_m0 = cosmo->get_OmegaM();
context.mass_to_msunh = mean_particle_mass_msunh;
context.velocity_to_kms = 100.0 * boxsize / a;
auto halos = FML::ANALYSIS::find_halos<3>(part.get_particles_ptr(), part.get_npart(), fof_settings, context);
FML::ANALYSIS::write_halo_outputs<3>(snapshot_folder, halos, fof_settings, context);
```

SimulationCode itself is not changed: its `HaloFinding.h` and `BaryonificationOutput.cpp` are where this code
comes from, and switching it over is a separate step.

## Tests

`tests/test_analysis.cpp` (`make run`, or `make run USE_MPI=true` for 1, 2 and 4 tasks): the context and units, SO
definitions and thresholds, FoF/SO/overlap on particles with known halos, the mass function, redistribution and
baryonification (particles conserved and owned, hosts, P(k) unchanged on large scales), the void finders on a
particle distribution with a spherical hole, and (with `USE_CGAL=true`) ZOBOV and the DTFE velocity units.
