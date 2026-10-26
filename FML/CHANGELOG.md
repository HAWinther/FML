# Changelog

All notable changes to FML. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The entries below are relative to upstream commit `f9f666a` (April 2026).

## [Unreleased]

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
