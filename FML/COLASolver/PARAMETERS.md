# COLASolver parameters

Generated with `./nbody --parameters PARAMETERS.md` from `src/ParameterSchema.h`.

Parameters marked *optional* take the default value if they are not in the parameterfile, unless `all_parameters_must_be_in_file = true`. A parameter is only used if the condition in the *Used when* column holds.

## General

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `all_parameters_must_be_in_file` | bool | `false` | always | If true every parameter that is used must be in the parameterfile (no default values are used) |

## Simulation

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `simulation_boxsize` | double | *required* | always | Boxsize of the simulation (Mpc/h) |
| `simulation_name` | string | *required* | always | Label used in the output filenames |
| `simulation_use_cola` | bool | `false` | always | Use the COLA method |
| `simulation_use_scaledependent_cola` | bool | `true` | always | If the gravity model has scale-dependent growth: use scale-dependent COLA. If false we use the k = 0 limit of the growth factors when doing COLA |
| `simulation_enforce_LPT_trajectories` | bool | `false` | simulation_use_cola = true | Do not compute the PM force, just let particles follow the LPT trajectories |
| `simulation_cola_LPT_order` | int | `2` | simulation_use_cola = true | The LPT order of COLA (1, 2 or 3). Has to be <= ic_LPT_order. If < 1 COLA is turned off |

## Cosmology

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `cosmology_model` | string | `"LCDM"` | always | The cosmology (background evolution). Options: `LCDM`, `w0waCDM`, `DGP`, `JBD`, `Quintessence`, `EDE`, `Tabulated` |
| `cosmology_Omegab` | double | *required* | always | Baryon density parameter |
| `cosmology_OmegaMNu` | double | `0` | always | Massive neutrino density parameter |
| `cosmology_OmegaCDM` | double | *required* | always | CDM density parameter |
| `cosmology_OmegaK` | double | *required* | always | Curvature density parameter (-k/H0^2) |
| `cosmology_Neffective` | double | *required* | always | Effective number of relativistic species |
| `cosmology_TCMB_kelvin` | double | *required* | always | Temperature of the CMB today (Kelvin) |
| `cosmology_h` | double | *required* | always | Hubble parameter H0 = 100h km/s/Mpc |
| `cosmology_JBD_wBD` | double | *required* | cosmology_model = JBD | The JBD parameter (wBD -> infinity gives GR) |
| `cosmology_JBD_GeffG_today` | double | `1` | cosmology_model = JBD | The value of G today relative to Newtons constant (GeffG_today != 1 effectively gives a different Newtons constant) |
| `cosmology_JBD_density_parameter_definition` | string | `"hi-class"` | cosmology_model = JBD | Definition of the density parameters Omega = 8pi G_* rho / 3H0^2 (which G_* to use). Options: `Gbare`, `Gnewton`, `Gtoday`, `hi-class` |
| `cosmology_w0` | double | *required* | cosmology_model = w0waCDM | Dark energy equation of state w(a) = w0 + wa(1-a): w0 |
| `cosmology_wa` | double | *required* | cosmology_model = w0waCDM | Dark energy equation of state w(a) = w0 + wa(1-a): wa |
| `cosmology_quintessence_potential` | string | *required* | cosmology_model = Quintessence | The potential V = V0 u(phi), phi in units of Mpl: exponential u = exp(-lambda phi), inverse_power_law u = phi^-alpha, pngb u = 1 + cos(phi/f), hilltop u = (1 - (phi/mu)^p)^2. V0 is found so that OmegaDE today = 1 - OmegaM - OmegaR - OmegaK. Options: `exponential`, `inverse_power_law`, `pngb`, `hilltop` |
| `cosmology_quintessence_phi_ini` | double | `0` | cosmology_model = Quintessence | The initial value of the field (units of Mpl; deep in the radiation era) |
| `cosmology_quintessence_dphidlna_ini` | double | `0` | cosmology_model = Quintessence | The initial dphi/dlna (0: frozen) |
| `cosmology_quintessence_lambda` | double | *required* | cosmology_model = Quintessence and cosmology_quintessence_potential = exponential | The slope lambda |
| `cosmology_quintessence_alpha` | double | *required* | cosmology_model = Quintessence and cosmology_quintessence_potential = inverse_power_law | The power alpha |
| `cosmology_quintessence_f` | double | *required* | cosmology_model = Quintessence and cosmology_quintessence_potential = pngb | The scale f (units of Mpl) |
| `cosmology_quintessence_mu` | double | *required* | cosmology_model = Quintessence and cosmology_quintessence_potential = hilltop | The position of the minimum mu (units of Mpl) |
| `cosmology_quintessence_p` | double | *required* | cosmology_model = Quintessence and cosmology_quintessence_potential = hilltop | The power p |
| `cosmology_ede_fEDE` | double | *required* | cosmology_model = EDE | Early dark energy: the maximum of rho_EDE / rho_total |
| `cosmology_ede_zc` | double | *required* | cosmology_model = EDE | Early dark energy: the redshift of the maximum |
| `cosmology_ede_theta_ini` | double | `2.8` | cosmology_model = EDE | Early dark energy: the initial angle phi_ini / f |
| `cosmology_ede_n` | double | `3` | cosmology_model = EDE | Early dark energy: V = m^2 f^2 (1 - cos(phi/f))^n |
| `cosmology_tabulated_filename` | string | *required* | cosmology_model = Tabulated | The file with the background |
| `cosmology_tabulated_format` | string | *required* | cosmology_model = Tabulated | H: columns z, H/H0. CLASS: a CLASS background file. w: columns a, w(a) of the dark energy. Options: `H`, `CLASS`, `w` |
| `cosmology_dgp_OmegaRC` | double | *required* | cosmology_model = DGP | The cross-over scale of the self-accelerating DGP model (Omega_rc = 1/(4 rc^2 H0^2)) |

## Gravity model

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `gravity_model` | string | `"GR"` | always | The gravity model (growth of perturbations and forces). Options: `GR`, `f(R)`, `DGP`, `JBD`, `Symmetron`, `Geff`, `mbeta` |
| `gravity_model_geff_geffofa_filename` | string | *required* | gravity_model = Geff | File with the format [a, Geff/G(a)] |
| `gravity_model_mbeta_params` | double list | *required* | gravity_model = mbeta | The parameters (beta0, n, m0, m) of beta(a) = beta0 a^n and m(a) = m0 H0 a^m |
| `gravity_model_fofr_fofr0` | double | *required* | gravity_model = f(R) | The value of f_R today |
| `gravity_model_fofr_nfofr` | double | `1` | gravity_model = f(R) | The index n of the Hu-Sawicky model |
| `gravity_model_symmetron_assb` | double | `0.5` | gravity_model = Symmetron | Symmetry breaking scale factor (no fifth-force for a < assb) |
| `gravity_model_symmetron_beta` | double | `1` | gravity_model = Symmetron | Coupling strength relative to gravity |
| `gravity_model_symmetron_L_mpch` | double | `1` | gravity_model = Symmetron | Range of the fifth force in the background today (Mpc/h) |
| `gravity_model_dgp_rcH0overc` | double | *required* | gravity_model = DGP | The cross-over scale rc H0 / c |
| `gravity_model_screening` | bool | `true` | gravity_model in {mbeta, f(R), Symmetron, DGP} | Use the approximate screening model (otherwise linear) |
| `gravity_model_dgp_smoothing_filter` | string | `"tophat"` | gravity_model in {mbeta, f(R), Symmetron, DGP} and gravity_model_screening = true and gravity_model = DGP | Screening approximation: the filter used to smooth the density field. Options: `tophat`, `sharpk`, `gaussian` |
| `gravity_model_dgp_smoothing_scale_over_boxsize` | double | `1` | gravity_model in {mbeta, f(R), Symmetron, DGP} and gravity_model_screening = true and gravity_model = DGP | Screening approximation: the smoothing scale R/boxsize |
| `gravity_model_screening_enforce_largescale_linear` | bool | `false` | gravity_model in {mbeta, f(R), Symmetron, DGP} and gravity_model_screening = true | Combine the screened solution with the linear solution to get the correct linear evolution on large scales |
| `gravity_model_screening_linear_scale_hmpc` | double | `0.05` | gravity_model in {mbeta, f(R), Symmetron, DGP} and gravity_model_screening = true | Use the linear solution for k < k* and the screened solution for k > k* (h/Mpc) |
| `gravity_model_screening_efficiency` | double | `1` | gravity_model in {mbeta, f(R), Symmetron, DGP} and gravity_model_screening = true and gravity_model in {mbeta, f(R)} | Screening efficiency (1 is standard). Increases or decreases the amount of screening |
| `gravity_model_fofr_exact_solution` | bool | `false` | gravity_model = f(R) | Solve the exact f(R) equation with the multigrid solver (slow, mainly for testing) |
| `gravity_model_symmetron_exact_solution` | bool | `false` | gravity_model = Symmetron | Solve the exact symmetron equation with the multigrid solver (slow, mainly for testing) |
| `gravity_model_dgp_exact_solution` | bool | `false` | gravity_model = DGP | Solve the exact DGP equation with the multigrid solver (slow, mainly for testing; can be hard to get to converge) |
| `multigrid_nsweeps` | int | `10` | solving the exact equation of the gravity model | Multigrid solver: Newton-Gauss-Seidel sweeps on every level |
| `multigrid_nsweeps_first_step` | int | `20` | solving the exact equation of the gravity model | Multigrid solver: sweeps on the first step (increase if the solver fails to converge) |
| `multigrid_solver_residual_convergence` | double | `1e-06` | solving the exact equation of the gravity model | Multigrid solver: converged when the residual is below this |

## Primordial power-spectrum

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `cosmology_As` | double | `2e-09` | always | Primordial amplitude |
| `cosmology_ns` | double | `1` | always | Spectral index |
| `cosmology_kpivot_mpc` | double | `0.05` | always | Pivot scale (1/Mpc) |

## Particles

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `particle_Npart_1D` | int | *required* | always | Number of CDM+baryon particles per dimension |
| `particle_baryons` | bool | `false` | always | Evolve CDM and baryons as two sets of particles (particle_Npart_1D^3 of each, IC from the CDM and baryon transfer functions at zini). false: one set of CDM+baryon particles |
| `particle_neutrinos` | bool | `false` | always | Massive neutrinos as particles with the delta-f method (Elbers et al. 2021): thermal Fermi-Dirac velocities, relativistic equations of motion, the particles only carry the perturbation. Replaces force_linear_massive_neutrinos (needs cosmology_OmegaMNu > 0, transferinfofile, timestep_method = Quinn, no COLA) |
| `particle_neutrino_Npart_1D` | int | `0` | always | particle_neutrinos: number of neutrino particles per dimension (0: particle_Npart_1D) |
| `particle_neutrino_deltaf` | bool | `true` | always | particle_neutrinos: use the delta-f weights (false: plain particles, much noisier; for testing) |
| `particle_neutrino_energy_density` | bool | `false` | always | particle_neutrinos: source the Poisson equation with the energy density of the neutrinos (each particle weighted by E/m = sqrt(1 + (q/am)^2)) instead of their mass density. The neutrino P(k) is then that of the energy-density contrast, as in CAMB/CLASS. Matters at high redshift for light neutrinos (<E/m> = 1.04 at z = 50 for 0.1 eV) |
| `particle_baryon_lattice` | string | `"same"` | always | particle_baryons: start the baryons from the same lattice as the CDM (same, recommended: the baryon-CDM difference then follows linear theory on all scales) or from a lattice offset by half a particle spacing (offset: gives a spurious baryon-CDM difference on small scales). Options: `same`, `offset` |
| `particle_allocation_factor` | double | `1.5` | always | Allocate this factor more particles than the average per task (particles move between tasks) |
| `particle_sort_every_nsteps` | int | `0` | always | Reorder the particles in memory by position (blocks of 4^3 force-grid cells) every this many steps, which makes the density assignment and the force interpolation faster when the particles have moved far from their initial (lattice) order. 0: never. Changes the results only by round-off (the order of the sums) and the order of the particles in the output |

## Time-stepping

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `timestep_nsteps` | int list | *required* | always | Number of steps between the outputs (one per output redshift). If only one number it is the total number of steps |
| `timestep_algorithm` | string | `"KDK"` | always | The time-stepping algorithm |
| `timestep_method` | string | `"Quinn"` | always | The time-stepping method. Quinn: standard leapfrog. Tassev: modified leapfrog for COLA. FastPM: leapfrog that follows linear growth exactly (Feng et al. 2016). BullFrog: follows 2LPT growth after every step (Rampf, List & Hahn 2024), not for COLA. Options: `Quinn`, `Tassev`, `FastPM`, `BullFrog` |
| `timestep_cola_nLPT` | double | `-2.5` | always | The nLPT parameter of the Tassev time-stepping |
| `timestep_scalefactor_spacing` | string | `"linear"` | always | Spacing of the steps in the scale factor (powerlaw: uniform in a^timestep_spacing_power, growthfactor: uniform in the linear growth factor D(a), as in Rampf, List & Hahn 2024). Options: `linear`, `logarithmic`, `powerlaw`, `growthfactor` |
| `timestep_spacing_power` | double | `1` | timestep_scalefactor_spacing = powerlaw | The steps are uniform in a^power |

## Initial conditions

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `ic_initial_redshift` | double | *required* | always | The initial redshift of the simulation |
| `ic_random_seed` | int | *required* | always | The random seed |
| `ic_random_generator` | string | `"MT19937"` | always | The random number generator (GSL is gsl_rng_ranlxd1 as used in 2LPTic, for comparisons). Options: `GSL`, `MT19937` |
| `ic_nmesh` | int | *required* | always | The grid size used to generate the IC |
| `ic_lattice_offset` | string | `"fiducial"` | always | Offset of the initial particle lattice from the nodes of the force grid. fiducial: half a force grid cell if force_nmesh > particle_Npart_1D, otherwise none. With particles exactly on the nodes of a finer grid, CIC responds to the size of the displacement and gives large spurious forces (see the force grid section of README.md). Options: `fiducial`, `none`, `half_force_cell` |
| `ic_type_of_input` | string | *required* | always | The input: powerspectrum (file with [k (h/Mpc), P(k) (Mpc/h)^3]), transferfunction (file with [k (h/Mpc), T(k) Mpc^2]) or transferinfofile (file listing T(k,z) files from CAMB/CLASS). Options: `powerspectrum`, `transferfunction`, `transferinfofile` |
| `ic_type_of_input_fileformat` | string | `"CAMB"` | always | Format of the transfer function files (CLASS run with format=camb). Options: `CAMB`, `CLASS`, `AXIONCAMB` |
| `ic_input_filename` | string | *required* | always | The input file |
| `ic_input_redshift` | double | *required* | always | The redshift of the input P(k) / T(k) |
| `ic_fix_amplitude` | bool | `true` | always | Fix the amplitude of the modes when generating the random field |
| `ic_reverse_phases` | bool | `false` | always | Reverse the phases (for paired simulations) |
| `ic_random_field_type` | string | `"gaussian"` | always | Type of IC: gaussian, nongaussian, read_particles (read a GADGET file and use that, reconstructing the LPT fields if COLA) or read_phases (take the phases from a GADGET file). Options: `gaussian`, `nongaussian`, `read_particles`, `read_phases` |
| `ic_LPT_order` | int | `2` | always | The LPT order used to make the IC |
| `ic_sigma8_normalization` | bool | `false` | always | Normalize P(k) to a given sigma8 (otherwise use the input) |
| `ic_sigma8_redshift` | double | `0` | always | The redshift of the sigma8 value |
| `ic_sigma8` | double | `0.8` | always | The sigma8 value to normalize to |
| `ic_use_gravity_model_GR` | bool | `false` | always | For modified gravity: the input P(k) is for LCDM, use GR growth factors to scale it to the initial redshift (same IC as the LCDM simulation) |
| `ic_fnl_type` | string | *required* | ic_random_field_type = nongaussian | Type of non-gaussianity. Options: `local`, `equilateral`, `orthogonal` |
| `ic_fnl` | double | *required* | ic_random_field_type = nongaussian | The value of fNL |
| `ic_fnl_redshift` | double | *required* | ic_random_field_type = nongaussian | The redshift at which the non-gaussian potential is applied |
| `ic_reconstruct_gadgetfilepath` | string | *required* | ic_random_field_type in {read_particles, read_phases} | Path to the GADGET files without the file number: binary (path.0, path.1, ...) or HDF5 (path.hdf5 or path.0.hdf5, ...), detected automatically |
| `ic_reconstruct_assigment_method` | string | `"CIC"` | ic_random_field_type in {read_particles, read_phases} | Density assignment method used to reconstruct the density field. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |
| `ic_reconstruct_smoothing_filter` | string | `"sharpk"` | ic_random_field_type in {read_particles, read_phases} | Filter used to remove small-scale modes from the reconstructed density field. Options: `tophat`, `sharpk`, `gaussian` |
| `ic_reconstruct_dimless_smoothing_scale` | double | `0` | ic_random_field_type in {read_particles, read_phases} | Smoothing scale R/boxsize of the reconstructed density field |
| `ic_reconstruct_interlacing` | bool | `false` | ic_random_field_type in {read_particles, read_phases} | Use interlacing when reconstructing the density field |
| `ic_reconstruct_exact` | bool | `false` | ic_random_field_type in {read_particles, read_phases} | Reconstruct the LPT fields exactly from the positions and velocities (always works for 1LPT IC; for 2LPT IC the particles must come from a regular grid) |

## Forces

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `force_nmesh` | int | *required* | always | Grid size used to compute the PM forces |
| `force_density_assignment_method` | string | `"CIC"` | always | Density assignment method. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |
| `force_greens_function_kernel` | string | `"fiducial"` | always | Kernel for D^2 when solving the Poisson equation (fiducial = continuous). Options: `fiducial`, `continuous`, `discrete_2pt`, `discrete_4pt` |
| `force_gradient_kernel` | string | `"fiducial"` | always | Kernel for D when computing the forces. fiducial = continuous if force_nmesh = particle_Npart_1D, discrete_4pt if force_nmesh is up to 3x larger and discrete_2pt if it is even larger (the continuous kernel gives spurious forces from the particle lattice when the force grid is finer than it). Options: `fiducial`, `continuous`, `discrete_2pt`, `discrete_4pt` |
| `force_density_method` | string | `"assignment"` | always | How the density field for the force is computed. assignment: particles to grid with force_density_assignment_method. nufft: the exact Fourier modes of the particle distribution (non-uniform FFT; no window function or aliasing). sheet: spectral sheet interpolation, R^NDIM tracers per particle placed by Fourier interpolating the displacement field x - q, assigned with force_density_assignment_method. sheet_nufft: sheet interpolation with the exact Fourier modes of the tracers (NUFFT). The force is always interpolated back to the particles with force_density_assignment_method. Options: `assignment`, `nufft`, `sheet`, `sheet_nufft` |
| `force_interlacing_order` | int | `1` | always | Force interlacing: compute the force for this many copies of the particles shifted by a fraction of a cell relative to the grid and average (at the particles). 1: off, 2: shift (1/2,1/2,1/2), 4: the half-cell face diagonals, 8: all half-cell corners, 64: quarter cells. Cancels the leading aliasing of the density assignment and the force interpolation at the cost of this many force calculations |
| `force_deconvolve_window` | bool | `false` | always | Divide the force by W(k)^2 where W is the window function of force_density_assignment_method (corrects for the smoothing of the assignment and the interpolation). Unstable unless the aliasing is suppressed (force_interlacing_order > 1, see README) |
| `force_early_method` | string | `"none"` | always | A more accurate force at early times (when the particles are still close to the lattice and CIC is inaccurate): nufft (NUFFT density and interpolation) or interlaced (force_early_interlacing_order shifted grids with force_early_density_assignment_method and window deconvolution) for steps at z > force_early_until_redshift, then the normal force settings. none: off. Options: `none`, `nufft`, `interlaced` |
| `force_early_until_redshift` | double | `10` | force_early_method in {nufft, interlaced} | Use the early force for z > this |
| `force_early_interlacing_order` | int | `2` | force_early_method = interlaced | Number of shifted grids at early times |
| `force_early_density_assignment_method` | string | `"TSC"` | force_early_method = interlaced | Assignment kernel at early times. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |
| `force_interpolation_method` | string | `"fiducial"` | always | How the force is interpolated to the particles. assignment: with the force_density_assignment_method kernel. nufft: exact evaluation of the Fourier series of the force at the particles (non-uniform FFT). fiducial: nufft if force_density_method = nufft (then assignment and interpolation are both exact, so there is no self-force), otherwise assignment. Options: `fiducial`, `assignment`, `nufft` |
| `force_nufft_epsilon` | double | `1e-05` | always | Relative accuracy of the NUFFT (force_density_method = nufft or sheet_nufft, or force_interpolation_method = nufft) |
| `force_sheet_refinement` | int | `2` | force_density_method in {sheet, sheet_nufft} | R: the number of tracers per particle per dimension for sheet interpolation |
| `force_linear_massive_neutrinos` | bool | `false` | always | Include massive neutrinos in the density field using linear theory (requires ic_type_of_input = transferinfofile) |
| `force_nbody_gauge` | bool | `false` | always | Run in the N-body gauge: add the linear source from radiation, neutrinos and GR (made with CLASS by scripts/make_nbody_gauge_transfer.py) to the density field |
| `force_nbody_gauge_filename` | string | *required* | force_nbody_gauge = true | The table T_extra(k,z) (nbgauge_extra.txt from scripts/make_nbody_gauge_transfer.py) |
| `force_use_finite_difference_force` | bool | `false` | always | Experimental: compute forces with finite differences of the potential |
| `force_finite_difference_stencil_order` | int | `4` | always | Experimental: order of the finite difference stencil |

## Lightcone

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `lightcone` | bool | `false` | always | Make a lightcone |
| `plc_pos_observer` | double list | `{0, 0, 0}` | lightcone = true | Position of the observer in units of the boxsize |
| `plc_z_init` | double | `0` | lightcone = true | The redshift we start recording the lightcone |
| `plc_z_finish` | double | `0` | lightcone = true | The redshift we stop recording the lightcone |
| `plc_boundary_mpch` | double | `20` | lightcone = true | Boundary region around the shell to make sure we get all particles (Mpc/h) |
| `plc_use_replicas` | bool | `false` | lightcone = true | Replicate the box to get the sky coverage we want |
| `plc_ndim_rep` | int | `0` | lightcone = true | Number of dimensions with replicas in both directions (3D: 0 = octant, 3 = full sky) |
| `plc_output_gadgetfile` | bool | `false` | lightcone = true | Output the lightcone particles in GADGET format |
| `plc_output_asciifile` | bool | `false` | lightcone = true | Output the lightcone particles in ascii |
| `plc_output_hdf5file` | bool | `false` | lightcone = true | Output the lightcone particles as GADGET HDF5 files (needs USE_HDF5; positions in Mpc/h, peculiar velocities in km/s) |
| `plc_output_in_batches` | bool | `false` | lightcone = true | Output in batches to save memory |
| `plc_make_onion_density_maps` | bool | `false` | lightcone = true | Make (Healpix) density maps delta(z, theta) |
| `plc_nside` | int | `256` | lightcone = true and plc_make_onion_density_maps = true | Healpix nside of the maps (npix = 12 nside^2) |
| `plc_da_maps` | double | `0.02` | lightcone = true and plc_make_onion_density_maps = true | Roughly the size of the bins in a for the maps |
| `plc_use_chunkpix` | bool | `false` | lightcone = true and plc_make_onion_density_maps = true | Store the maps with chunkpix (useful for sparse maps) |
| `plc_nside_chunks` | int | `8` | lightcone = true and plc_make_onion_density_maps = true and plc_use_chunkpix = true | Healpix nside of the chunks |

## Output

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `output_folder` | string | *required* | always | Folder to put the output in ("" is the current folder) |
| `output_redshifts` | double list | *required* | always | The output redshifts (decreasing) |
| `output_particles` | bool | `true` | always | Write the particles at the output redshifts |
| `output_fileformat` | string | `"GADGET"` | always | File format for the particles. GADGET: binary GADGET files. HDF5: GADGET HDF5 files (compile with USE_HDF5=true). Options: `GADGET`, `HDF5` |

## Halo finding

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `fof` | bool | `false` | always | Locate friends-of-friends halos at every output |
| `fof_nmin_per_halo` | int | `20` | fof = true | Minimum number of particles per halo (of one species with particle_baryons) |
| `fof_linking_length` | double | `0.2` | fof = true | The linking length in units of the mean particle separation (of one species with particle_baryons) |
| `fof_nmesh_max` | int | `0` | fof = true | Maximum grid size used to bin particles to speed up the linking (0: the code chooses) |
| `fof_buffer_length_mpch` | double | `3` | fof = true | Buffer region (larger than the largest halo) from neighbor tasks (Mpc/h). The code warns if halos reach its edge (they might then be cut off and found twice) |
| `fof_output_fileformat` | string | `"ascii"` | always | fof: the halo catalogue halos_z<z>.txt (ascii, one row per halo) or halos_z<z>.hdf5 (HDF5, compile with USE_HDF5=true). Options: `ascii`, `HDF5` |
| `fof_massfunction_nbins` | int | `30` | always | fof: number of logarithmic bins of the mass function |
| `fof_massfunction_Mmin` | double | `1e+10` | always | fof: smallest mass of the mass function (Msun/h) |
| `fof_massfunction_Mmax` | double | `1e+16` | always | fof: largest mass of the mass function (Msun/h) |
| `fof_so_definitions` | string list | `{}` | always | fof: spherical overdensity masses around the FoF halos, e.g. {"200m", "500c", "vir"} (<Delta>m: Delta times the mean CDM+baryon density, <Delta>c: the critical density, vir: Bryan & Norman 1998). Empty: none. The spheres must fit in fof_buffer_length_mpch |

## Power-spectrum

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `pofk` | bool | `false` | always | Compute the power-spectrum at every output |
| `pofk_every_step` | bool | `true` | always | Write P(k) of the density field used for the force at every step (cheap; same grid and assignment as the force, the window is deconvolved): pofk_<name>_cb_z<z>.txt in output_folder |
| `pofk_nmesh` | int | `0` | pofk = true | Grid size |
| `pofk_interlacing` | bool | `false` | pofk = true | Use interlaced grids to reduce aliasing |
| `pofk_subtract_shotnoise` | bool | `false` | pofk = true | Subtract shot-noise |
| `pofk_density_assignment_method` | string | `"CIC"` | pofk = true | Density assignment method. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |

## Power-spectrum multipoles

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `pofk_multipole` | bool | `false` | always | Compute the redshift space multipoles P_ell(k) at every output |
| `pofk_multipole_nmesh` | int | *required* | pofk_multipole = true | Grid size |
| `pofk_multipole_interlacing` | bool | `false` | pofk_multipole = true | Use interlaced grids to reduce aliasing |
| `pofk_multipole_subtract_shotnoise` | bool | `false` | pofk_multipole = true | Subtract shot-noise from P0 |
| `pofk_multipole_ellmax` | int | `4` | pofk_multipole = true | Compute P_ell for ell = 0, 2, ..., ellmax |
| `pofk_multipole_density_assignment_method` | string | `"CIC"` | pofk_multipole = true | Density assignment method. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |

## Bispectrum

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `bispectrum` | bool | `false` | always | Compute the bispectrum at every output |
| `bispectrum_nmesh` | int | *required* | bispectrum = true | Grid size |
| `bispectrum_nbins` | int | *required* | bispectrum = true | Number of k-bins (memory and time scale as nbins grids and nbins^3 integrals) |
| `bispectrum_density_assignment_method` | string | `"CIC"` | bispectrum = true | Density assignment method. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |
| `bispectrum_interlacing` | bool | `true` | bispectrum = true | Use interlaced grids to reduce aliasing |
| `bispectrum_subtract_shotnoise` | bool | `false` | bispectrum = true | Subtract shot-noise |

## Hybrid bias spectra

| Parameter | Type | Default | Used when | Description |
|---|---|---|---|---|
| `hybridbias` | bool | `false` | always | Compute the hybrid bias (Lagrangian bias basis) spectra P_ab(k) at every output: the cross spectra of the operators 1, delta_L, delta_L^2, s^2, nabla^2 delta_L advected with the particles (Modi et al. 2020, Kokron et al. 2021). delta_L is the linear field of the IC scaled to the output redshift with the growth factor of the gravity model |
| `hybridbias_nmesh` | int | *required* | hybridbias = true | Grid size for the advected fields |
| `hybridbias_noperators` | int | `5` | hybridbias = true | Use the first noperators of (1, delta, delta^2, s^2, nabla^2 delta): 15 spectra for 5 |
| `hybridbias_density_assignment_method` | string | `"CIC"` | hybridbias = true | Density assignment method. Options: `NGP`, `CIC`, `TSC`, `PCS`, `PQS` |
| `hybridbias_interlacing` | bool | `true` | hybridbias = true | Use interlaced grids to reduce aliasing |
| `hybridbias_subtract_shotnoise` | bool | `false` | hybridbias = true | Subtract the Poisson shot noise mean(w_a w_b)/Npart (the usual basis spectra keep it) |
| `hybridbias_smoothing_scale` | double | `0` | hybridbias = true | Smooth the linear field with this scale (Mpc/h) before forming the operators (0: none) |
| `hybridbias_smoothing_filter` | string | `"gaussian"` | hybridbias = true | The smoothing filter. Options: `tophat`, `sharpk`, `gaussian` |
| `hybridbias_kcut_over_nyquist` | double | `1` | hybridbias = true | Keep only the modes with k < this times the Nyquist frequency of the particle lattice in the linear field (a spherical cut, so the operators are isotropic as assumed in the LPT prediction). 0: keep all the modes of the lattice grid (a cube) |
| `hybridbias_zeldovich` | bool | `false` | hybridbias = true | Also write the Zel'dovich (resummed 1LPT) prediction of the spectra with the same cut and smoothing of the operators: hybridbias_zeldovich_z<z>.txt |
| `hybridbias_oneloop` | bool | `false` | hybridbias = true | Also write the one-loop LPT (CLEFT) prediction of the spectra (no counterterms): hybridbias_oneloop_z<z>.txt |

## Options with separate particle species

Not everything is implemented yet for CDM and baryons as separate particles (`particle_baryons`) and for neutrino particles (`particle_neutrinos`). The code stops with a message if one of these is not satisfied (generated from `species_requirements()` in `src/SimulationParameters.h`):

| Needs | `particle_baryons` | `particle_neutrinos` |
|---|---|---|
| `timestep_method = Quinn` |  | yes |
| `simulation_use_cola = false` |  | yes |
| `timestep_method` other than `Tassev` with COLA (the modified Tassev operators get the baryon-CDM relative velocity wrong: +30% in P_b-c with 20 linear steps) | yes |  |
| `simulation_cola_LPT_order` <= `ic_LPT_order` (with COLA) | yes |  |
| `simulation_enforce_LPT_trajectories = false` | yes | yes |
| `ic_type_of_input = transferinfofile` (the transfer functions of the species at zini) | yes | yes |
| `ic_random_field_type = gaussian` or `nongaussian` | yes | yes |
| `ic_LPT_order = 1` or `2` | yes |  |
| `force_linear_massive_neutrinos = false` (the particles replace the linear neutrinos) |  | yes |
| `force_density_method = assignment` | yes | yes |
| `force_interpolation_method` other than `nufft` | yes | yes |
| `force_interlacing_order = 1` | yes | yes |
| `force_early_method = none` | yes | yes |
| `force_nbody_gauge = false` | yes | yes |
| `lightcone = false` | yes | yes |
| `pofk_multipole = false` | yes | yes |
| `bispectrum = false` | yes | yes |
| `hybridbias = false` | yes |  |
