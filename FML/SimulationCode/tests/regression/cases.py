#!/usr/bin/env python3
#==========================================================================
# Regression test cases for the SimulationCode
#
# Each case is a set of overrides of the BASE parameters below. The cases
# are small (32^3 particles) so the whole suite runs in about a minute, but
# together they exercise all the main code paths of the simulation:
# COLA / no COLA, scale-dependent COLA, the time-stepping methods and
# spacings, all the IC input types and random field types, massive
# neutrinos, modified gravity, the lightcone and all on-the-fly analysis
#
# Usage: python3 cases.py <outdir>   (writes <outdir>/<case>/parameterfile.lua)
#==========================================================================
import os
import sys

# The SimulationCode input folder (power spectra, transfer functions)
INPUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "input"))

BASE = dict(
    all_parameters_must_be_in_file=True,
    simulation_name="Reg",
    simulation_boxsize=256.0,
    simulation_use_cola=True,
    simulation_use_scaledependent_cola=False,
    simulation_enforce_LPT_trajectories=False,
    simulation_cola_LPT_order=2,
    cosmology_model="LCDM",
    cosmology_OmegaCDM=0.2685,
    cosmology_Omegab=0.049,
    cosmology_OmegaMNu=0.0,
    cosmology_neutrino_masses_eV=[],
    cosmology_OmegaK=0.0,
    cosmology_Neffective=3.046,
    cosmology_TCMB_kelvin=2.7255,
    cosmology_h=0.671,
    cosmology_As=2.215e-9,
    cosmology_ns=0.966,
    cosmology_kpivot_mpc=0.05,
    gravity_model="GR",
    particle_Npart_1D=32,
    particle_allocation_factor=1.5,
    output_redshifts=[1.0, 0.0],
    output_particles=True,
    output_fileformat="GADGET",
    output_folder="output",
    timestep_nsteps=[3, 3],
    timestep_method="Quinn",
    timestep_cola_nLPT=-2.5,
    timestep_algorithm="KDK",
    timestep_scalefactor_spacing="linear",
    timestep_spacing_power=1.0,
    ic_random_seed=1234,
    ic_random_generator="GSL",
    ic_fix_amplitude=False,
    ic_reverse_phases=False,
    ic_random_field_type="gaussian",
    ic_nmesh=32,
    ic_use_gravity_model_GR=False,
    ic_LPT_order=2,
    ic_type_of_input="powerspectrum",
    ic_input_filename=INPUT + "/example_power_spectrum_cb_z0.000.txt",
    ic_input_redshift=0.0,
    ic_initial_redshift=20.0,
    ic_sigma8_normalization=False,
    ic_sigma8_redshift=0.0,
    ic_sigma8=0.83,
    force_nmesh=32,
    force_density_assignment_method="CIC",
    force_greens_function_kernel="fiducial",
    force_gradient_kernel="fiducial",
    force_linear_massive_neutrinos=True,
    force_use_finite_difference_force=False,
    force_finite_difference_stencil_order=4,
    lightcone=False,
    fof=False,
    fof_nmin_per_halo=20,
    fof_linking_length=0.2,
    fof_nmesh_max=0,
    fof_buffer_length_mpch=3.0,
    pofk=True,
    pofk_nmesh=32,
    pofk_interlacing=True,
    pofk_subtract_shotnoise=False,
    pofk_density_assignment_method="PCS",
    pofk_multipole=False,
    pofk_multipole_nmesh=32,
    pofk_multipole_interlacing=True,
    pofk_multipole_subtract_shotnoise=False,
    pofk_multipole_ellmax=4,
    pofk_multipole_density_assignment_method="PCS",
    bispectrum=False,
    bispectrum_nmesh=32,
    bispectrum_nbins=4,
    bispectrum_interlacing=True,
    bispectrum_subtract_shotnoise=False,
    bispectrum_density_assignment_method="PCS",
)

# Particles written at the initial redshift by the 'cola_quinn_all' case are used
# as input for the read_particles / read_phases cases
IC_GADGET = "../cola_quinn_all/output/snapshot_Reg_z20.000/gadget_z20.000"
# ...and by 'hdf5_output' for 'read_particles_hdf5'
IC_GADGET_HDF5 = "../hdf5_output/output/snapshot_Reg_z20.000/gadget_z20.000"

CASES = {
    # COLA, Quinn, all analysis on, output also at the initial redshift, sigma8 normalization
    "cola_quinn_all": dict(
        output_redshifts=[20.0, 1.0, 0.0],
        timestep_nsteps=[0, 3, 3],
        ic_sigma8_normalization=True,
        fof=True,
        fof_nmin_per_halo=10,
        pofk_multipole=True,
        bispectrum=True,
    ),
    # No COLA, Tassev time-stepping, logarithmic spacing, fixed amplitude + reversed phases,
    # a single total number of steps split over three outputs, transfer function input,
    # IC from GR growth, FML particle output
    "nocola_tassev_log": dict(
        simulation_use_cola=False,
        timestep_method="Tassev",
        timestep_scalefactor_spacing="logarithmic",
        # 10 steps split over the outputs: 8/1/1 (8 gave 7/0/1, i.e. no step between z = 1 and 0.5: now an error)
        timestep_nsteps=[10],
        output_redshifts=[1.0, 0.5, 0.0],
        ic_fix_amplitude=True,
        ic_reverse_phases=True,
        ic_type_of_input="transferfunction",
        ic_input_filename=INPUT + "/example_transfer_function_cb_z0.000.txt",
        ic_use_gravity_model_GR=True,
        ic_random_generator="MT19937",
    ),
    # Scale-dependent COLA with massive neutrinos from transfer tables (input/transfer_tables, tests/make_transfer_tables.py)
    "scaledep_cola_neutrinos": dict(
        simulation_use_scaledependent_cola=True,
        cosmology_OmegaMNu=0.0048,
        cosmology_OmegaCDM=0.2637,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.2.txt",
        ic_input_redshift=20.0,
    ),
    # f(R) gravity (approximate screening), 1LPT COLA, TSC assignment
    "fofr_cola1_tsc": dict(
        gravity_model="f(R)",
        gravity_model_fofr_fofr0=1e-5,
        gravity_model_fofr_nfofr=1.0,
        gravity_model_fofr_exact_solution=False,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=True,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=5,
        multigrid_coarse_factor=1,
        simulation_cola_LPT_order=1,
        force_density_assignment_method="TSC",
    ),
    # f(R) and DGP with the exact field equations solved with the multigrid solver
    "fofr_exact_multigrid": dict(
        gravity_model="f(R)",
        gravity_model_fofr_fofr0=1e-5,
        gravity_model_fofr_nfofr=1.0,
        gravity_model_fofr_exact_solution=True,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=10,
        multigrid_coarse_factor=1,
    ),
    "symmetron_exact_multigrid": dict(
        gravity_model="Symmetron",
        gravity_model_symmetron_assb=0.5,
        gravity_model_symmetron_beta=1.0,
        gravity_model_symmetron_L_mpch=8.0,
        gravity_model_symmetron_N=2.0,
        gravity_model_symmetron_M=4.0,
        gravity_model_symmetron_exact_solution=True,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=10,
        multigrid_coarse_factor=1,
    ),
    "dilaton_exact_multigrid": dict(
        gravity_model="Dilaton",
        gravity_model_dilaton_A2=5e5,
        gravity_model_dilaton_beta0=0.5,
        gravity_model_dilaton_r=1.0,
        gravity_model_dilaton_xi=1e-3,
        gravity_model_dilaton_exact_solution=True,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=10,
        multigrid_coarse_factor=1,
        multigrid_start_redshift=4.0,
        multigrid_force_before_start="linear",
    ),
    "dgp_exact_multigrid": dict(
        gravity_model="DGP",
        gravity_model_dgp_rcH0overc=1.0,
        gravity_model_dgp_exact_solution=True,
        gravity_model_screening=True,
        gravity_model_dgp_smoothing_filter="tophat",
        gravity_model_dgp_smoothing_scale_over_boxsize=0.0,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=10,
        multigrid_coarse_factor=1,
    ),
    # Non-gaussian (local fNL) initial conditions
    "nongaussian": dict(
        ic_random_field_type="nongaussian",
        ic_fnl_type="local",
        ic_fnl=100.0,
        ic_fnl_redshift=20.0,
    ),
    # Power-law spacing of the time steps
    "powerlaw_spacing": dict(
        timestep_scalefactor_spacing="powerlaw",
        timestep_spacing_power=2.0,
    ),
    # Lightcone (particles written as ascii, no maps), DGP gravity
    "lightcone_dgp": dict(
        gravity_model="DGP",
        gravity_model_dgp_rcH0overc=1.0,
        gravity_model_dgp_exact_solution=False,
        gravity_model_screening=True,
        gravity_model_dgp_smoothing_filter="tophat",
        gravity_model_dgp_smoothing_scale_over_boxsize=0.0,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=True,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-4,
        multigrid_nsweeps=2,
        multigrid_nsweeps_first_step=2,
        multigrid_coarse_factor=1,
        lightcone=True,
        plc_pos_observer=[0.0, 0.0, 0.0],
        plc_boundary_mpch=20.0,
        plc_z_init=0.05,
        plc_z_finish=0.0,
        plc_use_replicas=False,
        plc_ndim_rep=3,
        plc_output_gadgetfile=False,
        plc_output_asciifile=True,
        plc_output_in_batches=True,
        plc_make_onion_density_maps=False,
        output_redshifts=[0.0],
        timestep_nsteps=[6],
    ),
    # N-body gauge: IC from the N-body gauge transfer functions and the linear source from radiation,
    # neutrinos and GR (the tables of the internal solver in the N-body gauge)
    "nbody_gauge": dict(
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nbody_gauge.txt",
        ic_input_redshift=20.0,
        force_nbody_gauge=True,
    ),
    # Force grid finer than the particle grid (the fiducial gradient kernel is then discrete_4pt and the lattice is offset), PCS
    "fine_force_grid": dict(
        force_nmesh=64,
        force_density_assignment_method="PCS",
    ),
    # simulation_use_scaledependent_cola = true in GR with a force grid finer than the particle grid: the growth is
    # scale-independent, so standard COLA is used and the result is that of fine_force_grid (scale-dependent COLA with
    # the lattice offset is covered by mu_bz_scaledep_cola)
    "scaledep_cola_fine_force_grid": dict(
        simulation_use_scaledependent_cola=True,
        force_nmesh=64,
        force_density_assignment_method="PCS",
    ),
    # Hybrid bias spectra with the Zel'dovich prediction
    "hybridbias_zeldovich": dict(
        hybridbias=True,
        hybridbias_nmesh=32,
        hybridbias_noperators=5,
        hybridbias_density_assignment_method="PCS",
        hybridbias_interlacing=True,
        hybridbias_subtract_shotnoise=False,
        hybridbias_smoothing_scale=0.0,
        hybridbias_smoothing_filter="gaussian",
        hybridbias_kcut_over_nyquist=0.5,
        hybridbias_zeldovich=True,
        hybridbias_oneloop=False,
    ),
    # Force with NUFFTs (exact density and force interpolation, no window functions)
    "nufft_force": dict(
        force_density_method="nufft",
    ),
    # Density from spectral sheet interpolation (R = 2) with the exact Fourier modes of the tracers
    "sheet_nufft_force": dict(
        force_density_method="sheet_nufft",
        force_sheet_refinement=2,
        force_interpolation_method="nufft",
    ),
    # Force interlacing (2 shifted grids) with PCS and deconvolution of the window function
    "interlaced_force": dict(
        force_interlacing_order=2,
        force_density_assignment_method="PCS",
        force_deconvolve_window=True,
    ),
    # BullFrog from (almost) the Big Bang: Zel'dovich initial conditions at z = 1e7
    "bullfrog_bigbang": dict(
        simulation_use_cola=False,
        timestep_method="BullFrog",
        ic_initial_redshift=1e7,
        ic_LPT_order=1,
        force_density_method="nufft",
    ),
    # Early-time force switch: NUFFT for z > 5, then the normal CIC force
    "early_force_switch": dict(
        force_early_method="nufft",
        force_early_until_redshift=5.0,
    ),
    # FastPM time-stepping with COLA, logarithmic spacing, f(R) gravity
    "fastpm_cola_fofr": dict(
        timestep_method="FastPM",
        timestep_scalefactor_spacing="logarithmic",
        gravity_model="f(R)",
        gravity_model_fofr_fofr0=1e-5,
        gravity_model_fofr_nfofr=1.0,
        gravity_model_fofr_exact_solution=False,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=True,
        gravity_model_screening_linear_scale_hmpc=0.1,
    ),
    # BullFrog time-stepping (no COLA) with the lightcone
    "bullfrog_lightcone": dict(
        simulation_use_cola=False,
        timestep_method="BullFrog",
        lightcone=True,
        plc_pos_observer=[0.0, 0.0, 0.0],
        plc_boundary_mpch=20.0,
        plc_z_init=0.05,
        plc_z_finish=0.0,
        plc_use_replicas=False,
        plc_ndim_rep=3,
        plc_output_gadgetfile=False,
        plc_output_asciifile=True,
        plc_output_in_batches=True,
        plc_make_onion_density_maps=False,
        output_redshifts=[1.0, 0.0],
        timestep_nsteps=[3, 3],
    ),
    # Quintessence (exponential potential) with BullFrog
    "quintessence_exponential": dict(
        cosmology_model="Quintessence",
        cosmology_quintessence_potential="exponential",
        cosmology_quintessence_lambda=1.0,
        cosmology_quintessence_phi_ini=0.0,
        cosmology_quintessence_dphidlna_ini=0.0,
        simulation_use_cola=False,
        timestep_method="BullFrog",
    ),
    # Thawing PNGB quintessence with COLA
    "quintessence_pngb": dict(
        cosmology_model="Quintessence",
        cosmology_quintessence_potential="pngb",
        cosmology_quintessence_f=1.0,
        cosmology_quintessence_phi_ini=1.5,
        cosmology_quintessence_dphidlna_ini=0.0,
    ),
    # Early dark energy
    "early_dark_energy": dict(
        cosmology_model="EDE",
        cosmology_ede_fEDE=0.1,
        cosmology_ede_zc=3500.0,
        cosmology_ede_theta_ini=2.8,
        cosmology_ede_n=3.0,
    ),
    # Background from a CLASS file (w0 = -0.9, wa = 0.1)
    "tabulated_class": dict(
        cosmology_model="Tabulated",
        cosmology_tabulated_filename=INPUT + "/class_background_w0wa.dat",
        cosmology_tabulated_format="CLASS",
    ),
    # Dark matter - dark energy scattering: the velocity friction in the growth (COLA frame, IC) and in the time
    # stepping (COLA + Quinn; no COLA + FastPM)
    "dark_scattering_cola": dict(
        cosmology_model="DarkScattering",
        cosmology_ds_w=-0.9,
        cosmology_ds_xi=50.0,
        cosmology_ds_cs2=1.0,
    ),
    "dark_scattering_fastpm": dict(
        cosmology_model="DarkScattering",
        cosmology_ds_w=-0.9,
        cosmology_ds_xi=50.0,
        cosmology_ds_cs2=1.0,
        simulation_use_cola=False,
        timestep_method="FastPM",
    ),
    # Dark scattering with CDM and baryons as separate particles: the CDM has the friction, the baryons none (their
    # own kick, drift and velocity scale; with COLA the correction of the frame's LPT velocity)
    "dark_scattering_baryons_cola": dict(
        cosmology_model="DarkScattering",
        cosmology_ds_w=-0.9,
        cosmology_ds_xi=50.0,
        cosmology_ds_cs2=1.0,
        particle_baryons=True,
        particle_baryon_lattice="same",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/dark_scattering.txt",
        ic_input_redshift=20.0,
    ),
    "dark_scattering_baryons_fastpm": dict(
        cosmology_model="DarkScattering",
        cosmology_ds_w=-0.9,
        cosmology_ds_xi=50.0,
        cosmology_ds_cs2=1.0,
        particle_baryons=True,
        particle_baryon_lattice="same",
        simulation_use_cola=False,
        timestep_method="FastPM",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/dark_scattering.txt",
        ic_input_redshift=20.0,
    ),
    # The covariantised dark Thomson-like scattering (the drag alpha a^3 / (Omega_c E)): one fluid with COLA, and CDM
    # and baryon particles with FastPM
    "covariant_dark_scattering_cola": dict(
        cosmology_model="CovariantDarkScattering",
        cosmology_ds_w=-0.98,
        cosmology_cds_alpha=1.0,
        cosmology_ds_cs2=1.0,
    ),
    "covariant_dark_scattering_baryons_fastpm": dict(
        cosmology_model="CovariantDarkScattering",
        cosmology_ds_w=-0.98,
        cosmology_cds_alpha=1.0,
        cosmology_ds_cs2=1.0,
        particle_baryons=True,
        particle_baryon_lattice="same",
        simulation_use_cola=False,
        timestep_method="FastPM",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/covariant_dark_scattering.txt",
        ic_input_redshift=20.0,
    ),
    # Coupled quintessence: the CDM+baryon particles as one fluid (G_eff, friction and the CDM mass of the coupling),
    # and with CDM and baryon particles (the fifth force of the CDM on the CDM, the species' own steps)
    "coupled_quintessence_cola": dict(
        cosmology_model="CoupledQuintessence",
        gravity_model="CoupledQuintessence",
        cosmology_cq_lambda=0.5,
        cosmology_cq_beta=0.15,
    ),
    "coupled_quintessence_baryons_cola": dict(
        cosmology_model="CoupledQuintessence",
        gravity_model="CoupledQuintessence",
        cosmology_cq_lambda=0.5,
        cosmology_cq_beta=0.15,
        particle_baryons=True,
        particle_baryon_lattice="same",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/coupled_quintessence.txt",
        ic_input_redshift=20.0,
    ),
    "coupled_quintessence_baryons_fastpm": dict(
        cosmology_model="CoupledQuintessence",
        gravity_model="CoupledQuintessence",
        cosmology_cq_lambda=0.5,
        cosmology_cq_beta=0.15,
        particle_baryons=True,
        particle_baryon_lattice="same",
        simulation_use_cola=False,
        timestep_method="FastPM",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/coupled_quintessence.txt",
        ic_input_redshift=20.0,
    ),
    # Interacting dark energy: the CDM+baryon particles as one fluid with their own masses (the energy exchange with the
    # change of their volume; the mass growth in the growth factors and the COLA frame), and CDM particles with
    # masses and baryon particles (no COLA, Quinn)
    "ide_cola": dict(
        cosmology_model="IDE",
        cosmology_ide_w=-0.9,
        cosmology_ide_xi=-0.1,
        cosmology_ide_cs2=1.0,
    ),
    "ide_baryons_quinn": dict(
        cosmology_model="IDE",
        cosmology_ide_w=-0.9,
        cosmology_ide_xi=-0.1,
        cosmology_ide_cs2=1.0,
        particle_baryons=True,
        particle_baryon_lattice="same",
        simulation_use_cola=False,
        timestep_method="Quinn",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/ide.txt",
        ic_input_redshift=20.0,
    ),
    # The cubic Galileon (its tracker background and the force of the DGP form with r_c(a), beta(a)): the screening
    # approximation with COLA, and the exact field equation (DGP's multigrid solver)
    "cubic_galileon_cola": dict(
        cosmology_model="CubicGalileon",
        gravity_model="CubicGalileon",
        gravity_model_cubic_galileon_exact_solution=False,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
    ),
    "cubic_galileon_exact_multigrid": dict(
        cosmology_model="CubicGalileon",
        gravity_model="CubicGalileon",
        gravity_model_cubic_galileon_exact_solution=True,
        gravity_model_screening=True,
        gravity_model_screening_efficiency=1.0,
        gravity_model_screening_enforce_largescale_linear=False,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-7,
        multigrid_nsweeps=5,
        multigrid_nsweeps_first_step=10,
        multigrid_coarse_factor=1,
    ),
    # CDM and baryons as separate particles (same lattice), BullFrog, IC from the transfer functions at zini
    "two_species_bullfrog": dict(
        particle_baryons=True,
        particle_baryon_lattice="same",
        simulation_use_cola=False,
        timestep_method="BullFrog",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.0.txt",
        ic_input_redshift=20.0,
        pofk_subtract_shotnoise=True,
    ),
    # CDM and baryons (offset lattices) with linear massive neutrinos and HDF5 output (needs USE_HDF5)
    "two_species_neutrinos_hdf5": dict(
        particle_baryons=True,
        particle_baryon_lattice="offset",
        simulation_use_cola=False,
        cosmology_OmegaMNu=0.0048,
        cosmology_OmegaCDM=0.2637,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.2.txt",
        ic_input_redshift=20.0,
        output_fileformat="HDF5",
    ),
    # CDM and baryons as separate particles with COLA (shared COLA frame of the CDM+baryon field), Quinn steps
    "two_species_cola": dict(
        particle_baryons=True,
        simulation_use_cola=True,
        timestep_method="Quinn",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.0.txt",
        ic_input_redshift=20.0,
    ),
    # CDM and baryons with scale-dependent COLA and linear massive neutrinos
    "two_species_scaledep_cola_nu": dict(
        particle_baryons=True,
        particle_baryon_lattice="offset",
        simulation_use_cola=True,
        simulation_use_scaledependent_cola=True,
        timestep_method="Quinn",
        cosmology_OmegaMNu=0.0095,
        cosmology_OmegaCDM=0.259,
        force_linear_massive_neutrinos=True,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.4.txt",
        ic_input_redshift=20.0,
    ),
    # FoF halos with CDM and baryons as separate particles (linked together), HDF5 halo catalogue (needs USE_HDF5)
    "fof_two_species_hdf5": dict(
        particle_baryons=True,
        simulation_use_cola=False,
        timestep_method="Quinn",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.0.txt",
        ic_input_redshift=20.0,
        fof=True,
        fof_nmin_per_halo=10,
        fof_output_fileformat="HDF5",
        fof_so_definitions=["200m", "500c", "vir"],
        fof_buffer_length_mpch=6.0,
    ),
    # CDM and baryons with the mass-encoded relative mode (both follow the CDM+baryon flow, masses 1 + eps(q)) and
    # COLA, FoF and HDF5 output with per-particle masses (needs USE_HDF5)
    "two_species_mass_cola_fof_hdf5": dict(
        particle_baryons=True,
        particle_baryon_relative_mode="mass",
        simulation_use_cola=True,
        timestep_method="Quinn",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.0.txt",
        ic_input_redshift=20.0,
        output_fileformat="HDF5",
        fof=True,
        fof_nmin_per_halo=10,
        fof_so_definitions=["200m"],
        fof_buffer_length_mpch=6.0,
    ),
    # CDM and baryons with common displacements, masses at zini and their own velocities (mass_ini), GADGET output
    # with a MASS block
    "two_species_mass_ini": dict(
        particle_baryons=True,
        particle_baryon_relative_mode="mass_ini",
        simulation_use_cola=False,
        timestep_method="Quinn",
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.0.txt",
        ic_input_redshift=20.0,
    ),
    # Phenomenological modified gravity mu(a) = 1 + mu0 Omega_DE(a)/Omega_DE (no screening), COLA
    "mu_de_cola": dict(
        gravity_model="mu",
        gravity_model_mu_parametrization="DE",
        gravity_model_mu_mu0=0.5,
    ),
    # Scale-dependent mu(a,k) (Bertschinger-Zukin) with scale-dependent COLA and a 2x force grid (offset lattice)
    "mu_bz_scaledep_cola": dict(
        gravity_model="mu",
        gravity_model_mu_parametrization="BZ",
        gravity_model_mu_BZ_params=[4.0 / 3.0, 10.0, 4.0],
        simulation_use_scaledependent_cola=True,
        force_nmesh=64,
    ),
    # mu(a,k) from a table, without COLA
    "mu_table": dict(
        gravity_model="mu",
        gravity_model_mu_parametrization="table",
        gravity_model_mu_filename=INPUT + "/mu_table_BZ_example.txt",
        simulation_use_cola=False,
        timestep_method="FastPM",
    ),
    # FoF with spherical overdensity masses, text catalogue, COLA
    "fof_so": dict(
        output_redshifts=[1.0, 0.0],
        fof=True,
        fof_nmin_per_halo=10,
        fof_so_definitions=["200m", "200c"],
        fof_buffer_length_mpch=6.0,
    ),
    # Massive neutrinos as particles (delta-f) with 0.4 eV, IC from the transfer functions at zini
    "neutrino_particles": dict(
        particle_neutrinos=True,
        simulation_use_cola=False,
        timestep_method="Quinn",
        cosmology_OmegaMNu=0.0095,
        cosmology_OmegaCDM=0.259,
        force_linear_massive_neutrinos=False,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.4.txt",
        ic_input_redshift=20.0,
    ),
    # Light neutrinos (0.1 eV in total) as particles with the energy density in the Poisson equation
    "neutrino_particles_energy_density": dict(
        particle_neutrinos=True,
        force_neutrino_energy_density=True,
        simulation_use_cola=False,
        timestep_method="Quinn",
        cosmology_OmegaMNu=0.0024,
        cosmology_OmegaCDM=0.2661,
        force_linear_massive_neutrinos=False,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.1.txt",
        ic_input_redshift=20.0,
    ),
    # CDM, baryons and (fewer) neutrino particles without delta-f weights, HDF5 output (needs USE_HDF5)
    "neutrino_particles_baryons_hdf5": dict(
        particle_neutrinos=True,
        particle_neutrino_Npart_1D=16,
        particle_neutrino_deltaf=False,
        particle_baryons=True,
        simulation_use_cola=False,
        timestep_method="Quinn",
        cosmology_OmegaMNu=0.0024,
        cosmology_OmegaCDM=0.2661,
        force_linear_massive_neutrinos=False,
        ic_type_of_input="tables",
        ic_input_filename=INPUT + "/transfer_tables/lcdm_nu0.1.txt",
        ic_input_redshift=20.0,
        output_fileformat="HDF5",
    ),
    # The same as 'bullfrog_lightcone' with the lightcone particles in HDF5 (needs USE_HDF5)
    "hdf5_lightcone": dict(
        simulation_use_cola=False,
        timestep_method="BullFrog",
        lightcone=True,
        plc_pos_observer=[0.0, 0.0, 0.0],
        plc_boundary_mpch=20.0,
        plc_z_init=0.05,
        plc_z_finish=0.0,
        plc_use_replicas=False,
        plc_ndim_rep=3,
        plc_output_gadgetfile=False,
        plc_output_asciifile=True,
        plc_output_hdf5file=True,
        plc_output_in_batches=True,
        plc_make_onion_density_maps=False,
        output_redshifts=[1.0, 0.0],
        timestep_nsteps=[3, 3],
    ),
    # The potential, radial velocity and observed redshift of the lightcone particles (HDF5, needs USE_HDF5)
    "hdf5_lightcone_relativistic": dict(
        simulation_use_cola=False,
        timestep_method="BullFrog",
        lightcone=True,
        plc_pos_observer=[0.0, 0.0, 0.0],
        plc_boundary_mpch=20.0,
        plc_z_init=0.05,
        plc_z_finish=0.0,
        plc_use_replicas=False,
        plc_ndim_rep=3,
        plc_output_gadgetfile=False,
        plc_output_asciifile=True,
        plc_output_hdf5file=True,
        plc_output_in_batches=True,
        plc_make_onion_density_maps=False,
        output_redshifts=[1.0, 0.0],
        timestep_nsteps=[3, 3],
        plc_relativistic_effects=True,
    ),
    # BullFrog from the Big Bang with the steps uniform in the growth factor
    "bullfrog_growthfactor_spacing": dict(
        simulation_use_cola=False,
        timestep_method="BullFrog",
        timestep_scalefactor_spacing="growthfactor",
        ic_initial_redshift=1e7,
        ic_LPT_order=1,
        force_interlacing_order=2,
        force_density_assignment_method="TSC",
        force_deconvolve_window=True,
    ),
    # The same as 'cola_quinn_all' but with GADGET HDF5 output (needs USE_HDF5)
    "hdf5_output": dict(
        output_redshifts=[20.0, 1.0, 0.0],
        timestep_nsteps=[0, 3, 3],
        ic_sigma8_normalization=True,
        output_fileformat="HDF5",
    ),
    # Read the HDF5 particles written at z=20 by 'hdf5_output' (same as 'read_particles_exact')
    "read_particles_hdf5": dict(
        ic_random_field_type="read_particles",
        ic_reconstruct_gadgetfilepath=IC_GADGET_HDF5,
        ic_reconstruct_assigment_method="PCS",
        ic_reconstruct_interlacing=True,
        ic_reconstruct_smoothing_filter="sharpk",
        ic_reconstruct_dimless_smoothing_scale=0.01,
        ic_reconstruct_exact=True,
    ),
    # Read particles written at z=20 by 'cola_quinn_all' and reconstruct LPT fields exactly
    "read_particles_exact": dict(
        ic_random_field_type="read_particles",
        ic_reconstruct_gadgetfilepath=IC_GADGET,
        ic_reconstruct_assigment_method="PCS",
        ic_reconstruct_interlacing=True,
        ic_reconstruct_smoothing_filter="sharpk",
        ic_reconstruct_dimless_smoothing_scale=0.01,
        ic_reconstruct_exact=True,
    ),
    # Read particles and reconstruct LPT fields from the density field
    "read_particles_grid": dict(
        ic_random_field_type="read_particles",
        ic_reconstruct_gadgetfilepath=IC_GADGET,
        ic_reconstruct_assigment_method="PCS",
        ic_reconstruct_interlacing=True,
        ic_reconstruct_smoothing_filter="sharpk",
        ic_reconstruct_dimless_smoothing_scale=0.01,
        ic_reconstruct_exact=False,
    ),
    # Take the phases from the particles written by 'cola_quinn_all'
    "read_phases": dict(
        ic_random_field_type="read_phases",
        ic_reconstruct_gadgetfilepath=IC_GADGET,
        ic_reconstruct_assigment_method="PCS",
        ic_reconstruct_interlacing=True,
        ic_reconstruct_smoothing_filter="sharpk",
        ic_reconstruct_dimless_smoothing_scale=0.01,
        ic_reconstruct_exact=False,
    ),
}

# Cases that fail with the current code and are not run by default (run them by name)
KNOWN_FAILING = []

# Cases that need an optional feature (shown by ./nbody --features). They are only run
# if the executable has it
REQUIRES = {
    "hdf5_output": "hdf5",
    "hdf5_lightcone": "hdf5",
    "hdf5_lightcone_relativistic": "hdf5",
    "two_species_neutrinos_hdf5": "hdf5",
    "read_particles_hdf5": "hdf5",
    "neutrino_particles_baryons_hdf5": "hdf5",
    "fof_two_species_hdf5": "hdf5",
    "two_species_mass_cola_fof_hdf5": "hdf5",
}

# The order the cases must run in (the read_* cases need 'cola_quinn_all' / 'hdf5_output')
ORDER = [c for c in CASES.keys() if c not in KNOWN_FAILING]


def lua_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple)):
        return "{" + ", ".join(lua_value(x) for x in v) + "}"
    return repr(v)


def write_case(outdir, name):
    params = dict(BASE)
    params.update(CASES[name])
    # The IC-only axion parameters (read with ic_type_of_input = internal or tables; all parameters must be in the file)
    if params.get("ic_type_of_input") in ("internal", "internal_transfer", "tables"):
        params.setdefault("ic_axion_fraction", 0.0)
        params.setdefault("ic_axion_log10_m", -22.0)
        params.setdefault("ic_axion_switch_mH", 0.0)
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        f.write(f"-- Regression case '{name}' (generated by cases.py)\n")
        for k, v in params.items():
            f.write(f"{k} = {lua_value(v)}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        # --list [feature ...]: the cases that can run with these features
        features = sys.argv[2:]
        print(" ".join(c for c in ORDER if REQUIRES.get(c, None) in [None] + features))
        sys.exit(0)
    outdir = sys.argv[1]
    for name in CASES:
        write_case(outdir, name)
