#!/usr/bin/env python3
#==========================================================================
# Regression test cases for the COLASolver
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

# The COLASolver input folder (power spectra, transfer functions)
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
    ic_type_of_input_fileformat="CAMB",
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
        timestep_nsteps=[8],
        output_redshifts=[1.0, 0.5, 0.0],
        ic_fix_amplitude=True,
        ic_reverse_phases=True,
        ic_type_of_input="transferfunction",
        ic_input_filename=INPUT + "/example_transfer_function_cb_z0.000.txt",
        ic_use_gravity_model_GR=True,
        ic_random_generator="MT19937",
        output_fileformat="FML",
    ),
    # Scale-dependent COLA with massive neutrinos from transfer info file
    "scaledep_cola_neutrinos": dict(
        simulation_use_scaledependent_cola=True,
        cosmology_OmegaMNu=0.0048,
        cosmology_OmegaCDM=0.2637,
        ic_type_of_input="transferinfofile",
        ic_input_filename=INPUT + "/transfer_infofile_lcdm_nu0.2.txt",
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
        simulation_cola_LPT_order=1,
        force_density_assignment_method="TSC",
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
        gravity_model_screening_enforce_largescale_linear=True,
        gravity_model_screening_linear_scale_hmpc=0.1,
        multigrid_solver_residual_convergence=1e-4,
        multigrid_nsweeps=2,
        multigrid_nsweeps_first_step=2,
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

# The order the cases must run in (the read_* cases need 'cola_quinn_all')
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
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)

    # A transfer info file starts with the folder holding the transfer files. Write a copy
    # pointing to the folder in this repository so the case runs on any machine
    if params["ic_type_of_input"] == "transferinfofile":
        with open(params["ic_input_filename"]) as f:
            lines = f.read().split("\n")
        folder = os.path.basename(lines[0].split()[0].rstrip("/"))
        lines[0] = os.path.join(INPUT, folder) + " " + " ".join(lines[0].split()[1:])
        infofile = os.path.join(d, "transfer_infofile.txt")
        with open(infofile, "w") as f:
            f.write("\n".join(lines))
        params["ic_input_filename"] = infofile
    with open(os.path.join(d, "parameterfile.lua"), "w") as f:
        f.write(f"-- Regression case '{name}' (generated by cases.py)\n")
        for k, v in params.items():
            f.write(f"{k} = {lua_value(v)}\n")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print(" ".join(ORDER))
        sys.exit(0)
    outdir = sys.argv[1]
    for name in CASES:
        write_case(outdir, name)
