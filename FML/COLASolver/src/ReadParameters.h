#ifndef READPARAMETERS_HEADER
#define READPARAMETERS_HEADER

#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/ParameterMap/ParameterMap.h>

using ParameterMap = FML::UTILS::ParameterMap;
using LuaFileParser = FML::FILEUTILS::LuaFileParser;

void read_parameterfile(ParameterMap & param, std::string filename) {
    LuaFileParser lfp(filename);

    bool OPTIONAL = lfp.optional;
    bool REQUIRED = lfp.required;

    //=============================================================
    // Don't allow optional parameters?
    //=============================================================
    param["all_parameters_must_be_in_file"] = lfp.read_bool("all_parameters_must_be_in_file", false, OPTIONAL);
    if (param.get<bool>("all_parameters_must_be_in_file"))
        OPTIONAL = REQUIRED;

    //=============================================================
    // Simulation options
    //=============================================================
    param["simulation_boxsize"] = lfp.read_double("simulation_boxsize", 0.0, REQUIRED);
    param["simulation_name"] = lfp.read_string("simulation_name", "", REQUIRED);

    //=============================================================
    // COLA simulation
    //=============================================================
    param["simulation_use_cola"] = lfp.read_bool("simulation_use_cola", false, OPTIONAL);
    param["simulation_use_scaledependent_cola"] = lfp.read_bool("simulation_use_scaledependent_cola", true, OPTIONAL);

    //=============================================================
    // Cosmology options
    //=============================================================
    param["cosmology_model"] = lfp.read_string("cosmology_model", "LCDM", OPTIONAL);
    param["cosmology_Omegab"] = lfp.read_double("cosmology_Omegab", 0.0, REQUIRED);
    param["cosmology_OmegaMNu"] = lfp.read_double("cosmology_OmegaMNu", 0.0, OPTIONAL);
    param["cosmology_OmegaCDM"] = lfp.read_double("cosmology_OmegaCDM", 0.0, REQUIRED);
    param["cosmology_OmegaLambda"] = lfp.read_double("cosmology_OmegaLambda", 0.0, REQUIRED);
    param["cosmology_Neffective"] = lfp.read_double("cosmology_Neffective", 3.046, REQUIRED);
    param["cosmology_TCMB_kelvin"] = lfp.read_double("cosmology_TCMB_kelvin", 2.7255, REQUIRED);
    param["cosmology_h"] = lfp.read_double("cosmology_h", 0.0, REQUIRED);

    //=============================================================
    // Jordan-Brans-Dicke model
    //=============================================================
    if (param.get<std::string>("cosmology_model") == "JBD") {
        param["cosmology_JBD_wBD"] = lfp.read_double("cosmology_JBD_wBD", 10000.0, REQUIRED);
        param["cosmology_JBD_OmegaLambdah2"] = lfp.read_double("cosmology_JBD_OmegaLambdah2", 0.0, REQUIRED);
        param["cosmology_JBD_OmegaCDMh2"] = lfp.read_double("cosmology_JBD_OmegaCDMh2", 0.0, REQUIRED);
        param["cosmology_JBD_OmegaKh2"] = lfp.read_double("cosmology_JBD_OmegaKh2", 0.0, REQUIRED);
        param["cosmology_JBD_OmegaMNuh2"] = lfp.read_double("cosmology_JBD_OmegaMNuh2", 0.0, REQUIRED);
        param["cosmology_JBD_Omegabh2"] = lfp.read_double("cosmology_JBD_Omegabh2", 0.0, REQUIRED);
        param["cosmology_JBD_GeffG_today"] = lfp.read_double("cosmology_JBD_GeffG_today", 1.0, OPTIONAL);
    }

    //=============================================================
    // w0-wa Dark energy model
    //=============================================================
    if (param.get<std::string>("cosmology_model") == "w0waCDM") {
        param["cosmology_w0"] = lfp.read_double("cosmology_w0", -1.0, REQUIRED);
        param["cosmology_wa"] = lfp.read_double("cosmology_wa", 0.0, REQUIRED);
    }

    //=============================================================
    // DGP self-accelerating model
    //=============================================================
    if (param.get<std::string>("cosmology_model") == "DGP") {
        param["cosmology_dgp_OmegaRC"] = lfp.read_double("cosmology_dgp_OmegaRC", 0.0, REQUIRED);
    }

    //=============================================================
    // Gravity model
    //=============================================================
    param["gravity_model"] = lfp.read_string("gravity_model", "GR", OPTIONAL);

    if (param.get<std::string>("gravity_model") != "GR") {

        //=============================================================
        // f(R) model
        //=============================================================
        if (param.get<std::string>("gravity_model") == "f(R)") {
            param["gravity_model_fofr_fofr0"] = lfp.read_double("gravity_model_fofr_fofr0", 1e-5, REQUIRED);
            param["gravity_model_fofr_nfofr"] = lfp.read_double("gravity_model_fofr_nfofr", 1.0, OPTIONAL);

            // Screening approximation
            param["gravity_model_screening"] = lfp.read_bool("gravity_model_screening", true, OPTIONAL);
            if (param.get<bool>("gravity_model_screening")) {
                param["gravity_model_screening_enforce_largescale_linear"] =
                    lfp.read_bool("gravity_model_screening_enforce_largescale_linear", false, OPTIONAL);
                param["gravity_model_screening_linear_scale_hmpc"] =
                    lfp.read_double("gravity_model_screening_linear_scale_hmpc", 0.05, OPTIONAL);
            }

            // Solving the exact equation
            param["gravity_model_fofr_exact_solution"] =
                lfp.read_bool("gravity_model_fofr_exact_solution", false, OPTIONAL);
            if (param.get<bool>("gravity_model_fofr_exact_solution")) {
                param["multigrid_nsweeps"] = lfp.read_int("multigrid_nsweeps", 10, OPTIONAL);
                param["multigrid_nsweeps_first_step"] = lfp.read_int("multigrid_nsweeps_first_step", 20, OPTIONAL);
                param["multigrid_solver_residual_convergence"] =
                    lfp.read_double("multigrid_solver_residual_convergence", 1e-6, OPTIONAL);
            }
        }

        //=============================================================
        // Symmetron model
        //=============================================================

        if (param.get<std::string>("gravity_model") == "Symmetron") {
            param["gravity_model_symmetron_assb"] = lfp.read_double("gravity_model_symmetron_assb", 0.5, OPTIONAL);
            param["gravity_model_symmetron_beta"] = lfp.read_double("gravity_model_symmetron_beta", 1.0, OPTIONAL);
            param["gravity_model_symmetron_L_mpch"] = lfp.read_double("gravity_model_symmetron_L_mpch", 1.0, OPTIONAL);
            
            // Screening approximation
            param["gravity_model_screening"] = lfp.read_bool("gravity_model_screening", true, OPTIONAL);
            if (param.get<bool>("gravity_model_screening")) {
                param["gravity_model_screening_enforce_largescale_linear"] =
                    lfp.read_bool("gravity_model_screening_enforce_largescale_linear", false, OPTIONAL);
                param["gravity_model_screening_linear_scale_hmpc"] =
                    lfp.read_double("gravity_model_screening_linear_scale_hmpc", 0.05, OPTIONAL);
            }

            // Solving the exact equation
            param["gravity_model_symmetron_exact_solution"] =
                lfp.read_bool("gravity_model_symmetron_exact_solution", false, OPTIONAL);
            if (param.get<bool>("gravity_model_symmetron_exact_solution")) {
                param["multigrid_nsweeps"] = lfp.read_int("multigrid_nsweeps", 10, OPTIONAL);
                param["multigrid_nsweeps_first_step"] = lfp.read_int("multigrid_nsweeps_first_step", 20, OPTIONAL);
                param["multigrid_solver_residual_convergence"] =
                    lfp.read_double("multigrid_solver_residual_convergence", 1e-6, OPTIONAL);
            }
        }

        //=============================================================
        // DGP model
        //=============================================================
        if (param.get<std::string>("gravity_model") == "DGP") {
            param["gravity_model_dgp_rcH0overc"] = lfp.read_double("gravity_model_dgp_rcH0overc", 1.0, REQUIRED);

            // Screening approximation
            param["gravity_model_screening"] = lfp.read_bool("gravity_model_screening", true, OPTIONAL);
            if (param.get<bool>("gravity_model_screening")) {
                param["gravity_model_dgp_smoothing_filter"] =
                    lfp.read_string("gravity_model_dgp_smoothing_filter", "tophat", OPTIONAL);
                param["gravity_model_dgp_smoothing_scale_over_boxsize"] =
                    lfp.read_double("gravity_model_dgp_smoothing_scale_over_boxsize", 1.0, OPTIONAL);
                param["gravity_model_screening_enforce_largescale_linear"] =
                    lfp.read_bool("gravity_model_screening_enforce_largescale_linear", false, OPTIONAL);
                param["gravity_model_screening_linear_scale_hmpc"] =
                    lfp.read_double("gravity_model_screening_linear_scale_hmpc", 0.05, OPTIONAL);
            }
        }
    }

    //=============================================================
    // Primordial power-spectrum
    //=============================================================
    param["cosmology_As"] = lfp.read_double("cosmology_As", 2e-9, OPTIONAL);
    param["cosmology_ns"] = lfp.read_double("cosmology_ns", 1.0, OPTIONAL);
    param["cosmology_kpivot_mpc"] = lfp.read_double("cosmology_kpivot_mpc", 0.05, OPTIONAL);

    //=============================================================
    // Particles
    //=============================================================
    param["particle_Npart_1D"] = lfp.read_int("particle_Npart_1D", 0, REQUIRED);
    param["particle_allocation_factor"] = lfp.read_double("particle_allocation_factor", 1.5, OPTIONAL);

    //=============================================================
    // Time-stepping
    //=============================================================
    param["timestep_nsteps"] = lfp.read_number_array<int>("timestep_nsteps", {0}, REQUIRED);
    param["timestep_algorithm"] = lfp.read_string("timestep_algorithm", "KDK", OPTIONAL);
    param["timestep_method"] = lfp.read_string("timestep_method", "Quinn", OPTIONAL);
    param["timestep_cola_nLPT"] = lfp.read_double("timestep_cola_nLPT", -2.5, OPTIONAL);
    param["timestep_scalefactor_spacing"] = lfp.read_string("timestep_scalefactor_spacing", "linear", OPTIONAL);
    if (param.get<std::string>("timestep_scalefactor_spacing") == "powerlaw") {
        param["timestep_spacing_power"] = lfp.read_double("timestep_spacing_power", 1.0, OPTIONAL);
    }

    //=============================================================
    // Initial conditions
    //=============================================================
    param["ic_initial_redshift"] = lfp.read_double("ic_initial_redshift", 0.0, REQUIRED);
    param["ic_random_seed"] = lfp.read_int("ic_random_seed", 1234, REQUIRED);
    param["ic_random_generator"] = lfp.read_string("ic_random_generator", "MT19937", OPTIONAL);
    param["ic_nmesh"] = lfp.read_int("ic_nmesh", 0, REQUIRED);
    param["ic_type_of_input"] = lfp.read_string("ic_type_of_input", "powerspectrum", REQUIRED);
    param["ic_input_filename"] = lfp.read_string("ic_input_filename", "", REQUIRED);
    param["ic_input_redshift"] = lfp.read_double("ic_input_redshift", 0.0, REQUIRED);
    param["ic_fix_amplitude"] = lfp.read_bool("ic_fix_amplitude", true, OPTIONAL);
    param["ic_reverse_phases"] = lfp.read_bool("ic_reverse_phases", false, OPTIONAL);
    param["ic_random_field_type"] = lfp.read_string("ic_random_field_type", "gaussian", OPTIONAL);
    param["ic_LPT_order"] = lfp.read_int("ic_LPT_order", 2, OPTIONAL);
    param["ic_sigma8_normalization"] = lfp.read_bool("ic_sigma8_normalization", false, OPTIONAL);
    param["ic_sigma8_redshift"] = lfp.read_double("ic_sigma8_redshift", 0.0, OPTIONAL);
    param["ic_sigma8"] = lfp.read_double("ic_sigma8", 0.8, OPTIONAL);
    param["ic_use_gravity_model_GR"] = lfp.read_bool("ic_use_gravity_model_GR", false, OPTIONAL);

    if (param.get<std::string>("ic_random_field_type") != "gaussian") {
        //=============================================================
        // Non-gaussianity
        //=============================================================
        param["ic_fnl_type"] = lfp.read_string("ic_fnl_type", "local", REQUIRED);
        param["ic_fnl"] = lfp.read_double("ic_fnl", 100.0, REQUIRED);
        param["ic_fnl_redshift"] = lfp.read_double("ic_fnl_redshift", 0.0, REQUIRED);
    }

    if (param.get<std::string>("ic_random_field_type") == "reconstruct_from_particles") {
        //=============================================================
        // Reconstruct initial density field from particles and use
        // this to generate the IC (useful for COLA where we need
        // the displacement fields)
        //=============================================================
        param["ic_reconstruct_gadgetfilepath"] = lfp.read_string("ic_reconstruct_gadgetfilepath", "", REQUIRED);
        param["ic_reconstruct_assigment_method"] = lfp.read_string("ic_reconstruct_assigment_method", "CIC", OPTIONAL);
        param["ic_reconstruct_smoothing_filter"] =
            lfp.read_string("ic_reconstruct_smoothing_filter", "sharpk", OPTIONAL);
        param["ic_reconstruct_dimless_smoothing_scale"] =
            lfp.read_double("ic_reconstruct_dimless_smoothing_scale", 0.0, OPTIONAL);
        param["ic_reconstruct_interlacing"] = lfp.read_bool("ic_reconstruct_interlacing", false, OPTIONAL);
    }

    //=============================================================
    // Force calculation
    //=============================================================
    param["force_nmesh"] = lfp.read_int("force_nmesh", 0, REQUIRED);
    param["force_density_assignment_method"] = lfp.read_string("force_density_assignment_method", "CIC", OPTIONAL);
    param["force_kernel"] = lfp.read_string("force_kernel", "continuous_greens_function", OPTIONAL);
    param["force_linear_massive_neutrinos"] = lfp.read_bool("force_linear_massive_neutrinos", false, OPTIONAL);

    //=============================================================
    // Output
    //=============================================================
    param["output_folder"] = lfp.read_string("output_folder", "", REQUIRED);
    param["output_redshifts"] = lfp.read_number_array<double>("output_redshifts", {}, REQUIRED);
    param["output_particles"] = lfp.read_bool("output_particles", true, OPTIONAL);
    param["output_fileformat"] = lfp.read_string("output_fileformat", "GADGET", OPTIONAL);

    //=============================================================
    // Halofinding
    //=============================================================
    param["fof"] = lfp.read_bool("fof", false, OPTIONAL);
    if (param.get<bool>("fof")) {
        param["fof_nmin_per_halo"] = lfp.read_int("fof_nmin_per_halo", 20, OPTIONAL);
        param["fof_linking_length"] = lfp.read_double("fof_linking_length", 0.2, OPTIONAL);
        param["fof_nmesh_max"] = lfp.read_int("fof_nmesh_max", 0, OPTIONAL);
    }

    //=============================================================
    // Powerspectrum
    //=============================================================
    param["pofk"] = lfp.read_bool("pofk", false, OPTIONAL);
    if (param.get<bool>("pofk")) {
        param["pofk_nmesh"] = lfp.read_int("pofk_nmesh", 0, OPTIONAL);
        param["pofk_interlacing"] = lfp.read_bool("pofk_interlacing", false, OPTIONAL);
        param["pofk_subtract_shotnoise"] = lfp.read_bool("pofk_subtract_shotnoise", false, OPTIONAL);
        param["pofk_density_assignment_method"] = lfp.read_string("pofk_density_assignment_method", "CIC", OPTIONAL);
    }

    //=============================================================
    // Powerspectrum multipoles
    //=============================================================
    param["pofk_multipole"] = lfp.read_bool("pofk_multipole", false, OPTIONAL);
    if (param.get<bool>("pofk_multipole")) {
        param["pofk_multipole_nmesh"] = lfp.read_int("pofk_multipole_nmesh", 0, REQUIRED);
        param["pofk_multipole_interlacing"] = lfp.read_bool("pofk_multipole_interlacing", false, OPTIONAL);
        param["pofk_multipole_subtract_shotnoise"] =
            lfp.read_bool("pofk_multipole_subtract_shotnoise", false, OPTIONAL);
        param["pofk_multipole_ellmax"] = lfp.read_int("pofk_multipole_ellmax", 4, OPTIONAL);
        param["pofk_multipole_density_assignment_method"] =
            lfp.read_string("pofk_multipole_density_assignment_method", "CIC", OPTIONAL);
    }

    //=============================================================
    // Bispectrum
    //=============================================================
    param["bispectrum"] = lfp.read_bool("bispectrum", false, OPTIONAL);
    if (param.get<bool>("bispectrum")) {
        param["bispectrum_nmesh"] = lfp.read_int("bispectrum_nmesh", 0, REQUIRED);
        param["bispectrum_nbins"] = lfp.read_int("bispectrum_nbins", 0, REQUIRED);
        param["bispectrum_density_assignment_method"] =
            lfp.read_string("bispectrum_density_assignment_method", "CIC", OPTIONAL);
        param["bispectrum_interlacing"] = lfp.read_bool("bispectrum_interlacing", true, OPTIONAL);
        param["bispectrum_subtract_shotnoise"] = lfp.read_bool("bispectrum_subtract_shotnoise", false, OPTIONAL);
    }
}
#endif
