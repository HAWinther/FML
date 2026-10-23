#ifndef PARAMETERSCHEMA_HEADER
#define PARAMETERSCHEMA_HEADER

#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/ParameterMap/ParameterMap.h>

#include "Models.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

//=============================================================================
// All the parameters of the COLASolver in one table: name, type, default value,
// when it is used, allowed values and a description. This table is used to
//  - read the parameterfile (read_parameterfile)
//  - check string parameters against the allowed values
//  - warn about names in the parameterfile that are not parameters (typos)
//  - generate the parameter reference (./nbody --parameters, PARAMETERS.md)
//
// A parameter is only read if its condition holds (e.g. the f(R) parameters only
// if gravity_model = "f(R)"). Conditions can only depend on parameters earlier in
// the table. Optional parameters take the default value if not in the file,
// unless all_parameters_must_be_in_file = true
//=============================================================================

namespace COLASOLVER {

    using ParameterMap = FML::UTILS::ParameterMap;
    using ParameterValue = FML::UTILS::ParameterTypes;
    using Condition = std::function<bool(const ParameterMap &)>;

    enum class Need {
        Required,      // Must be in the parameterfile
        Optional,      // Takes the default value if not in the file (unless all_parameters_must_be_in_file)
        AlwaysOptional // Takes the default value if not in the file (experimental options)
    };

    struct ParameterSpec {
        std::string name;
        ParameterValue default_value; // Also defines the type of the parameter
        Need need;
        std::string description;
        std::vector<std::string> choices; // The allowed values (string parameters only, empty = anything)
        std::string section;
        Condition when;        // Only read if this holds (empty = always)
        std::string when_text; // ...in words
    };

    //=============================================================================
    /// Helper for writing the table
    //=============================================================================
    class SchemaBuilder {
      public:
        void section(const std::string & name) { current_section = name; }

        /// Parameters added inside body are only read if cond holds
        void when(Condition cond, const std::string & text, const std::function<void()> & body) {
            conditions.push_back({cond, text});
            body();
            conditions.pop_back();
        }

        template <class V>
        void required(const std::string & name,
                      V default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::Required, description, choices);
        }
        template <class V>
        void optional(const std::string & name,
                      V default_value,
                      const std::string & description,
                      std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::Optional, description, choices);
        }
        template <class V>
        void always_optional(const std::string & name, V default_value, const std::string & description) {
            add(name, to_value(default_value), Need::AlwaysOptional, description, {});
        }

        std::vector<ParameterSpec> specs;

      private:
        std::string current_section;
        std::vector<std::pair<Condition, std::string>> conditions;

        template <class V>
        static ParameterValue to_value(V v) {
            if constexpr (std::is_convertible_v<V, std::string> and not std::is_same_v<V, bool>)
                return ParameterValue(std::string(v));
            else
                return ParameterValue(v);
        }

        void add(const std::string & name,
                 ParameterValue default_value,
                 Need need,
                 const std::string & description,
                 const std::vector<std::string> & choices) {
            ParameterSpec spec{name, default_value, need, description, choices, current_section, {}, ""};
            if (not conditions.empty()) {
                auto conds = conditions;
                spec.when = [conds](const ParameterMap & p) {
                    for (auto & c : conds)
                        if (not c.first(p))
                            return false;
                    return true;
                };
                for (auto & c : conditions)
                    spec.when_text += (spec.when_text.empty() ? "" : " and ") + c.second;
            }
            specs.push_back(spec);
        }
    };

    // Conditions on earlier parameters
    inline Condition is_true(const std::string & name) {
        return [name](const ParameterMap & p) { return p.get<bool>(name); };
    }
    inline Condition is_one_of(const std::string & name, const std::vector<std::string> & values) {
        return [name, values](const ParameterMap & p) {
            return std::find(values.begin(), values.end(), p.get<std::string>(name)) != values.end();
        };
    }
    inline std::string one_of_text(const std::string & name, const std::vector<std::string> & values) {
        if (values.size() == 1)
            return name + " = " + values[0];
        std::string text = name + " in {";
        for (size_t i = 0; i < values.size(); i++)
            text += (i ? ", " : "") + values[i];
        return text + "}";
    }

    //=============================================================================
    /// The table of all parameters
    //=============================================================================
    template <int NDIM>
    const std::vector<ParameterSpec> & parameter_schema() {
        static const std::vector<ParameterSpec> schema = []() {
            SchemaBuilder b;
            const std::vector<std::string> assignment_methods{"NGP", "CIC", "TSC", "PCS", "PQS"};
            const std::vector<std::string> kernels{"fiducial", "continuous", "discrete_2pt", "discrete_4pt"};
            const std::vector<std::string> smoothing_filters{"tophat", "sharpk", "gaussian"};
            auto gravity_is = [&](std::vector<std::string> models, std::function<void()> body) {
                b.when(is_one_of("gravity_model", models), one_of_text("gravity_model", models), body);
            };
            auto if_true = [&](const std::string & flag, std::function<void()> body) {
                b.when(is_true(flag), flag + " = true", body);
            };

            b.section("General");
            b.optional("all_parameters_must_be_in_file",
                       false,
                       "If true every parameter that is used must be in the parameterfile (no default values are "
                       "used)");

            b.section("Simulation");
            b.required("simulation_boxsize", 0.0, "Boxsize of the simulation (Mpc/h)");
            b.required("simulation_name", "", "Label used in the output filenames");
            b.optional("simulation_use_cola", false, "Use the COLA method");
            b.optional("simulation_use_scaledependent_cola",
                       true,
                       "If the gravity model has scale-dependent growth: use scale-dependent COLA. If false we use "
                       "the k = 0 limit of the growth factors when doing COLA");
            if_true("simulation_use_cola", [&]() {
                b.optional("simulation_enforce_LPT_trajectories",
                           false,
                           "Do not compute the PM force, just let particles follow the LPT trajectories");
                b.optional("simulation_cola_LPT_order",
                           2,
                           "The LPT order of COLA (1, 2 or 3). Has to be <= ic_LPT_order. If < 1 COLA is turned off");
            });

            b.section("Cosmology");
            b.optional("cosmology_model",
                       "LCDM",
                       "The cosmology (background evolution)",
                       model_names(cosmology_models()));
            b.required("cosmology_Omegab", 0.0, "Baryon density parameter");
            b.optional("cosmology_OmegaMNu", 0.0, "Massive neutrino density parameter");
            b.required("cosmology_OmegaCDM", 0.0, "CDM density parameter");
            b.required("cosmology_OmegaK", 0.0, "Curvature density parameter (-k/H0^2)");
            b.required("cosmology_Neffective", 3.046, "Effective number of relativistic species");
            b.required("cosmology_TCMB_kelvin", 2.7255, "Temperature of the CMB today (Kelvin)");
            b.required("cosmology_h", 0.0, "Hubble parameter H0 = 100h km/s/Mpc");
            b.when(is_one_of("cosmology_model", {"JBD"}), "cosmology_model = JBD", [&]() {
                b.required("cosmology_JBD_wBD", 10000.0, "The JBD parameter (wBD -> infinity gives GR)");
                b.optional("cosmology_JBD_GeffG_today",
                           1.0,
                           "The value of G today relative to Newtons constant (GeffG_today != 1 effectively gives a "
                           "different Newtons constant)");
                b.optional("cosmology_JBD_density_parameter_definition",
                           "hi-class",
                           "Definition of the density parameters Omega = 8pi G_* rho / 3H0^2 (which G_* to use)",
                           {"Gbare", "Gnewton", "Gtoday", "hi-class"});
            });
            b.when(is_one_of("cosmology_model", {"w0waCDM"}), "cosmology_model = w0waCDM", [&]() {
                b.required("cosmology_w0", -1.0, "Dark energy equation of state w(a) = w0 + wa(1-a): w0");
                b.required("cosmology_wa", 0.0, "Dark energy equation of state w(a) = w0 + wa(1-a): wa");
            });
            b.when(is_one_of("cosmology_model", {"DGP"}), "cosmology_model = DGP", [&]() {
                b.required("cosmology_dgp_OmegaRC",
                           0.0,
                           "The cross-over scale of the self-accelerating DGP model "
                           "(Omega_rc = 1/(4 rc^2 H0^2))");
            });

            b.section("Gravity model");
            b.optional("gravity_model",
                       "GR",
                       "The gravity model (growth of perturbations and forces)",
                       model_names(gravity_models<NDIM>()));
            gravity_is({"Geff"}, [&]() {
                b.required("gravity_model_geff_geffofa_filename", "", "File with the format [a, Geff/G(a)]");
            });
            gravity_is({"mbeta"}, [&]() {
                b.required("gravity_model_mbeta_params",
                           std::vector<double>{},
                           "The parameters (beta0, n, m0, m) of beta(a) = beta0 a^n and m(a) = m0 H0 a^m");
            });
            gravity_is({"f(R)"}, [&]() {
                b.required("gravity_model_fofr_fofr0", 1e-5, "The value of f_R today");
                b.optional("gravity_model_fofr_nfofr", 1.0, "The index n of the Hu-Sawicky model");
            });
            gravity_is({"Symmetron"}, [&]() {
                b.optional("gravity_model_symmetron_assb",
                           0.5,
                           "Symmetry breaking scale factor (no fifth-force for a < assb)");
                b.optional("gravity_model_symmetron_beta", 1.0, "Coupling strength relative to gravity");
                b.optional("gravity_model_symmetron_L_mpch",
                           1.0,
                           "Range of the fifth force in the background today "
                           "(Mpc/h)");
            });
            gravity_is({"DGP"},
                       [&]() { b.required("gravity_model_dgp_rcH0overc", 1.0, "The cross-over scale rc H0 / c"); });
            gravity_is({"mbeta", "f(R)", "Symmetron", "DGP"}, [&]() {
                b.optional("gravity_model_screening", true, "Use the approximate screening model (otherwise linear)");
                if_true("gravity_model_screening", [&]() {
                    gravity_is({"DGP"}, [&]() {
                        b.optional("gravity_model_dgp_smoothing_filter",
                                   "tophat",
                                   "Screening approximation: the filter used to smooth the density field",
                                   smoothing_filters);
                        b.optional("gravity_model_dgp_smoothing_scale_over_boxsize",
                                   1.0,
                                   "Screening approximation: the smoothing scale R/boxsize");
                    });
                    b.optional("gravity_model_screening_enforce_largescale_linear",
                               false,
                               "Combine the screened solution with the linear solution to get the correct linear "
                               "evolution on large scales");
                    b.optional("gravity_model_screening_linear_scale_hmpc",
                               0.05,
                               "Use the linear solution for k < k* and the screened solution for k > k* (h/Mpc)");
                    gravity_is({"mbeta", "f(R)"}, [&]() {
                        b.optional("gravity_model_screening_efficiency",
                                   1.0,
                                   "Screening efficiency (1 is standard). Increases or decreases the amount of "
                                   "screening");
                    });
                });
            });
            gravity_is({"f(R)"}, [&]() {
                b.optional("gravity_model_fofr_exact_solution",
                           false,
                           "Solve the exact f(R) equation with the multigrid solver (slow, mainly for testing)");
            });
            gravity_is({"Symmetron"}, [&]() {
                b.optional("gravity_model_symmetron_exact_solution",
                           false,
                           "Solve the exact symmetron equation with the multigrid solver (slow, mainly for testing)");
            });
            gravity_is({"DGP"}, [&]() {
                b.optional("gravity_model_dgp_exact_solution",
                           false,
                           "Solve the exact DGP equation with the multigrid solver (slow, mainly for testing; can "
                           "be hard to get to converge)");
            });
            auto exact_solution = [](const ParameterMap & p) {
                auto model = p.get<std::string>("gravity_model");
                return (model == "f(R)" and p.get<bool>("gravity_model_fofr_exact_solution")) or
                       (model == "Symmetron" and p.get<bool>("gravity_model_symmetron_exact_solution")) or
                       (model == "DGP" and p.get<bool>("gravity_model_dgp_exact_solution"));
            };
            b.when(exact_solution, "solving the exact equation of the gravity model", [&]() {
                b.optional("multigrid_nsweeps", 10, "Multigrid solver: Newton-Gauss-Seidel sweeps on every level");
                b.optional("multigrid_nsweeps_first_step",
                           20,
                           "Multigrid solver: sweeps on the first step (increase if the solver fails to converge)");
                b.optional("multigrid_solver_residual_convergence",
                           1e-6,
                           "Multigrid solver: converged when the residual is below this");
            });

            b.section("Primordial power-spectrum");
            b.optional("cosmology_As", 2e-9, "Primordial amplitude");
            b.optional("cosmology_ns", 1.0, "Spectral index");
            b.optional("cosmology_kpivot_mpc", 0.05, "Pivot scale (1/Mpc)");

            b.section("Particles");
            b.required("particle_Npart_1D", 0, "Number of CDM+baryon particles per dimension");
            b.optional("particle_allocation_factor",
                       1.5,
                       "Allocate this factor more particles than the average per task (particles move between "
                       "tasks)");

            b.section("Time-stepping");
            b.required("timestep_nsteps",
                       std::vector<int>{0},
                       "Number of steps between the outputs (one per output redshift). If only one number it is the "
                       "total number of steps");
            b.optional("timestep_algorithm", "KDK", "The time-stepping algorithm");
            b.optional("timestep_method",
                       "Quinn",
                       "The time-stepping method. Quinn: standard leapfrog. Tassev: modified leapfrog for COLA. "
                       "FastPM: leapfrog that follows linear growth exactly (Feng et al. 2016). BullFrog: follows "
                       "2LPT growth after every step (Rampf, List & Hahn 2024), not for COLA",
                       {"Quinn", "Tassev", "FastPM", "BullFrog"});
            b.optional("timestep_cola_nLPT", -2.5, "The nLPT parameter of the Tassev time-stepping");
            b.optional("timestep_scalefactor_spacing",
                       "linear",
                       "Spacing of the steps in the scale factor (powerlaw: uniform in a^timestep_spacing_power)",
                       {"linear", "logarithmic", "powerlaw"});
            b.when(is_one_of("timestep_scalefactor_spacing", {"powerlaw"}),
                   "timestep_scalefactor_spacing = powerlaw",
                   [&]() { b.optional("timestep_spacing_power", 1.0, "The steps are uniform in a^power"); });

            b.section("Initial conditions");
            b.required("ic_initial_redshift", 0.0, "The initial redshift of the simulation");
            b.required("ic_random_seed", 1234, "The random seed");
            b.optional("ic_random_generator",
                       "MT19937",
                       "The random number generator (GSL is gsl_rng_ranlxd1 as used in 2LPTic, for comparisons)",
                       {"GSL", "MT19937"});
            b.required("ic_nmesh", 0, "The grid size used to generate the IC");
            b.required("ic_type_of_input",
                       "powerspectrum",
                       "The input: powerspectrum (file with [k (h/Mpc), P(k) (Mpc/h)^3]), transferfunction (file "
                       "with [k (h/Mpc), T(k) Mpc^2]) or transferinfofile (file listing T(k,z) files from CAMB/CLASS)",
                       {"powerspectrum", "transferfunction", "transferinfofile"});
            b.optional("ic_type_of_input_fileformat",
                       "CAMB",
                       "Format of the transfer function files (CLASS run with format=camb)",
                       {"CAMB", "CLASS", "AXIONCAMB"});
            b.required("ic_input_filename", "", "The input file");
            b.required("ic_input_redshift", 0.0, "The redshift of the input P(k) / T(k)");
            b.optional("ic_fix_amplitude", true, "Fix the amplitude of the modes when generating the random field");
            b.optional("ic_reverse_phases", false, "Reverse the phases (for paired simulations)");
            b.optional("ic_random_field_type",
                       "gaussian",
                       "Type of IC: gaussian, nongaussian, read_particles (read a GADGET file and use that, "
                       "reconstructing the LPT fields if COLA) or read_phases (take the phases from a GADGET file)",
                       {"gaussian", "nongaussian", "read_particles", "read_phases"});
            b.optional("ic_LPT_order", 2, "The LPT order used to make the IC");
            b.optional("ic_sigma8_normalization", false, "Normalize P(k) to a given sigma8 (otherwise use the input)");
            b.optional("ic_sigma8_redshift", 0.0, "The redshift of the sigma8 value");
            b.optional("ic_sigma8", 0.8, "The sigma8 value to normalize to");
            b.optional("ic_use_gravity_model_GR",
                       false,
                       "For modified gravity: the input P(k) is for LCDM, use GR growth factors to scale it to the "
                       "initial redshift (same IC as the LCDM simulation)");
            b.when(is_one_of("ic_random_field_type", {"nongaussian"}), "ic_random_field_type = nongaussian", [&]() {
                b.required("ic_fnl_type", "local", "Type of non-gaussianity", {"local", "equilateral", "orthogonal"});
                b.required("ic_fnl", 100.0, "The value of fNL");
                b.required("ic_fnl_redshift", 0.0, "The redshift at which the non-gaussian potential is applied");
            });
            b.when(is_one_of("ic_random_field_type", {"read_particles", "read_phases"}),
                   "ic_random_field_type in {read_particles, read_phases}",
                   [&]() {
                       b.required("ic_reconstruct_gadgetfilepath",
                                  "",
                                  "Path to the GADGET files without the file number: binary (path.0, path.1, ...) "
                                  "or HDF5 (path.hdf5 or path.0.hdf5, ...), detected automatically");
                       b.optional("ic_reconstruct_assigment_method",
                                  "CIC",
                                  "Density assignment method used to reconstruct the density field",
                                  assignment_methods);
                       b.optional("ic_reconstruct_smoothing_filter",
                                  "sharpk",
                                  "Filter used to remove small-scale modes from the reconstructed density field",
                                  smoothing_filters);
                       b.optional("ic_reconstruct_dimless_smoothing_scale",
                                  0.0,
                                  "Smoothing scale R/boxsize of the reconstructed density field");
                       b.optional("ic_reconstruct_interlacing",
                                  false,
                                  "Use interlacing when reconstructing the density field");
                       b.optional("ic_reconstruct_exact",
                                  false,
                                  "Reconstruct the LPT fields exactly from the positions and velocities (always works "
                                  "for 1LPT IC; for 2LPT IC the particles must come from a regular grid)");
                   });

            b.section("Forces");
            b.required("force_nmesh", 0, "Grid size used to compute the PM forces");
            b.optional("force_density_assignment_method", "CIC", "Density assignment method", assignment_methods);
            b.optional("force_greens_function_kernel",
                       "fiducial",
                       "Kernel for D^2 when solving the Poisson equation (fiducial = continuous)",
                       kernels);
            b.optional("force_gradient_kernel",
                       "fiducial",
                       "Kernel for D when computing the forces. fiducial = continuous if force_nmesh = "
                       "particle_Npart_1D and discrete_2pt if force_nmesh is larger (the continuous kernel gives "
                       "spurious forces from the particle lattice when the force grid is finer than it)",
                       kernels);
            b.optional("force_linear_massive_neutrinos",
                       false,
                       "Include massive neutrinos in the density field using linear theory (requires "
                       "ic_type_of_input = transferinfofile)");
            b.always_optional("force_nbody_gauge",
                              false,
                              "Run in the N-body gauge: add the linear source from radiation, neutrinos and GR "
                              "(made with CLASS by scripts/make_nbody_gauge_transfer.py) to the density field");
            b.when(is_true("force_nbody_gauge"), "force_nbody_gauge = true", [&]() {
                b.required("force_nbody_gauge_filename",
                           "",
                           "The table T_extra(k,z) (nbgauge_extra.txt from scripts/make_nbody_gauge_transfer.py)");
            });
            b.always_optional("force_use_finite_difference_force",
                              false,
                              "Experimental: compute forces with finite differences of the potential");
            b.always_optional("force_finite_difference_stencil_order",
                              4,
                              "Experimental: order of the finite difference stencil");

            b.section("Lightcone");
            b.optional("lightcone", false, "Make a lightcone");
            if_true("lightcone", [&]() {
                b.optional("plc_pos_observer",
                           std::vector<double>{0.0, 0.0, 0.0},
                           "Position of the observer in units of the boxsize");
                b.optional("plc_z_init", 0.0, "The redshift we start recording the lightcone");
                b.optional("plc_z_finish", 0.0, "The redshift we stop recording the lightcone");
                b.optional("plc_boundary_mpch",
                           20.0,
                           "Boundary region around the shell to make sure we get all particles (Mpc/h)");
                b.optional("plc_use_replicas", false, "Replicate the box to get the sky coverage we want");
                b.optional("plc_ndim_rep",
                           0,
                           "Number of dimensions with replicas in both directions (3D: 0 = octant, 3 = full sky)");
                b.optional("plc_output_gadgetfile", false, "Output the lightcone particles in GADGET format");
                b.optional("plc_output_asciifile", false, "Output the lightcone particles in ascii");
                b.optional("plc_output_in_batches", false, "Output in batches to save memory");
                b.optional("plc_make_onion_density_maps", false, "Make (Healpix) density maps delta(z, theta)");
                if_true("plc_make_onion_density_maps", [&]() {
                    b.optional("plc_nside", 256, "Healpix nside of the maps (npix = 12 nside^2)");
                    b.optional("plc_da_maps", 0.02, "Roughly the size of the bins in a for the maps");
                    b.optional("plc_use_chunkpix", false, "Store the maps with chunkpix (useful for sparse maps)");
                    if_true("plc_use_chunkpix",
                            [&]() { b.optional("plc_nside_chunks", 8, "Healpix nside of the chunks"); });
                });
            });

            b.section("Output");
            b.required("output_folder", "", "Folder to put the output in (\"\" is the current folder)");
            b.required("output_redshifts", std::vector<double>{}, "The output redshifts (decreasing)");
            b.optional("output_particles", true, "Write the particles at the output redshifts");
            b.optional("output_fileformat",
                       "GADGET",
                       "File format for the particles. GADGET: binary GADGET files. HDF5: GADGET HDF5 files "
                       "(compile with USE_HDF5=true). FML: raw dump of the particles (can be read back by FML)",
                       {"GADGET", "HDF5", "FML"});

            b.section("Halo finding");
            b.optional("fof", false, "Locate friends-of-friends halos at every output");
            if_true("fof", [&]() {
                b.optional("fof_nmin_per_halo", 20, "Minimum number of particles per halo");
                b.optional("fof_linking_length", 0.2, "The linking length in units of the mean particle separation");
                b.optional("fof_nmesh_max",
                           0,
                           "Maximum grid size used to bin particles to speed up the linking (0: the code chooses)");
                b.optional("fof_buffer_length_mpch",
                           3.0,
                           "Buffer region (larger than the largest halo) from neighbor tasks (Mpc/h)");
            });

            b.section("Power-spectrum");
            b.optional("pofk", false, "Compute the power-spectrum at every output");
            if_true("pofk", [&]() {
                b.optional("pofk_nmesh", 0, "Grid size");
                b.optional("pofk_interlacing", false, "Use interlaced grids to reduce aliasing");
                b.optional("pofk_subtract_shotnoise", false, "Subtract shot-noise");
                b.optional("pofk_density_assignment_method", "CIC", "Density assignment method", assignment_methods);
            });

            b.section("Power-spectrum multipoles");
            b.optional("pofk_multipole", false, "Compute the redshift space multipoles P_ell(k) at every output");
            if_true("pofk_multipole", [&]() {
                b.required("pofk_multipole_nmesh", 0, "Grid size");
                b.optional("pofk_multipole_interlacing", false, "Use interlaced grids to reduce aliasing");
                b.optional("pofk_multipole_subtract_shotnoise", false, "Subtract shot-noise from P0");
                b.optional("pofk_multipole_ellmax", 4, "Compute P_ell for ell = 0, 2, ..., ellmax");
                b.optional("pofk_multipole_density_assignment_method",
                           "CIC",
                           "Density assignment method",
                           assignment_methods);
            });

            b.section("Bispectrum");
            b.optional("bispectrum", false, "Compute the bispectrum at every output");
            if_true("bispectrum", [&]() {
                b.required("bispectrum_nmesh", 0, "Grid size");
                b.required("bispectrum_nbins",
                           0,
                           "Number of k-bins (memory and time scale as nbins grids and nbins^3 integrals)");
                b.optional("bispectrum_density_assignment_method",
                           "CIC",
                           "Density assignment method",
                           assignment_methods);
                b.optional("bispectrum_interlacing", true, "Use interlaced grids to reduce aliasing");
                b.optional("bispectrum_subtract_shotnoise", false, "Subtract shot-noise");
            });
            return b.specs;
        }();
        return schema;
    }

    //=============================================================================
    // Helpers
    //=============================================================================

    /// The value as a string (for printing)
    inline std::string value_to_string(const ParameterValue & value) {
        std::ostringstream os;
        std::visit(
            [&](auto && v) {
                using V = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<V, bool>) {
                    os << (v ? "true" : "false");
                } else if constexpr (std::is_same_v<V, std::string>) {
                    os << "\"" << v << "\"";
                } else if constexpr (std::is_same_v<V, int> or std::is_same_v<V, double>) {
                    os << v;
                } else {
                    os << "{";
                    for (size_t i = 0; i < v.size(); i++)
                        os << (i ? ", " : "") << v[i];
                    os << "}";
                }
            },
            value);
        return os.str();
    }

    inline std::string type_name(const ParameterValue & value) {
        static const char * names[] = {"bool", "int", "double", "string", "int list", "double list", "string list"};
        return names[value.index()];
    }

    /// Levenshtein distance (for suggesting the parameter that was meant)
    inline size_t edit_distance(const std::string & a, const std::string & b) {
        std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
        for (size_t j = 0; j <= b.size(); j++)
            prev[j] = j;
        for (size_t i = 1; i <= a.size(); i++) {
            cur[0] = i;
            for (size_t j = 1; j <= b.size(); j++)
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
            std::swap(prev, cur);
        }
        return prev[b.size()];
    }

    //=============================================================================
    /// Read the parameterfile (a Lua script) into param using the schema
    //=============================================================================
    template <int NDIM>
    void read_parameterfile(ParameterMap & param, const std::string & filename) {
        FML::FILEUTILS::LuaFileParser lfp(filename);
        const auto & schema = parameter_schema<NDIM>();

        bool all_required = false;
        for (auto & spec : schema) {
            if (spec.when and not spec.when(param))
                continue;
            const bool required = spec.need == Need::Required or (spec.need == Need::Optional and all_required);

            std::visit(
                [&](auto && default_value) {
                    using V = std::decay_t<decltype(default_value)>;
                    if constexpr (std::is_same_v<V, bool>)
                        param[spec.name] = lfp.read_bool(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, int>)
                        param[spec.name] = lfp.read_int(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, double>)
                        param[spec.name] = lfp.read_double(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, std::string>)
                        param[spec.name] = lfp.read_string(spec.name, default_value, required);
                    else if constexpr (std::is_same_v<V, std::vector<int>> or std::is_same_v<V, std::vector<double>>)
                        param[spec.name] =
                            lfp.read_number_array<typename V::value_type>(spec.name, default_value, required);
                    else
                        param[spec.name] = lfp.read_string_array(spec.name, default_value, required);
                },
                spec.default_value);

            if (spec.name == "all_parameters_must_be_in_file")
                all_required = param.get<bool>(spec.name);

            // Check that string parameters have one of the allowed values
            if (not spec.choices.empty()) {
                const auto value = param.get<std::string>(spec.name);
                if (std::find(spec.choices.begin(), spec.choices.end(), value) == spec.choices.end()) {
                    std::string valid;
                    for (auto & c : spec.choices)
                        valid += " " + c;
                    throw std::runtime_error("Parameterfile: unknown value [" + value + "] for " + spec.name +
                                             ". Valid options are:" + valid);
                }
            }
        }

        // Warn about names in the parameterfile that are not parameters (most likely typos)
        if (FML::ThisTask == 0) {
            std::set<std::string> known;
            for (auto & spec : schema)
                known.insert(spec.name);
            for (auto & name : lfp.get_global_variable_names()) {
                if (known.count(name))
                    continue;
                std::string closest;
                size_t best = 4;
                for (auto & k : known) {
                    auto d = edit_distance(name, k);
                    if (d < best) {
                        best = d;
                        closest = k;
                    }
                }
                std::cout << "WARNING: [" << name << "] in the parameterfile is not a parameter and is ignored";
                if (not closest.empty())
                    std::cout << " (did you mean [" << closest << "]?)";
                std::cout << "\n";
            }
        }
    }

    //=============================================================================
    /// Write the parameter reference (markdown)
    //=============================================================================
    template <int NDIM>
    void print_parameter_reference(std::ostream & os) {
        os << "# COLASolver parameters\n\n";
        os << "Generated with `./nbody --parameters PARAMETERS.md` from `src/ParameterSchema.h`.\n\n";
        os << "Parameters marked *optional* take the default value if they are not in the parameterfile, unless "
              "`all_parameters_must_be_in_file = true`. A parameter is only used if the condition in the *Used when* "
              "column holds.\n";
        std::string section;
        for (auto & spec : parameter_schema<NDIM>()) {
            if (spec.section != section) {
                section = spec.section;
                os << "\n## " << section << "\n\n";
                os << "| Parameter | Type | Default | Used when | Description |\n";
                os << "|---|---|---|---|---|\n";
            }
            std::string description = spec.description;
            if (not spec.choices.empty()) {
                description += ". Options:";
                for (size_t i = 0; i < spec.choices.size(); i++)
                    description += (i ? ", " : " ") + std::string("`") + spec.choices[i] + "`";
            }
            std::string default_text =
                spec.need == Need::Required ? "*required*" : "`" + value_to_string(spec.default_value) + "`";
            os << "| `" << spec.name << "` | " << type_name(spec.default_value) << " | " << default_text << " | "
               << (spec.when_text.empty() ? "always" : spec.when_text) << " | " << description << " |\n";
        }
    }

} // namespace COLASOLVER

#endif
