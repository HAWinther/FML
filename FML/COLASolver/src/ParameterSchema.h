#ifndef PARAMETERSCHEMA_HEADER
#define PARAMETERSCHEMA_HEADER

#include <FML/Global/Global.h>
#include <FML/LuaFileParser/LuaFileParser.h>
#include <FML/ParameterMap/ParameterMap.h>

#include "Models.h"
#include "ParameterDefaults.h"
#include "SimulationParameters.h"

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
        void always_optional(const std::string & name,
                             V default_value,
                             const std::string & description,
                             std::vector<std::string> choices = {}) {
            add(name, to_value(default_value), Need::AlwaysOptional, description, choices);
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
                           "The LPT order of COLA (1 or 2). Has to be <= ic_LPT_order. If < 1 COLA is turned off");
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
            b.when(is_one_of("cosmology_model", {"Quintessence"}), "cosmology_model = Quintessence", [&]() {
                b.required("cosmology_quintessence_potential",
                           "exponential",
                           "The potential V = V0 u(phi), phi in units of Mpl: exponential u = exp(-lambda phi), "
                           "inverse_power_law u = phi^-alpha, pngb u = 1 + cos(phi/f), hilltop u = (1 - (phi/mu)^p)^2. "
                           "V0 is found so that OmegaDE today = 1 - OmegaM - OmegaR - OmegaK",
                           {"exponential", "inverse_power_law", "pngb", "hilltop"});
                b.optional("cosmology_quintessence_phi_ini",
                           0.0,
                           "The initial value of the field (units of Mpl; deep in the radiation era)");
                b.optional("cosmology_quintessence_dphidlna_ini", 0.0, "The initial dphi/dlna (0: frozen)");
                b.when(is_one_of("cosmology_quintessence_potential", {"exponential"}),
                       "cosmology_quintessence_potential = exponential",
                       [&]() { b.required("cosmology_quintessence_lambda", 1.0, "The slope lambda"); });
                b.when(is_one_of("cosmology_quintessence_potential", {"inverse_power_law"}),
                       "cosmology_quintessence_potential = inverse_power_law",
                       [&]() { b.required("cosmology_quintessence_alpha", 1.0, "The power alpha"); });
                b.when(is_one_of("cosmology_quintessence_potential", {"pngb"}),
                       "cosmology_quintessence_potential = pngb",
                       [&]() { b.required("cosmology_quintessence_f", 1.0, "The scale f (units of Mpl)"); });
                b.when(
                    is_one_of("cosmology_quintessence_potential", {"hilltop"}),
                    "cosmology_quintessence_potential = hilltop",
                    [&]() {
                        b.required("cosmology_quintessence_mu", 1.0, "The position of the minimum mu (units of Mpl)");
                        b.required("cosmology_quintessence_p", 2.0, "The power p");
                    });
            });
            b.when(is_one_of("cosmology_model", {"EDE"}), "cosmology_model = EDE", [&]() {
                b.required("cosmology_ede_fEDE", 0.1, "Early dark energy: the maximum of rho_EDE / rho_total");
                b.required("cosmology_ede_zc", 3500.0, "Early dark energy: the redshift of the maximum");
                b.optional("cosmology_ede_theta_ini", 2.8, "Early dark energy: the initial angle phi_ini / f");
                b.optional("cosmology_ede_n", 3.0, "Early dark energy: V = m^2 f^2 (1 - cos(phi/f))^n");
            });
            b.when(is_one_of("cosmology_model", {"Tabulated"}), "cosmology_model = Tabulated", [&]() {
                b.required("cosmology_tabulated_filename", std::string(""), "The file with the background");
                b.required("cosmology_tabulated_format",
                           "H",
                           "H: columns z, H/H0. CLASS: a CLASS background file. w: columns a, w(a) of the dark energy",
                           {"H", "CLASS", "w"});
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
            gravity_is({"mu"}, [&]() {
                b.required("gravity_model_mu_parametrization",
                           "DE",
                           "Poisson equation with G_eff/G = mu(a,k), no screening. DE: mu = 1 + mu0 "
                           "Omega_DE(a)/Omega_DE(a=1). BZ: mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + lambda1^2 k^2 "
                           "a^s) (Bertschinger & Zukin 2008). table: mu(a) or mu(a,k) from a file",
                           {"DE", "BZ", "table"});
                b.when(is_one_of("gravity_model_mu_parametrization", {"DE"}),
                       "gravity_model_mu_parametrization = DE",
                       [&]() {
                           b.required("gravity_model_mu_mu0", 0.0, "mu0 in mu = 1 + mu0 Omega_DE(a)/Omega_DE(a=1)");
                       });
                b.when(is_one_of("gravity_model_mu_parametrization", {"BZ"}),
                       "gravity_model_mu_parametrization = BZ",
                       [&]() {
                           b.required("gravity_model_mu_BZ_params",
                                      std::vector<double>{1.0, 0.0, 4.0},
                                      "{beta1, lambda1 (Mpc/h), s} in mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + "
                                      "lambda1^2 k^2 a^s) (k in h/Mpc)");
                       });
                b.when(is_one_of("gravity_model_mu_parametrization", {"table"}),
                       "gravity_model_mu_parametrization = table",
                       [&]() {
                           b.required("gravity_model_mu_filename",
                                      "",
                                      "File with rows 'a mu' or 'a k mu' (k in h/Mpc, a regular grid in a and k; "
                                      "e.g. from hi_class). Splined in log a, log k; the closest value outside");
                       });
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
                                   "Screening approximation: the smoothing scale R/boxsize (the default 1 smooths over "
                                   "the whole "
                                   "box, which makes the screening very weak; see GravityModel_DGP.h)");
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
                b.when(is_true("gravity_model_fofr_exact_solution"), "gravity_model_fofr_exact_solution = true", [&]() {
                    b.always_optional("gravity_model_fofr_multigrid_method",
                                      "sqrt",
                                      "The field the multigrid solver solves for. sqrt: u = sqrt(|f_R|) with the "
                                      "equation in a cell solved exactly (a cubic for n = 1), robust for any f_R0, "
                                      "n and redshift. log: ln|f_R| with Newton steps (the original method; can give "
                                      "NaN for small f_R0 or at high redshift)",
                                      {"sqrt", "log"});
                });
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
            b.always_optional("particle_baryons",
                              false,
                              "Evolve CDM and baryons as two sets of particles (particle_Npart_1D^3 of each, IC from "
                              "the CDM and baryon transfer functions at zini). false: one set of CDM+baryon particles");
            b.always_optional("particle_neutrinos",
                              false,
                              "Massive neutrinos as particles with the delta-f method (Elbers et al. 2021): thermal "
                              "Fermi-Dirac velocities, relativistic equations of motion, the particles only carry the "
                              "perturbation. Replaces force_linear_massive_neutrinos (needs cosmology_OmegaMNu > 0, "
                              "transferinfofile, timestep_method = Quinn, no COLA)");
            b.always_optional("particle_neutrino_Npart_1D",
                              0,
                              "particle_neutrinos: number of neutrino particles per dimension (0: particle_Npart_1D)");
            b.always_optional("particle_neutrino_deltaf",
                              true,
                              "particle_neutrinos: use the delta-f weights (false: plain particles, much noisier; for "
                              "testing)");
            b.always_optional(
                "particle_baryon_lattice",
                "same",
                "particle_baryons: start the baryons from the same lattice as the CDM (same, recommended: "
                "the baryon-CDM difference then follows linear theory on all scales) or from a lattice "
                "offset by half a particle spacing (offset: gives a spurious baryon-CDM difference on "
                "small scales)",
                {"same", "offset"});
            b.always_optional(
                "particle_baryon_relative_mode",
                "displacement",
                "particle_baryons: how the baryon-CDM relative density is set up. displacement: each "
                "species is displaced by its own linear field (T_i/T_cb) and has its own velocity. mass: "
                "both species follow the total CDM+baryon flow (same displacements and velocities) and the "
                "relative density is carried by the particle masses m_i = 1 + eps_i(q) with eps_i = "
                "delta_cb(zini) (T_i - T_cb)(z=0) / T_cb(zini), the linear relative mode today "
                "(Hahn, Rampf & Uhlemann 2021; exact at z = 0 in linear theory, the relative mode is then "
                "constant in time). mass_ini: common displacements, masses with eps_i = T_i / T_cb - 1 at "
                "zini and each species keeps its own velocity (the relative mode then grows from zini). "
                "The mass modes need particle_baryon_lattice = same",
                {"displacement", "mass", "mass_ini"});
            b.optional("particle_allocation_factor",
                       1.5,
                       "Allocate this factor more particles than the average per task (particles move between "
                       "tasks)");
            b.always_optional("particle_sort_every_nsteps",
                              0,
                              "Reorder the particles in memory by position (blocks of 4^3 force-grid cells) every this "
                              "many steps, which makes the density assignment and the force interpolation faster when "
                              "the particles have moved far from their initial (lattice) order. 0: never. Changes the "
                              "results only by round-off (the order of the sums) and the order of the particles in "
                              "the output");

            b.section("Time-stepping");
            b.required("timestep_nsteps",
                       std::vector<int>{0},
                       "Number of steps between the outputs (one per output redshift). If only one number it is the "
                       "total number of steps");
            b.optional("timestep_algorithm", "KDK", "The time-stepping algorithm (only KDK is implemented)", {"KDK"});
            b.optional("timestep_method",
                       "Quinn",
                       "The time-stepping method. Quinn: standard leapfrog. Tassev: modified leapfrog for COLA. "
                       "FastPM: leapfrog that follows linear growth exactly (Feng et al. 2016). BullFrog: follows "
                       "2LPT growth after every step (Rampf, List & Hahn 2024), not for COLA",
                       {"Quinn", "Tassev", "FastPM", "BullFrog"});
            b.optional("timestep_cola_nLPT", -2.5, "The nLPT parameter of the Tassev time-stepping");
            b.optional("timestep_scalefactor_spacing",
                       "linear",
                       "Spacing of the steps in the scale factor (powerlaw: uniform in a^timestep_spacing_power, "
                       "growthfactor: uniform in the linear growth factor D(a), as in Rampf, List & Hahn 2024)",
                       {"linear", "logarithmic", "powerlaw", "growthfactor"});
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
            b.always_optional("ic_random_field_backend",
                              "ngenic",
                              "Fourier mode sampling: ngenic preserves legacy line seeds and IC; philox uses "
                              "independent mode counters",
                              {"ngenic", "philox"});
            b.required("ic_nmesh", 0, "The grid size used to generate the IC");
            b.always_optional(
                "ic_lattice_offset",
                "fiducial",
                "Offset of the initial particle lattice from the nodes of the force grid. fiducial: half a "
                "force grid cell if force_nmesh > particle_Npart_1D, otherwise none. With particles "
                "exactly on the nodes of a finer grid, CIC responds to the size of the displacement and gives large "
                "spurious forces (see the force grid section of README.md)",
                {"fiducial", "none", "half_force_cell"});
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
            if_true("ic_sigma8_normalization", [&]() {
                b.optional("ic_sigma8_redshift", 0.0, "The redshift of the sigma8 value");
                b.optional("ic_sigma8", 0.8, "The sigma8 value to normalize to");
            });
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
            b.optional(
                "force_gradient_kernel",
                "fiducial",
                "Kernel for D when computing the forces. fiducial = continuous if force_nmesh = "
                "particle_Npart_1D, discrete_4pt if force_nmesh is up to 3x larger and discrete_2pt if it is even "
                "larger (the continuous kernel gives "
                "spurious forces from the particle lattice when the force grid is finer than it)",
                kernels);
            b.always_optional("force_density_method",
                              "assignment",
                              "How the density field for the force is computed. assignment: particles to grid with "
                              "force_density_assignment_method. nufft: the exact Fourier modes of the particle "
                              "distribution (non-uniform FFT; no window function or aliasing). sheet: spectral sheet "
                              "interpolation, R^NDIM tracers per particle placed by Fourier interpolating the "
                              "displacement field x - q, assigned with force_density_assignment_method. sheet_nufft: "
                              "sheet interpolation with the exact Fourier modes of the tracers (NUFFT). The force is "
                              "always interpolated back to the particles with force_density_assignment_method",
                              {"assignment", "nufft", "sheet", "sheet_nufft"});
            b.always_optional("force_interlacing_order",
                              1,
                              "Force interlacing: compute the force for this many copies of the particles shifted "
                              "by a fraction of a cell relative to the grid and average (at the particles). 1: off, "
                              "2: shift (1/2,1/2,1/2), 4: the half-cell face diagonals, 8: all half-cell corners, "
                              "64: quarter cells. Cancels the leading aliasing of the density assignment and the force "
                              "interpolation at the cost of this many force calculations");
            b.always_optional("force_deconvolve_window",
                              false,
                              "Divide the force by W(k)^2 where W is the window function of "
                              "force_density_assignment_method (corrects for the smoothing of the assignment and the "
                              "interpolation). Unstable unless the aliasing is suppressed (force_interlacing_order "
                              "> 1, see README)");
            b.always_optional("force_early_method",
                              "none",
                              "A more accurate force at early times (when the particles are still close to the "
                              "lattice and CIC is inaccurate): nufft (NUFFT density and interpolation) or interlaced "
                              "(force_early_interlacing_order shifted grids with force_early_density_assignment_method "
                              "and window deconvolution) for steps at z > force_early_until_redshift, then the "
                              "normal force settings. none: off",
                              {"none", "nufft", "interlaced"});
            b.when(
                is_one_of("force_early_method", {"nufft", "interlaced"}),
                one_of_text("force_early_method", {"nufft", "interlaced"}),
                [&]() { b.always_optional("force_early_until_redshift", 10.0, "Use the early force for z > this"); });
            b.when(is_one_of("force_early_method", {"interlaced"}), "force_early_method = interlaced", [&]() {
                b.always_optional("force_early_interlacing_order", 2, "Number of shifted grids at early times");
                b.always_optional("force_early_density_assignment_method",
                                  "TSC",
                                  "Assignment kernel at early times",
                                  assignment_methods);
            });
            b.always_optional("force_interpolation_method",
                              "fiducial",
                              "How the force is interpolated to the particles. assignment: with the "
                              "force_density_assignment_method kernel. nufft: exact evaluation of the Fourier series "
                              "of the force at the particles (non-uniform FFT). fiducial: nufft if "
                              "force_density_method = nufft (then assignment and interpolation are both exact, so "
                              "there is no self-force), otherwise assignment",
                              {"fiducial", "assignment", "nufft"});
            b.always_optional("force_nufft_epsilon",
                              1e-5,
                              "Relative accuracy of the NUFFT (force_density_method = nufft or sheet_nufft, or "
                              "force_interpolation_method = nufft)");
            b.when(is_one_of("force_density_method", {"sheet", "sheet_nufft"}),
                   one_of_text("force_density_method", {"sheet", "sheet_nufft"}),
                   [&]() {
                       b.always_optional("force_sheet_refinement",
                                         2,
                                         "R: the number of tracers per particle per dimension for sheet interpolation");
                   });
            b.optional("force_linear_massive_neutrinos",
                       false,
                       "Include massive neutrinos in the density field using linear theory (requires "
                       "ic_type_of_input = transferinfofile)");
            b.always_optional(
                "force_neutrino_energy_density",
                false,
                "Source the Poisson equation with the energy density of the massive neutrinos instead of "
                "their mass density: f_nu <E/m>(a) delta_nu with <E/m> = <sqrt(1 + (q/am)^2)> over the "
                "Fermi-Dirac distribution (both for force_linear_massive_neutrinos, where delta_nu from "
                "CAMB/CLASS is already the energy-density contrast, and for particle_neutrinos, where each "
                "particle is weighted by its E/m and the neutrino P(k) is then the energy-density "
                "contrast). Matters at high redshift for light neutrinos (<E/m> = 1.33 at z = 50 for "
                "0.033 eV)");
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
                b.always_optional("plc_output_hdf5file",
                                  false,
                                  "Output the lightcone particles as GADGET HDF5 files (needs USE_HDF5; positions in "
                                  "Mpc/h, peculiar velocities in km/s)");
                b.optional("plc_output_in_batches", false, "Output in batches to save memory");
                b.always_optional(
                    "plc_relativistic_effects",
                    false,
                    "Store the potential Phi/c^2 at the lightcone particles and write it with the radial "
                    "velocity, the cosmological redshift and the observed redshift 1 + z_obs = (1 + z)(1 + "
                    "v_r/c - Phi/c^2) (Doppler + gravitational redshift; observer at rest with Phi = 0) to "
                    "the HDF5 and ascii lightcone files. With plc_make_onion_density_maps also Healpix maps "
                    "of Phi/c^2 at the shells. Phi from the Poisson equation with the density used for the "
                    "force (the GR potential also for modified gravity models). Costs one extra FFT per "
                    "step while recording the lightcone");
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
                       "(compile with USE_HDF5=true)",
                       {"GADGET", "HDF5"});

            b.section("Baryonification (snapshot post-processing)");
            b.always_optional("baryonification",
                              false,
                              "Correct a temporary snapshot copy; 3D GR, no massive neutrinos, single CDM+baryon "
                              "species; needs pofk=true");
            if_true("baryonification", [&]() {
                BaryonificationSettings defaults;
                defaults.parameters([&](const char * name, auto &, const auto & value, const char * description) {
                    b.always_optional(std::string("baryonification_") + name, value, description);
                });
            });

            b.section("Halo finding");
            b.optional("fof", false, "Locate friends-of-friends halos at every output");
            if_true("fof", [&]() {
                b.optional("fof_nmin_per_halo",
                           20,
                           "Minimum number of particles per halo (of one species with particle_baryons)");
                b.optional("fof_linking_length",
                           0.2,
                           "The linking length in units of the mean particle separation (of one species with "
                           "particle_baryons)");
                b.optional("fof_nmesh_max",
                           0,
                           "Maximum grid size used to bin particles to speed up the linking (0: the code chooses)");
                b.optional(
                    "fof_buffer_length_mpch",
                    3.0,
                    "Buffer region (larger than the largest halo) from neighbor tasks (Mpc/h). The code warns if "
                    "halos reach its edge (they might then be cut off and found twice)");
            });
            b.always_optional("fof_output_fileformat",
                              "ascii",
                              "fof: the halo catalogue halos_z<z>.txt (ascii, one row per halo) or halos_z<z>.hdf5 "
                              "(HDF5, compile with USE_HDF5=true)",
                              {"ascii", "HDF5"});
            b.always_optional("fof_massfunction_nbins", 30, "fof: number of logarithmic bins of the mass function");
            b.always_optional("fof_massfunction_Mmin", 1e10, "fof: smallest mass of the mass function (Msun/h)");
            b.always_optional("fof_massfunction_Mmax", 1e16, "fof: largest mass of the mass function (Msun/h)");
            b.always_optional("fof_so_definitions",
                              std::vector<std::string>{},
                              "fof: spherical overdensity masses around the FoF halos, e.g. {\"200m\", \"500c\", "
                              "\"vir\"} (<Delta>m: Delta times the mean CDM+baryon density, <Delta>c: the critical "
                              "density, vir: Bryan & Norman 1998). Empty: none. The spheres must fit in "
                              "fof_buffer_length_mpch");

            b.section("Power-spectrum");
            b.optional("pofk", false, "Compute the power-spectrum at every output");
            b.always_optional("pofk_every_step",
                              true,
                              "Write P(k) of the density field used for the force at every step (cheap; same grid "
                              "and assignment as the force, the window is deconvolved): pofk_<name>_cb_z<z>.txt in "
                              "output_folder");
            b.when([](const ParameterMap & p) { return p.get<bool>("pofk") or p.get<bool>("pofk_every_step"); },
                   "pofk = true or pofk_every_step = true",
                   [&]() {
                       b.optional("pofk_subtract_shotnoise", false, "Subtract shot-noise (pofk and pofk_every_step)");
                   });
            if_true("pofk", [&]() {
                b.required("pofk_nmesh", 0, "Grid size");
                b.optional("pofk_interlacing", false, "Use interlaced grids to reduce aliasing");
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

            b.section("Hybrid bias spectra");
            b.always_optional("hybridbias",
                              false,
                              "Compute the hybrid bias (Lagrangian bias basis) spectra P_ab(k) at every output: the "
                              "cross spectra of the operators 1, delta_L, delta_L^2, s^2, nabla^2 delta_L advected "
                              "with the particles (Modi et al. 2020, Kokron et al. 2021). delta_L is the linear field "
                              "of the IC scaled to the output redshift with the growth factor of the gravity model");
            if_true("hybridbias", [&]() {
                b.required("hybridbias_nmesh", 0, "Grid size for the advected fields");
                b.optional("hybridbias_noperators",
                           5,
                           "Use the first noperators of (1, delta, delta^2, s^2, nabla^2 delta): 15 spectra for 5");
                b.optional("hybridbias_density_assignment_method",
                           "CIC",
                           "Density assignment method",
                           assignment_methods);
                b.optional("hybridbias_interlacing", true, "Use interlaced grids to reduce aliasing");
                b.optional("hybridbias_subtract_shotnoise",
                           false,
                           "Subtract the Poisson shot noise mean(w_a w_b)/Npart (the usual basis spectra keep it)");
                b.optional("hybridbias_smoothing_scale",
                           0.0,
                           "Smooth the linear field with this scale (Mpc/h) before forming the operators (0: none)");
                b.optional("hybridbias_smoothing_filter", "gaussian", "The smoothing filter", smoothing_filters);
                b.optional(
                    "hybridbias_kcut_over_nyquist",
                    1.0,
                    "Keep only the modes with k < this times the Nyquist frequency of the particle lattice in the "
                    "linear field (a spherical cut, so the operators are isotropic as assumed in the LPT "
                    "prediction). 0: keep all the modes of the lattice grid (a cube)");
                b.optional("hybridbias_zeldovich",
                           false,
                           "Also write the Zel'dovich (resummed 1LPT) prediction of the spectra with the same cut and "
                           "smoothing of the operators: hybridbias_zeldovich_z<z>.txt");
                b.optional("hybridbias_oneloop",
                           false,
                           "Also write the one-loop LPT (CLEFT) prediction of the spectra (no counterterms): "
                           "hybridbias_oneloop_z<z>.txt");
                b.always_optional(
                    "hybridbias_oneloop_kernels",
                    "EdS",
                    "The kernels of the one-loop prediction. EdS: the Einstein-de Sitter kernels with the "
                    "linear P(k) of the gravity model. model: the scale- and time-dependent LPT kernels "
                    "of the gravity model (f(R) and DGP with their screening if gravity_model_screening, "
                    "the other models from G_eff(k,a) without screening, LCDM for GR)",
                    {"EdS", "model"});
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
    /// Register the lookup of the default values in the table (ParameterDefaults.h: get_parameter)
    //=============================================================================
    inline const bool parameter_default_lookup_registered = []() {
        parameter_default_lookup() = [](const std::string & name) -> const ParameterValue * {
            for (auto & spec : parameter_schema<3>())
                if (spec.name == name)
                    return &spec.default_value;
            return nullptr;
        };
        return true;
    }();

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
        os << "\n## Options with separate particle species\n\n";
        os << "Not everything is implemented yet for CDM and baryons as separate particles (`particle_baryons`) and "
              "for neutrino particles (`particle_neutrinos`). The code stops with a message if one of these is not "
              "satisfied (generated from `species_requirements()` in `src/SimulationParameters.h`):\n\n";
        print_species_requirements(os);
    }

} // namespace COLASOLVER

#endif
