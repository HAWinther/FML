#ifndef SIMULATIONPARAMETERS_HEADER
#define SIMULATIONPARAMETERS_HEADER
#include <FML/NBody/TimeStepTypes.h>

#include "BaryonificationParameters.h"
#include "ParameterDefaults.h"
#include "ParameterFields.h"
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/ParameterMap/ParameterMap.h>

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// The parameters of an N-body simulation, grouped by what they are used for.
// Every settings struct lists its parameters once in fields(f) (name, member, default, description, allowed
// values and when they are used; see ParameterFields.h): the parameter table (ParameterSchema.h) and the reading
// (SimulationParameters::read) both come from it. The parameters are checked for consistency in
// SimulationParameters::validate so that mistakes in the parameterfile are caught before the simulation starts
//=============================================================================

namespace SIMCODE {

    using ParameterMap = FML::UTILS::ParameterMap;

    //=============================================================================
    // Options that are given as strings in the parameterfile
    //=============================================================================
    enum class ICRandomFieldType { Gaussian, NonGaussian, ReadParticles, ReadPhases };
    enum class ICInputType { PowerSpectrum, TransferFunction, CambFiles, ClassFiles, InternalTransfer, TransferTables };
    enum class RandomGeneratorType { GSL, MT19937 };
    enum class RandomFieldBackend { NGenIC, Philox };
    using TimestepMethod = FML::NBODY::TimestepMethod;
    using ScaleFactorSpacing = FML::NBODY::ScaleFactorSpacing;
    enum class ParticleFileFormat { GADGET, HDF5 };

    //=============================================================================
    /// Reads parameters from a ParameterMap and remembers what was read so that
    /// it can all be printed at the end
    //=============================================================================
    class ParameterReader {
      public:
        explicit ParameterReader(const ParameterMap & param) : param(param) {}

        /// Read a parameter: its value in the map (a parameter is in the map if its condition in the table holds)
        /// or else the default value of the parameter table. Only the parameters in the map are printed
        template <class V>
        void get(const std::string & name, V & value) {
            value = get_parameter<V>(param, name);
            if (param.contains(name))
                record(name, value);
        }

        /// Read an optional parameter whose default is defined next to the setting (the parameter table takes the
        /// default from there too, e.g. BaryonificationSettings)
        template <class V>
        void get(const std::string & name, V & value, const V & default_value) {
            check_parameter_name(name);
            value = param.get<V>(name, default_value);
            record(name, value);
        }

        /// Read a string parameter that must be one of the given options
        template <class E>
        void get_choice(const std::string & name, E & value, const std::vector<std::pair<std::string, E>> & options) {
            const auto str = get_parameter<std::string>(param, name);
            if (param.contains(name))
                record(name, str);
            for (auto & option : options) {
                if (option.first == str) {
                    value = option.second;
                    return;
                }
            }
            std::string valid;
            for (auto & option : options)
                valid += " " + option.first;
            throw std::runtime_error("Unknown value [" + str + "] for parameter " + name +
                                     ". Valid options are:" + valid);
        }

        /// Check the value of a parameter that is in use (in the map: its conditions hold)
        template <class V>
        void check(const std::string & name, const V & value, const Check & c) const {
            if (param.contains(name))
                c.apply(name, value);
        }

        /// Change the value printed for a parameter that was read (if it was in the parameterfile)
        template <class V>
        void update(const std::string & name, const V & value) {
            std::ostringstream os;
            os << std::boolalpha << value;
            for (auto & entry : entries)
                if (entry.first == name)
                    entry.second = os.str();
        }

        /// Record a value that was not read directly from the parameterfile
        template <class V>
        void record(const std::string & name, const V & value) {
            std::ostringstream os;
            os << std::boolalpha << value;
            entries.push_back({name, os.str()});
        }

        void print(const std::string & title) const {
            if (FML::ThisTask != 0)
                return;
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# " << title << "\n";
            std::cout << "#=====================================================\n\n";
            for (auto & entry : entries)
                std::cout << std::left << std::setw(41) << entry.first << ": " << entry.second << "\n";
            std::cout << std::right;
        }

      private:
        const ParameterMap & param;
        std::vector<std::pair<std::string, std::string>> entries;

        template <class V>
        void record(const std::string & name, const std::vector<V> & values) {
            std::ostringstream os;
            for (auto & v : values)
                os << v << " ";
            entries.push_back({name, os.str()});
        }
    };

    //=============================================================================
    /// The visitor of the fields(f) of the settings structs that reads them (ParameterFields.h)
    //=============================================================================
    class FieldReader : public FieldVisitor<FieldReader> {
      public:
        explicit FieldReader(ParameterReader & r) : r(r) {}
        /// Read the member and check it (if the parameter is in use)
        template <class V, class D>
        void field(Need,
                   const std::string & name,
                   V & value,
                   const D &,
                   const std::string &,
                   const std::vector<std::string> &,
                   const Check & c) {
            r.get(name, value);
            r.check(name, value, c);
        }
        template <class E, class D>
        void choice(Need,
                    const std::string & name,
                    E & value,
                    const D &,
                    const std::string &,
                    const EnumOptions<E> & options) {
            r.get_choice(name, value, options);
        }
        /// (a parameter is read from the map if it is there, otherwise it gets its default: the conditions only
        /// matter for the table)
        template <class B>
        void when(const Condition &, const std::string &, B && body) {
            body();
        }
        template <class B>
        void if_true(const std::string &, B && body) {
            body();
        }

      private:
        ParameterReader & r;
    };

    //=============================================================================
    // The groups of parameters
    //=============================================================================

    struct SimulationSettings {
        std::string name;              // The name of sim used for outputs
        double boxsize;                // The boxsize in Mpc/h
        bool use_cola;                 // Use the cola method?
        bool use_scaledependent_cola;  // If cola, use cola with scaledependent growth?
        bool enforce_LPT_trajectories; // No PM force, particles follow LPT trajectories (requires COLA)
        int cola_LPT_order;            // The LPT order for COLA
        std::string fftw_planner;      // FFTW planner flag: estimate, measure, patient or exhaustive
        std::string fftw_wisdom_file;  // FFTW wisdom (plans) to read at the start ("" = none)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("simulation_boxsize", boxsize, 0.0, "Boxsize of the simulation (Mpc/h)");
            f.required("simulation_name", name, "", "Label used in the output filenames");
            f.optional("simulation_use_cola", use_cola, false, "Use the COLA method");
            f.optional("simulation_use_scaledependent_cola",
                       use_scaledependent_cola,
                       true,
                       "If the growth is scale-dependent (massive neutrinos, k-dependent gravity models): use "
                       "scale-dependent COLA. If false we use the k = 0 limit of the growth factors when doing COLA. "
                       "With scale-independent growth (or without COLA) it is set to false (standard COLA gives the "
                       "same result at a much lower cost)");
            f.if_true("simulation_use_cola", [&]() {
                f.optional("simulation_enforce_LPT_trajectories",
                           enforce_LPT_trajectories,
                           false,
                           "Do not compute the PM force, just let particles follow the LPT trajectories");
                f.optional("simulation_cola_LPT_order",
                           cola_LPT_order,
                           2,
                           "The LPT order of COLA (1 or 2). Has to be <= ic_LPT_order. If < 1 COLA is turned off");
            });

            f.always_optional(
                "simulation_fftw_planner",
                fftw_planner,
                "estimate",
                "How FFTW makes its plans. estimate: no measuring (fast to start, slower transforms). "
                "measure, patient, exhaustive: time candidate algorithms the first time each grid size is "
                "transformed (faster transforms, often 1.5-2x for large grids, but planning can take "
                "seconds to minutes). The plans made are written to fftw_wisdom_[simulation_name].txt in "
                "the output folder (not for estimate): give that file as simulation_fftw_wisdom_file to "
                "skip the planning in later runs",
                {"estimate", "measure", "patient", "exhaustive"});
            f.always_optional(
                "simulation_fftw_wisdom_file",
                fftw_wisdom_file,
                "",
                "FFTW wisdom (plans) to read at the start, e.g. the fftw_wisdom_[simulation_name].txt "
                "written by an earlier run (\"\": none). Plans in it are used for the transforms they "
                "match (same FFTW version, precision, grid size and number of threads; a different machine "
                "or thread count gives plans that are valid but possibly slow). Other transforms are "
                "planned with simulation_fftw_planner");
        }
    };

    struct ForceSettings {
        int nmesh;                             // The gridsize to bin particles to and compute PM forces
        std::string density_assignment_method; // Density assignment (NGP,CIC,TSC,PCS,PQS)
        bool linear_massive_neutrinos;         // Include the effects of massive neutrinos using linear theory
        bool neutrino_energy_density;          // Source the Poisson equation with the neutrino energy density
        bool nbody_gauge;                      // Add the linear N-body gauge source (radiation, neutrinos, GR)
        std::string nbody_gauge_filename;      // The table T_extra(k,z) (from scripts/make_nbody_gauge_transfer.py)
        std::string greens_function_kernel;    // Kernel used for the Greens function of the Poisson equation
        std::string gradient_kernel;           // Kernel used for computing the gradient of the potential
        std::string density_method;            // assignment, nufft, sheet or sheet_nufft
        double nufft_epsilon;                  // Accuracy of the NUFFT (density_method = nufft)
        int sheet_refinement;                  // R: R^NDIM tracers per particle (density_method = sheet)
        std::string interpolation_method;      // Force to particles: assignment (kernel) or nufft
        int interlacing_order;                 // Average the force over this many shifted copies of the particles
        bool deconvolve_window;                // Divide the force by W(k)^2 of the assignment kernel
        // Early-time force: a more accurate (and expensive) force before force_early_until_redshift
        std::string early_method;                    // none, nufft or interlaced
        double early_until_redshift;                 // Use the early force for steps with z > this
        int early_interlacing_order;                 // early_method = interlaced: number of shifted grids
        std::string early_density_assignment_method; // early_method = interlaced: assignment kernel

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("force_nmesh", nmesh, 0, "Grid size used to compute the PM forces", check::positive());
            f.optional("force_density_assignment_method",
                       density_assignment_method,
                       "CIC",
                       "Density assignment method",
                       choices::assignment_methods);
            f.optional("force_greens_function_kernel",
                       greens_function_kernel,
                       "fiducial",
                       "Kernel for D^2 when solving the Poisson equation (fiducial = continuous)",
                       choices::kernels);
            f.optional(
                "force_gradient_kernel",
                gradient_kernel,
                "fiducial",
                "Kernel for D when computing the forces. fiducial = continuous if force_nmesh = "
                "particle_Npart_1D, discrete_4pt if force_nmesh is up to 3x larger and discrete_2pt if it is even "
                "larger (the continuous kernel gives "
                "spurious forces from the particle lattice when the force grid is finer than it)",
                choices::kernels);
            f.always_optional("force_density_method",
                              density_method,
                              "assignment",
                              "How the density field for the force is computed. assignment: particles to grid with "
                              "force_density_assignment_method. nufft: the exact Fourier modes of the particle "
                              "distribution (non-uniform FFT; no window function or aliasing). sheet: spectral sheet "
                              "interpolation, R^NDIM tracers per particle placed by Fourier interpolating the "
                              "displacement field x - q, assigned with force_density_assignment_method. sheet_nufft: "
                              "sheet interpolation with the exact Fourier modes of the tracers (NUFFT). The force is "
                              "always interpolated back to the particles with force_density_assignment_method",
                              {"assignment", "nufft", "sheet", "sheet_nufft"});
            f.always_optional("force_interlacing_order",
                              interlacing_order,
                              1,
                              "Force interlacing: compute the force for this many copies of the particles shifted "
                              "by a fraction of a cell relative to the grid and average (at the particles). 1: off, "
                              "2: shift (1/2,1/2,1/2), 4: the half-cell face diagonals, 8: all half-cell corners, "
                              "64: quarter cells. Cancels the leading aliasing of the density assignment and the force "
                              "interpolation at the cost of this many force calculations");
            f.always_optional("force_deconvolve_window",
                              deconvolve_window,
                              false,
                              "Divide the force by W(k)^2 where W is the window function of "
                              "force_density_assignment_method (corrects for the smoothing of the assignment and the "
                              "interpolation). Unstable unless the aliasing is suppressed (force_interlacing_order "
                              "> 1, see README)");
            f.always_optional("force_early_method",
                              early_method,
                              "none",
                              "A more accurate force at early times (when the particles are still close to the "
                              "lattice and CIC is inaccurate): nufft (NUFFT density and interpolation) or interlaced "
                              "(force_early_interlacing_order shifted grids with force_early_density_assignment_method "
                              "and window deconvolution) for steps at z > force_early_until_redshift, then the "
                              "normal force settings. none: off",
                              {"none", "nufft", "interlaced"});
            f.when(is_one_of("force_early_method", {"nufft", "interlaced"}),
                   one_of_text("force_early_method", {"nufft", "interlaced"}),
                   [&]() {
                       f.always_optional("force_early_until_redshift",
                                         early_until_redshift,
                                         10.0,
                                         "Use the early force for z > this");
                   });
            f.when(is_one_of("force_early_method", {"interlaced"}), "force_early_method = interlaced", [&]() {
                f.always_optional("force_early_interlacing_order",
                                  early_interlacing_order,
                                  2,
                                  "Number of shifted grids at early times",
                                  check::at_least(2));
                f.always_optional("force_early_density_assignment_method",
                                  early_density_assignment_method,
                                  "TSC",
                                  "Assignment kernel at early times",
                                  choices::assignment_methods);
            });
            f.always_optional("force_interpolation_method",
                              interpolation_method,
                              "fiducial",
                              "How the force is interpolated to the particles. assignment: with the "
                              "force_density_assignment_method kernel. nufft: exact evaluation of the Fourier series "
                              "of the force at the particles (non-uniform FFT). fiducial: nufft if "
                              "force_density_method = nufft (then assignment and interpolation are both exact, so "
                              "there is no self-force), otherwise assignment",
                              {"fiducial", "assignment", "nufft"});
            f.when(
                [](const ParameterMap & p) {
                    const auto d = p.get<std::string>("force_density_method");
                    return d == "nufft" or d == "sheet_nufft" or
                           p.get<std::string>("force_interpolation_method") == "nufft" or
                           p.get<std::string>("force_early_method") == "nufft";
                },
                "force_density_method in {nufft, sheet_nufft} or force_interpolation_method = nufft or "
                "force_early_method = nufft",
                [&]() {
                    f.always_optional("force_nufft_epsilon",
                                      nufft_epsilon,
                                      1e-5,
                                      "Relative accuracy of the NUFFT (force_density_method = nufft or sheet_nufft, "
                                      "force_interpolation_method = nufft or force_early_method = nufft)",
                                      check::in_open_range(0, 0.1));
                });
            f.when(is_one_of("force_density_method", {"sheet", "sheet_nufft"}),
                   one_of_text("force_density_method", {"sheet", "sheet_nufft"}),
                   [&]() {
                       f.always_optional("force_sheet_refinement",
                                         sheet_refinement,
                                         2,
                                         "R: the number of tracers per particle per dimension for sheet interpolation",
                                         check::at_least(1));
                   });
            f.optional("force_linear_massive_neutrinos",
                       linear_massive_neutrinos,
                       false,
                       "Include massive neutrinos in the density field using linear theory (requires "
                       "ic_type_of_input = camb, class, internal or tables)");
            f.always_optional(
                "force_neutrino_energy_density",
                neutrino_energy_density,
                false,
                "Source the Poisson equation with the energy density of the massive neutrinos instead of "
                "their mass density: f_nu <E/m>(a) delta_nu with <E/m> = <sqrt(1 + (q/am)^2)> over the "
                "Fermi-Dirac distribution (both for force_linear_massive_neutrinos, where delta_nu from "
                "CAMB/CLASS is already the energy-density contrast, and for particle_neutrinos, where each "
                "particle is weighted by its E/m and the neutrino P(k) is then the energy-density "
                "contrast). Matters at high redshift for light neutrinos (<E/m> = 1.33 at z = 50 for "
                "0.033 eV)");
            f.always_optional("force_nbody_gauge",
                              nbody_gauge,
                              false,
                              "Run in the N-body gauge: the initial conditions in the N-body gauge and the linear "
                              "source from radiation, neutrinos and GR added to the density field. With "
                              "ic_type_of_input = internal both come from the Einstein-Boltzmann solver; with camb "
                              "from the tables made with CLASS by scripts/make_nbody_gauge_transfer.py");
            f.when(
                [](const ParameterMap & p) {
                    const auto type = p.get<std::string>("ic_type_of_input");
                    return p.get<bool>("force_nbody_gauge") and (type == "camb" or type == "transferinfofile");
                },
                "force_nbody_gauge = true and ic_type_of_input = camb",
                [&]() {
                    f.required("force_nbody_gauge_filename",
                               nbody_gauge_filename,
                               "",
                               "The table T_extra(k,z) (nbgauge_extra.txt from scripts/make_nbody_gauge_transfer.py)");
                });
        }
    };

    struct ICSettings {
        ICRandomFieldType random_field_type; // gaussian, nongaussian, read_particles, read_phases
        double initial_redshift;             // The initial redshift of the sim
        int nmesh;                           // The Nmesh used to generate the IC (use particle_Npart_1D)
        std::string lattice_offset;          // Offset of the particle lattice from the force grid nodes
        int random_seed;                     // The random seed
        RandomFieldBackend random_field_backend;
        RandomGeneratorType random_generator; // The generator: GSL or MT19937
        int LPT_order;                        // The LPT order to use to make IC (1,2,3)

        // Input file (power-spectrum / transfer functions)
        ICInputType type_of_input;  // powerspectrum, transferfunction, camb, class, internal or tables
        std::string input_filename; // The filename (class: the root of the CLASS run)
        std::string input_gauge;    // class: the gauge CLASS was run in (synchronous or newtonian)
        double input_redshift;      // The redshift of P(k,z) / T(k,z) that we read in
        bool use_gravity_model_GR;  // Input power-spectrum is for LCDM so if MG use LCDM to set the IC

        // Amplitude fixed / paired IC
        bool fix_amplitude;  // Fix amplitude of delta(k,zini) to that of P(k)
        bool reverse_phases; // Set delta(k,zini) -> -delta(k,zini)

        // Non-gaussianity (random_field_type = nongaussian)
        std::string fnl_type; // local, equilateral, orthogonal
        double fnl;           // fNL
        double fnl_redshift;  // Redshift of which to generate the primordial potential

        // Normalization using sigma8
        bool sigma8_normalization; // Normalize wrt to a sigma8 value?
        double sigma8_redshift;    // The redshift of which to normalize
        double sigma8;             // The value of sigma8 at this redshift we want

        // Read IC from file (and reconstruct the LPT fields if COLA)
        std::string reconstruct_gadgetfilepath;     // /path/to/gadget
        std::string reconstruct_assigment_method;   // Density assignment method (NGP, CIC, ...)
        std::string reconstruct_smoothing_filter;   // Smoothing filter (tophat, sharpk, gaussian)
        double reconstruct_dimless_smoothing_scale; // Smoothing scales R/boxsize
        bool reconstruct_interlacing;               // Use interlacing (probably not)?
        bool reconstruct_exact;                     // Reconstruct the LPT fields exactly from pos+vel

        // ic_type_of_input = internal: the settings of the Einstein-Boltzmann solver (read by TransferInput.h)
        std::vector<std::string> internal_solver_parameters;

        double a_initial() const { return 1.0 / (1.0 + initial_redshift); }
        /// The value of ic_type_of_input
        std::string type_of_input_name() const {
            switch (type_of_input) {
                case ICInputType::PowerSpectrum:
                    return "powerspectrum";
                case ICInputType::TransferFunction:
                    return "transferfunction";
                case ICInputType::CambFiles:
                    return "camb";
                case ICInputType::ClassFiles:
                    return "class";
                case ICInputType::InternalTransfer:
                    return "internal";
                case ICInputType::TransferTables:
                    return "tables";
            }
            return "";
        }
        /// The input has transfer functions of all species at all redshifts (CAMB, CLASS or the internal solver)
        bool has_transfer_tables() const {
            return type_of_input == ICInputType::CambFiles or type_of_input == ICInputType::ClassFiles or
                   from_internal_solver();
        }
        /// The tables come from the internal solver (internal, or tables written by ./nbody --write-transfers)
        bool from_internal_solver() const {
            return type_of_input == ICInputType::InternalTransfer or type_of_input == ICInputType::TransferTables;
        }
        bool reads_particles() const {
            return random_field_type == ICRandomFieldType::ReadParticles or
                   random_field_type == ICRandomFieldType::ReadPhases;
        }

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("ic_initial_redshift", initial_redshift, 0.0, "The initial redshift of the simulation");
            f.required("ic_random_seed", random_seed, 1234, "The random seed");
            f.choice(Need::Optional,
                     "ic_random_generator",
                     random_generator,
                     "MT19937",
                     "The random number generator (GSL is gsl_rng_ranlxd1 as used in 2LPTic, for comparisons)",
                     {{"GSL", RandomGeneratorType::GSL}, {"MT19937", RandomGeneratorType::MT19937}});
            f.choice(Need::AlwaysOptional,
                     "ic_random_field_backend",
                     random_field_backend,
                     "ngenic",
                     "Fourier mode sampling: ngenic preserves legacy line seeds and IC; philox uses "
                     "independent mode counters",
                     {{"ngenic", RandomFieldBackend::NGenIC}, {"philox", RandomFieldBackend::Philox}});
            f.required("ic_nmesh", nmesh, 0, "The grid size used to generate the IC", check::positive());
            f.always_optional(
                "ic_lattice_offset",
                lattice_offset,
                "fiducial",
                "Offset of the initial particle lattice from the nodes of the force grid. fiducial: half a "
                "force grid cell if force_nmesh > particle_Npart_1D, otherwise none. With particles "
                "exactly on the nodes of a finer grid, CIC responds to the size of the displacement and gives large "
                "spurious forces (see the force grid section of README.md)",
                {"fiducial", "none", "half_force_cell"});
            f.choice(Need::Required,
                     "ic_type_of_input",
                     type_of_input,
                     "powerspectrum",
                     "The input: powerspectrum (file with [k (h/Mpc), P(k) (Mpc/h)^3]), transferfunction (file "
                     "with [k (h/Mpc), T(k) Mpc^2]), camb (a transfer info file listing T(k,z) files from CAMB; the "
                     "old name "
                     "transferinfofile is also accepted; the "
                     "columns are read by name from the header, or by position: 13 CAMB, 9 AxionCAMB, 7 old CAMB / "
                     "CLASS with format=camb), class (native CLASS output: ic_input_filename is the root of the run, "
                     "with format = class, output = mTk,vTk, write_background = yes), internal (FML's "
                     "Einstein-Boltzmann solver run with the cosmology of the simulation, cosmology_model included; "
                     "the old name internal_transfer is also accepted) or tables (the tables of the internal solver "
                     "in one file, made with ./nbody --write-transfers parameterfile.lua file; checked against the "
                     "cosmology of the simulation). See FML/TransferProviders and README.md",
                     {{"powerspectrum", ICInputType::PowerSpectrum},
                      {"transferfunction", ICInputType::TransferFunction},
                      {"camb", ICInputType::CambFiles},
                      {"class", ICInputType::ClassFiles},
                      {"internal", ICInputType::InternalTransfer},
                      {"tables", ICInputType::TransferTables},
                      {"internal_transfer", ICInputType::InternalTransfer},
                      {"transferinfofile", ICInputType::CambFiles}});
            f.when(is_one_of("ic_type_of_input", {"class"}), "ic_type_of_input = class", [&]() {
                f.optional("ic_input_gauge",
                           input_gauge,
                           "synchronous",
                           "The gauge CLASS was run in (newtonian is converted to the CDM-synchronous gauge the IC "
                           "need; it needs output = mTk,vTk)",
                           {"synchronous", "newtonian"});
            });
            f.optional("ic_input_filename",
                       input_filename,
                       "",
                       "The input file (needed except for ic_type_of_input = internal, where it is an optional .ini "
                       "file of the Einstein-Boltzmann solver with its precision, recombination and reionization "
                       "settings: see ic_internal_solver_parameters)");
            f.when(is_one_of("ic_type_of_input", {"internal", "internal_transfer"}),
                   "ic_type_of_input = internal",
                   [&]() {
                       f.optional(
                           "ic_internal_solver_parameters",
                           internal_solver_parameters,
                           std::vector<std::string>{},
                           "Settings of the Einstein-Boltzmann solver as \"name=value\", e.g. {\"Yp=0.245\", "
                           "\"z_reion=8\", \"precision=fast\"}, applied after those of the .ini file "
                           "ic_input_filename (the names of Cosmology/EinsteinBoltzmannSolver/params/default.ini). "
                           "The cosmology (h, OmegaB, ..., model and its parameters, A_s, n_s, k_pivot, the "
                           "neutrino masses, preset), alpha_s and the k-grid (pert_k_min, pert_k_max, pert_n_k, "
                           "pert_dk_max) are set by the simulation and are an error here and in the .ini file. "
                           "See README.md");
                   });
            f.required("ic_input_redshift", input_redshift, 0.0, "The redshift of the input P(k) / T(k)");
            f.optional("ic_fix_amplitude",
                       fix_amplitude,
                       true,
                       "Fix the amplitude of the modes when generating the random field");
            f.optional("ic_reverse_phases", reverse_phases, false, "Reverse the phases (for paired simulations)");
            f.choice(Need::Optional,
                     "ic_random_field_type",
                     random_field_type,
                     "gaussian",
                     "Type of IC: gaussian, nongaussian, read_particles (read a GADGET file and use that, "
                     "reconstructing the LPT fields if COLA) or read_phases (take the phases from a GADGET file)",
                     {{"gaussian", ICRandomFieldType::Gaussian},
                      {"nongaussian", ICRandomFieldType::NonGaussian},
                      {"read_particles", ICRandomFieldType::ReadParticles},
                      {"read_phases", ICRandomFieldType::ReadPhases}});
            f.optional("ic_LPT_order", LPT_order, 2, "The LPT order used to make the IC");
            f.optional("ic_sigma8_normalization",
                       sigma8_normalization,
                       false,
                       "Normalize P(k) to a given sigma8 (otherwise use the input)");
            f.if_true("ic_sigma8_normalization", [&]() {
                f.optional("ic_sigma8_redshift", sigma8_redshift, 0.0, "The redshift of the sigma8 value");
                f.optional("ic_sigma8", sigma8, 0.8, "The sigma8 value to normalize to");
            });
            f.optional("ic_use_gravity_model_GR",
                       use_gravity_model_GR,
                       false,
                       "For modified gravity: the input P(k) is for LCDM, use GR growth factors to scale it to the "
                       "initial redshift (same IC as the LCDM simulation)");
            f.when(is_one_of("ic_random_field_type", {"nongaussian"}), "ic_random_field_type = nongaussian", [&]() {
                f.required("ic_fnl_type",
                           fnl_type,
                           "local",
                           "Type of non-gaussianity",
                           {"local", "equilateral", "orthogonal"});
                f.required("ic_fnl", fnl, 100.0, "The value of fNL");
                f.required(
                    "ic_fnl_redshift",
                    fnl_redshift,
                    0.0,
                    "The redshift of the potential Phi = (3/5) g(z) zeta (g = D/a) that fNL multiplies: 0 is the "
                    "usual large-scale-structure convention, a matter-era redshift (e.g. 30) the CMB convention");
            });
            f.when(is_one_of("ic_random_field_type", {"read_particles", "read_phases"}),
                   "ic_random_field_type in {read_particles, read_phases}",
                   [&]() {
                       f.required("ic_reconstruct_gadgetfilepath",
                                  reconstruct_gadgetfilepath,
                                  "",
                                  "Path to the GADGET files without the file number: binary (path.0, path.1, ...) "
                                  "or HDF5 (path.hdf5 or path.0.hdf5, ...), detected automatically");
                       f.optional("ic_reconstruct_assigment_method",
                                  reconstruct_assigment_method,
                                  "CIC",
                                  "Density assignment method used to reconstruct the density field",
                                  choices::assignment_methods);
                       f.optional("ic_reconstruct_smoothing_filter",
                                  reconstruct_smoothing_filter,
                                  "sharpk",
                                  "Filter used to remove small-scale modes from the reconstructed density field",
                                  choices::smoothing_filters);
                       f.optional("ic_reconstruct_dimless_smoothing_scale",
                                  reconstruct_dimless_smoothing_scale,
                                  0.0,
                                  "Smoothing scale R/boxsize of the reconstructed density field");
                       f.optional("ic_reconstruct_interlacing",
                                  reconstruct_interlacing,
                                  false,
                                  "Use interlacing when reconstructing the density field");
                       f.optional("ic_reconstruct_exact",
                                  reconstruct_exact,
                                  false,
                                  "Reconstruct the LPT fields exactly from the positions and velocities (always works "
                                  "for 1LPT IC; for 2LPT IC the particles must come from a regular grid)");
                   });
        }
    };

    struct ParticleSettings {
        int Npart_1D;                     // Number of particles per dimension (total is N^NDIM)
        double allocation_factor;         // Allocate this many more particles
        bool baryons;                     // CDM and baryons as two sets of particles (otherwise one set "cb")
        std::string baryon_lattice;       // same: the baryons start from the CDM lattice, offset: half a spacing
        std::string baryon_relative_mode; // displacement, mass or mass_ini (see ParameterSchema.h)
        bool neutrinos;                   // Massive neutrinos as particles (delta-f method)
        int neutrino_Npart_1D;            // Number of neutrino particles per dimension (0: particle_Npart_1D)
        bool neutrino_deltaf;             // Use the delta-f weights (false: plain particles, for testing)
        int sort_every_nsteps; // Reorder the particles in memory by position every this many steps (0: never)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("particle_Npart_1D",
                       Npart_1D,
                       0,
                       "Number of CDM+baryon particles per dimension",
                       check::positive());
            f.always_optional("particle_baryons",
                              baryons,
                              false,
                              "Evolve CDM and baryons as two sets of particles (particle_Npart_1D^3 of each, IC from "
                              "the CDM and baryon transfer functions at zini). false: one set of CDM+baryon particles");
            f.always_optional("particle_neutrinos",
                              neutrinos,
                              false,
                              "Massive neutrinos as particles with the delta-f method (Elbers et al. 2021): thermal "
                              "Fermi-Dirac velocities, relativistic equations of motion, the particles only carry the "
                              "perturbation. Replaces force_linear_massive_neutrinos (needs cosmology_OmegaMNu > 0, "
                              "ic_type_of_input = camb, class, internal or tables, timestep_method = Quinn, no COLA)");
            f.if_true("particle_neutrinos", [&]() {
                f.always_optional(
                    "particle_neutrino_Npart_1D",
                    neutrino_Npart_1D,
                    0,
                    "particle_neutrinos: number of neutrino particles per dimension (0: particle_Npart_1D)");
                f.always_optional(
                    "particle_neutrino_deltaf",
                    neutrino_deltaf,
                    true,
                    "particle_neutrinos: use the delta-f weights (false: plain particles, much noisier; for "
                    "testing)");
            });
            f.if_true("particle_baryons", [&]() {
                f.always_optional(
                    "particle_baryon_lattice",
                    baryon_lattice,
                    "same",
                    "particle_baryons: start the baryons from the same lattice as the CDM (same, recommended: "
                    "the baryon-CDM difference then follows linear theory on all scales) or from a lattice "
                    "offset by half a particle spacing (offset: gives a spurious baryon-CDM difference on "
                    "small scales)",
                    {"same", "offset"});
                f.always_optional(
                    "particle_baryon_relative_mode",
                    baryon_relative_mode,
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
            });
            f.optional("particle_allocation_factor",
                       allocation_factor,
                       1.5,
                       "Allocate this factor more particles than the average per task (particles move between "
                       "tasks)");
            f.always_optional("particle_sort_every_nsteps",
                              sort_every_nsteps,
                              0,
                              "Reorder the particles in memory by position (blocks of 4^3 force-grid cells) every this "
                              "many steps, which makes the density assignment and the force interpolation faster when "
                              "the particles have moved far from their initial (lattice) order. 0: never. Changes the "
                              "results only by round-off (the order of the sums) and the order of the particles in "
                              "the output");
        }
    };

    struct TimestepSettings {
        std::vector<int> nsteps;    // Steps between the outputs (if just 1 number the total amount of steps)
        TimestepMethod method;      // Quinn, Tassev (for COLA mainly), FastPM or BullFrog (see TimeStepping.h)
        std::string algorithm;      // KDK - Kick-Drift-Kick is the main method we use
        double cola_nLPT;           // nLPT. For method = Tassev
        ScaleFactorSpacing spacing; // linear, logarithmic, powerlaw, growthfactor
        double spacing_power;       // For spacing = powerlaw

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("timestep_nsteps",
                       nsteps,
                       std::vector<int>{0},
                       "Number of steps between the outputs (one per output redshift). If only one number it is the "
                       "total number of steps");
            f.optional("timestep_algorithm",
                       algorithm,
                       "KDK",
                       "The time-stepping algorithm (only KDK is implemented)",
                       {"KDK"});
            f.choice(Need::Optional,
                     "timestep_method",
                     method,
                     "Quinn",
                     "The time-stepping method. Quinn: standard leapfrog. Tassev: modified leapfrog for COLA. "
                     "FastPM: leapfrog that follows linear growth exactly (Feng et al. 2016). BullFrog: follows "
                     "2LPT growth after every step (Rampf, List & Hahn 2024), not for COLA",
                     {{"Quinn", TimestepMethod::Quinn},
                      {"Tassev", TimestepMethod::Tassev},
                      {"FastPM", TimestepMethod::FastPM},
                      {"BullFrog", TimestepMethod::BullFrog}});
            f.optional("timestep_cola_nLPT", cola_nLPT, -2.5, "The nLPT parameter of the Tassev time-stepping");
            f.choice(Need::Optional,
                     "timestep_scalefactor_spacing",
                     spacing,
                     "linear",
                     "Spacing of the steps in the scale factor (powerlaw: uniform in a^timestep_spacing_power, "
                     "growthfactor: uniform in the linear growth factor D(a), as in Rampf, List & Hahn 2024)",
                     {{"linear", ScaleFactorSpacing::Linear},
                      {"logarithmic", ScaleFactorSpacing::Logarithmic},
                      {"powerlaw", ScaleFactorSpacing::Powerlaw},
                      {"growthfactor", ScaleFactorSpacing::GrowthFactor}});
            f.when(is_one_of("timestep_scalefactor_spacing", {"powerlaw"}),
                   "timestep_scalefactor_spacing = powerlaw",
                   [&]() {
                       f.optional("timestep_spacing_power", spacing_power, 1.0, "The steps are uniform in a^power");
                   });
        }
    };

    struct FoFSettings {
        bool enabled;                  // Locate halos when we output
        int nmin_per_halo;             // Minimum particles per halo
        int nmesh_max;                 // For speeding it up: the maximum gridsize to bin particle to
        double linking_length;         // The linking length in units of the mean partice separation (i.e. ~0.2)
        double buffer_length_mpch;     // The buffer region (from nbor tasks) we use in the FoF finding
        std::string output_fileformat; // The halo catalogue: ascii or HDF5
        int massfunction_nbins;        // The mass function: number of bins...
        double massfunction_Mmin;      // ...and the range (Msun/h)
        double massfunction_Mmax;
        std::vector<std::string> so_definitions; // Spherical overdensity masses around the halos (e.g. 200m, 500c, vir)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.optional("fof", enabled, false, "Locate friends-of-friends halos at every output");
            f.if_true("fof", [&]() {
                f.optional("fof_nmin_per_halo",
                           nmin_per_halo,
                           20,
                           "Minimum number of particles per halo (of one species with particle_baryons)");
                f.optional("fof_linking_length",
                           linking_length,
                           0.2,
                           "The linking length in units of the mean particle separation (of one species with "
                           "particle_baryons)");
                f.optional("fof_nmesh_max",
                           nmesh_max,
                           0,
                           "Maximum grid size used to bin particles to speed up the linking (0: the code chooses)");
                f.optional(
                    "fof_buffer_length_mpch",
                    buffer_length_mpch,
                    3.0,
                    "Buffer region (larger than the largest halo) from neighbor tasks (Mpc/h). The code warns if "
                    "halos reach its edge (they might then be cut off and found twice)");
                f.always_optional("fof_output_fileformat",
                                  output_fileformat,
                                  "ascii",
                                  "fof: the halo catalogue halos_z<z>.txt (ascii, one row per halo) or halos_z<z>.hdf5 "
                                  "(HDF5, compile with USE_HDF5=true)",
                                  {"ascii", "HDF5"});
                f.always_optional("fof_massfunction_nbins",
                                  massfunction_nbins,
                                  30,
                                  "fof: number of logarithmic bins of the mass function",
                                  check::positive());
                f.always_optional("fof_massfunction_Mmin",
                                  massfunction_Mmin,
                                  1e10,
                                  "fof: smallest mass of the mass function (Msun/h)");
                f.always_optional("fof_massfunction_Mmax",
                                  massfunction_Mmax,
                                  1e16,
                                  "fof: largest mass of the mass function (Msun/h)");
                f.always_optional("fof_so_definitions",
                                  so_definitions,
                                  std::vector<std::string>{},
                                  "fof: spherical overdensity masses around the FoF halos, e.g. {\"200m\", \"500c\", "
                                  "\"vir\"} (<Delta>m: Delta times the mean CDM+baryon density, <Delta>c: the critical "
                                  "density, vir: Bryan & Norman 1998). Empty: none. The spheres must fit in "
                                  "fof_buffer_length_mpch");
            });
        }
    };

    struct PofkSettings {
        bool enabled;                          // Compute power-spectrum when we output
        int nmesh;                             // The grid size to use for this
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract shotnoise? (also used for P(k) every step)
        bool every_step;                       // P(k) of the force density field at every step
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.optional("pofk", enabled, false, "Compute the power-spectrum at every output");
            f.always_optional("pofk_every_step",
                              every_step,
                              true,
                              "Write P(k) of the density field used for the force at every step (cheap; same grid "
                              "and assignment as the force, the window is deconvolved): pofk_<name>_cb_z<z>.txt in "
                              "output_folder");
            f.when([](const ParameterMap & p) { return p.get<bool>("pofk") or p.get<bool>("pofk_every_step"); },
                   "pofk = true or pofk_every_step = true",
                   [&]() {
                       f.optional("pofk_subtract_shotnoise",
                                  subtract_shotnoise,
                                  false,
                                  "Subtract shot-noise (pofk and pofk_every_step)");
                   });
            f.if_true("pofk", [&]() {
                f.required("pofk_nmesh", nmesh, 0, "Grid size", check::positive());
                f.optional("pofk_interlacing", interlacing, false, "Use interlaced grids to reduce aliasing");
                f.optional("pofk_density_assignment_method",
                           density_assignment_method,
                           "CIC",
                           "Density assignment method",
                           choices::assignment_methods);
            });
        }
    };

    struct PofkMultipoleSettings {
        bool enabled;                          // Compute power-spectrum multipoles when we output
        int nmesh;                             // The grid size to use for this
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract shotnoise for P0?
        int ellmax;                            // Compute P_ell for ell=0,2,4,...,ellmax
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.optional("pofk_multipole",
                       enabled,
                       false,
                       "Compute the redshift space multipoles P_ell(k) at every output");
            f.if_true("pofk_multipole", [&]() {
                f.required("pofk_multipole_nmesh", nmesh, 0, "Grid size", check::positive());
                f.optional("pofk_multipole_interlacing", interlacing, false, "Use interlaced grids to reduce aliasing");
                f.optional("pofk_multipole_subtract_shotnoise",
                           subtract_shotnoise,
                           false,
                           "Subtract shot-noise from P0");
                f.optional("pofk_multipole_ellmax",
                           ellmax,
                           4,
                           "Compute P_ell for ell = 0, 2, ..., ellmax",
                           check::non_negative());
                f.optional("pofk_multipole_density_assignment_method",
                           density_assignment_method,
                           "CIC",
                           "Density assignment method",
                           choices::assignment_methods);
            });
        }
    };

    struct BispectrumSettings {
        bool enabled;                          // Compute bispectrum when we output
        int nmesh;                             // The grid size to use for this
        int nbins;                             // The number of bins in k to use (memory req. scales as nbins^3)
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract shotnoise?

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.optional("bispectrum", enabled, false, "Compute the bispectrum at every output");
            f.if_true("bispectrum", [&]() {
                f.required("bispectrum_nmesh", nmesh, 0, "Grid size", check::positive());
                f.required("bispectrum_nbins",
                           nbins,
                           0,
                           "Number of k-bins (memory and time scale as nbins grids and nbins^3 integrals)",
                           check::positive());
                f.optional("bispectrum_density_assignment_method",
                           density_assignment_method,
                           "CIC",
                           "Density assignment method",
                           choices::assignment_methods);
                f.optional("bispectrum_interlacing", interlacing, true, "Use interlaced grids to reduce aliasing");
                f.optional("bispectrum_subtract_shotnoise", subtract_shotnoise, false, "Subtract shot-noise");
            });
        }
    };

    struct HybridBiasSettings {
        bool enabled;                          // Compute the hybrid bias (Lagrangian bias basis) spectra when we output
        int nmesh;                             // The grid size for the advected fields
        int noperators;                        // Use the first noperators of 1, delta, delta^2, s^2, nabla^2 delta
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract the Poisson shot noise mean(w_a w_b) / Npart?
        double smoothing_scale;                // Smoothing of the linear field before forming the operators (Mpc/h)
        std::string smoothing_filter;          // gaussian, sharpk or tophat
        double kcut_over_nyquist;              // Spherical cut of the linear field at this times the lattice Nyquist
        bool zeldovich;                        // Also output the Zel'dovich prediction
        bool oneloop;                          // Also output the one-loop LPT prediction
        std::string oneloop_kernels;           // EdS or model (the scale-dependent kernels of the gravity model)

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.always_optional("hybridbias",
                              enabled,
                              false,
                              "Compute the hybrid bias (Lagrangian bias basis) spectra P_ab(k) at every output: the "
                              "cross spectra of the operators 1, delta_L, delta_L^2, s^2, nabla^2 delta_L advected "
                              "with the particles (Modi et al. 2020, Kokron et al. 2021). delta_L is the linear field "
                              "of the IC scaled to the output redshift with the growth factor of the gravity model");
            f.if_true("hybridbias", [&]() {
                f.required("hybridbias_nmesh", nmesh, 0, "Grid size for the advected fields", check::positive());
                f.optional("hybridbias_noperators",
                           noperators,
                           5,
                           "Use the first noperators of (1, delta, delta^2, s^2, nabla^2 delta): 15 spectra for 5",
                           check::in_range(1, 5));
                f.optional("hybridbias_density_assignment_method",
                           density_assignment_method,
                           "CIC",
                           "Density assignment method",
                           choices::assignment_methods);
                f.optional("hybridbias_interlacing", interlacing, true, "Use interlaced grids to reduce aliasing");
                f.optional("hybridbias_subtract_shotnoise",
                           subtract_shotnoise,
                           false,
                           "Subtract the Poisson shot noise mean(w_a w_b)/Npart (the usual basis spectra keep it)");
                f.optional("hybridbias_smoothing_scale",
                           smoothing_scale,
                           0.0,
                           "Smooth the linear field with this scale (Mpc/h) before forming the operators (0: none)",
                           check::non_negative());
                f.optional("hybridbias_smoothing_filter",
                           smoothing_filter,
                           "gaussian",
                           "The smoothing filter",
                           choices::smoothing_filters);
                f.optional(
                    "hybridbias_kcut_over_nyquist",
                    kcut_over_nyquist,
                    1.0,
                    "Keep only the modes with k < this times the Nyquist frequency of the particle lattice in the "
                    "linear field (a spherical cut, so the operators are isotropic as assumed in the LPT "
                    "prediction). 0: keep all the modes of the lattice grid (a cube)",
                    check::non_negative());
                f.optional("hybridbias_zeldovich",
                           zeldovich,
                           false,
                           "Also write the Zel'dovich (resummed 1LPT) prediction of the spectra with the same cut and "
                           "smoothing of the operators: hybridbias_zeldovich_z<z>.txt");
                f.optional("hybridbias_oneloop",
                           oneloop,
                           false,
                           "Also write the one-loop LPT (CLEFT) prediction of the spectra (no counterterms): "
                           "hybridbias_oneloop_z<z>.txt");
                f.always_optional(
                    "hybridbias_oneloop_kernels",
                    oneloop_kernels,
                    "EdS",
                    "The kernels of the one-loop prediction. EdS: the Einstein-de Sitter kernels with the "
                    "linear P(k) of the gravity model. model: the scale- and time-dependent LPT kernels "
                    "of the gravity model (f(R) and DGP with their screening if gravity_model_screening, "
                    "the other models from G_eff(k,a) without screening, LCDM for GR)",
                    {"EdS", "model"});
            });
        }
    };

    struct OutputSettings {
        std::vector<double> redshifts; // List of output redshift from large to small
        bool particles;                // Output particles?
        ParticleFileFormat fileformat; // Fileformat for particles
        std::string folder;            // Folder to store output ("" is the current folder)

        /// Path for a file in the output folder
        std::string path(const std::string & filename) const { return folder + (folder == "" ? "" : "/") + filename; }

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.required("output_folder", folder, "", "Folder to put the output in (\"\" is the current folder)");
            f.required("output_redshifts", redshifts, std::vector<double>{}, "The output redshifts (decreasing)");
            f.optional("output_particles", particles, true, "Write the particles at the output redshifts");
            f.choice(Need::Optional,
                     "output_fileformat",
                     fileformat,
                     "GADGET",
                     "File format for the particles. GADGET: binary GADGET files. HDF5: GADGET HDF5 files "
                     "(compile with USE_HDF5=true)",
                     {{"GADGET", ParticleFileFormat::GADGET}, {"HDF5", ParticleFileFormat::HDF5}});
        }
    };

    struct LightconeSettings {
        bool enabled;                     // Make a lightcone
        std::vector<double> pos_observer; // Position of the observer in units of the boxsize
        double z_init;                    // The redshift we start recording the lightcone
        double z_finish;                  // The redshift we stop recording the lightcone
        double boundary_mpch;             // Boundary around the shell to get all the particles (Mpc/h)
        bool use_replicas;                // Replicate the box
        int ndim_rep;                     // Number of dimensions with replicas in both directions
        bool output_gadgetfile;           // Output the lightcone particles (GADGET, ascii, HDF5)
        bool output_asciifile;
        bool output_hdf5file;
        bool output_in_batches;       // Output in batches to save memory
        bool relativistic_effects;    // The potential and observed redshift at the particles
        bool make_onion_density_maps; // (Healpix) density maps delta(z, theta)
        int nside;                    // Healpix nside of the maps
        double da_maps;               // Roughly the size of the bins in a for the maps
        bool use_chunkpix;            // Store the maps with chunkpix
        int nside_chunks;             // Healpix nside of the chunks

        /// The parameters (see ParameterFields.h)
        template <class F>
        void fields(F && f) {
            f.optional("lightcone", enabled, false, "Make a lightcone");
            f.if_true("lightcone", [&]() {
                f.optional("plc_pos_observer",
                           pos_observer,
                           std::vector<double>{0.0, 0.0, 0.0},
                           "Position of the observer in units of the boxsize");
                f.optional("plc_z_init", z_init, 0.0, "The redshift we start recording the lightcone");
                f.optional("plc_z_finish", z_finish, 0.0, "The redshift we stop recording the lightcone");
                f.optional("plc_boundary_mpch",
                           boundary_mpch,
                           20.0,
                           "Boundary region around the shell to make sure we get all particles (Mpc/h)");
                f.optional("plc_use_replicas",
                           use_replicas,
                           false,
                           "Replicate the box to get the sky coverage we want");
                f.optional("plc_ndim_rep",
                           ndim_rep,
                           0,
                           "Number of dimensions with replicas in both directions (3D: 0 = octant, 3 = full sky)");
                f.optional("plc_output_gadgetfile",
                           output_gadgetfile,
                           false,
                           "Output the lightcone particles in GADGET format");
                f.optional("plc_output_asciifile", output_asciifile, false, "Output the lightcone particles in ascii");
                f.always_optional("plc_output_hdf5file",
                                  output_hdf5file,
                                  false,
                                  "Output the lightcone particles as GADGET HDF5 files (needs USE_HDF5; positions in "
                                  "Mpc/h, peculiar velocities in km/s)");
                f.optional("plc_output_in_batches", output_in_batches, false, "Output in batches to save memory");
                f.always_optional(
                    "plc_relativistic_effects",
                    relativistic_effects,
                    false,
                    "Store the potential Phi/c^2 at the lightcone particles and write it with the radial "
                    "velocity, the cosmological redshift and the observed redshift 1 + z_obs = (1 + z)(1 + "
                    "v_r/c - Phi/c^2) (Doppler + gravitational redshift; observer at rest with Phi = 0) to "
                    "the HDF5 and ascii lightcone files. With plc_make_onion_density_maps also Healpix maps "
                    "of Phi/c^2 at the shells. Phi from the Poisson equation with the density used for the "
                    "force (the GR potential also for modified gravity models). Costs one extra FFT per "
                    "step while recording the lightcone");
                f.optional("plc_make_onion_density_maps",
                           make_onion_density_maps,
                           false,
                           "Make (Healpix) density maps delta(z, theta)");
                f.if_true("plc_make_onion_density_maps", [&]() {
                    f.optional("plc_nside", nside, 256, "Healpix nside of the maps (npix = 12 nside^2; at most 8192)");
                    f.optional("plc_da_maps", da_maps, 0.02, "Roughly the size of the bins in a for the maps");
                    f.optional("plc_use_chunkpix",
                               use_chunkpix,
                               false,
                               "Store the maps with chunkpix (useful for sparse maps)");
                    f.if_true("plc_use_chunkpix", [&]() {
                        f.optional("plc_nside_chunks", nside_chunks, 8, "Healpix nside of the chunks (at most 8192)");
                    });
                });
            });
        }
    };

    struct SimulationParameters;

    //=============================================================================
    /// A spherical overdensity definition of fof_so_definitions: "vir" (Bryan & Norman 1998) or <Delta><reference>
    /// with reference m or b (the mean CDM+baryon density) or c (the critical density), e.g. 200m, 500c. Throws for
    /// anything else
    //=============================================================================
    struct SODefinition {
        bool virial{false};
        double Delta{0.0};
        char reference{'m'};
    };
    inline SODefinition parse_so_definition(const std::string & definition) {
        auto unknown = [&]() {
            return std::runtime_error("fof_so_definitions: unknown definition [" + definition +
                                      "] (<Delta>m or <Delta>b: mean CDM+baryon density, <Delta>c: critical density, "
                                      "vir)");
        };
        if (definition == "vir")
            return {true, 0.0, 'm'};
        if (definition.size() < 2)
            throw unknown();
        SODefinition def;
        def.reference = definition.back();
        try {
            def.Delta = std::stod(definition.substr(0, definition.size() - 1));
        } catch (...) {
            throw unknown();
        }
        if (not(def.Delta > 0.0) or (def.reference != 'm' and def.reference != 'b' and def.reference != 'c'))
            throw unknown();
        return def;
    }

    //=============================================================================
    /// What the separate particle species need (particle_baryons, particle_neutrinos): one row per requirement,
    /// whether it applies to the baryon and to the neutrino particles and a test of a set of parameters. Used to
    /// validate the parameters and for the table in PARAMETERS.md
    //=============================================================================
    struct SpeciesRequirement {
        std::string requirement;
        bool baryons;
        bool neutrinos;
        std::function<bool(const SimulationParameters &)> satisfied;
    };
    inline std::vector<SpeciesRequirement> species_requirements();

    //=============================================================================
    /// All the parameters of the simulation
    //=============================================================================
    struct SimulationParameters {
        SimulationSettings simulation;
        ForceSettings force;
        ICSettings ic;
        ParticleSettings particle;
        TimestepSettings timestep;
        FoFSettings fof;
        BaryonificationSettings baryonification;
        PofkSettings pofk;
        PofkMultipoleSettings pofk_multipole;
        BispectrumSettings bispectrum;
        HybridBiasSettings hybridbias;
        OutputSettings output;
        LightconeSettings lightcone;
        bool old_input_type_name{false}; // ic_type_of_input = transferinfofile (the old name of camb)

        /// Read all the parameters, check them and print them. growth_is_scaledependent: of the gravity model (for
        /// simulation_use_scaledependent_cola). The checks that depend on the models: check_parameters_against_models
        /// (ModelCompatibility.h)
        void read(const ParameterMap & param, bool growth_is_scaledependent) {
            ParameterReader r(param);

            // Every parameter is read by the fields() of its settings: its value in the parameterfile (or its default
            // if its condition in the table holds) or else the default of the table, so the conditions and the defaults
            // are only in fields()
            FieldReader f(r);
            simulation.fields(f);

            force.fields(f);
            if (force.interpolation_method == "fiducial")
                force.interpolation_method = force.density_method == "nufft" ? "nufft" : "assignment";

            ic.fields(f);
            old_input_type_name = param.get<std::string>("ic_type_of_input") == "transferinfofile";

            particle.fields(f);
            if (particle.neutrinos and particle.neutrino_Npart_1D <= 0)
                particle.neutrino_Npart_1D = particle.Npart_1D;

            timestep.fields(f);
            fof.fields(f);
            baryonification.read(r);
            pofk.fields(f);
            pofk_multipole.fields(f);
            bispectrum.fields(f);
            hybridbias.fields(f);
            output.fields(f);
            lightcone.fields(f);

            validate();

            // Scale-dependent COLA only when the growth is scale-dependent (massive neutrinos, k-dependent gravity):
            // otherwise it gives the same result as standard COLA at a much higher cost (4x the time and 40% more
            // memory for 64^3 in GR)
            if (simulation.use_scaledependent_cola and not(simulation.use_cola and growth_is_scaledependent)) {
                simulation.use_scaledependent_cola = false;
                r.update("simulation_use_scaledependent_cola", false);
                if (FML::ThisTask == 0 and simulation.use_cola)
                    std::cout << "NB: simulation_use_scaledependent_cola = true, but the growth is scale-independent: "
                                 "using standard COLA (simulation_use_scaledependent_cola = false)\n";
            }

            r.print("NBodySimulation::read_parameters");
            print_warnings();
        }

        /// Warnings about settings that are allowed but probably not what is wanted (task 0)
        void print_warnings() const {
            if (old_input_type_name and FML::ThisTask == 0)
                std::cout << "WARNING: ic_type_of_input = transferinfofile is the old name of ic_type_of_input = camb\n";

            // The accurate force options (NUFFT, deconvolution, early-time force) remove the smoothing of the force
            // near the grid Nyquist frequency. With a grid finer than the particle grid the modes between the particle
            // and the grid Nyquist frequency are dominated by the discreteness of the particles and these options then
            // give far too much small-scale power (+15-60% at k = 0.2 h/Mpc in tests with a 2x grid)
            const bool exact_force = force.density_method != "assignment" or force.deconvolve_window or
                                     force.interpolation_method == "nufft" or force.early_method != "none";
            if (exact_force and force.nmesh > particle.Npart_1D and FML::ThisTask == 0)
                std::cout << "WARNING: force_density_method / force_deconvolve_window / force_early_method are meant for "
                             "force_nmesh = particle_Npart_1D. With a finer force grid they amplify the discreteness of the "
                             "particles (see README)\n";

            // Starting from the Big Bang the force at early times (particles close to the lattice) has to be accurate: the
            // CIC and interlaced forces make the modes with k > ~k_Ny/8 grow wrongly there (see README)
            if (ic.initial_redshift > 100.0 and force.early_method == "none" and
                force.density_method == "assignment" and force.interpolation_method != "nufft" and
                FML::ThisTask == 0)
                std::cout << "WARNING: starting at z = " << ic.initial_redshift
                          << " without force_early_method: use force_early_method = nufft (force_early_until_redshift ~ 10) "
                             "for an accurate force while the particles are close to the lattice (see README)\n";

            // The baryon-CDM relative mode in the displacements of the particles (a ~50 kpc/h CDM-baryon offset per pair)
            // gets a spurious discrete-mode growth once the force grid resolves it, and is partly lost in collapsed regions
            // (P(delta_c - delta_b) ~ 0.5 of linear theory at z = 0 on large scales). See README
            // (also the part of it that grows after zini with mass_ini)
            if (particle.baryons and particle.baryon_relative_mode != "mass" and
                force.nmesh > 2 * particle.Npart_1D and FML::ThisTask == 0)
                std::cout << "WARNING: particle_baryon_relative_mode = " << particle.baryon_relative_mode
                          << " with force_nmesh > 2 particle_Npart_1D: the baryon-CDM relative mode is not reliable (use "
                             "particle_baryon_relative_mode = mass, see README)\n";
        }

        /// Check that the parameters are consistent. Throws if not
        void validate() {
            auto fail = [](const std::string & msg) { throw std::runtime_error("Invalid parameters: " + msg); };

            if (baryonification.enabled) {
                baryonification.validate();
                if (particle.baryons || particle.neutrinos)
                    fail("baryonification currently needs a single CDM+baryon particle species");
                if (!pofk.enabled)
                    fail("baryonification needs pofk = true (uses its mesh, assignment and shot-noise settings)");
#ifndef USE_HDF5
                if (baryonification.output_particles && output.fileformat == ParticleFileFormat::HDF5)
                    fail("baryonification_output_particles with HDF5 needs USE_HDF5=true");
#endif
            }

            // COLA
            if (simulation.use_cola and simulation.cola_LPT_order > ic.LPT_order)
                fail("The highest COLA LPT order is ic_LPT_order (i.e. simulation_cola_LPT_order must be <= "
                     "ic_LPT_order)");
            // COLA has the 1LPT and 2LPT frames (the IC can still be 3LPT)
            if (simulation.use_cola and simulation.cola_LPT_order > 2)
                fail("simulation_cola_LPT_order must be 1 or 2 (COLA is implemented with the 1LPT and 2LPT frames; "
                     "the IC can be 3LPT)");
            if (simulation.use_cola and simulation.cola_LPT_order < 1) {
                // Without COLA nothing would move the particles (no PM force and no LPT trajectories)
                if (simulation.enforce_LPT_trajectories)
                    fail("simulation_enforce_LPT_trajectories = true needs COLA (simulation_cola_LPT_order >= 1)");
                simulation.use_cola = simulation.use_scaledependent_cola = false;
                if (FML::ThisTask == 0)
                    std::cout << "NB: simulation_cola_LPT_order < 1 so turning off COLA\n";
            }
            if (fof.enabled) {
                if (fof.output_fileformat != "ascii" and fof.output_fileformat != "HDF5")
                    fail("fof_output_fileformat must be ascii or HDF5");
#ifndef USE_HDF5
                if (fof.output_fileformat == "HDF5")
                    fail("fof_output_fileformat = HDF5 needs the code compiled with USE_HDF5=true");
#endif
                for (auto & def : fof.so_definitions) {
                    try {
                        parse_so_definition(def);
                    } catch (std::exception & e) {
                        fail(e.what());
                    }
                }
                if (not(fof.massfunction_Mmin > 0.0 and fof.massfunction_Mmax > fof.massfunction_Mmin))
                    fail("fof_massfunction_Mmin and fof_massfunction_Mmax must be 0 < fof_massfunction_Mmin < "
                         "fof_massfunction_Mmax");
            }
#ifndef USE_HDF5
            if (output.particles and output.fileformat == ParticleFileFormat::HDF5)
                fail("output_fileformat = HDF5 needs HDF5: compile with USE_HDF5=true (make USE_HDF5=true)");
#endif
            // (the grid sizes and bin counts, ellmax, ... are checked in fields())
            // The sheet density needs the particles on their lattice (the Lagrangian coordinates)
            if (force.density_method == "sheet" or force.density_method == "sheet_nufft") {
                if (ic.random_field_type == ICRandomFieldType::ReadParticles)
                    fail("force_density_method = sheet needs particles that start on a regular lattice (not "
                         "ic_random_field_type = read_particles)");
            }
            if (hybridbias.enabled) {
                if (ic.random_field_type == ICRandomFieldType::ReadParticles)
                    fail("hybridbias needs the linear density field of the initial conditions and particles that start "
                         "on a regular lattice (not ic_random_field_type = read_particles)");
                if ((hybridbias.zeldovich or hybridbias.oneloop) and hybridbias.kcut_over_nyquist <= 0.0)
                    fail("hybridbias_zeldovich and hybridbias_oneloop need an isotropic cut of the operators "
                         "(hybridbias_kcut_over_nyquist > 0)");
                if (hybridbias.oneloop and hybridbias.smoothing_scale > 0.0)
                    fail("hybridbias_oneloop uses the same filter for the operators and the displacement, so it cannot "
                         "be used with hybridbias_smoothing_scale > 0");
            }
            if (force.interlacing_order > 1) {
                if (force.density_method != "assignment" or force.interpolation_method != "assignment")
                    fail("force_interlacing_order > 1 needs force_density_method = assignment and "
                         "force_interpolation_method = assignment (the NUFFT has no aliasing to remove)");
                try {
                    FML::INTERPOLATION::generate_interlacing_shifts<3>(force.interlacing_order, 1.0);
                } catch (std::exception & e) {
                    fail("force_interlacing_order = " + std::to_string(force.interlacing_order) +
                         " is not supported (use 1, 2, 4, 8, 16, 27, 32, 64, ...)");
                }
            }
            if (ic.input_filename.empty() and ic.type_of_input != ICInputType::InternalTransfer and
                not ic.reads_particles())
                fail("ic_input_filename is needed for ic_type_of_input = " + ic.type_of_input_name());
            // N-body gauge: the IC must be the N-body gauge transfer functions at the initial redshift (internal: the
            // solver gives them; camb: the tables of scripts/make_nbody_gauge_transfer.py)
            if (force.nbody_gauge) {
                if (ic.type_of_input != ICInputType::CambFiles and not ic.from_internal_solver())
                    fail("force_nbody_gauge = true needs ic_type_of_input = internal or tables (the N-body gauge "
                         "from the Einstein-Boltzmann solver) or camb (the N-body gauge transfer functions made by "
                         "scripts/make_nbody_gauge_transfer.py)");
                if (ic.type_of_input == ICInputType::CambFiles and
                    std::fabs(ic.input_redshift - ic.initial_redshift) > 1e-6)
                    fail("force_nbody_gauge = true needs ic_input_redshift = ic_initial_redshift (scaling the "
                         "transfer functions with the growth factor would not be in the N-body gauge)");
                if (force.linear_massive_neutrinos) {
                    force.linear_massive_neutrinos = false;
                    if (FML::ThisTask == 0)
                        std::cout << "NB: force_linear_massive_neutrinos is ignored with force_nbody_gauge = true "
                                     "(the N-body gauge source already includes the massive neutrinos)\n";
                }
            }
            if (simulation.use_cola and timestep.method == TimestepMethod::BullFrog)
                fail("timestep_method = BullFrog already follows 2LPT and cannot be combined with COLA "
                     "(simulation_use_cola = false)");

#if !defined(NDIM_SPACE) || NDIM_SPACE == 3
            // Healpix_Map uses the int Healpix_Base (nside <= 2^13; npix = 12 nside^2 overflowed when the first map
            // was allocated at the first lightcone step). Checked here, before the check for Healpix (in 2D the maps
            // are not Healpix maps and nside is the number of angular bins)
            const int nside_max = 8192;
            const bool maps = lightcone.enabled and lightcone.make_onion_density_maps;
            for (auto [name, nside] :
                 {std::pair<std::string, int>{"plc_nside", maps ? lightcone.nside : 0},
                  std::pair<std::string, int>{"plc_nside_chunks",
                                              maps and lightcone.use_chunkpix ? lightcone.nside_chunks : 0}})
                if (nside > nside_max)
                    fail(name + " = " + std::to_string(nside) + " must be <= " + std::to_string(nside_max) +
                         " (the Healpix maps use int pixel indices: npix = 12 nside^2)");
#ifndef USE_HEALPIX
            if (maps)
                fail("plc_make_onion_density_maps = true needs Healpix: compile with USE_HEALPIX=true (make "
                     "USE_HEALPIX=true)");
#endif
#endif
#ifndef USE_HDF5
            if (lightcone.enabled and lightcone.output_hdf5file)
                fail("plc_output_hdf5file = true needs HDF5: compile with USE_HDF5=true (make USE_HDF5=true)");
#endif

            // Output redshifts: decreasing and not before the start of the simulation
            // (a first output within 1e-3 of the initial redshift is taken to be at the initial redshift)
            auto & z = output.redshifts;
            if (z.empty())
                fail("output_redshifts is empty");
            if (std::fabs(z[0] - ic.initial_redshift) < 1e-3)
                z[0] = ic.initial_redshift;
            if (z[0] > ic.initial_redshift)
                fail("The first output cannot be before the simulation starts (output_redshifts[0] > "
                     "ic_initial_redshift)");
            for (size_t i = 1; i < z.size(); i++)
                if (not(z[i] < z[i - 1]))
                    fail("output_redshifts must be strictly decreasing");

            // Steps: either the total number of steps or the number of steps for every output
            auto & n = timestep.nsteps;
            if (n.size() != 1 and n.size() != z.size())
                fail("timestep_nsteps must have one entry (total number of steps) or one entry per output redshift "
                     "(it has " +
                     std::to_string(n.size()) + ", there are " + std::to_string(z.size()) + " outputs)");
            for (auto steps : n)
                if (steps < 0)
                    fail("timestep_nsteps cannot be negative");
            // An output with no steps would write the particles of the previous output (or the IC) at another z
            if (n.size() == z.size())
                for (size_t i = 0; i < n.size(); i++)
                    if (n[i] == 0 and (i > 0 or z[0] != ic.initial_redshift))
                        fail("timestep_nsteps: output " + std::to_string(i) + " (z = " + std::to_string(z[i]) +
                             ") has 0 steps (only an output at ic_initial_redshift can have 0 steps)");
            if (n.size() == 1 and n[0] == 0 and (z.size() > 1 or z[0] != ic.initial_redshift))
                fail("timestep_nsteps = 0 but there are outputs after ic_initial_redshift");
            // Tassev: (a_h^n - a_l^n) / n is 0/0 for n = 0
            if (timestep.method == TimestepMethod::Tassev and timestep.cola_nLPT == 0.0)
                fail("timestep_cola_nLPT cannot be 0 with timestep_method = Tassev");
            if (timestep.spacing == ScaleFactorSpacing::Powerlaw and timestep.spacing_power == 0.0)
                fail("timestep_spacing_power cannot be 0 (use logarithmic spacing)");

            // The mass-encoded relative mode: CDM and baryons start from the same lattice (the masses compensate)
            if (particle.baryons and particle.baryon_relative_mode != "displacement" and
                particle.baryon_lattice != "same")
                fail("particle_baryon_relative_mode = " + particle.baryon_relative_mode +
                     " needs particle_baryon_lattice = same");

            // What is implemented for the particle species beyond one set of CDM+baryon particles
            for (auto & req : species_requirements()) {
                if (req.satisfied(*this))
                    continue;
                if (particle.baryons and req.baryons)
                    fail("particle_baryons = true needs " + req.requirement + " (not implemented for two species yet)");
                if (particle.neutrinos and req.neutrinos)
                    fail("particle_neutrinos = true needs " + req.requirement +
                         " (not implemented for neutrino particles yet)");
            }
        }
    };

    inline std::vector<SpeciesRequirement> species_requirements() {
        using P = const SimulationParameters &;
        return {
            {"`timestep_method = Quinn`", false, true, [](P p) { return p.timestep.method == TimestepMethod::Quinn; }},
            {"`simulation_use_cola = false`", false, true, [](P p) { return not p.simulation.use_cola; }},
            {"`timestep_method` other than `Tassev` with COLA (the modified Tassev operators get the baryon-CDM "
             "relative "
             "velocity wrong: +30% in P_b-c with 20 linear steps)",
             true,
             false,
             [](P p) { return not p.simulation.use_cola or p.timestep.method != TimestepMethod::Tassev; }},
            {"`simulation_cola_LPT_order` <= `ic_LPT_order` (with COLA)",
             true,
             false,
             [](P p) { return not p.simulation.use_cola or p.simulation.cola_LPT_order <= p.ic.LPT_order; }},
            {"`simulation_enforce_LPT_trajectories = false`",
             true,
             true,
             [](P p) { return not p.simulation.enforce_LPT_trajectories; }},
            {"`ic_type_of_input = camb`, `class`, `internal` or `tables` (species transfers at zini)",
             true,
             true,
             [](P p) { return p.ic.has_transfer_tables(); }},
            {"`ic_random_field_type = gaussian` or `nongaussian`",
             true,
             true,
             [](P p) {
                 return p.ic.random_field_type == ICRandomFieldType::Gaussian or
                        p.ic.random_field_type == ICRandomFieldType::NonGaussian;
             }},
            {"`ic_LPT_order = 1` or `2`", true, false, [](P p) { return p.ic.LPT_order <= 2; }},
            {"`force_linear_massive_neutrinos = false` (the particles replace the linear neutrinos)",
             false,
             true,
             [](P p) { return not p.force.linear_massive_neutrinos; }},
            {"`force_density_method = assignment`",
             true,
             true,
             [](P p) { return p.force.density_method == "assignment"; }},
            {"`force_interpolation_method` other than `nufft`",
             true,
             true,
             [](P p) { return p.force.interpolation_method != "nufft"; }},
            {"`force_interlacing_order = 1`", true, true, [](P p) { return p.force.interlacing_order <= 1; }},
            {"`force_early_method = none`", true, true, [](P p) { return p.force.early_method == "none"; }},
            {"`force_nbody_gauge = false`", true, true, [](P p) { return not p.force.nbody_gauge; }},
            {"`lightcone = false`", true, true, [](P p) { return not p.lightcone.enabled; }},
            {"`pofk_multipole = false`", true, true, [](P p) { return not p.pofk_multipole.enabled; }},
            {"`bispectrum = false`", true, true, [](P p) { return not p.bispectrum.enabled; }},
            {"`hybridbias = false`", true, false, [](P p) { return not p.hybridbias.enabled; }},
        };
    }

    /// The table of what the particle species need (markdown, for PARAMETERS.md)
    inline void print_species_requirements(std::ostream & os) {
        os << "| Needs | `particle_baryons` | `particle_neutrinos` |\n";
        os << "|---|---|---|\n";
        for (auto & req : species_requirements())
            os << "| " << req.requirement << " | " << (req.baryons ? "yes" : "") << " | "
               << (req.neutrinos ? "yes" : "") << " |\n";
    }

} // namespace SIMCODE

#endif
