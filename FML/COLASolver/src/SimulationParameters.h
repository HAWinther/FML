#ifndef SIMULATIONPARAMETERS_HEADER
#define SIMULATIONPARAMETERS_HEADER

#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// The parameters of an N-body simulation, grouped by what they are used for.
// Every setting is read (and printed) in one place: SimulationParameters::read
// and the parameters are checked for consistency in SimulationParameters::validate
// so that mistakes in the parameterfile are caught before the simulation starts
//=============================================================================

namespace COLASOLVER {

    using ParameterMap = FML::UTILS::ParameterMap;

    //=============================================================================
    // Options that are given as strings in the parameterfile
    //=============================================================================
    enum class ICRandomFieldType { Gaussian, NonGaussian, ReadParticles, ReadPhases };
    enum class ICInputType { PowerSpectrum, TransferFunction, TransferInfoFile };
    enum class RandomGeneratorType { GSL, MT19937 };
    enum class TimestepMethod { Quinn, Tassev };
    enum class ScaleFactorSpacing { Linear, Logarithmic, Powerlaw };
    enum class ParticleFileFormat { GADGET, FML };

    //=============================================================================
    /// Reads parameters from a ParameterMap and remembers what was read so that
    /// it can all be printed at the end
    //=============================================================================
    class ParameterReader {
      public:
        explicit ParameterReader(const ParameterMap & param) : param(param) {}

        /// Read a required parameter
        template <class V>
        void get(const std::string & name, V & value) {
            value = param.get<V>(name);
            record(name, value);
        }

        /// Read an optional parameter
        template <class V>
        void get(const std::string & name, V & value, const V & default_value) {
            value = param.get<V>(name, default_value);
            record(name, value);
        }

        /// Read a string parameter that must be one of the given options
        template <class E>
        void get_choice(const std::string & name, E & value, const std::vector<std::pair<std::string, E>> & options) {
            const auto str = param.get<std::string>(name);
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
    // The groups of parameters
    //=============================================================================

    struct SimulationSettings {
        std::string name;                     // The name of sim used for outputs
        double boxsize;                       // The boxsize in Mpc/h
        bool use_cola;                        // Use the cola method?
        bool use_scaledependent_cola;         // If cola, use cola with scaledependent growth?
        bool enforce_LPT_trajectories{false}; // No PM force, particles follow LPT trajectories (requires COLA)
        int cola_LPT_order{2};                // The LPT order for COLA
    };

    struct ForceSettings {
        int nmesh;                             // The gridsize to bin particles to and compute PM forces
        std::string density_assignment_method; // Density assignment (NGP,CIC,TSC,PCS,PQS)
        bool linear_massive_neutrinos;         // Include the effects of massive neutrinos using linear theory
        std::string greens_function_kernel;    // Kernel used for the Greens function of the Poisson equation
        std::string gradient_kernel;           // Kernel used for computing the gradient of the potential
    };

    struct ICSettings {
        ICRandomFieldType random_field_type;  // gaussian, nongaussian, read_particles, read_phases
        double initial_redshift;              // The initial redshift of the sim
        int nmesh;                            // The Nmesh used to generate the IC (use particle_Npart_1D)
        int random_seed;                      // The random seed
        RandomGeneratorType random_generator; // The generator: GSL or MT19937
        int LPT_order;                        // The LPT order to use to make IC (1,2,3)

        // Input file (power-spectrum / transfer functions)
        ICInputType type_of_input;            // powerspectrum, transferfunction, transferinfofile
        std::string type_of_input_fileformat; // Format, CAMB, CLASS (with format=camb), ..., for transfer files
        std::string input_filename;           // The filename
        double input_redshift;                // The redshift of P(k,z) / T(k,z) that we read in
        bool use_gravity_model_GR;            // Input power-spectrum is for LCDM so if MG use LCDM to set the IC

        // Amplitude fixed / paired IC
        bool fix_amplitude;  // Fix amplitude of delta(k,zini) to that of P(k)
        bool reverse_phases; // Set delta(k,zini) -> -delta(k,zini)

        // Non-gaussianity (random_field_type = nongaussian)
        std::string fnl_type; // local, equilateral, orthogonal
        double fnl{0.0};      // fNL
        double fnl_redshift;  // Redshift of which to generate the primordial potential

        // Normalization using sigma8
        bool sigma8_normalization;   // Normalize wrt to a sigma8 value?
        double sigma8_redshift{0.0}; // The redshift of which to normalize
        double sigma8;               // The value of sigma8 at this redshift we want

        // Read IC from file (and reconstruct the LPT fields if COLA)
        std::string reconstruct_gadgetfilepath;     // /path/to/gadget
        std::string reconstruct_assigment_method;   // Density assignment method (NGP, CIC, ...)
        std::string reconstruct_smoothing_filter;   // Smoothing filter (tophat, sharpk, gaussian)
        double reconstruct_dimless_smoothing_scale; // Smoothing scales R/boxsize
        bool reconstruct_interlacing;               // Use interlacing (probably not)?
        bool reconstruct_exact;                     // Reconstruct the LPT fields exactly from pos+vel

        double a_initial() const { return 1.0 / (1.0 + initial_redshift); }
        bool reads_particles() const {
            return random_field_type == ICRandomFieldType::ReadParticles or
                   random_field_type == ICRandomFieldType::ReadPhases;
        }
    };

    struct ParticleSettings {
        int Npart_1D;             // Number of particles per dimension (total is N^NDIM)
        double allocation_factor; // Allocate this many more particles
    };

    struct TimestepSettings {
        std::vector<int> nsteps;    // Steps between the outputs (if just 1 number the total amount of steps)
        TimestepMethod method;      // The method of Quinn or the method of Tassev (for COLA mainly)
        std::string algorithm;      // KDK - Kick-Drift-Kick is the main method we use
        double cola_nLPT;           // nLPT. For method = Tassev
        ScaleFactorSpacing spacing; // linear, logarithmic, powerlaw
        double spacing_power{1.0};  // For spacing = powerlaw
    };

    struct FoFSettings {
        bool enabled;              // Locate halos when we output
        int nmin_per_halo;         // Minimum particles per halo
        int nmesh_max;             // For speeding it up: the maximum gridsize to bin particle to
        double linking_length;     // The linking length in units of the mean partice separation (i.e. ~0.2)
        double buffer_length_mpch; // The buffer region (from nbor tasks) we use in the FoF finding
    };

    struct PofkSettings {
        bool enabled;                          // Compute power-spectrum when we output
        int nmesh;                             // The grid size to use for this
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise{true};         // Subtract shotnoise? (also used for P(k) every step)
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)
    };

    struct PofkMultipoleSettings {
        bool enabled;                          // Compute power-spectrum multipoles when we output
        int nmesh;                             // The grid size to use for this
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract shotnoise for P0?
        int ellmax;                            // Compute P_ell for ell=0,2,4,...,ellmax
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)
    };

    struct BispectrumSettings {
        bool enabled;                          // Compute bispectrum when we output
        int nmesh;                             // The grid size to use for this
        int nbins;                             // The number of bins in k to use (memory req. scales as nbins^3)
        std::string density_assignment_method; // Density assignment method (NGP, CIC, ...)
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise;               // Subtract shotnoise?
    };

    struct OutputSettings {
        std::vector<double> redshifts; // List of output redshift from large to small
        bool particles;                // Output particles?
        ParticleFileFormat fileformat; // Fileformat for particles
        std::string folder;            // Folder to store output ("" is the current folder)

        /// Path for a file in the output folder
        std::string path(const std::string & filename) const { return folder + (folder == "" ? "" : "/") + filename; }
    };

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
        PofkSettings pofk;
        PofkMultipoleSettings pofk_multipole;
        BispectrumSettings bispectrum;
        OutputSettings output;

        /// Read all the parameters, check them and print them
        void read(const ParameterMap & param) {
            ParameterReader r(param);

            // General parameters and COLA
            r.get("simulation_name", simulation.name);
            r.get("simulation_boxsize", simulation.boxsize);
            r.get("simulation_use_cola", simulation.use_cola);
            r.get("simulation_use_scaledependent_cola", simulation.use_scaledependent_cola);
            if (simulation.use_cola) {
                r.get("simulation_enforce_LPT_trajectories", simulation.enforce_LPT_trajectories, false);
                r.get("simulation_cola_LPT_order", simulation.cola_LPT_order, 2);
            }

            // Computing forces
            r.get("force_nmesh", force.nmesh);
            r.get("force_density_assignment_method", force.density_assignment_method);
            r.get("force_linear_massive_neutrinos", force.linear_massive_neutrinos);
            r.get("force_greens_function_kernel", force.greens_function_kernel);
            r.get("force_gradient_kernel", force.gradient_kernel);

            // Initial conditions
            r.get_choice("ic_type_of_input",
                         ic.type_of_input,
                         {{"powerspectrum", ICInputType::PowerSpectrum},
                          {"transferfunction", ICInputType::TransferFunction},
                          {"transferinfofile", ICInputType::TransferInfoFile}});
            r.get("ic_type_of_input_fileformat", ic.type_of_input_fileformat, std::string("CAMB"));
            r.get("ic_input_filename", ic.input_filename);
            r.get_choice("ic_random_field_type",
                         ic.random_field_type,
                         {{"gaussian", ICRandomFieldType::Gaussian},
                          {"nongaussian", ICRandomFieldType::NonGaussian},
                          {"read_particles", ICRandomFieldType::ReadParticles},
                          {"read_phases", ICRandomFieldType::ReadPhases}});
            r.get("ic_input_redshift", ic.input_redshift);
            r.get("ic_use_gravity_model_GR", ic.use_gravity_model_GR);
            r.get("ic_initial_redshift", ic.initial_redshift);
            r.get("ic_nmesh", ic.nmesh);
            r.get("ic_random_seed", ic.random_seed);
            r.get_choice("ic_random_generator",
                         ic.random_generator,
                         {{"GSL", RandomGeneratorType::GSL}, {"MT19937", RandomGeneratorType::MT19937}});
            r.get("ic_LPT_order", ic.LPT_order);
            r.get("ic_fix_amplitude", ic.fix_amplitude);
            r.get("ic_reverse_phases", ic.reverse_phases);
            if (ic.random_field_type == ICRandomFieldType::NonGaussian) {
                r.get("ic_fnl_type", ic.fnl_type);
                r.get("ic_fnl", ic.fnl);
                r.get("ic_fnl_redshift", ic.fnl_redshift);
            }
            r.get("ic_sigma8_normalization", ic.sigma8_normalization);
            if (ic.sigma8_normalization) {
                r.get("ic_sigma8_redshift", ic.sigma8_redshift);
                r.get("ic_sigma8", ic.sigma8);
            }
            if (ic.reads_particles()) {
                r.get("ic_reconstruct_gadgetfilepath", ic.reconstruct_gadgetfilepath);
                r.get("ic_reconstruct_assigment_method", ic.reconstruct_assigment_method);
                r.get("ic_reconstruct_smoothing_filter", ic.reconstruct_smoothing_filter);
                r.get("ic_reconstruct_dimless_smoothing_scale", ic.reconstruct_dimless_smoothing_scale);
                r.get("ic_reconstruct_interlacing", ic.reconstruct_interlacing);
                r.get("ic_reconstruct_exact", ic.reconstruct_exact);
            }

            // Particles
            r.get("particle_Npart_1D", particle.Npart_1D);
            r.get("particle_allocation_factor", particle.allocation_factor);

            // Time-stepping
            r.get("timestep_nsteps", timestep.nsteps);
            r.get_choice("timestep_method",
                         timestep.method,
                         {{"Quinn", TimestepMethod::Quinn}, {"Tassev", TimestepMethod::Tassev}});
            r.get("timestep_algorithm", timestep.algorithm);
            r.get("timestep_cola_nLPT", timestep.cola_nLPT);
            r.get_choice("timestep_scalefactor_spacing",
                         timestep.spacing,
                         {{"linear", ScaleFactorSpacing::Linear},
                          {"logarithmic", ScaleFactorSpacing::Logarithmic},
                          {"powerlaw", ScaleFactorSpacing::Powerlaw}});
            if (timestep.spacing == ScaleFactorSpacing::Powerlaw)
                r.get("timestep_spacing_power", timestep.spacing_power);

            // FoF halofinding
            r.get("fof", fof.enabled);
            if (fof.enabled) {
                r.get("fof_nmin_per_halo", fof.nmin_per_halo);
                r.get("fof_linking_length", fof.linking_length);
                r.get("fof_nmesh_max", fof.nmesh_max);
                r.get("fof_buffer_length_mpch", fof.buffer_length_mpch);
            }

            // Power-spectrum
            r.get("pofk", pofk.enabled);
            if (pofk.enabled) {
                r.get("pofk_nmesh", pofk.nmesh);
                r.get("pofk_interlacing", pofk.interlacing);
                r.get("pofk_subtract_shotnoise", pofk.subtract_shotnoise);
                r.get("pofk_density_assignment_method", pofk.density_assignment_method);
            }

            // Power-spectrum multipoles
            r.get("pofk_multipole", pofk_multipole.enabled);
            if (pofk_multipole.enabled) {
                r.get("pofk_multipole_nmesh", pofk_multipole.nmesh);
                r.get("pofk_multipole_interlacing", pofk_multipole.interlacing);
                r.get("pofk_multipole_subtract_shotnoise", pofk_multipole.subtract_shotnoise);
                r.get("pofk_multipole_ellmax", pofk_multipole.ellmax);
                r.get("pofk_multipole_density_assignment_method", pofk_multipole.density_assignment_method);
            }

            // Bispectrum
            r.get("bispectrum", bispectrum.enabled);
            if (bispectrum.enabled) {
                r.get("bispectrum_nmesh", bispectrum.nmesh);
                r.get("bispectrum_nbins", bispectrum.nbins);
                r.get("bispectrum_density_assignment_method", bispectrum.density_assignment_method);
                r.get("bispectrum_interlacing", bispectrum.interlacing);
                r.get("bispectrum_subtract_shotnoise", bispectrum.subtract_shotnoise);
            }

            // Output
            r.get("output_redshifts", output.redshifts);
            r.get("output_particles", output.particles);
            if (output.particles) {
                r.get_choice("output_fileformat",
                             output.fileformat,
                             {{"GADGET", ParticleFileFormat::GADGET}, {"FML", ParticleFileFormat::FML}});
            }
            r.get("output_folder", output.folder);

            validate();
            r.print("NBodySimulation::read_parameters");
        }

        /// Check that the parameters are consistent. Throws if not
        void validate() {
            auto fail = [](const std::string & msg) { throw std::runtime_error("Invalid parameters: " + msg); };

            // COLA
            if (simulation.use_cola and simulation.cola_LPT_order > ic.LPT_order)
                fail("The highest COLA LPT order is ic_LPT_order (i.e. simulation_cola_LPT_order must be <= "
                     "ic_LPT_order)");
            if (simulation.use_cola and simulation.cola_LPT_order < 1) {
                simulation.use_cola = simulation.use_scaledependent_cola = false;
                if (FML::ThisTask == 0)
                    std::cout << "NB: simulation_cola_LPT_order < 1 so turning off COLA\n";
            }

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
            if (timestep.spacing == ScaleFactorSpacing::Powerlaw and timestep.spacing_power == 0.0)
                fail("timestep_spacing_power cannot be 0 (use logarithmic spacing)");
        }
    };

} // namespace COLASOLVER

#endif
