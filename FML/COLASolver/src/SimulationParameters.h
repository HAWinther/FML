#ifndef SIMULATIONPARAMETERS_HEADER
#define SIMULATIONPARAMETERS_HEADER

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
    enum class TimestepMethod { Quinn, Tassev, FastPM, BullFrog };
    enum class ScaleFactorSpacing { Linear, Logarithmic, Powerlaw, GrowthFactor };
    enum class ParticleFileFormat { GADGET, HDF5 };

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
        bool nbody_gauge{false};               // Add the linear N-body gauge source (radiation, neutrinos, GR)
        std::string nbody_gauge_filename;      // The table T_extra(k,z) (from scripts/make_nbody_gauge_transfer.py)
        std::string greens_function_kernel;    // Kernel used for the Greens function of the Poisson equation
        std::string gradient_kernel;           // Kernel used for computing the gradient of the potential
        std::string density_method{"assignment"}; // assignment, nufft, sheet or sheet_nufft
        double nufft_epsilon{1e-5};            // Accuracy of the NUFFT (density_method = nufft)
        int sheet_refinement{2};               // R: R^NDIM tracers per particle (density_method = sheet)
        std::string interpolation_method{"assignment"}; // Force to particles: assignment (kernel) or nufft
        int interlacing_order{1};              // Average the force over this many shifted copies of the particles
        bool deconvolve_window{false};         // Divide the force by W(k)^2 of the assignment kernel
        // Early-time force: a more accurate (and expensive) force before force_early_until_redshift
        std::string early_method{"none"};      // none, nufft or interlaced
        double early_until_redshift{10.0};     // Use the early force for steps with z > this
        int early_interlacing_order{2};        // early_method = interlaced: number of shifted grids
        std::string early_density_assignment_method{"TSC"}; // early_method = interlaced: assignment kernel
    };

    struct ICSettings {
        ICRandomFieldType random_field_type;  // gaussian, nongaussian, read_particles, read_phases
        double initial_redshift;              // The initial redshift of the sim
        int nmesh;                            // The Nmesh used to generate the IC (use particle_Npart_1D)
        std::string lattice_offset{"fiducial"}; // Offset of the particle lattice from the force grid nodes
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
        bool baryons{false};      // CDM and baryons as two sets of particles (otherwise one set "cb")
        std::string baryon_lattice{"same"}; // same: the baryons start from the CDM lattice, offset: half a spacing
        bool neutrinos{false};    // Massive neutrinos as particles (delta-f method)
        int neutrino_Npart_1D{0}; // Number of neutrino particles per dimension (0: particle_Npart_1D)
        bool neutrino_deltaf{true}; // Use the delta-f weights (false: plain particles, for testing)
        bool neutrino_energy_density{false}; // Source the Poisson equation with the neutrino energy density
        int sort_every_nsteps{0};   // Reorder the particles in memory by position every this many steps (0: never)
    };

    struct TimestepSettings {
        std::vector<int> nsteps;    // Steps between the outputs (if just 1 number the total amount of steps)
        TimestepMethod method;      // Quinn, Tassev (for COLA mainly), FastPM or BullFrog (see TimeStepping.h)
        std::string algorithm;      // KDK - Kick-Drift-Kick is the main method we use
        double cola_nLPT;           // nLPT. For method = Tassev
        ScaleFactorSpacing spacing; // linear, logarithmic, powerlaw, growthfactor
        double spacing_power{1.0};  // For spacing = powerlaw
    };

    struct FoFSettings {
        bool enabled;              // Locate halos when we output
        int nmin_per_halo;         // Minimum particles per halo
        int nmesh_max;             // For speeding it up: the maximum gridsize to bin particle to
        double linking_length;     // The linking length in units of the mean partice separation (i.e. ~0.2)
        double buffer_length_mpch; // The buffer region (from nbor tasks) we use in the FoF finding
        std::string output_fileformat{"ascii"}; // The halo catalogue: ascii or HDF5
        int massfunction_nbins{30};             // The mass function: number of bins...
        double massfunction_Mmin{1e10};         // ...and the range (Msun/h)
        double massfunction_Mmax{1e16};
        std::vector<std::string> so_definitions; // Spherical overdensity masses around the halos (e.g. 200m, 500c, vir)
    };

    struct PofkSettings {
        bool enabled;                          // Compute power-spectrum when we output
        int nmesh;                             // The grid size to use for this
        bool interlacing;                      // Use interlacing for alias reduction?
        bool subtract_shotnoise{false};        // Subtract shotnoise? (also used for P(k) every step)
        bool every_step{true};                 // P(k) of the force density field at every step
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
    };

    struct OutputSettings {
        std::vector<double> redshifts; // List of output redshift from large to small
        bool particles;                // Output particles?
        ParticleFileFormat fileformat; // Fileformat for particles
        std::string folder;            // Folder to store output ("" is the current folder)
        bool lightcone{false};         // Make a lightcone (the settings are read by the Lightcone class)

        /// Path for a file in the output folder
        std::string path(const std::string & filename) const { return folder + (folder == "" ? "" : "/") + filename; }
    };

    struct SimulationParameters;

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
        PofkSettings pofk;
        PofkMultipoleSettings pofk_multipole;
        BispectrumSettings bispectrum;
        HybridBiasSettings hybridbias;
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
            r.get("force_nbody_gauge", force.nbody_gauge);
            if (force.nbody_gauge)
                r.get("force_nbody_gauge_filename", force.nbody_gauge_filename);
            r.get("force_greens_function_kernel", force.greens_function_kernel);
            r.get("force_gradient_kernel", force.gradient_kernel);
            r.get("force_density_method", force.density_method, std::string("assignment"));
            if (force.density_method == "nufft" or force.density_method == "sheet_nufft")
                r.get("force_nufft_epsilon", force.nufft_epsilon, 1e-5);
            if (force.density_method == "sheet" or force.density_method == "sheet_nufft")
                r.get("force_sheet_refinement", force.sheet_refinement, 2);
            r.get("force_interlacing_order", force.interlacing_order, 1);
            r.get("force_deconvolve_window", force.deconvolve_window, false);
            r.get("force_early_method", force.early_method, std::string("none"));
            if (force.early_method != "none") {
                r.get("force_early_until_redshift", force.early_until_redshift, 10.0);
                if (force.early_method == "interlaced") {
                    r.get("force_early_interlacing_order", force.early_interlacing_order, 2);
                    r.get("force_early_density_assignment_method",
                          force.early_density_assignment_method,
                          std::string("TSC"));
                }
            }
            r.get("force_interpolation_method", force.interpolation_method, std::string("fiducial"));
            if (force.interpolation_method == "fiducial")
                force.interpolation_method = force.density_method == "nufft" ? "nufft" : "assignment";
            if (force.interpolation_method == "nufft")
                r.get("force_nufft_epsilon", force.nufft_epsilon, 1e-5);

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
            r.get("ic_lattice_offset", ic.lattice_offset);
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
            r.get("particle_sort_every_nsteps", particle.sort_every_nsteps);
            r.get("particle_baryons", particle.baryons);
            if (particle.baryons)
                r.get("particle_baryon_lattice", particle.baryon_lattice);
            r.get("particle_neutrinos", particle.neutrinos);
            if (particle.neutrinos) {
                r.get("particle_neutrino_Npart_1D", particle.neutrino_Npart_1D);
                r.get("particle_neutrino_deltaf", particle.neutrino_deltaf);
                r.get("particle_neutrino_energy_density", particle.neutrino_energy_density);
                if (particle.neutrino_Npart_1D <= 0)
                    particle.neutrino_Npart_1D = particle.Npart_1D;
            }

            // Time-stepping
            r.get("timestep_nsteps", timestep.nsteps);
            r.get_choice("timestep_method",
                         timestep.method,
                         {{"Quinn", TimestepMethod::Quinn},
                          {"Tassev", TimestepMethod::Tassev},
                          {"FastPM", TimestepMethod::FastPM},
                          {"BullFrog", TimestepMethod::BullFrog}});
            r.get("timestep_algorithm", timestep.algorithm);
            r.get("timestep_cola_nLPT", timestep.cola_nLPT);
            r.get_choice("timestep_scalefactor_spacing",
                         timestep.spacing,
                         {{"linear", ScaleFactorSpacing::Linear},
                          {"logarithmic", ScaleFactorSpacing::Logarithmic},
                          {"powerlaw", ScaleFactorSpacing::Powerlaw},
                          {"growthfactor", ScaleFactorSpacing::GrowthFactor}});
            if (timestep.spacing == ScaleFactorSpacing::Powerlaw)
                r.get("timestep_spacing_power", timestep.spacing_power);

            // FoF halofinding
            r.get("fof", fof.enabled);
            if (fof.enabled) {
                r.get("fof_nmin_per_halo", fof.nmin_per_halo);
                r.get("fof_linking_length", fof.linking_length);
                r.get("fof_nmesh_max", fof.nmesh_max);
                r.get("fof_buffer_length_mpch", fof.buffer_length_mpch);
                r.get("fof_output_fileformat", fof.output_fileformat, std::string("ascii"));
                r.get("fof_massfunction_nbins", fof.massfunction_nbins, 30);
                r.get("fof_massfunction_Mmin", fof.massfunction_Mmin, 1e10);
                r.get("fof_massfunction_Mmax", fof.massfunction_Mmax, 1e16);
                r.get("fof_so_definitions", fof.so_definitions, std::vector<std::string>{});
            }

            // Power-spectrum
            r.get("pofk", pofk.enabled);
            r.get("pofk_every_step", pofk.every_step);
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

            // Hybrid bias spectra
            r.get("hybridbias", hybridbias.enabled);
            if (hybridbias.enabled) {
                r.get("hybridbias_nmesh", hybridbias.nmesh);
                r.get("hybridbias_noperators", hybridbias.noperators);
                r.get("hybridbias_density_assignment_method", hybridbias.density_assignment_method);
                r.get("hybridbias_interlacing", hybridbias.interlacing);
                r.get("hybridbias_subtract_shotnoise", hybridbias.subtract_shotnoise);
                r.get("hybridbias_smoothing_scale", hybridbias.smoothing_scale);
                r.get("hybridbias_smoothing_filter", hybridbias.smoothing_filter);
                r.get("hybridbias_kcut_over_nyquist", hybridbias.kcut_over_nyquist);
                r.get("hybridbias_zeldovich", hybridbias.zeldovich);
                r.get("hybridbias_oneloop", hybridbias.oneloop);
            }

            // Output
            r.get("output_redshifts", output.redshifts);
            r.get("output_particles", output.particles);
            if (output.particles) {
                r.get_choice("output_fileformat",
                             output.fileformat,
                             {{"GADGET", ParticleFileFormat::GADGET},
                              {"HDF5", ParticleFileFormat::HDF5}});
            }
            r.get("output_folder", output.folder);
            r.get("lightcone", output.lightcone, false);

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
#ifndef USE_HDF5
            if (fof.enabled) {
                if (fof.output_fileformat != "ascii" and fof.output_fileformat != "HDF5")
                    fail("fof_output_fileformat must be ascii or HDF5");
#ifndef USE_HDF5
                if (fof.output_fileformat == "HDF5")
                    fail("fof_output_fileformat = HDF5 needs the code compiled with USE_HDF5=true");
#endif
                for (auto & def : fof.so_definitions) {
                    bool ok = def == "vir";
                    if (not ok and def.size() >= 2 and (def.back() == 'm' or def.back() == 'b' or def.back() == 'c')) {
                        try {
                            ok = std::stod(def.substr(0, def.size() - 1)) > 0.0;
                        } catch (...) {
                            ok = false;
                        }
                    }
                    if (not ok)
                        fail("fof_so_definitions: unknown definition [" + def +
                             "] (<Delta>m or <Delta>b: mean CDM+baryon density, <Delta>c: critical density, vir)");
                }
                if (not(fof.massfunction_nbins > 0 and fof.massfunction_Mmin > 0.0 and
                        fof.massfunction_Mmax > fof.massfunction_Mmin))
                    fail("fof_massfunction_nbins must be > 0 and 0 < fof_massfunction_Mmin < fof_massfunction_Mmax");
            }
            if (output.particles and output.fileformat == ParticleFileFormat::HDF5)
                fail("output_fileformat = HDF5 needs HDF5: compile with USE_HDF5=true (make USE_HDF5=true)");
#endif
            // N-body gauge: the IC must be the N-body gauge transfer functions at the initial redshift
            if (force.density_method == "sheet" or force.density_method == "sheet_nufft") {
                if (ic.random_field_type == ICRandomFieldType::ReadParticles)
                    fail("force_density_method = sheet needs particles that start on a regular lattice (not "
                         "ic_random_field_type = read_particles)");
                if (force.sheet_refinement < 1)
                    fail("force_sheet_refinement must be >= 1");
            }
            if (hybridbias.enabled) {
                if (ic.random_field_type == ICRandomFieldType::ReadParticles)
                    fail("hybridbias needs the linear density field of the initial conditions and particles that start "
                         "on a regular lattice (not ic_random_field_type = read_particles)");
                if (hybridbias.noperators < 1 or hybridbias.noperators > 5)
                    fail("hybridbias_noperators must be 1, ..., 5 (1, delta, delta^2, s^2, nabla^2 delta)");
                if (hybridbias.nmesh <= 0)
                    fail("hybridbias_nmesh must be > 0");
                if (hybridbias.smoothing_scale < 0.0)
                    fail("hybridbias_smoothing_scale must be >= 0");
                if (hybridbias.kcut_over_nyquist < 0.0)
                    fail("hybridbias_kcut_over_nyquist must be >= 0");
                if ((hybridbias.zeldovich or hybridbias.oneloop) and hybridbias.kcut_over_nyquist <= 0.0)
                    fail("hybridbias_zeldovich and hybridbias_oneloop need an isotropic cut of the operators "
                         "(hybridbias_kcut_over_nyquist > 0)");
                if (hybridbias.oneloop and hybridbias.smoothing_scale > 0.0)
                    fail("hybridbias_oneloop uses the same filter for the operators and the displacement, so it cannot "
                         "be used with hybridbias_smoothing_scale > 0");
            }
            if (force.early_method == "interlaced" and force.early_interlacing_order < 2)
                fail("force_early_interlacing_order must be >= 2");
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
            if (not(force.nufft_epsilon > 0.0 and force.nufft_epsilon < 0.1))
                fail("force_nufft_epsilon must be in (0, 0.1)");
            if (force.nbody_gauge) {
                if (ic.type_of_input != ICInputType::TransferInfoFile)
                    fail("force_nbody_gauge = true needs ic_type_of_input = transferinfofile (the N-body gauge "
                         "transfer functions made by scripts/make_nbody_gauge_transfer.py)");
                if (std::fabs(ic.input_redshift - ic.initial_redshift) > 1e-6)
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
            {"`timestep_method` other than `Tassev` with COLA (the modified Tassev operators get the baryon-CDM relative "
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
            {"`ic_type_of_input = transferinfofile` (the transfer functions of the species at zini)",
             true,
             true,
             [](P p) { return p.ic.type_of_input == ICInputType::TransferInfoFile; }},
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
            {"`force_density_method = assignment`", true, true, [](P p) { return p.force.density_method == "assignment"; }},
            {"`force_interpolation_method` other than `nufft`",
             true,
             true,
             [](P p) { return p.force.interpolation_method != "nufft"; }},
            {"`force_interlacing_order = 1`", true, true, [](P p) { return p.force.interlacing_order <= 1; }},
            {"`force_early_method = none`", true, true, [](P p) { return p.force.early_method == "none"; }},
            {"`force_nbody_gauge = false`", true, true, [](P p) { return not p.force.nbody_gauge; }},
            {"`lightcone = false`", true, true, [](P p) { return not p.output.lightcone; }},
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
            os << "| " << req.requirement << " | " << (req.baryons ? "yes" : "") << " | " << (req.neutrinos ? "yes" : "")
               << " |\n";
    }

} // namespace COLASOLVER

#endif
