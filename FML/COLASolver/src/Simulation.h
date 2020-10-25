#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>

#include "AnalyzeOutput.h"
#include "COLA.h"
#include "Cosmology.h"
#include "GravityModel.h"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// Type alises we use below
//=============================================================================
using RandomGenerator = FML::RANDOM::RandomGenerator;
using GSLRandomGenerator = FML::RANDOM::GSLRandomGenerator;
using DVector = FML::INTERPOLATION::SPLINE::DVector;
using Spline = FML::INTERPOLATION::SPLINE::Spline;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
template <class T1>
using MPIParticles = FML::PARTICLE::MPIParticles<T1>;
using ParameterMap = FML::UTILS::ParameterMap;
using GadgetReader = FML::FILEUTILS::GADGET::GadgetReader;
using LinearTransferData = FML::FILEUTILS::LinearTransferData;
template <int N>
using PowerSpectrumBinning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<N>;

template <int NDIM, class T>
class NBodySimulation {
  protected:
    //=============================================================================
    /// Everything related to the background evolution
    //=============================================================================
    std::shared_ptr<Cosmology> cosmo;

    //=============================================================================
    /// Everything related to gravity: growth factors, computing forces
    //=============================================================================
    std::shared_ptr<GravityModel<NDIM>> grav;

    //=============================================================================
    /// Everything related to linear perturbations, transfer functions, initial pofk
    /// This is only used if ic_type_of_input == transferinfofile
    //=============================================================================
    std::shared_ptr<LinearTransferData> transferdata;

    //=============================================================================
    /// A copy of the simulation parameters just to have it. All parameters
    /// *should* be read in from this during read_parameters
    //=============================================================================
    std::shared_ptr<ParameterMap> parameters;

    //=============================================================================
    /// All the particles
    //=============================================================================
    MPIParticles<T> part;

    //=============================================================================
    /// The initial density field delta_cb(zini,k) computed with the same Nmesh
    /// as the forces (used for linear massive neutrinos - not allocated otherwise)
    /// NB: we cannot use phi_1LPT_ini_fourier below as it might have a different gridsize
    //=============================================================================
    FFTWGrid<NDIM> initial_density_field_fourier;

    //=============================================================================
    // LPT potentials NB: these grids will only get allocated if they are required!
    //=============================================================================
    FFTWGrid<NDIM> phi_1LPT_ini_fourier;
    FFTWGrid<NDIM> phi_2LPT_ini_fourier;
    FFTWGrid<NDIM> phi_3LPTa_ini_fourier;
    FFTWGrid<NDIM> phi_3LPTb_ini_fourier;

    // Do timings of the code
    FML::UTILS::Timings timer;

    // Splines used to generate the IC
    Spline power_primordial_spline{"P_primordial(k)"}; // Spline of primordial power-spectrum (h/Mpc) vs (Mpc/h)^3
    Spline power_initial_spline{"P(k,zini)"};          // Spline of baryon-CDM power-spectrum (h/Mpc) vs (Mpc/h)^3
    Spline transfer_initial_spline{"T(k,zini)"};       // Spline of baryon-CDM transfer function (h/Mpc) vs Mpc^2

    //=============================================================================
    // Parameters of the simulation
    //=============================================================================

    std::string simulation_name;             // The name of sim used for outputs
    double simulation_boxsize;               // The boxsize in Mpc/h
    bool simulation_use_cola;                // Use the cola method?
    bool simulation_use_scaledependent_cola; // If cola, use cola with scaledependent growth?

    // Force and density assignment
    int force_nmesh;                             // The gridsize to bin particles to and compute PM forces
    std::string force_density_assignment_method; // Density assignment (NGP,CIC,TSC,PCS,PQS)
    std::string force_kernel;                    // The force kernel (see relevant files)
    bool force_linear_massive_neutrinos;         // Include the effects of massive neutrinos using linear theory

    // Initial conditions
    std::string ic_random_field_type; // gaussian, nongaussian, reconstruct_from_particles, read_particles
    double ic_initial_redshift;       // The initial redshift of the sim
    int ic_nmesh;                     // The Nmesh used to generate the IC (use particle_Npart_1D)
    int ic_random_seed;               // The random seed
    std::string ic_random_generator;  // The generator: GSL or MT19937 (fiducial)
    int ic_LPT_order;                 // The LPT order to use to make IC (1,2,3)

    // Initial conditions: input file (power-spectrum / transfer functions)
    std::string ic_type_of_input;  // Type of input (powerspectrum, transferfuntion, transferinfofile)
    std::string ic_input_filename; // The filename
    double ic_input_redshift;      // The redshift of P(k,z) / T(k,z) that we read in
    bool ic_use_gravity_model_GR;  // Input power-spectrum is for LCDM so if MG use LCDM to set the IC

    // Initial conditions: Ampitude fixed IC
    bool ic_fix_amplitude;  // Fix amplitude of delta(k,zini) to that of P(k)
    bool ic_reverse_phases; // Set delta(k,zini) -> -delta(k,zini)

    // Initial conditions: Non-gaussianity (ic_random_field_type = nongaussian)
    std::string ic_fnl_type; // local, equilateral, orthogonal
    double ic_fnl;           // fNL
    double ic_fnl_redshift;  // Redshift of which to generate the primordial potential

    // Initial conditions: Normalization using sigma8?
    bool ic_sigma8_normalization; // Normalize wrt to a sigma8 value?
    double ic_sigma8_redshift;    // The redshift of which to normalize
    double ic_sigma8;             // The value of sigma8 at this redshift we want

    // Initial conditions: Read IC from file and reconstruct the phases
    std::string ic_reconstruct_gadgetfilepath;     // /path/to/gadget
    std::string ic_reconstruct_assigment_method;   // Density assignment method (NGP, CIC, ...)
    std::string ic_reconstruct_smoothing_filter;   // Smoothing filter (tophat, sharpk, gaussian)
    double ic_reconstruct_dimless_smoothing_scale; // Smoothing scales R/boxsize
    bool ic_reconstruct_interlacing;               // Use interlacing (probably not)?

    // Particles
    int particle_Npart_1D;             // Number of particles per dimension (total is N^3)
    double particle_allocation_factor; // Allocation this many more particles

    // Time-stepping
    std::vector<int> timestep_nsteps;         // Steps between the outputs (if just 1 number the total amount of steps)
    std::string timestep_method;              // The method of Quinn or the method of Tassev (for COLA mainly)
    std::string timestep_algorithm;           // KDK - Kick-Drift-Kick is the main method we use
    double timestep_cola_nLPT;                // nLPT. For timestep_method = Tassev
    std::string timestep_scalefactor_spacing; // linear, logarithmic, powerlaw
    double timestep_spacing_power;            // For timestep_scalefactor_spacing = powerlaw

    // FoF halofinding
    bool fof;                  // Locate halos when we output
    int fof_nmin_per_halo;     // Minimum particles per halo
    int fof_nmesh_max;         // For speeding it up: the maximum gridsize to bin particle to
    double fof_linking_length; // The linking length in units of the boxsize (i.e. ~0.2 / Npart_1D for bFoF = 0.2)

    // Power-spectrum
    bool pofk;                                  // Compute power-spectrum when we output
    int pofk_nmesh;                             // The grid size to use for this
    bool pofk_interlacing;                      // Use interlacing for alias reduction?
    bool pofk_subtract_shotnoise;               // Subtract shotnoise?
    std::string pofk_density_assignment_method; // Density assignment method (NGP, CIC, ...)

    // Power-spectrum multipoles
    bool pofk_multipole;                                  // Compute power-spectrum multipoles when we output
    int pofk_multipole_nmesh;                             // The grid size to use for this
    bool pofk_multipole_interlacing;                      // Use interlacing for alias reduction?
    bool pofk_multipole_subtract_shotnoise;               // Subtract shotnoise for P0?
    int pofk_multipole_ellmax;                            // Compute P_ell for ell=0,2,4,...,ellmax
    std::string pofk_multipole_density_assignment_method; // Density assignment method (NGP, CIC, ...)

    // Bispectrum
    bool bispectrum;                                  // Compute bispectrum when we output
    int bispectrum_nmesh;                             // The grid size to use for this
    int bispectrum_nbins;                             // The number of bins in k to use (memory req. scales as nbins^3)
    std::string bispectrum_density_assignment_method; // Density assignment method (NGP, CIC, ...)
    bool bispectrum_interlacing;                      // Use interlacing for alias reduction?
    bool bispectrum_subtract_shotnoise;               // Subtract shotnoise?

    // Output
    std::vector<double> output_redshifts; // List of output redshift from large to small
    bool output_particles;                // Output particles?
    std::string output_fileformat;        // Fileformat for particles (GADGET)
    std::string output_folder;            // Folder to store output

    //=============================================================================
    // Some of the stuff we compute and output is small so we also keep it
    // in the class in case one wants to process it later
    //=============================================================================
    // A list of (z, P(k)) for the particles at every step (naive binning - not using the same pofk_* setting!)
    std::vector<std::pair<double, PowerSpectrumBinning<NDIM>>> pofk_cb_every_step;
    std::vector<std::pair<double, PowerSpectrumBinning<NDIM>>> pofk_total_every_step;
    // A list of (z, P(k)) for the particles that we compute if pofk = true
    std::vector<std::pair<double, PowerSpectrumBinning<NDIM>>> pofk_every_output;
    // A list of (z, P_ell(k)) for the particles that we compute every output if pofk_multipoles = true
    std::vector<std::pair<double, std::vector<PowerSpectrumBinning<NDIM>>>> pofk_multipoles_every_output;

  public:
    NBodySimulation() = default;
    NBodySimulation(std::shared_ptr<Cosmology> cosmo, std::shared_ptr<GravityModel<NDIM>> grav)
        : cosmo(cosmo), grav(grav) {
        timer.StartTiming("The whole simulation");
    }

    // Move in particles from an external source
    NBodySimulation(std::shared_ptr<Cosmology> cosmo,
                    std::shared_ptr<GravityModel<NDIM>> grav,
                    MPIParticles<T> && _part)
        : NBodySimulation(cosmo, grav) {
        part = std::move(_part);
    }

    // Compute the time-steps for updates of position and velocity
    std::pair<std::vector<double>, std::vector<double>> compute_scalefactors_KDK(double amin, double amax, int nsteps);
    // Compute the delta_t to use for each of the steps
    std::pair<std::vector<double>, std::vector<double>> compute_deltatime_KDK(double amin, double amax, int nsteps);

    /// Read in and set parameters we need
    void read_parameters(ParameterMap & param);

    /// Initialize simulation. Create initial conditions. Make it ready to run
    void init();

    /// This method reconstructs delta(k,zini) and uses that to make IC from scratch
    void reconstruct_ic_from_particles(FFTWGrid<NDIM> & delta_fourier);

    /// This method simply read particles and uses them for the simulation
    void read_ic();

    /// Run simulation
    void run();

    /// From particles to density field
    void compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a);

    /// Compute stuff on the fly and output
    void analyze_and_output(int ioutput, double redshift);

    // Generation of IC (to be separated out in own file)
    template <int _NDIM, class _T>
    friend void generate_initial_conditions(NBodySimulation<_NDIM, _T> & sim);

    // On the fly analysis methods
    template <int _NDIM, class _T>
    friend void compute_fof_halos(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void compute_power_spectrum(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void
    compute_power_spectrum_multipoles(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void compute_bispectrum(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void output_gadget(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void output_fml(NBodySimulation<_NDIM, _T> & sim, double redshift, std::string snapshot_folder);
    template <int _NDIM, class _T>
    friend void output_pofk_for_every_step(NBodySimulation<_NDIM, _T> & sim);

    // Free all memory
    void free();
};

template <int NDIM, class T>
auto NBodySimulation<NDIM, T>::compute_deltatime_KDK(double amin, double amax, int nsteps)
    -> std::pair<std::vector<double>, std::vector<double>> {

    if (nsteps == 0)
        return {{}, {}};

    // The explicit timeevolution of the prefactor to the force (1.5*OmegaM*a in GR)
    // (In general this will depend on scale, but we are not that crazy that we compute
    // a factor per scale. Probably perfectly fine to simply use the GR expression in general)
    auto poisson_factor = [&](double a) { return 1.5 * cosmo->get_OmegaM() * grav->GeffOverG(a) * a; };

    //=====================================================
    // Timestep for a kick-step: Tassev et al.
    //=====================================================
    auto compute_kick_timestep_tassev = [&](double alow, double ahigh, int istep, int nsteps) {
        double amid = (ahigh + alow) / 2.;

        if (istep == 0)
            amid = alow;
        else if (istep == nsteps)
            amid = ahigh;

        const double da = (std::pow(ahigh, timestep_cola_nLPT) - std::pow(alow, timestep_cola_nLPT)) /
                          (timestep_cola_nLPT * std::pow(amid, timestep_cola_nLPT - 1.0));
        return da / (amid * amid * amid * cosmo->HoverH0_of_a(amid));
    };

    //=====================================================
    // Timestep for a kick-step: Quinn et al.
    //=====================================================
    auto compute_kick_timestep_quinn = [&](double alow, double ahigh, int istep, int nsteps) {
        double amid = (ahigh + alow) / 2.;

        if (istep == 0)
            amid = alow;
        else if (istep == nsteps)
            amid = ahigh;

        ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
            dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a)) * poisson_factor(a) / poisson_factor(amid);
            return GSL_SUCCESS;
        };

        // Solve the integral
        DVector tini{0.0};
        DVector avec{alow, ahigh};
        ODESolver ode;
        ode.solve(deriv, avec, tini);
        return ode.get_final_data()[0];
    };

    //=====================================================
    // Timestep for a drift-step: Tassev et al.
    //=====================================================
    auto compute_drift_timestep_tassev =
        [&](double alow, double ahigh, [[maybe_unused]] int istep, [[maybe_unused]] int nsteps) {
            double amid = (ahigh + alow) / 2.;

            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a));
                dtda[0] *= std::pow(a / amid, timestep_cola_nLPT);
                return GSL_SUCCESS;
            };

            // Solve the integral
            DVector tini{0.0};
            DVector avec{alow, ahigh};
            ODESolver ode;
            ode.solve(deriv, avec, tini);
            return ode.get_final_data()[0];
        };

    //=====================================================
    // Timestep for a drift-step: Quinn et al.
    //=====================================================
    auto compute_drift_timestep_quinn =
        [&](double alow, double ahigh, [[maybe_unused]] int istep, [[maybe_unused]] int nsteps) {
            ODEFunction deriv = [&](double a, [[maybe_unused]] const double * t, double * dtda) {
                dtda[0] = 1.0 / (a * a * a * cosmo->HoverH0_of_a(a));
                return GSL_SUCCESS;
            };

            // Solve the integral
            DVector tini{0.0};
            DVector avec{alow, ahigh};
            ODESolver ode;
            ode.solve(deriv, avec, tini);
            return ode.get_final_data()[0];
        };

    //=====================================================
    // Select the method to use
    //=====================================================
    auto compute_drift_timestep = [&](double alow, double ahigh, int istep, int nsteps) {
        if (timestep_method == "Quinn") {
            return compute_drift_timestep_quinn(alow, ahigh, istep, nsteps);
        } else if (timestep_method == "Tassev") {
            return compute_drift_timestep_tassev(alow, ahigh, istep, nsteps);
        } else {
            throw std::runtime_error("Unknown timestep_method [" + timestep_method + "]");
        }
    };
    auto compute_kick_timestep = [&](double alow, double ahigh, int istep, int nsteps) {
        if (timestep_method == "Quinn") {
            return compute_kick_timestep_quinn(alow, ahigh, istep, nsteps);
        } else if (timestep_method == "Tassev") {
            return compute_kick_timestep_tassev(alow, ahigh, istep, nsteps);
        } else {
            throw std::runtime_error("Unknown timestep_method [" + timestep_method + "]");
        }
    };

    //=====================================================
    // Select the spacing between steps
    //=====================================================
    auto scale_factor_of_step = [&](double i) {
        if (timestep_scalefactor_spacing == "linear") {
            return amin + (amax - amin) * i / double(nsteps);
        } else if (timestep_scalefactor_spacing == "logarithmic") {
            return std::exp(std::log(amin) + std::log(amax / amin) * i / double(nsteps));
        } else if (timestep_scalefactor_spacing == "powerlaw") {
            double n = timestep_spacing_power;
            return amin * std::pow(1.0 + (std::pow(amax / amin, n) - 1.0) * i / double(nsteps), 1.0 / n);
        } else {
            throw std::runtime_error("Unknown timestep_scalefactor_spacing [" + timestep_scalefactor_spacing + "]");
        }
    };

    //=====================================================
    // Timesteps: first step we displace pos and vel by 1/2 timestep to do leapfrog
    // Last time-step we synchronize back by only moving vel
    //=====================================================
    std::vector<double> delta_time_drift;
    std::vector<double> delta_time_kick;
    std::vector<double> pos_timestep;
    std::vector<double> vel_timestep;
    pos_timestep.push_back(amin);
    vel_timestep.push_back(amin);
    for (int i = 0; i <= nsteps; i++) {
        double apos_old = scale_factor_of_step(i);
        double apos_new = (i == nsteps) ? amax : scale_factor_of_step(i + 1.0);
        double avel_old = (i == 0) ? amin : scale_factor_of_step(i - 0.5);
        double avel_new = (i == 0) ? scale_factor_of_step(0.5) : (i == nsteps ? amax : scale_factor_of_step(i + 0.5));
        delta_time_drift.push_back(compute_drift_timestep(apos_old, apos_new, i, nsteps));
        delta_time_kick.push_back(compute_kick_timestep(avel_old, avel_new, i, nsteps));
        pos_timestep.push_back(apos_new);
        vel_timestep.push_back(avel_old);
    }
    return {delta_time_drift, delta_time_kick};
}

template <int NDIM, class T>
std::pair<std::vector<double>, std::vector<double>>
NBodySimulation<NDIM, T>::compute_scalefactors_KDK(double amin, double amax, int nsteps) {

    //=====================================================
    // Select the spacing between steps
    //=====================================================
    auto scale_factor_of_step = [&](double i) {
        if (nsteps == 0)
            return amin;
        if (timestep_scalefactor_spacing == "linear") {
            return amin + (amax - amin) * i / double(nsteps);
        } else if (timestep_scalefactor_spacing == "logarithmic") {
            return std::exp(std::log(amin) + std::log(amax / amin) * i / double(nsteps));
        } else if (timestep_scalefactor_spacing == "powerlaw") {
            return amin + (amax - amin) * std::pow(i / double(nsteps), timestep_spacing_power);
        } else {
            throw std::runtime_error("Unknown timestep_scalefactor_spacing [" + timestep_scalefactor_spacing + "]");
        }
    };

    std::vector<double> scalefactor_pos;
    std::vector<double> scalefactor_vel;
    scalefactor_pos.push_back(amin);
    scalefactor_vel.push_back(amin);
    for (int i = 0; i <= nsteps; i++) {
        double apos_new = (i == nsteps) ? amax : scale_factor_of_step(i + 1.0);
        double avel_new = (i == 0) ? scale_factor_of_step(0.5) : (i == nsteps ? amax : scale_factor_of_step(i + 0.5));
        scalefactor_pos.push_back(apos_new);
        scalefactor_vel.push_back(avel_new);
    }
    return {scalefactor_pos, scalefactor_vel};
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_parameters(ParameterMap & param) {

    parameters = std::make_shared<ParameterMap>(param);

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# NBodySimulation::read_parameters\n";
        std::cout << "#=====================================================\n\n";
    }

    // General parameters
    simulation_name = param.get<std::string>("simulation_name");
    simulation_boxsize = param.get<double>("simulation_boxsize");
    simulation_use_cola = param.get<bool>("simulation_use_cola");
    simulation_use_scaledependent_cola = param.get<bool>("simulation_use_scaledependent_cola");

    if (FML::ThisTask == 0) {
        std::cout << "simulation_name                          : " << simulation_name << "\n";
        std::cout << "simulation_boxsize                       : " << simulation_boxsize << "\n";
        std::cout << "simulation_use_cola                      : " << simulation_use_cola << "\n";
        std::cout << "simulation_use_scaledependent_cola       : " << simulation_use_scaledependent_cola << "\n";

        // We cannot use COLA if the particle type is not compatible with it
        if (simulation_use_cola and not FML::PARTICLE::has_get_D_1LPT<T>()) {
            throw std::runtime_error("Set to use COLA, but particle-type does not even have 1LPT growthfactors!");
        }
    }

    // Computing forces
    force_nmesh = param.get<int>("force_nmesh");
    force_density_assignment_method = param.get<std::string>("force_density_assignment_method");
    force_kernel = param.get<std::string>("force_kernel");
    force_linear_massive_neutrinos = param.get<bool>("force_linear_massive_neutrinos");

    if (FML::ThisTask == 0) {
        std::cout << "force_nmesh                              : " << force_nmesh << "\n";
        std::cout << "force_kernel                             : " << force_kernel << "\n";
        std::cout << "force_density_assignment_method          : " << force_density_assignment_method << "\n";
        std::cout << "force_linear_massive_neutrinos           : " << force_linear_massive_neutrinos << "\n";
    }

    // Initial conditions
    ic_type_of_input = param.get<std::string>("ic_type_of_input");
    ic_input_filename = param.get<std::string>("ic_input_filename");
    ic_random_field_type = param.get<std::string>("ic_random_field_type");
    ic_input_redshift = param.get<double>("ic_input_redshift");
    ic_use_gravity_model_GR = param.get<bool>("ic_use_gravity_model_GR");
    ic_initial_redshift = param.get<double>("ic_initial_redshift");
    ic_nmesh = param.get<int>("ic_nmesh");
    ic_random_seed = param.get<int>("ic_random_seed");
    ic_random_generator = param.get<std::string>("ic_random_generator");
    ic_LPT_order = param.get<int>("ic_LPT_order");
    ic_fix_amplitude = param.get<bool>("ic_fix_amplitude");
    ic_reverse_phases = param.get<bool>("ic_reverse_phases");
    if (ic_random_field_type == "nongaussian") {
        ic_fnl_type = param.get<std::string>("ic_fnl_type");
        ic_fnl = param.get<double>("ic_fnl");
        ic_fnl_redshift = param.get<double>("ic_fnl_redshift");
    }
    ic_sigma8_normalization = param.get<bool>("ic_sigma8_normalization");
    if (ic_sigma8_normalization) {
        ic_sigma8_redshift = param.get<double>("ic_sigma8_redshift");
        ic_sigma8 = param.get<double>("ic_sigma8");
    }
    if (ic_random_field_type == "reconstruct_from_particles") {
        ic_reconstruct_gadgetfilepath = param.get<std::string>("ic_reconstruct_gadgetfilepath");
        ic_reconstruct_assigment_method = param.get<std::string>("ic_reconstruct_assigment_method");
        ic_reconstruct_smoothing_filter = param.get<std::string>("ic_reconstruct_smoothing_filter");
        ic_reconstruct_dimless_smoothing_scale = param.get<double>("ic_reconstruct_dimless_smoothing_scale");
        ic_reconstruct_interlacing = param.get<bool>("ic_reconstruct_interlacing");
    }

    if (FML::ThisTask == 0) {
        std::cout << "ic_type_of_input                         : " << ic_type_of_input << "\n";
        std::cout << "ic_input_filename                        : " << ic_input_filename << "\n";
        std::cout << "ic_random_field_type                     : " << ic_random_field_type << "\n";
        std::cout << "ic_input_redshift                        : " << ic_input_redshift << "\n";
        std::cout << "ic_nmesh                                 : " << ic_nmesh << "\n";
        std::cout << "ic_random_seed                           : " << ic_random_seed << "\n";
        std::cout << "ic_random_generator                      : " << ic_random_generator << "\n";
        std::cout << "ic_LPT_order                             : " << ic_LPT_order << "\n";
        std::cout << "ic_fix_amplitude                         : " << ic_fix_amplitude << "\n";
        std::cout << "ic_reverse_phases                        : " << ic_reverse_phases << "\n";
        if (ic_random_field_type == "nongaussian") {
            std::cout << "ic_fnl_type                              : " << ic_fnl_type << "\n";
            std::cout << "ic_fnl                                   : " << ic_fnl << "\n";
        }
        if (ic_sigma8_normalization) {
            std::cout << "ic_sigma8                                : " << ic_sigma8 << "\n";
            std::cout << "ic_sigma8_redshift                       : " << ic_sigma8_redshift << "\n";
        }
        if (ic_random_field_type == "reconstruct_from_particles") {
            std::cout << "ic_reconstruct_gadgetfilepath            : " << ic_reconstruct_gadgetfilepath << "\n";
            std::cout << "ic_reconstruct_assigment_method          : " << ic_reconstruct_assigment_method << "\n";
            std::cout << "ic_reconstruct_smoothing_filter          : " << ic_reconstruct_smoothing_filter << "\n";
            std::cout << "ic_reconstruct_dimless_smoothing_scale   : " << ic_reconstruct_dimless_smoothing_scale
                      << "\n";
            std::cout << "ic_reconstruct_interlacing               : " << ic_reconstruct_interlacing << "\n";
        }
    }

    // Particles
    particle_Npart_1D = param.get<int>("particle_Npart_1D");
    particle_allocation_factor = param.get<double>("particle_allocation_factor");

    if (FML::ThisTask == 0) {
        std::cout << "ic_type_of_input                         : " << ic_type_of_input << "\n";
        std::cout << "particle_Npart_1D                        : " << particle_Npart_1D << "\n";
        std::cout << "particle_allocation_factor               : " << particle_allocation_factor << "\n";
    }

    // Timestepping
    timestep_nsteps = param.get<std::vector<int>>("timestep_nsteps");
    timestep_method = param.get<std::string>("timestep_method");
    timestep_algorithm = param.get<std::string>("timestep_algorithm");
    timestep_cola_nLPT = param.get<double>("timestep_cola_nLPT");
    timestep_scalefactor_spacing = param.get<std::string>("timestep_scalefactor_spacing");
    if (timestep_scalefactor_spacing == "powerlaw") {
        timestep_spacing_power = param.get<double>("timestep_spacing_power");
    }

    if (FML::ThisTask == 0) {
        std::cout << "timestep_nsteps                          : ";
        for (auto & n : timestep_nsteps)
            std::cout << n << " ";
        std::cout << "\n";
        std::cout << "timestep_method                          : " << timestep_method << "\n";
        std::cout << "timestep_algorithm                       : " << timestep_algorithm << "\n";
        std::cout << "timestep_scalefactor_spacing             : " << timestep_scalefactor_spacing << "\n";
        if (timestep_method == "Tassev") {
            std::cout << "timestep_cola_nLPT                       : " << timestep_cola_nLPT << "\n";
        }
        if (timestep_scalefactor_spacing == "powerlaw") {
            std::cout << "timestep_spacing_power                   : " << timestep_spacing_power << "\n";
        }
    }

    // FoF halofinding
    fof = param.get<bool>("fof");
    if (fof) {
        fof_nmin_per_halo = param.get<int>("fof_nmin_per_halo");
        fof_linking_length = param.get<double>("fof_linking_length");
        fof_nmesh_max = param.get<int>("fof_nmesh_max");

        if (FML::ThisTask == 0) {
            std::cout << "fof                                      : " << fof << "\n";
            std::cout << "fof_nmin_per_halo                        : " << fof_nmin_per_halo << "\n";
            std::cout << "fof_linking_length                       : " << fof_linking_length << "\n";
            std::cout << "fof_nmesh_max                            : " << fof_nmesh_max << "\n";
        }
    }

    // Powerspectrum
    pofk = param.get<bool>("pofk");
    if (pofk) {
        pofk_nmesh = param.get<int>("pofk_nmesh");
        pofk_interlacing = param.get<bool>("pofk_interlacing");
        pofk_subtract_shotnoise = param.get<bool>("pofk_subtract_shotnoise");
        pofk_density_assignment_method = param.get<std::string>("pofk_density_assignment_method");

        if (FML::ThisTask == 0) {
            std::cout << "pofk                                     : " << pofk << "\n";
            std::cout << "pofk_nmesh                               : " << pofk_nmesh << "\n";
            std::cout << "pofk_interlacing                         : " << pofk_interlacing << "\n";
            std::cout << "pofk_subtract_shotnoise                  : " << pofk_subtract_shotnoise << "\n";
            std::cout << "pofk_density_assignment_method           : " << pofk_density_assignment_method << "\n";
        }
    }

    // Powerspectrum multipoles
    pofk_multipole = param.get<bool>("pofk_multipole");
    if (pofk_multipole) {
        pofk_multipole_nmesh = param.get<int>("pofk_multipole_nmesh");
        pofk_multipole_interlacing = param.get<bool>("pofk_multipole_interlacing");
        pofk_multipole_subtract_shotnoise = param.get<bool>("pofk_multipole_subtract_shotnoise");
        pofk_multipole_ellmax = param.get<int>("pofk_multipole_ellmax");
        pofk_multipole_density_assignment_method = param.get<std::string>("pofk_multipole_density_assignment_method");

        if (FML::ThisTask == 0) {
            std::cout << "pofk_multipole                           : " << pofk_multipole << "\n";
            std::cout << "pofk_multipole_nmesh                     : " << pofk_multipole_nmesh << "\n";
            std::cout << "pofk_multipole_interlacing               : " << pofk_multipole_interlacing << "\n";
            std::cout << "pofk_multipole_subtract_shotnoise        : " << pofk_multipole_subtract_shotnoise << "\n";
            std::cout << "pofk_multipole_ellmax                    : " << pofk_multipole_ellmax << "\n";
            std::cout << "pofk_multipole_density_assignment_method : " << pofk_multipole_density_assignment_method
                      << "\n";
        }
    }

    // Bispectrum
    bispectrum = param.get<bool>("bispectrum");
    if (bispectrum) {
        bispectrum_nmesh = param.get<int>("bispectrum_nmesh");
        bispectrum_nbins = param.get<int>("bispectrum_nbins");
        bispectrum_density_assignment_method = param.get<std::string>("bispectrum_density_assignment_method");
        bispectrum_interlacing = param.get<bool>("bispectrum_interlacing");
        bispectrum_subtract_shotnoise = param.get<bool>("bispectrum_subtract_shotnoise");

        if (FML::ThisTask == 0) {
            std::cout << "bispectrum                               : " << bispectrum << "\n";
            std::cout << "bispectrum_nmesh                         : " << bispectrum_nmesh << "\n";
            std::cout << "bispectrum_nbins                         : " << bispectrum_nbins << "\n";
            std::cout << "bispectrum_density_assignment_method     : " << bispectrum_density_assignment_method << "\n";
            std::cout << "bispectrum_interlacing                   : " << bispectrum_interlacing << "\n";
            std::cout << "bispectrum_subtract_shotnoise            : " << bispectrum_subtract_shotnoise << "\n";
        }
    }

    // Output
    output_redshifts = param.get<std::vector<double>>("output_redshifts");
    output_particles = param.get<bool>("output_particles");
    output_fileformat = param.get<std::string>("output_fileformat");
    output_folder = param.get<std::string>("output_folder");

    if (FML::ThisTask == 0) {
        std::cout << "output_particles                         : " << output_particles << "\n";
        std::cout << "output_redshifts                         : ";
        for (auto & z : output_redshifts)
            std::cout << z << " , ";
        std::cout << "\n";
        std::cout << "output_fileformat                        : " << output_fileformat << "\n";
        std::cout << "output_folder                            : " << output_folder << "\n";
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::init() {

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Create initial conditions\n";
        std::cout << "#=====================================================\n";
    }

    // Empty output_folder means to output to current directory. Otherwise make folder
    if (output_folder != "") {
        if (not FML::create_folder(output_folder)) {
            throw std::runtime_error("Cannot create output directory [" + output_folder + "]");
        }
    }

    //=============================================================
    // Output the cosmology and growth functions
    //=============================================================
    if (FML::ThisTask == 0) {
        std::string cosmoprefix = output_folder;
        std::string gravprefix = output_folder;
        cosmoprefix = cosmoprefix + (cosmoprefix == "" ? "" : "/") + "cosmology_" + simulation_name;
        gravprefix = gravprefix + (gravprefix == "" ? "" : "/") + "gravitymodel_" + simulation_name;
        cosmo->output(cosmoprefix + ".txt");
        grav->output(gravprefix + "_k0.00001.txt", 1e-5 / grav->H0_hmpc);
        grav->output(gravprefix + "_k0.0001.txt", 1e-4 / grav->H0_hmpc);
        grav->output(gravprefix + "_k0.001.txt", 1e-3 / grav->H0_hmpc);
        grav->output(gravprefix + "_k0.01.txt", 1e-2 / grav->H0_hmpc);
        grav->output(gravprefix + "_k0.1.txt", 1e-1 / grav->H0_hmpc);
        grav->output(gravprefix + "_k1.0.txt", 1.0 / grav->H0_hmpc);
        grav->output(gravprefix + "_k10.0.txt", 1e1 / grav->H0_hmpc);
    }

    //=============================================================
    // Read in power-spectra / transfer function
    // Make spline of P(k,zini) and T(k,zini)
    // If the input it at a different redshift we use growth-factor
    // to bring it to the initial time
    //=============================================================

    DVector karr;
    DVector power;
    DVector power_primordial;
    DVector transfer;

    // If we have a MG model and want exactly the same IC as for LCDM
    // we can supply LCDM P(k) and use the GR growth factors to scale it back
    std::shared_ptr<GravityModel<NDIM>> grav_ic;
    if (ic_use_gravity_model_GR) {
        grav_ic = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        grav_ic->read_parameters(*parameters);
        grav_ic->init();
    } else {
        grav_ic = grav;
    }

    if (ic_type_of_input == "powerspectrum") {

        // Fileformat: [k (h/Mpc)  P(k)  (Mpc/h)^3]
        const int ncols = 2;
        const int col_k = 0;
        const int col_pofk = 1;
        std::vector<int> cols_to_keep{col_k, col_pofk};
        const int nheaderlines = 1;
        auto pofkdata = FML::FILEUTILS::read_regular_ascii(ic_input_filename, ncols, cols_to_keep, nheaderlines);

        karr.resize(pofkdata.size());
        power.resize(pofkdata.size());
        transfer.resize(pofkdata.size());
        power_primordial.resize(pofkdata.size());

        // Read in power-spectrum, compute transfer function
        // And use growth functions to translate it to initial redshift
        const double ainput = 1.0 / (1.0 + ic_input_redshift);
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        for (size_t i = 0; i < karr.size(); i++) {
            karr[i] = pofkdata[i][0];
            double k_hmpc = karr[i];
            double k_mpc = karr[i] * cosmo->get_h();
            double DinioverDinput = grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                                    grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
            power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
            power[i] = DinioverDinput * DinioverDinput * pofkdata[i][1];
            transfer[i] = std::sqrt(power[i] / power_primordial[i]) / (k_mpc * k_mpc);
        }

        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Reading power-spectrum from ascii file [" << ic_input_filename << "]\n";
            std::cout << "# The P(k) is at the redshift: " << ic_input_redshift << "\n";
            std::cout << "# Using growth factors we scaled it to redshift: " << ic_initial_redshift << "\n";
            std::cout << "# Found n: " << power.size() << " lines in file (sample of it below):\n";
            std::cout << "#=====================================================\n";
            std::cout << "# k (h/Mpc)      P(k,zini)  (Mpc/h)^3    T(k,zini)  (Mpc)^2\n";
            for (size_t i = 0; i < karr.size(); i++) {
                if (i % (karr.size() / 10) == 0) {
                    std::cout << std::setw(15) << karr[i] << " ";
                    std::cout << std::setw(15) << power[i] << " ";
                    std::cout << std::setw(15) << transfer[i] << "\n";
                }
            }
        }

    } else if (ic_type_of_input == "transferfunction") {

        // Fileformat: [k (h/Mpc)  T(k)  (Mpc)^2] (change below)
        const int ncols = 2;
        const int col_k = 0;
        const int col_tofk = 1;
        std::vector<int> cols_to_keep{col_k, col_tofk};
        const int nheaderlines = 1;
        auto tofkdata = FML::FILEUTILS::read_regular_ascii(ic_input_filename, ncols, cols_to_keep, nheaderlines);

        karr.resize(tofkdata.size());
        power.resize(tofkdata.size());
        transfer.resize(tofkdata.size());
        power_primordial.resize(tofkdata.size());

        // Read in transfer function, compute power-spectrum
        // And use growth functions to translate it to initial redshift
        const double ainput = 1.0 / (1.0 + ic_input_redshift);
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        for (size_t i = 0; i < karr.size(); i++) {
            karr[i] = tofkdata[i][0];
            double k_hmpc = karr[i];
            double k_mpc = karr[i] * cosmo->get_h();
            double DinioverDinput = grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                                    grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
            power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
            transfer[i] = DinioverDinput * tofkdata[i][1];
            power[i] = power_primordial[i] * std::pow(transfer[i] * k_mpc * k_mpc, 2);
        }

        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Reading transfer function from ascii file [" << ic_input_filename << "]\n";
            std::cout << "# The T(k) is at the redshift: " << ic_input_redshift << "\n";
            std::cout << "# Using growth factors we scaled it to redshift: " << ic_initial_redshift << "\n";
            std::cout << "# Found n: " << power.size() << " lines in file (sample of it below)\n";
            std::cout << "#=====================================================\n";
            std::cout << "# k (h/Mpc)      P(k,zini)  (Mpc/h)^3    T(k,zini)  (Mpc)^2:\n";
            for (size_t i = 0; i < karr.size(); i++) {
                if (i % (karr.size() / 10) == 0) {
                    std::cout << std::setw(15) << karr[i] << " ";
                    std::cout << std::setw(15) << power[i] << " ";
                    std::cout << std::setw(15) << transfer[i] << "\n";
                }
            }
        }

    } else if (ic_type_of_input == "transferinfofile") {

        // Read in all transfer functions T_i(k,z) unless we already have done so in the gravity-model
        // and in that case fetch it from there
        transferdata = grav->get_transferdata();
        if (not transferdata) {
            transferdata = std::make_shared<LinearTransferData>(cosmo->get_Omegab(),
                                                                cosmo->get_OmegaCDM(),
                                                                cosmo->get_kpivot_mpc(),
                                                                cosmo->get_As(),
                                                                cosmo->get_ns(),
                                                                cosmo->get_h());
            transferdata->read_transfer(ic_input_filename);

            // Make sure the gravity model also gets a pointer to this
            grav->set_transferdata(transferdata);
        }

        // Create the arrays. For consistency we generate them at ic_input_redshift
        // and use growth-factors to the initial time (this way we can pick backscaling or not
        // by selecting ic_input_redshift)
        const int nk = 1000;
        const double kmin = transferdata->get_kmin_hmpc_splines();
        const double kmax = transferdata->get_kmax_hmpc_splines();
        const double ainput = 1.0 / (1.0 + ic_input_redshift);
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        karr.resize(nk);
        power.resize(nk);
        power_primordial.resize(nk);
        transfer.resize(nk);
        for (size_t i = 0; i < karr.size(); i++) {
            const double k_hmpc = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(nk));
            karr[i] = k_hmpc;
            transfer[i] = transferdata->get_cdm_baryon_transfer_function(karr[i], ainput);
            power[i] = transferdata->get_cdm_baryon_power_spectrum(karr[i], ainput);
            double DinioverDinput = grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                                    grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
            power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
            transfer[i] *= DinioverDinput;
            power[i] *= DinioverDinput * DinioverDinput;
        }

    } else {
        throw std::runtime_error("Unknown ic type[" + ic_type_of_input + "]");
    }
    power_initial_spline.create(karr, power, "P(k,zini)");
    power_primordial_spline.create(karr, power_primordial, "P_primordial(k)");
    transfer_initial_spline.create(karr, transfer, "T(k,zini)");

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Testing splines: \n";
        std::cout << "#=====================================================\n";
        std::cout << "k =  0.01: P(k,zini) = " << power_initial_spline(0.01) << " (Mpc/h)^3\n";
        std::cout << "k =  0.10: P(k,zini) = " << power_initial_spline(0.1) << " (Mpc/h)^3\n";
        std::cout << "k =  1.00: P(k,zini) = " << power_initial_spline(1.0) << " (Mpc/h)^3\n";
        std::cout << "k = 10.00: P(k,zini) = " << power_initial_spline(10.0) << " (Mpc/h)^3\n";

        // Check for NaN
        FML::assert_mpi(power_initial_spline(0.01) == power_initial_spline(0.01), "NaN detected");
    }

    // Make some functions we need below to compute sigma(R,z)
    auto pofk_today_over_volume = [&](double kBox) {
        double D_today = grav_ic->get_D_1LPT(1.0, kBox / simulation_boxsize / grav_ic->H0_hmpc);
        double D_initial =
            grav_ic->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), kBox / simulation_boxsize / grav_ic->H0_hmpc);
        return std::pow(D_today / D_initial, 2) * power_initial_spline(kBox / simulation_boxsize) /
               std::pow(simulation_boxsize, NDIM);
    };
    auto pofk_atnormtime_over_volume = [&](double kBox) {
        double D_normtime =
            grav_ic->get_D_1LPT(1.0 / (1.0 + ic_sigma8_redshift), kBox / simulation_boxsize / grav_ic->H0_hmpc);
        double D_initial =
            grav_ic->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift), kBox / simulation_boxsize / grav_ic->H0_hmpc);
        return std::pow(D_normtime / D_initial, 2) * power_initial_spline(kBox / simulation_boxsize) /
               std::pow(simulation_boxsize, NDIM);
    };

    // Compute sigma8 today and at the redshift we want to normalize
    const double sigma8_today = FML::NBODY::compute_sigma_of_R(pofk_today_over_volume, 8.0, simulation_boxsize);
    const double sigma8 = FML::NBODY::compute_sigma_of_R(pofk_atnormtime_over_volume, 8.0, simulation_boxsize);

    // If we are to normalize P(k) according to a given sigma8 value
    const double pofk_norm_factor = ic_sigma8_normalization ? std::pow(ic_sigma8 / sigma8, 2) : 1.0;

    // If we normalize wrt sigma8 we need to update the splines, the transfer data and the cosmology
    if (pofk_norm_factor != 1.0) {
        for (auto & p : power) {
            p *= pofk_norm_factor;
        }
        for (auto & p : power_primordial) {
            p *= pofk_norm_factor;
        }
        for (auto & t : transfer) {
            t *= std::sqrt(pofk_norm_factor);
        }
        power_initial_spline.create(karr, power, "P(k,zini) sigma8 normalized");
        power_primordial_spline.create(karr, power_primordial, "P_primordial(k,zini)  sigma8 normalized");
        transfer_initial_spline.create(karr, transfer, "T(k,zini) sigma8 normalized");

        // Update transfer data. This also updates data in gravity model (as its the same object)
        if (transferdata) {
            transferdata->set_As(transferdata->get_As() * pofk_norm_factor);
        }
        // Update this value in the cosmology
        cosmo->set_As(cosmo->get_As() * pofk_norm_factor);

        // Print updated info
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "Updated cosmology and gravitymodel to reflect change of As:\n";
            std::cout << "#=====================================================\n";
            cosmo->info();
            grav->info();
        }
    }

    // Compute sigma(R) today. Also good to compute deltac. Then use this to estimate mass-function
    // which we can output together with the result from the FoF
    auto make_sigma_spline = [&]() {
        const int nr = 30;
        const double Rmin_mpch = 0.01;
        const double Rmax_mpch = 100.0;
        DVector R_mpch(nr);
        DVector sigma(nr);
        for (int i = 0; i < nr; i++) {
            R_mpch[i] = std::exp(std::log(Rmin_mpch) + std::log(Rmax_mpch / Rmin_mpch) * i / double(nr));
            sigma[i] = FML::NBODY::compute_sigma_of_R(pofk_today_over_volume, R_mpch[i], simulation_boxsize);
        }
        return Spline(R_mpch, sigma, "sigma(R_mpch,z=0.0)");
    };
    auto sigma_spline = make_sigma_spline();

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15) << sigma8_today << "\n";
        if (ic_sigma8_normalization) {
            std::cout << "# We will normalize to Sigma8 = " << std::setw(15) << ic_sigma8 << " at z = " << std::setw(15)
                      << ic_sigma8_redshift << "\n";
            std::cout << "# New Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15)
                      << sigma8_today * std::sqrt(pofk_norm_factor) << "\n";
            std::cout << "# <delta^2> = "
                      << sigma_spline(simulation_boxsize / ic_nmesh) *
                             grav_ic->get_D_1LPT(1.0 / (1.0 + ic_initial_redshift)) /
                             grav_ic->get_D_1LPT(1.0 / (1.0 + 0.0))
                      << "\n";
        }
        std::cout << "#=====================================================\n";
    }

    //=============================================================
    // Generate initial conditions
    //=============================================================

    // Set up random generator
    std::shared_ptr<RandomGenerator> rng;
    if (ic_random_generator == "GSL")
        rng = std::make_shared<GSLRandomGenerator>();
    else if (ic_random_generator == "MT19937")
        rng = std::make_shared<RandomGenerator>();
    else
        throw std::runtime_error("Unknown random generator " + ic_random_generator);
    rng->set_seed(ic_random_seed);

    // Make a grid for holding the initial density field
    auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment("CIC");
    FFTWGrid<NDIM> delta_ini_fourier(ic_nmesh, nextra.first, nextra.second);
    delta_ini_fourier.add_memory_label("delta_ini_fourier(k)");
    delta_ini_fourier.set_grid_status_real(false);

    // Make a gaussian or non-local non-gaussian random field in fourier space
    // We generate it at the initial redshift
    timer.StartTiming("InitialDensityField");
    if (ic_random_field_type == "gaussian") {

        auto Pofk_of_kBox_over_volume = [&](double kBox) {
            return power_initial_spline(kBox / simulation_boxsize) / std::pow(simulation_boxsize, NDIM);
        };
        FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(
            delta_ini_fourier, rng.get(), Pofk_of_kBox_over_volume, ic_fix_amplitude);

    } else if (ic_random_field_type == "nongaussian") {

        // The power-spectrum of the Bardeen potential (-Phi) at the given redshift
        const double afnl = 1.0 / (1.0 + ic_fnl_redshift);
        const double H0Box = simulation_boxsize * grav_ic->H0_hmpc;
        const double factor = 1.5 * cosmo->get_OmegaM() / afnl;
        auto Pofk_of_kBox_over_volume_primordial = [&](double kBox) {
            return power_primordial_spline(kBox / simulation_boxsize) / std::pow(simulation_boxsize, NDIM) *
                   std::pow(factor * grav_ic->get_D_1LPT(afnl, kBox / H0Box) / grav_ic->get_D_1LPT(1.0, kBox / H0Box),
                            2);
        };
        // The product of this and the function above is the initial power-spectrum for delta
        auto Pofk_of_kBox_over_Pofk_primordal = [&](double kBox) {
            return power_initial_spline(kBox / simulation_boxsize) / std::pow(simulation_boxsize, NDIM) /
                   Pofk_of_kBox_over_volume_primordial(kBox);
        };
        FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier_cosmology(
            delta_ini_fourier,
            rng.get(),
            Pofk_of_kBox_over_Pofk_primordal,
            Pofk_of_kBox_over_volume_primordial,
            ic_fix_amplitude,
            ic_fnl,
            ic_fnl_type);

    } else if (ic_random_field_type == "reconstruct_from_particles") {

        // Reconstruct the initial densityfield and use that to make IC from scratch (useful for testing)
        reconstruct_ic_from_particles(delta_ini_fourier);

    } else if (ic_random_field_type == "read_particles") {
        // Do nothing, we do this below
    } else {
        throw std::runtime_error("Unknown ic_random_field_type [" + ic_random_field_type + "]");
    }
    timer.EndTiming("InitialDensityField");

    // Reverse the phases? For doing pair-fixed simulations
    if (ic_reverse_phases) {
        auto Local_nx = delta_ini_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && fourier_index : delta_ini_fourier.get_fourier_range(islice, islice + 1)) {
                delta_ini_fourier.set_fourier_from_index(
                    fourier_index, -1.0 * delta_ini_fourier.get_fourier_from_index(fourier_index));
            }
        }
    }

    // If we simply read IC from file (useful for testing)
    if (ic_random_field_type == "read_particles") {
        FML::assert_mpi(simulation_use_cola == false,
                        "Cannot do a cola simulation without displacementfields. Use reconstruct_from_particles "
                        "instead of read_particles");
        read_ic();
    } else {

        // Generate IC from a given fourier grid. The growth rate is used to generate the velocities
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        const double fac = aini * aini * cosmo->HoverH0_of_a(std::exp(std::log(aini)));
        std::vector<double> velocity_norms{fac * grav_ic->get_f_1LPT(aini),
                                           fac * grav_ic->get_f_2LPT(aini),
                                           fac * grav_ic->get_f_3LPTa(aini),
                                           fac * grav_ic->get_f_3LPTb(aini)};
        timer.StartTiming("InitialConditions");

        // Store the LPT potentials we need from the IC (max 2LPT)
        // We store 2LPT, 3LPTa, 3LPTb depending on the size of the vector we send in
        std::vector<FFTWGrid<NDIM>> phi_nLPT_potentials;
        if (simulation_use_cola and simulation_use_scaledependent_cola) {
            if (FML::PARTICLE::has_get_D_2LPT<T>())
                phi_nLPT_potentials.resize(1);
            if (FML::PARTICLE::has_get_D_3LPTa<T>() and FML::PARTICLE::has_get_D_3LPTb<T>())
                phi_nLPT_potentials.resize(3);
        }

        FML::NBODY::NBodyInitialConditions<NDIM, T>(part,
                                                    particle_Npart_1D,
                                                    particle_allocation_factor,
                                                    delta_ini_fourier,
                                                    phi_nLPT_potentials,
                                                    ic_LPT_order,
                                                    simulation_boxsize,
                                                    ic_initial_redshift,
                                                    velocity_norms);

        // Store potential in the class
        if (simulation_use_cola and simulation_use_scaledependent_cola) {
            if (FML::PARTICLE::has_get_D_1LPT<T>()) {
                // Store phi_1LPT (D^2 phi_1LPT = -delta)
                phi_1LPT_ini_fourier = delta_ini_fourier;
                phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini)");
                auto Local_x_start = phi_1LPT_ini_fourier.get_local_x_start();
                auto Local_nx = phi_1LPT_ini_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    double kmag2;
                    std::array<double, NDIM> kvec;
                    for (auto && fourier_index : phi_1LPT_ini_fourier.get_fourier_range(islice, islice + 1)) {
                        if (Local_x_start == 0 and fourier_index == 0)
                            continue; // DC mode k = 0
                        phi_1LPT_ini_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                        auto value = phi_1LPT_ini_fourier.get_fourier_from_index(fourier_index) / kmag2;
                        phi_1LPT_ini_fourier.set_fourier_from_index(fourier_index, value);
                    }
                }
                // Deal with DC mode
                if (Local_x_start == 0)
                    phi_1LPT_ini_fourier.set_fourier_from_index(0, 0.0);
            }
            if (FML::PARTICLE::has_get_D_2LPT<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing initial 2LPT potential \n";
                phi_2LPT_ini_fourier = phi_nLPT_potentials[0];
                phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini)");
            }
            if (FML::PARTICLE::has_get_D_3LPTa<T>() and FML::PARTICLE::has_get_D_3LPTb<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing initial 3LPT potentials \n";
                phi_3LPTa_ini_fourier = phi_nLPT_potentials[1];
                phi_3LPTb_ini_fourier = phi_nLPT_potentials[2];
                phi_3LPTa_ini_fourier.add_memory_label("phi_3LPTa(k,zini)");
                phi_3LPTb_ini_fourier.add_memory_label("phi_3LPTb(k,zini)");
            }
        }

        timer.EndTiming("InitialConditions");
    }

    //============================================================
    // In the COLA frame v=0 so reset velocities
    //============================================================
    if (simulation_use_cola) {
        auto np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < np; i++) {
            auto * vel = FML::PARTICLE::GetVel(part[i]);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] = 0.0;
            }
        }
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::run() {
    timer.StartTiming("Timestepping");

    // Number of extra slices we need for density assignement
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force_density_assignment_method);

    //================================================================
    // Check that the first output redshift is not larger than the initial redshift
    //================================================================
    if (std::fabs(output_redshifts[0] - ic_initial_redshift) < 1e-3)
        output_redshifts[0] = ic_initial_redshift;
    FML::assert_mpi(output_redshifts[0] <= ic_initial_redshift,
                    "The first output cannot be before the simulation starts");

    //================================================================
    // If we only give the total number of steps then compute how many steps between outputs
    //================================================================
    if (timestep_nsteps.size() == 1) {
        std::vector<int> nsteps_between_outputs;
        auto as = compute_scalefactors_KDK(
            1.0 / (1.0 + ic_initial_redshift), 1.0 / (1.0 + output_redshifts.back()), timestep_nsteps[0]);
        int icount = 0;
        for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {
            int i = 0;
            while (1) {
                double z = 1.0 / as.first[icount + i] - 1.0;
                if (z < output_redshifts[ioutput] + 1e-3)
                    break;
                i++;
            }
            nsteps_between_outputs.push_back(i);
            icount += i;
        }
        timestep_nsteps = nsteps_between_outputs;
    }

    //================================================================
    // Print all the steps we will take
    //================================================================
    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# The time-steps and outputs we plan to take\n";
        std::cout << "#=====================================================\n\n";
        for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {
            std::cout << "Next output z = " << output_redshifts[ioutput] << " nsteps: " << timestep_nsteps[ioutput]
                      << "\n";
            double zmin = (ioutput == 0) ? ic_initial_redshift : output_redshifts[ioutput - 1];
            double zmax = output_redshifts[ioutput];
            auto as = compute_scalefactors_KDK(1.0 / (1.0 + zmin), 1.0 / (1.0 + zmax), timestep_nsteps[ioutput]);
            for (size_t i = 0; i < as.first.size() - 1; i++) {
                std::cout << "Step: " << std::setw(4) << i << " / " << std::setw(4) << timestep_nsteps[ioutput] << "\n";
                std::cout << "Positions  from z = " << std::setw(15) << 1.0 / as.first[i] - 1.0 << " -> "
                          << std::setw(15) << 1.0 / as.first[i + 1] - 1.0 << "\n";
                std::cout << "Velocities from z = " << std::setw(15) << 1.0 / as.second[i] - 1.0 << " -> "
                          << std::setw(15) << 1.0 / as.second[i + 1] - 1.0 << "\n";
            }
            std::cout << "\n";
        }
    }

    //=============================================================
    // Main time-stepping loop
    //=============================================================

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Starting main time-stepping loop\n";
        std::cout << "#=====================================================\n\n";
    }

    int istep_total = 0;
    for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {

        // Fetch the list of steps to take
        const double amin =
            (ioutput == 0) ? 1.0 / (1.0 + ic_initial_redshift) : 1.0 / (1.0 + output_redshifts[ioutput - 1]);
        const double amax = 1.0 / (1.0 + output_redshifts[ioutput]);
        const auto asteps = compute_scalefactors_KDK(amin, amax, timestep_nsteps[ioutput]);

        //=============================================================
        // Set up time-steps between output times
        //=============================================================
        auto delta_time = compute_deltatime_KDK(amin, amax, timestep_nsteps[ioutput]);

        //=============================================================
        // Time-step till the next output
        //=============================================================
        if (timestep_nsteps[ioutput] > 0)
            for (int istep = 0; istep <= timestep_nsteps[ioutput]; istep++) {

                const double apos = asteps.first[istep];
                const double avel = asteps.second[istep];
                const double apos_new = asteps.first[istep + 1];
                const double avel_new = asteps.second[istep + 1];
                const double delta_time_drift = delta_time.first[istep];
                const double delta_time_kick = delta_time.second[istep];

                if (FML::ThisTask == 0) {
                    std::cout << "\n";
                    std::cout << "Taking substep: " << istep << " / " << timestep_nsteps[ioutput]
                              << " Total steps taken: " << istep_total << "\n";
                    std::cout << "Positions from  z = " << std::setw(15) << 1.0 / apos - 1.0 << " -> " << std::setw(15)
                              << 1.0 / apos_new - 1.0 << "\n";
                    std::cout << "Velocities from z = " << std::setw(15) << 1.0 / avel - 1.0 << " -> " << std::setw(15)
                              << 1.0 / avel_new - 1.0 << "\n";
                    std::cout << "deltatime_pos = " << std::setw(15) << delta_time_drift
                              << " deltatime_vel = " << std::setw(15) << delta_time_kick << "\n";
                }

                // The last step is just a sync step so don't count towards the total
                if (istep < timestep_nsteps[ioutput])
                    istep_total++;

                // Compute total density field
                FFTWGrid<NDIM> density_grid_fourier(force_nmesh, nleftright.first, nleftright.second);
                if (delta_time_kick != 0.0) {
                    timer.StartTiming("ComputeDensityField");
                    compute_density_field_fourier(density_grid_fourier, apos);
                    timer.EndTiming("ComputeDensityField");
                }

                // Compute forces
                std::array<FFTWGrid<NDIM>, NDIM> force_real;
                if (delta_time_kick != 0.0) {
                    timer.StartTiming("ComputeForce");
                    grav->compute_force(apos,
                                        grav->H0_hmpc * simulation_boxsize,
                                        density_grid_fourier,
                                        force_density_assignment_method,
                                        force_real);
                    timer.EndTiming("ComputeForce");
                }

                // Kick particles (updates velocity)
                if (delta_time_kick != 0.0) {
                    timer.StartTiming("Kick");
                    FML::NBODY::KickParticles<NDIM>(force_real, part, delta_time_kick, force_density_assignment_method);
                    timer.EndTiming("Kick");
                }

                // For COLA we can do the kick and drift at the same time
                if (simulation_use_cola) {
                    timer.StartTiming("COLA");
                    // If the growth factors are scaledependent then we use the scaledependent version
                    // unless simulation_use_scaledependent_cola is set to false
                    const double aini = 1.0 / (1.0 + ic_initial_redshift);
                    if (simulation_use_scaledependent_cola and grav->scaledependent_growth) {
                        cola_kick_drift_scaledependent<NDIM, T>(part,
                                                                grav,
                                                                phi_1LPT_ini_fourier,
                                                                phi_2LPT_ini_fourier,
                                                                phi_3LPTa_ini_fourier,
                                                                phi_3LPTb_ini_fourier,
                                                                grav->H0_hmpc * simulation_boxsize,
                                                                aini,
                                                                apos,
                                                                apos_new,
                                                                delta_time_kick,
                                                                delta_time_drift);
                    } else {
                        cola_kick_drift<NDIM, T>(part, grav, aini, apos, apos_new, delta_time_kick, delta_time_drift);
                    }
                    timer.EndTiming("COLA");
                }

                // Drift particles (updates positions)
                if (delta_time_drift != 0.0) {
                    timer.StartTiming("Drift");
                    FML::NBODY::DriftParticles<NDIM, T>(part, delta_time_drift);
                    timer.EndTiming("Drift");
                }

                // Show info about particles
                part.info();

                // Show info about system memory use
                FML::print_system_memory_use();
            }

        //=============================================================
        // Analyze data and output
        //=============================================================
        analyze_and_output(ioutput, output_redshifts[ioutput]);
    }
    timer.EndTiming("Timestepping");

    //=============================================================
    // Print all timings
    //=============================================================
    timer.EndTiming("The whole simulation");
    if (FML::ThisTask == 0) {
        timer.PrintAllTimings();
    }

#ifdef MEMORY_LOGGING
    // Simulation is over, output the memory usage (of what we log)
    FML::MemoryLog::get()->print();
#endif
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a) {

    if (FML::ThisTask == 0) {
        std::cout << "Adding particles (baryons+CDM) to the densityfield\n";
    }

    //=============================================================
    // Particles to grid
    //=============================================================
    FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                          part.get_npart(),
                                          part.get_npart_total(),
                                          density_grid_fourier,
                                          force_density_assignment_method);

    //=============================================================
    // Fourier transform
    //=============================================================
    density_grid_fourier.fftw_r2c();

    //=============================================================
    // Bin up power-spectrum (its basically free as we have the density field)
    //=============================================================
    const double redshift = 1.0 / a - 1.0;
    PowerSpectrumBinning<NDIM> pofk_particles(density_grid_fourier.get_nmesh() / 2);
    pofk_particles.subtract_shotnoise = false;
    FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(
        density_grid_fourier, pofk_particles, force_density_assignment_method);
    pofk_particles.scale(simulation_boxsize);
    pofk_cb_every_step.push_back({redshift, pofk_particles});

    //=============================================================
    // Add on contribution from massive neutrinos, radiation etc.
    // We need to have transfer functions for what follows and for that we
    // check [transferdata] which is created if "transferinfofile" is used
    //=============================================================
    if (force_linear_massive_neutrinos and cosmo->get_OmegaMNu() > 0.0 and transferdata) {

        // First step we store the initial  density field
        if (not initial_density_field_fourier) {
            initial_density_field_fourier = density_grid_fourier;
            initial_density_field_fourier.add_memory_label("density(k,zini)");
            FML::assert_mpi(std::fabs(redshift - ic_initial_redshift) < 1e-3, "This should not happen");
        }

        if (FML::ThisTask == 0) {
            std::cout << "Adding linear massive neutrinos to the densityfield\n";
        }

        // Function to translate delta_cb(zini,k) -> delta_nu(z,k) using transfer functions
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        const double koverkBox = 1.0 / simulation_boxsize;
        auto norm = [&](double kBox) {
            double T_mnu = transferdata->get_massive_neutrino_transfer_function(kBox * koverkBox, a);
            double T_cb_ini = transferdata->get_cdm_baryon_transfer_function(kBox * koverkBox, aini);
            return (T_mnu / T_cb_ini);
        };

        // We compute the total matter density-field deltaM = (OmegaCB deltaCB + OmegaMNu deltaMNu)/OmegaM
        const double OmegaM = cosmo->get_OmegaM();
        const double OmegaMNu = cosmo->get_OmegaMNu();
        const double fMNu = OmegaMNu / OmegaM;

        auto Local_nx = initial_density_field_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            [[maybe_unused]] double kmag;
            [[maybe_unused]] std::array<double, NDIM> kvec;
            for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
                density_grid_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                auto delta_ini = initial_density_field_fourier.get_fourier_from_index(fourier_index);
                auto deltaCB = density_grid_fourier.get_fourier_from_index(fourier_index);
                auto deltaM = deltaCB * (1.0 - fMNu) + delta_ini * norm(kmag) * fMNu;
                density_grid_fourier.set_fourier_from_index(fourier_index, deltaM);
            }
        }

        //=============================================================
        // Bin up total power-spectrum (its basically free)
        // NB: the density field is here the sum of W*deltaCB + deltaNu
        // so the last part is going to be added up as deltaNu/W^2 so
        // its not fully correct for large k, but this is not a high
        // quality P(k) anyway... should improve this at some point
        // One way is to bin up <(W*deltaCB)^2>, <deltaNu^2> and the cross
        // <(W*deltaCB) deltaNu> and deconvolve them seperately
        //=============================================================
        PowerSpectrumBinning<NDIM> pofk_total(density_grid_fourier.get_nmesh() / 2);
        pofk_total.subtract_shotnoise = false;
        FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(
            density_grid_fourier, pofk_total, force_density_assignment_method);
        pofk_total.scale(simulation_boxsize);
        pofk_total_every_step.push_back({redshift, pofk_total});
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::analyze_and_output(int ioutput, double redshift) {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    std::string redshiftstring = stream.str();
    std::string snapshot_folder = output_folder;

    //=============================================================
    // Create snapshotfolder
    //=============================================================
    snapshot_folder =
        output_folder + (output_folder == "" ? "" : "/") + "snapshot_" + simulation_name + "_z" + redshiftstring;
    if (not FML::create_folder(snapshot_folder)) {
        throw std::runtime_error("Failed to create snapshot folder [" + snapshot_folder + "]");
    }

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Analyze and output\n";
        std::cout << "#=====================================================\n";
        std::cout << "Doing output " << ioutput << " at redshift z = " << redshift << "\n";
        std::cout << "Snapshot will be stored in folder " << snapshot_folder << "\n";
    }

    auto add_on_LPT_velocity = [&](double addsubtract_sign) {
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        const double a = 1.0 / (1.0 + redshift);
        if (simulation_use_scaledependent_cola and grav->scaledependent_growth) {
            cola_add_on_LPT_velocity_scaledependent<NDIM, T>(part,
                                                             grav,
                                                             phi_1LPT_ini_fourier,
                                                             phi_2LPT_ini_fourier,
                                                             phi_3LPTa_ini_fourier,
                                                             phi_3LPTb_ini_fourier,
                                                             grav->H0_hmpc * simulation_boxsize,
                                                             aini,
                                                             a,
                                                             addsubtract_sign);
        } else {
            cola_add_on_LPT_velocity<NDIM, T>(part, grav, aini, a, addsubtract_sign);
        }
    };

    //=============================================================
    // Change velocities to true velocities
    // In the COLA frame the initial velocity is zero, i.e. we have subtracted the
    // velocity predicted by LPT. Here we add on the LPT velocity to the particles
    //=============================================================
    if (simulation_use_cola) {
        timer.StartTiming("COLA output");
        add_on_LPT_velocity(+1.0);
        timer.EndTiming("COLA output");
    }

    //=============================================================
    // Every step we (might) have computed Pcb(k) and possibly Ptotal(k)
    // and stored this. Output these to file
    //=============================================================
    if (pofk_cb_every_step.size() > 0 or pofk_total_every_step.size() > 0) {
        output_pofk_for_every_step(*this);
    }

    //=============================================================
    // Power-spectrum
    //=============================================================
    if (pofk) {
        timer.StartTiming("Power-spectrum");
        compute_power_spectrum(*this, redshift, snapshot_folder);
        timer.EndTiming("Power-spectrum");
    }

    //=============================================================
    // Power-spectrum multipoles
    //=============================================================
    if (pofk_multipole) {
        timer.StartTiming("Power-spectrum multipoles");
        compute_power_spectrum_multipoles(*this, redshift, snapshot_folder);
        timer.EndTiming("Power-spectrum multipoles");
    }

    //=============================================================
    // Halo finding
    //=============================================================
    if (fof) {
        timer.StartTiming("FOF");
        compute_fof_halos(*this, redshift, snapshot_folder);
        timer.EndTiming("FOF");
    }

    //=============================================================
    // Bispectrum
    //=============================================================
    if (bispectrum) {
        timer.StartTiming("Bispectrum");
        compute_bispectrum(*this, redshift, snapshot_folder);
        timer.EndTiming("Bispectrum");
    }

    //=============================================================
    // Write particles to file
    //=============================================================
    if (output_particles) {
        timer.StartTiming("Output particles");
        if (output_fileformat == "GADGET")
            output_gadget(*this, redshift, snapshot_folder);
        if (output_fileformat == "FML") {
            output_fml(*this, redshift, snapshot_folder);
        }
        timer.EndTiming("Output particles");
    }

    //=============================================================
    // Change velocities back to true COLA velocities
    //=============================================================
    if (simulation_use_cola) {
        timer.StartTiming("COLA output");
        add_on_LPT_velocity(-1.0);
        timer.EndTiming("COLA output");
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::free() {
    part.free();
    initial_density_field_fourier.free();
    phi_1LPT_ini_fourier.free();
    phi_2LPT_ini_fourier.free();
    phi_3LPTa_ini_fourier.free();
    phi_3LPTb_ini_fourier.free();
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::reconstruct_ic_from_particles(FFTWGrid<NDIM> & delta_fourier) {
    // This assumes the IC was generated with 1LPT, though the error is small if it was 2LPT
    // And its too much of a hazzle to do it properly (not trivial)

    // Read in gadget files (all tasks reads the same files)
    GadgetReader g;
    std::vector<T> externalpart;
    const std::string fileprefix = ic_reconstruct_gadgetfilepath;
    const bool only_keep_part_in_domain = true;
    const double buffer_factor = 1.0;
    const bool verbose = false;
    g.read_gadget(fileprefix, externalpart, buffer_factor, only_keep_part_in_domain, verbose);

    size_t NumPart = externalpart.size();
    size_t NumPartTotal = NumPart;
    FML::SumOverTasks(&NumPartTotal);

    // Reallocate if we need more extra slices
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(ic_reconstruct_assigment_method);
    auto Nmesh = delta_fourier.get_nmesh();
    if (delta_fourier.get_n_extra_slices_left() < nleftright.first or
        delta_fourier.get_n_extra_slices_right() < nleftright.second)
        delta_fourier = FFTWGrid<NDIM>(Nmesh, nleftright.first, nleftright.second);

    // Assign particles to grid
    FML::INTERPOLATION::particles_to_fourier_grid(externalpart.data(),
                                                  NumPart,
                                                  NumPartTotal,
                                                  delta_fourier,
                                                  ic_reconstruct_assigment_method,
                                                  ic_reconstruct_interlacing);

    // Deconvolve window function
    FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(delta_fourier, ic_reconstruct_assigment_method);

    // Smoothing (should just use a sharpk filter to set modes beyond Nmesh used to generate the IC to zero)
    // If we use a larger grid then the highest frequency modes, beyond knyuist of the grid used to
    // generate the IC, will just be noise and will lead to problems so these modes needs to be killed
    FML::GRID::smoothing_filter_fourier_space(
        delta_fourier, ic_reconstruct_dimless_smoothing_scale, ic_reconstruct_smoothing_filter);
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_ic() {

    // Read in gadget files (all tasks reads the same files)
    // Using FML::Vector in case we have memory logging on to allow us to
    // move the data into mpiparticles
    GadgetReader g;
    FML::Vector<T> externalpart;
    const std::string fileprefix = ic_reconstruct_gadgetfilepath;
    const bool only_keep_part_in_domain = true;
    const bool verbose = false;
    g.read_gadget(fileprefix, externalpart, particle_allocation_factor, only_keep_part_in_domain, verbose);

    auto header = g.get_header();
    const double scale_factor = header.time;

    // Velocities we get from the gadget reader is peculiar km/s
    // so scale to code units
    const double vel_norm = scale_factor / (100 * simulation_boxsize);
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < externalpart.size(); i++) {
        auto * vel = FML::PARTICLE::GetVel(externalpart[i]);
        for (int idim = 0; idim < NDIM; idim++)
            vel[idim] *= vel_norm;
    }

    // Move them into MPIParticles
    part.move_from(std::move(externalpart));
}

#endif
