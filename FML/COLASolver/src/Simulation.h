#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/Interpolation/SheetInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>

#include "AnalyzeOutput.h"
#include "COLA.h"
#include "Cosmology.h"
#include "ExternalIC.h"
#include "GravityModel.h"
#include "GravityModel_GR.h"
#include "InitialConditions.h"
#include "InitialPowerSpectrum.h"
#include "Lightcone/Lightcone.h"
#include "NBodyGauge.h"
#include "Neutrinos.h"
#include "Particles.h"
#include "SimulationParameters.h"
#include "TimeStepping.h"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
/// A particle-mesh N-body simulation (optionally with COLA).
///
/// Usage: construct with a cosmology and a gravity model, then call
/// read_parameters(param), init() (creates the initial conditions) and run().
///
/// The pieces live in separate headers:
///   SimulationParameters.h : all the parameters (read, printed and validated)
///   TimeStepping.h         : where the steps are and the size of the kick/drift steps
///   InitialPowerSpectrum.h : P(k,zini) / T(k,zini) the initial conditions are made from
///   InitialConditions.h    : the initial density field
///   ExternalIC.h           : starting from particles read from file
///   COLA.h                 : the COLA method (ColaState)
///   AnalyzeOutput.h        : on the fly analysis and output
//=============================================================================
template <int NDIM, class T>
class NBodySimulation {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using MPIParticles = FML::PARTICLE::MPIParticles<T>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using LinearTransferData = FML::FILEUTILS::LinearTransferData;
    using PowerSpectrumBinning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>;
    using Parameters = COLASOLVER::SimulationParameters;

  protected:
    /// Everything related to the background evolution
    std::shared_ptr<Cosmology> cosmo;

    /// Everything related to gravity: growth factors, computing forces
    std::shared_ptr<GravityModel<NDIM>> grav;
    /// The gravity model used to make the IC (GR if ic_use_gravity_model_GR, otherwise grav)
    std::shared_ptr<GravityModel<NDIM>> grav_ic;

    /// Linear perturbations, transfer functions, initial pofk
    /// This is only used if ic_type_of_input == transferinfofile
    std::shared_ptr<LinearTransferData> transferdata;

    /// A copy of the parameterfile (passed on to the models we create)
    std::shared_ptr<ParameterMap> parameters;

    /// The parameters of the simulation
    Parameters params;

    /// All the particles (CDM+baryons as one species "cb", or the CDM particles if particle_baryons)
    MPIParticles part;
    /// The baryon particles (only if particle_baryons)
    MPIParticles part_baryon;
    /// The weights of the species in the CDM+baryon density: OmegaCDM / OmegaCB and Omegab / OmegaCB
    double weight_cdm{1.0};
    double weight_baryon{0.0};
    /// The massive neutrino particles (only if particle_neutrinos; delta-f method)
    FML::PARTICLE::MPIParticles<ParticleNeutrino<NDIM>> part_nu;
    /// Neutrino mass (one species, eV) and temperature today (eV)
    double mnu_eV{0.0};
    double Tnu0_eV{0.0};

    /// The initial density field delta_cb(zini,k) computed with the same Nmesh
    /// as the forces (used for linear massive neutrinos and the N-body gauge - not allocated otherwise)
    FFTWGrid<NDIM> initial_density_field_fourier;
    // Shift of the particles relative to the grid when computing the density (force interlacing)
    std::array<double, NDIM> density_shift{};
    // The force settings used in the current step (params.force or the early-time settings)
    COLASOLVER::ForceSettings current_force;
    void set_force_settings_for_step(double a);

    /// The linear N-body gauge source (if force_nbody_gauge)
    std::shared_ptr<COLASOLVER::NBodyGaugeSource> nbody_gauge;

    /// COLA settings and the LPT potentials at the initial time
    ColaState<NDIM, T> cola;

    /// The lightcone construction
    std::shared_ptr<Lightcone<NDIM, T>> lightcone;

    /// Timings of the code
    FML::UTILS::Timings timer;

    // Splines used to generate the IC
    Spline power_primordial_spline{"P_primordial(k)"}; // Spline of primordial power-spectrum (h/Mpc) vs (Mpc/h)^3
    Spline power_initial_spline{"P(k,zini)"};          // Spline of baryon-CDM power-spectrum (h/Mpc) vs (Mpc/h)^3
    Spline transfer_initial_spline{"T(k,zini)"};       // Spline of baryon-CDM transfer function (h/Mpc) vs Mpc^2

    //=============================================================================
    // Some of the stuff we compute and output is small so we also keep it
    // in the class in case one wants to process it later
    //=============================================================================
    // A list of (z, P(k)) for the particles at every step (naive binning - not using the same pofk_* setting!)
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_cb_every_step;
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_total_every_step;
    // ...and of the CDM and baryon particles separately (if particle_baryons)
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_cdm_every_step;
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_baryon_every_step;
    // ...and of the neutrino particles (if particle_neutrinos)
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_nu_every_step;
    // A list of (z, P(k)) for the particles that we compute if pofk = true
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_every_output;
    // A list of (z, P_ell(k)) for the particles that we compute every output if pofk_multipoles = true
    std::vector<std::pair<double, std::vector<PowerSpectrumBinning>>> pofk_multipoles_every_output;

  public:
    NBodySimulation() = default;
    NBodySimulation(std::shared_ptr<Cosmology> cosmo, std::shared_ptr<GravityModel<NDIM>> grav)
        : cosmo(cosmo), grav(grav) {
        timer.StartTiming("The whole simulation");
    }

    // Move in particles from an external source
    NBodySimulation(std::shared_ptr<Cosmology> cosmo, std::shared_ptr<GravityModel<NDIM>> grav, MPIParticles && _part)
        : NBodySimulation(cosmo, grav) {
        part = std::move(_part);
    }

    /// Read (and check) the parameters
    void read_parameters(ParameterMap & param);

    /// Generate the initial conditions
    void init();

    /// Run simulation
    void run();

    /// Free all memory
    void free();

    //=============================================================================
    // Access to the state of the simulation (used by the analysis in AnalyzeOutput.h)
    //=============================================================================
    const Parameters & get_parameters() const { return params; }
    MPIParticles & get_particles() { return part; }
    /// Separate CDM and baryon particles? Then get_particles() are the CDM particles
    bool has_baryons() const { return params.particle.baryons; }
    MPIParticles & get_baryon_particles() { return part_baryon; }
    double get_weight_cdm() const { return weight_cdm; }
    double get_weight_baryon() const { return weight_baryon; }
    /// Massive neutrinos as particles (delta-f method)?
    bool has_neutrino_particles() const { return params.particle.neutrinos; }
    FML::PARTICLE::MPIParticles<ParticleNeutrino<NDIM>> & get_neutrino_particles() { return part_nu; }
    /// The shot noise of the (weighted) neutrino density field: sum w^2 / N^2 (1/N without the weights)
    double neutrino_shotnoise() const;
    /// Call f(particles) for all the species
    template <class F>
    void for_each_species(F && f) {
        f(part);
        if (has_baryons())
            f(part_baryon);
    }
    std::shared_ptr<Cosmology> get_cosmology() const { return cosmo; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model() const { return grav; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model_ic() const { return grav_ic; }
    std::shared_ptr<LinearTransferData> get_transferdata() const { return transferdata; }
    const Spline & get_power_initial_spline() const { return power_initial_spline; }
    const auto & get_pofk_cb_every_step() const { return pofk_cb_every_step; }
    const auto & get_pofk_total_every_step() const { return pofk_total_every_step; }
    const auto & get_pofk_cdm_every_step() const { return pofk_cdm_every_step; }
    const auto & get_pofk_baryon_every_step() const { return pofk_baryon_every_step; }
    const auto & get_pofk_nu_every_step() const { return pofk_nu_every_step; }
    const auto & get_pofk_every_output() const { return pofk_every_output; }
    const auto & get_pofk_multipoles_every_output() const { return pofk_multipoles_every_output; }

    /// The time-stepping (where the steps are and their sizes)
    COLASOLVER::TimeStepping<NDIM> time_stepping() const { return {params.timestep, cosmo, grav}; }

    /// The scale factors of positions and velocities after each step from amin to amax
    std::pair<std::vector<double>, std::vector<double>> compute_scalefactors_KDK(double amin, double amax, int nsteps) {
        return time_stepping().scalefactors_KDK(amin, amax, nsteps);
    }

    /// The size of the drift and kick steps from amin to amax
    std::pair<std::vector<double>, std::vector<double>> compute_deltatime_KDK(double amin, double amax, int nsteps) {
        return time_stepping().deltatime_KDK(amin, amax, nsteps);
    }

  protected:
    // The parts of init()
    void output_background_and_growth_functions();
    void setup_initial_power_spectrum();
    void generate_initial_conditions();
    void generate_initial_conditions_two_species(const FFTWGrid<NDIM> & delta_cb_fourier);
    void apply_lattice_offset();
    void generate_neutrino_particles(const FFTWGrid<NDIM> & delta_cb_fourier);
    void read_ic();
    void read_phases(FFTWGrid<NDIM> & delta_fourier);
    void setup_nbody_gauge();

    // The parts of run()
    struct StepInfo {
        double apos;             // Scale factor of the positions at the start of the step
        double apos_new;         // ...and at the end of the step
        double avel_new;         // Scale factor of the velocities at the end of the step
        double delta_time_drift; // The size of the drift step
        double delta_time_kick;  // The size of the kick step
        double velocity_scale;   // Multiply the velocities by this before the kick
    };
    void step(const StepInfo & s);
    void drift_and_lightcone(const StepInfo & s);
    void kick_with_interlaced_force(const StepInfo & s);
    void kick_neutrinos(std::array<FFTWGrid<NDIM>, NDIM> & force_real, const StepInfo & s);
    void drift_neutrinos(const StepInfo & s);
    void update_neutrino_weights();
    void add_neutrino_particles_to_density(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void analyze_and_output(int ioutput, double redshift);
};

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_parameters(ParameterMap & param) {
    parameters = std::make_shared<ParameterMap>(param);
    params.read(param);

    // We cannot use COLA if the particle type is not compatible with it
    if (params.simulation.use_cola and not FML::PARTICLE::has_get_D_1LPT<T>()) {
        throw std::runtime_error("Set to use COLA, but particle-type does not even have 1LPT growthfactors!");
    }

    FML::NBODY::set_fiducial_greens_functions_kernel(params.force.greens_function_kernel);

    // The fiducial gradient kernel: the continuous kernel (D = ik) is the most accurate when the force grid is
    // the same as the particle grid. When the force grid is finer the regular particle lattice (of the IC) puts
    // power just below the Nyquist frequency of the grid where ik is large, which gives large spurious forces
    // (tens of percent errors in the growth of individual large-scale modes and too much small-scale power).
    // The finite difference kernels (which vanish at the Nyquist frequency) suppress this. Together with the
    // half-cell lattice offset (ic_lattice_offset = fiducial) the four-point kernel gives the most accurate
    // nonlinear P(k) for force grids up to 3 times the particle grid and the two-point kernel for finer grids
    // (see the README)
    std::string gradient_kernel = params.force.gradient_kernel;
    if ((gradient_kernel == "fiducial" or gradient_kernel == "") and params.force.nmesh > params.particle.Npart_1D) {
        gradient_kernel = params.force.nmesh <= 3 * params.particle.Npart_1D ? "discrete_4pt" : "discrete_2pt";
        if (FML::ThisTask == 0)
            std::cout << "NB: force_nmesh > particle_Npart_1D so we use force_gradient_kernel = " << gradient_kernel
                      << " (set force_gradient_kernel to continuous, discrete_2pt or discrete_4pt to change this)\n";
    }
    FML::NBODY::set_fiducial_gradient_kernel(gradient_kernel);
    FML::NBODY::FIDUCIAL_DECONVOLVE_FORCE = params.force.deconvolve_window;
    current_force = params.force;

    // The accurate force options (NUFFT, deconvolution, early-time force) remove the smoothing of the force
    // near the grid Nyquist frequency. With a grid finer than the particle grid the modes between the particle
    // and the grid Nyquist frequency are dominated by the discreteness of the particles and these options then
    // give far too much small-scale power (+15-60% at k = 0.2 h/Mpc in tests with a 2x grid)
    const bool exact_force = params.force.density_method != "assignment" or params.force.deconvolve_window or
                             params.force.interpolation_method == "nufft" or params.force.early_method != "none";
    if (exact_force and params.force.nmesh > params.particle.Npart_1D and FML::ThisTask == 0)
        std::cout << "WARNING: force_density_method / force_deconvolve_window / force_early_method are meant for "
                     "force_nmesh = particle_Npart_1D. With a finer force grid they amplify the discreteness of the "
                     "particles (see README)\n";

    if (params.particle.neutrinos and param.get<bool>("lightcone", false))
        throw std::runtime_error("Invalid parameters: particle_neutrinos = true: needs lightcone = false (not "
                                 "implemented for neutrino particles yet)");
    if (params.particle.baryons and param.get<bool>("lightcone", false))
        throw std::runtime_error("Invalid parameters: particle_baryons = true: needs lightcone = false (not "
                                 "implemented for two species yet)");

    cola.enabled = params.simulation.use_cola;
    cola.scaledependent = params.simulation.use_scaledependent_cola;
    cola.LPT_order = params.simulation.cola_LPT_order;
    cola.aini = params.ic.a_initial();
    cola.H0Box = grav->H0_hmpc * params.simulation.boxsize;
    cola.grav = grav;
}

//=============================================================================
// Initial conditions
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::init() {

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Create initial conditions\n";
        std::cout << "#=====================================================\n";
    }

    // Empty output_folder means to output to current directory. Otherwise make folder
    if (params.output.folder != "") {
        if (not FML::create_folder(params.output.folder)) {
            throw std::runtime_error("Cannot create output directory [" + params.output.folder + "]");
        }
    }

    output_background_and_growth_functions();

    // The weights of CDM and baryons in the CDM+baryon density (when they are separate particles)
    if (has_baryons()) {
        const double OmegaCB = cosmo->get_OmegaCDM() + cosmo->get_Omegab();
        weight_cdm = cosmo->get_OmegaCDM() / OmegaCB;
        weight_baryon = cosmo->get_Omegab() / OmegaCB;
        if (FML::ThisTask == 0)
            std::cout << "# CDM and baryons as separate particles. Weights in the CDM+baryon density: CDM "
                      << weight_cdm << " baryons " << weight_baryon << "\n";
    }

    // Massive neutrinos as particles
    if (has_neutrino_particles()) {
        if (not(cosmo->get_OmegaMNu() > 0.0 and cosmo->get_Mnu_eV() > 0.0))
            throw std::runtime_error("particle_neutrinos = true needs massive neutrinos (cosmology_OmegaMNu > 0)");
        mnu_eV = cosmo->get_Mnu_eV() / cosmo->get_N_nu();
        Tnu0_eV = cosmo->get_neutrino_temperature_eV(1.0);
        if (FML::ThisTask == 0)
            std::cout << "# Massive neutrinos as particles (" << params.particle.neutrino_Npart_1D << "^" << NDIM
                      << ", delta-f weights: " << std::boolalpha << params.particle.neutrino_deltaf
                      << "). m_nu = " << mnu_eV << " eV (x" << cosmo->get_N_nu() << "), T_nu0 = " << Tnu0_eV
                      << " eV, f_nu = " << cosmo->get_fMNu() << "\n";
    }

    // If we have a MG model and want exactly the same IC as for LCDM
    // we can supply LCDM P(k) and use the GR growth factors to scale it back
    if (params.ic.use_gravity_model_GR) {
        grav_ic = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        grav_ic->read_parameters(*parameters);
        grav_ic->init();
    } else {
        grav_ic = grav;
    }

    setup_initial_power_spectrum();

    generate_initial_conditions();

    // The N-body gauge source needs the initial density field of the particles
    if (params.force.nbody_gauge)
        setup_nbody_gauge();

    // In the COLA frame v=0 so reset velocities
    if (cola.enabled) {
        cola_initialize_velocities<NDIM, T>(part);
    }

    // Set up the lightcone
    std::vector<std::string> particle_types_lightcone{"cb"};
    lightcone = std::make_shared<Lightcone<NDIM, T>>(cosmo);
    lightcone->read_parameters(*parameters);
    lightcone->init(particle_types_lightcone);
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::output_background_and_growth_functions() {
    if (FML::ThisTask != 0)
        return;
    const std::string cosmoprefix = params.output.path("cosmology_" + params.simulation.name);
    const std::string gravprefix = params.output.path("gravitymodel_" + params.simulation.name);
    cosmo->output(cosmoprefix + ".txt");
    grav->output(gravprefix + "_k0.00001.txt", 1e-5 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.0001.txt", 1e-4 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.001.txt", 1e-3 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.01.txt", 1e-2 / grav->H0_hmpc);
    grav->output(gravprefix + "_k0.1.txt", 1e-1 / grav->H0_hmpc);
    grav->output(gravprefix + "_k1.0.txt", 1e+0 / grav->H0_hmpc);
    grav->output(gravprefix + "_k10.0.txt", 1e+1 / grav->H0_hmpc);
}

//=============================================================================
/// Read the input P(k) / T(k), bring it to the initial redshift, make splines
/// and (optionally) normalize it to a given sigma8
//=============================================================================
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_initial_power_spectrum() {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;

    auto table = COLASOLVER::read_initial_power_spectrum<NDIM>(ic, cosmo, grav, grav_ic, transferdata);
    auto make_splines = [&](std::string label) {
        power_initial_spline.create(table.k, table.power, "P(k,zini)" + label);
        power_primordial_spline.create(table.k, table.power_primordial, "P_primordial(k)" + label);
        transfer_initial_spline.create(table.k, table.transfer, "T(k,zini)" + label);
    };
    make_splines("");

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Testing splines: \n";
        std::cout << "#=====================================================\n";
        for (double k : {0.01, 0.1, 1.0, 10.0})
            std::cout << "k = " << std::setw(5) << k << ": P(k,zini) = " << power_initial_spline(k) << " (Mpc/h)^3\n";
        FML::assert_mpi(power_initial_spline(0.01) == power_initial_spline(0.01), "NaN detected");
    }

    // sigma(R,z) from the current P(k,zini)
    auto sigma_of_R = [&](double R_mpch, double redshift) {
        return COLASOLVER::sigma_of_R<NDIM>(power_initial_spline,
                                            grav_ic,
                                            boxsize,
                                            ic.initial_redshift,
                                            redshift,
                                            R_mpch);
    };
    const double sigma8_today = sigma_of_R(8.0, 0.0);

    // Normalize P(k) to a given sigma8 value. We then need to update the splines,
    // the transfer data and the cosmology
    double pofk_norm_factor = 1.0;
    if (ic.sigma8_normalization) {
        const double sigma8 = sigma_of_R(8.0, ic.sigma8_redshift);
        pofk_norm_factor = std::pow(ic.sigma8 / sigma8, 2);
    }
    if (pofk_norm_factor != 1.0) {
        table.scale(pofk_norm_factor);
        make_splines(" sigma8 normalized");

        // Update transfer data. This also updates data in gravity model (as its the same object)
        if (transferdata) {
            transferdata->set_As(transferdata->get_As() * pofk_norm_factor);
        }
        cosmo->set_As(cosmo->get_As() * pofk_norm_factor);

        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "Updated cosmology and gravitymodel to reflect change of As:\n";
            std::cout << "#=====================================================\n";
            cosmo->info();
            grav->info();
        }
    }

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15) << sigma8_today << "\n";
        if (ic.sigma8_normalization) {
            std::cout << "# We will normalize to Sigma8 = " << std::setw(15) << ic.sigma8 << " at z = " << std::setw(15)
                      << ic.sigma8_redshift << "\n";
            std::cout << "# New Sigma(R = 8 Mpc/h, z = 0.0 ) : " << std::setw(15)
                      << sigma8_today * std::sqrt(pofk_norm_factor) << "\n";
            std::cout << "# Sigma(R = Box/ic_nmesh, z = zini) : " << std::setw(15)
                      << sigma_of_R(boxsize / ic.nmesh, ic.initial_redshift) << "\n";
        }
        std::cout << "#=====================================================\n";
    }
}

//=============================================================================
/// Create the particles: either generate the initial conditions from a random
/// field (gaussian / non-gaussian / phases taken from a file) or read them from file
//=============================================================================
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions() {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;

    // If we simply read IC from file (useful for testing)
    if (ic.random_field_type == COLASOLVER::ICRandomFieldType::ReadParticles) {
        read_ic();
        return;
    }

    //=============================================================
    // Make the initial density field delta(k,zini)
    //=============================================================
    auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment("CIC");
    if (ic.random_field_type == COLASOLVER::ICRandomFieldType::ReadPhases) {
        // If read_phases then delta_ini is computed from particles
        nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(ic.reconstruct_assigment_method);
    }
    FFTWGrid<NDIM> delta_ini_fourier(ic.nmesh, nextra.first, nextra.second);
    delta_ini_fourier.add_memory_label("delta_ini_fourier(k)");
    delta_ini_fourier.set_grid_status_real(false);

    timer.StartTiming("InitialDensityField");
    auto rng = COLASOLVER::make_random_generator(ic);
    COLASOLVER::generate_initial_density_field<NDIM>(delta_ini_fourier,
                                                     ic,
                                                     boxsize,
                                                     power_initial_spline,
                                                     power_primordial_spline,
                                                     cosmo,
                                                     grav_ic,
                                                     rng.get());
    timer.EndTiming("InitialDensityField");

    // Take the phases from the particles in a file, but use our P(k) for the amplitudes
    if (ic.random_field_type == COLASOLVER::ICRandomFieldType::ReadPhases) {
        read_phases(delta_ini_fourier);
        COLASOLVER::whiten_and_set_power_spectrum<NDIM>(delta_ini_fourier, power_initial_spline, boxsize);
    }

    // Reverse the phases? For doing pair-fixed simulations
    if (ic.reverse_phases) {
        COLASOLVER::reverse_phases<NDIM>(delta_ini_fourier);
    }

    // Massive neutrino particles (made from the same random field)
    if (has_neutrino_particles()) {
        timer.StartTiming("InitialConditionsNeutrinos");
        generate_neutrino_particles(delta_ini_fourier);
        timer.EndTiming("InitialConditionsNeutrinos");
    }

    // Separate CDM and baryon particles
    if (has_baryons()) {
        timer.StartTiming("InitialConditions");
        generate_initial_conditions_two_species(delta_ini_fourier);
        apply_lattice_offset();
        timer.EndTiming("InitialConditions");
        return;
    }

    //=============================================================
    // Displace particles from a regular grid using LPT
    // The growth rate is used to generate the velocities
    //=============================================================
    const double aini = ic.a_initial();
    const double fac = aini * aini * cosmo->HoverH0_of_a(std::exp(std::log(aini)));
    std::vector<double> velocity_norms{fac * grav_ic->get_f_1LPT(aini),
                                       fac * grav_ic->get_f_2LPT(aini),
                                       fac * grav_ic->get_f_3LPTa(aini),
                                       fac * grav_ic->get_f_3LPTb(aini)};
    timer.StartTiming("InitialConditions");

    // Which LPT potentials we need to keep for scale-dependent COLA (max 3LPT)
    const bool store_potentials = cola.enabled and cola.scaledependent;
    const bool store_1LPT = store_potentials and FML::PARTICLE::has_get_D_1LPT<T>() and cola.LPT_order >= 1;
    const bool store_2LPT = store_potentials and FML::PARTICLE::has_get_D_2LPT<T>() and cola.LPT_order >= 2;
    const bool store_3LPT = store_potentials and FML::PARTICLE::has_get_D_3LPTa<T>() and
                            FML::PARTICLE::has_get_D_3LPTb<T>() and cola.LPT_order >= 3;

    // We get 2LPT, 3LPTa, 3LPTb depending on the size of the vector we send in
    std::vector<FFTWGrid<NDIM>> phi_nLPT_potentials;
    if (store_2LPT)
        phi_nLPT_potentials.resize(1);
    if (store_3LPT)
        phi_nLPT_potentials.resize(3);

    // In the N-body gauge the growth rate (and so the 1LPT velocity) is scale-dependent on large scales.
    // We take it from the (N-body gauge) transfer functions
    std::function<double(double)> velocity_norm_1LPT_of_kBox = nullptr;
    Spline growth_rate_of_logk{"f_cb(k,zini)"};
    if (params.force.nbody_gauge) {
        FML::assert_mpi(transferdata != nullptr, "force_nbody_gauge needs the transfer functions");
        const double kmin = transferdata->get_kmin_hmpc_splines();
        const double kmax = transferdata->get_kmax_hmpc_splines();
        const int nk = 512;
        DVector logk(nk), f(nk);
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            f[i] = transferdata->get_cdm_baryon_growth_rate(std::exp(logk[i]), aini);
        }
        growth_rate_of_logk.create(logk, f, "f_cb(k,zini)");
        velocity_norm_1LPT_of_kBox = [&, kmin, kmax](double kBox) {
            const double k = std::min(std::max(kBox / boxsize, kmin), kmax);
            return fac * growth_rate_of_logk(std::log(k));
        };
        if (FML::ThisTask == 0)
            std::cout << "# N-body gauge: growth rate at zini f(k) = " << growth_rate_of_logk(std::log(kmin)) << " (k = "
                      << kmin << " h/Mpc), " << growth_rate_of_logk(std::log(0.1)) << " (k = 0.1 h/Mpc). "
                      << "Scale-independent value " << grav_ic->get_f_1LPT(aini) << "\n";
    }

    FML::NBODY::NBodyInitialConditions<NDIM, T>(part,
                                                params.particle.Npart_1D,
                                                params.particle.allocation_factor,
                                                delta_ini_fourier,
                                                phi_nLPT_potentials,
                                                ic.LPT_order,
                                                boxsize,
                                                ic.initial_redshift,
                                                velocity_norms,
                                                velocity_norm_1LPT_of_kBox);

    apply_lattice_offset();

    // Store the potentials
    if (store_1LPT) {
        cola.phi_1LPT_ini_fourier = COLASOLVER::compute_phi_1LPT_from_delta<NDIM>(delta_ini_fourier);
        cola.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini)");
    }
    if (store_2LPT) {
        if (FML::ThisTask == 0)
            std::cout << "Storing phi_2LPT(k,zini) for scale-dependent COLA\n";
        cola.phi_2LPT_ini_fourier = phi_nLPT_potentials[0];
        cola.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini)");
    }
    if (store_3LPT) {
        if (FML::ThisTask == 0)
            std::cout << "Storing phi_3LPTa(k,zini) and phi_3LPTb(k,zini) for scale-dependent COLA\n";
        cola.phi_3LPTa_ini_fourier = phi_nLPT_potentials[1];
        cola.phi_3LPTb_ini_fourier = phi_nLPT_potentials[2];
        cola.phi_3LPTa_ini_fourier.add_memory_label("phi_3LPTa(k,zini)");
        cola.phi_3LPTb_ini_fourier.add_memory_label("phi_3LPTb(k,zini)");
    }

    timer.EndTiming("InitialConditions");
}

/// Offset the particle lattice from the nodes of the force grid. With a force grid finer than the particle
/// grid the particles would otherwise start exactly on grid nodes where CIC responds to |displacement|
/// (a rectified response) and this gives large spurious forces (see notes/force_grid_and_lattice.md).
/// This is a uniform translation of all the particles (all species) so it does not change the physics
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::apply_lattice_offset() {
    const auto & ic = params.ic;
    const bool offset_lattice =
        ic.lattice_offset == "half_force_cell" or
        (ic.lattice_offset == "fiducial" and params.force.nmesh > params.particle.Npart_1D);
    if (not offset_lattice)
        return;
    const double shift = 0.5 / double(params.force.nmesh);
    if (FML::ThisTask == 0)
        std::cout << "Offsetting the particle lattice by half a force grid cell (" << shift * params.simulation.boxsize
                  << " Mpc/h) in each direction\n";
    for_each_species([&](MPIParticles & particles) {
        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<T>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q[idim] += shift;
                    if (q[idim] >= 1.0)
                        q[idim] -= 1.0;
                }
            }
        }
        particles.communicate_particles();
    });
    if (has_neutrino_particles()) {
        for (auto & p : part_nu) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
        }
        part_nu.communicate_particles();
    }
}

/// The initial conditions with separate CDM and baryon particles. Both are made from the same random field
/// delta_cb(k,zini): delta_i(k) = delta_cb(k) R_i(k,zini) with R_i = T_i / T_cb from the transfer functions
/// (w_c R_c + w_b R_b = 1). The 1LPT velocity uses f_i = f_cb + dlog(R_i)/dlna where f_cb is the growth rate of
/// the gravity model (so the total CDM+baryon velocity is exactly the growing mode, as for one species, and the
/// relative velocity comes from the transfer functions; the growth rates computed directly from the transfer
/// functions are inaccurate near the ends of the redshift range of the table). The 2LPT displacement is that of
/// the total (CDM+baryon) field for both. By default (particle_baryon_lattice = same) the baryons start from the same
/// lattice as the CDM. With "offset" their lattice is offset by half a particle spacing in each direction (their
/// displacement field is then evaluated at the offset positions). Their IDs follow those of the CDM.
/// NB: interleaved (offset) lattices give a large spurious growth of the baryon-CDM difference on scales k > k_Ny/8
/// of the particles (and a few % too little CDM+baryon power at k ~ k_Ny/2) in this PM code; with the same lattice the
/// difference follows linear theory on all scales and the CDM+baryon P(k) is that of one species (see the README)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions_two_species(const FFTWGrid<NDIM> & delta_cb_fourier) {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const int Npart_1D = params.particle.Npart_1D;
    const double aini = ic.a_initial();
    FML::assert_mpi(transferdata != nullptr, "particle_baryons needs the transfer functions (transferinfofile)");
    if (aini < 1.0 / (1.0 + transferdata->get_zmax_splines()) * (1.0 - 1e-10))
        throw std::runtime_error("particle_baryons: the transfer functions only go up to z = " +
                                 std::to_string(transferdata->get_zmax_splines()) + " (need zini)");

    // T_i / T_cb and f_i at zini
    const double kmin = transferdata->get_kmin_hmpc_splines();
    const double kmax = transferdata->get_kmax_hmpc_splines();
    const int nk = 512;
    DVector logk(nk), ratio_cdm(nk), ratio_baryon(nk), f_cdm(nk), f_baryon(nk);
    double max_err = 0.0;
    // dlog(R_i)/dlna by finite differences over +-dx in log(a) (one-sided at the end of the table)
    const double dx = 0.05;
    const double xini = std::log(aini);
    const double xmin_table = std::log(1.0 / (1.0 + transferdata->get_zmax_splines()));
    const double xlo = std::max(xini - dx, xmin_table), xhi = xini + dx;
    auto ratios = [&](double k, double a) {
        const double Tcb = transferdata->get_cdm_baryon_transfer_function(k, a);
        return std::pair<double, double>(transferdata->get_cdm_transfer_function(k, a) / Tcb,
                                         transferdata->get_baryon_transfer_function(k, a) / Tcb);
    };
    for (int i = 0; i < nk; i++) {
        logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
        const double k = std::exp(logk[i]);
        const auto r = ratios(k, aini), rlo = ratios(k, std::exp(xlo)), rhi = ratios(k, std::exp(xhi));
        ratio_cdm[i] = r.first;
        ratio_baryon[i] = r.second;
        const double fcb = grav_ic->get_f_1LPT(aini, k / grav_ic->H0_hmpc);
        f_cdm[i] = fcb + std::log(rhi.first / rlo.first) / (xhi - xlo);
        f_baryon[i] = fcb + std::log(rhi.second / rlo.second) / (xhi - xlo);
        max_err = std::max(max_err, std::fabs(weight_cdm * ratio_cdm[i] + weight_baryon * ratio_baryon[i] - 1.0));
    }
    Spline ratio_cdm_spline(logk, ratio_cdm, "T_cdm/T_cb(k,zini)");
    Spline ratio_baryon_spline(logk, ratio_baryon, "T_b/T_cb(k,zini)");
    Spline f_cdm_spline(logk, f_cdm, "f_cdm(k,zini)");
    Spline f_baryon_spline(logk, f_baryon, "f_b(k,zini)");
    if (FML::ThisTask == 0) {
        std::cout << "# Two species IC at z = " << ic.initial_redshift << ": T_b/T_c = ";
        for (double k : {0.01, 0.1, 1.0})
            std::cout << ratio_baryon_spline(std::log(k)) / ratio_cdm_spline(std::log(k)) << " (k = " << k << ") ";
        std::cout << "\n# f_cdm, f_b at k = 0.1: " << f_cdm_spline(std::log(0.1)) << " " << f_baryon_spline(std::log(0.1))
                  << ". Max |w_c T_c/T_cb + w_b T_b/T_cb - 1| = " << max_err << "\n";
    }
    if (max_err > 1e-2)
        throw std::runtime_error("particle_baryons: OmegaCDM and Omegab do not match the transfer functions "
                                 "(w_c T_c + w_b T_b != T_cb)");

    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    const std::vector<double> velocity_norms{fac * grav_ic->get_f_1LPT(aini),
                                             fac * grav_ic->get_f_2LPT(aini),
                                             fac * grav_ic->get_f_3LPTa(aini),
                                             fac * grav_ic->get_f_3LPTb(aini)};
    auto clamp_logk = [&](double kBox) { return std::log(std::min(std::max(kBox / boxsize, kmin), kmax)); };

    auto make_species = [&](MPIParticles & particles,
                            const Spline & ratio,
                            const Spline & growth_rate,
                            double shift,
                            long long int id_offset,
                            const std::string & label) {
        if (FML::ThisTask == 0)
            std::cout << "# Making the " << label << " particles\n";
        const std::array<double, NDIM> shift_vec = [&] {
            std::array<double, NDIM> v;
            v.fill(shift);
            return v;
        }();
        // delta_i(k) on the (offset) lattice: f(q) -> f(q + shift) so that the particle created at q gets the
        // displacement at q + shift (and is then moved there)
        FFTWGrid<NDIM> delta = delta_cb_fourier;
        FFTWGrid<NDIM> delta_total = delta_cb_fourier;
        const auto Local_nx = delta.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            [[maybe_unused]] double kmag;
            [[maybe_unused]] std::array<double, NDIM> kvec;
            for (auto && fourier_index : delta.get_fourier_range(islice, islice + 1)) {
                delta.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                const auto value = delta.get_fourier_from_index(fourier_index);
                delta.set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(ratio(clamp_logk(kmag))));
            }
        }
        if (shift != 0.0) {
            FML::GRID::shift_grid_fourier<NDIM>(delta, shift_vec, +1.0);
            FML::GRID::shift_grid_fourier<NDIM>(delta_total, shift_vec, +1.0);
        }
        std::function<double(double)> velocity_norm_1LPT = [&](double kBox) {
            return fac * growth_rate(clamp_logk(kBox));
        };
        std::vector<FFTWGrid<NDIM>> no_potentials;
        FML::NBODY::NBodyInitialConditions<NDIM, T>(particles,
                                                    Npart_1D,
                                                    params.particle.allocation_factor,
                                                    delta,
                                                    no_potentials,
                                                    ic.LPT_order,
                                                    boxsize,
                                                    ic.initial_redshift,
                                                    velocity_norms,
                                                    velocity_norm_1LPT,
                                                    ic.LPT_order >= 2 ? &delta_total : nullptr);
        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<T>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q[idim] += shift;
                    if (q[idim] >= 1.0)
                        q[idim] -= 1.0;
                }
            }
            if constexpr (FML::PARTICLE::has_set_id<T>())
                FML::PARTICLE::SetID(p, FML::PARTICLE::GetID(p) + id_offset);
        }
        particles.communicate_particles();
    };
    make_species(part, ratio_cdm_spline, f_cdm_spline, 0.0, 0, "CDM");
    make_species(part_baryon,
                 ratio_baryon_spline,
                 f_baryon_spline,
                 params.particle.baryon_lattice == "offset" ? 0.5 / double(Npart_1D) : 0.0,
                 (long long int)FML::power(size_t(Npart_1D), NDIM),
                 "baryon");
}

//=============================================================================
// Massive neutrino particles (delta-f method, Elbers et al. 2021)
//=============================================================================

/// The neutrino particles. The bulk part is made like the CDM: delta_nu(k) = delta_cb(k) T_nu(k,zini) / T_cb(k,zini)
/// from the same random field, 1LPT displacements and velocities v = a^2 H f_nu(k) Psi with f_nu from the transfer
/// functions. On top of this each particle gets a thermal momentum q = y k_B T_nu0 / c in a random direction, with y
/// drawn from the Fermi-Dirac distribution (the random numbers depend only on the seed and the particle ID).
/// We store y and the linear delta_nu at the particle (needed for the delta-f weights)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_neutrino_particles(const FFTWGrid<NDIM> & delta_cb_fourier) {
    namespace NU = COLASOLVER::NEUTRINOS;
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const int Npart_1D = params.particle.neutrino_Npart_1D;
    const double aini = ic.a_initial();
    FML::assert_mpi(transferdata != nullptr, "particle_neutrinos needs the transfer functions (transferinfofile)");
    if (aini < 1.0 / (1.0 + transferdata->get_zmax_splines()) * (1.0 - 1e-10))
        throw std::runtime_error("particle_neutrinos: the transfer functions only go up to z = " +
                                 std::to_string(transferdata->get_zmax_splines()) + " (need zini)");

    // T_nu / T_cb and f_nu at zini
    const double kmin = transferdata->get_kmin_hmpc_splines();
    const double kmax = transferdata->get_kmax_hmpc_splines();
    const int nk = 512;
    DVector logk(nk), ratio(nk), f_nu(nk);
    for (int i = 0; i < nk; i++) {
        logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
        const double k = std::exp(logk[i]);
        ratio[i] = transferdata->get_massive_neutrino_transfer_function(k, aini) /
                   transferdata->get_cdm_baryon_transfer_function(k, aini);
        f_nu[i] = transferdata->get_massive_neutrino_growth_rate(k, aini);
    }
    Spline ratio_spline(logk, ratio, "T_nu/T_cb(k,zini)");
    Spline f_nu_spline(logk, f_nu, "f_nu(k,zini)");
    if (FML::ThisTask == 0) {
        std::cout << "# Neutrino particles at z = " << ic.initial_redshift << ": T_nu/T_cb, f_nu =";
        for (double k : {0.01, 0.1, 1.0})
            std::cout << " " << ratio_spline(std::log(k)) << ", " << f_nu_spline(std::log(k)) << " (k = " << k << ")";
        std::cout << "\n";
    }
    auto clamp_logk = [&](double kBox) { return std::log(std::min(std::max(kBox / boxsize, kmin), kmax)); };

    // delta_nu(k, zini)
    FFTWGrid<NDIM> delta_nu = delta_cb_fourier;
    delta_nu.add_memory_label("delta_nu(k,zini)");
    const auto Local_nx = delta_nu.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        [[maybe_unused]] double kmag;
        [[maybe_unused]] std::array<double, NDIM> kvec;
        for (auto && fourier_index : delta_nu.get_fourier_range(islice, islice + 1)) {
            delta_nu.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
            const auto value = delta_nu.get_fourier_from_index(fourier_index);
            delta_nu.set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(ratio_spline(clamp_logk(kmag))));
        }
    }

    // The bulk motion (1LPT)
    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    const std::vector<double> velocity_norms{fac * f_nu_spline(std::log(0.1)), 0.0, 0.0, 0.0};
    std::function<double(double)> velocity_norm_1LPT = [&](double kBox) {
        return fac * f_nu_spline(clamp_logk(kBox));
    };
    std::vector<FFTWGrid<NDIM>> no_potentials;
    FML::NBODY::NBodyInitialConditions<NDIM, ParticleNeutrino<NDIM>>(part_nu,
                                                                     Npart_1D,
                                                                     params.particle.allocation_factor,
                                                                     delta_nu,
                                                                     no_potentials,
                                                                     1,
                                                                     boxsize,
                                                                     ic.initial_redshift,
                                                                     velocity_norms,
                                                                     velocity_norm_1LPT);

    // The linear delta_nu at the particles
    delta_nu.fftw_c2r();
    delta_nu.communicate_boundaries();
    std::vector<FML::GRID::FloatType> delta_at_particles;
    FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, ParticleNeutrino<NDIM>>(
        delta_nu, part_nu.get_particles_ptr(), part_nu.get_npart(), delta_at_particles, "CIC");
    delta_nu.free();

    // The IDs follow those of the CDM (and baryons) and the thermal momenta
    const long long int id_offset =
        (long long int)FML::power(size_t(params.particle.Npart_1D), NDIM) * (has_baryons() ? 2 : 1);
    const double vth_norm = Tnu0_eV / mnu_eV * NU::c_code(boxsize);
    const NU::FermiDiracSampler sampler;
    auto * p = part_nu.get_particles_ptr();
    double min_delta = 1e100;
#ifdef USE_OMP
#pragma omp parallel for reduction(min : min_delta)
#endif
    for (size_t i = 0; i < part_nu.get_npart(); i++) {
        p[i].id += id_offset;
        p[i].delta_ini = float(delta_at_particles[i]);
        min_delta = std::min(min_delta, double(delta_at_particles[i]));
        NU::ParticleRNG rng(uint64_t(ic.random_seed), uint64_t(p[i].id));
        const double y = sampler(rng.uniform());
        p[i].y_ini = float(y);
        const double cos_theta = 2.0 * rng.uniform() - 1.0;
        const double phi = 2.0 * M_PI * rng.uniform();
        const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
        const double dir[3] = {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
        for (int idim = 0; idim < NDIM; idim++)
            p[i].vel[idim] += y * vth_norm * (NDIM == 3 ? dir[idim] : (idim == 0 ? 1.0 : 0.0));
    }
    FML::MinOverTasks(&min_delta);
    if (min_delta <= -1.0)
        throw std::runtime_error("particle_neutrinos: delta_nu <= -1 in the initial conditions (start earlier)");
    update_neutrino_weights();
    if (FML::ThisTask == 0)
        std::cout << "# Neutrino thermal velocity scale (T_nu0 / m_nu) c = " << vth_norm << " (code units), "
                  << vth_norm * 100.0 * boxsize << " km/s today. Min delta_nu(zini) at the particles = " << min_delta
                  << "\n";
}

/// The delta-f weights w = 1 - f0(y) / f_i where f_i = f0(y_ini) (1 + delta_ini) is the (conserved) phase-space
/// density of the particle and y the current momentum in units of k_B T_nu0 / c. The weighted density
/// (N_grid^3 / N_nu) sum w_i W(x - x_i) is then delta_nu (not 1 + delta_nu). Without delta-f all weights are 1
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::update_neutrino_weights() {
    namespace NU = COLASOLVER::NEUTRINOS;
    auto * p = part_nu.get_particles_ptr();
    const size_t npart = part_nu.get_npart();
    if (not params.particle.neutrino_deltaf) {
        for (size_t i = 0; i < npart; i++)
            p[i].weight = 1.0f;
        return;
    }
    const double inv_c = 1.0 / NU::c_code(params.simulation.boxsize);
    const double y_over_v = inv_c * mnu_eV / Tnu0_eV;
    double sum_w = 0.0, sum_w2 = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum_w, sum_w2)
#endif
    for (size_t i = 0; i < npart; i++) {
        double v2 = 0.0;
        for (int idim = 0; idim < NDIM; idim++)
            v2 += p[i].vel[idim] * p[i].vel[idim];
        const double y = std::sqrt(v2) * y_over_v;
        const double fi = NU::fermi_dirac(p[i].y_ini) * (1.0 + p[i].delta_ini);
        const double w = 1.0 - NU::fermi_dirac(y) / fi;
        p[i].weight = float(w);
        sum_w += w;
        sum_w2 += w * w;
    }
    FML::SumOverTasks(&sum_w);
    FML::SumOverTasks(&sum_w2);
    const double N = double(part_nu.get_npart_total());
    if (FML::ThisTask == 0)
        std::cout << "[Neutrinos] delta-f weights: mean " << sum_w / N << " rms " << std::sqrt(sum_w2 / N) << "\n";
}

template <int NDIM, class T>
double NBodySimulation<NDIM, T>::neutrino_shotnoise() const {
    const double N = double(part_nu.get_npart_total());
    if (not params.particle.neutrino_deltaf)
        return 1.0 / N;
    double sum_w2 = 0.0;
    for (auto & p : const_cast<FML::PARTICLE::MPIParticles<ParticleNeutrino<NDIM>> &>(part_nu))
        sum_w2 += double(p.weight) * double(p.weight);
    FML::SumOverTasks(&sum_w2);
    return sum_w2 / (N * N);
}

/// Kick the neutrinos with the relativistic factor (1 + 2u^2) / sqrt(1 + u^2) at the time of the force
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::kick_neutrinos(std::array<FFTWGrid<NDIM>, NDIM> & force_real, const StepInfo & s) {
    namespace NU = COLASOLVER::NEUTRINOS;
    for (int idim = 0; idim < NDIM; idim++)
        force_real[idim].communicate_boundaries();
    auto * p = part_nu.get_particles_ptr();
    const size_t npart = part_nu.get_npart();
    std::array<std::vector<FML::GRID::FloatType>, NDIM> force;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, ParticleNeutrino<NDIM>>(
        force_real, p, npart, force, current_force.density_assignment_method);
    const double inv_ac = 1.0 / (s.apos * NU::c_code(params.simulation.boxsize));
    double max_u = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_u)
#endif
    for (size_t i = 0; i < npart; i++) {
        double v2 = 0.0;
        for (int idim = 0; idim < NDIM; idim++) {
            p[i].vel[idim] *= s.velocity_scale;
            v2 += p[i].vel[idim] * p[i].vel[idim];
        }
        const double u2 = v2 * inv_ac * inv_ac;
        max_u = std::max(max_u, u2);
        const double R = NU::kick_factor(u2);
        for (int idim = 0; idim < NDIM; idim++)
            p[i].vel[idim] -= force[idim][i] * s.delta_time_kick * R;
    }
    FML::MaxOverTasks(&max_u);
    if (FML::ThisTask == 0)
        std::cout << "[Kick neutrinos] max u = q/(amc) = " << std::sqrt(max_u) << "\n";
    update_neutrino_weights();
}

/// Drift the neutrinos: dx = v dt_drift / sqrt(1 + u^2) with u evaluated at the middle of the drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::drift_neutrinos(const StepInfo & s) {
    namespace NU = COLASOLVER::NEUTRINOS;
    auto * p = part_nu.get_particles_ptr();
    const size_t npart = part_nu.get_npart();
    const double amid = 0.5 * (s.apos + s.apos_new);
    const double inv_ac = 1.0 / (amid * NU::c_code(params.simulation.boxsize));
    double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
    for (size_t i = 0; i < npart; i++) {
        double v2 = 0.0;
        for (int idim = 0; idim < NDIM; idim++)
            v2 += p[i].vel[idim] * p[i].vel[idim];
        const double fac = s.delta_time_drift * NU::drift_factor(v2 * inv_ac * inv_ac);
        for (int idim = 0; idim < NDIM; idim++) {
            const double disp = p[i].vel[idim] * fac;
            max_disp = std::max(max_disp, std::fabs(disp));
            double x = p[i].pos[idim] + disp;
            x -= std::floor(x);
            if (x >= 1.0)
                x = 0.0;
            p[i].pos[idim] = x;
        }
    }
    FML::MaxOverTasks(&max_disp);
    if (FML::ThisTask == 0)
        std::cout << "[Drift neutrinos] Max displacement: " << max_disp << "\n";
    part_nu.communicate_particles();
}

/// Assign the (weighted) neutrino particles to a grid and add them to the density field: on input the CDM+baryon
/// density delta_cb(k), on output delta_m = (1 - f_nu) delta_cb + f_nu delta_nu. Also bins up P_nu(k)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::add_neutrino_particles_to_density(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const auto & force = current_force;
    if (FML::ThisTask == 0)
        std::cout << "Adding the neutrino particles to the densityfield\n";
    FFTWGrid<NDIM> density_nu(density_grid_fourier.get_nmesh(),
                              density_grid_fourier.get_n_extra_slices_left(),
                              density_grid_fourier.get_n_extra_slices_right());
    FML::INTERPOLATION::particles_to_grid(part_nu.get_particles_ptr(),
                                          part_nu.get_npart(),
                                          part_nu.get_npart_total(),
                                          density_nu,
                                          force.density_assignment_method);
    density_nu.fftw_r2c();
    // Without the delta-f weights the density is 1 + delta
    if (FML::ThisTask == 0)
        density_nu.set_fourier_from_index(0, 0.0);

    if (params.pofk.every_step) {
        PowerSpectrumBinning pofk(density_nu.get_nmesh() / 2);
        FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(density_nu, pofk, force.density_assignment_method);
        if (params.pofk.subtract_shotnoise) {
            const double shotnoise = neutrino_shotnoise();
            for (int i = 0; i < pofk.n; i++)
                pofk.pofk[i] -= shotnoise;
        }
        pofk.scale(params.simulation.boxsize);
        pofk_nu_every_step.push_back({1.0 / a - 1.0, pofk});
    }

    const auto fnu = FML::GRID::FloatType(cosmo->get_fMNu());
    const auto Local_nx = density_grid_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
            const auto dcb = density_grid_fourier.get_fourier_from_index(fourier_index);
            const auto dnu = density_nu.get_fourier_from_index(fourier_index);
            density_grid_fourier.set_fourier_from_index(fourier_index, dcb * (FML::GRID::FloatType(1.0) - fnu) + dnu * fnu);
        }
    }
}

/// Read the N-body gauge source, check it and store the initial density field of the particles
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_nbody_gauge() {
    const auto & force = params.force;
    if (grav->get_name() != "GR" or cosmo->get_name() != "LCDM")
        throw std::runtime_error("force_nbody_gauge is only implemented for cosmology_model = LCDM and gravity_model "
                                 "= GR (the source is computed with CLASS)");
    FML::assert_mpi(transferdata != nullptr, "force_nbody_gauge needs the transfer functions");

    if (FML::ThisTask == 0)
        std::cout << "# Running in the N-body gauge. Reading the linear source [" << force.nbody_gauge_filename << "]\n";
    nbody_gauge = std::make_shared<COLASOLVER::NBodyGaugeSource>();
    nbody_gauge->read(force.nbody_gauge_filename);
    nbody_gauge->check_cosmology(*cosmo, params.ic.initial_redshift, params.simulation.boxsize, force.nmesh);

    if (FML::ThisTask == 0) {
        const double aini = params.ic.a_initial();
        for (double a : {aini, 0.5, 1.0}) {
            const auto ratio = nbody_gauge->ratio_spline(a, aini, *transferdata);
            std::cout << "# N-body gauge source: delta_extra(k,z) / delta_cb(k,zini) at z = " << 1.0 / a - 1.0 << ":";
            for (double k : {1e-3, 1e-2, 1e-1})
                std::cout << " " << ratio(std::log(k)) << " (k = " << k << ")";
            std::cout << "\n";
        }
    }

    // delta_cb(k,zini) of the particles on the force grid
    const auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force.density_assignment_method);
    initial_density_field_fourier = FFTWGrid<NDIM>(force.nmesh, nleftright.first, nleftright.second);
    initial_density_field_fourier.add_memory_label("density(k,zini)");
    FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                          part.get_npart(),
                                          part.get_npart_total(),
                                          initial_density_field_fourier,
                                          force.density_assignment_method);
    initial_density_field_fourier.fftw_r2c();
}

/// Compute delta(k) from particles in gadget files (used to take the phases from them)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_phases(FFTWGrid<NDIM> & delta_fourier) {
    COLASOLVER::delta_from_gadget_particles<NDIM, T>(delta_fourier, params.ic, params.particle.allocation_factor);
}

/// Read the particles from gadget files. With COLA we also reconstruct the LPT displacement fields
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_ic() {
    COLASOLVER::read_ic_from_gadget<NDIM, T>(part,
                                             params,
                                             cosmo,
                                             grav_ic,
                                             cola.phi_1LPT_ini_fourier,
                                             cola.phi_2LPT_ini_fourier);
}

//=============================================================================
// Running the simulation
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::run() {
    timer.StartTiming("Timestepping");

    const auto & output_redshifts = params.output.redshifts;
    const double zini = params.ic.initial_redshift;
    const auto stepper = time_stepping();

    // If we only give the total number of steps then compute how many steps between outputs
    std::vector<int> nsteps = params.timestep.nsteps;
    if (nsteps.size() == 1)
        nsteps = stepper.steps_between_outputs(zini, output_redshifts, nsteps[0]);
    stepper.print_steps(zini, output_redshifts, nsteps);

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Starting main time-stepping loop\n";
        std::cout << "#=====================================================\n\n";
    }

    int istep_total = 0;
    for (size_t ioutput = 0; ioutput < output_redshifts.size(); ioutput++) {

        // The steps to take till the next output
        const double amin = (ioutput == 0) ? 1.0 / (1.0 + zini) : 1.0 / (1.0 + output_redshifts[ioutput - 1]);
        const double amax = 1.0 / (1.0 + output_redshifts[ioutput]);
        const auto steps = stepper.steps(amin, amax, nsteps[ioutput]);

        if (nsteps[ioutput] > 0) {
            for (int istep = 0; istep <= nsteps[ioutput]; istep++) {
                // The last step is just a sync step so don't count towards the total
                if (istep < nsteps[ioutput])
                    istep_total++;

                if (FML::ThisTask == 0) {
                    std::cout << std::setprecision(5) << "\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "# Step " << istep_total << " (" << istep << " / " << nsteps[ioutput]
                              << " till output " << ioutput << ")\n";
                    std::cout << "# Pos: " << steps[istep].apos << " -> " << steps[istep].apos_new
                              << "  Vel: " << steps[istep].avel << " -> " << steps[istep].avel_new << "\n";
                    std::cout << "#=====================================================\n";
                }

                const auto & s = steps[istep];
                step({s.apos, s.apos_new, s.avel_new, s.delta_time_drift, s.delta_time_kick, s.velocity_scale});
            }
        }

        analyze_and_output(ioutput, output_redshifts[ioutput]);
    }
    timer.EndTiming("Timestepping");

    // Lightcone create kappa maps
    if (lightcone->lightcone_active()) {
        lightcone->create_weak_lensing_maps();
    }

    timer.EndTiming("The whole simulation");
    if (FML::ThisTask == 0) {
        timer.PrintAllTimings();
    }

#ifdef MEMORY_LOGGING
    // Simulation is over, output the memory usage (of what we log)
    FML::MemoryLog::get()->print();
#endif
}

/// Choose the force settings for a step where the force is evaluated at scale factor a: the early-time
/// settings (force_early_method) for z > force_early_until_redshift, otherwise the normal ones
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::set_force_settings_for_step(double a) {
    const auto & f = params.force;
    current_force = f;
    const bool early = f.early_method != "none" and 1.0 / a - 1.0 > f.early_until_redshift;
    if (early and f.early_method == "nufft") {
        current_force.density_method = "nufft";
        current_force.interpolation_method = "nufft";
        current_force.interlacing_order = 1;
        current_force.deconvolve_window = false;
    }
    if (early and f.early_method == "interlaced") {
        current_force.density_method = "assignment";
        current_force.interpolation_method = "assignment";
        current_force.interlacing_order = f.early_interlacing_order;
        current_force.density_assignment_method = f.early_density_assignment_method;
        current_force.deconvolve_window = true;
    }
    FML::NBODY::FIDUCIAL_DECONVOLVE_FORCE = current_force.deconvolve_window;
    if (early and FML::ThisTask == 0)
        std::cout << "Using the early-time force (" << f.early_method << ") at z = " << 1.0 / a - 1.0 << "\n";
}

/// One kick-drift step: density -> force -> kick -> (COLA kick-drift) -> lightcone -> drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::step(const StepInfo & s) {
    const auto & sim = params.simulation;
    set_force_settings_for_step(s.apos);
    const auto & force = current_force;

    // Force interlacing: average the force over copies of the particles shifted relative to the grid
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force.density_assignment_method);
    if (force.interlacing_order > 1 and s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        kick_with_interlaced_force(s);
        return drift_and_lightcone(s);
    }

    // Compute total density field
    FFTWGrid<NDIM> density_grid_fourier(force.nmesh, nleftright.first, nleftright.second);
    if (s.delta_time_kick != 0.0) {
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");
    }

    // Compute forces and kick particles (updates velocity)
    std::array<FFTWGrid<NDIM>, NDIM> force_real;
    if (s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("ComputeForce");
        grav->compute_force(s.apos,
                            grav->H0_hmpc * sim.boxsize,
                            density_grid_fourier,
                            force.density_assignment_method,
                            force_real);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        if (s.velocity_scale != 1.0) {
            for_each_species([&](MPIParticles & particles) {
                auto * p = particles.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < particles.get_npart(); i++) {
                    auto * vel = FML::PARTICLE::GetVel(p[i]);
                    for (int idim = 0; idim < NDIM; idim++)
                        vel[idim] *= s.velocity_scale;
                }
            });
        }
        if (force.interpolation_method == "nufft") {
            // Exact evaluation of the Fourier series of the force at the particle positions
            FML::NUFFT::ESKernel kernel(force.nufft_epsilon);
            std::vector<FML::GRID::FloatType> force_particle;
            auto * p = part.get_particles_ptr();
            for (int idim = 0; idim < NDIM; idim++) {
                force_real[idim].fftw_r2c();
                FML::NUFFT::nufft_type2<NDIM>(force_real[idim], p, part.get_npart(), force_particle, kernel);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < part.get_npart(); i++)
                    FML::PARTICLE::GetVel(p[i])[idim] -= force_particle[i] * s.delta_time_kick;
            }
        } else {
            for_each_species([&](MPIParticles & particles) {
                FML::NBODY::KickParticles<NDIM>(
                    force_real, particles, s.delta_time_kick, force.density_assignment_method);
            });
        }
        if (has_neutrino_particles())
            kick_neutrinos(force_real, s);
        timer.EndTiming("Kick");
    }
    drift_and_lightcone(s);
}

/// The rest of a step after the (PM) kick: COLA kick-drift, lightcone and drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::drift_and_lightcone(const StepInfo & s) {
    const auto & sim = params.simulation;
    if (sim.enforce_LPT_trajectories and FML::ThisTask == 0) {
        std::cout << "Enforcing LPT trajectories so we do not compute forces and apply (non COLA) KICK/DRIFT "
                     "operators\n";
    }

    // For COLA we can do the kick and drift at the same time
    if (cola.enabled) {
        timer.StartTiming("COLA");
        cola.kick_drift(part, s.apos, s.apos_new, s.delta_time_kick, s.delta_time_drift);
        if (not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Build the lightcone (with the true velocities of the particles)
    if (lightcone->lightcone_active() and s.delta_time_drift != 0.0) {
        ScopedLPTVelocity<NDIM, T> true_velocities(cola, part, s.avel_new, &timer);
        timer.StartTiming("Lightcone");
        lightcone->create_lightcone(part, s.apos, s.apos_new, s.avel_new, s.delta_time_drift);
        timer.EndTiming("Lightcone");
    }

    // Drift particles (updates positions)
    if (s.delta_time_drift != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("Drift");
        for_each_species(
            [&](MPIParticles & particles) { FML::NBODY::DriftParticles<NDIM, T>(particles, s.delta_time_drift); });
        if (has_neutrino_particles())
            drift_neutrinos(s);
        timer.EndTiming("Drift");
    }

    // Show info about particles and memory use
    for_each_species([&](MPIParticles & particles) { particles.info(); });
    if (has_neutrino_particles())
        part_nu.info();
    FML::print_system_memory_use();
}

/// The kick with force interlacing: the force is computed for force_interlacing_order copies of the particles
/// shifted by a fraction of a cell relative to the grid (see FML::INTERPOLATION::generate_interlacing_shifts),
/// interpolated to the shifted particles and averaged. This cancels the leading aliasing terms of both the
/// density assignment and the force interpolation
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::kick_with_interlaced_force(const StepInfo & s) {
    const auto & sim = params.simulation;
    const auto & force = current_force;
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force.density_assignment_method);
    const auto shifts =
        FML::INTERPOLATION::generate_interlacing_shifts<NDIM>(force.interlacing_order, 1.0 / double(force.nmesh));
    // With one task the shifted particles can be wrapped in all directions. Otherwise not in x (they stay on
    // their task and the grid has one extra slice on the right for the ones that end up outside the domain)
    const int wrap_type = FML::NTasks == 1 ? 1 : 2;

    auto * p = part.get_particles_ptr();
    const size_t npart = part.get_npart();
    std::array<std::vector<FML::GRID::FloatType>, NDIM> force_sum;
    for (auto & f : force_sum)
        f.assign(npart, 0.0);
    for (auto & shift : shifts) {
        FML::PARTICLE::shift_particles<T, NDIM>(p, npart, shift, +1.0, wrap_type);
        density_shift = shift;

        FFTWGrid<NDIM> density_grid_fourier(force.nmesh, nleftright.first, nleftright.second + 1);
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");

        timer.StartTiming("ComputeForce");
        std::array<FFTWGrid<NDIM>, NDIM> force_real;
        grav->compute_force(s.apos,
                            grav->H0_hmpc * sim.boxsize,
                            density_grid_fourier,
                            force.density_assignment_method,
                            force_real);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        for (int idim = 0; idim < NDIM; idim++)
            force_real[idim].communicate_boundaries();
        std::array<std::vector<FML::GRID::FloatType>, NDIM> force_particle;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(
            force_real, p, npart, force_particle, force.density_assignment_method);
        const double norm = 1.0 / double(shifts.size());
        for (int idim = 0; idim < NDIM; idim++)
            for (size_t i = 0; i < npart; i++)
                force_sum[idim][i] += force_particle[idim][i] * norm;
        timer.EndTiming("Kick");

        FML::PARTICLE::shift_particles<T, NDIM>(p, npart, shift, -1.0, wrap_type);
    }
    density_shift = {};

    // Kick: v -> velocity_scale * v - F dt
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < npart; i++) {
        auto * vel = FML::PARTICLE::GetVel(p[i]);
        for (int idim = 0; idim < NDIM; idim++)
            vel[idim] = vel[idim] * s.velocity_scale - force_sum[idim][i] * s.delta_time_kick;
    }
}

/// Assign the particles to a grid, Fourier transform it and add on the contribution from
/// massive neutrinos (if any). Also bins up P(k) of the particles (and the total P(k))
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const auto & force = current_force;
    const double boxsize = params.simulation.boxsize;
    const double ic_initial_redshift = params.ic.initial_redshift;
    const bool pofk_subtract_shotnoise = params.pofk.subtract_shotnoise;

    if (FML::ThisTask == 0) {
        std::cout << (has_baryons() ? "Adding the CDM and baryon particles to the densityfield\n" :
                                      "Adding particles (baryons+CDM) to the densityfield\n");
    }

    // Particles to grid and Fourier transform (or NUFFT / sheet interpolation)
    if (force.density_method == "nufft") {
        // The exact Fourier modes (no window function, no aliasing) for |m_i| < Nmesh/2
        FML::NUFFT::particles_to_fourier_grid_nufft<NDIM>(part.get_particles_ptr(),
                                                          part.get_npart(),
                                                          part.get_npart_total(),
                                                          density_grid_fourier,
                                                          force.nufft_epsilon);
    } else if (force.density_method == "sheet" or force.density_method == "sheet_nufft") {
        if constexpr (FML::PARTICLE::has_get_q<T>()) {
            FML::INTERPOLATION::particles_to_fourier_grid_sheet<NDIM>(
                part.get_particles_ptr(),
                part.get_npart(),
                params.particle.Npart_1D,
                force.sheet_refinement,
                density_grid_fourier,
                force.density_method == "sheet" ? force.density_assignment_method : std::string("NUFFT"),
                force.nufft_epsilon);
        } else {
            FML::assert_mpi(false, "force_density_method = sheet needs particles with a Lagrangian position q");
        }
    } else if (not has_baryons()) {
        FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                              part.get_npart(),
                                              part.get_npart_total(),
                                              density_grid_fourier,
                                              force.density_assignment_method);
        density_grid_fourier.fftw_r2c();
    } else {
        // CDM and baryons: delta_cb = w_c delta_c + w_b delta_b (and P(k) of each species)
        FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                              part.get_npart(),
                                              part.get_npart_total(),
                                              density_grid_fourier,
                                              force.density_assignment_method);
        density_grid_fourier.fftw_r2c();
        FFTWGrid<NDIM> density_baryon(density_grid_fourier.get_nmesh(),
                                      density_grid_fourier.get_n_extra_slices_left(),
                                      density_grid_fourier.get_n_extra_slices_right());
        FML::INTERPOLATION::particles_to_grid(part_baryon.get_particles_ptr(),
                                              part_baryon.get_npart(),
                                              part_baryon.get_npart_total(),
                                              density_baryon,
                                              force.density_assignment_method);
        density_baryon.fftw_r2c();
        if (params.pofk.every_step) {
            auto bin_species = [&](const FFTWGrid<NDIM> & grid, size_t npart_total, auto & list) {
                PowerSpectrumBinning pofk(grid.get_nmesh() / 2);
                FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(grid, pofk, force.density_assignment_method);
                if (params.pofk.subtract_shotnoise)
                    for (int i = 0; i < pofk.n; i++)
                        pofk.pofk[i] -= 1.0 / double(npart_total);
                pofk.scale(boxsize);
                list.push_back({1.0 / a - 1.0, pofk});
            };
            bin_species(density_grid_fourier, part.get_npart_total(), pofk_cdm_every_step);
            bin_species(density_baryon, part_baryon.get_npart_total(), pofk_baryon_every_step);
        }
        const auto wc = FML::GRID::FloatType(weight_cdm);
        const auto wb = FML::GRID::FloatType(weight_baryon);
        const auto Local_nx = density_grid_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
                const auto dc = density_grid_fourier.get_fourier_from_index(fourier_index);
                const auto db = density_baryon.get_fourier_from_index(fourier_index);
                density_grid_fourier.set_fourier_from_index(fourier_index, dc * wc + db * wb);
            }
        }
    }

    // P(k) from the density field: deconvolve the assignment window (not needed for the NUFFT). The shot noise
    // (if subtracted) is 1/N for one species and w_c^2/N_c + w_b^2/N_b for CDM + baryons
    auto bin_up_pofk = [&](PowerSpectrumBinning & pofk) {
        if (force.density_method == "nufft" or force.density_method == "sheet_nufft")
            FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(density_grid_fourier, pofk);
        else
            FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(
                density_grid_fourier, pofk, force.density_assignment_method);
    };
    auto subtract_cb_shotnoise = [&](PowerSpectrumBinning & pofk) {
        if (params.pofk.subtract_shotnoise) {
            double shotnoise = 1.0 / double(part.get_npart_total());
            if (has_baryons())
                shotnoise = weight_cdm * weight_cdm / double(part.get_npart_total()) +
                            weight_baryon * weight_baryon / double(part_baryon.get_npart_total());
            for (int i = 0; i < pofk.n; i++)
                pofk.pofk[i] -= shotnoise;
        }
    };

    // Bin up power-spectrum (its basically free as we have the density field)
    const double redshift = 1.0 / a - 1.0;
    PowerSpectrumBinning pofk_particles(density_grid_fourier.get_nmesh() / 2);
    pofk_particles.subtract_shotnoise = pofk_subtract_shotnoise;
    // With force interlacing the particles might be shifted relative to the grid: then we have the linear
    // fields below in the shifted frame and do not bin up P(k) again
    const bool shifted = std::any_of(density_shift.begin(), density_shift.end(), [](double x) { return x != 0.0; });
    FFTWGrid<NDIM> initial_density_field_shifted;
    auto initial_density = [&]() -> const FFTWGrid<NDIM> & {
        if (not shifted)
            return initial_density_field_fourier;
        if (not initial_density_field_shifted) {
            // The particles were moved by +s so delta_s(k) = exp(-ik.s) delta(k)
            initial_density_field_shifted = initial_density_field_fourier;
            FML::GRID::shift_grid_fourier<NDIM>(initial_density_field_shifted, density_shift, -1.0);
        }
        return initial_density_field_shifted;
    };
    const bool pofk_every_step = params.pofk.every_step;
    if (not shifted and pofk_every_step) {
        bin_up_pofk(pofk_particles);
        subtract_cb_shotnoise(pofk_particles);
        pofk_particles.scale(boxsize);
        pofk_cb_every_step.push_back({redshift, pofk_particles});
    }

    //=============================================================
    // N-body gauge: add the linear source from radiation, neutrinos and GR
    // delta_extra(k,a) = delta_cb(k,aini) T_extra(k,a) / T_cb(k,aini)
    //=============================================================
    if (nbody_gauge) {
        if (FML::ThisTask == 0)
            std::cout << "Adding the N-body gauge linear source (radiation, neutrinos, GR) to the densityfield\n";
        const double aini = 1.0 / (1.0 + ic_initial_redshift);
        const Spline ratio = nbody_gauge->ratio_spline(a, aini, *transferdata);
        const double logkmin = std::log(nbody_gauge->get_kmin_hmpc());
        const double logkmax = std::log(nbody_gauge->get_kmax_hmpc());

        // The force is normalized to OmegaM = OmegaCB + OmegaMNu so the source is (1-fMNu)(delta_cb + delta_extra)
        const double fMNu = cosmo->get_fMNu();
        const auto & delta_ini_grid = initial_density();
        auto Local_nx = density_grid_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            [[maybe_unused]] double kmag;
            [[maybe_unused]] std::array<double, NDIM> kvec;
            for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
                density_grid_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                const double logk = std::min(std::max(std::log(kmag / boxsize), logkmin), logkmax);
                const double r = kmag > 0.0 ? ratio(logk) : 0.0;
                auto delta_ini = delta_ini_grid.get_fourier_from_index(fourier_index);
                auto deltaCB = density_grid_fourier.get_fourier_from_index(fourier_index);
                auto delta = (deltaCB + delta_ini * FML::GRID::FloatType(r)) * FML::GRID::FloatType(1.0 - fMNu);
                density_grid_fourier.set_fourier_from_index(fourier_index, delta);
            }
        }

        // P(k) of the total source (the same caveat about deconvolution as for the neutrinos below)
        if (not shifted and pofk_every_step) {
            PowerSpectrumBinning pofk_total(density_grid_fourier.get_nmesh() / 2);
            pofk_total.subtract_shotnoise = pofk_subtract_shotnoise;
            bin_up_pofk(pofk_total);
            pofk_total.scale(boxsize);
            pofk_total_every_step.push_back({redshift, pofk_total});
        }
        return;
    }

    //=============================================================
    // Massive neutrino particles: delta_m = (1 - f_nu) delta_cb + f_nu delta_nu
    //=============================================================
    if (has_neutrino_particles()) {
        add_neutrino_particles_to_density(density_grid_fourier, a);
        if (shifted or not pofk_every_step)
            return;
        PowerSpectrumBinning pofk_total(density_grid_fourier.get_nmesh() / 2);
        pofk_total.subtract_shotnoise = pofk_subtract_shotnoise;
        bin_up_pofk(pofk_total);
        if (pofk_subtract_shotnoise) {
            const double fnu = cosmo->get_fMNu();
            double shotnoise_cb = 1.0 / double(part.get_npart_total());
            if (has_baryons())
                shotnoise_cb = weight_cdm * weight_cdm / double(part.get_npart_total()) +
                               weight_baryon * weight_baryon / double(part_baryon.get_npart_total());
            const double shotnoise = (1.0 - fnu) * (1.0 - fnu) * shotnoise_cb + fnu * fnu * neutrino_shotnoise();
            for (int i = 0; i < pofk_total.n; i++)
                pofk_total.pofk[i] -= shotnoise;
        }
        pofk_total.scale(boxsize);
        pofk_total_every_step.push_back({redshift, pofk_total});
        return;
    }

    //=============================================================
    // Add on contribution from massive neutrinos.
    // We need to have transfer functions for this and for that we
    // check [transferdata] which is created if "transferinfofile" is used
    //=============================================================
    if (not(force.linear_massive_neutrinos and cosmo->get_fMNu() > 0.0 and transferdata))
        return;

    // First step we store the initial density field
    if (not initial_density_field_fourier) {
        initial_density_field_fourier = density_grid_fourier;
        initial_density_field_fourier.add_memory_label("density(k,zini)");
        FML::assert_mpi(std::fabs(redshift - ic_initial_redshift) < 1e-3, "This should not happen");
    }

    if (FML::ThisTask == 0) {
        std::cout << "Adding on the massive neutrinos to the densityfield using linear theory\n";
    }

    // Function to translate delta_cb(zini,k) -> delta_nu(z,k) using transfer functions
    const double aini = 1.0 / (1.0 + ic_initial_redshift);
    const double koverkBox = 1.0 / boxsize;
    auto norm = [&](double kBox) {
        double T_mnu = transferdata->get_massive_neutrino_transfer_function(kBox * koverkBox, a);
        double T_cb_ini = transferdata->get_cdm_baryon_transfer_function(kBox * koverkBox, aini);
        return (T_mnu / T_cb_ini);
    };

    // We compute the total matter density-field deltaM = (OmegaCB deltaCB + OmegaMNu deltaMNu)/OmegaM
    const double fMNu = cosmo->get_fMNu();
    const auto & delta_ini_grid = initial_density();
    auto Local_nx = initial_density_field_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        [[maybe_unused]] double kmag;
        [[maybe_unused]] std::array<double, NDIM> kvec;
        for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
            density_grid_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
            auto delta_ini = delta_ini_grid.get_fourier_from_index(fourier_index);
            auto deltaCB = density_grid_fourier.get_fourier_from_index(fourier_index);
            auto deltaM =
                deltaCB * FML::GRID::FloatType(1.0 - fMNu) + delta_ini * FML::GRID::FloatType(norm(kmag) * fMNu);
            density_grid_fourier.set_fourier_from_index(fourier_index, deltaM);
        }
    }

    //=============================================================
    // Bin up total power-spectrum (its basically free)
    // XXX: the density field is here the sum of W*deltaCB + deltaNu
    // so the last part is going to be added up as deltaNu/W^2 so
    // its not fully correct for large k, but this is not a high
    // quality P(k) anyway... should improve this at some point
    // One way is to bin up <(W*deltaCB)^2>, <deltaNu^2> and the cross
    // <(W*deltaCB) deltaNu> and deconvolve them separately
    //=============================================================
    if (shifted or not pofk_every_step)
        return;
    PowerSpectrumBinning pofk_total(density_grid_fourier.get_nmesh() / 2);
    pofk_total.subtract_shotnoise = pofk_subtract_shotnoise;
    bin_up_pofk(pofk_total);
    pofk_total.scale(boxsize);
    pofk_total_every_step.push_back({redshift, pofk_total});
}

/// Do all the on the fly analysis that is turned on and write the output
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::analyze_and_output(int ioutput, double redshift) {

    // Create snapshot folder
    const std::string snapshot_folder =
        params.output.path("snapshot_" + params.simulation.name + "_z" + redshift_to_string(redshift));
    if (not FML::create_folder(snapshot_folder)) {
        throw std::runtime_error("Failed to create snapshot folder [" + snapshot_folder + "]");
    }

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Analyze and output\n";
        std::cout << "#=====================================================\n";
        std::cout << "Doing output " << ioutput << " at redshift z = " << redshift << "\n";
    }

    // In the COLA frame the velocities are relative to LPT. While this object exists the
    // particles have their true velocities
    ScopedLPTVelocity<NDIM, T> true_velocities(cola, part, 1.0 / (1.0 + redshift), &timer);

    // Every step we (might) have computed Pcb(k) and possibly Ptotal(k). Output these to file
    // (only the steps since the last output: the files of earlier steps are already written)
    if (pofk_cb_every_step.size() > 0 or pofk_total_every_step.size() > 0) {
        output_pofk_for_every_step(*this);
        pofk_cb_every_step.clear();
        pofk_total_every_step.clear();
        pofk_cdm_every_step.clear();
        pofk_baryon_every_step.clear();
        pofk_nu_every_step.clear();
    }

    struct Analysis {
        std::string name;
        bool enabled;
        std::function<void()> run;
    };
    const std::vector<Analysis> analyses{
        {"Power-spectrum",
         params.pofk.enabled,
         [&]() { pofk_every_output.push_back({redshift, compute_power_spectrum(*this, redshift, snapshot_folder)}); }},
        {"Power-spectrum multipoles",
         params.pofk_multipole.enabled,
         [&]() {
             pofk_multipoles_every_output.push_back(
                 {redshift, compute_power_spectrum_multipoles(*this, redshift, snapshot_folder)});
         }},
        {"FOF", params.fof.enabled, [&]() { compute_fof_halos(*this, redshift, snapshot_folder); }},
        {"Bispectrum", params.bispectrum.enabled, [&]() { compute_bispectrum(*this, redshift, snapshot_folder); }},
        {"Output particles",
         params.output.particles,
         [&]() {
             if (params.output.fileformat == COLASOLVER::ParticleFileFormat::GADGET)
                 output_gadget(*this, redshift, snapshot_folder);
             else
                 output_gadget_hdf5(*this, redshift, snapshot_folder);
         }},
    };
    for (auto & analysis : analyses) {
        if (not analysis.enabled)
            continue;
        timer.StartTiming(analysis.name);
        analysis.run();
        timer.EndTiming(analysis.name);
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::free() {
    part.free();
    part_baryon.free();
    part_nu.free();
    initial_density_field_fourier.free();
    cola.free();
}

#endif
