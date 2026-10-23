#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
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

    /// All the particles
    MPIParticles part;

    /// The initial density field delta_cb(zini,k) computed with the same Nmesh
    /// as the forces (used for linear massive neutrinos - not allocated otherwise)
    FFTWGrid<NDIM> initial_density_field_fourier;

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
    std::shared_ptr<Cosmology> get_cosmology() const { return cosmo; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model() const { return grav; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model_ic() const { return grav_ic; }
    std::shared_ptr<LinearTransferData> get_transferdata() const { return transferdata; }
    const Spline & get_power_initial_spline() const { return power_initial_spline; }
    const auto & get_pofk_cb_every_step() const { return pofk_cb_every_step; }
    const auto & get_pofk_total_every_step() const { return pofk_total_every_step; }
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
    void read_ic();
    void read_phases(FFTWGrid<NDIM> & delta_fourier);

    // The parts of run()
    struct StepInfo {
        double apos;             // Scale factor of the positions at the start of the step
        double apos_new;         // ...and at the end of the step
        double avel_new;         // Scale factor of the velocities at the end of the step
        double delta_time_drift; // The size of the drift step
        double delta_time_kick;  // The size of the kick step
    };
    void step(const StepInfo & s);
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
    FML::NBODY::set_fiducial_gradient_kernel(params.force.gradient_kernel);

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

    FML::NBODY::NBodyInitialConditions<NDIM, T>(part,
                                                params.particle.Npart_1D,
                                                params.particle.allocation_factor,
                                                delta_ini_fourier,
                                                phi_nLPT_potentials,
                                                ic.LPT_order,
                                                boxsize,
                                                ic.initial_redshift,
                                                velocity_norms);

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
        const auto asteps = stepper.scalefactors_KDK(amin, amax, nsteps[ioutput]);
        const auto delta_time = stepper.deltatime_KDK(amin, amax, nsteps[ioutput]);

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
                    std::cout << "# Pos: " << asteps.first[istep] << " -> " << asteps.first[istep + 1]
                              << "  Vel: " << asteps.second[istep] << " -> " << asteps.second[istep + 1] << "\n";
                    std::cout << "#=====================================================\n";
                }

                step({asteps.first[istep],
                      asteps.first[istep + 1],
                      asteps.second[istep + 1],
                      delta_time.first[istep],
                      delta_time.second[istep]});
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

/// One kick-drift step: density -> force -> kick -> (COLA kick-drift) -> lightcone -> drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::step(const StepInfo & s) {
    const auto & sim = params.simulation;
    const auto & force = params.force;

    // Compute total density field
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force.density_assignment_method);
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
        FML::NBODY::KickParticles<NDIM>(force_real, part, s.delta_time_kick, force.density_assignment_method);
        timer.EndTiming("Kick");
    }
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
        FML::NBODY::DriftParticles<NDIM, T>(part, s.delta_time_drift);
        timer.EndTiming("Drift");
    }

    // Show info about particles and memory use
    part.info();
    FML::print_system_memory_use();
}

/// Assign the particles to a grid, Fourier transform it and add on the contribution from
/// massive neutrinos (if any). Also bins up P(k) of the particles (and the total P(k))
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a) {
    const auto & force = params.force;
    const double boxsize = params.simulation.boxsize;
    const double ic_initial_redshift = params.ic.initial_redshift;
    const bool pofk_subtract_shotnoise = params.pofk.subtract_shotnoise;

    if (FML::ThisTask == 0) {
        std::cout << "Adding particles (baryons+CDM) to the densityfield\n";
    }

    // Particles to grid and Fourier transform
    FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                          part.get_npart(),
                                          part.get_npart_total(),
                                          density_grid_fourier,
                                          force.density_assignment_method);
    density_grid_fourier.fftw_r2c();

    // Bin up power-spectrum (its basically free as we have the density field)
    const double redshift = 1.0 / a - 1.0;
    PowerSpectrumBinning pofk_particles(density_grid_fourier.get_nmesh() / 2);
    pofk_particles.subtract_shotnoise = pofk_subtract_shotnoise;
    FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(density_grid_fourier,
                                                                 pofk_particles,
                                                                 force.density_assignment_method);
    pofk_particles.scale(boxsize);
    pofk_cb_every_step.push_back({redshift, pofk_particles});

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
    PowerSpectrumBinning pofk_total(density_grid_fourier.get_nmesh() / 2);
    pofk_total.subtract_shotnoise = pofk_subtract_shotnoise;
    FML::CORRELATIONFUNCTIONS::bin_up_deconvolved_power_spectrum(density_grid_fourier,
                                                                 pofk_total,
                                                                 force.density_assignment_method);
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
    if (pofk_cb_every_step.size() > 0 or pofk_total_every_step.size() > 0) {
        output_pofk_for_every_step(*this);
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
                 output_fml(*this, redshift, snapshot_folder);
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
    initial_density_field_fourier.free();
    cola.free();
}

#endif
