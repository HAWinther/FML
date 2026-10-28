#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

#include "ParameterDefaults.h"

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
#include "BaryonificationOutput.h"
#include "COLA.h"
#include "Cosmology.h"
#include "ExternalIC.h"
#include "GravityModel.h"
#include "GravityModel_GR.h"
#include "InitialConditions.h"
#include "InitialPowerSpectrum.h"
#include "KickDrift.h"
#include "Lightcone/Lightcone.h"
#include "NBodyGauge.h"
#include "NeutrinoParticles.h"
#include "Particles.h"
#include "SimulationParameters.h"
#include "Species.h"
#include "TimeStepping.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
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
///   InitialConditions.h    : the initial density field and making the particles of a species from it
///   ExternalIC.h           : starting from particles read from file
///   COLA.h                 : the COLA method (ColaState)
///   Species.h              : the particle species (CDM+baryons, or CDM and baryons, and neutrinos)
///   NeutrinoParticles.h    : massive neutrinos as particles (delta-f method)
///   KickDrift.h            : the kick and drift of a species (non-relativistic or relativistic)
///   SimulationDensity.h    : the density field the force is computed from (part of this class)
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

    /// The particle species (see Species.h): "cb", or "cdm" and "baryon", and "nu"
    std::vector<COLASOLVER::SpeciesInfo> species;
    /// All the particles (CDM+baryons as one species "cb", or the CDM particles if particle_baryons)
    MPIParticles part;
    /// The baryon particles (only if particle_baryons)
    MPIParticles part_baryon;
    /// The massive neutrino particles (only if particle_neutrinos; delta-f method)
    COLASOLVER::NeutrinoParticles<NDIM> neutrinos;

    /// Phi/c^2 in real space for the lightcone (plc_relativistic_effects), computed at scale factor
    /// lightcone_potential_a from the density field of the step
    FFTWGrid<NDIM> lightcone_potential;
    double lightcone_potential_a{-1.0};
    void compute_lightcone_potential(const FFTWGrid<NDIM> & density_grid_fourier, double a);

    /// The initial density field delta_cb(zini,k) computed with the same Nmesh
    /// as the forces (used for linear massive neutrinos and the N-body gauge - not allocated otherwise)
    FFTWGrid<NDIM> initial_density_field_fourier;
    // ...and in the frame of the particles when they are shifted (only during compute_density_field_fourier)
    FFTWGrid<NDIM> initial_density_field_shifted;
    /// The linear density field delta_cb(zini,k) the particles were displaced with, on a grid of the size of the
    /// particle lattice (only if hybridbias)
    FFTWGrid<NDIM> hybridbias_linear_field_fourier;
    // Shift of the particles relative to the grid when computing the density (force interlacing)
    std::array<double, NDIM> density_shift{};
    // The force settings used in the current step (params.force or the early-time settings)
    COLASOLVER::ForceSettings current_force;
    FML::NBODY::ForceOptions force_options;
    void set_force_settings_for_step(double a);

    /// The linear N-body gauge source (if force_nbody_gauge)
    std::shared_ptr<COLASOLVER::NBodyGaugeSource> nbody_gauge;

    /// COLA settings and the LPT potentials at the initial time
    ColaState<NDIM, T> cola;
    // The initial velocities are already relative to the COLA frame (scale-dependent growth with scale-independent
    // COLA: the residual 1LPT velocity (f(k) - f(0)) Psi), so they are not reset to zero
    bool ic_velocities_relative_to_cola_frame{false};

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
    // ...and of the species separately when there are several (cdm, baryon, nu)
    std::map<std::string, std::vector<std::pair<double, PowerSpectrumBinning>>> pofk_species_every_step;
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
    /// Massive neutrinos as particles (delta-f method)?
    bool has_neutrino_particles() const { return params.particle.neutrinos; }
    COLASOLVER::NeutrinoParticles<NDIM> & get_neutrinos() { return neutrinos; }
    typename COLASOLVER::NeutrinoParticles<NDIM>::Container & get_neutrino_particles() { return neutrinos.particles(); }

    /// The particle species (in the order of the IDs)
    const std::vector<COLASOLVER::SpeciesInfo> & get_species() const { return species; }
    /// Call f(particles, info) for the CDM+baryon species ("cb", or "cdm" and "baryon"; all of type T)
    template <class F>
    void for_each_cb_species(F && f) {
        for (auto & info : species) {
            if (info.name == "cb" or info.name == "cdm")
                f(part, info);
            else if (info.name == "baryon")
                f(part_baryon, info);
        }
    }
    /// Call f(particles, info) for all the species (the particle types differ, so use a generic lambda)
    template <class F>
    void for_each_species(F && f) {
        for_each_cb_species(f);
        for (auto & info : species)
            if (info.name == "nu")
                f(neutrinos.particles(), info);
    }
    /// The shot noise of the density field of a species (in units of the box volume): 1/N (sum w^2 / N^2 for the
    /// delta-f neutrinos). Collective
    double species_shotnoise(const COLASOLVER::SpeciesInfo & info) {
        if (info.name == "nu")
            return neutrinos.shotnoise();
        return 1.0 / double((info.name == "baryon" ? part_baryon : part).get_npart_total());
    }
    /// The shot noise of delta_cb = sum_i w_i delta_i: sum_i w_i^2 / N_i
    double cb_shotnoise() {
        double shotnoise = 0.0;
        for_each_cb_species([&](MPIParticles & particles, const COLASOLVER::SpeciesInfo & info) {
            shotnoise += info.weight_cb * info.weight_cb / double(particles.get_npart_total());
        });
        return shotnoise;
    }
    /// The shot noise of the total delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with neutrino particles (with
    /// force_neutrino_energy_density and a > 0: of (1 - f_nu) delta_cb + f_nu <E/m>(a) delta_nu, the source of
    /// the Poisson equation). Collective
    double total_shotnoise(double a = -1.0) {
        const double fnu = cosmo->get_fMNu();
        if (params.force.neutrino_energy_density and a > 0.0) {
            typename COLASOLVER::NeutrinoParticles<NDIM>::ScopedEnergyWeights energy_weights(neutrinos, a, true);
            const double fnu_eff = fnu * neutrino_energy_factor(a);
            return (1.0 - fnu) * (1.0 - fnu) * cb_shotnoise() + fnu_eff * fnu_eff * neutrinos.shotnoise();
        }
        return (1.0 - fnu) * (1.0 - fnu) * cb_shotnoise() + fnu * fnu * neutrinos.shotnoise();
    }
    /// The mean energy per mass of the neutrinos <E/m>(a) with force_neutrino_energy_density, otherwise 1
    double neutrino_energy_factor(double a) const {
        if (not params.force.neutrino_energy_density)
            return 1.0;
        return COLASOLVER::NEUTRINOS::mean_energy_factor(a,
                                                         cosmo->get_Mnu_eV() / cosmo->get_N_nu(),
                                                         cosmo->get_neutrino_temperature_eV(1.0));
    }
    /// Use the energy density of the neutrinos (force_neutrino_energy_density)
    bool neutrino_energy_density() const { return params.force.neutrino_energy_density; }
    std::shared_ptr<Cosmology> get_cosmology() const { return cosmo; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model() const { return grav; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model_ic() const { return grav_ic; }
    std::shared_ptr<LinearTransferData> get_transferdata() const { return transferdata; }
    const Spline & get_power_initial_spline() const { return power_initial_spline; }
    const auto & get_pofk_cb_every_step() const { return pofk_cb_every_step; }
    const auto & get_pofk_total_every_step() const { return pofk_total_every_step; }
    const auto & get_pofk_species_every_step() const { return pofk_species_every_step; }
    const auto & get_pofk_every_output() const { return pofk_every_output; }
    const auto & get_pofk_multipoles_every_output() const { return pofk_multipoles_every_output; }
    /// The linear density field delta_cb(zini,k) of the initial conditions (only if hybridbias)
    const FFTWGrid<NDIM> & get_hybridbias_linear_field() const { return hybridbias_linear_field_fourier; }

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
    // The density field (SimulationDensity.h)
    void compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void assign_cb_particles_to_grid(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_nbody_gauge_source(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_neutrino_particles(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_linear_neutrinos(FFTWGrid<NDIM> & density_grid_fourier, double a);
    const FFTWGrid<NDIM> & initial_density_in_particle_frame();
    bool particles_shifted() const {
        return std::any_of(density_shift.begin(), density_shift.end(), [](double x) { return x != 0.0; });
    }
    void bin_up_power_spectrum_every_step(const FFTWGrid<NDIM> & grid,
                                          bool deconvolve,
                                          double shotnoise,
                                          double a,
                                          std::vector<std::pair<double, PowerSpectrumBinning>> & list);
    void analyze_and_output(int ioutput, double redshift);
};

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_parameters(ParameterMap & param) {
    parameters = std::make_shared<ParameterMap>(param);
    params.read(param);
    if (params.baryonification.enabled) {
        if (NDIM != 3 || grav->get_name() != "GR" || cosmo->get_OmegaMNu() != 0.0)
            throw std::runtime_error(
                "baryonification currently requires NDIM=3, gravity_model=GR and cosmology_OmegaMNu=0");
        params.baryonification.profile_model(cosmo->get_Omegab() / cosmo->get_OmegaM()).validate();
    }

    // We cannot use COLA if the particle type is not compatible with it
    if (params.simulation.use_cola and not FML::PARTICLE::has_get_D_1LPT<T>()) {
        throw std::runtime_error("Set to use COLA, but particle-type does not even have 1LPT growthfactors!");
    }

    force_options.laplace_kernel = FML::NBODY::laplace_kernel_from_name(params.force.greens_function_kernel);

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
    force_options.gradient_kernel = FML::NBODY::gradient_kernel_from_name(gradient_kernel);
    force_options.deconvolve = params.force.deconvolve_window;
    current_force = params.force;

    // The (experimental) finite difference force: only the stencils 2, 4 and 6 are implemented and the gradient is
    // the finite difference stencil, so an explicitly chosen Fourier gradient kernel would be ignored
    if (COLASOLVER::get_parameter<bool>(param, "force_use_finite_difference_force")) {
        const int stencil_order = COLASOLVER::get_parameter<int>(param, "force_finite_difference_stencil_order");
        if (not FML::NBODY::is_valid_finite_difference_stencil_order(stencil_order))
            throw std::runtime_error("Parameterfile: force_finite_difference_stencil_order = " +
                                     std::to_string(stencil_order) + " is not implemented. Valid options are: 2 4 6");
        if (params.force.gradient_kernel != "fiducial" and params.force.gradient_kernel != "")
            throw std::runtime_error("Parameterfile: force_gradient_kernel = " + params.force.gradient_kernel +
                                     " is not used with force_use_finite_difference_force = true (the gradient is "
                                     "the finite difference stencil set by force_finite_difference_stencil_order). "
                                     "Use force_gradient_kernel = fiducial");
    }

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

    // Starting from the Big Bang the force at early times (particles close to the lattice) has to be accurate: the
    // CIC and interlaced forces make the modes with k > ~k_Ny/8 grow wrongly there (see README)
    if (params.ic.initial_redshift > 100.0 and params.force.early_method == "none" and
        params.force.density_method == "assignment" and params.force.interpolation_method != "nufft" and
        FML::ThisTask == 0)
        std::cout << "WARNING: starting at z = " << params.ic.initial_redshift
                  << " without force_early_method: use force_early_method = nufft (force_early_until_redshift ~ 10) "
                     "for an accurate force while the particles are close to the lattice (see README)\n";

    // The baryon-CDM relative mode in the displacements of the particles (a ~50 kpc/h CDM-baryon offset per pair)
    // gets a spurious discrete-mode growth once the force grid resolves it, and is partly lost in collapsed regions
    // (P(delta_c - delta_b) ~ 0.5 of linear theory at z = 0 on large scales). See README
    // (also the part of it that grows after zini with mass_ini)
    if (params.particle.baryons and params.particle.baryon_relative_mode != "mass" and
        params.force.nmesh > 2 * params.particle.Npart_1D and FML::ThisTask == 0)
        std::cout << "WARNING: particle_baryon_relative_mode = " << params.particle.baryon_relative_mode
                  << " with force_nmesh > 2 particle_Npart_1D: the baryon-CDM relative mode is not reliable (use "
                     "particle_baryon_relative_mode = mass, see README)\n";

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

    // The particle species (CDM+baryons as one, or CDM and baryons, and massive neutrinos)
    if (has_neutrino_particles() and not(cosmo->get_OmegaMNu() > 0.0 and cosmo->get_Mnu_eV() > 0.0))
        throw std::runtime_error("particle_neutrinos = true needs massive neutrinos (cosmology_OmegaMNu > 0)");
    species = COLASOLVER::make_species_list(params.particle, *cosmo, NDIM);
    if (FML::ThisTask == 0 and species.size() > 1) {
        std::cout << "# Particle species (name, Npart_1D, GADGET type, Omega_i / OmegaM, weight in delta_cb):\n";
        for (auto & info : species)
            std::cout << "#   " << info.name << " " << info.Npart_1D << " " << info.gadget_type << " "
                      << info.mass_fraction << " " << info.weight_cb << "\n";
    }
    if (has_neutrino_particles()) {
        const double mnu_eV = cosmo->get_Mnu_eV() / cosmo->get_N_nu();
        const double Tnu0_eV = cosmo->get_neutrino_temperature_eV(1.0);
        neutrinos.init(mnu_eV, Tnu0_eV, params.simulation.boxsize, params.particle.neutrino_deltaf);
        if (FML::ThisTask == 0)
            std::cout << "# Massive neutrinos as particles (delta-f weights: " << std::boolalpha
                      << params.particle.neutrino_deltaf << "). m_nu = " << mnu_eV << " eV (x" << cosmo->get_N_nu()
                      << "), T_nu0 = " << Tnu0_eV << " eV, f_nu = " << cosmo->get_fMNu() << "\n";
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

    // In the COLA frame v=0 so reset velocities. With CDM and baryons as separate particles the initial conditions
    // already have the velocities relative to the shared COLA frame (the baryon-CDM relative velocity)
    if (cola.enabled and not has_baryons() and not ic_velocities_relative_to_cola_frame) {
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

    // The splines are constant beyond the table, so modes of the IC grid with k > k_max of the input would silently get
    // P(k_max). Only a warning: the power there is often negligible (and a table only slightly too short is common)
    const double kmax_grid = std::sqrt(double(NDIM)) * M_PI * ic.nmesh / boxsize;
    if (FML::ThisTask == 0 and ic.random_field_type != COLASOLVER::ICRandomFieldType::ReadParticles and
        not table.k.empty() and table.k.back() < kmax_grid)
        std::cout << "WARNING: the input P(k) / T(k) only goes to k = " << table.k.back()
                  << " h/Mpc, but the IC grid has modes up to sqrt(NDIM) pi ic_nmesh / simulation_boxsize = "
                  << kmax_grid << " h/Mpc. P(k) is taken to be constant (P(k_max)) beyond the table\n";

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

    // Keep the linear field for the hybrid bias spectra (the Lagrangian operators are formed at the lattice points so
    // we only need the modes the lattice resolves)
    if (params.hybridbias.enabled) {
        hybridbias_linear_field_fourier = FFTWGrid<NDIM>(params.particle.Npart_1D);
        hybridbias_linear_field_fourier.add_memory_label("delta_L(k,zini) for hybrid bias");
        FML::NUFFT::copy_fourier_modes<NDIM>(delta_ini_fourier,
                                             hybridbias_linear_field_fourier,
                                             [](const std::array<int, NDIM> &) { return 1.0; });
    }

    // Massive neutrino particles (made from the same random field)
    if (has_neutrino_particles()) {
        timer.StartTiming("InitialConditionsNeutrinos");
        FML::assert_mpi(transferdata != nullptr, "particle_neutrinos needs the transfer functions (transferinfofile)");
        const auto & nu = species.back();
        neutrinos.generate(delta_ini_fourier,
                           *transferdata,
                           ic.initial_redshift,
                           cosmo->HoverH0_of_a(ic.a_initial()),
                           nu.Npart_1D,
                           params.particle.allocation_factor,
                           ic.random_seed,
                           nu.id_offset);
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
        // With COLA the velocities are relative to the COLA frame, which has the growth rate of the gravity model
        // (at k = 0 for scale-independent COLA) for 1LPT and all of the 2LPT velocity. Zeroing the velocities (as for
        // scale-independent growth) would replace f_nbg(k) by f(0) (0.75% too large velocity on the largest scales)
        const bool relative_to_cola_frame = cola.enabled;
        velocity_norm_1LPT_of_kBox = [&, kmin, kmax, relative_to_cola_frame](double kBox) {
            const double k = std::min(std::max(kBox / boxsize, kmin), kmax);
            const double f_frame = not relative_to_cola_frame ? 0.0 :
                                   cola.scaledependent        ? grav->get_f_1LPT(aini, kBox / boxsize / grav->H0_hmpc) :
                                                                grav->get_f_1LPT(aini);
            return fac * (growth_rate_of_logk(std::log(k)) - f_frame);
        };
        if (relative_to_cola_frame)
            ic_velocities_relative_to_cola_frame = true;
        if (FML::ThisTask == 0)
            std::cout << "# N-body gauge: growth rate at zini f(k) = " << growth_rate_of_logk(std::log(kmin))
                      << " (k = " << kmin << " h/Mpc), " << growth_rate_of_logk(std::log(0.1)) << " (k = 0.1 h/Mpc). "
                      << "Scale-independent value " << grav_ic->get_f_1LPT(aini) << "\n";
    }

    // Scale-dependent growth (massive neutrinos with the transfer functions, modified gravity): the 1LPT velocity
    // is f(k, zini) Psi (the growth rate at k = 0 for all k overestimated the velocity: for 0.4 eV neutrinos
    // f = 0.998 instead of f_cb = 0.976 for k >> k_fs, which gave +1.7% in P_cb(z = 0) on linear scales). With
    // scale-dependent COLA the COLA frame already has this velocity; with scale-independent COLA it has f(k = 0) Psi
    // and the particles get the residual (f(k) - f(0)) Psi
    if (not params.force.nbody_gauge and grav_ic->is_growth_scaledependent() and
        not(cola.enabled and cola.scaledependent)) {
        const double f0 = cola.enabled ? grav->get_f_1LPT(aini) : 0.0;
        velocity_norm_1LPT_of_kBox = [&, f0](double kBox) {
            return fac * (grav_ic->get_f_1LPT(aini, kBox / boxsize / grav_ic->H0_hmpc) - f0);
        };
        if (cola.enabled)
            ic_velocities_relative_to_cola_frame = true;
        if (FML::ThisTask == 0)
            std::cout << "# Scale-dependent growth rate at zini f(k) = "
                      << grav_ic->get_f_1LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                      << grav_ic->get_f_1LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                      << grav_ic->get_f_1LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                      << grav_ic->get_f_1LPT(aini) << "\n";
    }

    // With COLA the velocities in the particles are relative to the COLA frame, which only has the LPT orders up to
    // simulation_cola_LPT_order (1LPT, 2LPT). The higher orders (2LPT with simulation_cola_LPT_order = 1, 3LPT) keep
    // their full IC velocity: zeroing all the velocities (cola_initialize_velocities) dropped them (1.7% rms in the
    // velocity at z = 20 with ic_LPT_order = 2 and simulation_cola_LPT_order = 1)
    if (cola.enabled and ic.LPT_order > cola.LPT_order)
        ic_velocities_relative_to_cola_frame = true;
    if (ic_velocities_relative_to_cola_frame) {
        // velocity_norms = {1LPT, 2LPT, 3LPTa, 3LPTb}: the frame has the first cola.LPT_order (<= 2) of them
        for (int i = 0; i < cola.LPT_order; i++)
            velocity_norms[i] = 0.0;
    }

    // The same for the 2LPT velocity: f_2LPT(k, zini) Psi_2LPT with k the wavenumber of the 2LPT field (as in the
    // COLA frame of scale-dependent COLA, which therefore needs nothing extra). Using f_2LPT(k = 0) for all k gave
    // velocities that differed from those of scale-dependent COLA by 4.3e-4 rms for 0.4 eV neutrinos. With
    // scale-independent COLA the frame has f_2LPT(0) Psi_2LPT and the particles get the residual. In the N-body gauge
    // (above) only the 1LPT growth rate differs from that of the gravity model (on horizon scales, from the transfer
    // functions); the 2LPT velocity is that of the gravity model in both cases (as in the COLA frame).
    // With simulation_cola_LPT_order = 1 the frame has no 2LPT velocity and the particles get all of it
    std::function<double(double)> velocity_norm_2LPT_of_kBox = nullptr;
    const bool cola_frame_has_2LPT = cola.enabled and cola.LPT_order >= 2;
    if (grav_ic->is_growth_scaledependent() and ic.LPT_order >= 2 and
        (not cola_frame_has_2LPT or not cola.scaledependent)) {
        const double f2_frame = cola_frame_has_2LPT ? grav->get_f_2LPT(aini) : 0.0;
        velocity_norm_2LPT_of_kBox = [&, f2_frame](double kBox) {
            return fac * (grav_ic->get_f_2LPT(aini, kBox / boxsize / grav_ic->H0_hmpc) - f2_frame);
        };
        // (with COLA the velocities are already relative to the frame, see above)
        FML::assert_mpi(not cola.enabled or ic_velocities_relative_to_cola_frame,
                        "Internal error: scale-dependent 2LPT velocity with COLA must be relative to the COLA frame");
        if (FML::ThisTask == 0)
            std::cout << "# Scale-dependent 2LPT growth rate at zini f_2(k) = "
                      << grav_ic->get_f_2LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                      << grav_ic->get_f_2LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                      << grav_ic->get_f_2LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                      << grav_ic->get_f_2LPT(aini) << "\n";
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
                                                velocity_norm_1LPT_of_kBox,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                velocity_norm_2LPT_of_kBox);

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
    const bool offset_lattice = ic.lattice_offset == "half_force_cell" or
                                (ic.lattice_offset == "fiducial" and params.force.nmesh > params.particle.Npart_1D);
    if (not offset_lattice)
        return;
    const double shift = 0.5 / double(params.force.nmesh);
    // Scale-dependent COLA interpolates its LPT potentials (of the lattice before the offset) at q - shift
    cola.lattice_offset = shift;
    if (FML::ThisTask == 0)
        std::cout << "Offsetting the particle lattice by half a force grid cell (" << shift * params.simulation.boxsize
                  << " Mpc/h) in each direction\n";
    for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo &) {
        using P = std::remove_reference_t<decltype(*particles.get_particles_ptr())>;
        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<P>()) {
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
}

/// The initial conditions with separate CDM and baryon particles (see two_species_ic_splines and
/// generate_species_from_delta_cb in InitialConditions.h). The 2LPT displacement is that of the total (CDM+baryon)
/// field for both. By default (particle_baryon_lattice = same) the baryons start from the same lattice as the CDM.
/// With "offset" their lattice is offset by half a particle spacing in each direction. Their IDs follow those of the
/// CDM.
/// NB: interleaved (offset) lattices give a large spurious growth of the baryon-CDM difference on scales k > k_Ny/8
/// of the particles (and a few % too little CDM+baryon power at k ~ k_Ny/2) in this PM code; with the same lattice the
/// difference follows linear theory on all scales and the CDM+baryon P(k) is that of one species (see the README)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions_two_species(const FFTWGrid<NDIM> & delta_cb_fourier) {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const double aini = ic.a_initial();
    FML::assert_mpi(transferdata != nullptr, "particle_baryons needs the transfer functions (transferinfofile)");
    const auto & cdm = species[0];
    const auto & baryon = species[1];
    const auto splines = COLASOLVER::two_species_ic_splines<NDIM>(*transferdata,
                                                                  *grav_ic,
                                                                  ic.initial_redshift,
                                                                  cdm.weight_cb,
                                                                  baryon.weight_cb);

    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    const std::vector<double> velocity_norms{fac * grav_ic->get_f_1LPT(aini),
                                             fac * grav_ic->get_f_2LPT(aini),
                                             fac * grav_ic->get_f_3LPTa(aini),
                                             fac * grav_ic->get_f_3LPTb(aini)};
    const double kmin = transferdata->get_kmin_hmpc_splines();
    const double kmax = transferdata->get_kmax_hmpc_splines();
    auto clamp_logk = [&](double kBox) { return std::log(std::min(std::max(kBox / boxsize, kmin), kmax)); };
    // The baryon-CDM relative mode: in the displacement (each species its own displacement and velocity), the
    // masses m = 1 + eps (mass: shared CDM+baryon flow, eps = the linear relative mode today; mass_ini: shared
    // displacement, eps at zini and each species its own velocity)
    const std::string & relative_mode = params.particle.baryon_relative_mode;
    const bool mass_mode = relative_mode == "mass" or relative_mode == "mass_ini";
    std::pair<Spline, Spline> eps_today;
    if (relative_mode == "mass") {
        eps_today = COLASOLVER::two_species_mass_contrast_today(*transferdata, ic.initial_redshift);
        if (FML::ThisTask == 0) {
            std::cout << "# Mass-encoded relative mode: eps_b / delta_cb(zini) = ";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << eps_today.second(std::log(k)) << " (k = " << k << ") ";
            std::cout << "\n";
        }
    }
    auto make_species = [&](MPIParticles & particles,
                            const COLASOLVER::SpeciesInfo & info,
                            const Spline & ratio,
                            const Spline & growth_rate,
                            const Spline * eps_of_logk,
                            double shift) {
        if (FML::ThisTask == 0)
            std::cout << "# Making the " << info.name << " particles\n";
        // With COLA the species share the COLA frame of the total CDM+baryon field and the velocities we set are
        // relative to it: v_i - v_frame = a^2 H [(f_i - f_frame / R_i) Psi_i] for 1LPT (Psi_i = R_i Psi_cb) and zero
        // for 2LPT (the same for all species; all of the 2LPT velocity with simulation_cola_LPT_order = 1 as the frame
        // then has none, see generate_initial_conditions). The frame velocity is f_frame Psi_cb with f_frame = f_cb(k)
        // for scale-dependent COLA, but the growth rate at k = 0 for scale-independent COLA (the frame adds back
        // grav->get_f_1LPT(a), as for one species). Using f_cb(k) there with scale-dependent growth gave the
        // velocities (f_cb(k) - f(0)) Psi_cb too little (-1.75% rms for 0.4 eV neutrinos)
        std::vector<double> residual_velocity_norms = velocity_norms;
        for (int i = 0; i < cola.LPT_order; i++)
            residual_velocity_norms[i] = 0.0;
        auto f_frame = [&](double logk) {
            return cola.scaledependent ? grav_ic->get_f_1LPT(aini, std::exp(logk) / grav_ic->H0_hmpc) :
                                         grav->get_f_1LPT(aini);
        };
        std::function<double(double)> velocity_norm_1LPT = [&](double kBox) {
            const double logk = clamp_logk(kBox);
            if (not cola.enabled)
                return fac * growth_rate(logk);
            return fac * (growth_rate(logk) - f_frame(logk) / ratio(logk));
        };
        // The 2LPT velocity with scale-dependent growth: f_2LPT(k) Psi_2LPT (relative to the frame with f_2LPT(0) for
        // scale-independent COLA, nothing for scale-dependent COLA whose frame has f_2LPT(k); see
        // generate_initial_conditions; all of it with simulation_cola_LPT_order = 1 where the frame has no 2LPT)
        std::function<double(double)> velocity_norm_2LPT = nullptr;
        const bool cola_frame_has_2LPT = cola.enabled and cola.LPT_order >= 2;
        if (grav_ic->is_growth_scaledependent() and ic.LPT_order >= 2 and
            (not cola_frame_has_2LPT or not cola.scaledependent)) {
            const double f2_frame = cola_frame_has_2LPT ? grav->get_f_2LPT(aini) : 0.0;
            velocity_norm_2LPT = [&, f2_frame](double kBox) {
                return fac * (grav_ic->get_f_2LPT(aini, kBox / boxsize / grav_ic->H0_hmpc) - f2_frame);
            };
        }
        std::function<double(double)> displacement_ratio = [&](double kBox) { return ratio(clamp_logk(kBox)); };
        std::function<double(double)> mass_contrast = nullptr;
        if (mass_mode) {
            // Both species are displaced by Psi_cb. mass: the CDM+baryon velocity f_cb Psi_cb. mass_ini: the velocity
            // of the species v_i = f_i R_i Psi_cb. With COLA relative to the frame velocity f_frame Psi_cb
            displacement_ratio = [](double) { return 1.0; };
            velocity_norm_1LPT = [&](double kBox) {
                const double logk = clamp_logk(kBox);
                const double f_cb = grav_ic->get_f_1LPT(aini, std::exp(logk) / grav_ic->H0_hmpc);
                const double f_i = relative_mode == "mass" ? f_cb : growth_rate(logk) * ratio(logk);
                return fac * (cola.enabled ? f_i - f_frame(logk) : f_i);
            };
            if (relative_mode == "mass")
                mass_contrast = [&](double kBox) { return (*eps_of_logk)(clamp_logk(kBox)); };
            else
                mass_contrast = [&](double kBox) { return ratio(clamp_logk(kBox)) - 1.0; };
        }
        COLASOLVER::generate_species_from_delta_cb<NDIM, T>(particles,
                                                            delta_cb_fourier,
                                                            displacement_ratio,
                                                            velocity_norm_1LPT,
                                                            cola.enabled ? residual_velocity_norms : velocity_norms,
                                                            ic.LPT_order,
                                                            info.Npart_1D,
                                                            params.particle.allocation_factor,
                                                            boxsize,
                                                            ic.initial_redshift,
                                                            shift,
                                                            info.id_offset,
                                                            cola.enabled,
                                                            mass_contrast,
                                                            velocity_norm_2LPT);
    };
    make_species(part, cdm, splines.ratio_cdm, splines.f_cdm, &eps_today.first, 0.0);
    const double baryon_shift = params.particle.baryon_lattice == "offset" ? 0.5 / double(baryon.Npart_1D) : 0.0;
    make_species(part_baryon, baryon, splines.ratio_baryon, splines.f_baryon, &eps_today.second, baryon_shift);

    // Scale-dependent COLA: the LPT potentials of the total CDM+baryon field (the shared COLA frame)
    if (cola.enabled and cola.scaledependent) {
        if (FML::PARTICLE::has_get_D_1LPT<T>() and cola.LPT_order >= 1) {
            cola.phi_1LPT_ini_fourier = COLASOLVER::compute_phi_1LPT_from_delta<NDIM>(delta_cb_fourier);
            cola.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini)");
        }
        if (FML::PARTICLE::has_get_D_2LPT<T>() and cola.LPT_order >= 2 and ic.LPT_order >= 2) {
            FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_cb_fourier, cola.phi_2LPT_ini_fourier);
            cola.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini)");
        }
        // The baryons on the offset lattice: the potentials of the shifted field, as for their IC (see
        // ColaState::ShiftedSpecies)
        if (baryon_shift != 0.0) {
            if (FML::ThisTask == 0)
                std::cout << "Storing the LPT potentials of the offset baryon lattice for scale-dependent COLA\n";
            typename ColaState<NDIM, T>::ShiftedSpecies sp;
            sp.part = &part_baryon;
            sp.shift = baryon_shift;
            FFTWGrid<NDIM> delta_shifted = delta_cb_fourier;
            std::array<double, NDIM> shift_vec;
            shift_vec.fill(baryon_shift);
            FML::GRID::shift_grid_fourier<NDIM>(delta_shifted, shift_vec, +1.0);
            if (cola.phi_1LPT_ini_fourier) {
                sp.phi_1LPT_ini_fourier = COLASOLVER::compute_phi_1LPT_from_delta<NDIM>(delta_shifted);
                sp.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini) baryons");
            }
            if (cola.phi_2LPT_ini_fourier) {
                FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_shifted, sp.phi_2LPT_ini_fourier);
                sp.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini) baryons");
            }
            cola.shifted_species.push_back(std::move(sp));
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
        std::cout << "# Running in the N-body gauge. Reading the linear source [" << force.nbody_gauge_filename
                  << "]\n";
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
    const auto nleftright =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(force.density_assignment_method);
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
    COLASOLVER::delta_from_gadget_particles<NDIM, T>(delta_fourier, params, *cosmo);
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

    // With the LPT fields reconstructed from the density field the COLA frame velocity is only an approximation to the
    // velocities in the file (smoothed, from the Eulerian density). Keep the velocities of the file: store them
    // relative to the frame instead of replacing them by it. (The exact reconstruction reproduces them, so nothing to
    // do there)
    if (cola.enabled and not params.ic.reconstruct_exact) {
        cola.add_on_LPT_velocity(part, params.ic.a_initial(), -1.0);
        ic_velocities_relative_to_cola_frame = true;
    }
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
    if (nsteps.size() == 1) {
        nsteps = stepper.steps_between_outputs(zini, output_redshifts, nsteps[0]);
        // An output with no steps would write the particles of the previous output at another redshift
        for (size_t i = 0; i < nsteps.size(); i++)
            if (not(nsteps[i] > 0 or (i == 0 and output_redshifts[0] == zini)))
                throw std::runtime_error("Splitting timestep_nsteps over the outputs gives 0 steps before output z = " +
                                         std::to_string(output_redshifts[i]) +
                                         ": use more steps or give the number of steps for every output");
    }
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

                // Reorder the particles in memory by position (faster density assignment and interpolation)
                const int sort_every = params.particle.sort_every_nsteps;
                if (sort_every > 0 and istep < nsteps[ioutput] and istep_total % sort_every == 0) {
                    timer.StartTiming("SortParticles");
                    const int nblocks = std::max(1, params.force.nmesh / 4);
                    for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo &) {
                        particles.sort_by_position(nblocks);
                    });
                    timer.EndTiming("SortParticles");
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
    force_options.deconvolve = current_force.deconvolve_window;
    if (early and FML::ThisTask == 0)
        std::cout << "Using the early-time force (" << f.early_method << ") at z = " << 1.0 / a - 1.0 << "\n";
}

/// One kick-drift step: density -> force -> kick -> COLA kick -> lightcone -> COLA drift -> drift
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
        // The potential for the lightcone (from the same density field as the force)
        if (lightcone->needs_potential(s.apos, s.apos_new))
            compute_lightcone_potential(density_grid_fourier, s.apos);
    }

    // Compute forces and kick particles (updates velocity)
    std::array<FFTWGrid<NDIM>, NDIM> force_real;
    if (s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("ComputeForce");
        grav->compute_force(s.apos,
                            grav->H0_hmpc * sim.boxsize,
                            density_grid_fourier,
                            force.density_assignment_method,
                            force_real,
                            force_options);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        if (force.interpolation_method == "nufft") {
            // Exact evaluation of the Fourier series of the force at the particle positions (one species)
            auto * p = part.get_particles_ptr();
            if (s.velocity_scale != 1.0) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < part.get_npart(); i++) {
                    auto * vel = FML::PARTICLE::GetVel(p[i]);
                    for (int idim = 0; idim < NDIM; idim++)
                        vel[idim] *= s.velocity_scale;
                }
            }
            FML::NUFFT::ESKernel kernel(force.nufft_epsilon);
            std::vector<FML::GRID::FloatType> force_particle;
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
            for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo & info) {
                COLASOLVER::kick_particles<NDIM>(particles,
                                                 info,
                                                 force_real,
                                                 s.apos,
                                                 s.delta_time_kick,
                                                 s.velocity_scale,
                                                 force.density_assignment_method,
                                                 sim.boxsize);
            });
        }
        // The delta-f weights depend on the momenta
        if (has_neutrino_particles())
            neutrinos.update_weights();
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

    // For COLA we can do the kick and drift at the same time. The lightcone drifts the particles from apos with
    // their true velocity (which includes the LPT velocity), so then the COLA drift (the LPT displacement over
    // the step) is done after the lightcone
    const bool make_lightcone = lightcone->lightcone_active() and s.delta_time_drift != 0.0;
    if (cola.enabled) {
        timer.StartTiming("COLA");
        // The COLA kick subtracts the LPT force from the COLA velocities (which get the PM force). When enforcing the
        // LPT trajectories there is no PM kick, so there must be no COLA kick either: the COLA velocities stay zero
        // and the particles have the LPT velocity (with it the velocities were close to the initial LPT velocity)
        const double delta_time_kick_cola = sim.enforce_LPT_trajectories ? 0.0 : s.delta_time_kick;
        for_each_cb_species([&](MPIParticles & particles, const COLASOLVER::SpeciesInfo &) {
            if (make_lightcone)
                cola.kick(particles, s.apos, s.apos_new, delta_time_kick_cola);
            else
                cola.kick_drift(particles, s.apos, s.apos_new, delta_time_kick_cola, s.delta_time_drift);
        });
        if (not make_lightcone and not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Build the lightcone (with the true velocities of the particles)
    if (make_lightcone) {
        ScopedLPTVelocity<NDIM, T> true_velocities(cola, part, s.avel_new, &timer);
        timer.StartTiming("Lightcone");
        if (lightcone->needs_potential(s.apos, s.apos_new)) {
            // The potential of this step (if not computed with the force, e.g. with force interlacing: from the
            // CDM+baryon density only) at the particles
            if (lightcone_potential_a != s.apos) {
                const auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(
                    current_force.density_assignment_method);
                FFTWGrid<NDIM> density(current_force.nmesh, nleftright.first, nleftright.second);
                assign_cb_particles_to_grid(density, s.apos);
                compute_lightcone_potential(density, s.apos);
            }
            std::vector<FML::GRID::FloatType> potential;
            FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, T>(
                lightcone_potential,
                part.get_particles_ptr(),
                part.get_npart(),
                potential,
                current_force.density_assignment_method);
            lightcone->create_lightcone(part,
                                        s.apos,
                                        s.apos_new,
                                        s.avel_new,
                                        s.delta_time_drift,
                                        &potential,
                                        &lightcone_potential,
                                        current_force.density_assignment_method);
            lightcone_potential.free();
            lightcone_potential_a = -1.0;
        } else {
            lightcone->create_lightcone(part, s.apos, s.apos_new, s.avel_new, s.delta_time_drift);
        }
        timer.EndTiming("Lightcone");
    }
    if (cola.enabled and make_lightcone) {
        timer.StartTiming("COLA");
        for_each_cb_species([&](MPIParticles & particles, const COLASOLVER::SpeciesInfo &) {
            cola.drift(particles, s.apos, s.apos_new);
        });
        if (not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Drift particles (updates positions)
    if (s.delta_time_drift != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("Drift");
        for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo & info) {
            COLASOLVER::drift_particles<NDIM>(particles, info, s.apos, s.apos_new, s.delta_time_drift, sim.boxsize);
        });
        timer.EndTiming("Drift");
    }

    // Show info about particles and memory use
    for_each_species([&](auto & particles, const COLASOLVER::SpeciesInfo &) { particles.info(); });
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
    // The positions are restored exactly after each shift (shifting back changes them in the last digit)
    const auto saved_pos = FML::PARTICLE::save_particle_positions<T, NDIM>(p, npart);
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
                            force_real,
                            force_options);
        timer.EndTiming("ComputeForce");

        timer.StartTiming("Kick");
        for (int idim = 0; idim < NDIM; idim++)
            force_real[idim].communicate_boundaries();
        std::array<std::vector<FML::GRID::FloatType>, NDIM> force_particle;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(force_real,
                                                                                   p,
                                                                                   npart,
                                                                                   force_particle,
                                                                                   force.density_assignment_method);
        const double norm = 1.0 / double(shifts.size());
        for (int idim = 0; idim < NDIM; idim++)
            for (size_t i = 0; i < npart; i++)
                force_sum[idim][i] += force_particle[idim][i] * norm;
        timer.EndTiming("Kick");

        FML::PARTICLE::restore_particle_positions<T, NDIM>(p, npart, saved_pos);
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
    std::optional<ScopedLPTVelocity<NDIM, T>> true_velocities_baryons;
    if (has_baryons())
        true_velocities_baryons.emplace(cola, part_baryon, 1.0 / (1.0 + redshift), &timer);

    // Every step we (might) have computed Pcb(k) and possibly Ptotal(k). Output these to file
    // (only the steps since the last output: the files of earlier steps are already written)
    if (pofk_cb_every_step.size() > 0 or pofk_total_every_step.size() > 0) {
        output_pofk_for_every_step(*this);
        pofk_cb_every_step.clear();
        pofk_total_every_step.clear();
        pofk_species_every_step.clear();
    }

    struct Analysis {
        std::string name;
        bool enabled;
        std::function<void()> run;
    };
    const std::vector<Analysis> analyses{
        {"Baryonification",
         params.baryonification.enabled,
         [&]() {
             if constexpr (NDIM == 3)
                 COLASOLVER::output_baryonified_snapshot(
                     COLASOLVER::copy_baryonification_particles(part.get_particles_ptr(), part.get_npart()),
                     params,
                     cosmo,
                     redshift,
                     snapshot_folder);
         }},
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
        {"Hybrid bias spectra",
         params.hybridbias.enabled,
         [&]() { compute_hybrid_bias_spectra(*this, redshift, snapshot_folder); }},
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
    neutrinos.free();
    initial_density_field_fourier.free();
    hybridbias_linear_field_fourier.free();
    cola.free();
}

// The density field the force is computed from (member functions of NBodySimulation)
#include "SimulationDensity.h"

/// Phi/c^2 in real space from the density contrast delta(k) of the step: Phi/c^2(k) = -1.5 Omega_M (H0/c)^2 delta(k)
/// / (a k^2) (the GR Poisson equation, k in h/Mpc), with the same window deconvolution as the force. Stored in
/// lightcone_potential (with the boundaries communicated, ready for interpolation)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::compute_lightcone_potential(const FFTWGrid<NDIM> & density_grid_fourier, double a) {
    timer.StartTiming("LightconePotential");
    lightcone_potential = density_grid_fourier;
    lightcone_potential.add_memory_label("Lightcone potential");
    const double boxsize = params.simulation.boxsize;
    const double c_over_H0 = 2997.92458; // Mpc/h
    const double norm = -1.5 * cosmo->get_OmegaM() / (a * c_over_H0 * c_over_H0) * boxsize * boxsize;
    const bool deconvolve = current_force.deconvolve_window;
    const FML::INTERPOLATION::WindowFunctionTable<NDIM> window(current_force.density_assignment_method,
                                                               lightcone_potential.get_nmesh());
    const auto Local_nx = lightcone_potential.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        std::array<double, NDIM> kvec;
        double kmag2;
        for (auto && index : lightcone_potential.get_fourier_range(islice, islice + 1)) {
            lightcone_potential.get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag2);
            if (kmag2 == 0.0) {
                lightcone_potential.set_fourier_from_index(index, 0.0);
                continue;
            }
            double fac = norm / kmag2;
            if (deconvolve) {
                const double W = window(kvec);
                fac /= W * W;
            }
            lightcone_potential.set_fourier_from_index(index,
                                                       lightcone_potential.get_fourier_from_index(index) *
                                                           FML::GRID::FloatType(fac));
        }
    }
    lightcone_potential.fftw_c2r();
    lightcone_potential.communicate_boundaries();
    lightcone_potential_a = a;
    timer.EndTiming("LightconePotential");
}

#endif
