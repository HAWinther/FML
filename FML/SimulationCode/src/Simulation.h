#ifndef SIMULATION_HEADER
#define SIMULATION_HEADER

#include "ParameterDefaults.h"

#include "TransferInput.h"
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
    using TransferProvider = SIMCODE::TransferProvider;
    using PowerSpectrumBinning = FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM>;
    using Parameters = SIMCODE::SimulationParameters;

  protected:
    /// Everything related to the background evolution
    std::shared_ptr<Cosmology> cosmo;

    /// Everything related to gravity: growth factors, computing forces
    std::shared_ptr<GravityModel<NDIM>> grav;
    /// The gravity model used to make the IC (GR if ic_use_gravity_model_GR, otherwise grav)
    std::shared_ptr<GravityModel<NDIM>> grav_ic;

    /// The transfer functions (ic_type_of_input = camb, class or internal; see TransferInput.h).
    /// The same object as in the gravity model (grav)
    std::shared_ptr<TransferProvider> transfers;

    /// A copy of the parameterfile (passed on to the models we create)
    std::shared_ptr<ParameterMap> parameters;

    /// The parameters of the simulation
    Parameters params;

    /// The particle species (see Species.h): "cb", or "cdm" and "baryon", and "nu"
    std::vector<SIMCODE::SpeciesInfo> species;
    /// All the particles (CDM+baryons as one species "cb", or the CDM particles if particle_baryons)
    MPIParticles part;
    /// The baryon particles (only if particle_baryons)
    MPIParticles part_baryon;
    /// The massive neutrino particles (only if particle_neutrinos; delta-f method)
    SIMCODE::NeutrinoParticles<NDIM> part_neutrinos;

    /// The initial density field delta_cb(zini,k) computed with the same Nmesh
    /// as the forces (used for linear massive neutrinos and the N-body gauge - not allocated otherwise)
    FFTWGrid<NDIM> initial_density_field_fourier;
    // ...with the density assignment of the early-time force (force_early_method; only while it is used)
    FFTWGrid<NDIM> initial_density_field_early;
    // ...and in the frame of the particles when they are shifted (only during compute_density_field_fourier)
    FFTWGrid<NDIM> initial_density_field_shifted;
    /// The linear density field delta_cb(zini,k) the particles were displaced with, on a grid of the size of the
    /// particle lattice (only if hybridbias)
    FFTWGrid<NDIM> hybridbias_linear_field_fourier;
    /// The loop functions of the one-loop hybrid-bias prediction with the model's kernels at all the outputs
    /// (computed at the first output: the kernels are integrated from early times once)
    std::shared_ptr<const FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::MGLoopTables> hybridbias_mg_tables;
    // Shift of the particles relative to the grid when computing the density (force interlacing)
    std::array<double, NDIM> density_shift{};
    // The force settings used in the current step (params.force or the early-time settings)
    SIMCODE::ForceSettings current_force;
    bool early_force{false}; // current_force are the early-time settings
    FML::NBODY::ForceOptions force_options;
    void set_force_settings_for_step(double a);

    /// The linear N-body gauge source (if force_nbody_gauge)
    std::shared_ptr<SIMCODE::NBodyGaugeSource> nbody_gauge;

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

    //=============================================================================
    // Some of the stuff we compute and output is small so we also keep it
    // in the class in case one wants to process it later
    //=============================================================================
    // A list of (z, P(k)) for the particles at every step (naive binning - not using the same pofk_* setting!)
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_cb_every_step;
    std::vector<std::pair<double, PowerSpectrumBinning>> pofk_total_every_step;
    // ...and of the species separately when there are several (cdm, baryon, nu)
    std::map<std::string, std::vector<std::pair<double, PowerSpectrumBinning>>> pofk_species_every_step;

  public:
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
    SIMCODE::NeutrinoParticles<NDIM> & get_neutrinos() { return part_neutrinos; }
    typename SIMCODE::NeutrinoParticles<NDIM>::Container & get_neutrino_particles() {
        return part_neutrinos.particles();
    }

    /// The particle species (in the order of the IDs)
    const std::vector<SIMCODE::SpeciesInfo> & get_species() const { return species; }
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
                f(part_neutrinos.particles(), info);
    }
    /// The shot noise of the density field of a species (in units of the box volume): 1/N (sum w^2 / N^2 for the
    /// delta-f neutrinos). Collective
    double species_shotnoise(const SIMCODE::SpeciesInfo & info) {
        if (info.name == "nu")
            return part_neutrinos.shotnoise();
        return 1.0 / double((info.name == "baryon" ? part_baryon : part).get_npart_total());
    }
    /// The shot noise of delta_cb = sum_i w_i delta_i: sum_i w_i^2 / N_i
    double cb_shotnoise() {
        double shotnoise = 0.0;
        for_each_cb_species([&](MPIParticles & particles, const SIMCODE::SpeciesInfo & info) {
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
            typename SIMCODE::NeutrinoParticles<NDIM>::ScopedEnergyWeights energy_weights(part_neutrinos, a, true);
            const double fnu_eff = fnu * neutrino_energy_factor(a);
            return (1.0 - fnu) * (1.0 - fnu) * cb_shotnoise() + fnu_eff * fnu_eff * part_neutrinos.shotnoise();
        }
        return (1.0 - fnu) * (1.0 - fnu) * cb_shotnoise() + fnu * fnu * part_neutrinos.shotnoise();
    }
    /// The mean energy per mass of the neutrinos <E/m>(a) with force_neutrino_energy_density, otherwise 1
    double neutrino_energy_factor(double a) const {
        if (not params.force.neutrino_energy_density)
            return 1.0;
        return SIMCODE::NEUTRINOS::mean_energy_factor(
            a, cosmo->get_neutrino_masses_eV(), cosmo->get_neutrino_temperature_eV(1.0));
    }
    /// Use the energy density of the neutrinos (force_neutrino_energy_density)
    bool neutrino_energy_density() const { return params.force.neutrino_energy_density; }
    std::shared_ptr<Cosmology> get_cosmology() const { return cosmo; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model() const { return grav; }
    std::shared_ptr<GravityModel<NDIM>> get_gravity_model_ic() const { return grav_ic; }
    std::shared_ptr<TransferProvider> get_transfers() const { return transfers; }
    const Spline & get_power_initial_spline() const { return power_initial_spline; }
    /// The linear CDM+baryon P(k, a) (k in h/Mpc, P in (Mpc/h)^3): P(k, zini) (D(k, a) / D(k, aini))^2 with the
    /// growth of the gravity model; without P(k, zini) the transfer functions' P_cb(k, a) (0 without them)
    double linear_pofk_cb(double k, double a) const {
        if (power_initial_spline) {
            const double D = grav->get_D_1LPT(a, k / grav->H0_hmpc);
            const double Dini = grav->get_D_1LPT(1.0 / (1.0 + params.ic.initial_redshift), k / grav->H0_hmpc);
            return power_initial_spline(k) * std::pow(D / Dini, 2);
        }
        if (transfers)
            return transfers->power(SIMCODE::TransferSpecies::CDMBaryon, k, a);
        return 0.0;
    }
    /// (T_species(k, a) / T_cb(k, a))^2 from the transfer functions (1 without them): P_species = P_cb times this
    double species_linear_ratio(const std::string & species, double k, double a) const {
        if (not transfers)
            return 1.0;
        const double Ti = SIMCODE::species_transfer_function(species, *transfers, k, a);
        return std::pow(Ti / transfers->density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, k, a), 2);
    }
    const auto & get_pofk_cb_every_step() const { return pofk_cb_every_step; }
    const auto & get_pofk_total_every_step() const { return pofk_total_every_step; }
    const auto & get_pofk_species_every_step() const { return pofk_species_every_step; }
    /// The linear density field delta_cb(zini,k) of the initial conditions (only if hybridbias)
    const FFTWGrid<NDIM> & get_hybridbias_linear_field() const { return hybridbias_linear_field_fourier; }
    auto & get_hybridbias_mg_tables() { return hybridbias_mg_tables; }

    /// The time-stepping (where the steps are and their sizes)
    SIMCODE::TimeStepping<NDIM> time_stepping() const { return {params.timestep, cosmo, grav}; }


  protected:
    // FFTW planner flag and wisdom (plans): read at the start, written to the output folder when new plans are made
    void setup_fftw();
    void save_fftw_wisdom_if_new_plans();
    std::string fftw_wisdom_output_filename() const;
    size_t fftw_plans_in_saved_wisdom{0};

    // The density and force grids of the PM step, kept from step to step: allocating new ones every step makes
    // the operating system zero and map the memory of all of them again (page faults on ~4 full grids per step).
    // Released before the analysis at the outputs, which needs the memory for other grids
    FFTWGrid<NDIM> pm_density_grid;
    std::array<FFTWGrid<NDIM>, NDIM> pm_force_grids;
    FFTWGrid<NDIM> & pm_density_workspace(int nmesh, int nleft, int nright);
    void release_pm_workspace();

    // The parts of init()
    void output_background_and_growth_functions();
    void setup_initial_power_spectrum();
    void generate_initial_conditions();
    FFTWGrid<NDIM> make_initial_density_field();
    void generate_neutrino_particles(const FFTWGrid<NDIM> & delta_ini_fourier);
    void generate_initial_conditions_one_species(const FFTWGrid<NDIM> & delta_cb_fourier);
    void generate_initial_conditions_two_species(const FFTWGrid<NDIM> & delta_cb_fourier);
    std::vector<double> ic_velocity_norms(double fac) const;
    bool cola_frame_growth_differs() const;
    std::function<double(double)> ic_velocity_norm_1LPT_of_kBox(double fac);
    std::function<double(double)> ic_velocity_norm_2LPT_of_kBox(double fac) const;
    void store_cola_potentials(const FFTWGrid<NDIM> & delta_cb_fourier, FFTWGrid<NDIM> * phi_2LPT_ini_fourier);
    void apply_lattice_offset();
    void read_ic();
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
    void assign_cb_particles_to_grid(FFTWGrid<NDIM> & density_grid_fourier, double a, bool bin_pofk = true);
    void store_initial_density_field(const FFTWGrid<NDIM> & density_grid_fourier, double a);
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
    if (SIMCODE::get_parameter<bool>(param, "force_use_finite_difference_force")) {
        const int stencil_order = SIMCODE::get_parameter<int>(param, "force_finite_difference_stencil_order");
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

    setup_fftw();

    output_background_and_growth_functions();

    // The particle species (CDM+baryons as one, or CDM and baryons, and massive neutrinos)
    if (has_neutrino_particles() and not(cosmo->get_OmegaMNu() > 0.0 and cosmo->get_Mnu_eV() > 0.0))
        throw std::runtime_error("particle_neutrinos = true needs massive neutrinos (cosmology_OmegaMNu > 0 or "
                                 "cosmology_neutrino_masses_eV)");
    species = SIMCODE::make_species_list(params.particle, *cosmo, NDIM);
    if (FML::ThisTask == 0 and species.size() > 1) {
        std::cout << "# Particle species (name, Npart_1D, GADGET type, Omega_i / OmegaM, weight in delta_cb):\n";
        for (auto & info : species)
            std::cout << "#   " << info.name << " " << info.Npart_1D << " " << info.gadget_type << " "
                      << info.mass_fraction << " " << info.weight_cb << "\n";
    }
    if (has_neutrino_particles()) {
        const auto & masses = cosmo->get_neutrino_masses_eV();
        const double Tnu0_eV = cosmo->get_neutrino_temperature_eV(1.0);
        part_neutrinos.init(masses, Tnu0_eV, params.simulation.boxsize, params.particle.neutrino_deltaf);
        if (FML::ThisTask == 0) {
            std::cout << "# Massive neutrinos as particles (delta-f weights: " << std::boolalpha
                      << params.particle.neutrino_deltaf << "). m_nu =";
            for (double m : masses)
                std::cout << " " << m;
            std::cout << " eV (one set of particles for all species), T_nu0 = " << Tnu0_eV
                      << " eV, f_nu = " << cosmo->get_fMNu() << "\n";
        }
    }

    // If we have a MG model and want exactly the same IC as for LCDM
    // we can supply LCDM P(k) and use the GR growth factors to scale it back
    if (params.ic.use_gravity_model_GR) {
        grav_ic = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        grav_ic->set_transfers(grav->get_transfers()); // (loaded once in Main.cpp)
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
    lightcone->set_parameters(params);
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

    auto table = SIMCODE::read_initial_power_spectrum<NDIM>(ic,
                                                            cosmo,
                                                            grav,
                                                            grav_ic,
                                                            transfers);
    auto make_splines = [&](std::string label) {
        power_initial_spline.create(table.k, table.power, "P(k,zini)" + label);
        power_primordial_spline.create(table.k, table.power_primordial, "P_primordial(k)" + label);
    };
    make_splines("");

    // The splines are constant beyond the table, so modes of the IC grid with k > k_max of the input would silently get
    // P(k_max). Only a warning: the power there is often negligible (and a table only slightly too short is common)
    const double kmax_grid = std::sqrt(double(NDIM)) * M_PI * ic.nmesh / boxsize;
    if (FML::ThisTask == 0 and ic.random_field_type != SIMCODE::ICRandomFieldType::ReadParticles and
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
        return SIMCODE::sigma_of_R<NDIM>(power_initial_spline, grav_ic, boxsize, ic.initial_redshift, redshift, R_mpch);
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

        // Update the transfer functions: the one object (loaded in Main.cpp) of the gravity models (grav and grav_ic),
        // the initial conditions and the analysis
        if (transfers)
            transfers->rescale_primordial_power(pofk_norm_factor);
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
    // If we simply read IC from file (useful for testing)
    if (params.ic.random_field_type == SIMCODE::ICRandomFieldType::ReadParticles) {
        read_ic();
        return;
    }

    const FFTWGrid<NDIM> delta_ini_fourier = make_initial_density_field();

    // Massive neutrino particles (made from the same random field)
    if (has_neutrino_particles())
        generate_neutrino_particles(delta_ini_fourier);

    timer.StartTiming("InitialConditions");
    if (has_baryons())
        generate_initial_conditions_two_species(delta_ini_fourier);
    else
        generate_initial_conditions_one_species(delta_ini_fourier);
    apply_lattice_offset();
    timer.EndTiming("InitialConditions");
}

/// The initial CDM+baryon density field delta(k, zini) (and the linear field for the hybrid bias spectra)
template <int NDIM, class T>
FML::GRID::FFTWGrid<NDIM> NBodySimulation<NDIM, T>::make_initial_density_field() {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;

    auto nextra = FML::INTERPOLATION::get_extra_slices("CIC");
    if (ic.random_field_type == SIMCODE::ICRandomFieldType::ReadPhases) {
        // If read_phases then delta_ini is computed from particles
        nextra = FML::INTERPOLATION::get_extra_slices(ic.reconstruct_assigment_method);
    }
    FFTWGrid<NDIM> delta_ini_fourier(ic.nmesh, nextra.first, nextra.second);
    delta_ini_fourier.add_memory_label("delta_ini_fourier(k)");
    delta_ini_fourier.set_grid_status_real(false);

    timer.StartTiming("InitialDensityField");
    auto rng = SIMCODE::make_random_generator(ic);
    SIMCODE::generate_initial_density_field<NDIM>(delta_ini_fourier,
                                                  ic,
                                                  boxsize,
                                                  power_initial_spline,
                                                  power_primordial_spline,
                                                  cosmo,
                                                  grav_ic,
                                                  rng.get());
    timer.EndTiming("InitialDensityField");

    // Take the phases from the particles in a file, but use our P(k) for the amplitudes
    if (ic.random_field_type == SIMCODE::ICRandomFieldType::ReadPhases) {
        SIMCODE::delta_from_gadget_particles<NDIM, T>(delta_ini_fourier, params, *cosmo);
        SIMCODE::whiten_and_set_power_spectrum<NDIM>(delta_ini_fourier, power_initial_spline, boxsize);
    }

    // Reverse the phases? For doing pair-fixed simulations
    if (ic.reverse_phases) {
        SIMCODE::reverse_phases<NDIM>(delta_ini_fourier);
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
    return delta_ini_fourier;
}

/// The massive neutrino particles from the initial density field (and the transfer functions)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_neutrino_particles(const FFTWGrid<NDIM> & delta_ini_fourier) {
    const auto & ic = params.ic;
    timer.StartTiming("InitialConditionsNeutrinos");
    FML::assert_mpi(transfers != nullptr, "particle_neutrinos needs the transfer functions (camb or class)");
    const auto & nu = species.back();
    part_neutrinos.generate(delta_ini_fourier,
                            *transfers,
                            ic.initial_redshift,
                            cosmo->HoverH0_of_a(ic.a_initial()),
                            nu.Npart_1D,
                            params.particle.allocation_factor,
                            ic.random_seed,
                            nu.id_offset);
    timer.EndTiming("InitialConditionsNeutrinos");
}

//=============================================================================
// The IC velocities. The velocity of LPT order n is fac f_n Psi_n with fac = a^2 H / H0 at zini and f_n the growth
// rate of the IC (grav_ic). With COLA the velocities in the particles are relative to the COLA frame, which has the
// growth rates of the gravity model (grav): at k = 0 for scale-independent COLA, f(k) for scale-dependent COLA, and
// only the LPT orders up to simulation_cola_LPT_order. With ic_use_gravity_model_GR and a modified gravity model the
// IC have the growth rate of GR (grav_ic) but the COLA frame that of the gravity model (grav).
//=============================================================================

/// fac {f_1LPT, f_2LPT, f_3LPTa, f_3LPTb} of the IC at k = 0
template <int NDIM, class T>
std::vector<double> NBodySimulation<NDIM, T>::ic_velocity_norms(double fac) const {
    const double aini = params.ic.a_initial();
    return {fac * grav_ic->get_f_1LPT(aini),
            fac * grav_ic->get_f_2LPT(aini),
            fac * grav_ic->get_f_3LPTa(aini),
            fac * grav_ic->get_f_3LPTb(aini)};
}

/// The IC have the growth of GR (ic_use_gravity_model_GR) and the COLA frame that of the modified gravity model
template <int NDIM, class T>
bool NBodySimulation<NDIM, T>::cola_frame_growth_differs() const {
    return cola.enabled and grav_ic != grav and grav_ic->get_name() != grav->get_name();
}

/// The scale-dependent 2LPT velocity norm (nullptr if it is that of k = 0): f_2LPT(k, zini) Psi_2LPT with k the
/// wavenumber of the 2LPT field (as in the COLA frame of scale-dependent COLA, which therefore needs nothing extra).
/// Using f_2LPT(k = 0) for all k gave velocities that differed from those of scale-dependent COLA by 4.3e-4 rms for
/// 0.4 eV neutrinos. With scale-independent COLA the frame has f_2LPT(0) Psi_2LPT and the particles get the residual.
/// In the N-body gauge only the 1LPT growth rate differs from that of the gravity model (on horizon scales, from the
/// transfer functions); the 2LPT velocity is that of the gravity model in both cases (as in the COLA frame). With
/// simulation_cola_LPT_order = 1 the frame has no 2LPT velocity and the particles get all of it. With
/// ic_use_gravity_model_GR and modified gravity the residual relative to the frame of the gravity model. The same for
/// all species (the 2LPT displacement is that of the total CDM+baryon field)
template <int NDIM, class T>
std::function<double(double)> NBodySimulation<NDIM, T>::ic_velocity_norm_2LPT_of_kBox(double fac) const {
    const auto & ic = params.ic;
    const double boxsize = params.simulation.boxsize;
    const double aini = ic.a_initial();
    const bool cola_frame_has_2LPT = cola.enabled and cola.LPT_order >= 2;
    if (not(ic.LPT_order >= 2 and
            ((grav_ic->is_growth_scaledependent() and (not cola_frame_has_2LPT or not cola.scaledependent)) or
             (cola_frame_has_2LPT and cola_frame_growth_differs()))))
        return nullptr;
    if (FML::ThisTask == 0 and grav_ic->is_growth_scaledependent())
        std::cout << "# Scale-dependent 2LPT growth rate at zini f_2(k) = "
                  << grav_ic->get_f_2LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav_ic->get_f_2LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                  << grav_ic->get_f_2LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                  << grav_ic->get_f_2LPT(aini) << "\n";
    const double f2_frame0 = cola_frame_has_2LPT ? grav->get_f_2LPT(aini) : 0.0;
    const bool frame_scaledependent = cola_frame_has_2LPT and cola.scaledependent;
    auto g = grav;
    auto g_ic = grav_ic;
    return [g, g_ic, fac, aini, boxsize, f2_frame0, frame_scaledependent](double kBox) {
        const double f2_frame = frame_scaledependent ? g->get_f_2LPT(aini, kBox / boxsize / g->H0_hmpc) : f2_frame0;
        return fac * (g_ic->get_f_2LPT(aini, kBox / boxsize / g_ic->H0_hmpc) - f2_frame);
    };
}

/// The scale-dependent 1LPT velocity norm of one species (nullptr if it is that of k = 0). Sets
/// ic_velocities_relative_to_cola_frame when the velocities it gives are relative to the COLA frame.
/// N-body gauge: the growth rate (and so the 1LPT velocity) is scale-dependent on large scales; we take it from the
/// (N-body gauge) transfer functions.
/// Scale-dependent growth (massive neutrinos with the transfer functions, modified gravity): the 1LPT velocity is
/// f(k, zini) Psi (the growth rate at k = 0 for all k overestimated the velocity: for 0.4 eV neutrinos f = 0.998
/// instead of f_cb = 0.976 for k >> k_fs, which gave +1.7% in P_cb(z = 0) on linear scales). With scale-dependent COLA
/// the COLA frame already has this velocity; with scale-independent COLA it has f(k = 0) Psi and the particles get the
/// residual (f(k) - f(0)) Psi. With ic_use_gravity_model_GR the particles get the residual (f_IC(k) - f_frame(k)) Psi
/// for the orders the frame has. Zeroing the velocities (or taking the frame growth rate to be that of the IC) started
/// COLA with the velocity of the gravity model and the run without COLA with that of GR
template <int NDIM, class T>
std::function<double(double)> NBodySimulation<NDIM, T>::ic_velocity_norm_1LPT_of_kBox(double fac) {
    const double boxsize = params.simulation.boxsize;
    const double aini = params.ic.a_initial();
    auto g = grav;
    auto g_ic = grav_ic;
    const bool cola_enabled = cola.enabled;
    const bool cola_scaledependent = cola.scaledependent;

    if (params.force.nbody_gauge) {
        FML::assert_mpi(transfers != nullptr, "force_nbody_gauge needs the transfer functions");
        const double kmin = transfers->range().kmin_hmpc;
        const double kmax = transfers->range().kmax_hmpc;
        const auto & tr = *transfers;
        const Spline growth_rate_of_logk = SIMCODE::spline_of_logk(
            kmin,
            kmax,
            [&](double logk) {
                return tr.density_growth_rate(SIMCODE::TransferSpecies::CDMBaryon, std::exp(logk), aini);
            },
            "f_cb(k,zini)");
        if (FML::ThisTask == 0)
            std::cout << "# N-body gauge: growth rate at zini f(k) = " << growth_rate_of_logk(std::log(kmin))
                      << " (k = " << kmin << " h/Mpc), " << growth_rate_of_logk(std::log(0.1)) << " (k = 0.1 h/Mpc). "
                      << "Scale-independent value " << grav_ic->get_f_1LPT(aini) << "\n";
        // With COLA the velocities are relative to the COLA frame, which has the growth rate of the gravity model
        // (at k = 0 for scale-independent COLA) for 1LPT and all of the 2LPT velocity. Zeroing the velocities (as for
        // scale-independent growth) would replace f_nbg(k) by f(0) (0.75% too large velocity on the largest scales)
        if (cola_enabled)
            ic_velocities_relative_to_cola_frame = true;
        return
            [g, growth_rate_of_logk, fac, aini, boxsize, kmin, kmax, cola_enabled, cola_scaledependent](double kBox) {
                const double logk = SIMCODE::clamped_logk(kBox / boxsize, kmin, kmax);
                const double f_frame = not cola_enabled    ? 0.0 :
                                       cola_scaledependent ? g->get_f_1LPT(aini, kBox / boxsize / g->H0_hmpc) :
                                                             g->get_f_1LPT(aini);
                return fac * (growth_rate_of_logk(logk) - f_frame);
            };
    }

    if (not((grav_ic->is_growth_scaledependent() and not(cola_enabled and cola_scaledependent)) or
            cola_frame_growth_differs()))
        return nullptr;
    if (cola_enabled)
        ic_velocities_relative_to_cola_frame = true;
    if (FML::ThisTask == 0 and cola_frame_growth_differs())
        std::cout << "# The IC have the growth rate of GR at zini f = " << grav_ic->get_f_1LPT(aini)
                  << ", the COLA frame that of " << grav->get_name()
                  << " f(k) = " << grav->get_f_1LPT(aini, 0.01 / grav->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav->get_f_1LPT(aini, 1.0 / grav->H0_hmpc) << " (k = 1 h/Mpc): the particles get the "
                  << "difference\n";
    if (FML::ThisTask == 0 and grav_ic->is_growth_scaledependent())
        std::cout << "# Scale-dependent growth rate at zini f(k) = "
                  << grav_ic->get_f_1LPT(aini, 0.01 / grav_ic->H0_hmpc) << " (k = 0.01 h/Mpc), "
                  << grav_ic->get_f_1LPT(aini, 0.1 / grav_ic->H0_hmpc) << " (k = 0.1 h/Mpc), "
                  << grav_ic->get_f_1LPT(aini, 1.0 / grav_ic->H0_hmpc) << " (k = 1 h/Mpc). Scale-independent value "
                  << grav_ic->get_f_1LPT(aini) << "\n";
    const double f0 = cola_enabled ? grav->get_f_1LPT(aini) : 0.0;
    const bool frame_scaledependent = cola_enabled and cola_scaledependent;
    return [g, g_ic, fac, aini, boxsize, f0, frame_scaledependent](double kBox) {
        const double f_frame = frame_scaledependent ? g->get_f_1LPT(aini, kBox / boxsize / g->H0_hmpc) : f0;
        return fac * (g_ic->get_f_1LPT(aini, kBox / boxsize / g_ic->H0_hmpc) - f_frame);
    };
}

/// Scale-dependent COLA: the LPT potentials phi_1LPT and phi_2LPT (k, zini) of the CDM+baryon field (the COLA
/// frame). phi_2LPT_ini_fourier: the 2LPT potential if the IC generator returned it (else it is computed here)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::store_cola_potentials(const FFTWGrid<NDIM> & delta_cb_fourier,
                                                     FFTWGrid<NDIM> * phi_2LPT_ini_fourier) {
    if (not(cola.enabled and cola.scaledependent))
        return;
    if (FML::PARTICLE::has_get_D_1LPT<T>() and cola.LPT_order >= 1) {
        cola.phi_1LPT_ini_fourier = FFTWGrid<NDIM>();
        FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<NDIM>(delta_cb_fourier, cola.phi_1LPT_ini_fourier);
        cola.phi_1LPT_ini_fourier.add_memory_label("phi_1LPT(k,zini)");
    }
    if (FML::PARTICLE::has_get_D_2LPT<T>() and cola.LPT_order >= 2 and params.ic.LPT_order >= 2) {
        if (FML::ThisTask == 0)
            std::cout << "Storing phi_2LPT(k,zini) for scale-dependent COLA\n";
        if (phi_2LPT_ini_fourier)
            cola.phi_2LPT_ini_fourier = std::move(*phi_2LPT_ini_fourier);
        else
            FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_cb_fourier, cola.phi_2LPT_ini_fourier);
        cola.phi_2LPT_ini_fourier.add_memory_label("phi_2LPT(k,zini)");
    }
}

/// The initial conditions with one CDM+baryon species: the particles displaced from a regular grid with LPT, the
/// velocities from the growth rates (see ic_velocity_norm_1LPT_of_kBox and ic_velocity_norm_2LPT_of_kBox)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::generate_initial_conditions_one_species(const FFTWGrid<NDIM> & delta_cb_fourier) {
    const auto & ic = params.ic;
    const double aini = ic.a_initial();
    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    std::vector<double> velocity_norms = ic_velocity_norms(fac);

    // The IC generator returns the 2LPT potential if the vector has room for it (for scale-dependent COLA)
    const bool store_2LPT = cola.enabled and cola.scaledependent and FML::PARTICLE::has_get_D_2LPT<T>() and
                            cola.LPT_order >= 2 and ic.LPT_order >= 2;
    std::vector<FFTWGrid<NDIM>> phi_nLPT_potentials;
    if (store_2LPT)
        phi_nLPT_potentials.resize(1);

    const auto velocity_norm_1LPT_of_kBox = ic_velocity_norm_1LPT_of_kBox(fac);

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

    const auto velocity_norm_2LPT_of_kBox = ic_velocity_norm_2LPT_of_kBox(fac);
    // (with COLA the velocities are already relative to the frame, see above)
    FML::assert_mpi(not velocity_norm_2LPT_of_kBox or not cola.enabled or ic_velocities_relative_to_cola_frame,
                    "Internal error: scale-dependent 2LPT velocity with COLA must be relative to the COLA frame");

    FML::NBODY::NBodyInitialConditions<NDIM, T>(part,
                                                params.particle.Npart_1D,
                                                params.particle.allocation_factor,
                                                delta_cb_fourier,
                                                phi_nLPT_potentials,
                                                ic.LPT_order,
                                                params.simulation.boxsize,
                                                ic.initial_redshift,
                                                velocity_norms,
                                                velocity_norm_1LPT_of_kBox,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                velocity_norm_2LPT_of_kBox);

    store_cola_potentials(delta_cb_fourier, store_2LPT ? &phi_nLPT_potentials[0] : nullptr);
}

/// Offset the particle lattice from the nodes of the force grid. With a force grid finer than the particle
/// grid the particles would otherwise start exactly on grid nodes where CIC responds to |displacement|
/// (a rectified response) and this gives large spurious forces.
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
    for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo &) {
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
    FML::assert_mpi(transfers != nullptr,
                    "particle_baryons needs the transfer functions (camb, class or internal)");
    const auto & cdm = species[0];
    const auto & baryon = species[1];
    const auto splines = SIMCODE::two_species_ic_splines<NDIM>(*transfers,
                                                               *grav_ic,
                                                               ic.initial_redshift,
                                                               cdm.weight_cb,
                                                               baryon.weight_cb);

    const double fac = aini * aini * cosmo->HoverH0_of_a(aini);
    const std::vector<double> velocity_norms = ic_velocity_norms(fac);
    // The 2LPT velocity with scale-dependent growth (the same for both species, see ic_velocity_norm_2LPT_of_kBox)
    const auto velocity_norm_2LPT = ic_velocity_norm_2LPT_of_kBox(fac);
    const double kmin = transfers->range().kmin_hmpc;
    const double kmax = transfers->range().kmax_hmpc;
    auto clamp_logk = [&](double kBox) { return SIMCODE::clamped_logk(kBox / boxsize, kmin, kmax); };
    // The baryon-CDM relative mode: in the displacement (each species its own displacement and velocity), the
    // masses m = 1 + eps (mass: shared CDM+baryon flow, eps = the linear relative mode today; mass_ini: shared
    // displacement, eps at zini and each species its own velocity)
    const std::string & relative_mode = params.particle.baryon_relative_mode;
    const bool mass_mode = relative_mode == "mass" or relative_mode == "mass_ini";
    std::pair<Spline, Spline> eps_today;
    if (relative_mode == "mass") {
        eps_today = SIMCODE::two_species_mass_contrast_today(*transfers, ic.initial_redshift);
        if (FML::ThisTask == 0) {
            std::cout << "# Mass-encoded relative mode: eps_b / delta_cb(zini) = ";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << eps_today.second(std::log(k)) << " (k = " << k << ") ";
            std::cout << "\n";
        }
    }
    auto make_species = [&](MPIParticles & particles,
                            const SIMCODE::SpeciesInfo & info,
                            const Spline & ratio,
                            const Spline & growth_rate,
                            const Spline * eps_of_logk,
                            double shift) {
        if (FML::ThisTask == 0)
            std::cout << "# Making the " << info.name << " particles\n";
        // With COLA the species share the COLA frame of the total CDM+baryon field and the velocities we set are
        // relative to it: v_i - v_frame = a^2 H [(f_i - f_frame / R_i) Psi_i] for 1LPT (Psi_i = R_i Psi_cb) and zero
        // for 2LPT (the same for all species; all of the 2LPT velocity with simulation_cola_LPT_order = 1 as the frame
        // then has none, see ic_velocity_norm_2LPT_of_kBox). The frame velocity is f_frame Psi_cb with
        // f_frame = f_cb(k) for scale-dependent COLA, but the growth rate at k = 0 for scale-independent COLA (the
        // frame adds back grav->get_f_1LPT(a), as for one species). Using f_cb(k) there with scale-dependent growth
        // gave the velocities (f_cb(k) - f(0)) Psi_cb too little (-1.75% rms for 0.4 eV neutrinos). The frame has the
        // growth rate of the gravity model (grav), also with ic_use_gravity_model_GR where the IC have that of GR
        // (grav_ic)
        std::vector<double> residual_velocity_norms = velocity_norms;
        for (int i = 0; i < cola.LPT_order; i++)
            residual_velocity_norms[i] = 0.0;
        auto f_frame = [&](double logk) {
            return cola.scaledependent ? grav->get_f_1LPT(aini, std::exp(logk) / grav->H0_hmpc) :
                                         grav->get_f_1LPT(aini);
        };
        std::function<double(double)> velocity_norm_1LPT = [&](double kBox) {
            const double logk = clamp_logk(kBox);
            if (not cola.enabled)
                return fac * growth_rate(logk);
            return fac * (growth_rate(logk) - f_frame(logk) / ratio(logk));
        };
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
        SIMCODE::generate_species_from_delta_cb<NDIM, T>(particles,
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
    store_cola_potentials(delta_cb_fourier, nullptr);
    if (cola.enabled and cola.scaledependent) {
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
                sp.phi_1LPT_ini_fourier = FFTWGrid<NDIM>();
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<NDIM>(delta_shifted, sp.phi_1LPT_ini_fourier);
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

/// Read the N-body gauge source and check it
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_nbody_gauge() {
    const auto & force = params.force;
    if (grav->get_name() != "GR" or cosmo->get_name() != "LCDM")
        throw std::runtime_error("force_nbody_gauge is only implemented for cosmology_model = LCDM and gravity_model "
                                 "= GR (the source is computed with CLASS)");
    FML::assert_mpi(transfers != nullptr, "force_nbody_gauge needs the transfer functions");

    if (FML::ThisTask == 0)
        std::cout << "# Running in the N-body gauge. Reading the linear source [" << force.nbody_gauge_filename
                  << "]\n";
    nbody_gauge = std::make_shared<SIMCODE::NBodyGaugeSource>();
    nbody_gauge->read(force.nbody_gauge_filename);
    nbody_gauge->check_cosmology(*cosmo, params.ic.initial_redshift, params.simulation.boxsize, force.nmesh);

    if (FML::ThisTask == 0) {
        const double aini = params.ic.a_initial();
        for (double a : {aini, 0.5, 1.0}) {
            const auto ratio = nbody_gauge->ratio_spline(a, aini, *transfers);
            std::cout << "# N-body gauge source: delta_extra(k,z) / delta_cb(k,zini) at z = " << 1.0 / a - 1.0 << ":";
            for (double k : {1e-3, 1e-2, 1e-1})
                std::cout << " " << ratio(std::log(k)) << " (k = " << k << ")";
            std::cout << "\n";
        }
    }

    // delta_cb(k,zini) is stored from the density field of the first force evaluation (add_nbody_gauge_source) so that
    // it has the same window as the density fields it is added to
}

/// Read the particles from gadget files. With COLA we also reconstruct the LPT displacement fields
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::read_ic() {
    SIMCODE::read_ic_from_gadget<NDIM, T>(part,
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
// FFTW plans. With simulation_fftw_planner = measure (or slower) FFTW times
// candidate algorithms the first time each grid size is transformed. The plans
// (wisdom) are written to the output folder so that later runs can read them
// with simulation_fftw_wisdom_file and skip the planning
//=============================================================================

template <int NDIM, class T>
std::string NBodySimulation<NDIM, T>::fftw_wisdom_output_filename() const {
    return params.output.path("fftw_wisdom_" + params.simulation.name + ".txt");
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::setup_fftw() {
    const auto & planner = params.simulation.fftw_planner;
    const auto & wisdom_file = params.simulation.fftw_wisdom_file;
    FML::GRID::set_fftw_planner_flag(FML::GRID::fftw_planner_flag_from_name(planner));
    if (wisdom_file != "") {
        if (not FML::GRID::load_fftw_wisdom(wisdom_file))
            throw std::runtime_error("Cannot read FFTW wisdom from simulation_fftw_wisdom_file = [" + wisdom_file +
                                     "] (missing file, or wisdom from another FFTW version or precision)");
    }
    if (FML::ThisTask == 0) {
        std::cout << "# FFTW planner: " << planner << " with " << FML::GRID::get_fftw_nthreads() << " thread(s)";
        if (wisdom_file != "")
            std::cout << ", wisdom read from " << wisdom_file;
        if (planner != "estimate")
            std::cout << ". Plans made are written to " << fftw_wisdom_output_filename();
        std::cout << "\n";
    }
}

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::save_fftw_wisdom_if_new_plans() {
    // Plans made with FFTW_ESTIMATE are not worth keeping (no measuring was done)
    if (params.simulation.fftw_planner == "estimate")
        return;
    // Collective: all tasks must agree on whether to save (they make the same plans, but be safe)
    size_t nplans = FML::GRID::get_number_of_fftw_plans_made();
    FML::MaxOverTasks(&nplans);
    if (nplans == fftw_plans_in_saved_wisdom)
        return;
    const auto filename = fftw_wisdom_output_filename();
    if (FML::GRID::save_fftw_wisdom(filename)) {
        fftw_plans_in_saved_wisdom = nplans;
        if (FML::ThisTask == 0)
            std::cout << "# Wrote FFTW wisdom (" << nplans << " plans made in this run) to " << filename << "\n";
    } else if (FML::ThisTask == 0) {
        std::cout << "# Warning: could not write FFTW wisdom to " << filename << "\n";
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
                    for_each_species(
                        [&](auto & particles, const SIMCODE::SpeciesInfo &) { particles.sort_by_position(nblocks); });
                    timer.EndTiming("SortParticles");
                }

                const auto & s = steps[istep];
                step({s.apos, s.apos_new, s.avel_new, s.delta_time_drift, s.delta_time_kick, s.velocity_scale});

                // Most plans are made by now (initial conditions and the first step): save them already so that
                // they are kept also if the run does not finish
                if (istep_total == 1 and istep == 0)
                    save_fftw_wisdom_if_new_plans();
            }
        }

        analyze_and_output(ioutput, output_redshifts[ioutput]);
        save_fftw_wisdom_if_new_plans();
    }
    timer.EndTiming("Timestepping");

    // Lightcone create kappa maps
    if (lightcone->lightcone_active()) {
        lightcone->create_weak_lensing_maps();
    }
    save_fftw_wisdom_if_new_plans();

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
    early_force = early;
    if (not early)
        initial_density_field_early.free();
    if (early and FML::ThisTask == 0)
        std::cout << "Using the early-time force (" << f.early_method << ") at z = " << 1.0 / a - 1.0 << "\n";
}

/// The density grid of the PM step (the grid of the previous step if it has the same size and extra slices,
/// otherwise a new one). Its content is left over from the previous step: the density assignment overwrites it
template <int NDIM, class T>
auto NBodySimulation<NDIM, T>::pm_density_workspace(int nmesh, int nleft, int nright) -> FFTWGrid<NDIM> & {
    auto & grid = pm_density_grid;
    if (grid.get_nmesh() != nmesh or grid.get_n_extra_slices_left() != nleft or
        grid.get_n_extra_slices_right() != nright) {
        grid = FFTWGrid<NDIM>(nmesh, nleft, nright);
        grid.add_memory_label("NBodySimulation::pm_density_grid");
    }
    grid.set_grid_status_real(true);
    return grid;
}

/// Free the PM density and force grids (they are allocated again at the next step)
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::release_pm_workspace() {
    pm_density_grid.free();
    for (auto & grid : pm_force_grids)
        grid.free();
}

/// One kick-drift step: density -> force -> kick -> COLA kick -> lightcone -> COLA drift -> drift
template <int NDIM, class T>
void NBodySimulation<NDIM, T>::step(const StepInfo & s) {
    const auto & sim = params.simulation;
    set_force_settings_for_step(s.apos);
    const auto & force = current_force;

    // Force interlacing: average the force over copies of the particles shifted relative to the grid
    const auto nleftright = FML::INTERPOLATION::get_extra_slices(force.density_assignment_method);
    if (force.interlacing_order > 1 and s.delta_time_kick != 0.0 and not sim.enforce_LPT_trajectories) {
        kick_with_interlaced_force(s);
        return drift_and_lightcone(s);
    }

    // Compute total density field
    FFTWGrid<NDIM> & density_grid_fourier = pm_density_workspace(force.nmesh, nleftright.first, nleftright.second);
    if (s.delta_time_kick != 0.0) {
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");
        // The potential for the lightcone (from the same density field as the force)
        if (lightcone->needs_potential(s.apos, s.apos_new)) {
            timer.StartTiming("LightconePotential");
            lightcone->compute_potential(density_grid_fourier,
                                         s.apos,
                                         current_force.density_assignment_method,
                                         current_force.deconvolve_window);
            timer.EndTiming("LightconePotential");
        }
    }

    // Compute forces and kick particles (updates velocity)
    std::array<FFTWGrid<NDIM>, NDIM> & force_real = pm_force_grids;
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
            for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
                SIMCODE::kick_particles<NDIM>(particles,
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
            part_neutrinos.update_weights();
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
        for_each_cb_species([&](MPIParticles & particles, const SIMCODE::SpeciesInfo &) {
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
        // The potential of this step (if not computed with the force, e.g. with force interlacing: from the
        // CDM+baryon density only)
        if (lightcone->needs_potential(s.apos, s.apos_new) and not lightcone->has_potential(s.apos)) {
            const auto nleftright = FML::INTERPOLATION::get_extra_slices(current_force.density_assignment_method);
            FFTWGrid<NDIM> density(current_force.nmesh, nleftright.first, nleftright.second);
            assign_cb_particles_to_grid(density, s.apos);
            timer.StartTiming("LightconePotential");
            lightcone->compute_potential(
                density, s.apos, current_force.density_assignment_method, current_force.deconvolve_window);
            timer.EndTiming("LightconePotential");
        }
        lightcone->create_lightcone(
            part, s.apos, s.apos_new, s.avel_new, s.delta_time_drift, current_force.density_assignment_method);
        timer.EndTiming("Lightcone");
    }
    if (cola.enabled and make_lightcone) {
        timer.StartTiming("COLA");
        for_each_cb_species(
            [&](MPIParticles & particles, const SIMCODE::SpeciesInfo &) { cola.drift(particles, s.apos, s.apos_new); });
        if (not cola.scaledependent and sim.enforce_LPT_trajectories)
            part.communicate_particles();
        timer.EndTiming("COLA");
    }

    // Drift particles (updates positions)
    if (s.delta_time_drift != 0.0 and not sim.enforce_LPT_trajectories) {
        timer.StartTiming("Drift");
        for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
            SIMCODE::drift_particles<NDIM>(particles, info, s.apos, s.apos_new, s.delta_time_drift, sim.boxsize);
        });
        timer.EndTiming("Drift");
    }

    // Show info about particles and memory use
    for_each_species([&](auto & particles, const SIMCODE::SpeciesInfo &) { particles.info(); });
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
    const auto nleftright = FML::INTERPOLATION::get_extra_slices(force.density_assignment_method);
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

        FFTWGrid<NDIM> & density_grid_fourier =
            pm_density_workspace(force.nmesh, nleftright.first, nleftright.second + 1);
        timer.StartTiming("ComputeDensityField");
        compute_density_field_fourier(density_grid_fourier, s.apos);
        timer.EndTiming("ComputeDensityField");

        timer.StartTiming("ComputeForce");
        std::array<FFTWGrid<NDIM>, NDIM> & force_real = pm_force_grids;
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

    // The analysis allocates its own grids
    release_pm_workspace();

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
                 SIMCODE::output_baryonified_snapshot(
                     SIMCODE::copy_baryonification_particles(part.get_particles_ptr(), part.get_npart()),
                     params,
                     cosmo,
                     redshift,
                     snapshot_folder);
         }},
        {"Power-spectrum",
         params.pofk.enabled,
         [&]() { compute_power_spectrum(*this, redshift, snapshot_folder); }},
        {"Power-spectrum multipoles",
         params.pofk_multipole.enabled,
         [&]() { compute_power_spectrum_multipoles(*this, redshift, snapshot_folder); }},
        {"FOF", params.fof.enabled, [&]() { compute_fof_halos(*this, redshift, snapshot_folder); }},
        {"Bispectrum", params.bispectrum.enabled, [&]() { compute_bispectrum(*this, redshift, snapshot_folder); }},
        {"Hybrid bias spectra",
         params.hybridbias.enabled,
         [&]() { compute_hybrid_bias_spectra(*this, redshift, snapshot_folder); }},
        {"Output particles",
         params.output.particles,
         [&]() {
             if (params.output.fileformat == SIMCODE::ParticleFileFormat::GADGET)
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

// The density field the force is computed from (member functions of NBodySimulation)
#include "SimulationDensity.h"

#endif
