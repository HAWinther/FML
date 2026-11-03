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
#include "ICFromParameters.h"
#include "InitialDensityField.h"
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
/// read_parameters(param, params), init() (creates the initial conditions) and run().
/// The parameters are read once in Main.cpp (SimulationParameters::read and
/// check_parameters_against_models in ModelCompatibility.h)
///
/// The pieces live in separate headers:
///   SimulationParameters.h : all the parameters (read, printed and validated)
///   ModelCompatibility.h   : the checks of the parameters against the cosmology and gravity model
///   TimeStepping.h         : where the steps are and the size of the kick/drift steps
///   ICFromParameters.h     : P(k,zini) / T(k,zini) and the initial density field from the ic_* parameters (the
///                            particles of a species from it: FML::NBODY::generate_species_from_delta_cb)
///   ExternalIC.h           : starting from particles read from file
///   COLA.h                 : the COLA method (ColaState)
///   Species.h              : the particle species (CDM+baryons, or CDM and baryons, and neutrinos)
///   NeutrinoParticles.h    : massive neutrinos as particles (delta-f method)
///   KickDrift.h            : the kick and drift of a species (non-relativistic or relativistic)
///   InitialDensityField.h  : delta_cb(k,zini) for the linear sources added to the density field
///   SimulationInit.h       : reading the parameters and the initial conditions (part of this class)
///   SimulationStep.h       : the time-stepping and the analysis at the outputs (part of this class)
///   SimulationDensity.h    : the density field the force is computed from (part of this class)
///   AnalyzeOutput.h        : on the fly analysis and output
///   SnapshotOutput.h       : writing particles as GADGET / HDF5 snapshots
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

    /// The particle species (see Species.h): CDMBaryon, or CDM and Baryon, and MassiveNeutrino
    std::vector<SIMCODE::SpeciesInfo> species;
    /// All the particles (CDM+baryons as one species "cb", or the CDM particles if particle_baryons)
    MPIParticles part;
    /// The baryon particles (only if particle_baryons)
    MPIParticles part_baryon;
    /// The massive neutrino particles (only if particle_neutrinos; delta-f method)
    SIMCODE::NeutrinoParticles<NDIM> part_neutrinos;

    /// The initial density field delta_cb(zini,k) on the force grid (for linear massive neutrinos and the N-body
    /// gauge; see InitialDensityField.h)
    SIMCODE::InitialDensityField<NDIM> initial_density_field;
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
    // COLA: the residual 1LPT velocity (f(k) - f(0)) Psi; IC orders the frame does not have; velocities read from
    // file), so they are not reset to zero. Set by generate_initial_conditions_one_species and read_ic
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
    std::map<SIMCODE::ParticleSpecies, std::vector<std::pair<double, PowerSpectrumBinning>>> pofk_species_every_step;

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

    /// Set the parameters: params read (and checked) from param in Main.cpp. Sets up what follows from them
    void read_parameters(const ParameterMap & param, const Parameters & params);

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
    /// Do the CDM+baryon particles carry masses (the mass-encoded relative mode, particle_baryon_relative_mode = mass
    /// or mass_ini)? Otherwise their masses (if the particle type has them) are 1
    bool particles_have_masses() const {
        return params.particle.baryons and params.particle.baryon_relative_mode != "displacement";
    }
    /// Massive neutrinos as particles (delta-f method)?
    bool has_neutrino_particles() const { return params.particle.neutrinos; }
    SIMCODE::NeutrinoParticles<NDIM> & get_neutrinos() { return part_neutrinos; }
    typename SIMCODE::NeutrinoParticles<NDIM>::Container & get_neutrino_particles() {
        return part_neutrinos.particles();
    }

    /// The particle species (in the order of the IDs)
    const std::vector<SIMCODE::SpeciesInfo> & get_species() const { return species; }
    /// The CDM+baryon particles of a species (CDMBaryon and CDM: part, Baryon: part_baryon; all of type T)
    MPIParticles & cb_particles_of(const SIMCODE::SpeciesInfo & info) {
        FML::assert_mpi(info.in_cb(), "cb_particles_of: not a CDM+baryon species");
        return info.is(SIMCODE::ParticleSpecies::Baryon) ? part_baryon : part;
    }
    /// Call f(particles, info) for the CDM+baryon species (CDMBaryon, or CDM and Baryon; all of type T)
    template <class F>
    void for_each_cb_species(F && f) {
        for (auto & info : species)
            if (info.in_cb())
                f(cb_particles_of(info), info);
    }
    /// Call f(particles, info) for all the species (the particle types differ, so use a generic lambda)
    template <class F>
    void for_each_species(F && f) {
        for_each_cb_species(f);
        for (auto & info : species)
            if (info.is(SIMCODE::ParticleSpecies::MassiveNeutrino))
                f(part_neutrinos.particles(), info);
    }
    /// The shot noise of the density field of a species (in units of the box volume): 1/N (sum w^2 / N^2 for the
    /// delta-f neutrinos). Collective
    double species_shotnoise(const SIMCODE::SpeciesInfo & info) {
        if (info.is(SIMCODE::ParticleSpecies::MassiveNeutrino))
            return part_neutrinos.shotnoise();
        return 1.0 / double(cb_particles_of(info).get_npart_total());
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
        return cosmo->get_neutrino_mean_energy_factor(a);
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
    double species_linear_ratio(SIMCODE::TransferSpecies species, double k, double a) const {
        if (not transfers)
            return 1.0;
        const double Ti = transfers->density_over_k_mpc2(species, k, a);
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

    // The parts of run() (a step: where the positions and velocities are, and the kick/drift sizes)
    using Step = FML::NBODY::TimeStepping::Step;
    void step(const Step & s);
    void drift_and_lightcone(const Step & s);
    void kick_with_interlaced_force(const Step & s);
    // The density field (SimulationDensity.h)
    void compute_density_field_fourier(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void assign_cb_particles_to_grid(FFTWGrid<NDIM> & density_grid_fourier,
                                     double a,
                                     const SIMCODE::ForceSettings & force,
                                     bool bin_pofk = true);
    void store_initial_density_field(const FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_nbody_gauge_source(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_neutrino_particles(FFTWGrid<NDIM> & density_grid_fourier, double a);
    void add_linear_neutrinos(FFTWGrid<NDIM> & density_grid_fourier, double a);
    const FFTWGrid<NDIM> & initial_density_in_particle_frame() {
        return initial_density_field.in_particle_frame(early_force, density_shift);
    }
    bool particles_shifted() const { return SIMCODE::InitialDensityField<NDIM>::is_shifted(density_shift); }
    void bin_up_power_spectrum_every_step(const FFTWGrid<NDIM> & grid,
                                          bool deconvolve,
                                          double shotnoise,
                                          double a,
                                          std::vector<std::pair<double, PowerSpectrumBinning>> & list);
    void analyze_and_output(int ioutput, double redshift);
};

// The member functions of NBodySimulation: the setup and initial conditions, the time-stepping and output,
// and the density field the force is computed from
#include "SimulationInit.h"
#include "SimulationStep.h"
#include "SimulationDensity.h"

#endif
