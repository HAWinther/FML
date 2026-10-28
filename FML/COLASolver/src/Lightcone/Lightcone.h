#ifndef LIGHTCONE_HEADER
#define LIGHTCONE_HEADER

#include "../ParameterDefaults.h"

#include "Lightcone_Healpix.h"
#include "Lightcone_Replicas.h"
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/MPIParticles/MPIParticles.h>

#include <cmath>
#include <iostream>
#include <map>

// This version of the lightcone was originally written by Albert Izard.
// I added the replicas with help from the L-PICOLA implementation (written by Howlett Cullan).

// The particle-type we use to store the light-cone particles
// This is a minimal particle and we preferably will want to use float
// to reduce the memory consumption. We have added a mass to allow for halos
template <class T, int NDIM>
class LightConeParticle {
  private:
    T pos[NDIM];
    T vel[NDIM];
    T mass;
    T phi{}; // Phi/c^2 at the particle (plc_relativistic_effects)

  public:
    T * get_pos() { return pos; }
    T * get_vel() { return vel; }
    T get_mass() { return mass; }
    void set_mass(double _mass) { mass = _mass; }
    T get_potential() const { return phi; }
    void set_potential(double _phi) { phi = T(_phi); }

    constexpr int get_ndim() const { return NDIM; }

    // Assign this particle from another particle (of possibly different type)
    template <class U>
    void assign_from_particle(U & p) {
        auto * _pos = FML::PARTICLE::GetPos(p);
        auto * _vel = FML::PARTICLE::GetVel(p);
        if constexpr (FML::PARTICLE::has_get_mass<U>()) {
            mass = FML::PARTICLE::GetMass(p);
        }
        for (int idim = 0; idim < NDIM; idim++) {
            pos[idim] = _pos[idim];
            vel[idim] = _vel[idim];
        }
    }
};

// The Lightcone class. This deals with anything related to making lightcones
// and associated onion shell density-maps
template <int NDIM, class T, class U = LightConeParticle<float, NDIM>>
class Lightcone {
  protected:
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Point = std::array<double, NDIM>;
    using MapRealType = double;
    template <class T1>
    using MPIParticles = FML::PARTICLE::MPIParticles<T1>;

    // The cosmology needed to compute comoving distance r(a)
    std::shared_ptr<Cosmology> cosmo;

    // Lightcone parameters
    bool lightcone_on{false};

    // Store the potential at the particles and compute the observed redshift (Doppler + gravitational redshift)
    bool relativistic_effects{false};

    // Initial and final time for the lightcone
    double plc_a_init;
    double plc_a_finish;

    // Onion-maps for particle types
    bool make_onion_density_maps;
    std::map<std::string, OnionSlices<MapRealType, NDIM>> onionslices_of_type;

    // Replicas
    BoxReplicas<NDIM> replicas;

    // For setting filenames and doing outputs
    std::string simulation_name;
    std::string output_folder;
    std::string plc_folder;
    bool output_gadgetfile{false};
    bool output_asciifile{false};
    bool output_hdf5file{false};
    bool output_in_batches{false};
    // Output in batches in the current step (only if the particles may not fit in one batch; the same on all tasks)
    bool batches_this_step{false};
    bool store_particles{false};

    // Boxsize of the simulation
    double boxsize;
    // The origin of the light-cone
    Point pos_observer = {};
    // Sky coverage
    double fsky;
    // Number of calls to the light cone (starts counting at 0)
    int plc_step;

    // Mean density of the tracers in units where the boxsize is unity
    // so basically just n_part
    std::map<std::string, double> mean_density;

    // Ranges for splines for distances
    Spline dist_spline, dist_inv_spline;
    const int npts_loga = 200;
    const double alow = 1e-4;
    const double ahigh = 1.0;
    double dist_low, dist_high;

    // For allocating memory for particles
    // We allow for a 20% buffer. If it overshoots then push_back
    // will realloc which is expensive memory-wise!
    const double particle_allocation_factor{1.2};

  private:
    // Comoving distance
    void compute_comoving_distance();
    double get_distance(double a);
    double get_distance_inv(double dist);

    // Output methods
    void output_particles(FML::Vector<U> & part_lc, double astart, double aend, std::string label, int ibatch = 0);
    void output_ascii(FML::Vector<U> & part_lc, double astart, double aend, std::string label, int ibatch = 0);
    void output_gadget(FML::Vector<U> & part_lc, double astart, double aend, std::string label, int ibatch = 0);
    void output_hdf5(FML::Vector<U> & part_lc, double astart, double aend, std::string label, int ibatch = 0);

    // Useful function - volume of a shell (in any dim)
    double volume_shell(double rlow, double rhigh);

    // The fraction of the lightcone shell rlow < r < rhigh that is in the x-domain of this task
    double local_fraction_of_shell(double rlow, double rhigh);

  public:
    Lightcone(std::shared_ptr<Cosmology> cosmo) : cosmo(cosmo) {};
    Lightcone() = default;
    ~Lightcone() = default;

    void read_parameters(ParameterMap & param);

    void init(std::vector<std::string> & particle_types);

    void create_lightcone(MPIParticles<T> & part,
                          double apos,
                          double apos_new,
                          double avel,
                          double delta_time_drift,
                          const std::vector<FML::GRID::FloatType> * potential_at_particles = nullptr,
                          const FML::GRID::FFTWGrid<NDIM> * potential_grid = nullptr,
                          const std::string & interpolation_method = "CIC");

    bool lightcone_active() const { return lightcone_on; }

    /// Is the direction n (unit vector) at distance r (units of the box) from the observer inside the region
    /// covered by the lightcone, i.e. by the box and its replicas (a point on the boundary is outside)? Without
    /// replicas (plc_use_replicas = false) the box covers -o_i <= r n_i < 1 - o_i (o = plc_pos_observer), with
    /// replicas everything in the first plc_ndim_rep directions and r n_i >= -o_i in the others (for o = 0: the
    /// half-space n_i >= 0, i.e. fsky = 2^(ndim_rep-NDIM)). Used for the pixel centers of the onion maps
    bool is_observed(const std::array<double, NDIM> & n, double r) const {
        const double eps = 1e-10;
        for (int idim = 0; idim < NDIM; idim++) {
            const double x = r * n[idim];
            if (idim >= replicas.get_ndim_rep() and x < -pos_observer[idim] + eps)
                return false;
            if (not replicas.uses_replicas() and x > 1.0 - pos_observer[idim] - eps)
                return false;
        }
        return true;
    }

    /// Does the step apos -> apos_new record lightcone particles with the potential (plc_relativistic_effects)?
    /// The same conditions as in create_lightcone
    bool needs_potential(double apos, double apos_new) const {
        if (not lightcone_on or not relativistic_effects)
            return false;
        if (std::abs(apos_new / apos - 1.0) < 1e-6)
            return false;
        return not(apos_new < plc_a_init + 1e-3 or apos > plc_a_finish - 1e-3);
    }

    /// The radial velocity (km/s), the cosmological redshift and the observed redshift of a lightcone particle (its
    /// position in Mpc/h relative to the observer, velocity in km/s): 1 + z_obs = (1 + z) (1 + v_r/c - Phi/c^2),
    /// the Doppler and gravitational redshift to first order (the observer at rest with Phi_obs = 0: these add a
    /// dipole and a monopole). No integrated terms (ISW, time delay)
    std::array<double, 3> relativistic_quantities(U & p) {
        const auto * pos = FML::PARTICLE::GetPos(p);
        const auto * vel = FML::PARTICLE::GetVel(p);
        double r2 = 0.0, vr = 0.0;
        for (int idim = 0; idim < NDIM; idim++) {
            r2 += double(pos[idim]) * double(pos[idim]);
            vr += double(vel[idim]) * double(pos[idim]);
        }
        const double r = std::sqrt(r2);
        vr = r > 0.0 ? vr / r : 0.0;
        const double zcos = 1.0 / get_distance_inv(r / boxsize) - 1.0;
        const double c_kms = 299792.458;
        const double zobs = (1.0 + zcos) * (1.0 + vr / c_kms - double(p.get_potential())) - 1.0;
        return {vr, zcos, zobs};
    }

    /// Maps of Phi/c^2 sampled at the pixel centers at the mid-radius of the onion shells of the current step
    void make_potential_maps(const FML::GRID::FFTWGrid<NDIM> & potential_grid,
                             const std::string & interpolation_method);

    void create_weak_lensing_maps(std::string particle_type = "cb");

    void free();
};

// Read all the parameters we need for the lightcone
// This also calls the read_parameter function for the maps and replicas
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::read_parameters(ParameterMap & param) {

    lightcone_on = param.get<bool>("lightcone");
    if (lightcone_on) {
        simulation_name = param.get<std::string>("simulation_name");
        output_folder = param.get<std::string>("output_folder");
        boxsize = param.get<double>("simulation_boxsize");
        auto origin = param.get<std::vector<double>>("plc_pos_observer");
        FML::assert_mpi(origin.size() >= NDIM, "Position of observer much have NDIM components");
        for (int idim = 0; idim < NDIM; idim++)
            pos_observer[idim] = origin[idim];
        output_in_batches = param.get<bool>("plc_output_in_batches");
        double plc_z_init = param.get<double>("plc_z_init");
        plc_a_init = 1.0 / (1.0 + plc_z_init);
        double plc_z_finish = param.get<double>("plc_z_finish");
        plc_a_finish = 1.0 / (1.0 + plc_z_finish);
        output_gadgetfile = param.get<bool>("plc_output_gadgetfile");
        output_asciifile = param.get<bool>("plc_output_asciifile");
        output_hdf5file = COLASOLVER::get_parameter<bool>(param, "plc_output_hdf5file");
#ifndef USE_HDF5
        if (output_hdf5file)
            throw std::runtime_error("Invalid parameters: plc_output_hdf5file = true needs HDF5: compile with "
                                     "USE_HDF5=true (make USE_HDF5=true)");
#endif
        make_onion_density_maps = param.get<bool>("plc_make_onion_density_maps");
#ifndef USE_HEALPIX
        // Check this here and not at the first lightcone step
        if (make_onion_density_maps and NDIM == 3)
            throw std::runtime_error(
                "Invalid parameters: plc_make_onion_density_maps = true needs Healpix: compile with "
                "USE_HEALPIX=true (make USE_HEALPIX=true)");
#endif
        relativistic_effects = COLASOLVER::get_parameter<bool>(param, "plc_relativistic_effects");
        if (FML::ThisTask == 0) {
            std::cout << "\n# Read in Lightcone.h:\n";
            std::cout << "# lightcone                                : " << lightcone_on << "\n";
            std::cout << "# plc_pos_observer                         : ";
            for (auto & p : pos_observer)
                std::cout << p << " , ";
            std::cout << "\n";
            std::cout << "# plc_z_init                               : " << plc_z_init << "\n";
            std::cout << "# plc_z_finish                             : " << plc_z_finish << "\n";
            std::cout << "# plc_output_gadgetfile                    : " << output_gadgetfile << "\n";
            std::cout << "# plc_output_asciifile                     : " << output_asciifile << "\n";
            std::cout << "# plc_output_hdf5file                      : " << output_hdf5file << "\n";
            std::cout << "# output_in_batches                        : " << output_in_batches << "\n";
            std::cout << "# plc_make_onion_density_maps                        : " << make_onion_density_maps << "\n";
            std::cout << "# plc_relativistic_effects                 : " << relativistic_effects << "\n";
            if (relativistic_effects and output_gadgetfile)
                std::cout << "# NB: the potential and observed redshift are only written to the HDF5 and ascii files\n";
        }
        replicas.read_parameters(param);

        // Only allocate for cb now
        if (make_onion_density_maps)
            onionslices_of_type["cb"].read_parameters(param);
    }
}

// Initialize the lightcone
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::init(std::vector<std::string> & particle_types) {
    if (not lightcone_on)
        return;

    // Set up maps for all particle types
    auto onionslice_cb = onionslices_of_type["cb"];
    for (auto & v : particle_types) {
        if (v != "cb")
            onionslices_of_type[v] = onionslice_cb;
    }

    // Create output-folder
    plc_folder = output_folder + "/lightcone_" + simulation_name;
    if (not FML::create_folder(plc_folder))
        throw std::runtime_error("Failed to create output folder for lightcone " + plc_folder);

    if (FML::ThisTask == 0) {
        std::cout << "# Initializing lightcone\n";
        std::cout << "# Data will be stored in folder " << plc_folder << "\n";
    }

    // If no reps then fsky = 1/2^NDIM
    // If reps in every dimension then 1.0
    fsky = std::pow(2, replicas.get_ndim_rep() - NDIM);
    if (FML::ThisTask == 0) {
        std::cout << "# Lightcone will cover fsky = " << fsky << "\n";
    }

    // The lightcone has to end today or earlier
    plc_a_finish = std::min(plc_a_finish, 1.0);

    // Set up interpolation to compute distances
    compute_comoving_distance();

    // Init step number
    plc_step = -1;

    // Init the replicas
    replicas.init(plc_a_init, get_distance(plc_a_init), pos_observer);
}

// Creates the lightcone every step
// This method does not modify part in any way
// We only take a copy of the particles when they cross the lightcone
// and store them in part_lc
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::create_lightcone(MPIParticles<T> & part,
                                             double apos,
                                             double apos_new,
                                             double avel,
                                             double delta_time_drift,
                                             const std::vector<FML::GRID::FloatType> * potential_at_particles,
                                             const FML::GRID::FFTWGrid<NDIM> * potential_grid,
                                             const std::string & interpolation_method) {

    // In current version of the code we only have one particle type
    const std::string particle_type = "cb";

    // If the step is close to zero then just skip
    if (std::abs(apos_new / apos - 1.0) < 1e-6)
        return;
    // If lightcone is not on then do not continue
    if (not lightcone_on)
        return;

    // Select the right map for the given particle type
    auto & onionslices_curtype = onionslices_of_type["cb"];
    if (make_onion_density_maps)
        onionslices_curtype = onionslices_of_type[particle_type];

    // Mean density in units where boxsize is unity
    mean_density[particle_type] = part.get_npart_total();

    // Check if we are inside the lightcone range
    // The accuracy on the cut around plc_a_finish is irrelevant in most realistic cases.
    // Therefore we can ignore the buffer zone around apos.
    if (apos_new < plc_a_init + 1e-3 || apos > plc_a_finish - 1e-3) {
        if (FML::ThisTask == 0) {
            std::cout << "# Outside of the lightcone range so won't do lightcone drift\n";
        }
        return;
    }

    // Increase step-number
    plc_step++;

    // The comoving distance to the start and end of the step
    const double r_lc_apos = get_distance(apos);
    const double r_lc_apos_new = get_distance(apos_new);

    // The comoving distance to the lightcone. We have old,new = apos,apos_new
    // except if a_init/a_finish is in the middle of the step
    const double a_old = std::max(apos, plc_a_init);
    const double r_lc_old = get_distance(a_old);
    const double r_lc_old_squared = r_lc_old * r_lc_old;
    const double a_new = std::min(apos_new, plc_a_finish);
    const double r_lc_new = get_distance(a_new);
    const double r_lc_new_squared = r_lc_new * r_lc_new;
    // The same as fractions of the step (the particles move linearly in a over the step). A particle is in the
    // lightcone if it is inside r_lc_old at a_old and outside r_lc_new at a_new
    const double frac_old = a_old > apos ? (a_old - apos) / (apos_new - apos) : 0.0;
    const double frac_new = a_new < apos_new ? (a_new - apos) / (apos_new - apos) : 1.0;

    if (FML::ThisTask == 0) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# .____    .__       .__     __                                    \n";
        std::cout << "# |    |   |__| ____ |  |___/  |_  ____  ____   ____   ____        \n";
        std::cout << "# |    |   |  |/ ___\\|  |  \\   __\\/ ___\\/  _ \\ /    \\_/ __ \\\n";
        std::cout << "# |    |___|  / /_/  >   Y  \\  | \\  \\__(  <_> )   |  \\  ___/   \n";
        std::cout << "# |_______ \\__\\___  /|___|  /__|  \\___  >____/|___|  /\\___  >  \n";
        std::cout << "#         \\/ /_____/      \\/          \\/           \\/     \\/  \n";
        std::cout << "#=====================================================\n";
        std::cout << "# Drifting particles from a = " << apos << " -> " << apos_new << "\n";
        std::cout << "# We will output particles with a = " << a_old << " -> " << a_new << "\n";
        std::cout << "# Size of the slice: " << r_lc_new * boxsize << " -> " << r_lc_old * boxsize << " Mpc/h\n";
    }

    // Flag the replicates that don't need looping over
    replicas.flag(r_lc_old, r_lc_new);

    // If we output in batches, set the maximum number to store
    // We just use the number of particles we have on the task for simplicity
    // Can be changed to a user-provided number in case memory is really tight
    const size_t npart_per_batch = std::max(part.get_npart(), size_t(1));
    int ibatch = 0;
    size_t npart_in_written_batches = 0;

    // Estimate how many particles we should reserve based on the volume and the average number density
    // (the global number times the fraction of the shell in the domain of this task)
    // If no output method is picked then we don't save anything
    store_particles = output_asciifile or output_gadgetfile or output_hdf5file;
    const double num_particles_expected = mean_density[particle_type] * fsky * volume_shell(r_lc_new, r_lc_old);
    size_t num_particles_to_allocate =
        ceil(num_particles_expected * local_fraction_of_shell(r_lc_new, r_lc_old) * particle_allocation_factor);
    // Output in batches in this step only if the particles may not fit in one batch on some task. This is decided
    // for each step (and the same on all tasks as the filenames depend on it)
    batches_this_step = false;
    if (store_particles and output_in_batches) {
        int need_batches = num_particles_to_allocate >= npart_per_batch ? 1 : 0;
        FML::MaxOverTasks(&need_batches);
        batches_this_step = need_batches == 1;
        if (batches_this_step)
            num_particles_to_allocate = std::min(num_particles_to_allocate, npart_per_batch);
    }
    if (FML::ThisTask == 0) {
        if (store_particles) {
            std::cout << "# We will allocate storage for " << num_particles_to_allocate / double(part.get_npart())
                      << " times the number of particles we have (task 0)\n";
            if (batches_this_step)
                std::cout << "# We output in batches of " << npart_per_batch << " (task 0)\n";
            std::cout << "# Memory consumption for particles: "
                      << num_particles_to_allocate * sizeof(U) / double(1024. * 1024.)
                      << " MB on task 0. The particle-size is " << sizeof(U) << " bytes\n";
        } else {
            std::cout << "# We expect roughly ~ " << size_t(num_particles_expected) << " particles in the slice\n";
            std::cout << "# NB: No output picked so we will not store and output any particles\n";
        }
        if (make_onion_density_maps)
            std::cout << "# We will make and output onion slices for the density-contrast\n";
    }

    // Allocate and initialize the maps we need for the current time-step
    if (make_onion_density_maps)
        onionslices_curtype.init_current_step(a_old, a_new, dist_spline, boxsize, plc_folder);

    // Allocate lightcone particles
    FML::Vector<U> part_lc;
    if (store_particles)
        part_lc.reserve(num_particles_to_allocate + 1);

    // Loop over all particles
    double mean_error = 0.0;
    double mean_error_count = 0.0;
    double max_error = 0.0;
    double max_disp = 0.0;
    double vel_norm = 100 * boxsize / avel;
    int nproblem = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : mean_error) reduction(+ : mean_error_count) reduction(max : max_error)          \
    reduction(max : max_disp) reduction(+ : nproblem)
#endif
    for (size_t ipart = 0; ipart < part.get_npart(); ipart++) {

        // Take a copy of the particle reducing it to just position and velocity
        // This is what we are going to store and output so lets save some memory
        U p_bare;
        p_bare.assign_from_particle(part[ipart]);
        if (relativistic_effects and potential_at_particles)
            p_bare.set_potential((*potential_at_particles)[ipart]);

        // Measure the position relative to the observer position
        // and compute the distance per dimension that the particle moves over a full step
        Point delta_pos;
        auto * pos_bare = FML::PARTICLE::GetPos(p_bare);
        auto * vel_bare = FML::PARTICLE::GetVel(p_bare);
        for (int idim = 0; idim < NDIM; idim++) {
            pos_bare[idim] -= pos_observer[idim];
            delta_pos[idim] = vel_bare[idim] * delta_time_drift;
            max_disp = std::max(max_disp, std::abs(delta_pos[idim]));
        }

        // Loop over all replicas
        auto n_replicas_total = replicas.get_n_replicas();
        for (size_t i = 0; i < n_replicas_total; i++) {

            // Get the box-shift
            auto replica_shift = replicas.get_replica_position(i);

            // Get the coord of the replica
            auto coord = replicas.get_coord_of_replica(replica_shift);

            // If replica is in use then process particle
            if (replicas.is_flagged_in_use(coord)) {

                // Take a copy of the particle
                auto p_current = p_bare;
                auto * pos = FML::PARTICLE::GetPos(p_current);
                auto * vel = FML::PARTICLE::GetVel(p_current);

                // Add on the replica-shift
                for (int idim = 0; idim < NDIM; idim++)
                    pos[idim] += replica_shift[idim];

                // Compute radial comoving distance to the particle before the step (at a_old)
                double r_old_squared = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    double dx = frac_old == 0.0 ? pos[idim] : pos[idim] + delta_pos[idim] * frac_old;
                    r_old_squared += dx * dx;
                }

                // Do not process particle if comoving distance is larger than the size of the lightcone
                if (r_old_squared > r_lc_old_squared)
                    continue;

                // Compute radial comoving distance to the particle after the step (at a_new)
                double r_new_squared = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    double dx = frac_new == 1.0 ? pos[idim] + delta_pos[idim] : pos[idim] + delta_pos[idim] * frac_new;
                    r_new_squared += dx * dx;
                }

                // Do not process particle if comoving distance is smaller than the size of the lightcone after the step
                if (r_new_squared <= r_lc_new_squared)
                    continue;

                // Determine the crossing time and the fraction of
                // a step to take to the crossing
                double a_cross{};
                double fraction_of_step_to_take{};

                // METHOD 1 - The L-PICOLA method. Approximative, but much less operations
                // (r_lc(a) is approximated by the chord over the step [apos, apos_new])
                const double r_old = std::sqrt(r_old_squared);
                const double r_new = std::sqrt(r_new_squared);
                a_cross =
                    apos + (apos_new - apos) * (r_lc_apos - r_old) / ((r_new - r_old) - (r_lc_apos_new - r_lc_apos));
                fraction_of_step_to_take = (a_cross - apos) / (apos_new - apos);

                // If the lightcone starts or ends inside the step we know from the tests above that the particle
                // crosses the lightcone in [a_old, a_new], but the chord can put the crossing outside it (and
                // rejecting those particles loses particles). Then solve |x + f dx| = r_lc(apos + f (apos_new - apos))
                // for f in [frac_old, frac_new] by bisection
                const bool partial_step = frac_old > 0.0 or frac_new < 1.0;
                if (partial_step and (fraction_of_step_to_take < frac_old or fraction_of_step_to_take > frac_new)) {
                    double flow = frac_old, fhigh = frac_new;
                    for (int iter = 0; iter < 40; iter++) {
                        const double f = 0.5 * (flow + fhigh);
                        double r2 = 0.0;
                        for (int idim = 0; idim < NDIM; idim++) {
                            const double dx = pos[idim] + delta_pos[idim] * f;
                            r2 += dx * dx;
                        }
                        const double r_lc = get_distance(apos + (apos_new - apos) * f);
                        if (r2 > r_lc * r_lc)
                            fhigh = f;
                        else
                            flow = f;
                    }
                    fraction_of_step_to_take = 0.5 * (flow + fhigh);
                    a_cross = apos + (apos_new - apos) * fraction_of_step_to_take;
                }

                // Take a fraction of a step and get the crossing distance
                double r_crossing_squared = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    pos[idim] += delta_pos[idim] * fraction_of_step_to_take;
                    r_crossing_squared += pos[idim] * pos[idim];
                }

                // If the particle starts off very close to the boundary it might
                // slightly overshoot so cap r_crossing to boundary
                // Only for a very small fraction of particles, but lets record how many
                if (r_crossing_squared > r_lc_old_squared and (r_crossing_squared / r_lc_old_squared - 1) < 1e-2) {
                    r_crossing_squared = r_lc_old_squared;
                    nproblem++;
                } else if (r_crossing_squared < r_lc_new_squared and
                           (r_lc_new_squared / r_crossing_squared - 1) < 1e-2) {
                    r_crossing_squared = r_lc_new_squared;
                    nproblem++;
                }

                // Compute the error we do with the approximation above
                const double r_crossing = std::sqrt(r_crossing_squared);
                const double r_crossing_expected = get_distance(a_cross);
                const double error = std::abs(r_crossing / r_crossing_expected - 1.0);
                mean_error += error;
                mean_error_count += 1.0;
                max_error = std::max(max_error, error);

                // Add in units to the velocities (to get peculiar in km/s)
                // as this is not possible to know just from pos and vel data
                if (store_particles) {
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] *= boxsize;
                        vel[idim] *= vel_norm;
                    }
                }

                // Add particle to container
#ifdef USE_OMP
#pragma omp critical
#endif
                {
                    if (store_particles)
                        part_lc.push_back(p_current);

                    // Check if we should output
                    // Note: the way we do it is not great as other treads have to pause while we output
                    // Can be improved, but requires a lot of work
                    // (only with batches: then the output does no MPI communication)
                    if (batches_this_step and part_lc.size() >= npart_per_batch) {
                        output_particles(part_lc, a_old, a_new, particle_type, ibatch);
                        npart_in_written_batches += part_lc.size();
                        part_lc.resize(0);
                        ibatch++;
                    }
                }

                // Add the particle to the desired map
                if (make_onion_density_maps) {
                    auto & map = onionslices_curtype.get_map(r_crossing);
                    long int ipix = map.get_pixel_index(pos);
#ifdef USE_OMP
#pragma omp critical
#endif
                    map.add_particle_to_map(ipix);
                }
            }
        }
    } // End loop over particles

    // Compute some global info
    FML::SumOverTasks(&nproblem);
    FML::SumOverTasks(&mean_error);
    FML::SumOverTasks(&mean_error_count);
    FML::MaxOverTasks(&max_error);
    FML::MaxOverTasks(&max_disp);
    if (FML::ThisTask == 0) {
        std::cout << "# Error in |r_cross/r_lc-1|:  " << max_error * 100 << " % (max) "
                  << mean_error / mean_error_count * 100.0 << " % (mean)\n";
        std::cout << "# Maximum displacement in a step: " << max_disp * boxsize << " Mpc/h\n";
        std::cout << "# Number of particles that we had to snap to the boundary: " << nproblem << " = "
                  << nproblem / double(mean_error_count) * 100 << " %\n";
    }

    // Get number of particles and err_rmse across all procs
    size_t npart_lc_global = part_lc.size() + npart_in_written_batches;
    size_t npart_remaining_to_be_output = part_lc.size();
    FML::SumOverTasks(&npart_lc_global);
    FML::SumOverTasks(&npart_remaining_to_be_output);

    // Print stats about particles and mean overdensity and usage of memory
    if (FML::ThisTask == 0 and store_particles) {
        const double volume = fsky * volume_shell(r_lc_new, r_lc_old);
        const double density = npart_lc_global / volume;
        const double delta = density / mean_density[particle_type] - 1.0;
        std::cout << "# We have " << npart_lc_global
                  << " global particles = " << npart_lc_global / double(part.get_npart_total())
                  << " x NpartTotal in the lightcone\n";
        std::cout << "# Delta_shell = " << delta << "\n";
        std::cout << "# Fraction of allocated particles used: " << part_lc.size() / double(num_particles_to_allocate)
                  << "\n";
    }

    // Write to file if there are particles in the lightcone
    if (batches_this_step) {
        // The tasks have written different numbers of batches: all tasks write the same number (empty if
        // needed) so that each batch has NTasks files (as the headers say)
        int nbatches = ibatch + (part_lc.size() > 0 ? 1 : 0);
        FML::MaxOverTasks(&nbatches);
        while (ibatch < nbatches) {
            output_particles(part_lc, a_old, a_new, particle_type, ibatch++);
            part_lc.resize(0);
        }
    } else if (npart_remaining_to_be_output > 0) {
        output_particles(part_lc, a_old, a_new, particle_type, ibatch++);
    }
    if (FML::ThisTask == 0)
        std::cout << "# Done outputting particles" << std::endl;

    // Gather map data from across tasks and write files
    if (make_onion_density_maps) {

        // Reduce over tasks so that task 0 have a complete map
        onionslices_curtype.reduce_over_tasks();

        // Normalize map so that it becomes a density contrast (over the observed part of the sky, zero outside)
        onionslices_curtype.normalize_maps(
            mean_density[particle_type],
            [this](const std::array<double, NDIM> & n, double r) { return is_observed(n, r); });

        if (FML::ThisTask == 0) {

            // Output the map(s)
            const bool output_only_new_maps = true;
            const bool only_one_task_outputs = true;
            onionslices_curtype.output(plc_folder,
                                       "delta_" + particle_type,
                                       not output_only_new_maps,
                                       only_one_task_outputs);

            // Compute angular power-spectra (divided by the observed fsky) and output it with and without shot-noise
            const bool subtract_shotnoise = true;
            onionslices_curtype.compute_and_output_angular_powerspectrum(plc_folder,
                                                                         "delta_" + particle_type,
                                                                         not output_only_new_maps,
                                                                         subtract_shotnoise);
        }

        // Maps of the potential at the same shells
        if (relativistic_effects and potential_grid)
            make_potential_maps(*potential_grid, interpolation_method);

        // To reduce memory overhead we can free the map
        // We can easily just read it from file if we need it again
        onionslices_curtype.free();
    }

    if (FML::ThisTask == 0) {
        std::cout << "# Lightcone finished for current step!\n";
        std::cout << "#=====================================================\n\n";
    }

} // create_lightcone

template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::make_potential_maps([[maybe_unused]] const FML::GRID::FFTWGrid<NDIM> & potential_grid,
                                                [[maybe_unused]] const std::string & interpolation_method) {
#ifdef USE_HEALPIX
    if constexpr (NDIM == 3) {
        auto & slices = onionslices_of_type["cb"];
        auto & maps = slices.get_maps();
        const auto range = slices.current_step_range();
        struct Point {
            double pos[NDIM];
            double * get_pos() { return pos; }
            constexpr int get_ndim() const { return NDIM; }
        };
        for (int imap = range.first; imap < range.second; imap++) {
            const auto & dmap = maps[imap];
            OnionSlice<MapRealType, NDIM> phimap;
            phimap
                .init(dmap.get_rmin(), dmap.get_rmax(), dmap.get_amin(), dmap.get_amax(), slices.get_nside(), false, 0);
            const double rmid = 0.5 * (dmap.get_rmin() + dmap.get_rmax());
            const long npix = phimap.get_npix();

            // The pixel centers at rmid (periodic box) that are in the local domain
            std::vector<Point> points;
            std::vector<long> ipixs;
            for (long ipix = 0; ipix < npix; ipix++) {
                const auto n = phimap.pixel_direction(ipix);
                Point pt;
                for (int idim = 0; idim < NDIM; idim++) {
                    double x = pos_observer[idim] + rmid * n[idim];
                    x -= std::floor(x);
                    if (x >= 1.0)
                        x = 0.0;
                    pt.pos[idim] = x;
                }
                if (pt.pos[0] >= FML::xmin_domain and pt.pos[0] < FML::xmax_domain) {
                    points.push_back(pt);
                    ipixs.push_back(ipix);
                }
            }
            std::vector<FML::GRID::FloatType> values;
            FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, Point>(potential_grid,
                                                                                    points.data(),
                                                                                    points.size(),
                                                                                    values,
                                                                                    interpolation_method);
            for (size_t i = 0; i < points.size(); i++)
                phimap.set_pixel(ipixs[i], values[i]);
            phimap.sum_field_over_tasks(0);
            if (FML::ThisTask == 0) {
                const std::string label = "phi_cb";
                phimap.output(plc_folder + "/OnionMap_" + label + "_imap" + std::to_string(imap));
                phimap.compute_and_output_angular_powerspectrum(plc_folder + "/Cell_" + label + "_imap" +
                                                                    std::to_string(imap) + ".txt",
                                                                label,
                                                                false);
            }
            phimap.free();
        }
    }
#else
    if (FML::ThisTask == 0)
        std::cout << "# NB: potential maps need Healpix (USE_HEALPIX)\n";
#endif
}

// Compute comoving distance to redshift relation and set up spline interpolation
// This gives us a spline of chi(loga) in (Mpc/h)
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::compute_comoving_distance() {
    const double H0_hmpc = cosmo->get_H0_hmpc();
    const double H0Box = H0_hmpc * boxsize;

    if (FML::ThisTask == 0) {
        std::cout << "# Lightcone - Setting up distance interpolation\n";
    }

    DVector loga_arr(npts_loga);
    for (int i = 0; i < npts_loga; i++) {
        loga_arr[i] = std::log(ahigh) + std::log(alow / ahigh) * i / double(npts_loga);
    }

    // integrate dz/H0/E(z) from 0 to z
    // or -dlna/H0/(a*E(z)) from 0 to ln(a)
    FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, [[maybe_unused]] const double * y, double * dydx) {
        const double a = std::exp(x);
        dydx[0] = -1.0 / (H0Box * a * cosmo->HoverH0_of_a(a));
        return GSL_SUCCESS;
    };

    FML::SOLVERS::ODESOLVER::ODESolver ode;
    std::vector<double> yini{0.0};
    ode.solve(deriv, loga_arr, yini);

    auto dist = ode.get_data_by_component(0);
    dist_spline.create(loga_arr, dist, "dist_spline");
    dist_inv_spline.create(dist, loga_arr, "dist_inv_spline");

    // Get ranges we need to respect
    dist_low = get_distance(ahigh);
    dist_high = get_distance(alow);
}

// Convert scale factor to comoving distance
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::get_distance(double a) {
    if (a < alow || a > ahigh) {
        if (FML::ThisTask == 0) {
            std::cout << "WARNING! Scale factor outside spline range. a " << a << " [" << alow << ", " << ahigh
                      << "]\n";
        }
    }
    double loga = std::log(a);
    return dist_spline(loga);
}

// Convert comoving distance to scale factor
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::get_distance_inv(double dist) {
    if (dist < dist_low || dist > dist_high) {
        if (FML::ThisTask == 0) {
            std::cout << "WARNING! Distance outside spline range. dist " << dist * boxsize << " Mpc/h ["
                      << dist_low * boxsize << ", " << dist_high * boxsize << "]\n";
        }
    }
    return std::exp(dist_inv_spline(dist));
}

// Output the lightcone-particles as a gadget file
// NB: the velocities are not normalized as in "standard" gadget snapshot format (i.e. the sqrt(a))
// NB: as currently written this only work seamlessly for CDM as with multiple species we currently need MPI-calls
// in write_gadget_single and not all tasks might output.
// With output-in-batches the NumPartTotal in the gadget files will just be NumPart as we cannot easily get this
// information on the fly
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_gadget(FML::Vector<U> & part_lc,
                                          double astart,
                                          double aend,
                                          std::string label,
                                          int ibatch) {

    const int nfiles = FML::NTasks;
    std::string fileprefix;
    if (batches_this_step)
        fileprefix = plc_folder + "/" + "lightcone_" + label + "_gadget_step" + std::to_string(plc_step) + "_batch" +
                     std::to_string(ibatch) + ".";
    else
        fileprefix = plc_folder + "/" + "lightcone_" + label + "_gadget_step" + std::to_string(plc_step) + ".";
    double pos_norm = 1.0;
    double vel_norm = 1.0;
    double scale_factor = (astart + aend) / 2.0;

    if (FML::ThisTask == 0)
        std::cout << "# Writing lightcone to gadget file(s): " << fileprefix << "X\n";

    size_t npart_lc = part_lc.size();
    size_t npart_lc_global = npart_lc;

    // If we output all files at the same time then its fine to do MPI calls
    if (not batches_this_step) {
        FML::SumOverTasks(&npart_lc_global);
    } else {
        // In this case we don't have a good way of getting the total number of particles in all files
        // so we just set NumPartTot = NumPart
        npart_lc_global = npart_lc;
    }

    FML::FILEUTILS::GADGET::GadgetWriter gw;
    gw.write_gadget_single(fileprefix + std::to_string(FML::ThisTask),
                           part_lc.data(),
                           npart_lc,
                           npart_lc_global,
                           nfiles,
                           scale_factor,
                           boxsize,
                           cosmo->get_OmegaM(),
                           cosmo->get_OmegaLambda(),
                           cosmo->get_h(),
                           pos_norm,
                           vel_norm);
}

// Output the lightcone-particles as GADGET HDF5 files (one file per task, and per batch with output in batches).
// Positions are comoving in Mpc/h (not wrapped into the box: with replicas they extend beyond it; the observer
// is at plc_pos_observer * boxsize) and
// velocities are peculiar velocities in km/s (not the GADGET convention v/sqrt(a)), as for the GADGET lightcone.
// The header time is the mean of the crossing scale factors [astart, aend] of the step.
// With output in batches NumPartTotal is just NumPart (we cannot get the total on the fly)
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_hdf5([[maybe_unused]] FML::Vector<U> & part_lc,
                                        [[maybe_unused]] double astart,
                                        [[maybe_unused]] double aend,
                                        [[maybe_unused]] std::string label,
                                        [[maybe_unused]] int ibatch) {
#ifdef USE_HDF5
    std::string fileprefix = plc_folder + "/" + "lightcone_" + label + "_hdf5_step" + std::to_string(plc_step);
    if (batches_this_step)
        fileprefix += "_batch" + std::to_string(ibatch);
    if (FML::ThisTask == 0)
        std::cout << "# Writing lightcone to HDF5 file(s): " << fileprefix << ".X.hdf5\n";

    size_t npart_lc = part_lc.size();
    size_t npart_lc_global = npart_lc;
    if (not batches_this_step)
        FML::SumOverTasks(&npart_lc_global);

    FML::FILEUTILS::GADGET::GadgetHDF5Writer gw(NDIM);
    gw.set_periodic_wrap(false);
    if (relativistic_effects) {
        // The same as below plus PartType1/Potential (Phi/c^2), RadialVelocity (km/s), CosmologicalRedshift and
        // ObservedRedshift
        auto * p = part_lc.data();
        auto sp = FML::FILEUTILS::GADGET::make_gadget_species(1, p, npart_lc, npart_lc_global, 1.0);
        sp.extra_scalars.push_back({"Potential", [p](size_t i) { return double(p[i].get_potential()); }});
        sp.extra_scalars.push_back(
            {"RadialVelocity", [this, p](size_t i) { return relativistic_quantities(p[i])[0]; }});
        sp.extra_scalars.push_back(
            {"CosmologicalRedshift", [this, p](size_t i) { return relativistic_quantities(p[i])[1]; }});
        sp.extra_scalars.push_back(
            {"ObservedRedshift", [this, p](size_t i) { return relativistic_quantities(p[i])[2]; }});
        gw.write_gadget_species(fileprefix + "." + std::to_string(FML::ThisTask) + ".hdf5",
                                {sp},
                                FML::NTasks,
                                (astart + aend) / 2.0,
                                boxsize,
                                cosmo->get_OmegaM(),
                                cosmo->get_OmegaLambda(),
                                cosmo->get_h(),
                                1.0,
                                1.0);
        return;
    }
    gw.write_gadget_single(fileprefix + "." + std::to_string(FML::ThisTask) + ".hdf5",
                           part_lc.data(),
                           npart_lc,
                           npart_lc_global,
                           FML::NTasks,
                           (astart + aend) / 2.0,
                           boxsize,
                           cosmo->get_OmegaM(),
                           cosmo->get_OmegaLambda(),
                           cosmo->get_h(),
                           1.0,
                           1.0);
#else
    throw std::runtime_error("plc_output_hdf5file needs HDF5: compile with USE_HDF5=true");
#endif
}

template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_particles(FML::Vector<U> & part_lc,
                                             double astart,
                                             double aend,
                                             std::string label,
                                             int ibatch) {
    if (output_gadgetfile)
        output_gadget(part_lc, astart, aend, label, ibatch);
    if (output_asciifile)
        output_ascii(part_lc, astart, aend, label, ibatch);
    if (output_hdf5file)
        output_hdf5(part_lc, astart, aend, label, ibatch);
}

// Output the lightcone-particles as an ascii file
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_ascii(FML::Vector<U> & part_lc,
                                         double astart,
                                         double aend,
                                         std::string label,
                                         int ibatch) {
    std::string filename = plc_folder + "/" + "lightcone_" + label + "_ascii_step" + std::to_string(plc_step) + ".";
    if (FML::ThisTask == 0)
        std::cout << "# Writing lightcone to ascii file(s): " << filename << "X\n";
    filename += std::to_string(FML::ThisTask);

    size_t npart = part_lc.size();
    size_t npart_total = npart;
    if (not batches_this_step)
        FML::SumOverTasks(&npart_total);

    std::ofstream fp;
    if (ibatch == 0)
        fp.open(filename);
    else
        fp.open(filename, std::ios_base::app);

    // What to output
    const bool output_mass = false and FML::PARTICLE::has_get_mass<T>();
    const bool output_pos = true;
    const bool output_vel = false;
    const bool output_vr = false;
    std::string what_we_output = "# The column below are ";
    if (output_mass)
        what_we_output += " mass (in number of particles) ";
    if (output_pos)
        what_we_output += " pos_vec (Mpc/h) ";
    if (output_vel)
        what_we_output += " vel_vec (km/s) ";
    if (output_vr)
        what_we_output += " LOS_velocity (km/s) ";
    if (relativistic_effects)
        what_we_output += " Phi/c^2 v_r (km/s) z_cos z_obs ";

    // Output header only the first time
    if (ibatch == 0) {
        fp << "# Lightcone " << label << "-particles from Task " << FML::ThisTask + 1 << " / " << FML::NTasks << "\n";
        fp << "# The lightcone crossing scale-factors for this sample is in [" << std::to_string(astart) << " -> "
           << std::to_string(aend) << "]\n";
        fp << "# The corresponding comoving distances is in [" << get_distance(aend) * boxsize << " -> "
           << get_distance(astart) * boxsize << "] Mpc/h\n";
        if (batches_this_step)
            fp << "# This file is written in batches so no particle number information can be written here\n";
        else {
            fp << "# This file has " << npart << " / " << npart_total << " particles\n";
        }
        fp << what_we_output << "\n";
    }

    for (auto & p : part_lc) {
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * vel = FML::PARTICLE::GetVel(p);

        // Output the mass
        if (output_mass) {
            if constexpr (FML::PARTICLE::has_get_mass<T>()) {
                fp << std::setw(10) << FML::PARTICLE::GetMass(p) << " ";
            }
        }

        // Output the position
        if (output_pos) {
            for (int idim = 0; idim < NDIM; idim++)
                fp << std::setw(10) << pos[idim] << " ";
        }

        // Output the velocity
        if (output_vel) {
            for (int idim = 0; idim < NDIM; idim++)
                fp << std::setw(10) << vel[idim] << " ";
        }

        // Output the radial velocity
        if (output_vr) {
            double vr = 0.0, r_squared = 0.0;
            for (int idim = 0; idim < NDIM; idim++) {
                vr += vel[idim] * pos[idim];
                r_squared += pos[idim] * pos[idim];
            }
            vr /= std::sqrt(r_squared);
            fp << std::setw(10) << vr << " ";
        }

        // The potential, radial velocity and the cosmological and observed redshift
        if (relativistic_effects) {
            const auto q = relativistic_quantities(p);
            fp << std::setw(14) << std::setprecision(7) << p.get_potential() << " " << std::setw(12) << q[0] << " "
               << std::setw(14) << std::setprecision(9) << q[1] << " " << std::setw(14) << q[2] << " ";
            fp << std::setprecision(6);
        }
        fp << "\n";
    }

    // An option here is to save memory is to just output the radial velocity (needed for RSD)
    // E.g. double vr = 0.0, r_squared = 0.0;
    // for (int idim = 0; idim < NDIM; idim++) {
    // vr += vel[idim]*pos[idim]; r_squared += pos[idim]*pos[idim];
    // }
    // vr /= std::sqrt(r2);
}

// Volume of a spherical shell bounded by rlow and rhigh in NDIM dimension
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::volume_shell(double rlow, double rhigh) {
    if constexpr (NDIM == 3)
        return 4.0 / 3.0 * M_PI * (rhigh * rhigh * rhigh - rlow * rlow * rlow);
    else if (NDIM == 2)
        return M_PI * (rhigh * rhigh - rlow * rlow);
    else if (NDIM == 1)
        return 2.0 * (rhigh - rlow);
    else
        throw std::runtime_error("Volume-shell only implemented for ndim = 1,2,3");
}

// The fraction of the lightcone shell rlow < r < rhigh (units of the box) that is in the x-domain of this task,
// for reserving memory for the lightcone particles. The cross section of the shell at x (relative to the observer)
// is taken proportional to (rhigh^2 - x^2)^((NDIM-1)/2) - (rlow^2 - x^2)^((NDIM-1)/2), i.e. the same fraction of
// the full sphere at all x (exact for an observer at the origin), over the x-range of the box and its replicas
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::local_fraction_of_shell(double rlow, double rhigh) {
    if (FML::NTasks == 1)
        return 1.0;
    const auto nrep = replicas.get_replica_range(0);
    const double xmin = -pos_observer[0] - nrep.first;
    const double xmax = 1.0 - pos_observer[0] + nrep.second;
    const int npts = 1000 * (nrep.first + nrep.second + 1);
    const double dx = (xmax - xmin) / npts;
    auto cross_section = [&](double r, double x) {
        return r * r > x * x ? std::pow(r * r - x * x, 0.5 * (NDIM - 1)) : 0.0;
    };
    double local = 0.0, total = 0.0;
    for (int i = 0; i < npts; i++) {
        const double x = xmin + (i + 0.5) * dx;
        const double w = cross_section(rhigh, x) - cross_section(rlow, x);
        // The position in the box
        const double xbox = x + pos_observer[0] - std::floor(x + pos_observer[0]);
        total += w;
        if (xbox >= FML::xmin_domain and xbox < FML::xmax_domain)
            local += w;
    }
    return total > 0.0 ? local / total : 1.0 / FML::NTasks;
}

template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::free() {
    if (make_onion_density_maps)
        for (auto & maps : onionslices_of_type)
            maps.second.free();
    onionslices_of_type = std::map<std::string, OnionSlices<MapRealType, NDIM>>{};
}

template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::create_weak_lensing_maps(std::string particle_type) {
    // This routine currently assumes that all maps are reduced and reside on task 0
    if (FML::ThisTask != 0)
        return;
    if (not make_onion_density_maps)
        return;

    // Get a vector of all the delta-maps
    auto & maps = onionslices_of_type[particle_type].get_maps();
    if (maps.size() == 0)
        return;

    // Normalization factor for distances (which are in units of box)
    const double H0_hmpc = cosmo->get_H0_hmpc();
    const double H0Box = H0_hmpc * boxsize;
    const double prefactor_kappa = 1.5 * cosmo->get_OmegaM();

    // Define what we mean by the "r" of a delta-map over the range [rmin,rmax]
    // We put this to be the middle of the map
    auto get_r_of_delta_map = [&](OnionSlice<double, NDIM> & map) -> double {
        return (map.get_rmin() + map.get_rmax()) / 2.0;
    };

    // Allocate kappa-map container
    std::vector<OnionSlice<double, NDIM>> kappa_maps(maps.size());

    // Loop over the delta-maps (start with the map closest to the observer)
    for (int i = int(maps.size()) - 1; i >= 0; i--) {

        // The delta map
        auto & delta_map = maps[i];
        // Read the map we have saved from file
        delta_map.read_saved_map();
        // The number of pixels in the map
        const auto nside = delta_map.get_nside();
        const auto npix = delta_map.get_npix();
        // The raw data (a vector of floats from 0 -> npix)
        const auto delta_map_data = delta_map.get_map_data();
        // The a-range for the map
        const double amin = delta_map.get_amin();
        const double amax = delta_map.get_amax();
        // The r-range for the map
        const double rmin = delta_map.get_rmin();
        const double rmax = delta_map.get_rmax();

        // The r of the delta-map
        const double r_delta_map = get_r_of_delta_map(delta_map);
        [[maybe_unused]] double a_delta_map = (amin + amax) / 2.0;

        // Allocate all the kappa-maps
        if (i == int(maps.size()) - 1) {
            std::cout << "# Allocating kappa-maps\n";
            for (size_t j = 0; j < maps.size(); j++) {
                const auto & curmap = maps[j];
                const double amap = curmap.get_amin();
                const double rmap = curmap.get_rmax();
                kappa_maps[j].init(rmap, rmap, amap, amap, nside, false, 0);
            }
            std::cout << "# Memory used: "
                      << maps.size() * npix * sizeof(maps[0].get_map_data()[0]) / double(1024. * 1024.) << " MB\n";
        }

        std::cout << "# Processing delta-map i = " << i << " ";
        std::cout << "r_map = " << r_delta_map * boxsize << " Mpc/h ";
        std::cout << "[rmin = " << rmin * boxsize << " -> rmax = " << rmax * boxsize << " Mpc/h]\n";

        // Lopp over all kappa-maps and with j <= i and add the contribution to the
        // integral from the current slice
        for (int j = 0; j <= i; j++) {
            auto & kappa_map = kappa_maps[j];
            auto * kappa_map_data = kappa_map.get_map_data();
            const double r_kappa_map = kappa_map.get_rmax();

            // Integrate from amin -> amax
            const int npts = 100;
            const double xmin = std::log(amin);
            const double xmax = std::log(amax);
            const double dx = (xmax - xmin) / double(npts);
            double integral = 0.0;
            for (int k = 0; k < npts; k++) {
                const double x = xmin + k * dx;
                const double a = std::exp(x);
                const double r = get_distance(a);
                const double g = (r_kappa_map - r) / r_kappa_map;
                integral += g * (r * (H0Box)) / a * (dx / (a * cosmo->HoverH0_of_a(a)));
            }

            integral *= prefactor_kappa;

            // An alternative way of doing this (approximate the full integral Int delta g dchi as a discrete sum):
            // Mostly gives the same result up to a few percent
            // integral = prefactor / a_delta_map * ((r_kappa_map - r_delta_map) / r_kappa_map) * r_delta_map * (rmax -
            // rmin) * std::pow(H0Box, 2);

            std::cout << "# Adding delta i = " << i << " ";
            std::cout << "[" << rmin * boxsize << " -> " << rmax * boxsize << " Mpc/h] ";
            std::cout << "to kappa-map j = " << j << " rmap = " << r_kappa_map * boxsize << " Mpc/h\n";
            for (int k = 0; k < npix; k++) {
                kappa_map_data[k] += delta_map_data[k] * integral;
            }
        }

        // Only the part of the sky observed in all the delta-maps is complete: this is the observed part at the
        // distance of the kappa-map (the region covered by the box and its replicas can only shrink with the
        // distance). Zero outside
        kappa_maps[i].apply_mask([this](const std::array<double, NDIM> & n, double r) { return is_observed(n, r); },
                                 kappa_maps[i].get_rmax());

        // Output the kappa-map and free the memory (the first map is all zeros so skip this)
        std::cout << "# Outputting kappa-map with ";
        std::cout << "a = " << kappa_maps[i].get_amin() << " ";
        std::cout << "r = " << kappa_maps[i].get_rmax() * boxsize << " Mpc/h\n";
        std::string filename = plc_folder + "/OnionMap_kappa_" + particle_type + "_imap" + std::to_string(i);
        kappa_maps[i].output(filename);

        // Compute min/max of kappa
        double kappa_min = std::numeric_limits<double>::max();
        double kappa_max = -std::numeric_limits<double>::max();
        double kappa_mean = 0.0;
        const auto * kappa_data = kappa_maps[i].get_map_data();
        for (int k = 0; k < npix; k++) {
            const double kappa = kappa_data[k];
            kappa_min = std::min(kappa_min, kappa);
            kappa_max = std::max(kappa_max, kappa);
            kappa_mean += kappa;
        }
        kappa_mean /= double(npix);
        std::cout << "# Summary kappa: (min,mean,max) = (" << kappa_min << " , " << kappa_mean << " , " << kappa_max
                  << ")\n";

        // Compute and output C_kappakappa(ell)
        std::string filename_cell = plc_folder + "/Cell_kappa_" + particle_type + "_imap" + std::to_string(i) + ".txt";
        kappa_maps[i].compute_and_output_angular_powerspectrum(filename_cell, "kappa_" + particle_type, false);
        kappa_maps[i].free();

        // Free the density-map no longer needed
        delta_map.free();
    }

    // Output a list of the a and r of the map
    std::string filename_kappa_info = plc_folder + "/OnionMap_kappa_bininfo.txt";
    std::ofstream fp(filename_kappa_info);
    fp << "# Info about the maps we output\n";
    fp << "# imap     amap           rmap (Mpc/h)\n";
    for (size_t i = 0; i < kappa_maps.size(); i++) {
        fp << std::setw(3) << i << " " << std::setw(12) << kappa_maps[i].get_amin() << " " << std::setw(12)
           << kappa_maps[i].get_rmax() * boxsize << "\n";
    }
}

#endif
