#ifndef LIGHTCONE_HEADER
#define LIGHTCONE_HEADER

#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/ParameterMap/ParameterMap.h>
#include "Lightcone_Healpix.h"
#include "Lightcone_Replicas.h"

#include <iostream>
#include <cmath>

// This version of the lightcone was originally written by Albert Izard.
// I added the replicas with help from the L-PICOLA implementation (written by Howlett Cullan).

// The particle-type we use to store the light-cone particles
// This is a minimal particle and we preferably will want to use float
// to reduce the memory consumption
template<class T, int NDIM>
class LightConeParticle {
  private:
    T pos[NDIM];
    T vel[NDIM];
  public:
    T* get_pos() { return pos; }
    T* get_vel() { return vel; }
    
    constexpr int get_ndim() { return NDIM; }

    // Assign this particle from another particle (of possibly different type)
    template<class U>
      void assign_from_particle(U & p) {
        auto * _pos = FML::PARTICLE::GetPos(p);
        auto * _vel = FML::PARTICLE::GetVel(p);
        for(int idim = 0; idim < NDIM; idim++){
          pos[idim] = _pos[idim];
          vel[idim] = _vel[idim];
        }
      }
};

// The Lightcone class. This deals with anything related to making lightcones
// and associated healpix density-maps
template <int NDIM, class T, class U = LightConeParticle<float, NDIM>>
class Lightcone {
  protected:

    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Point = std::array<double, NDIM>;

    // The cosmology needed to compute comoving distance r(a)
    std::shared_ptr<Cosmology> cosmo;

    // Lightcone parameters
    bool lightcone_on{false};

    // Initial and final time for the lightcone
    double plc_a_init;
    double plc_a_finish; 
    
    // Healpix-maps
    bool build_healpix;
    HealpixMaps hp_maps;

    // Replicas
    BoxReplicas<NDIM> replicas;

    // For setting filenames and doing outputs
    std::string simulation_name;
    std::string output_folder;
    std::string plc_folder;
    bool output_gadgetfile;
    bool output_asciifile;
    bool store_particles;

    // Boxsize of the simulation
    double boxsize;
    // Mean density of the tracers
    double mean_density;
    // The origin of the light-cone
    Point pos_observer = {};
    // Sky coverage
    double fsky;
    // Number of calls to the light cone (starts counting at 0)
    int plc_step; 

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
    void output_ascii(std::vector<U> & part_lc, double astart, double aend);
    void output_gadget(std::vector<U> & part_lc, double astart, double aend);

    // Useful function - volume of a shell (in any dim)
    double volume_shell(double rlow, double rhigh);

  public:

    Lightcone(std::shared_ptr<Cosmology> cosmo) : cosmo(cosmo) {};

    void read_parameters(ParameterMap & param);

    void init();

    void create_lightcone(
        FML::PARTICLE::MPIParticles<T> & part,
        double apos,
        double apos_new,
        double avel,
        double delta_time_drift);

    bool lightcone_active() { return lightcone_on; } 
};

// Read all the parameters we need for the lightcone
// This also calls the read_parameter function for healpix and replicas
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::read_parameters(ParameterMap & param){

  lightcone_on = param.get<bool>("lightcone");
  if (lightcone_on) {
    simulation_name = param.get<std::string>("simulation_name");
    output_folder = param.get<std::string>("output_folder");
    boxsize = param.get<double>("simulation_boxsize");
    int npart_1D = param.get<int>("particle_Npart_1D");
    mean_density = std::pow(npart_1D, NDIM);
    auto origin = param.get<std::vector<double>>("plc_pos_observer");
    FML::assert_mpi(origin.size() >= NDIM, "Position of observer much have NDIM components");
    for(int idim = 0; idim < NDIM; idim++)
      pos_observer[idim] = origin[idim];

    double plc_z_init = param.get<double>("plc_z_init");
    plc_a_init   = 1.0 / (1.0+plc_z_init);
    double plc_z_finish = param.get<double>("plc_z_finish");
    plc_a_finish = 1.0 / (1.0+plc_z_finish);
    output_gadgetfile = param.get<bool>("plc_output_gadgetfile");
    output_asciifile = param.get<bool>("plc_output_asciifile");
    build_healpix = param.get<bool>("plc_build_healpix");
    build_healpix = build_healpix and NDIM == 3;
#ifndef USE_HEALPIX
    build_healpix = false;
#endif
    if (FML::ThisTask == 0) {
      std::cout << "lightcone                                : " << lightcone_on << "\n";
      std::cout << "plc_pos_observer                         : ";
      for(auto & p : pos_observer)
        std::cout << p << " , ";
      std::cout << "\n";
      std::cout << "plc_z_init                               : " << plc_z_init << "\n";
      std::cout << "plc_z_finish                             : " << plc_z_finish << "\n";
      std::cout << "plc_output_gadgetfile                    : " << output_gadgetfile << "\n";
      std::cout << "plc_output_asciifile                     : " << output_asciifile << "\n";
      std::cout << "plc_build_healpix                        : " << build_healpix << "\n";
      std::cout << "mean_density                             : " << mean_density << "\n";
    }
    replicas.read_parameters(param);
    if(build_healpix)
      hp_maps.read_parameters(param);
  }
}

// Initialize the lightcone
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::init() {
  if(not lightcone_on) return;

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
  if(FML::ThisTask == 0) {
    std::cout << "# Lightcone will cover fsky = " << fsky << "\n";
  }

  // We don't care about the part of the lightcone extremely close to the observer
  if(plc_a_finish > 0.995){
    plc_a_finish = 0.995;
    if(FML::ThisTask == 0) {
      std::cout << "# Warning: plc_a_finish was too high, changed to " << plc_a_finish << "\n";
    }
  }

  // Set up interpolation to compute distances
  compute_comoving_distance();

  // Init step number
  plc_step = -1;

  // Init the replicas
  replicas.init(plc_a_init, get_distance(plc_a_init));
}

// Creates the lightcone every step
// This method does not modify part in any way
// We only take a copy of the particles when they cross the lightcone
// and store them in part_lc
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::create_lightcone(
    FML::PARTICLE::MPIParticles<T> & part,
    double apos,
    double apos_new,
    double avel,
    double delta_time_drift) {

  if(not lightcone_on) return;

  // Check if we are inside the lightcone range
  // The accuracy on the cut around plc_a_finish is irrelevant in most realistic cases.
  // Therefore we can ignore the buffer zone around apos.
  if(apos_new < plc_a_init || apos > plc_a_finish) {
    if (FML::ThisTask == 0){
      std::cout << "# Outside of the lightcone range so won't do lightcone drift\n";
    }
    return;
  }
  
  // Increase step-number
  plc_step++;

  // Get distances for the initial/final limits of the light cone
  const double a_old = std::max(apos, plc_a_init);
  const double a_new = std::min(apos_new, plc_a_finish);
  const double r_lc_old = get_distance(a_old);
  const double r_lc_old_squared = r_lc_old * r_lc_old;
  const double r_lc_new = get_distance(a_new);
  const double r_lc_new_squared = r_lc_new * r_lc_new;

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
    std::cout << "# Drifting a = " << apos << " -> " << apos_new << "\n";
    std::cout << "# delta_time_drift " << delta_time_drift << "\n";
    std::cout << "# Slice: " << r_lc_new * boxsize << " - " << r_lc_old * boxsize << " Mpc/h\n";
  }

  // Flag the replicates that don't need looping over
  replicas.flag(r_lc_old, r_lc_new);

  // Estimate how many particles we should reserve based on the volume and the average number density
  // If no output method is picked then we don't save anything
  store_particles = output_asciifile or output_gadgetfile;
  size_t num_particles_to_allocate = ceil(mean_density * fsky * volume_shell(r_lc_new, r_lc_old) * particle_allocation_factor);
  if(FML::ThisTask == 0) {
    if(store_particles) {
      std::cout << "# We will allocate storage for " << num_particles_to_allocate / double(part.get_npart()) << " times the number of particles we have\n";
      std::cout << "# Memory consumption for particles: " << num_particles_to_allocate * sizeof(U) / double(1024.*1024.) / FML::NTasks << " MB average per task\n";
    } else {
      std::cout << "# NB: no output picked so we will not store and output particles\n";
    }
  }

  // Allocate and initialize the healpix maps we need for the current time-step
  if(build_healpix)
    hp_maps.init_current_step(a_old, a_new, dist_spline, boxsize, plc_folder);

  // Allocate lightcone particles
  std::vector<U> part_lc;
  if(store_particles)
    part_lc.reserve(num_particles_to_allocate);

  // Loop over all particles
  double mean_error = 0.0;
  double mean_error_count = 0.0;
  double max_error = 0.0;
  double max_disp = 0.0;
  double vel_norm = 100 * boxsize / avel;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : mean_error) reduction(+ : mean_error_count) reduction(max : max_error) reduction(max : max_disp)
#endif
  for (size_t ipart = 0; ipart < part.get_npart(); ipart++) {

    // Take a copy of the particle reducing it to just position and velocity
    // This is what we are going to store and output so lets save some memory
    U p_bare;
    p_bare.assign_from_particle(part[ipart]);

    // Measure the position relative to the observer position
    // and compute the distance per dimension that the particle moves over a full step
    Point delta_pos;
    auto * pos_bare = FML::PARTICLE::GetPos(p_bare);
    auto * vel_bare = FML::PARTICLE::GetVel(p_bare);
    for(int idim =0; idim<NDIM; idim++) {
      pos_bare[idim] -= pos_observer[idim];
      delta_pos[idim] = vel_bare[idim] * delta_time_drift;
      max_disp = std::max(max_disp, std::abs(delta_pos[idim]));
    }

    // Loop over all replicas
    auto n_replicas_total = replicas.get_n_replicas();
    for(size_t i = 0; i < n_replicas_total; i++) {

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
        for(int idim =0; idim<NDIM; idim++)
          pos[idim] += replica_shift[idim];

        // Compute radial comoving distance to the particle before the step
        double r_old_squared = 0.0;
        for(int idim = 0; idim<NDIM; idim++){
          double dx = pos[idim];
          r_old_squared += dx * dx;
        }

        // Do not process particle if comoving distance is larger than the size of the lightcone 
        if(r_old_squared > r_lc_old_squared)
          continue;

        // Compute radial comoving distance to the particle after a full step
        double r_new_squared = 0.0;
        for(int idim = 0; idim<NDIM; idim++){
          double dx = pos[idim] + delta_pos[idim];
          r_new_squared += dx * dx;
        }

        // Do not process particle if comoving distance is smaller than the size of the lightcone after the step
        if(r_new_squared <= r_lc_new_squared)
          continue;

        // Compute crossing scale-factor
        const double r_old = std::sqrt(r_old_squared);
        const double r_new = std::sqrt(r_new_squared);
        const double a_L = apos + (apos_new - apos) * (r_lc_old - r_old) / ((r_new - r_old) - (r_lc_new - r_lc_old));
        const double delta_factor = (a_L - apos) / (apos_new - apos);

        // Take a fraction of a step
        double r_crossing = 0.0;
        for(int idim =0; idim<NDIM; idim++){
          pos[idim] += delta_pos[idim] * delta_factor;
          double dx = pos[idim];
          r_crossing += dx * dx;
        }
        r_crossing = std::sqrt(r_crossing);

        // Compute the error we do with the approximation above
        const double r_crossing_expected = get_distance(a_L);
        const double error = std::abs(r_crossing / r_crossing_expected - 1.0);
        mean_error += error;
        mean_error_count += 1.0;
        max_error = std::max(max_error, error);

        // Add in units to the velocities (to get peculiar in km/s)
        // as this is not possible to know just from pos and vel data
        for(int idim =0; idim<NDIM; idim++){
          pos[idim] *= boxsize;
          vel[idim] *= vel_norm;
        }

        // Add particle to container
#ifdef USE_OMP
#pragma omp critical
#endif
        if(store_particles)
          part_lc.push_back(p_current);

        // Add the particle to the desired healpix map
        if (build_healpix) {
          auto & healpix_map = hp_maps.get_map(r_crossing);
          long int ipix;
          if constexpr (std::is_same_v<decltype(pos[0]), double>) {
            ipix = healpix_map.get_pixel_index(pos);
          } else {
            double pos_temp[NDIM];
            for(int idim = 0; idim < NDIM; idim++)
              pos_temp[idim] = pos[idim];
            ipix = healpix_map.get_pixel_index(pos_temp);
          }
#ifdef USE_OMP
#pragma omp critical
#endif
          healpix_map.add_particle_to_map(ipix);
        }
      }
    }
  } // End loop over particles

  // Compute some global info
  FML::SumOverTasks(&mean_error);
  FML::SumOverTasks(&mean_error_count);
  FML::MaxOverTasks(&max_error);
  FML::MaxOverTasks(&max_disp);
  if(FML::ThisTask == 0) {
    std::cout << "# Error in |r_cross/r_lc-1|:  " << max_error * 100 << " % (max) " << mean_error / mean_error_count * 100.0 << " % (mean)\n";
    std::cout << "# Maximum displacement in a step: " << max_disp * boxsize << " Mpc/h\n";
  }

  // Get number of particles and err_rmse across all procs
  size_t npart_lc_global = part_lc.size();
  FML::SumOverTasks(&npart_lc_global);

  // If particles have been stored and there are none then just exit
  if(npart_lc_global == 0 and store_particles) {
    if(FML::ThisTask == 0)
      std::cout << "# Zero particles in the lightcone current step\n";
    return;
  }

  // Print stats about particles and mean overdensity and usage of memory
  if (FML::ThisTask == 0 and store_particles) {
    double volume = fsky * volume_shell(r_lc_new, r_lc_old);
    double density = npart_lc_global / volume;
    double delta = density/mean_density-1.0;
    std::cout << "# We have " << npart_lc_global << " global particles = " << npart_lc_global / double(part.get_npart_total()) << " x NpartTotal in the lightcone. Delta_shell = " << delta << "\n";
    std::cout << "# Fraction of allocated particles used: " << part_lc.size() / double(num_particles_to_allocate) << "\n"; 
  }

  // Write to file if there are particles in the lightcone
  if(npart_lc_global > 0) {
    if(output_gadgetfile)
      output_gadget(part_lc, a_old, a_new);
    if(output_asciifile)
      output_ascii(part_lc, a_old, a_new);
  }

  // Gather healpix map data from across tasks and write files
  if(build_healpix){
    // Reduce over tasks so that task 0 have a complete map
    hp_maps.finalize();

    if(FML::ThisTask == 0) {
      // Normalize map so that it becomes a density contrast
      hp_maps.normalize_maps(mean_density);

      // Output the map(s)
      const bool output_only_new_maps = true;
      hp_maps.output(plc_folder, not output_only_new_maps); 
    }
  }

  // Deallocate lightcone particles
  part_lc.clear();
  part_lc.shrink_to_fit();

  if (FML::ThisTask == 0) 
    std::cout << "# Lightcone finished for current step!\n\n";

} // create_lightcone

// Compute comoving distance to redshift relation and set up spline interpolation
// This gives us a spline of chi(loga) in (Mpc/h)
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::compute_comoving_distance() {
  const double H0Box = boxsize / 2997.92458;

  if(FML::ThisTask == 0){
    std::cout << "# Lightcone - Setting up distance interpolation\n";
  }

  DVector loga_arr(npts_loga);
  for (int i = 0; i < npts_loga; i++) {
    loga_arr[i] = std::log(ahigh) + std::log(alow / ahigh) * i / double(npts_loga);
  }

  // integrate dz/H0/E(z) from 0 to z
  // or -dlna/H0/(a*E(z)) from 0 to ln(a)
  FML::SOLVERS::ODESOLVER::ODEFunction deriv =
    [&](double x, [[maybe_unused]] const double * y, double * dydx) {
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
  if(a<alow || a>ahigh){
    if(FML::ThisTask == 0){
      std::cout << "WARNING! Scale factor outside spline range. a "
        << a << " [" << alow << ", " << ahigh << "]\n";
    }
  }
  double loga = std::log(a);
  return dist_spline(loga);
}

// Convert comoving distance to scale factor
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::get_distance_inv(double dist) {
  if(dist<dist_low || dist>dist_high){
    if(FML::ThisTask==0){
      std::cout << "WARNING! Distance outside spline range. dist "
        << dist * boxsize << " Mpc/h [" << dist_low * boxsize << ", " << dist_high * boxsize << "]\n";
    }
  }
  return std::exp(dist_inv_spline(dist));
}

// Output the lightcone-particles as a gadget file
// NB: the velocities are not normalized as in "standard" gadget snapshot format (i.e. the sqrt(a))
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_gadget(std::vector<U> & part_lc, double astart, double aend) {

  const int nfiles = FML::NTasks;
  const std::string fileprefix = plc_folder + "/" + "lightcone_gadget_step" + std::to_string(plc_step) + ".";
  double pos_norm = 1.0;
  double vel_norm = 1.0;
  double scale_factor = (astart + aend)/2.0;

  if(FML::ThisTask == 0) std::cout << "# Writing lightcone to gadget file(s): " << fileprefix << "X\n";

  size_t npart_lc = part_lc.size();
  size_t npart_lc_global = npart_lc;
  FML::SumOverTasks(&npart_lc_global);

  FML::FILEUTILS::GADGET::GadgetWriter gw;
  gw.write_gadget_single(fileprefix + std::to_string(FML::ThisTask),
      &part_lc[0],
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

// Output the lightcone-particles as an ascii file
template <int NDIM, class T, class U>
void Lightcone<NDIM, T, U>::output_ascii(std::vector<U> & part_lc, double astart, double aend) {
  std::string filename = plc_folder + "/" + "lightcone_ascii_step" + std::to_string(plc_step) + ".";
  if(FML::ThisTask == 0)
    std::cout << "# Writing lightcone to ascii file(s): " << filename << "X\n";
  filename += std::to_string(FML::ThisTask);
  
  size_t npart = part_lc.size();
  size_t npart_total = npart;
  FML::SumOverTasks(&npart_total);
  
  std::ofstream fp(filename);
  fp << "# Lightcone particles from Task " << FML::ThisTask << " / " << FML::NTasks << "\n";
  fp << "# The lightcone crossing scale-factors for this sample is in [" << std::to_string(astart) << " -> " << std::to_string(aend) << "]\n";
  fp << "# The corresponding comoving distances is in [" << get_distance(aend) * boxsize << " -> " << get_distance(astart) * boxsize << "] Mpc/h\n";
  fp << "# This file has " << npart << " / " << npart_total << " particles\n";
  fp << "# The columns below are [pos_vec] (Mpc/h), [vel_vec] (km/s comoving)\n";
  for(auto & p : part_lc) {
    auto * pos = FML::PARTICLE::GetPos(p);
    auto * vel = FML::PARTICLE::GetVel(p);
    for(int idim = 0; idim < NDIM; idim++)
      fp << std::setw(15) << pos[idim] << "   ";
    for(int idim = 0; idim < NDIM; idim++)
      fp << std::setw(15) << vel[idim] << "   ";
    fp << "\n";
  }
}

// Volume of a spherical shell bounded by rlow and rhigh in NDIM dimension
template <int NDIM, class T, class U>
double Lightcone<NDIM, T, U>::volume_shell(double rlow, double rhigh) {
  if constexpr(NDIM == 3)
    return 4.0/3.0 * M_PI * (rhigh*rhigh*rhigh - rlow*rlow*rlow);
  else if(NDIM == 2)
    return M_PI * (rhigh*rhigh - rlow*rlow);
  else if(NDIM == 0)
    return rhigh - rlow;
  else
    throw std::runtime_error("Volume-shell only implemented for ndim = 1,2,3");
}

#endif
