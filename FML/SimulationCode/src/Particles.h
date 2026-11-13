#ifndef PARTICLES_HEADER
#define PARTICLES_HEADER

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

//=============================================================================
// The particle types the simulation can use. By default the simulation is compiled for one type, ParticleFiducial
// (all the fields below: 120 bytes in 3D), which can run every configuration. With USE_MINIMAL_PARTICLES (make
// USE_MINIMAL_PARTICLES=true) it is compiled for all of them and Main.cpp picks the smallest one that has what the
// simulation needs (less memory, 7 times the compile time):
//
//   ParticleBasic         (pos, vel, id)               no COLA
//   ParticleCOLA          (+ 1LPT and 2LPT Psi)        scale-independent COLA
//   ParticleCOLAScaleDep  (+ Lagrangian position q)    scale-dependent COLA
//   ParticleBasicQ        (pos, vel, id, q)            no COLA, but q is needed
//                                                      (force_density_method = sheet)
//   ParticleWithMass<P>   (P + a mass and volume)      particle_baryons with
//                                                      particle_baryon_relative_mode = mass or mass_ini;
//                                                      cosmology_model = IDE
//
// Free choice of types, so you can for example change to floats to save memory
// or add fields you want to carry along. See FML/ParticleTypes for more examples
// and FML/ParticleTypes/ReflectOnParticleMethods.h for the standard methods
//=============================================================================

/// Position, velocity and ID: all that is needed for a standard PM simulation
template <int NDIM>
class ParticleBasic {
  public:
    double pos[NDIM];
    double * get_pos() { return pos; }
    double vel[NDIM];
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }

    // ID of the particle (ok to skip this, but if so it will not be in the GADGET output files)
    long long int id;
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
};

/// A standard PM particle that also stores its Lagrangian position q (for sheet interpolation)
template <int NDIM>
class ParticleBasicQ : public ParticleBasic<NDIM> {
  public:
    double q[NDIM];
    double * get_q() { return q; }
};

/// For scale-independent COLA we also need the LPT displacement fields
template <int NDIM>
class ParticleCOLA : public ParticleBasic<NDIM> {
  public:
    // 1LPT displacement field Psi (needed for >= 1LPT COLA)
    float Psi_1LPT[NDIM];
    float * get_D_1LPT() { return Psi_1LPT; }

    // 2LPT displacement field Psi (needed for 2LPT COLA)
    float Psi_2LPT[NDIM];
    float * get_D_2LPT() { return Psi_2LPT; }
};

/// Massive neutrino particles (particle_neutrinos = true, delta-f method of Elbers et al. 2021). The velocity is the
/// comoving momentum p / (m H0 Box) (= the usual code velocity a^2 dx/dt / (H0 Box) for a non-relativistic particle).
/// The phase-space density the particle carries is f_i = f0(y_ini) (1 + delta_ini) where f0(y) = 1/(e^y + 1) is the
/// background Fermi-Dirac distribution, y = p c / (k_B T_nu0) the (comoving) momentum in units of the neutrino
/// temperature today and delta_ini the linear neutrino density at the particle in the initial conditions. The
/// weight is w = 1 - f0(y) / f_i and the density assignment then gives the neutrino density perturbation
template <int NDIM>
class ParticleNeutrino {
  public:
    double pos[NDIM];
    double * get_pos() { return pos; }
    double vel[NDIM];
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    long long int id;
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }

    float y_ini;     // momentum in units of k_B T_nu0 / c of the (unperturbed) thermal momentum at the start
    float delta_ini; // linear delta_nu at the particle at the start
    float weight;    // the delta-f weight w = 1 - f0(y) / (f0(y_ini) (1 + delta_ini)) (1 without delta-f)
    double get_deltaf_weight() const { return weight; }
};

/// For scale-dependent COLA we also need the initial Lagrangian position
template <int NDIM>
class ParticleCOLAScaleDep : public ParticleCOLA<NDIM> {
  public:
    // NB: should ideally have same type as [pos] to avoid truncating the
    // precision of pos (these are temporarily swapped by some algorithms)
    double q[NDIM];
    double * get_q() { return q; }
};

/// A particle P with a mass (in units of the mean particle mass of its species). Used for the mass-encoded
/// baryon-CDM relative mode (particle_baryon_relative_mode = mass or mass_ini): the CDM and baryon particles move
/// together with the total CDM+baryon flow and the relative density is carried by their masses m = 1 + eps(q). The
/// density assignment weights the particles by m / <m>. And for interacting dark energy (cosmology_model = IDE), where
/// the CDM particles gain or lose mass with the change of their volume: volume is then the comoving volume of the
/// particle relative to the mean, 1 / (1 + delta_n) with delta_n the number density at the particle when the masses
/// were last updated (Simulation::update_ide_masses)
template <class P>
class ParticleWithMass : public P {
  public:
    double mass{1.0};   // (1 unless set: the IC with a mass-encoded relative mode, or read from a file, or IDE)
    double volume{0.0}; // IDE: the volume at the last mass update (0: not set yet)
    double get_mass() const { return mass; }
    void set_mass(double _mass) { mass = _mass; }
};

/// The particle type that can do everything (COLA of all kinds, q for the sheet density, a mass for the
/// mass-encoded relative mode): the one the simulation is compiled for without USE_MINIMAL_PARTICLES
template <int NDIM>
using ParticleFiducial = ParticleWithMass<ParticleCOLAScaleDep<NDIM>>;

#endif
