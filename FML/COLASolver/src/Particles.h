#ifndef PARTICLES_HEADER
#define PARTICLES_HEADER

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

//=============================================================================
// The particle types the simulation can use. Main.cpp picks the smallest one
// that has what the simulation needs:
//
//   ParticleBasic         (pos, vel, id)               no COLA
//   ParticleCOLA          (+ 1LPT and 2LPT Psi)        scale-independent COLA
//   ParticleCOLAScaleDep  (+ Lagrangian position q)    scale-dependent COLA
//
// Free choice of types, so you can for example change to floats to save memory
// or add fields you want to carry along. See ExampleParticles.h for more examples
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

/// For scale-independent COLA we also need the LPT displacement fields
template <int NDIM>
class ParticleCOLA : public ParticleBasic<NDIM> {
  public:
    // 1LPT displacement field Psi (needed for >= 1LPT COLA)
    float Psi_1LPT[NDIM];
    float * get_D_1LPT() { return Psi_1LPT; }

    // 2LPT displacement field Psi (needed for >= 2LPT COLA)
    float Psi_2LPT[NDIM];
    float * get_D_2LPT() { return Psi_2LPT; }

    // 3LPT displacement fields (needed for 3LPT COLA)
    // float Psi_3LPTa[NDIM];
    // float Psi_3LPTb[NDIM];
    // float * get_D_3LPTa() { return Psi_3LPTa; }
    // float * get_D_3LPTb() { return Psi_3LPTb; }
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

#endif
