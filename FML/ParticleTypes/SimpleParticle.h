#ifndef SIMPLEPARTICLE_HEADER
#define SIMPLEPARTICLE_HEADER
#include <cstring>

/// Example of a simple particle with all the methods needed to be used with MPIParticles
/// Some algorithms might require more field like get_vol, id etc.
/// The types used below can be changed (automatically inferred by algorithms)
template <int NDIM>
struct SimpleParticle {

    /// Position of particle in [0,1)^NDIM
    double Pos[NDIM];
    /// Velocity of particle (in whatever units you want)
    double Vel[NDIM];

    /// Get the dimension of the position
    constexpr int get_ndim() const { return NDIM; }
    /// Get a pointer to the position of the particle
    double * get_pos() { return Pos; }
    /// Get a pointer to the velocity of the particle
    double * get_vel() { return Vel; }
};

/// Example of a particle for COLA N-body simulations
/// that is compatible with the make IC and N-body methods
/// If you only want 1LPT then simply remove all 2LPT stuff below
/// The types used below can be changed (automatically inferred by algorithms)
template <int NDIM>
struct COLAParticle {

    /// Position
    double Pos[NDIM];
    /// Lagrangian Position
    double q[NDIM];
    /// Velocity
    double Vel[NDIM];
    /// 1LPT displacement vector
    double D_1LPT[NDIM];
    /// 2LPT displacement vector
    double D_2LPT[NDIM];
    /// 3LPTa displacement vector
    double D_3LPTa[NDIM];
    /// 3LPTb displacement vector
    double D_3LPTb[NDIM];
    /// ID of the particle
    long long int id;

    /// Get the ID of the particle
    long long int get_id() const { return id; }
    /// Set the ID of the particle
    void set_id(long long int _id) { id = _id; }
    /// Get the dimension of the position
    constexpr int get_ndim() const { return NDIM; }
    /// Get a pointer to the position of the particle
    double * get_pos() { return Pos; }
    /// Get a pointer to the velocity of the particle
    double * get_vel() { return Vel; }
    /// Get a pointer to the Lagrangian position of the particle
    double * get_q() { return q; }
    /// Get a pointer to the initial 1LPT displacement field at the particles initial position
    double * get_D_1LPT() { return D_1LPT; }
    /// Get a pointer to the initial 2LPT displacement field at the particles initial position
    double * get_D_2LPT() { return D_2LPT; }
    /// Get a pointer to the initial 3LPTa displacement field at the particles initial position
    double * get_D_3LPTa() { return D_3LPTa; }
    /// Get a pointer to the initial 3LPTb displacement field at the particles initial position
    double * get_D_3LPTb() { return D_3LPTb; }
};

#endif
