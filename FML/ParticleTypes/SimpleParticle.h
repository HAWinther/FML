#ifndef SIMPLEPARTICLE_HEADER
#define SIMPLEPARTICLE_HEADER
#include <cstring>

/// Simple particle with all the methods needed to be used with MPIParticles
/// Some algorithms might require more field like get_vol, id etc.
template <int NDIM>
struct SimpleParticle {

    double Pos[NDIM];
    double Vel[NDIM];

    int get_ndim() { return NDIM; }
    double * get_pos() { return Pos; }
    double * get_vel() { return Vel; }
};

/// A particle for COLA N-body simulations
/// that is compatible with the make IC and N-body methods
/// If you only want 1LPT then simply remove all 2LPT stuff below
template <int NDIM>
struct COLAParticle {

    double Pos[NDIM];
    double q[NDIM];
    double Vel[NDIM];
    double D_1LPT[NDIM];
    double D_2LPT[NDIM];
    double D_3LPTa[NDIM];
    double D_3LPTb[NDIM];

    long long int id;
    long long int get_id() { return id; }
    void set_id(long long int _id) { id = _id; }

    constexpr int get_ndim() { return NDIM; }
    double * get_pos() { return Pos; }
    double * get_vel() { return Vel; }
    double * get_q() { return q; }
    double * get_D_1LPT() { return D_1LPT; }
    double * get_D_2LPT() { return D_2LPT; }
    double * get_D_3LPTa() { return D_3LPTa; }
    double * get_D_3LPTb() { return D_3LPTb; }
};

#endif
