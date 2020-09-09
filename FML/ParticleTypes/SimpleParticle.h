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

    double get_mass() { return 1.0; }

    // Methods needed for commmunication of particle
    int get_particle_byte_size() { return 2 * sizeof(double) * NDIM; }

    void append_to_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Pos, bytes);
        buffer += bytes;
        bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Vel, bytes);
        buffer += bytes;
    }

    void assign_from_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(Pos, buffer, bytes);
        buffer += bytes;
        bytes = sizeof(double) * NDIM;
        std::memcpy(Vel, buffer, bytes);
        buffer += bytes;
    }
};

/// A particle for COLA N-body simulations
template <int NDIM>
struct COLAParticle {

    double Pos[NDIM];
    double Vel[NDIM];
    double D[NDIM];

    int get_ndim() { return NDIM; }

    double * get_pos() { return Pos; }

    double * get_vel() { return Vel; }
    
    double * get_D() { return D; }

    double get_mass() { return 1.0; }

    // Methods needed for commmunication of particle
    int get_particle_byte_size() { return 3 * sizeof(double) * NDIM; }

    void append_to_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Pos, bytes);
        buffer += bytes;
        bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Vel, bytes);
        buffer += bytes;
        std::memcpy(buffer, D, bytes);
    }

    void assign_from_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(Pos, buffer, bytes);
        buffer += bytes;
        bytes = sizeof(double) * NDIM;
        std::memcpy(Vel, buffer, bytes);
        buffer += bytes;
        std::memcpy(D, buffer, bytes);
    }
};

/// A simple particle for LPT stuff with all the methods needed to be used with MPIParticles
/// This stores both the Eulerian position x and the Lagrangian position q 
/// By default x is the position returned by pos, but can be changed with the set_* methods
/// (The position_is_eulerian flag should be global and not stored with each particle, but fuck it)
template <int NDIM>
struct LPTParticle {
    double Pos_x[NDIM];
    double Pos_q[NDIM];
    double Vel[NDIM];
    bool position_is_eulerian{true};

    int get_ndim() { return NDIM; }

    double * get_pos() {
        if (position_is_eulerian)
            return Pos_x;
        else
            return Pos_q;
    }

    double * get_vel() { return Vel; }

    double get_mass() { return 1.0; }

    void set_eulerian_position(){ position_is_eulerian = true; }
    void set_lagrangian_position(){ position_is_eulerian = false; }

    // Methods needed for commmunication of particle
    int get_particle_byte_size() { return 3 * sizeof(double) * NDIM + sizeof(bool); }

    void append_to_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(buffer, Pos_x, bytes);
        buffer += bytes;
        std::memcpy(buffer, Pos_q, bytes);
        buffer += bytes;
        std::memcpy(buffer, Vel, bytes);
        buffer += bytes;
        std::memcpy(buffer, &position_is_eulerian, sizeof(bool));
    }

    void assign_from_buffer(char * buffer) {
        int bytes = sizeof(double) * NDIM;
        std::memcpy(Pos_x, buffer, bytes);
        buffer += bytes;
        std::memcpy(Pos_q, buffer, bytes);
        buffer += bytes;
        std::memcpy(Vel, buffer, bytes);
        buffer += bytes;
        std::memcpy(&position_is_eulerian, buffer, sizeof(bool));
    }
};

#endif
