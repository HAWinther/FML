#ifndef GENERALPARTICLE_HEADER
#define GENERALPARTICLE_HEADER
#include <cstring>
#include <stdlib.h>

template <int NDIM,
          typename PosType,
          bool HASPOS,
          typename VelType,
          bool HASVEL,
          typename IDType,
          bool HASID,
          typename MassType,
          bool HASMASS,
          typename VolType,
          bool HASVOL>

struct GeneralParticle {
    static_assert(HASPOS or HASID or HASVEL or HASID or HASMASS or HASVOL, "Particle cannot be completely empty");

    // All the data in a single char array to simplify communicating the particle
    char data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM + sizeof(IDType) * HASID +
              sizeof(MassType) * HASMASS + sizeof(VolType) * HASVOL];

    char * get_data() { return data; }

    int get_ndim() { return NDIM; }

    PosType * get_pos() {
        if (!HASPOS)
            return nullptr;
        return (PosType *)data;
    }

    PosType * get_vel() {
        if (!HASVEL)
            return nullptr;
        return (VelType *)&data[sizeof(VelType) * HASPOS * NDIM];
    }

    IDType get_id() {
        if (!HASID)
            return -1;
        return *(IDType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM];
    }

    MassType get_mass() {
        if (!HASMASS)
            return 1.0;
        return *(MassType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM +
                                  sizeof(IDType) * HASID];
    }

    VolType get_vol() {
        if (!HASVOL)
            return 1.0;
        return *(VolType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM +
                                 sizeof(IDType) * HASID + sizeof(MassType) * HASMASS];
    }

    void set_pos(PosType * pos) {
        if (!HASPOS)
            return;
        std::memcpy(get_pos(), pos, sizeof(PosType) * HASPOS * NDIM);
    }

    void set_vel(VelType * vel) {
        if (!HASVEL)
            return;
        std::memcpy(get_vel(), vel, sizeof(VelType) * HASVEL * NDIM);
    }

    void set_id(IDType id) {
        if (!HASID)
            return;
        IDType * id_ptr = (IDType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM];
        std::memcpy(id_ptr, &id, sizeof(IDType) * HASID);
    }

    void set_mass(MassType mass) {
        if (!HASMASS)
            return;
        MassType * mass_ptr = (MassType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM +
                                                sizeof(IDType) * HASID];
        std::memcpy(mass_ptr, &mass, sizeof(MassType) * HASMASS);
    }

    void set_vol(VolType vol) {
        if (!HASVOL)
            return;
        VolType * vol_ptr = (VolType *)&data[sizeof(PosType) * HASPOS * NDIM + sizeof(VelType) * HASVEL * NDIM +
                                             sizeof(IDType) * HASID + sizeof(MassType) * HASMASS];
        std::memcpy(vol_ptr, &vol, sizeof(VolType) * HASVOL);
    }

    bool has_pos() { return HASPOS; }

    bool has_vel() { return HASVEL; }

    bool has_id() { return HASID; }

    bool has_mass() { return HASMASS; }

    bool has_vol() { return HASVOL; }

    int get_particle_byte_size() { return sizeof(data); }

    void append_to_buffer(char * buffer) { std::memcpy(buffer, data, sizeof(data)); }

    void assign_from_buffer(char * buffer) { std::memcpy(data, buffer, sizeof(data)); }
};

#endif
