#ifndef SIMCODE_BARYONIFICATION_OUTPUT_H
#define SIMCODE_BARYONIFICATION_OUTPUT_H
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <memory>
#include <string>
#include "SimulationParameters.h"

class Cosmology;
namespace SIMCODE {
    /// Minimal snapshot copy: true output velocities in code units, positions in box units, original IDs.
    struct BaryonificationParticle {
        double pos[3], vel[3];
        long long id;
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        long long get_id() const { return id; }
        constexpr int get_ndim() const { return 3; }
    };

    template <class P>
    FML::Vector<BaryonificationParticle> copy_baryonification_particles(P * particles, size_t count) {
        FML::Vector<BaryonificationParticle> copy(count);
        for (size_t i = 0; i < count; ++i) {
            for (int d = 0; d < 3; ++d) {
                copy[i].pos[d] = FML::PARTICLE::GetPos(particles[i])[d];
                copy[i].vel[d] = FML::PARTICLE::GetVel(particles[i])[d];
            }
            copy[i].id = FML::PARTICLE::GetID(particles[i]);
        }
        return copy;
    }

    // Non-template implementation: compiled once, independent of the simulation's LPT particle type.
    void output_baryonified_snapshot(FML::Vector<BaryonificationParticle> particles,
                                     const SimulationParameters & settings,
                                     const std::shared_ptr<Cosmology> & cosmo,
                                     double redshift,
                                     const std::string & snapshot_folder);
} // namespace SIMCODE
#endif
