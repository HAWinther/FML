#ifndef FML_ANALYSIS_PARTICLEUTILS_H
#define FML_ANALYSIS_PARTICLEUTILS_H

#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace FML {
    namespace ANALYSIS {

        /// Send the particles to the task whose domain (slabs: [xmin_domain, xmax_domain)) holds their position after
        /// they have moved (positions in [0,1)), growing the storage first so that the receiving task has room for
        /// all its particles: MPIParticles::communicate_particles only uses the storage it has. Collective
        template <class P>
        void redistribute_particles(FML::PARTICLE::MPIParticles<P> & particles) {
            if (FML::NTasks == 1)
                return;
            const auto xmin = FML::GatherFromTasks(&FML::xmin_domain);
            const auto & domain = particles.get_domain();
            std::vector<size_t> counts(size_t(FML::NTasks), 0);
            for (size_t i = 0; i < particles.get_npart(); ++i) {
                const auto * pos = FML::PARTICLE::GetPos(particles[i]);
                // Slabs: the x-edges of the tasks; other domains (pencils): the owner
                const int rank = domain.is_slab() ?
                                     int(std::upper_bound(xmin.begin(), xmin.end(), double(pos[0])) - xmin.begin()) - 1 :
                                     domain.owner(pos);
                ++counts[size_t(std::clamp(rank, 0, FML::NTasks - 1))];
            }
            FML::SumArrayOverTasks(counts.data(), FML::NTasks);
            auto & storage = particles.get_particles();
            // The particles staying plus those arriving (= counts) must fit; a little slack for later moves
            const size_t needed = counts[size_t(FML::ThisTask)];
            const size_t want = needed + needed / 20 + 16;
            if (storage.size() < want)
                storage.resize(want);
            particles.communicate_particles();
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
