#ifndef SIMCODE_SHORT_RANGE_FORCE_H
#define SIMCODE_SHORT_RANGE_FORCE_H
#include "ShortRangeSettings.h"
#include <optional>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ShortRangeGravity/LongRangePM.h>
#include <FML/ShortRangeGravity/Octree.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>
namespace SIMCODE {
    // This is an acceleration -grad Phi, unlike the PM mesh's +grad Phi. Apply with PLUS kick integral.
    template <class P>
    FML::SHORT_RANGE_GRAVITY::Result short_range_acceleration(FML::PARTICLE::MPIParticles<P> & particles,
                                                              const ShortRangeSettings & settings,
                                                              double boxsize,
                                                              double poisson_normalization) {
        namespace SR = FML::SHORT_RANGE_GRAVITY;
        P prototype{};
        int valid = FML::PARTICLE::has_get_id<P>() && FML::PARTICLE::GetNDIM(prototype) == 3;
        FML::MinOverTasks(&valid);
        if (!valid)
            throw std::invalid_argument("Short-range forces require stable globally unique particle IDs");
        std::vector<SR::Particle> owned(particles.get_npart());
        auto * p = particles.get_particles_ptr();
        double total_mass = 0;
        for (size_t i = 0; i < owned.size(); ++i) {
            auto & q = owned[i];
            auto * x = FML::PARTICLE::GetPos(p[i]);
            for (int d = 0; d < 3; ++d)
                q.position[d] = x[d];
            q.mass = FML::PARTICLE::GetMass(p[i]);
            q.id = std::uint64_t(FML::PARTICLE::GetID(p[i]));
            valid &= std::isfinite(q.mass) && q.mass >= 0;
            total_mass += q.mass;
        }
        FML::MinOverTasks(&valid);
        FML::SumOverTasks(&total_mass);
        if (!valid || !(total_mass > 0) || !std::isfinite(total_mass))
            throw std::invalid_argument("Invalid short-range source mass");
        std::optional<SR::Kernel> storage;
        try {
            const auto coupling = SR::fml_acceleration_coupling(poisson_normalization, total_mass, 1);
            storage.emplace(settings.options(boxsize, coupling));
        } catch (const std::exception &) {
            valid = 0;
        }
        FML::MinOverTasks(&valid);
        if (!valid)
            throw std::invalid_argument("Invalid short-range kernel settings on at least one rank");
        const auto & kernel = *storage;
        auto sources = SR::exchange_slab_ghosts(owned, FML::xmin_domain, FML::xmax_domain, kernel);
        if (settings.backend == "cell")
            return SR::CellList(sources.particles, kernel).compute(owned);
        SR::TreeOptions tree;
        tree.opening_angle = settings.opening_angle;
        return SR::Octree(sources.particles, kernel, tree).compute(owned);
    }
    template <class P>
    void kick_short_range(FML::PARTICLE::MPIParticles<P> & particles,
                          const ShortRangeSettings & settings,
                          double boxsize,
                          double poisson_normalization,
                          double dt) {
        if (!settings.enabled || dt == 0)
            return;
        auto force = short_range_acceleration(particles, settings, boxsize, poisson_normalization);
        auto * p = particles.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < particles.get_npart(); ++i) {
            auto * v = FML::PARTICLE::GetVel(p[i]);
            for (int d = 0; d < 3; ++d)
                v[d] += dt * force.acceleration[i][d];
        }
    }
    template <int D>
    void filter_short_range_long_force(FML::GRID::FFTWGrid<D> & density,
                                       const ShortRangeSettings & settings,
                                       double boxsize) {
        if (!settings.enabled)
            return;
        if constexpr (D == 3) {
            auto options = settings.options(boxsize);
            options.table_points = 0; // Filtering does not need a pair-force table.
            FML::SHORT_RANGE_GRAVITY::apply_long_range_filter(density, FML::SHORT_RANGE_GRAVITY::Kernel(options));
        } else
            throw std::invalid_argument("Short-range force integration requires NDIM=3");
    }
} // namespace SIMCODE
#endif
