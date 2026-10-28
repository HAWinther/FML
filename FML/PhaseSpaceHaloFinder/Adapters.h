#ifndef FML_PHASE_SPACE_HALO_ADAPTERS_HEADER
#define FML_PHASE_SPACE_HALO_ADAPTERS_HEADER
#include <FML/PhaseSpaceHaloFinder/Distributed.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/HOD/HOD.h>
#include <FML/HaloHistory/HaloHistory.h>
namespace FML::PHASE_SPACE_HALOS {
    /// Scale FML particle fields explicitly; IDs must already be globally stable. No velocity convention is inferred.
    template <class T>
    std::vector<Particle>
    import_particles(T * p, std::size_t n, double position_scale, double velocity_scale, double mass_scale = 1) {
        static_assert(FML::PARTICLE::has_get_pos<T>() && FML::PARTICLE::has_get_vel<T>() &&
                          FML::PARTICLE::has_get_id<T>() && FML::PARTICLE::has_get_ndim<T>(),
                      "Phase-space halo finding requires positions, velocities and stable IDs");
        for (double s : {position_scale, velocity_scale, mass_scale})
            if (!std::isfinite(s) || s <= 0)
                throw std::invalid_argument("Invalid particle import scale");
        std::vector<Particle> result;
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (FML::PARTICLE::GetNDIM(p[i]) != 3)
                throw std::invalid_argument("Phase-space halo finding requires 3D particles");
            Particle x;
            const auto id = FML::PARTICLE::GetID(p[i]);
            if constexpr (std::is_signed_v<decltype(id)>)
                if (id < 0)
                    throw std::invalid_argument("Negative particle ID");
            x.id = ID(id);
            x.mass = FML::PARTICLE::GetMass(p[i]) * mass_scale;
            for (int d = 0; d < 3; ++d) {
                x.position[d] = FML::PARTICLE::GetPos(p[i])[d] * position_scale;
                x.velocity[d] = FML::PARTICLE::GetVel(p[i])[d] * velocity_scale;
            }
            result.push_back(x);
        }
        return result;
    }
    struct SOSeed {
        ID id;
        Vector position;
        double initial_radius;
    };
    /// Guesses for the existing SO routine: positions and radii in unit-box coordinates, no fabricated SO mass.
    inline std::vector<SOSeed> so_seeds(const Catalogue & c, double boxsize) {
        if (!std::isfinite(boxsize) || boxsize <= 0)
            throw std::invalid_argument("Invalid SO boxsize");
        std::vector<SOSeed> result;
        for (const auto & h : c.halos) {
            Vector x = h.position;
            for (auto & v : x)
                v /= boxsize;
            result.push_back({h.id, x, h.bound_radius / boxsize});
        }
        return result;
    }
    /// Collective: route SO guesses from parent-owner ranks to the spatial owner required by the existing SO API.
    inline std::vector<SOSeed> so_seeds_by_slab(const Catalogue & c, const Options & o) {
        bool valid = true;
        std::vector<SOSeed> seeds;
        try {
            o.validate();
            seeds = so_seeds(c, o.boxsize);
        } catch (const std::exception &) {
            valid = false;
        }
        detail::collective_require(valid, "Invalid SO seed options");
        detail::agree_options(o);
        std::array<double, 2> local{FML::xmin_domain, FML::xmax_domain};
        const auto slabs = FML::GatherFromTasks(&local);
        std::vector<std::vector<SOSeed>> outgoing(FML::NTasks);
        for (const auto & seed : seeds) {
            int owner = -1;
            for (int t = 0; t < FML::NTasks; ++t)
                if (seed.position[0] >= slabs[t][0] && seed.position[0] < slabs[t][1])
                    owner = t;
            if (owner < 0)
                valid = false;
            else
                outgoing[owner].push_back(seed);
        }
        detail::collective_require(valid, "SO centre has no spatial owner");
        return detail::route(outgoing, o);
    }
    /// Supply the intended HOD mass/radius explicitly, e.g. externally measured M200m/R200m in the same units.
    inline FML::HOD::Halo hod_halo(const Halo & h, double mass, double radius, double concentration) {
        if (!std::isfinite(mass) || mass <= 0 || !std::isfinite(radius) || radius <= 0 ||
            !std::isfinite(concentration) || concentration <= 0)
            throw std::invalid_argument("Invalid HOD mass/radius/concentration");
        FML::HOD::Halo out;
        out.id = h.id;
        out.position = h.position;
        out.velocity = h.velocity;
        out.mass = mass;
        out.radius = radius;
        out.concentration = concentration;
        out.velocity_dispersion = std::sqrt(detail::dot(h.velocity_dispersion, h.velocity_dispersion) / 3);
        return out;
    }
    /// Exclusive particle-ID catalogue for the separate history module. Units and mass definition remain explicit.
    inline FML::HALO_HISTORY::Snapshot history_snapshot(const Catalogue & c,
                                                        ID snapshot_id,
                                                        double scale_factor,
                                                        double boxsize,
                                                        FML::HALO_HISTORY::Metadata metadata) {
        FML::HALO_HISTORY::Snapshot s;
        s.id = snapshot_id;
        s.scale_factor = scale_factor;
        s.boxsize = boxsize;
        s.metadata = std::move(metadata);
        s.membership = FML::HALO_HISTORY::Membership::Exclusive;
        for (const auto & h : c.halos) {
            FML::HALO_HISTORY::Halo x;
            x.id = h.id;
            if (h.host != no_halo)
                x.host = h.host;
            x.position = h.position;
            x.velocity = h.velocity;
            x.mass = h.bound_mass;
            x.radius = h.bound_radius;
            x.vmax = h.vmax;
            x.particles = h.particle_ids;
            s.halos.push_back(std::move(x));
        }
        FML::HALO_HISTORY::validate(s);
        return s;
    }
} // namespace FML::PHASE_SPACE_HALOS
#endif
