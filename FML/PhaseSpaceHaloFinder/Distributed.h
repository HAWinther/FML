#ifndef FML_DISTRIBUTED_PHASE_SPACE_HALOS_HEADER
#define FML_DISTRIBUTED_PHASE_SPACE_HALOS_HEADER
#include <FML/PhaseSpaceHaloFinder/PhaseSpaceHaloFinder.h>
#include <FML/Global/PeriodicSlabExchange.h>
#include <string>
namespace FML::PHASE_SPACE_HALOS {
    namespace detail {
        inline void collective_require(bool valid, const std::string & message) {
            int failed = !valid;
            FML::MaxOverTasks(&failed);
            if (failed)
                throw std::runtime_error("[PhaseSpaceHalos] " + message);
        }
        inline int owner(ID id) { // Stable hash; lattice-aligned minimum IDs must not all select the same rank.
            id += 0x9e3779b97f4a7c15ULL;
            id = (id ^ (id >> 30)) * 0xbf58476d1ce4e5b9ULL;
            id = (id ^ (id >> 27)) * 0x94d049bb133111ebULL;
            return int((id ^ (id >> 31)) % std::uint64_t(FML::NTasks));
        }
        template <class Record>
        std::vector<Record> route(std::vector<std::vector<Record>> & outgoing, const Options & o) {
            std::vector<unsigned long long> counts(FML::NTasks), received(FML::NTasks);
            for (int t = 0; t < FML::NTasks; ++t)
                counts[t] = outgoing[t].size();
#ifdef USE_MPI
            MPI_Alltoall(counts.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         received.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
#else
            received = counts;
#endif
            std::size_t total = 0;
            bool fits = true;
            for (auto n : received) {
                if (n > o.max_local_particles - total) {
                    fits = false;
                    break;
                }
                total += std::size_t(n);
            }
            collective_require(fits, "Receive/workspace particle budget exceeded");
            std::vector<Record> result;
            try {
                result.reserve(total);
            } catch (const std::exception &) {
                fits = false;
            }
            collective_require(fits, "Unable to allocate receive workspace");
            FML::COMMUNICATION::exchange_records_chunked(
                outgoing,
                [&](const Record * data, std::size_t n) noexcept { result.insert(result.end(), data, data + n); },
                o.max_message_bytes);
            return result;
        }
        inline void agree_options(const Options & o) {
            const std::array<double, 13> values{o.boxsize,
                                                o.linking_length,
                                                o.retained_fraction,
                                                o.seed_significance,
                                                o.softening_support,
                                                o.velocity_floor,
                                                o.gravitational_constant,
                                                o.position_to_physical,
                                                o.hubble_velocity_gradient,
                                                o.assignment_density,
                                                o.potential_opening_angle,
                                                o.minimum_bound_fraction,
                                                o.minimum_peak_contrast};
            const std::array<std::uint64_t, 13> sizes{std::uint64_t(o.minimum_particles),
                                                      std::uint64_t(o.minimum_seed_particles),
                                                      std::uint64_t(o.max_parent_particles),
                                                      std::uint64_t(o.max_local_particles),
                                                      std::uint64_t(o.max_seeds),
                                                      o.max_neighbor_visits,
                                                      std::uint64_t(o.max_refinement_depth),
                                                      std::uint64_t(o.max_unbinding_iterations),
                                                      std::uint64_t(o.max_label_iterations),
                                                      std::uint64_t(o.max_message_bytes),
                                                      std::uint64_t(o.unbind),
                                                      std::uint64_t(o.saddle_samples),
                                                      std::uint64_t(o.vmax_inclusive)};
            auto copy = values;
            auto copies = FML::GatherFromTasks(&copy);
            auto integers = sizes;
            auto integer_copies = FML::GatherFromTasks(&integers);
            bool same = true;
            for (const auto & v : copies)
                same &= v == values;
            for (const auto & v : integer_copies)
                same &= v == sizes;
            collective_require(same, "Options differ between ranks");
        }
        struct Visible {
            Particle p;
            int owner{0};
            ID label{0};
        };
        struct Request {
            ID id{0};
            int rank{0};
        };
        struct Label {
            ID id{0}, label{0};
        };
        struct RoutedParticle {
            Particle p;
            ID fof{0};
            int origin{0};
        };
    } // namespace detail
    /// Collective on FML::communicator. Caller particles must occupy FML's periodic x slabs (unit-box globals).
    /// Particle positions themselves use Options::boxsize. No rank changes ownership of the caller's particles.
    inline Catalogue find_distributed(const std::vector<Particle> & particles, const Options & o = {}) {
        bool valid = true;
        try {
            o.validate();
            detail::validate_particles(particles, o);
        } catch (const std::exception &) {
            valid = false;
        }
        detail::collective_require(valid, "Invalid options or particles on at least one rank");
        detail::agree_options(o);
        const std::array<double, 2> slab{FML::xmin_domain, FML::xmax_domain};
        auto slab_copy = slab;
        const auto slabs = FML::GatherFromTasks(&slab_copy);
        valid = slab[0] >= 0 && slab[1] <= 1 && slab[0] <= slab[1];
        for (int rank = 0; rank < FML::NTasks; ++rank)
            valid &=
                slabs[rank][0] == (rank ? slabs[rank - 1][1] : 0) && (rank != FML::NTasks - 1 || slabs[rank][1] == 1);
        for (const auto & p : particles)
            valid &= p.position[0] / o.boxsize >= slab[0] && p.position[0] / o.boxsize < slab[1];
        detail::collective_require(valid, "Slabs must partition [0,1) and particles must already have migrated");
        std::vector<std::vector<ID>> ids(FML::NTasks);
        for (const auto & p : particles)
            ids[detail::owner(p.id)].push_back(p.id);
        auto checked = detail::route(ids, o);
        std::sort(checked.begin(), checked.end());
        detail::collective_require(std::adjacent_find(checked.begin(), checked.end()) == checked.end(),
                                   "Particle IDs must be globally unique");
        std::uint64_t total = particles.size();
        FML::SumOverTasks(&total);
        if (!total) {
            Catalogue empty;
            empty.options = o;
            return empty;
        }
        double mass = 0;
        for (const auto & p : particles)
            mass += p.mass;
        FML::SumOverTasks(&mass);
        const double density = o.assignment_density > 0 ? o.assignment_density : 200 * mass / std::pow(o.boxsize, 3);
        detail::collective_require(std::isfinite(density) && density > 0, "Invalid global mass density");
        const double radius = o.linking_length * o.boxsize / std::cbrt(double(total));
        detail::collective_require(radius < .5 * o.boxsize, "Linking radius exceeds half the box");
        auto owned = particles;
        std::sort(owned.begin(), owned.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
        std::vector<detail::Visible> visible;
        for (const auto & p : owned)
            visible.push_back({p, FML::ThisTask, p.id});
        FML::COMMUNICATION::append_periodic_slab_ghosts(
            visible,
            radius / o.boxsize,
            [&](const auto & x) { return x.p.position[0] / o.boxsize; },
            o.max_message_bytes);
        detail::collective_require(visible.size() <= o.max_local_particles,
                                   "Owned plus ghost particle budget exceeded");
        FML::KDTREE::pointVec positions;
        std::map<ID, std::size_t> location;
        for (std::size_t i = 0; i < visible.size(); ++i) {
            const auto & p = visible[i].p;
            positions.push_back({p.position[0], p.position[1], p.position[2]});
            location[p.id] = i;
        }
        FML::KDTREE::KDTree tree(positions, o.boxsize);
        detail::DisjointSet connected(visible.size());
        std::uint64_t visits = 0;
        valid = true;
        try {
            for (std::size_t i = 0; i < owned.size(); ++i) {
                const auto neighbors = tree.neighborhood_indices(positions[i], radius);
                detail::count_visits(visits, neighbors.size(), o);
                for (auto j : neighbors)
                    connected.join(i, j);
            }
        } catch (const std::exception &) {
            valid = false;
        }
        detail::collective_require(valid, "Spatial neighbour-work budget exceeded");
        // Subscribe once to the owners of visible ghosts. Only ID/label metadata crosses ranks during convergence.
        std::vector<std::vector<detail::Request>> queries(FML::NTasks);
        for (std::size_t i = owned.size(); i < visible.size(); ++i)
            queries[visible[i].owner].push_back({visible[i].p.id, FML::ThisTask});
        auto requested = detail::route(queries, o);
        std::vector<std::vector<std::size_t>> subscribers(FML::NTasks);
        for (const auto & request : requested)
            subscribers[request.rank].push_back(location.at(request.id));
        bool converged = false;
        std::uint64_t iterations = 0;
        for (int iteration = 0; iteration < o.max_label_iterations; ++iteration) {
            std::vector<ID> minimum(visible.size(), no_halo);
            for (std::size_t i = 0; i < visible.size(); ++i) {
                const auto root = connected.find(i);
                minimum[root] = std::min(minimum[root], visible[i].label);
            }
            int changed = 0;
            for (std::size_t i = 0; i < owned.size(); ++i) {
                const auto next = minimum[connected.find(i)];
                changed |= next != visible[i].label;
                visible[i].label = next;
            }
            std::vector<std::vector<detail::Label>> outgoing(FML::NTasks);
            for (int rank = 0; rank < FML::NTasks; ++rank)
                for (auto i : subscribers[rank])
                    outgoing[rank].push_back({visible[i].p.id, visible[i].label});
            const auto updated = detail::route(outgoing, o);
            for (const auto & label : updated) {
                auto & record = visible[location.at(label.id)];
                changed |= record.label != label.label;
                record.label = label.label;
            }
            FML::MaxOverTasks(&changed);
            ++iterations;
            if (!changed) {
                converged = true;
                break;
            }
        }
        detail::collective_require(converged,
                                   "Distributed FoF labels failed to converge; increase max_label_iterations");
        std::vector<std::vector<detail::RoutedParticle>> outgoing(FML::NTasks);
        for (std::size_t i = 0; i < owned.size(); ++i)
            outgoing[detail::owner(visible[i].label)].push_back({owned[i], visible[i].label, FML::ThisTask});
        auto received = detail::route(outgoing, o);
        // Parent analysis uses ID order, independent of arrival order and the MPI decomposition.
        std::sort(received.begin(), received.end(), [](const auto & a, const auto & b) { return a.p.id < b.p.id; });
        std::vector<Particle> complete;
        std::vector<ID> labels;
        std::map<ID, int> origins;
        for (const auto & record : received) {
            complete.push_back(record.p);
            labels.push_back(record.fof);
            origins[record.p.id] = record.origin;
        }
        Catalogue result;
        valid = true;
        std::string error;
        try {
            result = detail::analyze(std::move(complete), labels, o, density);
        } catch (const std::exception & e) {
            valid = false;
            error = e.what();
        }
        detail::collective_require(valid,
                                   "Parent analysis failed on at least one rank" +
                                       (error.empty() ? std::string{} : std::string(": ") + error));
        std::vector<std::vector<Assignment>> answers(FML::NTasks);
        for (const auto & assignment : result.assignments)
            answers[origins.at(assignment.particle)].push_back(assignment);
        result.assignments = detail::route(answers, o);
        std::sort(result.assignments.begin(), result.assignments.end(), [](const auto & a, const auto & b) {
            return a.particle < b.particle;
        });
        valid = result.assignments.size() == owned.size();
        if (valid)
            for (std::size_t i = 0; i < owned.size(); ++i)
                valid &= result.assignments[i].particle == owned[i].id;
        detail::collective_require(valid, "Missing or duplicate returned particle assignment");
        result.spatial_linking_radius = radius;
        result.statistics.label_iterations = iterations;
        result.statistics.ghost_particles = visible.size() - owned.size();
        result.statistics.routed_particles = received.size();
        return result;
    }
} // namespace FML::PHASE_SPACE_HALOS
#endif
