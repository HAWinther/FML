#ifndef FML_SHORT_RANGE_SLAB_GHOSTS_HEADER
#define FML_SHORT_RANGE_SLAB_GHOSTS_HEADER
#include <FML/Global/Global.h>
#include <FML/ShortRangeGravity/ShortRangeGravity.h>
#include <cstring>

namespace FML::SHORT_RANGE_GRAVITY {
    struct GhostSources {
        std::vector<Particle> particles; // Owned first, then each remote source once. Positions remain in the base box.
        std::uint64_t sent_particles{0}, received_particles{0};
    };
    namespace detail {
        inline double distance_to_periodic_slab(double x, double lo, double hi, double box) {
            double distance = box;
            for (int image = -1; image <= 1; ++image) {
                const double y = x + image * box;
                distance = std::min(distance, std::max({lo - y, 0.0, y - hi}));
            }
            return distance;
        }
    } // namespace detail
    /// Collective on FML::communicator. Slabs in rank order must partition [0,L); IDs must be globally unique.
    /// Exchanges only cutoff-relevant sources, including nonadjacent ranks; never all-gathers particle payloads.
    /// Call after ownership migration and rebuild after every drift. Identical kernel options required on every rank.
    inline GhostSources
    exchange_slab_ghosts(const std::vector<Particle> & owned, double xmin, double xmax, const Kernel & kernel) {
        int invalid = 0;
        try {
            detail::validate_particles(owned, kernel.options().boxsize, true);
            if (!std::isfinite(xmin) || !std::isfinite(xmax) || xmin < 0 || xmax > kernel.options().boxsize ||
                xmin >= xmax)
                invalid = 1;
            for (const auto & p : owned)
                if (p.position[0] < xmin || p.position[0] >= xmax)
                    invalid = 1;
        } catch (const std::invalid_argument &) {
            invalid = 1;
        }
        FML::MaxOverTasks(&invalid);
        if (invalid)
            throw std::invalid_argument("[ShortRangeGravity] Invalid owned particles or slab on at least one rank");
        GhostSources result;
        result.particles = owned;
#ifdef USE_MPI
        const auto & o = kernel.options();
        const auto model = kernel.model_parameters();
        constexpr int nsettings = 10;
        const std::array<double, nsettings> settings{xmin,
                                                     xmax,
                                                     o.boxsize,
                                                     o.split_scale,
                                                     o.softening_support,
                                                     o.cutoff_tolerance,
                                                     o.coupling,
                                                     model[0],
                                                     model[1],
                                                     model[2]};
        std::vector<double> all_settings(std::size_t(FML::NTasks) * settings.size());
        MPI_Allgather(settings.data(),
                      nsettings,
                      MPI_DOUBLE,
                      all_settings.data(),
                      nsettings,
                      MPI_DOUBLE,
                      FML::communicator);
        for (int rank = 0; rank < FML::NTasks; ++rank) {
            const auto base = std::size_t(rank) * nsettings;
            if (all_settings[base] != (rank ? all_settings[base - (nsettings - 1)] : 0) ||
                (rank == FML::NTasks - 1 && all_settings[base + 1] != o.boxsize))
                invalid = 1;
            for (int j = 2; j < nsettings; ++j)
                if (all_settings[base + j] != settings[j])
                    invalid = 1;
        }
        FML::MaxOverTasks(&invalid);
        if (invalid)
            throw std::invalid_argument("[ShortRangeGravity] Slabs must partition the box and kernels must agree");
        const auto needed = [&](const Particle & p, int rank) {
            const auto base = std::size_t(rank) * nsettings;
            return detail::distance_to_periodic_slab(p.position[0],
                                                     all_settings[base],
                                                     all_settings[base + 1],
                                                     o.boxsize) < kernel.cutoff();
        };
        std::vector<std::uint64_t> send_counts(FML::NTasks), recv_counts(FML::NTasks);
        for (int rank = 0; rank < FML::NTasks; ++rank)
            if (rank != FML::ThisTask)
                for (const auto & p : owned)
                    send_counts[rank] += needed(p, rank);
        MPI_Alltoall(send_counts.data(), 1, MPI_UINT64_T, recv_counts.data(), 1, MPI_UINT64_T, FML::communicator);
        // Send one destination at a time. Fixed-size packed records avoid padding, datatype extents and int overflow.
        constexpr std::size_t record_bytes = 4 * sizeof(double) + sizeof(std::uint64_t);
        constexpr std::size_t batch = 16384;
        std::vector<char> send_buffer(batch * record_bytes), recv_buffer(batch * record_bytes);
        for (int offset = 1; offset < FML::NTasks; ++offset) {
            const int dest = (FML::ThisTask + offset) % FML::NTasks;
            const int source = (FML::ThisTask - offset + FML::NTasks) % FML::NTasks;
            std::uint64_t rounds = std::max(send_counts[dest], recv_counts[source]) / batch +
                                   (std::max(send_counts[dest], recv_counts[source]) % batch != 0);
            MPI_Allreduce(MPI_IN_PLACE, &rounds, 1, MPI_UINT64_T, MPI_MAX, FML::communicator);
            std::size_t cursor = 0;
            std::uint64_t sent = 0, received = 0;
            for (std::uint64_t round = 0; round < rounds; ++round) {
                const auto ns = std::size_t(std::min<std::uint64_t>(batch, send_counts[dest] - sent));
                const auto nr = std::size_t(std::min<std::uint64_t>(batch, recv_counts[source] - received));
                std::size_t packed = 0;
                while (packed < ns) {
                    const auto & p = owned[cursor++];
                    if (!needed(p, dest))
                        continue;
                    char * bytes = send_buffer.data() + packed++ * record_bytes;
                    std::memcpy(bytes, p.position.data(), 3 * sizeof(double));
                    std::memcpy(bytes + 3 * sizeof(double), &p.mass, sizeof(double));
                    std::memcpy(bytes + 4 * sizeof(double), &p.id, sizeof(p.id));
                }
                MPI_Sendrecv(send_buffer.data(),
                             int(ns * record_bytes),
                             MPI_BYTE,
                             dest,
                             0,
                             recv_buffer.data(),
                             int(nr * record_bytes),
                             MPI_BYTE,
                             source,
                             0,
                             FML::communicator,
                             MPI_STATUS_IGNORE);
                for (std::size_t i = 0; i < nr; ++i) {
                    Particle p;
                    const char * bytes = recv_buffer.data() + i * record_bytes;
                    std::memcpy(p.position.data(), bytes, 3 * sizeof(double));
                    std::memcpy(&p.mass, bytes + 3 * sizeof(double), sizeof(double));
                    std::memcpy(&p.id, bytes + 4 * sizeof(double), sizeof(p.id));
                    result.particles.push_back(p);
                }
                sent += ns;
                received += nr;
            }
            result.sent_particles += sent;
            result.received_particles += received;
        }
#else
        if (xmin != 0 || xmax != kernel.options().boxsize)
            throw std::invalid_argument("[ShortRangeGravity] Serial slab must cover the whole box");
#endif
        return result;
    }
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
