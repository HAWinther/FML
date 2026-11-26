#ifndef FML_SHORT_RANGE_SLAB_GHOSTS_HEADER
#define FML_SHORT_RANGE_SLAB_GHOSTS_HEADER
#include <FML/Global/ChunkedExchange.h>
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
        // The region of a rank: [lo, hi) along x and y (all of z) in box units
        struct Box {
            double lo[2], hi[2];
        };
        // The periodic distance from a point to a box (the x-distance for a slab: no rounding from the y part)
        inline double distance_to_periodic_box(const double * pos, const Box & b, double box) {
            const double dx = distance_to_periodic_slab(pos[0], b.lo[0], b.hi[0], box);
            if (b.lo[1] == 0.0 and b.hi[1] == box)
                return dx;
            const double dy = distance_to_periodic_slab(pos[1], b.lo[1], b.hi[1], box);
            return std::hypot(dx, dy);
        }
        // The periodic gap between two intervals
        inline double periodic_gap(double lo1, double hi1, double lo2, double hi2, double box) {
            double gap = box;
            for (int image = -1; image <= 1; ++image)
                gap = std::min(gap, std::max({lo1 + image * box - hi2, lo2 - (hi1 + image * box), 0.0}));
            return gap;
        }
        inline GhostSources exchange_ghosts_in_boxes(const std::vector<Particle> & owned,
                                                     const std::vector<Box> & boxes,
                                                     const Kernel & kernel);
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
        std::vector<detail::Box> boxes(FML::NTasks);
        for (int rank = 0; rank < FML::NTasks; ++rank)
            boxes[rank] = {{all_settings[std::size_t(rank) * nsettings], 0.0},
                           {all_settings[std::size_t(rank) * nsettings + 1], o.boxsize}};
        result = detail::exchange_ghosts_in_boxes(owned, boxes, kernel);
#else
        if (xmin != 0 || xmax != kernel.options().boxsize)
            throw std::invalid_argument("[ShortRangeGravity] Serial slab must cover the whole box");
#endif
        return result;
    }
    namespace detail {
        // The exchange: every rank gets the particles within the cutoff of its box (also from ranks further away)
        inline GhostSources
        exchange_ghosts_in_boxes(const std::vector<Particle> & owned, const std::vector<Box> & boxes, const Kernel & kernel) {
            GhostSources result;
            result.particles = owned;
#ifdef USE_MPI
            const auto & o = kernel.options();
        const auto needed = [&](const Particle & p, int rank) {
            return distance_to_periodic_box(p.position.data(), boxes[rank], o.boxsize) < kernel.cutoff();
        };
        // The ranks whose slab is within the cutoff of ours (periodic): only they can need any of our particles, so
        // the particles are only tested against those (testing every particle against every rank was O(N NTasks))
        std::vector<char> candidate(FML::NTasks, 0);
        const auto & mine = boxes[FML::ThisTask];
        for (int rank = 0; rank < FML::NTasks; ++rank) {
            const auto & b = boxes[rank];
            // (the x-gap as for the slabs; with pencils also the y-gap)
            const double gx = periodic_gap(b.lo[0], b.hi[0], mine.lo[0], mine.hi[0], o.boxsize);
            const double gy = periodic_gap(b.lo[1], b.hi[1], mine.lo[1], mine.hi[1], o.boxsize);
            const double gap = gy == 0.0 ? gx : std::hypot(gx, gy);
            candidate[rank] = rank != FML::ThisTask and gap < kernel.cutoff();
        }
        std::vector<std::uint64_t> send_counts(FML::NTasks), recv_counts(FML::NTasks);
        for (int rank = 0; rank < FML::NTasks; ++rank)
            if (candidate[rank])
                for (const auto & p : owned)
                    send_counts[rank] += needed(p, rank);
        MPI_Alltoall(send_counts.data(), 1, MPI_UINT64_T, recv_counts.data(), 1, MPI_UINT64_T, FML::communicator);
        // The number of rounds of every pairwise exchange in one reduction; offsets with nothing to send anywhere
        // are skipped (instead of one reduction and an empty Sendrecv per offset)
        // Send one destination at a time. Fixed-size packed records avoid padding, datatype extents and int overflow.
        constexpr std::size_t record_bytes = 4 * sizeof(double) + sizeof(std::uint64_t);
        constexpr std::size_t batch = 16384;
        std::vector<char> send_buffer(batch * record_bytes), recv_buffer(batch * record_bytes);
        std::vector<std::uint64_t> all_rounds(FML::NTasks, 0);
        for (int offset = 1; offset < FML::NTasks; ++offset) {
            const int dest = (FML::ThisTask + offset) % FML::NTasks;
            const int source = (FML::ThisTask - offset + FML::NTasks) % FML::NTasks;
            const auto n = std::max(send_counts[dest], recv_counts[source]);
            all_rounds[offset] = n / batch + (n % batch != 0);
        }
        FML::MaxArrayOverTasks(all_rounds.data(), all_rounds.size());
        for (int offset = 1; offset < FML::NTasks; ++offset) {
            const int dest = (FML::ThisTask + offset) % FML::NTasks;
            const int source = (FML::ThisTask - offset + FML::NTasks) % FML::NTasks;
            const std::uint64_t rounds = all_rounds[offset];
            if (rounds == 0)
                continue;
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
#endif
            return result;
        }
    } // namespace detail

    /// The ghosts for particles on a Peano-Hilbert domain (contiguous ranges of the Hilbert key, Global/Domain.h): every
    /// rank gets the remote particles that can be within the cutoff of its region. The box is divided into coarse cells
    /// of the curve (2^level per side, at least the cutoff wide, at most 2^7); a coarse cell is a contiguous key range,
    /// so the ranks owning any part of it are a range of ranks. A particle goes to the ranks of the cells within the
    /// cutoff of it (the cells around its own one): a superset of the ranks that need it, never fewer. Each source is
    /// sent once per rank. Owned particles must be in this rank's domain (positions in the kernel's length unit).
    /// Collective
    inline GhostSources exchange_peano_hilbert_ghosts(const std::vector<Particle> & owned,
                                                      const FML::PeanoHilbertDomain & domain,
                                                      const Kernel & kernel) {
        namespace PH = FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;
        using Key = FML::PeanoHilbertDomain::Key;
        const double L = kernel.options().boxsize;
        int invalid = 0;
        try {
            detail::validate_particles(owned, L, true);
            for (const auto & p : owned)
                if (domain.owner_of_position({p.position[0] / L, p.position[1] / L, p.position[2] / L}) != FML::ThisTask)
                    invalid = 1;
        } catch (const std::exception &) {
            invalid = 1;
        }
        FML::MaxOverTasks(&invalid);
        if (invalid)
            throw std::invalid_argument("[ShortRangeGravity] Invalid owned particles or Peano-Hilbert domain on a rank");
        GhostSources result;
        result.particles = owned;
        if (FML::NTasks == 1)
            return result;

        // The coarse cells: 2^level per side, each at least the cutoff wide (so the cells within the cutoff of a point
        // are among the 27 around it), at most the resolution of the domain and 2^7
        const double cutoff = kernel.cutoff() / L;
        int level = 0;
        while (level < 7 and level < int(domain.bits()) and std::ldexp(1.0, -(level + 1)) >= cutoff)
            level++;
        const int n = 1 << level;
        const unsigned shift = 3 * (domain.bits() - unsigned(level));
        // The ranks owning the keys of each coarse cell: [first, last]
        std::vector<std::array<int, 2>> ranks(size_t(n) * n * n);
        for (int ix = 0; ix < n; ix++)
            for (int iy = 0; iy < n; iy++)
                for (int iz = 0; iz < n; iz++) {
                    const Key k0 = PH::key_from_grid_coordinate({uint32_t(ix), uint32_t(iy), uint32_t(iz)}, unsigned(level))
                                   << shift;
                    const Key k1 = k0 + (Key(1) << shift) - 1;
                    ranks[(size_t(ix) * n + iy) * n + iz] = {domain.owner_of_key(k0), domain.owner_of_key(k1)};
                }

        // The destinations of each particle
        struct Record {
            double position[3];
            double mass;
            std::uint64_t id;
        };
        std::vector<std::vector<Record>> outgoing(FML::NTasks);
        std::vector<int> dest;
        for (const auto & p : owned) {
            std::array<double, 3> x;
            std::array<int, 3> c;
            for (int d = 0; d < 3; d++) {
                x[d] = p.position[d] / L;
                c[d] = std::min(n - 1, int(x[d] * n));
            }
            dest.clear();
            for (int a = -1; a <= 1; a++)
                for (int b = -1; b <= 1; b++)
                    for (int e = -1; e <= 1; e++) {
                        const int o[3] = {a, b, e};
                        // The periodic distance from the particle to the neighbour cell
                        double dist2 = 0;
                        std::array<int, 3> cc;
                        for (int d = 0; d < 3; d++) {
                            cc[d] = ((c[d] + o[d]) % n + n) % n;
                            if (o[d] != 0) {
                                const double edge = o[d] > 0 ? double(c[d] + 1) / n : double(c[d]) / n;
                                const double dd = std::fabs(x[d] - edge);
                                dist2 += dd * dd;
                            }
                        }
                        if (n > 1 and dist2 >= cutoff * cutoff)
                            continue;
                        const auto & r = ranks[(size_t(cc[0]) * n + cc[1]) * n + cc[2]];
                        for (int rank = r[0]; rank <= r[1]; rank++)
                            if (rank != FML::ThisTask)
                                dest.push_back(rank);
                    }
            std::sort(dest.begin(), dest.end());
            dest.erase(std::unique(dest.begin(), dest.end()), dest.end());
            for (int rank : dest) {
                Record rec;
                for (int d = 0; d < 3; d++)
                    rec.position[d] = p.position[d];
                rec.mass = p.mass;
                rec.id = p.id;
                outgoing[rank].push_back(rec);
                result.sent_particles++;
            }
        }
        auto consume = [&](const Record * r, size_t count) noexcept {
            for (size_t i = 0; i < count; i++) {
                Particle p;
                for (int d = 0; d < 3; d++)
                    p.position[d] = r[i].position[d];
                p.mass = r[i].mass;
                p.id = r[i].id;
                result.particles.push_back(p);
                result.received_particles++;
            }
        };
        FML::COMMUNICATION::exchange_records_chunked<Record>(outgoing, consume);
        return result;
    }

    /// The same for the particles on a domain (Slab, Pencil or PeanoHilbert; FML::domain: the slabs of
    /// FML::xmin_domain, FML::xmax_domain or the pencils): the region of a rank is its slab or pencil (in units of the
    /// box), or its key ranges (exchange_peano_hilbert_ghosts)
    inline GhostSources
    exchange_domain_ghosts(const std::vector<Particle> & owned, const FML::Domain & domain, const Kernel & kernel) {
        const double L = kernel.options().boxsize;
        if (domain.holds<FML::PeanoHilbertDomain>())
            return exchange_peano_hilbert_ghosts(owned, domain.get<FML::PeanoHilbertDomain>(), kernel);
        assert_mpi(domain.is_slab() or domain.holds<FML::PencilDomain>(),
                   "[ShortRangeGravity] exchange_domain_ghosts needs a Slab, a Pencil or a PeanoHilbert domain");
        if (domain.is_slab())
            return exchange_slab_ghosts(owned, domain.x_min(FML::ThisTask) * L, domain.x_max(FML::ThisTask) * L, kernel);
        std::vector<detail::Box> boxes(FML::NTasks);
        const auto & pencils = domain.get<FML::PencilDomain>();
        for (int rank = 0; rank < FML::NTasks; ++rank) {
            const auto c = pencils.coords_of(rank);
            boxes[rank] = {{pencils.x_edges()[c[0]] * L, pencils.y_edges()[c[1]] * L},
                           {pencils.x_edges()[c[0] + 1] * L, pencils.y_edges()[c[1] + 1] * L}};
        }
        int invalid = 0;
        try {
            detail::validate_particles(owned, L, true);
            const auto & b = boxes[FML::ThisTask];
            for (const auto & p : owned)
                for (int d = 0; d < 2; d++)
                    if (p.position[d] < b.lo[d] || p.position[d] >= b.hi[d])
                        invalid = 1;
        } catch (const std::invalid_argument &) {
            invalid = 1;
        }
        FML::MaxOverTasks(&invalid);
        if (invalid)
            throw std::invalid_argument("[ShortRangeGravity] Invalid owned particles or pencil on at least one rank");
        return detail::exchange_ghosts_in_boxes(owned, boxes, kernel);
    }
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
