#ifndef FML_PERIODIC_SLAB_EXCHANGE_HEADER
#define FML_PERIODIC_SLAB_EXCHANGE_HEADER

#include <FML/Global/ChunkedExchange.h>
#include <FML/Global/Global.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace FML::COMMUNICATION {
    namespace detail {
        // Minimum distance on the unit circle to a closed interval in [0,1].
        inline double periodic_distance_to_slab(double x, double lo, double hi) {
            double distance = 1.0;
            for (int image = -1; image <= 1; ++image) {
                const double y = x + image;
                distance = std::min(distance, std::max({lo - y, y - hi, 0.0}));
            }
            return distance;
        }
    } // namespace detail

    /// Append remote records within buffer of the local periodic x slab, once per
    /// source record/destination. Input must contain only owned records, with x in
    /// [xmin_domain,xmax_domain) and [0,1). Coordinates remain wrapped; no duplicate
    /// periodic images are appended. Slabs may be nonuniform or empty, and buffers
    /// may differ by rank. Collective on FML::communicator, including empty ranks.
    /// Record must be trivially copyable. get_x(record) returns its x coordinate.
    /// Outgoing records are staged per intersecting peer; receive scratch is bounded
    /// by max_message_bytes. The final owned+ghost vector is reserved before receipt.
    template <class Record, class GetX>
    void append_periodic_slab_ghosts(std::vector<Record> & records,
                                     double buffer,
                                     GetX get_x,
                                     size_t max_message_bytes = 8 * 1024 * 1024) {
        static_assert(std::is_trivially_copyable_v<Record>, "Slab records must be trivially copyable");
        const double lo = FML::xmin_domain, hi = FML::xmax_domain;
        assert_mpi(std::isfinite(buffer) && buffer >= 0 && lo >= 0 && hi <= 1 && lo <= hi,
                   "[append_periodic_slab_ghosts] Invalid slab or buffer");
        bool owned = true;
        for (const auto & record : records) {
            const double x = get_x(record);
            owned = owned && std::isfinite(x) && x >= lo && x < hi;
        }
        assert_mpi(owned, "[append_periodic_slab_ghosts] Expected owned records only");
#ifdef USE_MPI
        if (FML::NTasks > 1) {
            const std::array<double, 3> local{lo, hi, buffer};
            std::vector<std::array<double, 3>> slabs(FML::NTasks);
            MPI_Allgather(local.data(), 3, MPI_DOUBLE, slabs.data(), 3, MPI_DOUBLE, FML::communicator);
            std::vector<std::vector<Record>> outgoing(FML::NTasks);
            std::vector<unsigned long long> nsend(FML::NTasks), nrecv(FML::NTasks);
            for (int peer = 0; peer < FML::NTasks; ++peer) {
                const auto & slab = slabs[peer];
                if (peer == FML::ThisTask || slab[0] == slab[1])
                    continue;
                bool intersects = false;
                for (int image = -1; image <= 1; ++image)
                    intersects = intersects || (hi >= slab[0] - slab[2] + image && lo <= slab[1] + slab[2] + image);
                if (!intersects)
                    continue;
                auto needed = [&](const Record & record) {
                    return detail::periodic_distance_to_slab(get_x(record), slab[0], slab[1]) <= slab[2];
                };
                size_t count = 0;
                for (const auto & record : records)
                    count += needed(record);
                auto & send = outgoing[peer];
                send.reserve(count);
                for (const auto & record : records)
                    if (needed(record))
                        send.push_back(record);
                nsend[peer] = count;
            }
            MPI_Alltoall(nsend.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         nrecv.data(),
                         1,
                         MPI_UNSIGNED_LONG_LONG,
                         FML::communicator);
            size_t total = records.size();
            for (auto count : nrecv) {
                assert_mpi(count <= records.max_size() - total,
                           "[append_periodic_slab_ghosts] Receive size exceeds vector capacity");
                total += size_t(count);
            }
            records.reserve(total);
            exchange_records_chunked(
                outgoing,
                [&](const Record * data, size_t count) noexcept {
                    // Trivial records and sufficient capacity: no allocation during receipt.
                    records.insert(records.end(), data, data + count);
                },
                max_message_bytes);
        }
#else
        (void)max_message_bytes;
#endif
    }
    /// The same for the particle domain FML::domain (Slab: the function above; Pencil: the records within buffer of
    /// the pencil of each task, the corners included). get_xy(record) returns {x, y} (in [0,1)); with a Slab domain
    /// only x is used
    template <class Record, class GetXY>
    void append_periodic_domain_ghosts(std::vector<Record> & records,
                                       double buffer,
                                       GetXY get_xy,
                                       size_t max_message_bytes = 8 * 1024 * 1024) {
        if (not FML::domain.holds<FML::PencilDomain>()) {
            append_periodic_slab_ghosts(
                records, buffer, [&](const Record & r) { return double(get_xy(r)[0]); }, max_message_bytes);
            return;
        }
        static_assert(std::is_trivially_copyable_v<Record>, "Records must be trivially copyable");
        const auto & pencils = FML::domain.get<FML::PencilDomain>();
        auto box_of = [&](int task) {
            const auto c = pencils.coords_of(task);
            return std::array<double, 4>{pencils.x_edges()[c[0]],
                                         pencils.x_edges()[c[0] + 1],
                                         pencils.y_edges()[c[1]],
                                         pencils.y_edges()[c[1] + 1]};
        };
        const auto mine = box_of(FML::ThisTask);
        bool owned = std::isfinite(buffer) && buffer >= 0;
        for (const auto & record : records) {
            const auto xy = get_xy(record);
            owned = owned && xy[0] >= mine[0] && xy[0] < mine[1] && xy[1] >= mine[2] && xy[1] < mine[3];
        }
        assert_mpi(owned, "[append_periodic_domain_ghosts] Invalid buffer or records that are not owned");
#ifdef USE_MPI
        if (FML::NTasks > 1) {
            std::vector<double> buffers(FML::NTasks);
            MPI_Allgather(&buffer, 1, MPI_DOUBLE, buffers.data(), 1, MPI_DOUBLE, FML::communicator);
            std::vector<std::vector<Record>> outgoing(FML::NTasks);
            // The periodic gap between the intervals [a0,a1] and [b0,b1] (0 if they overlap)
            auto interval_gap = [](double a0, double a1, double b0, double b1) {
                double gap = 1.0;
                for (int image = -1; image <= 1; ++image)
                    gap = std::min(gap, std::max(0.0, std::max(a0 - (b1 + image), (b0 + image) - a1)));
                return gap;
            };
            for (int peer = 0; peer < FML::NTasks; ++peer) {
                if (peer == FML::ThisTask)
                    continue;
                const auto b = box_of(peer);
                // No record of ours can be within the buffer of the peer if our whole box is not (skipping these
                // peers makes the cost O(neighbours x records) rather than O(NTasks x records))
                if (std::hypot(interval_gap(mine[0], mine[1], b[0], b[1]), interval_gap(mine[2], mine[3], b[2], b[3])) >
                    buffers[peer])
                    continue;
                for (const auto & record : records) {
                    const auto xy = get_xy(record);
                    const double dx = detail::periodic_distance_to_slab(xy[0], b[0], b[1]);
                    const double dy = detail::periodic_distance_to_slab(xy[1], b[2], b[3]);
                    if (std::hypot(dx, dy) <= buffers[peer])
                        outgoing[peer].push_back(record);
                }
            }
            size_t total = records.size();
            for (const auto & o : outgoing)
                total += o.size(); // (a guess of what we receive: the same order)
            records.reserve(total);
            exchange_records_chunked(
                outgoing,
                [&](const Record * data, size_t count) { records.insert(records.end(), data, data + count); },
                max_message_bytes);
        }
#else
        (void)max_message_bytes;
#endif
    }
} // namespace FML::COMMUNICATION
#endif
