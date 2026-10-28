#ifndef FML_PERIODIC_SLAB_EXCHANGE_HEADER
#define FML_PERIODIC_SLAB_EXCHANGE_HEADER

#include <FML/Global/ChunkedExchange.h>
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
    }

    /// Append remote records within buffer of the local periodic x slab, once per
    /// source record/destination. Input must contain only owned records, with x in
    /// [xmin_domain,xmax_domain) and [0,1). Coordinates remain wrapped; no duplicate
    /// periodic images are appended. Slabs may be nonuniform or empty, and buffers
    /// may differ by rank. Collective on FML::communicator, including empty ranks.
    /// Record must be trivially copyable. get_x(record) returns its x coordinate.
    /// Outgoing records are staged per intersecting peer; receive scratch is bounded
    /// by max_message_bytes. The final owned+ghost vector is reserved before receipt.
    template <class Record, class GetX>
    void append_periodic_slab_ghosts(std::vector<Record> & records, double buffer, GetX get_x,
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
                if (peer == FML::ThisTask || slab[0] == slab[1]) continue;
                bool intersects = false;
                for (int image = -1; image <= 1; ++image)
                    intersects = intersects || (hi >= slab[0] - slab[2] + image &&
                                                 lo <= slab[1] + slab[2] + image);
                if (!intersects) continue;
                auto needed = [&](const Record & record) {
                    return detail::periodic_distance_to_slab(get_x(record), slab[0], slab[1]) <= slab[2];
                };
                size_t count = 0;
                for (const auto & record : records) count += needed(record);
                auto & send = outgoing[peer];
                send.reserve(count);
                for (const auto & record : records)
                    if (needed(record)) send.push_back(record);
                nsend[peer] = count;
            }
            MPI_Alltoall(nsend.data(), 1, MPI_UNSIGNED_LONG_LONG,
                         nrecv.data(), 1, MPI_UNSIGNED_LONG_LONG, FML::communicator);
            size_t total = records.size();
            for (auto count : nrecv) {
                assert_mpi(count <= records.max_size() - total,
                           "[append_periodic_slab_ghosts] Receive size exceeds vector capacity");
                total += size_t(count);
            }
            records.reserve(total);
            exchange_records_chunked(outgoing, [&](const Record * data, size_t count) noexcept {
                // Trivial records and sufficient capacity: no allocation during receipt.
                records.insert(records.end(), data, data + count);
            }, max_message_bytes);
        }
#else
        (void)max_message_bytes;
#endif
    }
}
#endif
