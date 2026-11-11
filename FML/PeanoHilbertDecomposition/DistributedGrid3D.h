#ifndef FML_PEANO_HILBERT_DISTRIBUTED_GRID_3D_H
#define FML_PEANO_HILBERT_DISTRIBUTED_GRID_3D_H

#include <FML/PeanoHilbertDecomposition/PeanoHilbertDecomposition.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT {

    enum class HaloConnectivity { Faces, Full };

    /// Distributed cubic grid with local values and a cache of remotely owned
    /// cells fetched collectively from their Peano-Hilbert owner.
    template <class T> class DistributedGrid3D {
      public:
        using Coordinate = std::array<uint32_t, 3>;

        DistributedGrid3D(uint32_t cells_per_dimension, const Decomposition & decomposition,
                          const T & initial_value = T{})
            : n_(cells_per_dimension), decomposition_(decomposition) {
            if (decomposition_.nranks() < 1)
                throw std::invalid_argument("DistributedGrid3D requires a built Peano-Hilbert decomposition");
            if (n_ != (uint32_t(1) << decomposition_.bits()))
                throw std::invalid_argument("Grid resolution must equal 2^decomposition.bits()");

            const Key begin = decomposition_.key_begin(decomposition_.rank());
            const Key end = decomposition_.key_end(decomposition_.rank());
            const Key local_size = end - begin;
            if (local_size > uint64_t(std::numeric_limits<size_t>::max()))
                throw std::length_error("Local Peano-Hilbert grid does not fit in addressable memory");
            values_.assign(size_t(local_size), initial_value);
        }

        uint32_t cells_per_dimension() const { return n_; }
        int rank() const { return decomposition_.rank(); }
        int nranks() const { return decomposition_.nranks(); }
        size_t local_size() const { return values_.size(); }
        uint64_t global_size() const { return grid_cell_count(); }

        /// Number of cells in the global grid. This is independent of the number
        /// of particles used to form the decomposition.
        uint64_t grid_cell_count() const { return uint64_t(n_) * uint64_t(n_) * uint64_t(n_); }

        /// Key and coordinate for a local value, in increasing Hilbert-key order.
        Key local_key(size_t local_index) const {
            check_local_index(local_index);
            return decomposition_.key_begin(rank()) + Key(local_index);
        }
        Coordinate local_coordinate(size_t local_index) const {
            return grid_coordinate_from_key(local_key(local_index), decomposition_.bits());
        }
        Position local_cell_center(size_t local_index) const {
            const auto c = local_coordinate(local_index);
            const double inv_n = 1.0 / double(n_);
            return {(double(c[0]) + 0.5) * inv_n, (double(c[1]) + 0.5) * inv_n,
                    (double(c[2]) + 0.5) * inv_n};
        }

        /// Return the rank owning a global cell, without communicating.
        int owner(uint32_t i, uint32_t j, uint32_t k) const { return decomposition_.owner_of_key(cell_key(i, j, k)); }

        /// Access a local cell or a previously fetched remote/ghost copy.
        const T & at(uint32_t i, uint32_t j, uint32_t k) const {
            const Key key = cell_key(i, j, k);
            if (decomposition_.owner_of_key(key) == rank())
                return values_[local_index_for_key(key)];
            const auto it = ghost_lookup_.find(key);
            if (it == ghost_lookup_.end())
                throw std::out_of_range("Remote grid cell is not cached; fetch it or exchange the halo first");
            return ghost_values_[it->second];
        }
        T & at(uint32_t i, uint32_t j, uint32_t k) {
            const Key key = cell_key(i, j, k);
            if (decomposition_.owner_of_key(key) == rank())
                return values_[local_index_for_key(key)];
            const auto it = ghost_lookup_.find(key);
            if (it == ghost_lookup_.end())
                throw std::out_of_range("Remote grid cell is not cached; fetch it or exchange the halo first");
            return ghost_values_[it->second];
        }

        T & operator[](size_t local_index) {
            check_local_index(local_index);
            return values_[local_index];
        }
        const T & operator[](size_t local_index) const {
            check_local_index(local_index);
            return values_[local_index];
        }

        T * data() { return values_.data(); }
        const T * data() const { return values_.data(); }
        std::vector<T> & local_values() { return values_; }
        const std::vector<T> & local_values() const { return values_; }

        /// Initialize local cells from their center positions. The callback is
        /// evaluated only for cells owned by this rank.
        template <class Function> void initialize(Function && function) {
            for (size_t i = 0; i < values_.size(); ++i)
                values_[i] = function(local_cell_center(i));
        }

        /// Collectively fetch arbitrary cells from their owners and cache remote
        /// results. Every rank in comm calls this method, but query counts and
        /// coordinates may differ between ranks. T must be trivially copyable.
        std::vector<T> fetch_remote_cells(
            const std::vector<Coordinate> & coordinates,
#ifdef USE_MPI
            MPI_Comm comm = MPI_COMM_WORLD
#else
            int comm = 0
#endif
        ) {
            std::vector<Key> keys;
            keys.reserve(coordinates.size());
            bool valid = true;
            for (const auto & c : coordinates) {
                if (c[0] >= n_ || c[1] >= n_ || c[2] >= n_) {
                    valid = false;
                    keys.push_back(0);
                } else {
                    keys.push_back(key_from_grid_coordinate(c, decomposition_.bits()));
                }
            }
            collective_require(valid, "A rank requested a grid cell outside the global grid", comm);
            auto result = fetch_keys(keys, comm);
            cache_remote_values(keys, result);
            return result;
        }

        /// Build the unique remote-cell list for the requested stencil radius.
        /// This operation is local; call exchange_halo collectively afterwards.
        void build_halo(uint32_t width = 1, HaloConnectivity connectivity = HaloConnectivity::Full,
                        bool periodic = true) {
            std::vector<Key> requested;
            const int64_t n = int64_t(n_);
            const int64_t w = int64_t(width);
            for (size_t local = 0; local < local_size(); ++local) {
                const auto center = local_coordinate(local);
                for (int64_t di = -w; di <= w; ++di)
                    for (int64_t dj = -w; dj <= w; ++dj)
                        for (int64_t dk = -w; dk <= w; ++dk) {
                            if (di == 0 && dj == 0 && dk == 0)
                                continue;
                            if (connectivity == HaloConnectivity::Faces &&
                                (int(di != 0) + int(dj != 0) + int(dk != 0) != 1))
                                continue;
                            int64_t c[3] = {int64_t(center[0]) + di, int64_t(center[1]) + dj,
                                            int64_t(center[2]) + dk};
                            bool inside = true;
                            for (int axis = 0; axis < 3; ++axis) {
                                if (periodic) {
                                    c[axis] %= n;
                                    if (c[axis] < 0)
                                        c[axis] += n;
                                } else if (c[axis] < 0 || c[axis] >= n) {
                                    inside = false;
                                    break;
                                }
                            }
                            if (!inside)
                                continue;
                            const Key key = key_from_grid_coordinate(
                                {uint32_t(c[0]), uint32_t(c[1]), uint32_t(c[2])}, decomposition_.bits());
                            if (decomposition_.owner_of_key(key) != rank())
                                requested.push_back(key);
                        }
            }
            std::sort(requested.begin(), requested.end());
            requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
            halo_keys_ = std::move(requested);
            halo_width_ = width;
            halo_periodic_ = periodic;
            halo_connectivity_ = connectivity;
        }

        /// Refresh all cells collected by build_halo. This is collective over
        /// comm. Call it after owners update their local grid values.
        void exchange_halo(
#ifdef USE_MPI
            MPI_Comm comm = MPI_COMM_WORLD
#else
            int comm = 0
#endif
        ) {
            const auto values = fetch_keys(halo_keys_, comm);
            cache_remote_values(halo_keys_, values);
        }

        const std::vector<Key> & halo_keys() const { return halo_keys_; }
        uint32_t halo_width() const { return halo_width_; }
        bool halo_is_periodic() const { return halo_periodic_; }
        HaloConnectivity halo_connectivity() const { return halo_connectivity_; }

        const Decomposition & decomposition() const { return decomposition_; }

      private:
#ifdef USE_MPI
        static void collective_require(bool valid, const char * message, MPI_Comm comm) {
            int local = valid ? 1 : 0;
            int global = 0;
            MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
            if (!global)
                throw std::runtime_error(message);
        }
#else
        static void collective_require(bool valid, const char * message, int) {
            if (!valid)
                throw std::runtime_error(message);
        }
#endif

        std::vector<T> fetch_keys(
            const std::vector<Key> & keys,
#ifdef USE_MPI
            MPI_Comm comm
#else
            int comm
#endif
        ) {
            static_assert(std::is_trivially_copyable_v<T>,
                          "MPI remote grid access requires a trivially copyable cell value");
            const int comm_rank = [&]() {
#ifdef USE_MPI
                int r = 0;
                MPI_Comm_rank(comm, &r);
                return r;
#else
                return 0;
#endif
            }();
            const int comm_size = [&]() {
#ifdef USE_MPI
                int s = 1;
                MPI_Comm_size(comm, &s);
                return s;
#else
                return 1;
#endif
            }();
            collective_require(comm_rank == rank() && comm_size == nranks(),
                               "Grid decomposition and halo communicator do not match", comm);

            std::vector<std::vector<std::pair<Key, size_t>>> grouped(static_cast<size_t>(nranks()));
            bool keys_valid = true;
            for (size_t q = 0; q < keys.size(); ++q) {
                if (keys[q] >= grid_cell_count()) {
                    keys_valid = false;
                    continue;
                }
                const int owner_rank = decomposition_.owner_of_key(keys[q]);
                grouped[size_t(owner_rank)].emplace_back(keys[q], q);
            }
            collective_require(keys_valid, "A rank supplied an invalid Peano-Hilbert grid key", comm);

            std::vector<size_t> send_counts_size(static_cast<size_t>(nranks()));
            for (int p = 0; p < nranks(); ++p)
                send_counts_size[size_t(p)] = grouped[size_t(p)].size();
            std::vector<int> send_counts, send_displacements;
            size_t send_total = 0;
            bool send_layout_valid = make_mpi_layout(send_counts_size, send_counts, send_displacements, send_total);
            collective_require(send_layout_valid, "Grid request counts exceed MPI int limits", comm);

#ifdef USE_MPI
            std::vector<int> recv_counts(size_t(nranks()), 0), recv_displacements;
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
            std::vector<size_t> recv_counts_size(static_cast<size_t>(nranks()));
            for (int p = 0; p < nranks(); ++p)
                recv_counts_size[size_t(p)] = size_t(recv_counts[size_t(p)]);
            size_t recv_total = 0;
            bool recv_layout_valid = make_mpi_layout(recv_counts_size, recv_counts, recv_displacements, recv_total);
            collective_require(recv_layout_valid, "Grid receive request counts exceed MPI int limits", comm);

            std::vector<Key> send_keys(send_total), recv_keys(recv_total);
            std::vector<size_t> original_positions(send_total);
            for (int p = 0; p < nranks(); ++p) {
                size_t cursor = size_t(send_displacements[size_t(p)]);
                for (const auto & request : grouped[size_t(p)]) {
                    send_keys[cursor] = request.first;
                    original_positions[cursor] = request.second;
                    ++cursor;
                }
            }
            MPI_Alltoallv(send_keys.data(), send_counts.data(), send_displacements.data(), MPI_UINT64_T,
                          recv_keys.data(), recv_counts.data(), recv_displacements.data(), MPI_UINT64_T, comm);

            std::vector<T> response_send(recv_total);
            bool requests_owned = true;
            for (size_t q = 0; q < recv_total; ++q) {
                const Key key = recv_keys[q];
                if (decomposition_.owner_of_key(key) != rank()) {
                    requests_owned = false;
                    response_send[q] = T{};
                } else {
                    response_send[q] = values_[size_t(key - decomposition_.key_begin(rank()))];
                }
            }
            collective_require(requests_owned, "Grid request was routed to a non-owner rank", comm);

            std::vector<size_t> response_send_counts_size(static_cast<size_t>(nranks()));
            std::vector<size_t> response_recv_counts_size(static_cast<size_t>(nranks()));
            for (int p = 0; p < nranks(); ++p) {
                response_send_counts_size[size_t(p)] = size_t(recv_counts[size_t(p)]);
                response_recv_counts_size[size_t(p)] = size_t(send_counts[size_t(p)]);
            }
            std::vector<int> response_send_counts, response_send_displacements;
            std::vector<int> response_recv_counts, response_recv_displacements;
            size_t response_send_total = 0, response_recv_total = 0;
            bool response_layout_valid = sizeof(T) <= size_t(INT_MAX) &&
                make_mpi_layout(response_send_counts_size, response_send_counts, response_send_displacements,
                                response_send_total, sizeof(T)) &&
                make_mpi_layout(response_recv_counts_size, response_recv_counts, response_recv_displacements,
                                response_recv_total, sizeof(T));
            collective_require(response_layout_valid, "Grid value exchange exceeds MPI int byte limits", comm);
            std::vector<T> response_recv(send_total);
            MPI_Alltoallv(reinterpret_cast<const char *>(response_send.data()), response_send_counts.data(),
                          response_send_displacements.data(), MPI_BYTE,
                          reinterpret_cast<char *>(response_recv.data()), response_recv_counts.data(),
                          response_recv_displacements.data(), MPI_BYTE, comm);

            std::vector<T> result(keys.size());
            for (size_t q = 0; q < send_total; ++q)
                result[original_positions[q]] = response_recv[q];
            return result;
#else
            std::vector<T> result(keys.size());
            for (size_t q = 0; q < keys.size(); ++q) {
                if (decomposition_.owner_of_key(keys[q]) != rank())
                    throw std::runtime_error("Remote grid access requires USE_MPI");
                result[q] = values_[size_t(keys[q] - decomposition_.key_begin(rank()))];
            }
            return result;
#endif
        }

        static bool make_mpi_layout(const std::vector<size_t> & counts, std::vector<int> & mpi_counts,
                                    std::vector<int> & displacements, size_t & total, size_t scale = 1) {
            mpi_counts.assign(counts.size(), 0);
            displacements.assign(counts.size(), 0);
            size_t offset = 0;
            for (size_t i = 0; i < counts.size(); ++i) {
                if (counts[i] > size_t(INT_MAX) / scale || offset > size_t(INT_MAX) ||
                    counts[i] * scale > size_t(INT_MAX) - offset)
                    return false;
                mpi_counts[i] = int(counts[i] * scale);
                displacements[i] = int(offset);
                offset += counts[i] * scale;
            }
            total = offset / scale;
            if (offset % scale != 0)
                return false;
            return true;
        }

        void cache_remote_values(const std::vector<Key> & keys, const std::vector<T> & values) {
            if (keys.size() != values.size())
                throw std::logic_error("Remote grid key/value result size mismatch");
            for (size_t i = 0; i < keys.size(); ++i) {
                if (decomposition_.owner_of_key(keys[i]) == rank())
                    continue;
                auto it = ghost_lookup_.find(keys[i]);
                if (it == ghost_lookup_.end()) {
                    const size_t slot = ghost_values_.size();
                    ghost_lookup_.emplace(keys[i], slot);
                    ghost_keys_.push_back(keys[i]);
                    ghost_values_.push_back(values[i]);
                } else {
                    ghost_values_[it->second] = values[i];
                }
            }
        }

        Key cell_key(uint32_t i, uint32_t j, uint32_t k) const {
            if (i >= n_ || j >= n_ || k >= n_)
                throw std::out_of_range("DistributedGrid3D cell coordinate is outside the grid");
            return key_from_grid_coordinate({i, j, k}, decomposition_.bits());
        }

        size_t local_index_for_key(Key key) const {
            if (decomposition_.owner_of_key(key) != rank())
                throw std::out_of_range("Requested grid cell is owned by another MPI rank");
            const Key begin = decomposition_.key_begin(rank());
            const Key offset = key - begin;
            if (offset >= values_.size())
                throw std::logic_error("Owned grid cell is outside the local key interval");
            return size_t(offset);
        }

        void check_local_index(size_t local_index) const {
            if (local_index >= values_.size())
                throw std::out_of_range("DistributedGrid3D local cell index is outside this rank's storage");
        }

        uint32_t n_{0};
        Decomposition decomposition_;
        std::vector<T> values_;
        std::vector<Key> halo_keys_;
        uint32_t halo_width_{0};
        bool halo_periodic_{true};
        HaloConnectivity halo_connectivity_{HaloConnectivity::Full};
        std::vector<Key> ghost_keys_;
        std::vector<T> ghost_values_;
        std::unordered_map<Key, size_t> ghost_lookup_;
    };

} // namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT

#endif
