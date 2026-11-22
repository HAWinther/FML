#ifndef FML_FIELD_LEVEL_DISTRIBUTED_ASSIGNMENT_H
#define FML_FIELD_LEVEL_DISTRIBUTED_ASSIGNMENT_H
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/MPICounts.h>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <climits>
#include <numeric>
#include <limits>

namespace FML::FIELDLEVEL::FRAMEWORK::DETAIL {
    // Sparse scatter of global mesh-node contributions to their owning FFT slab.
    // Reverse sends each node's cotangent back in exactly the forward request order.
    // Particles stay on their Lagrangian owners; nodes can be on any rank.
    // Any grid decomposed in x (FFTWGrid slabs) or in x and y (PencilGrid pencils): the owners from the local boxes
    template <class GridType = FML::GRID::FFTWGrid<3>>
    class DistributedAssignmentT {
        using Grid = GridType;
        FML::GRID::CellOwners<3> owners;
        std::array<int, 3> local_start{}, local_count{};
        std::vector<int> send_count, recv_count, send_offset, recv_offset;
        // Only nodes owned by OTHER ranks are routed; owned nodes accumulate densely in `local`.
        std::vector<std::uint64_t> sent, received;
        std::unordered_map<std::uint64_t, size_t> slot;
        std::vector<double> local;
        size_t owned{};
        bool contiguous{false}, full_rows{false};
        std::uint64_t first_key{};
        int mesh, tasks, rank;
        bool ready{false};
        void check(const Grid & grid) const {
            FML::COMMUNICATION::require_count(
                tasks == FML::NTasks && rank == FML::ThisTask && grid.get_nmesh() == mesh &&
                    grid.get_local_range(0)[0] == local_start[0] && grid.get_local_range(0)[1] == local_count[0] &&
                    grid.get_local_range(1)[0] == local_start[1] && grid.get_local_range(1)[1] == local_count[1],
                "[Distributed assignment] Incompatible grid/layout");
        }
        static size_t offsets(const std::vector<int> & counts, std::vector<int> & offsets) {
            size_t total = 0;
            offsets.resize(counts.size());
            for (size_t p = 0; p < counts.size(); ++p) {
                if (counts[p] < 0 || size_t(counts[p]) > size_t(INT_MAX) - total)
                    throw std::overflow_error("[Distributed assignment] MPI int displacement limit exceeded");
                offsets[p] = int(total);
                total += size_t(counts[p]);
            }
            return total;
        }
        int owner(std::uint64_t node) const {
            const auto global_x = node / (std::uint64_t(mesh) * mesh);
            if (global_x >= std::uint64_t(mesh))
                throw std::invalid_argument("[Distributed assignment] Node outside mesh");
            const int task = owners.owner({int(global_x), int(node / mesh % mesh), int(node % mesh)});
            if (task < 0)
                throw std::invalid_argument("[Distributed assignment] Node outside mesh");
            return task;
        }
        // Compact owned index in real-range order (x, then y, then z), or SIZE_MAX if not owned here.
        size_t owned_index(std::uint64_t node) const {
            // Slabs (and one rank) own a contiguous key range: no division needed.
            if (contiguous)
                return node - first_key < owned ? size_t(node - first_key) : std::numeric_limits<size_t>::max();
            const std::uint64_t m = std::uint64_t(mesh), row = node / m;
            const long long x = (long long)(row / m) - local_start[0], y = (long long)(row % m) - local_start[1],
                            z = (long long)(node % m) - local_start[2];
            if (x < 0 || x >= local_count[0] || y < 0 || y >= local_count[1] || z < 0 || z >= local_count[2])
                return std::numeric_limits<size_t>::max();
            return (size_t(x) * size_t(local_count[1]) + size_t(y)) * size_t(local_count[2]) + size_t(z);
        }
        // Compact offset of the start of the owned z-row (x, y), or SIZE_MAX if not owned here.
        // The supported grids never split z, so a row is owned entirely by one rank.
        size_t owned_row(int x, int y) const {
            const int ix = x - local_start[0], iy = y - local_start[1];
            if (!full_rows || ix < 0 || ix >= local_count[0] || iy < 0 || iy >= local_count[1])
                return std::numeric_limits<size_t>::max();
            return (size_t(ix) * size_t(local_count[1]) + size_t(iy)) * size_t(mesh);
        }
        // Passed to deposit's visitor. add(key, value) adds one node; add.row(x, y, z, v, n) adds
        // v[c] at the nodes (x, y, z[c]), c = 0..n-1, in that order (the same per-node sequence).
        class Adder {
            DistributedAssignmentT & owner_;
            std::unordered_map<std::uint64_t, double> & remote_;
            const std::uint64_t nodes_;

          public:
            Adder(DistributedAssignmentT & owner, std::unordered_map<std::uint64_t, double> & remote,
                  std::uint64_t nodes)
                : owner_(owner), remote_(remote), nodes_(nodes) {}
            void operator()(std::uint64_t key, double value) const {
                if (key >= nodes_)
                    throw std::invalid_argument("[Distributed assignment] Node outside mesh");
                const size_t q = owner_.owned_index(key);
                if (q != std::numeric_limits<size_t>::max())
                    owner_.local[q] += value;
                else
                    remote_[key] += value;
            }
            void row(int x, int y, const int * z, const double * values, int n) const {
                const int m = owner_.mesh;
                if (x < 0 || x >= m || y < 0 || y >= m)
                    throw std::invalid_argument("[Distributed assignment] Node outside mesh");
                const size_t base = owner_.owned_row(x, y);
                if (base != std::numeric_limits<size_t>::max()) {
                    double * row = owner_.local.data() + base;
                    for (int c = 0; c < n; ++c) {
                        if (z[c] < 0 || z[c] >= m)
                            throw std::invalid_argument("[Distributed assignment] Node outside mesh");
                        row[z[c]] += values[c];
                    }
                    return;
                }
                const std::uint64_t key = (std::uint64_t(x) * std::uint64_t(m) + std::uint64_t(y)) * std::uint64_t(m);
                for (int c = 0; c < n; ++c)
                    (*this)(key + std::uint64_t(z[c]), values[c]);
            }
        };
        ptrdiff_t local_index(std::uint64_t node, const Grid & grid) const {
            const auto row = node / mesh;
            const auto x = row / mesh;
            return grid.get_index_real(
                {int(x) - local_start[0], int(row % mesh) - local_start[1], int(node % mesh) - local_start[2]});
        }

      public:
        explicit DistributedAssignmentT(const Grid & grid)
            : owners(grid), mesh(grid.get_nmesh()), tasks(FML::NTasks), rank(FML::ThisTask) {
            FML::COMMUNICATION::require_count(mesh > 0 && tasks > 0 && rank >= 0 && rank < tasks,
                                              "[Distributed assignment] Unallocated grid or invalid task metadata");
            owned = 1;
            for (int d = 0; d < 3; d++) {
                local_start[d] = int(grid.get_local_range(d)[0]);
                local_count[d] = int(grid.get_local_range(d)[1]);
                owned *= size_t(std::max(0, local_count[d]));
            }
            contiguous = local_start[1] == 0 && local_start[2] == 0 && local_count[1] == mesh && local_count[2] == mesh;
            first_key = std::uint64_t(local_start[0]) * std::uint64_t(mesh) * std::uint64_t(mesh);
            full_rows = local_start[2] == 0 && local_count[2] == mesh;
        }
        // Each node's total is accumulated in visit order, and owners add the per-rank totals in
        // ascending source-rank order: the same floating-point sequence as a full exchange.
        template <class Visit>
        void deposit(Grid & grid, Visit visit) {
            check(grid);
            ready = false;
            using FML::COMMUNICATION::collective_try;
            std::vector<double> values, incoming;
            collective_try(
                [&] {
                    std::unordered_map<std::uint64_t, double> remote;
                    local.assign(owned, 0.);
                    const std::uint64_t nodes = std::uint64_t(mesh) * mesh * mesh;
                    Adder add{*this, remote, nodes};
                    visit(add);
                    sent.clear();
                    slot.clear();
                    sent.reserve(remote.size());
                    for (const auto & entry : remote)
                        sent.push_back(entry.first);
                    std::vector<int> peer(sent.size());
                    std::vector<size_t> order(sent.size());
                    for (size_t j = 0; j < sent.size(); ++j) {
                        peer[j] = owner(sent[j]);
                        order[j] = j;
                    }
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return peer[a] == peer[b] ? sent[a] < sent[b] : peer[a] < peer[b];
                    });
                    std::vector<std::uint64_t> keys(sent.size());
                    send_count.assign(tasks, 0);
                    recv_count.assign(tasks, 0);
                    values.resize(sent.size());
                    for (size_t j = 0; j < order.size(); ++j) {
                        const int p = peer[order[j]];
                        if (p == rank)
                            throw std::logic_error("[Distributed assignment] Owned node routed remotely");
                        if (send_count[p] == INT_MAX)
                            throw std::overflow_error("[Distributed assignment] MPI count overflow");
                        ++send_count[p];
                        keys[j] = sent[order[j]];
                        values[j] = remote.at(keys[j]);
                        slot.emplace(keys[j], j);
                    }
                    sent.swap(keys);
                    offsets(send_count, send_offset);
                },
                "Assignment local scatter");
#ifdef USE_MPI
            MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, FML::communicator);
#else
            recv_count = send_count;
#endif
            collective_try(
                [&] {
                    const size_t n = offsets(recv_count, recv_offset);
                    received.resize(n);
                    incoming.resize(n);
                },
                "Assignment receive storage");
#ifdef USE_MPI
            MPI_Alltoallv(sent.data(),
                          send_count.data(),
                          send_offset.data(),
                          MPI_UINT64_T,
                          received.data(),
                          recv_count.data(),
                          recv_offset.data(),
                          MPI_UINT64_T,
                          FML::communicator);
            MPI_Alltoallv(values.data(),
                          send_count.data(),
                          send_offset.data(),
                          MPI_DOUBLE,
                          incoming.data(),
                          recv_count.data(),
                          recv_offset.data(),
                          MPI_DOUBLE,
                          FML::communicator);
#else
            received = sent;
            incoming = values;
#endif
            grid.fill_real_grid(FML::GRID::FloatType(0));
            for (int source = 0; source < tasks; ++source) {
                if (source == rank) {
                    size_t q = 0;
                    for (auto i : grid.get_real_range()) {
                        if (local[q] != 0 || std::signbit(local[q]))
                            grid.set_real_from_index(i, grid.get_real_from_index(i) + local[q]);
                        ++q;
                    }
                    continue;
                }
                const size_t begin = size_t(recv_offset[source]), end = begin + size_t(recv_count[source]);
                for (size_t j = begin; j < end; ++j) {
                    const auto index = local_index(received[j], grid);
                    grid.set_real_from_index(index, grid.get_real_from_index(index) + incoming[j]);
                }
            }
            ready = true;
        }
        // Owned node values (compact real-range order) followed by remote values in sent order.
        std::vector<double> reverse(const Grid & grid) const {
            check(grid);
            FML::COMMUNICATION::require_count(ready, "[Distributed assignment] Reverse requires successful deposit");
            std::vector<double> outgoing, result;
            FML::COMMUNICATION::collective_try(
                [&] {
                    outgoing.resize(received.size());
                    result.resize(owned + sent.size());
                    for (size_t j = 0; j < received.size(); ++j)
                        outgoing[j] = grid.get_real_from_index(local_index(received[j], grid));
                    size_t q = 0;
                    for (auto i : grid.get_real_range())
                        result[q++] = grid.get_real_from_index(i);
                },
                "Assignment reverse storage");
#ifdef USE_MPI
            MPI_Alltoallv(outgoing.data(),
                          recv_count.data(),
                          recv_offset.data(),
                          MPI_DOUBLE,
                          result.data() + owned,
                          send_count.data(),
                          send_offset.data(),
                          MPI_DOUBLE,
                          FML::communicator);
#else
            std::copy(outgoing.begin(), outgoing.end(), result.begin() + owned);
#endif
            return result;
        }
        // Owned z-row (x, y) of a reverse() result, indexed by z; nullptr if the row is remote
        // (then use at() per node).
        const double * row(const std::vector<double> & reversed, int x, int y) const {
            const size_t base = owned_row(x, y);
            return base == std::numeric_limits<size_t>::max() ? nullptr : reversed.data() + base;
        }
        double at(const std::vector<double> & reversed, std::uint64_t node) const {
            const size_t q = owned_index(node);
            if (q != std::numeric_limits<size_t>::max())
                return reversed[q];
            const auto it = slot.find(node);
            if (it == slot.end())
                throw std::out_of_range("[Distributed assignment] Node was not deposited");
            return reversed[owned + it->second];
        }
    };
    using DistributedAssignment = DistributedAssignmentT<>;
} // namespace FML::FIELDLEVEL::FRAMEWORK::DETAIL
#endif
