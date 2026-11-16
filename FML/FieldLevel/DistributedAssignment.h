#ifndef FML_FIELD_LEVEL_DISTRIBUTED_ASSIGNMENT_H
#define FML_FIELD_LEVEL_DISTRIBUTED_ASSIGNMENT_H
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/MPICounts.h>
#include <unordered_map>
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
        std::vector<std::uint64_t> sent, received;
        std::unordered_map<std::uint64_t, size_t> slot;
        int mesh, tasks, rank;
        // Small global meshes fit in bounded dense scratch. Preserve visit and
        // sorted packet order, including zero-weight keys; only the lookup changes.
        bool dense_small{false};
        std::vector<size_t> dense_slot;
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
            dense_small = mesh <= 16;
            for (int d = 0; d < 3; d++) {
                local_start[d] = int(grid.get_local_range(d)[0]);
                local_count[d] = int(grid.get_local_range(d)[1]);
            }
        }
        template <class Visit>
        void deposit(Grid & grid, Visit visit) {
            check(grid);
            ready = false;
            using FML::COMMUNICATION::collective_try;
            std::vector<double> values, incoming;
            collective_try(
                [&] {
                    std::unordered_map<std::uint64_t, double> sum;
                    std::vector<double> dense_sum;
                    std::vector<unsigned char> seen;
                    sent.clear();
                    slot.clear();
                    if (dense_small) {
                        const size_t nodes = size_t(mesh) * mesh * mesh;
                        dense_sum.assign(nodes, 0.);
                        seen.assign(nodes, 0);
                        dense_slot.assign(nodes, std::numeric_limits<size_t>::max());
                        visit([&](std::uint64_t key, double value) {
                            if (key >= nodes) throw std::invalid_argument("[Distributed assignment] Node outside mesh");
                            if (!seen[key]) { seen[key] = 1; sent.push_back(key); }
                            dense_sum[key] += value;
                        });
                    } else {
                        visit([&](std::uint64_t key, double value) { sum[key] += value; });
                        for (const auto & entry : sum) sent.push_back(entry.first);
                    }
                    std::sort(sent.begin(), sent.end(), [&](auto a, auto b) {
                        const int pa = owner(a), pb = owner(b);
                        return pa == pb ? a < b : pa < pb;
                    });
                    send_count.assign(tasks, 0);
                    recv_count.assign(tasks, 0);
                    values.resize(sent.size());
                    for (size_t j = 0; j < sent.size(); ++j) {
                        const int peer = owner(sent[j]);
                        if (send_count[peer] == INT_MAX)
                            throw std::overflow_error("[Distributed assignment] MPI count overflow");
                        ++send_count[peer];
                        values[j] = dense_small ? dense_sum[sent[j]] : sum.at(sent[j]);
                        if (dense_small) dense_slot[sent[j]] = j;
                        else slot.emplace(sent[j], j);
                    }
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
            for (size_t j = 0; j < received.size(); ++j) {
                const auto index = local_index(received[j], grid);
                grid.set_real_from_index(index, grid.get_real_from_index(index) + incoming[j]);
            }
            ready = true;
        }
        std::vector<double> reverse(const Grid & grid) const {
            check(grid);
            FML::COMMUNICATION::require_count(ready, "[Distributed assignment] Reverse requires successful deposit");
            std::vector<double> outgoing, result;
            FML::COMMUNICATION::collective_try(
                [&] {
                    outgoing.resize(received.size());
                    result.resize(sent.size());
                    for (size_t j = 0; j < received.size(); ++j)
                        outgoing[j] = grid.get_real_from_index(local_index(received[j], grid));
                },
                "Assignment reverse storage");
#ifdef USE_MPI
            MPI_Alltoallv(outgoing.data(),
                          recv_count.data(),
                          recv_offset.data(),
                          MPI_DOUBLE,
                          result.data(),
                          send_count.data(),
                          send_offset.data(),
                          MPI_DOUBLE,
                          FML::communicator);
#else
            result = outgoing;
#endif
            return result;
        }
        double at(const std::vector<double> & reversed, std::uint64_t node) const {
            if (dense_small) {
                if (node >= dense_slot.size() || dense_slot[node] == std::numeric_limits<size_t>::max())
                    throw std::out_of_range("[Distributed assignment] Node was not deposited");
                return reversed[dense_slot[node]];
            }
            return reversed[slot.at(node)];
        }
    };
    using DistributedAssignment = DistributedAssignmentT<>;
} // namespace FML::FIELDLEVEL::FRAMEWORK::DETAIL
#endif
