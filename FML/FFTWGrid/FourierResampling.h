#ifndef FML_FOURIER_RESAMPLING_HEADER
#define FML_FOURIER_RESAMPLING_HEADER
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/MPICounts.h>
#include <algorithm>
#include <climits>
#include <vector>

namespace FML::GRID {
    /// Reusable geometry for normalized, non-transposed FFTWGrid slab layouts.
    /// Keep only modes strictly inside the smaller mesh's Nyquist cube.
    template <int N>
    class FourierResamplingPlan {
        static_assert(N >= 2, "Fourier resampling requires the slab FFTWGrid layout (N >= 2)");
        using Grid = FFTWGrid<N>;
        using Count = FML::COMMUNICATION::Count;
        struct Block {
            ptrdiff_t source, target;
            size_t count;
        };
        int oldmesh, newmesh, tasks = FML::NTasks, rank = FML::ThisTask;
        std::array<int, 4> local_layout;
        std::vector<Block> local;
        std::vector<std::vector<Block>> send, receive;
        std::vector<Count> send_count, receive_count, max_per_step;
        std::vector<ComplexType> send_buffer, receive_buffer;
        size_t planning_bytes = 0;

        static void append(std::vector<Block> & blocks, Block block) {
            if (!blocks.empty() && blocks.back().source + ptrdiff_t(blocks.back().count) == block.source &&
                blocks.back().target + ptrdiff_t(blocks.back().count) == block.target)
                blocks.back().count += block.count;
            else
                blocks.push_back(block);
        }
        static std::array<int, 4> layout(const Grid & input, const Grid & output) {
            return {int(input.get_local_x_start()),
                    int(input.get_local_nx()),
                    int(output.get_local_x_start()),
                    int(output.get_local_nx())};
        }
        // A cursor consumes a peer's contiguous runs without per-coefficient indices on the wire.
        struct Cursor {
            size_t block = 0, used = 0;
            template <class Copy>
            void consume(const std::vector<Block> & blocks, size_t count, Copy copy) {
                size_t offset = 0;
                while (offset < count) {
                    const auto & b = blocks[block];
                    const size_t n = std::min(count - offset, b.count - used);
                    copy(b, used, offset, n);
                    offset += n;
                    used += n;
                    if (used == b.count) {
                        ++block;
                        used = 0;
                    }
                }
            }
        };

      public:
#ifdef FML_FOURIER_RESAMPLING_TEST_ALLOCATIONS
        static inline bool fail_geometry_once = false, fail_transport_once = false;
#endif
        FourierResamplingPlan(const Grid & input, const Grid & output)
            : oldmesh(input.get_nmesh()), newmesh(output.get_nmesh()), local_layout(layout(input, output)) {
            using namespace FML::COMMUNICATION;
            std::vector<std::array<int, 4>> layouts;
            collective_try([&] {
                if (oldmesh < 1 || newmesh < 1)
                    throw std::invalid_argument("[Fourier resampling] Unallocated grid");
                layouts.resize(tasks);
                send.resize(tasks);
                receive.resize(tasks);
                send_count.resize(tasks);
                receive_count.resize(tasks);
                max_per_step.resize(tasks);
            }, "Fourier remapping metadata");
#ifdef USE_MPI
            MPI_Allgather(local_layout.data(), 4, MPI_INT, layouts.data(), 4, MPI_INT, FML::communicator);
#else
            layouts[0] = local_layout;
#endif
            collective_try([&] {
#ifdef FML_FOURIER_RESAMPLING_TEST_ALLOCATIONS
                if (fail_geometry_once) {
                    fail_geometry_once = false;
                    throw std::bad_alloc();
                }
#endif
                auto intervals = [&](int axis, int mesh) {
                    std::vector<std::array<int, 3>> result;
                    result.reserve(tasks);
                    for (int t = 0; t < tasks; ++t) {
                        const int start = layouts[t][axis], width = layouts[t][axis + 1];
                        if (width < 0 || width > mesh || (width && (start < 0 || start > mesh - width)))
                            throw std::invalid_argument("[Fourier resampling] Invalid slab");
                        if (width)
                            result.push_back({start, width, t});
                    }
                    std::sort(result.begin(), result.end());
                    int end = 0;
                    for (auto slab : result) {
                        if (slab[0] != end)
                            throw std::invalid_argument("[Fourier resampling] Slabs must partition the mesh");
                        end += slab[1];
                    }
                    if (end != mesh)
                        throw std::invalid_argument("[Fourier resampling] Incomplete slab partition");
                    return result;
                };
                const auto sources = intervals(0, oldmesh), targets = intervals(2, newmesh);
                planning_bytes = layouts.capacity() * sizeof(layouts[0]) + sources.capacity() * sizeof(sources[0]) +
                                 targets.capacity() * sizeof(targets[0]);
                auto owner = [](const auto & slabs, int x) {
                    auto it = std::upper_bound(slabs.begin(), slabs.end(), x,
                                              [](int value, const auto & slab) { return value < slab[0]; });
                    return (*--it)[2];
                };
                for (int mesh : {oldmesh, newmesh}) {
                    Count size = mesh / 2 + 1;
                    for (int d = 0; d < N - 1; ++d)
                        if (!multiply_count(size, mesh, size, std::numeric_limits<ptrdiff_t>::max()))
                            throw std::overflow_error("[Fourier resampling] Mesh index product overflow");
                }
                const int kmax = (std::min(oldmesh, newmesh) - 1) / 2;
                auto visit_rows = [&](auto && visit) {
                    for (int x = 0; x < oldmesh; ++x) {
                        const int wave = x <= oldmesh / 2 ? x : x - oldmesh;
                        if (std::abs(wave) > kmax)
                            continue;
                        const int mapped = wave < 0 ? wave + newmesh : wave;
                        const int from = owner(sources, x), to = owner(targets, mapped);
                        if (from != rank && to != rank)
                            continue;
                        auto rows = [&](auto && self, int d, ptrdiff_t src, ptrdiff_t dst) -> void {
                            if (d == N - 1) {
                                visit(from, to, Block{src * (oldmesh / 2 + 1), dst * (newmesh / 2 + 1), size_t(kmax + 1)});
                                return;
                            }
                            for (int j = 0; j <= 2 * kmax; ++j) {
                                const int a = j <= kmax ? j : j + oldmesh - (2 * kmax + 1);
                                const int b = j <= kmax ? j : j + newmesh - (2 * kmax + 1);
                                self(self, d + 1, src * oldmesh + a, dst * newmesh + b);
                            }
                        };
                        rows(rows, 1, x - layouts[from][0], mapped - layouts[to][2]);
                    }
                };
                // Count before allocating geometry, so construction has no vector-growth transients.
                size_t local_rows = 0;
                visit_rows([&](int from, int to, const Block &) {
                    if (from == to) ++local_rows;
                    else {
                        if (from == rank) ++send_count[to];
                        if (to == rank) ++receive_count[from];
                    }
                });
                local.reserve(local_rows);
                if (local.capacity() != local_rows)
                    throw std::logic_error("[Fourier resampling] Unexpected local row capacity");
                for (int peer = 0; peer < tasks; ++peer) {
                    send[peer].reserve(size_t(send_count[peer]));
                    receive[peer].reserve(size_t(receive_count[peer]));
                    if (send[peer].capacity() != size_t(send_count[peer]) ||
                        receive[peer].capacity() != size_t(receive_count[peer]))
                        throw std::logic_error("[Fourier resampling] Unexpected peer row capacity");
                }
                std::fill(send_count.begin(), send_count.end(), 0);
                std::fill(receive_count.begin(), receive_count.end(), 0);
                visit_rows([&](int from, int to, const Block & block) {
                    if (from == to) append(local, block);
                    else {
                        if (from == rank) { append(send[to], block); send_count[to] += block.count; }
                        if (to == rank) { append(receive[from], block); receive_count[from] += block.count; }
                    }
                });
            }, "Fourier remapping geometry");
#ifdef USE_MPI
            for (int step = 1; step < tasks; ++step)
                max_per_step[step] = send_count[int((static_cast<long long>(rank) + step) % tasks)];
            FML::MaxArrayOverTasks(max_per_step.data(), size_t(tasks));
#endif
        }

        /// Per-rank geometry, construction and fixed-cap transport bound; no allocation or collective.
        static size_t workspace_bound(int input_mesh, int output_mesh, size_t input_slabs,
                                      size_t output_slabs, size_t max_message_bytes) {
            using namespace FML::COMMUNICATION;
            if (input_mesh < 1 || output_mesh < 1)
                throw std::invalid_argument("[Fourier resampling] Invalid planning width");
            const Count width = 2ULL * ((std::min(input_mesh, output_mesh) - 1) / 2) + 1;
            Count rows = input_slabs;
            bool good = add_count(rows, output_slabs);
            for (int d = 0; d < N - 2; ++d)
                good &= multiply_count(rows, width, rows);
            Count bytes = 0, metadata = 0;
            good &= multiply_count(rows, sizeof(Block), bytes);
            good &= multiply_count(size_t(FML::NTasks), 2 * sizeof(std::vector<Block>) + 3 * sizeof(Count) +
                                      sizeof(std::array<int, 4>) + 2 * sizeof(std::array<int, 3>), metadata);
            good &= add_count(bytes, metadata);
            good &= add_count(bytes, sizeof(FourierResamplingPlan));
            const Count buffer = FML::NTasks > 1 ? std::min(max_message_bytes, size_t(INT_MAX)) : 0;
            good &= add_count(bytes, 2 * buffer, std::numeric_limits<size_t>::max());
            if (!good)
                throw std::overflow_error("[Fourier resampling] Workspace bound overflow");
            return size_t(bytes);
        }
        static size_t serial_workspace_bound(int input_mesh, int output_mesh) {
            if (FML::NTasks != 1)
                throw std::invalid_argument("[Fourier resampling] Serial planning requires one rank");
            return workspace_bound(input_mesh, output_mesh, size_t(input_mesh), size_t(output_mesh), 0);
        }

        /// Geometry plus the larger of construction scratch or retained payload buffers.
        /// Excludes grids, allocator/reallocation transients and opaque MPI workspace.
        size_t workspace_storage_bytes() const {
            size_t bytes = sizeof(*this) + local.capacity() * sizeof(Block) +
                           (send.capacity() + receive.capacity()) * sizeof(std::vector<Block>) +
                           (send_count.capacity() + receive_count.capacity() + max_per_step.capacity()) * sizeof(Count);
            for (const auto & blocks : send)
                bytes += blocks.capacity() * sizeof(Block);
            for (const auto & blocks : receive)
                bytes += blocks.capacity() * sizeof(Block);
            return bytes +
                   std::max(planning_bytes, (send_buffer.capacity() + receive_buffer.capacity()) * sizeof(ComplexType));
        }

        /// Collective; reuse with different fields having the same meshes and local slab geometry.
        /// The two scratch buffers each use at most max_message_bytes (clamped to INT_MAX).
        void apply(const Grid & input, Grid & output, size_t max_message_bytes = 64 * 1024 * 1024) {
            using namespace FML::COMMUNICATION;
            Count bytes = std::min(max_message_bytes, size_t(INT_MAX));
            FML::MinOverTasks(&bytes);
            require_count(
                tasks == FML::NTasks && rank == FML::ThisTask && &input != &output && input.get_nmesh() == oldmesh &&
                    output.get_nmesh() == newmesh && layout(input, output) == local_layout &&
                    bytes >= sizeof(ComplexType),
                "[Fourier resampling] Incompatible plan, aliased grids or message limit smaller than a value");
            output.set_grid_status_real(false);
            for (auto i : output.get_fourier_range())
                output.set_fourier_from_index(i, 0.0);
            auto * out = output.get_fourier_grid();
            for (auto b : local)
                for (size_t i = 0; i < b.count; ++i)
                    out[b.target + ptrdiff_t(i)] = input.get_fourier_from_index(b.source + ptrdiff_t(i));
#ifdef USE_MPI
            const Count chunk = bytes / sizeof(ComplexType);
            const Count maxsend = *std::max_element(send_count.begin(), send_count.end());
            const Count maxrecv = *std::max_element(receive_count.begin(), receive_count.end());
            auto prepare = [&](auto & buffer, Count count) {
                const size_t size = size_t(std::min(chunk, count));
                // Retain storage across equal-cap calls, but honor a smaller cap without geometric over-allocation.
                if (buffer.capacity() > chunk || buffer.capacity() < size) {
                    std::vector<ComplexType>(size).swap(buffer);
                    FML::PERFORMANCE::transport_allocation(buffer.capacity() * sizeof(ComplexType));
                } else
                    buffer.resize(size);
            };
            collective_try([&] {
#ifdef FML_FOURIER_RESAMPLING_TEST_ALLOCATIONS
                if (fail_transport_once) {
                    fail_transport_once = false;
                    throw std::bad_alloc();
                }
#endif
                prepare(send_buffer, maxsend);
                prepare(receive_buffer, maxrecv);
            }, "Fourier remapping transport allocation");
            for (int step = 1; step < tasks; ++step) {
                const int dest = int((static_cast<long long>(rank) + step) % tasks);
                const int source = int((static_cast<long long>(rank) - step + tasks) % tasks);
                const Count rounds = max_per_step[step] / chunk + (max_per_step[step] % chunk != 0);
                Cursor pack, unpack;
                for (Count round = 0; round < rounds; ++round) {
                    const Count offset = round * chunk;
                    const size_t scount =
                        offset < send_count[dest] ? size_t(std::min(chunk, send_count[dest] - offset)) : 0;
                    const size_t rcount =
                        offset < receive_count[source] ? size_t(std::min(chunk, receive_count[source] - offset)) : 0;
                    pack.consume(send[dest], scount, [&](const Block & b, size_t used, size_t at, size_t n) {
                        for (size_t i = 0; i < n; ++i)
                            send_buffer[at + i] = input.get_fourier_from_index(b.source + ptrdiff_t(used + i));
                    });
                    MPI_Sendrecv(scount ? send_buffer.data() : nullptr,
                                 int(scount * sizeof(ComplexType)),
                                 MPI_BYTE,
                                 dest,
                                 1,
                                 rcount ? receive_buffer.data() : nullptr,
                                 int(rcount * sizeof(ComplexType)),
                                 MPI_BYTE,
                                 source,
                                 1,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    FML::PERFORMANCE::payload(scount * sizeof(ComplexType), rcount * sizeof(ComplexType));
                    FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportPackedBytes,
                                             scount * sizeof(ComplexType));
                    FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportUnpackedBytes,
                                             rcount * sizeof(ComplexType));
                    unpack.consume(receive[source], rcount, [&](const Block & b, size_t used, size_t at, size_t n) {
                        std::copy_n(receive_buffer.data() + at, n, out + b.target + ptrdiff_t(used));
                    });
                }
            }
#endif
        }
    };

    /// Copy normalized Fourier coefficients; excluded modes are zero, with no volume factor.
    template <int N>
    FFTWGrid<N> resize_fourier_grid(const FFTWGrid<N> & input, int nmesh, int nleft = 0, int nright = 0) {
        assert_mpi(input.get_nmesh() > 0 && nmesh > 0, "[Fourier resampling] Meshes must be allocated and positive");
        FFTWGrid<N> output(nmesh, nleft, nright);
        FourierResamplingPlan<N> plan(input, output);
        plan.apply(input, output);
        return output;
    }
} // namespace FML::GRID
#endif
