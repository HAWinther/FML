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
    template <int N> class FourierResamplingPlan {
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

        static void append(std::vector<Block> &blocks, Block block) {
            if (!blocks.empty() && blocks.back().source + ptrdiff_t(blocks.back().count) == block.source &&
                blocks.back().target + ptrdiff_t(blocks.back().count) == block.target)
                blocks.back().count += block.count;
            else
                blocks.push_back(block);
        }
        static std::array<int, 4> layout(const Grid &input, const Grid &output) {
            return {int(input.get_local_x_start()), int(input.get_local_nx()), int(output.get_local_x_start()),
                    int(output.get_local_nx())};
        }
        // A cursor consumes a peer's contiguous runs without per-coefficient indices on the wire.
        struct Cursor {
            size_t block = 0, used = 0;
            template <class Copy> void consume(const std::vector<Block> &blocks, size_t count, Copy copy) {
                size_t offset = 0;
                while (offset < count) {
                    const auto &b = blocks[block];
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
        FourierResamplingPlan(const Grid &input, const Grid &output)
            : oldmesh(input.get_nmesh()), newmesh(output.get_nmesh()), local_layout(layout(input, output)), send(tasks),
              receive(tasks), send_count(tasks), receive_count(tasks), max_per_step(tasks) {
            using namespace FML::COMMUNICATION;
            require_count(oldmesh > 0 && newmesh > 0, "[Fourier resampling] Meshes must be allocated and positive");
            std::vector<std::array<int, 4>> layouts(tasks);
#ifdef USE_MPI
            MPI_Allgather(local_layout.data(), 4, MPI_INT, layouts.data(), 4, MPI_INT, FML::communicator);
#else
            layouts[0] = local_layout;
#endif
            // Ignore empty slabs when sorting: FFTW may give them an arbitrary start.
            auto intervals = [&](int axis, int mesh) {
                std::vector<std::array<int, 3>> result;
                bool valid = true;
                for (int t = 0; t < tasks; ++t) {
                    const int start = layouts[t][axis], width = layouts[t][axis + 1];
                    valid &= width >= 0 && width <= mesh && (width == 0 || (start >= 0 && start <= mesh - width));
                    if (width > 0)
                        result.push_back({start, width, t});
                }
                std::sort(result.begin(), result.end());
                int end = 0;
                for (auto slab : result) {
                    valid &= slab[0] == end;
                    end = slab[0] + slab[1];
                }
                require_count(valid && end == mesh, "[Fourier resampling] Slabs must partition the mesh");
                return result;
            };
            const auto sources = intervals(0, oldmesh), targets = intervals(2, newmesh);
            planning_bytes = layouts.capacity() * sizeof(layouts[0]) + sources.capacity() * sizeof(sources[0]) +
                             targets.capacity() * sizeof(targets[0]);
            auto owner = [](const auto &slabs, int x) {
                auto it = std::upper_bound(slabs.begin(), slabs.end(), x,
                                           [](int value, const auto &slab) { return value < slab[0]; });
                return (*--it)[2];
            };
            // Bound flattened index arithmetic independently of MPI's smaller message-count limit.
            bool valid = true;
            for (int mesh : {oldmesh, newmesh}) {
                Count size = mesh / 2 + 1;
                for (int d = 0; d < N - 1; ++d)
                    valid &= multiply_count(size, mesh, size, std::numeric_limits<ptrdiff_t>::max());
            }
            require_count(valid, "[Fourier resampling] Mesh index product overflow");
            const int kmax = (std::min(oldmesh, newmesh) - 1) / 2;
            // Both endpoints construct the same source-ordered runs, so target indices need not travel.
            for (int x = 0; x < oldmesh; ++x) {
                const int wave = x <= oldmesh / 2 ? x : x - oldmesh;
                if (std::abs(wave) > kmax)
                    continue;
                const int mapped = wave < 0 ? wave + newmesh : wave;
                const int from = owner(sources, x), to = owner(targets, mapped);
                if (from != rank && to != rank)
                    continue;
                auto rows = [&](auto &&self, int d, ptrdiff_t src, ptrdiff_t dst) -> void {
                    if (d == N - 1) {
                        Block block{src * (oldmesh / 2 + 1), dst * (newmesh / 2 + 1), size_t(kmax + 1)};
                        if (from == to)
                            append(local, block);
                        else {
                            if (from == rank) {
                                append(send[to], block);
                                send_count[to] += block.count;
                            }
                            if (to == rank) {
                                append(receive[from], block);
                                receive_count[from] += block.count;
                            }
                        }
                        return;
                    }
                    for (int j = 0; j <= 2 * kmax; ++j) {
                        const int a = j <= kmax ? j : j + (oldmesh - (2 * kmax + 1));
                        const int b = j <= kmax ? j : j + (newmesh - (2 * kmax + 1));
                        self(self, d + 1, src * oldmesh + a, dst * newmesh + b);
                    }
                };
                rows(rows, 1, x - layouts[from][0], mapped - layouts[to][2]);
            }
#ifdef USE_MPI
            // One vector reduction for the whole plan, instead of a reduction at every peer step/call.
            for (int step = 1; step < tasks; ++step)
                max_per_step[step] = send_count[int((static_cast<long long>(rank) + step) % tasks)];
            MPI_Allreduce(MPI_IN_PLACE, max_per_step.data(), tasks, MPI_UNSIGNED_LONG_LONG, MPI_MAX, FML::communicator);
#endif
        }

        /// Geometry plus the larger of construction scratch or retained payload buffers.
        /// Excludes grids, allocator/reallocation transients and opaque MPI workspace.
        size_t workspace_storage_bytes() const {
            size_t bytes = sizeof(*this) + local.capacity() * sizeof(Block) +
                           (send.capacity() + receive.capacity()) * sizeof(std::vector<Block>) +
                           (send_count.capacity() + receive_count.capacity() + max_per_step.capacity()) * sizeof(Count);
            for (const auto &blocks : send)
                bytes += blocks.capacity() * sizeof(Block);
            for (const auto &blocks : receive)
                bytes += blocks.capacity() * sizeof(Block);
            return bytes +
                   std::max(planning_bytes, (send_buffer.capacity() + receive_buffer.capacity()) * sizeof(ComplexType));
        }

        /// Collective; reuse with different fields having the same meshes and local slab geometry.
        /// The two scratch buffers each use at most max_message_bytes (clamped to INT_MAX).
        void apply(const Grid &input, Grid &output, size_t max_message_bytes = 64 * 1024 * 1024) {
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
            auto *out = output.get_fourier_grid();
            for (auto b : local)
                for (size_t i = 0; i < b.count; ++i)
                    out[b.target + ptrdiff_t(i)] = input.get_fourier_from_index(b.source + ptrdiff_t(i));
#ifdef USE_MPI
            const Count chunk = bytes / sizeof(ComplexType);
            const Count maxsend = *std::max_element(send_count.begin(), send_count.end());
            const Count maxrecv = *std::max_element(receive_count.begin(), receive_count.end());
            auto prepare = [&](auto &buffer, Count count) {
                const size_t size = size_t(std::min(chunk, count));
                // Retain storage across equal-cap calls, but honor a smaller cap without geometric over-allocation.
                if (buffer.capacity() > chunk || buffer.capacity() < size) {
                    std::vector<ComplexType>(size).swap(buffer);
                    FML::PERFORMANCE::transport_allocation(buffer.capacity() * sizeof(ComplexType));
                } else
                    buffer.resize(size);
            };
            prepare(send_buffer, maxsend);
            prepare(receive_buffer, maxrecv);
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
                    pack.consume(send[dest], scount, [&](const Block &b, size_t used, size_t at, size_t n) {
                        for (size_t i = 0; i < n; ++i)
                            send_buffer[at + i] = input.get_fourier_from_index(b.source + ptrdiff_t(used + i));
                    });
                    MPI_Sendrecv(scount ? send_buffer.data() : nullptr, int(scount * sizeof(ComplexType)), MPI_BYTE,
                                 dest, 1, rcount ? receive_buffer.data() : nullptr, int(rcount * sizeof(ComplexType)),
                                 MPI_BYTE, source, 1, FML::communicator, MPI_STATUS_IGNORE);
                    FML::PERFORMANCE::payload(scount * sizeof(ComplexType), rcount * sizeof(ComplexType));
                    FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportPackedBytes,
                                             scount * sizeof(ComplexType));
                    FML::PERFORMANCE::record(FML::PERFORMANCE::Counter::TransportUnpackedBytes,
                                             rcount * sizeof(ComplexType));
                    unpack.consume(receive[source], rcount, [&](const Block &b, size_t used, size_t at, size_t n) {
                        std::copy_n(receive_buffer.data() + at, n, out + b.target + ptrdiff_t(used));
                    });
                }
            }
#endif
        }
    };

    /// Copy normalized Fourier coefficients; excluded modes are zero, with no volume factor.
    template <int N>
    FFTWGrid<N> resize_fourier_grid(const FFTWGrid<N> &input, int nmesh, int nleft = 0, int nright = 0) {
        assert_mpi(input.get_nmesh() > 0 && nmesh > 0, "[Fourier resampling] Meshes must be allocated and positive");
        FFTWGrid<N> output(nmesh, nleft, nright);
        FourierResamplingPlan<N> plan(input, output);
        plan.apply(input, output);
        return output;
    }
} // namespace FML::GRID
#endif
