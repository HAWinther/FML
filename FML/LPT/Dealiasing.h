#ifndef FML_LPT_DEALIASING_HEADER
#define FML_LPT_DEALIASING_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <algorithm>
#include <climits>
#include <limits>
#include <numeric>
#include <vector>

namespace FML::COSMOLOGY::LPT {
    /// Padding evaluates nonlinear products on a larger mesh and crops the result.
    /// None preserves the original, aliased calculation. Padding excludes input and
    /// output Nyquist planes (for even meshes); all strictly sub-Nyquist modes remain.
    enum class Dealiasing { None, Padding };

    namespace detail {
        inline int padded_lpt_mesh(int nmesh, int order) {
            // M >= (order+1) N/2 prevents a degree-order product from wrapping
            // into |k_i| < N/2. Keep the FFT mesh even and divisible by MPI tasks.
            const long long multiple = std::lcm(2, FML::NTasks);
            const long long minimum = ((order + 1LL) * nmesh + 1) / 2;
            const long long padded = ((minimum + multiple - 1) / multiple) * multiple;
            assert_mpi(padded <= INT_MAX, "[LPT padding] Mesh size exceeds int range");
            return int(padded);
        }

        /// Copy normalized Fourier coefficients between meshes without a real-space
        /// interpolation. No volume factor is needed: FFTWGrid normalizes its forward
        /// FFT. Only modes strictly inside the smaller mesh's Nyquist cube survive.
        /// MPI exchanges only retained modes, without replicating the global grid.
        template <int N>
        FML::GRID::FFTWGrid<N>
        resize_lpt_fourier(const FML::GRID::FFTWGrid<N> & input, int nmesh, int nleft = 0, int nright = 0) {
            assert_mpi(input.get_nmesh() > 0 && nmesh > 0, "[LPT padding] Meshes must be allocated and positive");
            using Grid = FML::GRID::FFTWGrid<N>;
            using Complex = FML::GRID::ComplexType;
            Grid output(nmesh, nleft, nright);
            output.set_grid_status_real(false);
            for (auto index : output.get_fourier_range())
                output.set_fourier_from_index(index, 0.0);
            const int oldmesh = input.get_nmesh();
            const int cutoff_mesh = std::min(oldmesh, nmesh);
            std::vector<int> starts(FML::NTasks), widths(FML::NTasks);
            int start = output.get_local_x_start(), width = output.get_local_nx();
#ifdef USE_MPI
            MPI_Allgather(&start, 1, MPI_INT, starts.data(), 1, MPI_INT, FML::communicator);
            MPI_Allgather(&width, 1, MPI_INT, widths.data(), 1, MPI_INT, FML::communicator);
#else
            starts[0] = start;
            widths[0] = width;
#endif
            // Indices refer to the receiver's local grid. Sending records avoids
            // assumptions about FFTW's distribution or a matching slab layout.
            struct Mode {
                ptrdiff_t index;
                Complex value;
            };
            std::vector<std::vector<Mode>> outgoing(FML::NTasks);
            for (auto index : input.get_fourier_range()) {
                auto coord = input.get_fourier_coord_from_index(index);
                coord[0] += input.get_local_x_start();
                bool keep = true;
                for (int d = 0; d < N; d++) {
                    int wave = coord[d];
                    if (d < N - 1 && wave > oldmesh / 2)
                        wave -= oldmesh;
                    if (2LL * std::abs(wave) >= cutoff_mesh)
                        keep = false;
                    coord[d] = wave < 0 ? wave + nmesh : wave;
                }
                if (!keep)
                    continue;
                int owner = int(std::upper_bound(starts.begin(), starts.end(), coord[0]) - starts.begin()) - 1;
                assert_mpi(owner >= 0 && coord[0] < starts[owner] + widths[owner],
                           "[LPT padding] Cannot locate Fourier slab");
                coord[0] -= starts[owner];
                ptrdiff_t target = coord[0];
                for (int d = 1; d < N; d++)
                    target = target * (d == N - 1 ? nmesh / 2 + 1 : nmesh) + coord[d];
                auto value = input.get_fourier_from_index(index);
                if (owner == FML::ThisTask)
                    output.set_fourier_from_index(target, value);
                else
                    outgoing[owner].push_back({target, value});
            }
#ifdef USE_MPI
            // A ring schedule visits every source/destination pair. Chunking keeps
            // MPI byte counts representable even for multi-gigabyte messages.
            constexpr unsigned long long chunk = INT_MAX / sizeof(Mode);
            for (int step = 1; step < FML::NTasks; step++) {
                const int dest = (FML::ThisTask + step) % FML::NTasks;
                const int source = (FML::ThisTask - step + FML::NTasks) % FML::NTasks;
                auto & send = outgoing[dest];
                unsigned long long nsend = send.size(), nrecv = 0;
                MPI_Sendrecv(&nsend,
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             dest,
                             0,
                             &nrecv,
                             1,
                             MPI_UNSIGNED_LONG_LONG,
                             source,
                             0,
                             FML::communicator,
                             MPI_STATUS_IGNORE);
                unsigned long long rounds = (std::max(nsend, nrecv) + chunk - 1) / chunk;
                MPI_Allreduce(MPI_IN_PLACE, &rounds, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, FML::communicator);
                std::vector<Mode> receive(std::min(chunk, nrecv));
                for (unsigned long long round = 0; round < rounds; round++) {
                    const auto offset = round * chunk;
                    const auto scount = offset < nsend ? std::min(chunk, nsend - offset) : 0;
                    const auto rcount = offset < nrecv ? std::min(chunk, nrecv - offset) : 0;
                    MPI_Sendrecv(scount ? send.data() + offset : nullptr,
                                 int(scount * sizeof(Mode)),
                                 MPI_BYTE,
                                 dest,
                                 1,
                                 rcount ? receive.data() : nullptr,
                                 int(rcount * sizeof(Mode)),
                                 MPI_BYTE,
                                 source,
                                 1,
                                 FML::communicator,
                                 MPI_STATUS_IGNORE);
                    for (size_t i = 0; i < rcount; i++)
                        output.set_fourier_from_index(receive[i].index, receive[i].value);
                }
                std::vector<Mode>().swap(send);
            }
#endif
            return output;
        }
    } // namespace detail
} // namespace FML::COSMOLOGY::LPT
#endif
