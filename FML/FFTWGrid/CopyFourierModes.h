#ifndef FML_COPY_FOURIER_MODES_H
#define FML_COPY_FOURIER_MODES_H

// =====================================================================================================================
// Copy the Fourier modes between grids of any size, decomposition and type (FFTWGrid, PencilGrid, ...): each mode is
// sent to the task that has it in the destination grid. resize_fourier_grid for the grids other than FFTWGrid (which
// has its own in FourierResampling.h)
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/ChunkedExchange.h>
#include <FML/Global/Global.h>

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace GRID {

        namespace detail {
            /// Which task has a Fourier cell of a grid (from the local Fourier boxes of all tasks, which must form a
            /// product of the ranges along each axis: true for the slabs and the pencils)
            template <int N>
            class FourierOwners {
              public:
                template <class Grid>
                explicit FourierOwners(const Grid & grid) {
                    const int n = grid.get_nmesh();
                    std::vector<int> box(2 * N);
                    for (int idim = 0; idim < N; idim++) {
                        const auto r = grid.get_local_fourier_range(idim);
                        box[2 * idim] = int(r[0]);
                        box[2 * idim + 1] = int(r[1]);
                    }
                    std::vector<int> boxes(2 * N * FML::NTasks);
#ifdef USE_MPI
                    MPI_Allgather(box.data(), 2 * N, MPI_INT, boxes.data(), 2 * N, MPI_INT, FML::communicator);
#else
                    boxes = box;
#endif
                    auto empty = [&](int task) {
                        for (int idim = 0; idim < N; idim++)
                            if (boxes[2 * N * task + 2 * idim + 1] == 0)
                                return true;
                        return false;
                    };
                    std::array<std::vector<int>, N> starts;
                    for (int idim = 0; idim < N; idim++) {
                        const int extent = idim == N - 1 ? n / 2 + 1 : n;
                        for (int task = 0; task < FML::NTasks; task++)
                            if (not empty(task))
                                starts[idim].push_back(boxes[2 * N * task + 2 * idim]);
                        std::sort(starts[idim].begin(), starts[idim].end());
                        starts[idim].erase(std::unique(starts[idim].begin(), starts[idim].end()), starts[idim].end());
                        nparts[idim] = int(starts[idim].size());
                        part[idim].resize(extent);
                        for (int c = 0; c < extent; c++)
                            part[idim][c] =
                                int(std::upper_bound(starts[idim].begin(), starts[idim].end(), c) - starts[idim].begin()) - 1;
                    }
                    size_t ntot = 1;
                    for (int idim = 0; idim < N; idim++)
                        ntot *= size_t(nparts[idim]);
                    task.assign(ntot, -1);
                    for (int t = 0; t < FML::NTasks; t++) {
                        if (empty(t))
                            continue;
                        std::array<int, N> g;
                        for (int idim = 0; idim < N; idim++)
                            g[idim] = boxes[2 * N * t + 2 * idim];
                        task[flat(g)] = t;
                    }
                    for (auto t : task)
                        assert_mpi(t >= 0, "[NUFFT::FourierOwners] The Fourier boxes of the tasks are not a product\n");
                }
                int owner(const std::array<int, N> & g) const { return task[flat(g)]; }

              private:
                size_t flat(const std::array<int, N> & g) const {
                    size_t k = 0;
                    for (int idim = 0; idim < N; idim++)
                        k = k * size_t(nparts[idim]) + size_t(part[idim][g[idim]]);
                    return k;
                }
                std::array<std::vector<int>, N> part;
                std::array<int, N> nparts{};
                std::vector<int> task;
            };
        } // namespace detail

        //================================================================================
        /// @brief Copy all Fourier modes with \f$ |m_i| \leq (\min(N_{\rm src},N_{\rm dst})-1)/2 \f$ from one grid to
        /// another of a different size, multiplying by factor(m). All other modes of dst are set to zero. Works with
        /// MPI (the modes are sent to the task that owns them in dst).
        ///
        /// @param[in] src The source grid (Fourier space)
        /// @param[out] dst The destination grid (allocated, Fourier space on output)
        /// @param[in] factor Function of the integer wavevector
        //================================================================================
        namespace detail {
            // (all_modes: also the Nyquist planes, for grids of the same size)
            template <int N, template <int> class GridSrc, template <int> class GridDst>
            void copy_fourier_modes_impl(const GridSrc<N> & src,
                                         GridDst<N> & dst,
                                         const std::function<double(const std::array<int, N> &)> & factor,
                                         bool all_modes) {
            const int nsrc = src.get_nmesh();
            const int ndst = dst.get_nmesh();
            assert_mpi(not all_modes or nsrc == ndst, "[copy_fourier_modes] all modes needs grids of the same size");
            // Keep |m| <= (min(nsrc, ndst) - 1) / 2: for even meshes this drops the Nyquist plane, for odd meshes
            // there is no Nyquist plane and all modes of the smaller grid are kept (as FourierResamplingPlan)
            const int mmax = all_modes ? nsrc : (std::min(nsrc, ndst) - 1) / 2 + 1; // keep |m| < mmax
            assert_mpi(nsrc > 0 && ndst > 0, "[NUFFT::copy_fourier_modes] Grids must be allocated");

            dst.fill_fourier_grid(0.0);
            dst.set_grid_status_real(false);

            // The modes are sent to the task that has them in dst (the grids can be decomposed differently, e.g. an
            // FFTWGrid and a PencilGrid)
            const detail::FourierOwners<N> owners(dst);
            struct Mode {
                int coord[N];
                ComplexType value;
            };
            std::vector<std::vector<Mode>> send(FML::NTasks);
            for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(src.get_local_nrows_fourier()); irow++) {
                for (auto && fourier_index : src.get_fourier_range_of_rows(irow, irow + 1)) {
                    auto m = src.get_fourier_integer_wavevector_from_index(fourier_index);
                    bool keep = true;
                    for (int idim = 0; idim < N; idim++)
                        keep = keep and std::abs(m[idim]) < mmax;
                    if (not keep)
                        continue;
                    Mode mode;
                    std::array<int, N> g;
                    for (int idim = 0; idim < N; idim++)
                        g[idim] = mode.coord[idim] = (m[idim] >= 0 or idim == N - 1) ? m[idim] : m[idim] + ndst;
                    mode.value = src.get_fourier_from_index(fourier_index) * FloatType(factor(m));
                    send[owners.owner(g)].push_back(mode);
                }
            }

            FML::COMMUNICATION::exchange_records_chunked(send, [&](const Mode * modes, size_t count) {
                for (size_t i = 0; i < count; ++i) {
                    std::array<int, N> coord;
                    for (int idim = 0; idim < N; idim++)
                        coord[idim] = modes[i].coord[idim];
                    dst.set_fourier_from_index(dst.get_index_fourier_global(coord), modes[i].value);
                }
            });
            }
        } // namespace detail

        /// Copy the Fourier modes of src into dst (any sizes, decompositions and grid types), each times factor(integer
        /// wave-vector). Only the modes strictly inside the Nyquist cube of the smaller mesh are kept (|m| <= (min(Nsrc, Ndst) -
        /// 1) / 2 along every axis: the Nyquist plane of an even mesh is dropped, an odd mesh has none); the others are
        /// zero in dst. Collective: each mode is sent to its owner in dst
        template <int N, template <int> class GridSrc, template <int> class GridDst>
        void copy_fourier_modes(const GridSrc<N> & src,
                                GridDst<N> & dst,
                                const std::function<double(const std::array<int, N> &)> & factor) {
            detail::copy_fourier_modes_impl<N>(src, dst, factor, false);
        }

        /// Copy all the Fourier modes (also the Nyquist planes) to a grid of the same size decomposed differently (e.g.
        /// an FFTWGrid to a PencilGrid): the same field
        template <int N, template <int> class GridSrc, template <int> class GridDst>
        void redistribute_fourier_grid(const GridSrc<N> & src, GridDst<N> & dst) {
            detail::copy_fourier_modes_impl<N>(src, dst, [](const std::array<int, N> &) { return 1.0; }, true);
        }

        /// Copy normalized Fourier coefficients into a grid of another size (the modes strictly inside the smaller
        /// mesh's Nyquist cube; the others are zero), decomposed as the input. For the grids other than FFTWGrid
        template <int N,
                  template <int>
                  class Grid,
                  std::enable_if_t<not std::is_same_v<Grid<N>, FFTWGrid<N>>, int> = 0>
        Grid<N> resize_fourier_grid(const Grid<N> & input, int nmesh, int nleft = 0, int nright = 0) {
            assert_mpi(input.get_nmesh() > 0 && nmesh > 0,
                       "[resize_fourier_grid] Meshes must be allocated and positive");
            auto output = make_grid_like(input, nmesh, nleft, nright);
            copy_fourier_modes<N>(input, output, [](const std::array<int, N> &) { return 1.0; });
            return output;
        }

    } // namespace GRID
} // namespace FML
#endif
