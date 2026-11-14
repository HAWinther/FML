#ifndef FML_GRID_INTERFACE_H
#define FML_GRID_INTERFACE_H

// =====================================================================================================================
// The interface of a distributed Fourier grid that generic algorithms are written against, so that they work with
// FFTWGrid<N> (slabs, FFTW-MPI) and PencilGrid<N> (any PencilDomain) alike. The rules for generic code:
//
//   - Loop over the local Fourier modes by rows: for irow < get_local_nrows_fourier(), the indices of
//     get_fourier_range_of_rows(irow, irow + 1) (or use for_each_nonzero_fourier_mode below)
//   - Never decode a Fourier index or read the raw Fourier coordinates (FFTWGrid: coord[0] is the local kx;
//     PencilGrid: coord[0] is the global kx, coord[1], coord[2] the local ky, kz). Use
//     get_fourier_integer_wavevector_from_index or get_fourier_wavevector(_and_norm)_by_index
//   - The last component of the integer wave-vector is in [0, Nmesh/2] (r2c): the weight of a mode is
//     fourier_mode_multiplicity(k[N-1], Nmesh)
//   - k = 0 is the mode whose integer wave-vector is zero (not "index 0 on the task with x-start 0"): has_dc_mode
//
// And in position space:
//
//   - Make grids with make_grid<Grid>(Nmesh, left, right, domain) or make_grid_like(grid, ...) on the domain of the
//     particles (the particles are on the task of their cells), not with a constructor
//   - The local cells along an axis are get_local_range(axis) = {first, count} (global cells; an axis that is not split
//     is whole). CellOwners gives the task of any cell
//   - Never use a real-space index of one grid in another grid with different extra cells: a PencilGrid index depends
//     on them (the y-extras are inside each x-slice), an FFTWGrid index does not. Go through the coordinates. Fourier
//     indices do not depend on the extra cells
//   - Data per cell with ghost layers from the neighbours (any type, slabs and pencils): GhostedCells.h
//
// Typical generic code: template <int N, template <int> class Grid = FFTWGrid> with an FFTWGrid fast path behind
// if constexpr (is_fftwgrid_v<Grid<N>>). PencilDomain/README.md describes the pencils and their use in FML.
//
// has_fourier_grid_interface<Grid> checks the methods at compile time.
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

namespace FML {
    namespace GRID {

        /// The type T in a context where it is not deduced (as std::type_identity_t in C++20): e.g. a pointer to a grid
        /// that may be nullptr
        template <class T>
        struct non_deduced {
            using type = T;
        };
        template <class T>
        using non_deduced_t = typename non_deduced<T>::type;

        /// Is a type an FFTWGrid (of any dimension)?
        template <class Grid>
        struct is_fftwgrid : std::false_type {};
        template <int N>
        struct is_fftwgrid<FFTWGrid<N>> : std::true_type {};
        template <class Grid>
        inline constexpr bool is_fftwgrid_v = is_fftwgrid<Grid>::value;

        /// Does Grid have the Fourier-space interface (the methods generic Fourier-space code may use)?
        template <class Grid, class = void>
        struct has_fourier_grid_interface : std::false_type {};
        template <class Grid>
        struct has_fourier_grid_interface<
            Grid,
            std::void_t<
                decltype(std::declval<const Grid &>().get_nmesh()),
                decltype(std::declval<const Grid &>().get_ndim()),
                decltype(std::declval<const Grid &>().get_local_nrows_fourier()),
                decltype(std::declval<const Grid &>().get_fourier_range_of_rows(IndexIntType(0), IndexIntType(1))),
                decltype(std::declval<const Grid &>().get_fourier_from_index(IndexIntType(0))),
                decltype(std::declval<const Grid &>().get_fourier_integer_wavevector_from_index(IndexIntType(0))),
                decltype(std::declval<Grid &>().set_fourier_from_index(IndexIntType(0), ComplexType{}))>>
            : std::true_type {};

        /// @brief Loop over the local Fourier modes of a grid except k = 0, calling f(fourier_index, kvec, |k|,
        /// weight) with kvec = 2 pi n (multiply by 1/Boxsize for physical units) and weight = 2 for the modes that also
        /// represent -k (0 < k_last < Nmesh/2) and 1 otherwise. OpenMP over the rows (f must be thread safe). For
        /// FFTWGrid<3> a fast path that does not decode the indices (the same values in the same order)
        template <int N, template <int> class Grid, class F>
        void for_each_nonzero_fourier_mode(const Grid<N> & grid, F && f) {
            static_assert(has_fourier_grid_interface<Grid<N>>::value,
                          "for_each_nonzero_fourier_mode: the grid does not have the Fourier-space interface");
            const int Nmesh = grid.get_nmesh();
            const double twopi = 2.0 * M_PI;

            if constexpr (N == 3 and std::is_same_v<Grid<N>, FFTWGrid<3>>) {
                const int Local_nx = int(grid.get_local_nx());
                const int Local_x_start = int(grid.get_local_x_start());
                const int nover2 = Nmesh / 2;
                const int nover2plus1 = Nmesh / 2 + 1;
                // OpenMP over the rows (islice, iy): a task can have fewer slices than threads
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (int iy = 0; iy < Nmesh; iy++) {
                        std::array<double, 3> kvec;
                        const int ix = islice + Local_x_start;
                        kvec[0] = twopi * double(ix <= nover2 ? ix : ix - Nmesh);
                        kvec[1] = twopi * double(iy <= nover2 ? iy : iy - Nmesh);
                        size_t fourier_index = (size_t(islice) * Nmesh + iy) * nover2plus1;
                        for (int iz = 0; iz < nover2plus1; iz++, fourier_index++) {
                            if (Local_x_start == 0 and fourier_index == 0)
                                continue; // DC mode (k=0)
                            kvec[2] = twopi * double(iz <= nover2 ? iz : iz - Nmesh);
                            const double kmag = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);
                            const double weight = fourier_mode_multiplicity(iz, Nmesh);
                            f(fourier_index, kvec, kmag, weight);
                        }
                    }
                }
            } else {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexIntType irow = 0; irow < IndexIntType(grid.get_local_nrows_fourier()); irow++) {
                    for (auto && fourier_index : grid.get_fourier_range_of_rows(irow, irow + 1)) {
                        const auto k = grid.get_fourier_integer_wavevector_from_index(fourier_index);
                        std::array<double, N> kvec;
                        double kmag2 = 0.0;
                        bool zero = true;
                        for (int idim = 0; idim < N; idim++) {
                            zero = zero and k[idim] == 0;
                            kvec[idim] = twopi * double(k[idim]);
                            kmag2 += kvec[idim] * kvec[idim];
                        }
                        if (zero)
                            continue; // DC mode (k=0)
                        f(fourier_index, kvec, std::sqrt(kmag2), double(fourier_mode_multiplicity(k[N - 1], Nmesh)));
                    }
                }
            }
        }

        namespace detail {
            template <class Grid, class = void>
            struct has_get_domain : std::false_type {};
            template <class Grid>
            struct has_get_domain<Grid, std::void_t<decltype(std::declval<const Grid &>().get_domain())>>
                : std::true_type {};
        } // namespace detail

        /// The domain of the particles that go with a grid (they are on the task of their cells): a grid with a domain
        /// (get_domain(), e.g. PencilGrid): its particle_domain(); FFTWGrid: the global slabs FML::domain
        template <class Grid>
        FML::Domain particle_domain_of(const Grid & grid) {
            if constexpr (detail::has_get_domain<Grid>::value)
                return grid.get_domain().particle_domain();
            else
                return FML::domain;
        }

        /// A grid of another size decomposed the same way (NUFFT: the fine grid). FFTWGrid: the FFTW slabs (see
        /// fits_grid_like). Other grids have their own (e.g. PencilGrid.h), found by argument dependent lookup
        template <int N>
        FFTWGrid<N> make_grid_like(const FFTWGrid<N> &, int Nmesh, int n_extra_left, int n_extra_right) {
            return FFTWGrid<N>(Nmesh, n_extra_left, n_extra_right);
        }
        /// Do grids of Nmesh cells have the decomposition of the particles of the grid? FFTWGrid: the slabs of
        /// FML::domain need Nmesh divisible by the number of tasks
        template <int N>
        bool fits_grid_like(const FFTWGrid<N> &, int Nmesh) {
            return Nmesh % FML::NTasks == 0;
        }

        /// Does this task have the k = 0 mode? It is then the local Fourier index 0 (FFTWGrid: the task with x-start 0;
        /// PencilGrid: the task with ky-start and kz-start 0). Generic code: has_dc_mode(grid) and fourier_index == 0
        template <class Grid>
        bool has_dc_mode(const Grid & grid) {
            if (grid.get_local_nrows_fourier() == 0)
                return false;
            const auto k = grid.get_fourier_integer_wavevector_from_index(0);
            for (auto ki : k)
                if (ki != 0)
                    return false;
            return true;
        }

        /// The type of a grid (to make one: make_grid)
        template <class Grid>
        struct grid_type_tag {};

        /// An FFTWGrid: the FFTW slabs (the domain is not used: the slabs of FML::domain are those of FFTW for Nmesh
        /// divisible by the number of tasks)
        template <int N>
        FFTWGrid<N> make_grid_on_domain(grid_type_tag<FFTWGrid<N>>, int Nmesh, int n_extra_left, int n_extra_right,
                                        const FML::Domain & domain) {
            assert_mpi(domain.is_slab(), "[make_grid] An FFTWGrid is decomposed in slabs: the domain must be a SlabDomain");
            return FFTWGrid<N>(Nmesh, n_extra_left, n_extra_right);
        }

        /// A grid of type Grid (FFTWGrid, PencilGrid, ...) decomposed as a domain (default: the fiducial FML::domain).
        /// Functions that make grids for particles use the domain of the particles. Other grid types provide
        /// make_grid_on_domain (found by argument dependent lookup, e.g. in PencilGrid.h)
        template <class Grid>
        Grid make_grid(int Nmesh, int n_extra_left = 0, int n_extra_right = 0, const FML::Domain & domain = FML::domain) {
            return make_grid_on_domain(grid_type_tag<Grid>{}, Nmesh, n_extra_left, n_extra_right, domain);
        }

        /// The task that has a cell of a distributed grid (FFTWGrid slabs, PencilGrid pencils: the local boxes are a
        /// product of x-blocks and, for pencils, y-blocks). Collective constructor; owner(global coordinates)
        template <int N>
        class CellOwners {
            std::vector<int> xstarts, ystarts, table;
            static size_t block(const std::vector<int> & starts, int c) {
                return size_t(std::upper_bound(starts.begin(), starts.end(), c) - starts.begin()) - 1;
            }

          public:
            /// An empty table
            CellOwners() = default;
            /// The table of the local boxes of all tasks of a grid (collective: an Allgather of the local ranges)
            template <class Grid>
            explicit CellOwners(const Grid & grid) {
                long long mine[4] = {(long long)(grid.get_local_range(0)[0]), (long long)(grid.get_local_range(0)[1]), 0, 1};
                if constexpr (N >= 2) {
                    mine[2] = (long long)(grid.get_local_range(1)[0]);
                    mine[3] = (long long)(grid.get_local_range(1)[1]);
                }
                std::vector<long long> all(4 * size_t(FML::NTasks));
#ifdef USE_MPI
                MPI_Allgather(mine, 4, MPI_LONG_LONG, all.data(), 4, MPI_LONG_LONG, FML::communicator);
#else
                std::copy(mine, mine + 4, all.begin());
#endif
                auto has_cells = [&](int t) { return all[4 * t + 1] > 0 and all[4 * t + 3] > 0; };
                for (int t = 0; t < FML::NTasks; ++t)
                    if (has_cells(t)) {
                        xstarts.push_back(int(all[4 * t]));
                        ystarts.push_back(int(all[4 * t + 2]));
                    }
                for (auto * v : {&xstarts, &ystarts}) {
                    std::sort(v->begin(), v->end());
                    v->erase(std::unique(v->begin(), v->end()), v->end());
                }
                table.assign(xstarts.size() * ystarts.size(), -1);
                for (int t = 0; t < FML::NTasks; ++t)
                    if (has_cells(t))
                        table[block(xstarts, int(all[4 * t])) * ystarts.size() + block(ystarts, int(all[4 * t + 2]))] =
                            t;
            }
            /// The task with the cell (global coordinates in [0, Nmesh))
            int owner(const std::array<int, N> & c) const {
                const int y = N >= 2 ? c[N >= 2 ? 1 : 0] : 0;
                return table[block(xstarts, c[0]) * ystarts.size() + block(ystarts, y)];
            }
        };

    } // namespace GRID
} // namespace FML

#endif
