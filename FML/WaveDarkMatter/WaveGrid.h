#ifndef FML_WAVE_GRID_H
#define FML_WAVE_GRID_H

// The grids of the wave solver: any complex grid with complex-to-complex transforms and the interface of
// FML::GRID::CFFTWGrid (FFTW-MPI slabs, the default) or FML::GRID::CPencilGrid (pencils; include
// <FML/PencilDomain/CPencilGrid.h> and use SolverT<FML::GRID::CPencilGrid<3>>). The solver's grids have no extra
// cells, so the local cells in position space have the indices [0, get_ntot_real()) and those in Fourier space
// [0, get_ntot_fourier()) (the two layouts and counts differ on pencils).

#include <FML/FFTWGrid/CFFTWGrid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#ifndef USE_FFTW
#error "WaveDarkMatter requires USE_FFTW"
#endif

namespace FML::WAVEDM {
    using Real = FML::GRID::FloatType;
    using Complex = FML::GRID::ComplexType;

    namespace detail {
        inline void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        template <class T>
        inline void same(T value, const char * message) {
            T lo = value, hi = value;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, message);
        }

        template <class Grid>
        inline constexpr bool is_slab_grid_v = std::is_same_v<Grid, FML::GRID::CFFTWGrid<3>>;

        /// A wave grid of mesh n without extra cells: CFFTWGrid slabs (any n: FFTW may leave tasks without slabs).
        /// Other grids (pencils): the fiducial domain FML::domain if its edges lie on cells, otherwise the grid's
        /// default decomposition for n (pencils: the process grid of FML::pencil_domain() with the edges on cells),
        /// so any n works as for slabs
        template <class Grid>
        Grid make_grid(int n) {
            if constexpr (is_slab_grid_v<Grid>) {
                return Grid(n);
            } else {
                bool on_cells = false;
                if (FML::domain.holds<FML::PencilDomain>()) {
                    on_cells = true;
                    const auto & pd = FML::domain.get<FML::PencilDomain>();
                    for (const auto * edges : {&pd.x_edges(), &pd.y_edges()})
                        for (double e : *edges)
                            on_cells &= std::abs(e * n - std::round(e * n)) < 1e-6;
                }
                return on_cells ? FML::GRID::make_grid<Grid>(n, 0, 0, FML::domain) : Grid(n);
            }
        }

        /// Copy the local values and the status of a grid with the same layout (no reallocation)
        template <class Grid>
        void copy_values(Grid & dst, const Grid & src) {
            require(dst.get_nmesh() == src.get_nmesh() && dst.get_ntot_real() == src.get_ntot_real() &&
                        dst.get_ntot_fourier() == src.get_ntot_fourier() &&
                        dst.get_local_range(0) == src.get_local_range(0) &&
                        dst.get_local_range(1) == src.get_local_range(1),
                    "[WaveDM] Complex layouts differ");
            const ptrdiff_t n = std::max(src.get_ntot_real(), src.get_ntot_fourier());
            std::copy_n(src.get_fourier_grid(), n, dst.get_fourier_grid());
            dst.set_grid_status_real(src.get_grid_status_real());
        }

        /// The global integer coordinates of the local position-space cell with index i
        template <class Grid>
        std::array<int, 3> global_coordinate(const Grid & grid, size_t i) {
            auto c = grid.get_coord_from_index(FML::IndexIntType(i));
            for (int d = 0; d < 3; d++)
                c[d] += int(grid.get_local_range(d)[0]);
            return c;
        }
    } // namespace detail
} // namespace FML::WAVEDM
#endif
