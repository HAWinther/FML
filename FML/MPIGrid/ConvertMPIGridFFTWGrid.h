#ifndef CONVERTGRIDS_HEADER
#define CONVERTGRIDS_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/MPIGrid/MPIGrid.h>

#include <optional>
#include <type_traits>

//=========================================================================
// Method to convert data in a MPIGrid to a FFTWGrid for the case where T is
// float or double.
// NB: we only copy the main grid and we assume its a real grid
//=========================================================================

template <int N, class T>
void ConvertToFFTWGrid(FML::GRID::MPIGrid<N, T> & from_grid, FML::GRID::FFTWGrid<N> & to_grid) {

    auto Nmesh = from_grid.get_N();
    auto nleft = from_grid.get_n_extra_slices_left();
    auto nright = from_grid.get_n_extra_slices_right();

    to_grid = FML::GRID::FFTWGrid<N>(Nmesh, nleft, nright);

    auto Ntot = from_grid.get_NtotLocal();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (long long int index = 0; index < Ntot; index++) {
        auto coord = from_grid.coord_from_index(index);
        auto value = from_grid.get_y(index);
        to_grid.set_real(coord, value);
    }
}

template <int N, class T>
void ConvertToMPIGrid(FML::GRID::FFTWGrid<N> & from_grid, FML::GRID::MPIGrid<N, T> & to_grid) {

    auto Nmesh = from_grid.get_nmesh();
    auto nleft = from_grid.get_n_extra_slices_left();
    auto nright = from_grid.get_n_extra_slices_right();

    to_grid = FML::GRID::MPIGrid<N, T>(Nmesh, true, nleft, nright);

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (FML::IndexIntType irow = 0; irow < from_grid.get_local_nrows(); irow++) {
        for (auto && real_index : from_grid.get_real_range_of_rows(irow, irow + 1)) {
            auto coord = from_grid.get_coord_from_index(real_index);
            auto value = from_grid.get_real_from_index(real_index);
            to_grid.set_y(coord, value);
        }
    }
}

//=========================================================================
// The same for any grid with the interface (e.g. PencilGrid): the MPIGrid is made on the domain of the grid (its
// pencils) so that the local cells are the same. FFTWGrid uses the functions above (the fiducial MPIGrid slabs)
//=========================================================================

/// The domain an MPIGrid (and the multigrid solver) must use to have the cells of a grid: none for an FFTWGrid (the
/// fiducial MPIGrid slabs are those of FFTW), the pencils for a grid with a domain (e.g. PencilGrid)
template <class Grid>
std::optional<FML::Domain> multigrid_domain_of(const Grid & grid) {
    if constexpr (FML::GRID::detail::has_get_domain<Grid>::value)
        return FML::Domain(grid.get_domain());
    else
        return std::nullopt;
}

/// Copy a grid in position space (any grid type) into an MPIGrid with the same local cells (made on
/// multigrid_domain_of(grid))
template <int N, class T, template <int> class Grid>
void ConvertToMPIGrid(Grid<N> & from_grid, FML::GRID::MPIGrid<N, T> & to_grid) {
    if constexpr (std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>) {
        ConvertToMPIGrid<N, T>(static_cast<FML::GRID::FFTWGrid<N> &>(from_grid), to_grid);
    } else {
        const auto Nmesh = from_grid.get_nmesh();
        const auto nleft = from_grid.get_n_extra_slices_left();
        const auto nright = from_grid.get_n_extra_slices_right();
        to_grid = FML::GRID::MPIGrid<N, T>(Nmesh, true, *multigrid_domain_of(from_grid), nleft, nright);
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(from_grid.get_local_nrows()); irow++) {
            for (auto && real_index : from_grid.get_real_range_of_rows(irow, irow + 1)) {
                auto coord = from_grid.get_coord_from_index(real_index);
                to_grid.set_y(coord, from_grid.get_real_from_index(real_index));
            }
        }
    }
}

/// Copy the main cells of an MPIGrid to a grid (FFTWGrid: made anew as ConvertToFFTWGrid; other grids: made on the
/// domain of the MPIGrid unless already of that size)
template <int N, class T, template <int> class Grid>
void ConvertToGrid(FML::GRID::MPIGrid<N, T> & from_grid, Grid<N> & to_grid) {
    if constexpr (std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>) {
        ConvertToFFTWGrid<N, T>(from_grid, to_grid);
    } else {
        const auto Nmesh = from_grid.get_N();
        assert_mpi(from_grid.get_domain().has_value(), "[ConvertToGrid] The MPIGrid has no domain\n");
        if (not to_grid or to_grid.get_nmesh() != Nmesh)
            to_grid = Grid<N>(Nmesh,
                              from_grid.get_n_extra_slices_left(),
                              from_grid.get_n_extra_slices_right(),
                              from_grid.get_domain()->template get<FML::PencilDomain>());
        const auto Ntot = from_grid.get_NtotLocal();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (long long int index = 0; index < Ntot; index++)
            to_grid.set_real(from_grid.coord_from_index(index), from_grid.get_y(index));
        to_grid.set_grid_status_real(true);
    }
}

#endif
