#ifndef CONVERTGRIDS_HEADER
#define CONVERTGRIDS_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIGrid/MPIGrid.h>

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


    std::cout << nleft << " " << nright << "\n";

    to_grid = FML::GRID::FFTWGrid<N>(Nmesh, nleft, nright);

    auto Ntot = from_grid.get_NtotLocal();
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

    for (auto real_index : from_grid.get_real_range()) {
        auto coord = from_grid.get_coord_from_index(real_index);
        auto value = from_grid.get_real_from_index(real_index);
        to_grid.set_y(coord, value);
    }
}

#endif
