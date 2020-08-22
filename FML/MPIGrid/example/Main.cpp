#include <FML/MPIGrid/MPIGrid.h>
#include <cassert>
#include <iostream>

int main() {

    //===================================================
    //
    // The grid is split among task along the x-direction
    // Each task has a main grid and extra x-slices on the
    // left and right that is used for having the boundary
    // when doing operations. These cells are filled by
    // running communicate_boundaries()
    //          ____________________________
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |____|________________|____|
    //
    //          nleft     main grid    nright
    //
    //===================================================

    const int NDIM = 3;
    const int Nmesh = 16;
    const int nleft = 1;
    const int nright = 1;
    const bool periodic = true;
    FML::GRID::MPIGrid<NDIM, double> grid(Nmesh, periodic, nleft, nright);

    // How many of the x-slices we have on current task
    auto nx_local = grid.get_NLocal();

    // The ix value the grid starts at
    // The local grid goes from 0 -> Local_nx-1
    // which corresponds to xStartLocal -> xStartLocal + Local_nx - 1
    // in the global grid
    auto xStartLocal = grid.get_xStartLocal();

    // Total number of cells in the grid
    auto NtotLocal = grid.get_NtotLocal();

    std::cout << "Grid has " << Nmesh << " cells per dim. Task " << FML::ThisTask << " has x-slices in the range ["
              << xStartLocal << " -> " << xStartLocal + nx_local << ")\n";

    // Get a pointer to the main grid
    auto * y = grid.get_y();

    // Number of cells per slice
    long long int NperSlice = FML::power(Nmesh, NDIM - 1);

    // The start of the local ith-slices (i = [-nleft, ..., ] [0, 1, 2, .., nx_local -1], [nx_local, ... , nx_local +
    // nright - 1]) Fetch the first slice on the right (after the main grid - this does not exist unless nright >= 1)
    int i = nx_local;
    auto * y1 = &y[NperSlice * i];
    y1[0] = 0.0; // Sets the value of the cell with coord (nx_local,0,0,0,...) to 0.0

    // Loop over all cells
    for (long long int i = 0; i < NtotLocal; i++) {

        // Get coordinate of the cell in the global grid
        auto globalcoord = grid.globalcoord_from_index(i);

        // Get local coordinates of the cell
        // (i.e. coord[0] goes from 0 to local_nx-1 and the rest from 0 to nmesh-1)
        auto coord = grid.coord_from_index(i);

        // The difference between these two is just in coord[0]
        assert(globalcoord[0] == coord[0] + xStartLocal);
        for (int idim = 1; idim < NDIM; idim++)
            assert(coord[idim] == globalcoord[idim]);

        // Get the index from local coords
        // The cell with (local) coord (ix,iy,...) has index, here in 3D,
        //    index = iz + Nmesh*(iy + Nmesh * ix)
        // i.e. the last index runs first
        // NB: all methods asking for coord assume *local* coordinates
        auto index = grid.index_from_coord(coord);

        // ...which is the same as i
        assert(index == i);

        // Set the value of a cell
        double value = 1.0;
        grid.set_y(i, value);
    }

    // If we have set the grid and want to communicate the boundaries to neighboring tasks
    grid.communicate_boundaries();

    // Free up all memory in the grid (after which its just a shell)
    grid.free();
}
