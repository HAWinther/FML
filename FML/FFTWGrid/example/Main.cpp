#include <FML/FFTWGrid/FFTWGrid.h>
#include <numeric>

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

int main() {

    //===================================================
    //
    // The grid is split among task along the x-direction
    // Each task has a main grid and extra x-slices on the
    // left and right that is used for having the boundary
    // when doing operations. These cells are filled by
    // running communicate_boundaries()
    //
    //                Local gridslices
    //       ...____________________________...
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //          |    |                |    |
    //       ...|____|________________|____|...
    //
    //          nleft     main grid    nright
    //
    //===================================================

    const int Ndim = 3;
    const int Nmesh = 16;
    const int nleft = 0;
    const int nright = 0;
    FFTWGrid<Ndim> grid(Nmesh, nleft, nright);

    // Set the status of the grid. This is just for your convenience
    // If you try to use real methods on the fourier grid it will trigger
    // a warning if compiled with DEBUG_FFTWGRID so can help find bugs more easily.
    // The status changes every time you call do a FFT with the fftw_r2c / c2r routines.
    grid.set_grid_status_real(true);

    // Fill it to zero
    grid.fill_real_grid(0.0);

    // ...use the 'wrong' method (in this example it does not matter, but will show warning)
    grid.fill_fourier_grid(0.0);

    // Print some info about the grid
    grid.info();

    // The solution of D^2f = source with source given below
    const int n = 2;
    auto solution = [&](std::array<double, Ndim> & pos) -> double {
        auto value = 0.0;
        for (auto & x : pos)
            value += sin(n * 2 * M_PI * x);
        return value;
    };
    // Some function we want to fill the grid with
    auto source = [&](std::array<double, Ndim> & pos) -> double {
        return -(n * 2 * M_PI) * (n * 2 * M_PI) * solution(pos);
    };

    //==============================================
    // Loop over all real cells and set the values
    //==============================================
    for (auto && index : grid.get_real_range()) {
        // The (local) coordinates of the cell
        auto coord = grid.get_coord_from_index(index);

        // The (global) position of the cell
        auto pos = grid.get_real_position(coord);

        // Compute the source from the position
        auto value = source(pos);

        // Set the value to the grid
        grid.set_real_from_index(index, value);
    }

    // Fourier transform it
    grid.fftw_r2c();

    // Test saving and loading the grid
    // grid.dump_to_file("output_grid");
    // grid.fill_real_grid(0.0);
    // grid.load_from_file("output_grid");
    // grid.info();

    //==============================================
    // Loop over complex grid and print some values
    // (Only ki = +-2 should be non-zero)
    //==============================================
    if (FML::ThisTask == 0)
        std::cout << "Print non-zero fourier coefficients (Expecting only |k|=" << n << "):\n";
    for (auto && index : grid.get_fourier_range()) {
        // The (global) Fourier wave-vector of the cell (2*pi * integer_wavenumber)
        auto kvec = grid.get_fourier_wavevector_from_index(index);
        auto knorm2 = std::inner_product(kvec.begin(), kvec.end(), kvec.begin(), 0.0);

        // Fetch the value (should be +- N^3 / 2)
        auto value = grid.get_fourier_from_index(index);

        // Print non-zero values
        if (std::abs(value) > 1e-5) {
            std::cout << "Task: " << FML::ThisTask << " | ";
            for (auto & k : kvec) {
                std::cout << " ki: " << k / (2.0 * M_PI) << " ";
            }
            std::cout << " ==> " << value << "\n";
            // FML::assert_mpi(std::fabs( std::sqrt(knorm2)/(2.0*M_PI) - n) < 1e-5, "Non-zero fourier coefficient with
            // |k| != n\n");
        }

        // Divide by -k^2 and set value
        if (knorm2 > 0.0) {
            grid.set_fourier_from_index(index, -value / FML::GRID::FloatType(knorm2));
        } else {
            grid.set_fourier_from_index(index, 0.0);
        }
    }

    // Fourier transform it back to real-space
    grid.fftw_c2r();

    //==============================================
    // Print solution of Poisson equation and
    // compare with analytical solution
    //==============================================
    if (FML::ThisTask == 0)
        std::cout << "Compare solution to analytical solution:\n";
    for (auto && index : grid.get_real_range()) {
        // The (local) coordinates of the cell
        auto coord = grid.get_coord_from_index(index);

        // The (global) position of the cell
        auto pos = grid.get_real_position(coord);

        // The value
        auto value = grid.get_real_from_index(index);
        auto value_analytic = solution(pos);

        // Show error if its large enough
        if (std::abs(value - value_analytic) > 1e-10)
            std::cout << "Solution: " << value << " Analytic: " << value_analytic
                      << " Error: " << std::abs(value - value_analytic) << "\n";
    }
    if (FML::ThisTask == 0)
        std::cout << "(No output above means the solution matches the analytical one to < 1e-10)\n";
}
