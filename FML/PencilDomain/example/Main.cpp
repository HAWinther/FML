#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>

#include <algorithm>
#include <cmath>
#include <iostream>

// MPI, FML and FFTW are initialized (and finalized) by FML automatically (Global.cpp, FMLSetup)
int main() {
    using namespace FML::GRID;

    // A 3D grid on the default pencils with one extra cell on each side in x and y
    const int Nmesh = 32;
    PencilGrid<3> grid(Nmesh, 1, 1);
    grid.add_memory_label("Example");
    grid.info();

    auto func = [](std::array<double, 3> & x) {
        return FloatType(std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]) + 0.1 * std::cos(6.0 * M_PI * x[2]));
    };
    grid.fill_real_grid(std::function<FloatType(std::array<double, 3> &)>(func));

    // Fourier transform, apply a filter (here the identity, a Gaussian with a huge width), transform back
    grid.fftw_r2c();
    for (auto && index : grid.get_fourier_range()) {
        std::array<double, 3> kvec;
        double kmag;
        grid.get_fourier_wavevector_and_norm_by_index(index, kvec, kmag);
        grid.set_fourier_from_index(index,
                                    grid.get_fourier_from_index(index) * FloatType(std::exp(-kmag * kmag * 1e-30)));
    }
    grid.fftw_c2r();
    grid.communicate_boundaries();

    // Compare with the function (also the extra cells)
    double max_error = 0.0;
    for (int ix = -1; ix <= grid.get_local_nx(); ix++)
        for (int iy = -1; iy <= grid.get_local_ny(); iy++)
            for (int iz = 0; iz < Nmesh; iz++) {
                auto x = grid.get_real_position({ix, iy, iz});
                max_error = std::max(max_error, std::abs(double(grid.get_real({ix, iy, iz}) - func(x))));
            }
    FML::MaxOverTasks(&max_error);
    if (FML::ThisTask == 0)
        std::cout << "Maximum error of the FFT round trip: " << max_error << "\n";
    return 0;
}
