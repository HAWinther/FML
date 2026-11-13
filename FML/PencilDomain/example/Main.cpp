#include <FML/PencilDomain/PencilGrid.h>

#include <cmath>
#include <iostream>
#include <memory>

// MPI, FML and FFTW are initialized (and finalized) by FML automatically (Global.cpp, FMLSetup)
int main() {
    {
        using namespace FML::GRID;
        auto domain = std::make_shared<PencilDomain>(std::array<int,3>{16,16,16}, 1.0);
        PencilGrid grid(domain, 1);
        grid.fill_real_grid([](std::array<double,3>& x) { return std::sin(2.0 * M_PI * x[0]) * std::cos(2.0 * M_PI * x[1]); });
        grid.fftw_r2c();
        grid.fftw_c2r();
        double local_error = 0.0;
        for (auto c : grid.get_real_range()) {
            auto x = grid.get_real_position({c.i,c.j,c.k});
            double expected = std::sin(2.0*M_PI*x[0]) * std::cos(2.0*M_PI*x[1]);
            local_error = std::max(local_error, std::abs(grid[c] - expected));
        }
        double error = 0.0;
        MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, domain->communicator());
        if (domain->rank() == 0) std::cout << "maximum FFT round-trip error: " << error << '\n';
    }
    return 0;
}
