// PencilGrid::dump_to_file / load_from_file (checkpoints) give back the same grid: size, extra cells,
// pencils, status and every value (real space with the extra cells, and Fourier space), 2D and 3D
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cstdio>
#include <string>

using FML::TESTS::check;
using namespace FML::TESTS::GRIDS;

namespace {
    template <int N>
    void roundtrip(const FML::PencilDomain & layout, const std::string & name) {
        const int n = FML::NTasks % 3 == 0 ? 12 : 16;
        FML::GRID::PencilGrid<N> g(n, 1, 2, layout);
        fill<N>(g, [](auto x) {
            double v = 0;
            for (int d = 0; d < N; d++)
                v += std::sin(2 * M_PI * (d + 1) * x[d] + 0.1 * d);
            return v;
        });
        g.communicate_boundaries();
        const std::string prefix = "test_pencil_checkpoint_grid";
        // Real space, the extra cells included
        g.dump_to_file(prefix);
        FML::GRID::PencilGrid<N> r;
        r.load_from_file(prefix);
        bool same = r.get_nmesh() == n and r.get_n_extra_slices_left() == 1 and r.get_n_extra_slices_right() == 2 and
                    r.get_grid_status_real() and r.get_domain() == g.get_domain();
        for (long long i = 0; same and i < g.get_ntot_fourier_alloc(); i++)
            same = std::memcmp(&g.get_fourier_grid()[i], &r.get_fourier_grid()[i], sizeof(ComplexType)) == 0;
        // Fourier space
        g.fftw_r2c();
        g.dump_to_file(prefix);
        r.load_from_file(prefix);
        same = same and not r.get_grid_status_real() and max_rel_diff(global_fourier(r), global_fourier(g)) == 0.0;
        int good = same ? 1 : 0;
        FML::MinOverTasks(&good);
        check(good == 1, name + std::to_string(N) + "D: dump_to_file / load_from_file give back the grid");
        if (FML::ThisTask >= 0)
            std::remove((prefix + "." + std::to_string(FML::ThisTask)).c_str());
    }
} // namespace

int main() {
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) + ", ";
        roundtrip<3>(FML::PencilDomain::uniform(p0, FML::NTasks / p0), name);
    }
    roundtrip<2>(FML::PencilDomain::uniform(FML::NTasks, 1), "pencils " + std::to_string(FML::NTasks) + "x1, ");
    return FML::TESTS::summary("PencilGrid checkpoints");
}
