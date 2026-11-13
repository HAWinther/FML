// Benchmark: the time of a forward and an inverse FFT of a 3D grid, FFTWGrid (FFTW-MPI slabs) against PencilGrid on
// slabs (P1 = 1, the same layout as FFTWGrid) and on the default pencils (MPI_Dims_create), for FFTW_ESTIMATE and
// FFTW_MEASURE plans. Usage: mpirun -np P ./benchmark_pencil_grid [Nmesh ...] (default 64 128 256)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace FML::GRID;

namespace {

    struct Timing {
        double r2c{0.0}, c2r{0.0};
    };

    // The time of f (the slowest task)
    template <class F>
    double timed(F && f) {
        MPI_Barrier(FML::communicator);
        const double t0 = MPI_Wtime();
        f();
        double t = MPI_Wtime() - t0;
        FML::MaxOverTasks(&t);
        return t;
    }

    // The median over repeats of the forward and the inverse transform (after one untimed round trip)
    template <class Grid>
    Timing benchmark(Grid & grid, int repeats) {
        auto fill = [](std::array<double, 3> & x) {
            return FloatType(std::sin(2 * M_PI * x[0]) + std::cos(4 * M_PI * x[1]) * std::sin(2 * M_PI * x[2]));
        };
        grid.fill_real_grid(std::function<FloatType(std::array<double, 3> &)>(fill));
        grid.fftw_r2c();
        grid.fftw_c2r();
        std::vector<double> r2c, c2r;
        for (int i = 0; i < repeats; i++) {
            r2c.push_back(timed([&] { grid.fftw_r2c(); }));
            c2r.push_back(timed([&] { grid.fftw_c2r(); }));
        }
        std::sort(r2c.begin(), r2c.end());
        std::sort(c2r.begin(), c2r.end());
        return {r2c[r2c.size() / 2], c2r[c2r.size() / 2]};
    }

    void report(const std::string & name, int n, const std::string & flag, Timing t, double reference) {
        if (FML::ThisTask != 0)
            return;
        const double total = t.r2c + t.c2r;
        std::printf("%-22s %5d %-8s %10.2f %10.2f %10.2f %8.2f\n",
                    name.c_str(),
                    n,
                    flag.c_str(),
                    1e3 * t.r2c,
                    1e3 * t.c2r,
                    1e3 * total,
                    reference > 0 ? total / reference : 1.0);
    }

} // namespace

int main(int argc, char ** argv) {
    std::vector<int> meshes;
    for (int i = 1; i < argc; i++)
        meshes.push_back(std::atoi(argv[i]));
    if (meshes.empty())
        meshes = {64, 128, 256};
    const auto P3 = FML::pencil_domain().get<FML::PencilDomain>().process_grid();

    if (FML::ThisTask == 0) {
        std::printf("# %d tasks, %d thread(s) per task, default pencils %d x %d\n",
                    FML::NTasks,
                    FML::NThreads,
                    P3[0],
                    P3[1]);
        std::printf("%-22s %5s %-8s %10s %10s %10s %8s\n",
                    "grid",
                    "Nmesh",
                    "plans",
                    "r2c [ms]",
                    "c2r [ms]",
                    "sum [ms]",
                    "/ FFTW");
    }
    for (int n : meshes) {
        const int repeats = n <= 64 ? 21 : (n <= 128 ? 11 : 5);
        for (unsigned flag : {unsigned(FFTW_ESTIMATE), unsigned(FFTW_MEASURE)}) {
            set_fftw_planner_flag(flag);
            const std::string flagname = fftw_planner_flag_name(flag);
            double reference = -1.0;
            if (n % FML::NTasks == 0) {
                FFTWGrid<3> grid(n);
                const auto t = benchmark(grid, repeats);
                reference = t.r2c + t.c2r;
                report("FFTWGrid", n, flagname, t, reference);
            }
            {
                PencilGrid<3> grid(n, 0, 0, FML::PencilDomain::uniform(FML::NTasks, 1));
                report("PencilGrid slabs", n, flagname, benchmark(grid, repeats), reference);
            }
            if (P3[1] > 1) {
                PencilGrid<3> grid(n);
                report("PencilGrid pencils", n, flagname, benchmark(grid, repeats), reference);
            }
        }
    }
    return 0;
}
