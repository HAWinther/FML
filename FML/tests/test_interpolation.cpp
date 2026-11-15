//==========================================================
// Tests for FML::INTERPOLATION (Interpolation/ParticleGridInterpolation.h):
//   - NGP vector interpolation (used for the forces) of a constant field gives the constant for every particle (was 0
//     for particles in the right half of their cell in any dimension: 87.5%)
//   - lattice particles on the boundary of the slabs (x * Nmesh rounds to the cell of the left task) are assigned to
//     the grid: sum(delta) = 0 for NGP, ..., PQS. Tested with particles 1 ulp from the slab boundaries; with a
//     lattice it only happens for some task counts, e.g. 10: mpirun -np 10 ./test_interpolation slab
//   - interpolation of a linear field with CIC, ..., PQS (scalar and vector) is exact
//   - convolve_grid_with_kernel with a top-hat of 3 cells (ORDER 3) and a constant kernel (ORDER 1) on a cosine
// Bad input that must stop with a clear message (run by make run):
//   ./test_interpolation unknown : unknown method name (was silently ignored)
//   ./test_interpolation thin    : fewer slices on a task than the kernel needs (with 4 tasks; lost mass before)
//   ./test_interpolation outside : a particle far outside the slab of its task (2 tasks; was moved to the edge and
//                                  lost mass without notice), outside_interp the same for the interpolation
//   ./test_interpolation zero    : no particles (NumPartTot = 0 filled the grid with inf)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;

using namespace FML;
using FFTWGrid = FML::GRID::FFTWGrid<3>;
using FloatType = FML::GRID::FloatType;


using Particle = FML::TESTS::PositionParticle<3>;

const std::vector<std::string> methods{"NGP", "CIC", "TSC", "PCS", "PQS"};

std::vector<Particle> random_particles(int n, int seed) {
    std::mt19937 gen(seed + 1000 * ThisTask);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<Particle> part(n);
    for (auto & p : part) {
        p.pos[0] = xmin_domain + (xmax_domain - xmin_domain) * U(gen);
        if (p.pos[0] >= xmax_domain)
            p.pos[0] = xmin_domain;
        p.pos[1] = U(gen);
        p.pos[2] = U(gen);
    }
    return part;
}


void test_vector_ngp() {
    const int Nm = 24;
    auto part = random_particles(20000, 1);
    FFTWGrid g(Nm, 0, 0);
    g.fill_real_grid(1.0);
    g.communicate_boundaries();
    std::array<FFTWGrid, 3> gv{g, g, g};
    std::array<std::vector<FloatType>, 3> out;
    std::vector<FloatType> sc;
    INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, Particle>(gv, part.data(), part.size(), out, "NGP");
    INTERPOLATION::interpolate_grid_to_particle_positions<3, Particle>(g, part.data(), part.size(), sc, "NGP");
    double bad = 0, n = part.size();
    for (size_t i = 0; i < part.size(); i++)
        for (int d = 0; d < 3; d++)
            bad += (out[d][i] != 1.0) + (d == 0 and sc[i] != 1.0);
    SumOverTasks(&bad);
    SumOverTasks(&n);
    check_local(bad == 0,
                "NGP vector interpolation of a constant field = the constant for all particles (" +
                    num(100 * bad / (3 * n), 3) + "% wrong; was 87.5% zero)");
}

void test_interpolation_exact() {
    // f = a + b * y_node: every kernel reproduces a linear function (with the periodic wrap in y the field is not
    // linear across y = 1, so only particles away from it are checked)
    const int Nm = 24;
    auto part = random_particles(20000, 2);
    for (auto & m : methods) {
        auto nlr = INTERPOLATION::get_extra_slices(m);
        FFTWGrid g(Nm, nlr.first, nlr.second);
        for (auto && real_index : g.get_real_range()) {
            auto coord = g.get_coord_from_index(real_index);
            g.set_real_from_index(real_index, 2.0 + 3.0 * coord[1] / double(Nm));
        }
        g.communicate_boundaries();
        std::array<FFTWGrid, 3> gv{g, g, g};
        std::array<std::vector<FloatType>, 3> out;
        std::vector<FloatType> sc;
        INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, Particle>(gv, part.data(), part.size(), out, m);
        INTERPOLATION::interpolate_grid_to_particle_positions<3, Particle>(g, part.data(), part.size(), sc, m);
        double maxerr = 0.0;
        for (size_t i = 0; i < part.size(); i++) {
            const double y = part[i].pos[1];
            if (y < 4.0 / Nm or y > 1.0 - 4.0 / Nm)
                continue;
#ifdef CELLCENTERSHIFTED
            const double expected = 2.0 + 3.0 * (y - 0.5 / Nm);
#else
            const double expected = m == "NGP" ? 2.0 + 3.0 * std::floor(y * Nm) / Nm : 2.0 + 3.0 * y;
#endif
            maxerr = std::max({maxerr, std::fabs(sc[i] - expected), std::fabs(out[1][i] - expected)});
        }
        MaxOverTasks(&maxerr);
        check_local(maxerr < 1e-12,
                    m + " interpolation (scalar and vector) of a linear field: max error " + num(maxerr, 3));
    }
}

// Particles within round-off of the slab boundaries: x = xmin_domain - 1 ulp (as the round-off in a lattice gives),
// and x = xmax_domain - 1 ulp (x * Nmesh can round up to the first cell of the right task)
void test_slab_boundary_ulp() {
    if (NTasks == 1)
        return;
    const int Nm = 24;
    std::vector<Particle> part;
    std::mt19937 gen(7 + ThisTask);
    std::uniform_real_distribution<double> U(0, 1);
    for (int i = 0; i < 2000; i++) {
        double x = (i % 2 == 0) ? std::nextafter(xmin_domain, 0.0) : std::nextafter(xmax_domain, 0.0);
        if (ThisTask == 0 and i % 2 == 0)
            x = 0.0;
        part.push_back({{x, U(gen), U(gen)}});
    }
    size_t NP = part.size(), NPT = NP;
    SumOverTasks(&NPT);
    for (auto & m : methods) {
        auto nlr = INTERPOLATION::get_extra_slices(m);
        FFTWGrid g(Nm, nlr.first, nlr.second);
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), NP, NPT, g, m);
        double sum = 0.0;
        for (auto && idx : g.get_real_range())
            sum += g.get_real_from_index(idx);
        SumOverTasks(&sum);
        check_local(std::fabs(sum) < 1e-6,
                    m + ": particles 1 ulp from the slab boundaries are all assigned: sum(delta) = " + num(sum, 3) +
                        " (were outside the grid: lost, or written out of bounds)");
    }
}

void test_slab_boundary() {
    // Find a lattice Np = Nmesh (multiple of NTasks) where x = i/Np on this or another task gives int(x*Nm) outside
    // the task's slab
    int Nfound = 0;
    for (int Nm = NTasks; Nm <= 300 and Nfound == 0; Nm += NTasks) {
        int hit = 0;
        for (int task = 1; task < NTasks; task++) {
            const double xmin = task / double(NTasks);
            const int start = task * (Nm / NTasks);
            for (int i = 0; i < Nm; i++) {
                const double x = i / double(Nm);
                if (x >= xmin and x < xmin + 1.0 / NTasks and int(x * Nm) < start)
                    hit = 1;
            }
        }
        if (hit)
            Nfound = Nm;
    }
    if (Nfound == 0) {
        check_local(true,
                    "slab boundary: no lattice puts a particle outside its slab with " + std::to_string(NTasks) +
                        " task(s) (try 10 tasks)");
        return;
    }
    const int Nm = Nfound, Np = Nfound;
    std::vector<Particle> part;
    for (int i = 0; i < Np; i++) {
        const double x = i / double(Np);
        if (not(x >= xmin_domain and x < xmax_domain))
            continue;
        for (int j = 0; j < Np; j++)
            for (int k = 0; k < Np; k++)
                part.push_back({{x, j / double(Np), k / double(Np)}});
    }
    size_t NP = part.size(), NPT = NP;
    SumOverTasks(&NPT);
    for (auto & m : methods) {
        auto nlr = INTERPOLATION::get_extra_slices(m);
        FFTWGrid g(Nm, nlr.first, nlr.second);
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), NP, NPT, g, m);
        double sum = 0.0, maxd = 0.0;
        for (auto && idx : g.get_real_range()) {
            sum += g.get_real_from_index(idx);
            maxd = std::max(maxd, std::fabs(double(g.get_real_from_index(idx))));
        }
        SumOverTasks(&sum);
        MaxOverTasks(&maxd);
        // (for NGP the cell of a particle exactly on a node is decided by round-off, so only the sum is checked)
        check_local(std::fabs(sum) < 1e-6 and (m == "NGP" or maxd < 1e-10),
                    m + ": lattice Np = Nmesh = " + std::to_string(Nm) + " on " + std::to_string(NTasks) +
                        " tasks, particles on the slab boundary: sum(delta) = " + num(sum, 3) + ", max|delta| = " +
                        num(maxd, 3) + " (NGP lost a plane of particles, CIC wrote outside the grid)");
    }
}

// grid_out(x) = sum_d K(d) grid_in(x + d): for f = cos(2 pi m x_c / Nmesh) along x (across the tasks) and y and a
// top-hat of 3 cells, every cosine is multiplied by (1 + 2 cos(2 pi m / Nmesh)) / 3; ORDER 1 multiplies by K(0)
void test_convolve() {
    const int Nm = 24, mx = 2, my = 3;
    const auto nlr = INTERPOLATION::get_extra_slices_needed_by_order<3>();
    FFTWGrid g(Nm, nlr.first, nlr.second);
    for (auto && real_index : g.get_real_range()) {
        auto coord = g.get_coord_from_index(real_index);
        const double xc = coord[0] + double(g.get_local_x_start());
        g.set_real_from_index(real_index,
                              std::cos(2 * M_PI * mx * xc / Nm) + std::cos(2 * M_PI * my * coord[1] / Nm));
    }
    g.communicate_boundaries();
    const double fx = (1 + 2 * std::cos(2 * M_PI * mx / Nm)) / 3.0, fy = (1 + 2 * std::cos(2 * M_PI * my / Nm)) / 3.0;

    FFTWGrid out3, out1;
    INTERPOLATION::convolve_grid_with_kernel<3, 3>(
        g, out3, [](std::array<double, 3> &) { return FloatType(1.0 / 27.0); });
    INTERPOLATION::convolve_grid_with_kernel<3, 1>(g, out1, [](std::array<double, 3> & dx) {
        return FloatType(dx[0] == 0 and dx[1] == 0 and dx[2] == 0 ? 2.0 : 0.0);
    });
    double err3 = 0.0, err1 = 0.0;
    for (auto && real_index : g.get_real_range()) {
        auto coord = g.get_coord_from_index(real_index);
        const double xc = coord[0] + double(g.get_local_x_start());
        const double cx = std::cos(2 * M_PI * mx * xc / Nm), cy = std::cos(2 * M_PI * my * coord[1] / Nm);
        err3 = std::max(err3, std::fabs(out3.get_real_from_index(real_index) - (fx * cx + fy * cy)));
        err1 = std::max(err1, std::fabs(out1.get_real_from_index(real_index) - 2 * (cx + cy)));
    }
    MaxOverTasks(&err3);
    MaxOverTasks(&err1);
    check_local(err3 < 1e-12, "convolve_grid_with_kernel, 3-cell top-hat on a cosine: max error " + num(err3, 3));
    check_local(err1 < 1e-12, "convolve_grid_with_kernel, ORDER 1 multiplies by K(0): max error " + num(err1, 3) +
                                  " (ignored the kernel; did not compile before)");
}

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "unknown") {
        FFTWGrid g(8, 1, 1);
        std::vector<Particle> part(1, Particle{{xmin_domain, 0.5, 0.5}});
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), 1, NTasks, g, "cic");
        return 0;
    }
    if (mode == "thin") {
        // 4 tasks, Nmesh 4: one slice per task, PQS needs 2 extra slices
        auto nlr = INTERPOLATION::get_extra_slices("PQS");
        FFTWGrid g(4, nlr.first, nlr.second);
        std::vector<Particle> part(1, Particle{{xmin_domain, 0.5, 0.5}});
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), 1, NTasks, g, "PQS");
        return 0;
    }

    if (mode == "outside" or mode == "outside_interp") {
        // 2 tasks, Nmesh 16: the particle of task 0 is 0.1 * 16 = 1.6 cells into the slab of task 1
        FFTWGrid g(16, 1, 2);
        std::vector<Particle> part(1, Particle{{xmin_domain + (ThisTask == 0 ? 0.6 : 0.1), 0.5, 0.5}});
        if (mode == "outside") {
            INTERPOLATION::particles_to_grid<3, Particle>(part.data(), 1, NTasks, g, "CIC");
        } else {
            g.fill_real_grid(1.0);
            g.communicate_boundaries();
            std::vector<FloatType> values;
            INTERPOLATION::interpolate_grid_to_particle_positions<3, Particle>(g, part.data(), 1, values, "CIC");
        }
        return 0;
    }
    if (mode == "zero") {
        FFTWGrid g(8, 1, 2);
        std::vector<Particle> part;
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), 0, 0, g, "CIC");
        return 0;
    }

    if (mode == "slab") {
        // Only this test (with e.g. 10 tasks, which do not divide the grid sizes of the other tests)
        test_slab_boundary();
    } else {
        test_vector_ngp();
        test_interpolation_exact();
        test_slab_boundary_ulp();
        test_convolve();
    }

    return FML::TESTS::summary("interpolation");
}
