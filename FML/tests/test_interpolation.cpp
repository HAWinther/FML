//==========================================================
// Tests for FML::INTERPOLATION (Interpolation/ParticleGridInterpolation.h):
//   - NGP vector interpolation (used for the forces) of a constant field gives the constant for every particle (was 0
//     for particles in the right half of their cell in any dimension: 87.5%)
//   - lattice particles on the boundary of the slabs (x * Nmesh rounds to the cell of the left task) are assigned to
//     the grid: sum(delta) = 0 for NGP, ..., PQS. Tested with particles 1 ulp from the slab boundaries; with a
//     lattice it only happens for some task counts, e.g. 10: mpirun -np 10 ./test_interpolation slab
//   - interpolation of a linear field with CIC, ..., PQS (scalar and vector) is exact
// Bad input that must stop with a clear message (run by make run):
//   ./test_interpolation unknown : unknown method name (was silently ignored)
//   ./test_interpolation thin    : fewer slices on a task than the kernel needs (with 4 tasks; lost mass before)
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

using namespace FML;
using FFTWGrid = FML::GRID::FFTWGrid<3>;
using FloatType = FML::GRID::FloatType;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(3);
    os << x;
    return os.str();
}

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

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
    check(bad == 0,
          "NGP vector interpolation of a constant field = the constant for all particles (" + num(100 * bad / (3 * n)) +
              "% wrong; was 87.5% zero)");
}

void test_interpolation_exact() {
    // f = a + b * y_node: every kernel reproduces a linear function (with the periodic wrap in y the field is not
    // linear across y = 1, so only particles away from it are checked)
    const int Nm = 24;
    auto part = random_particles(20000, 2);
    for (auto & m : methods) {
        auto nlr = INTERPOLATION::get_extra_slices_needed_for_density_assignment(m);
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
        check(maxerr < 1e-12, m + " interpolation (scalar and vector) of a linear field: max error " + num(maxerr));
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
        auto nlr = INTERPOLATION::get_extra_slices_needed_for_density_assignment(m);
        FFTWGrid g(Nm, nlr.first, nlr.second);
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), NP, NPT, g, m);
        double sum = 0.0;
        for (auto && idx : g.get_real_range())
            sum += g.get_real_from_index(idx);
        SumOverTasks(&sum);
        check(std::fabs(sum) < 1e-6,
              m + ": particles 1 ulp from the slab boundaries are all assigned: sum(delta) = " + num(sum) +
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
        check(true,
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
        auto nlr = INTERPOLATION::get_extra_slices_needed_for_density_assignment(m);
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
        check(std::fabs(sum) < 1e-6 and (m == "NGP" or maxd < 1e-10),
              m + ": lattice Np = Nmesh = " + std::to_string(Nm) + " on " + std::to_string(NTasks) +
                  " tasks, particles on the slab boundary: sum(delta) = " + num(sum) + ", max|delta| = " + num(maxd) +
                  " (NGP lost a plane of particles, CIC wrote outside the grid)");
    }
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
        auto nlr = INTERPOLATION::get_extra_slices_needed_for_density_assignment("PQS");
        FFTWGrid g(4, nlr.first, nlr.second);
        std::vector<Particle> part(1, Particle{{xmin_domain, 0.5, 0.5}});
        INTERPOLATION::particles_to_grid<3, Particle>(part.data(), 1, NTasks, g, "PQS");
        return 0;
    }

    if (mode == "slab") {
        // Only this test (with e.g. 10 tasks, which do not divide the grid sizes of the other tests)
        test_slab_boundary();
    } else {
        test_vector_ngp();
        test_interpolation_exact();
        test_slab_boundary_ulp();
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All interpolation tests passed\n" :
                                   std::to_string(fails) + " interpolation test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
