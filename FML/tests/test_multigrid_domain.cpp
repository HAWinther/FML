// The multigrid solver on a domain (MPIGrid / MPIMultiGrid / MultiGridSolver with a FML::Domain). On
// Slab and on every Pencil layout of the tasks:
//   - MPIGrid: the extra cells (x-slices, y-layers and the corners, 1-3 layers, also more than the local cells) are
//     the cells of the neighbor tasks (periodic) or untouched outside a non-periodic box, and get_neighbor_gridindex
//     / get_shifted_index find them
//   - MultiGridSolver: the same solution as the fiducial slabs with the same number of levels for a nonlinear
//     equation (the 2 NDIM + 1 cells), with the cube ordering (the 3^NDIM cells, get_cube_gridindex) and with a
//     4th order stencil (get_Gradient2, two extra layers), and the deeper hierarchy pencils allow converges
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/MultiGridSolver.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using IndexInt = FML::GRID::IndexInt;

namespace {

    std::vector<FML::Domain> domains() {
        std::vector<FML::Domain> d{FML::Domain::uniform_slab(FML::NTasks)};
        for (int p0 = 1; p0 <= FML::NTasks; p0++)
            if (FML::NTasks % p0 == 0)
                d.push_back(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
        return d;
    }

    std::string name(const FML::Domain & d) {
        return d.is_slab() ? "Slab" : "Pencil " + std::to_string(d.pencil_px()) + "x" + std::to_string(d.pencil_py());
    }

    //==========================================================================
    // The extra cells
    //==========================================================================
    template <int NDIM>
    bool check_extra_cells(const FML::Domain & domain, int N, int nextra, bool periodic) {
        FML::GRID::MPIGrid<NDIM, double> grid(N, periodic, domain, nextra, nextra);
        auto value = [&](std::array<int, NDIM> g) {
            double v = 0.0;
            for (int d = 0; d < NDIM; d++)
                v = v * N + g[d];
            return v;
        };
        // Sentinels everywhere, then the main grid
        std::array<int, NDIM> lo, hi, start;
        for (int d = 0; d < NDIM; d++) {
            const auto r = grid.get_local_range(d);
            start[d] = r[0];
            const bool split = r[1] < N;
            const int extra = (d == 0 or split) ? nextra : 0;
            lo[d] = -extra;
            hi[d] = r[1] + extra;
        }
        auto for_all = [&](auto && f) {
            std::array<int, NDIM> c = lo;
            while (true) {
                f(c);
                int d = NDIM - 1;
                while (d >= 0 and ++c[d] == hi[d]) {
                    c[d] = lo[d];
                    d--;
                }
                if (d < 0)
                    break;
            }
        };
        for_all([&](const std::array<int, NDIM> & c) { grid[grid.index_from_coord(c)] = -1.0; });
        for (IndexInt i = 0; i < grid.get_NtotLocal(); i++)
            grid[i] = value(grid.globalcoord_from_index(i));
        grid.communicate_boundaries();

        bool ok = true;
        // Every cell: the value of the global cell (periodic) or untouched outside a non-periodic box. With one task
        // along x the extra x-slices are the periodic copies of our own (as for the slabs)
        for_all([&](const std::array<int, NDIM> & c) {
            std::array<int, NDIM> g;
            bool outside = false;
            for (int d = 0; d < NDIM; d++) {
                g[d] = c[d] + start[d];
                outside = outside or g[d] < 0 or g[d] >= N;
                g[d] = (g[d] % N + N) % N;
            }
            const double expected = (outside and not periodic) ? -1.0 : value(g);
            const IndexInt index = grid.index_from_coord(c);
            ok = ok and grid[index] == expected;
            // The index and the coordinate are inverses also in the extra cells
            ok = ok and grid.coord_from_index(index) == c;
        });
        // The neighbors of the main cells
        for (IndexInt i = 0; i < grid.get_NtotLocal(); i++) {
            const auto c = grid.coord_from_index(i);
            const auto g = grid.globalcoord_from_index(i);
            const auto list = grid.get_neighbor_gridindex(i);
            for (int d = 0; d < NDIM; d++)
                for (int s : {-1, 1}) {
                    auto gn = g;
                    gn[d] += s;
                    if (not periodic and (gn[d] < 0 or gn[d] >= N))
                        continue;
                    gn[d] = (gn[d] + N) % N;
                    ok = ok and grid[list[2 * d + (s > 0 ? 2 : 1)]] == value(gn);
                    for (int k = 1; k <= nextra; k++) {
                        auto gk = g;
                        gk[d] += s * k;
                        if (not periodic and (gk[d] < 0 or gk[d] >= N))
                            continue;
                        gk[d] = (gk[d] % N + N) % N;
                        ok = ok and grid[grid.get_shifted_index(i, c, d, s * k)] == value(gk);
                    }
                }
        }
        int bad = ok ? 0 : 1;
        FML::SumOverTasks(&bad);
        return bad == 0;
    }

    //==========================================================================
    // The solves
    //==========================================================================
    template <int NDIM>
    using Solver = FML::SOLVERS::MULTIGRIDSOLVER::MultiGridSolver<NDIM, double>;

    template <int NDIM>
    double source(const std::array<double, NDIM> & x) {
        double s = 0.0;
        for (int d = 0; d < NDIM; d++)
            s += std::sin(2.0 * M_PI * (d + 1) * x[d] + 0.3 * d) + 0.5 * std::cos(6.0 * M_PI * x[d]);
        double p = 1.0;
        for (int d = 0; d < NDIM; d++)
            p *= std::sin(2.0 * M_PI * x[d]);
        return 10.0 * (s + 3.0 * p);
    }

    // The equations: 0: the Laplacian (2 NDIM + 1 cells) - f - f^3 / 10, 1: the 3^NDIM cells (cube ordering), 2: the
    // 4th order Laplacian (two extra layers)
    template <int NDIM>
    std::vector<double> solve(int equation, int N, int nlevels, const FML::Domain * domain, int & nlevels_used,
                              int & nvcycles, int maxsteps = 200) {
        const int nextra = equation == 2 ? 2 : 1;
        Solver<NDIM> g = domain ? Solver<NDIM>(N, *domain, nlevels, false, true, nextra, nextra) :
                                  Solver<NDIM>(N, nlevels, false, true, nextra, nextra);
        g.set_epsilon(1e-11);
        g.set_maxsteps(maxsteps);
        g.set_ngs_sweeps(3, 3, 3);
        if (equation == 1)
            g.set_cube_ordering(true);
        g.set_initial_guess(0.0);

        typename FML::SOLVERS::MULTIGRIDSOLVER::MultiGridFunction<NDIM, double> eq;
        if (equation == 0) {
            eq = [](Solver<NDIM> * sol, int level, IndexInt index) {
                auto list = sol->get_neighbor_gridindex(level, index);
                const double f = sol->get_Field(level, index);
                const double l = sol->get_Laplacian(level, list) - f - f * f * f / 10.0 -
                                 (level == 0 ? source<NDIM>(sol->get_Coordinate(level, index)) : 0.0);
                const double dl = sol->get_derivLaplacian(level, list) - 1.0 - 3.0 * f * f / 10.0;
                return std::pair<double, double>{l, dl};
            };
        } else if (equation == 1) {
            eq = [](Solver<NDIM> * sol, int level, IndexInt index) {
                const auto list = sol->get_cube_gridindex(level, index);
                constexpr int ncells = FML::power(3, NDIM);
                const double h = sol->get_Gridspacing(level);
                const double f = sol->get_Field(level, index);
                // (a symmetric operator: the weight of a cell is one over its squared distance)
                double sum = 0.0, wsum = 0.0;
                for (int k = 0; k < ncells; k++) {
                    int r2 = 0;
                    for (int d = 0, kk = k; d < NDIM; d++, kk /= 3)
                        r2 += (kk % 3 - 1) * (kk % 3 - 1);
                    if (r2 == 0)
                        continue;
                    sum += sol->get_Field(level, list[k]) / r2;
                    wsum += 1.0 / r2;
                }
                const double norm = 2.0 * NDIM / (h * h * wsum);
                const double l = norm * (sum - wsum * f) - f -
                                 (level == 0 ? source<NDIM>(sol->get_Coordinate(level, index)) : 0.0);
                return std::pair<double, double>{l, -norm * wsum - 1.0};
            };
        } else {
            eq = [](Solver<NDIM> * sol, int level, IndexInt index) {
                const auto d2 = sol->get_Gradient2(level, index, 2);
                const double h = sol->get_Gridspacing(level);
                const double f = sol->get_Field(level, index);
                double lap = 0.0;
                for (int d = 0; d < NDIM; d++)
                    lap += d2[d];
                const double l = lap - f - (level == 0 ? source<NDIM>(sol->get_Coordinate(level, index)) : 0.0);
                return std::pair<double, double>{l, -NDIM * 2.5 / (h * h) - 1.0};
            };
        }
        g.solve(eq, g.ConvergenceCriterionResidual);
        nlevels_used = nlevels;
        nvcycles = g.get_solve_result().iterations;
        const bool converged = g.get_solve_result().converged();
        if (not converged and maxsteps > 10)
            nvcycles = -1;

        // The solution on the global grid (on every task)
        auto & grid = g.get_grid(0);
        std::vector<double> global(FML::power(size_t(N), NDIM), 0.0);
        for (IndexInt i = 0; i < grid.get_NtotLocal(); i++) {
            const auto c = grid.globalcoord_from_index(i);
            size_t k = 0;
            for (int d = 0; d < NDIM; d++)
                k = k * N + c[d];
            global[k] = grid[i];
        }
        FML::SumArrayOverTasks(global.data(), global.size());
        return global;
    }

    double max_diff(const std::vector<double> & a, const std::vector<double> & b) {
        double d = 0.0, m = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            d = std::max(d, std::abs(a[i] - b[i]));
            m = std::max(m, std::abs(b[i]));
        }
        return d / m;
    }

    template <int NDIM>
    void run(int N) {
        const std::string dim = std::to_string(NDIM) + "D";
        for (auto & domain : domains()) {
            bool ok = true;
            for (bool periodic : {true, false})
                for (int nextra = 1; nextra <= 3; nextra++) {
                    // (N cells and 6 cells per dim: up to three extra layers with one local layer)
                    // (12 cells per dim and one layer per task)
                    ok = ok and check_extra_cells<NDIM>(domain, 12, nextra, periodic);
                    ok = ok and check_extra_cells<NDIM>(domain, std::max(FML::NTasks, 2), nextra, periodic);
                }
            check(ok, dim + " " + name(domain) + ": the extra cells, the neighbors and the shifted cells");
        }

        const std::vector<std::string> equations{"nonlinear", "cube ordering", "4th order"};
        for (int equation = 0; equation < 3; equation++) {
            // The fiducial slabs
            const int nlevels_slab = FML::GRID::MPIMultiGrid<NDIM, double>::max_levels(N);
            int nl0, nv0;
            const auto ref = solve<NDIM>(equation, N, nlevels_slab, nullptr, nl0, nv0);
            for (auto & domain : domains()) {
                int nl, nv;
                // The same levels: the same field after 3 V-cycles (the red-black sweeps do not depend on the
                // decomposition; not on an odd coarsest level, where the colours do not alternate across the periodic
                // boundary and the result of a sweep depends on the order of the updates)
                if (equation != 2) {
                    int nl3 = nlevels_slab, nv3;
                    while ((N >> (nl3 - 1)) % 2 != 0)
                        nl3--;
                    const auto ref3 = solve<NDIM>(equation, N, nl3, nullptr, nl3, nv3, 3);
                    const auto f3 = solve<NDIM>(equation, N, nl3, &domain, nl3, nv3, 3);
                    const double diff3 = max_diff(f3, ref3);
                    check(diff3 < 1e-13,
                          dim + " " + equations[equation] + ", " + name(domain) +
                              ": the field of the slabs after 3 V-cycles (rel. diff " + num(diff3, 2) + ")");
                }
                // ... and the same solution
                const auto f = solve<NDIM>(equation, N, nlevels_slab, &domain, nl, nv);
                const double diff = max_diff(f, ref);
                // (the 4th order stencil reads cells of the same colour two cells away, which are updated in the
                // same sweep: the result depends on the order of the updates, i.e. on the decomposition, as it does
                // for the slabs and one task. The others are the same up to the order of the sums over the tasks)
                const bool exact = equation != 2;
                check(nv > 0 and (exact ? diff < 1e-12 and nv == nv0 : diff < 1e-8),
                      dim + " " + equations[equation] + ", " + name(domain) + ": the solution of the slabs (" +
                          std::to_string(nlevels_slab) + " levels, " + std::to_string(nv) + " V-cycles, rel. diff " +
                          num(diff, 2) + ")");

                // All the levels the domain allows
                const int nlevels_max = FML::GRID::MPIMultiGrid<NDIM, double>::max_levels(N, domain);
                if (nlevels_max > nlevels_slab) {
                    const auto f2 = solve<NDIM>(equation, N, nlevels_max, &domain, nl, nv);
                    const double diff2 = max_diff(f2, ref);
                    check(diff2 < 1e-8 and nv > 0,
                          dim + " " + equations[equation] + ", " + name(domain) + ": " +
                              std::to_string(nlevels_max) + " levels (slabs " + std::to_string(nlevels_slab) +
                              "): converges to the same solution in " + std::to_string(nv) + " V-cycles (slabs " +
                              std::to_string(nv0) + ", rel. diff " + num(diff2, 2) + ")");
                }
            }
        }
    }

} // namespace

int main() {
    // (N / 2^l divisible by the number of tasks on the levels of the slabs)
    run<3>(FML::NTasks % 3 == 0 ? 48 : 32);
    run<2>(FML::NTasks % 3 == 0 ? 96 : 64);
    return FML::TESTS::summary("multigrid on a domain");
}
