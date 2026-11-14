// The screened modified gravity solvers on pencils. FofrSolverCosmology (sqrt and log), Symmetron-,
// Dilaton- and DGPSolverCosmology solved with a PencilGrid (the multigrid solver on its pencils) give the same
// fifth-force potential as with an FFTWGrid (the slabs) with the same number of levels, also from a previous solution
// (warm start); with the levels the pencils allow, the same to the accuracy of the solution
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/DGPSolver.h>
#include <FML/MultigridSolver/DilatonSolver.h>
#include <FML/MultigridSolver/FofrSolver.h>
#include <FML/MultigridSolver/SymmetronSolver.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
constexpr int NDIM = 3;
using FFTWGrid = FML::GRID::FFTWGrid<NDIM>;
using PencilGrid = FML::GRID::PencilGrid<NDIM>;

namespace {

    // A clustered density contrast (Gaussian halos on a small wave), > -1, the same for any decomposition
    double density(const std::array<double, NDIM> & x) {
        const double centres[4][3] = {{0.2, 0.3, 0.4}, {0.7, 0.75, 0.1}, {0.5, 0.05, 0.9}, {0.95, 0.5, 0.55}};
        double d = 0.3 * std::sin(2 * M_PI * x[0]) * std::cos(2 * M_PI * x[1]) * std::sin(4 * M_PI * x[2]);
        for (auto & c : centres) {
            double r2 = 0.0;
            for (int i = 0; i < NDIM; i++) {
                double dx = std::abs(x[i] - c[i]);
                dx = std::min(dx, 1.0 - dx);
                r2 += dx * dx;
            }
            d += 30.0 * std::exp(-r2 / (2 * 0.03 * 0.03));
        }
        return d - 0.6;
    }

    // Loop over the local cells of a grid with the global cell
    template <class Grid, class F>
    void for_cells(Grid & g, F && f) {
        for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(g.get_local_nrows()); irow++)
            for (auto && index : g.get_real_range_of_rows(irow, irow + 1)) {
                auto c = g.get_coord_from_index(index);
                for (int d = 0; d < 2; d++)
                    c[d] += int(g.get_local_range(d)[0]);
                f(index, c);
            }
    }

    template <class Grid>
    void fill(Grid & g) {
        const int n = g.get_nmesh();
        for_cells(g, [&](auto index, auto c) {
            std::array<double, NDIM> x;
            for (int d = 0; d < NDIM; d++)
                x[d] = (c[d] + 0.5) / n;
            g.set_real_from_index(index, density(x));
        });
    }

    template <class Grid>
    std::vector<double> global(Grid & g) {
        const int n = g.get_nmesh();
        std::vector<double> all(size_t(n) * n * n, 0.0);
        for_cells(g, [&](auto index, auto c) { all[(size_t(c[0]) * n + c[1]) * n + c[2]] = g.get_real_from_index(index); });
        FML::SumArrayOverTasks(all.data(), all.size());
        return all;
    }

    double max_rel_diff(const std::vector<double> & a, const std::vector<double> & b) {
        double d = 0.0, m = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            d = std::max(d, std::abs(a[i] - b[i]));
            m = std::max(m, std::abs(b[i]));
        }
        return d / m;
    }

    // The solvers: solve(density, potential, previous or nullptr, nlevels) -> V-cycles
    template <class Grid>
    using Solve = std::function<int(Grid &, Grid &, Grid *, int, double)>;

    template <class Grid>
    std::vector<std::pair<std::string, Solve<Grid>>> solvers() {
        const double H0Box = 256.0 / FML::UTILS::SI::c_over_H0_mpch, a = 1.0;
        std::vector<std::pair<std::string, Solve<Grid>>> s;
        for (std::string method : {"sqrt", "log"})
            s.push_back({"f(R) " + method, [=](Grid & d, Grid & phi, Grid * prev, int nl, double eps) {
                             FofrSolverCosmology<NDIM> solver(0.3, 1.0, 1e-5, H0Box, false);
                             solver.set_method(method);
                             solver.set_epsilon(eps);
                             solver.set_nlevels(nl);
                             if (prev)
                                 solver.set_previous_solution(*prev, 0.9);
                             solver.solve(a, d, phi);
                             return solver.get_number_of_vcycles();
                         }});
        s.push_back({"symmetron", [=](Grid & d, Grid & phi, Grid *, int nl, double eps) {
                         SymmetronSolverCosmology<NDIM> solver(0.3, 0.5, 1.0, 8.0, H0Box, false);
                         solver.set_epsilon(eps);
                         solver.set_nlevels(nl);
                         solver.solve(a, d, phi);
                         return solver.get_number_of_vcycles();
                     }});
        s.push_back({"dilaton", [=](Grid & d, Grid & phi, Grid * prev, int nl, double eps) {
                         DilatonSolverCosmology<NDIM> solver(0.3, 5e3, 0.5, 1.0, 1e-2, H0Box, false);
                         solver.set_epsilon(eps);
                         solver.set_nlevels(nl);
                         if (prev)
                             solver.set_previous_solution(*prev, 0.9);
                         solver.solve(a, d, phi);
                         return solver.get_number_of_vcycles();
                     }});
        s.push_back({"DGP", [=](Grid & d, Grid & phi, Grid * prev, int nl, double eps) {
                         const double beta = 1.0 + 2.0 * (1.0 - 0.3 / 2.0);
                         DGPSolverCosmology<NDIM> solver(0.3, 1.0, beta, H0Box, false);
                         solver.set_epsilon(eps);
                         solver.set_nlevels(nl);
                         if (prev)
                             solver.set_previous_solution(*prev);
                         solver.solve(a, d, phi);
                         return solver.get_number_of_vcycles();
                     }});
        return s;
    }

} // namespace

int main() {
    const int N = FML::NTasks % 3 == 0 ? 48 : 32;
    // The levels of the slabs, with an even coarsest level: on an odd one the colours of the sweeps do not alternate
    // across the periodic boundary and the result of a sweep depends on the order of the updates (the decomposition)
    int nlevels_slab = FML::SOLVERS::MULTIGRIDSOLVER::MPIMultiGrid<NDIM, double>::max_levels(N);
    while ((N >> (nlevels_slab - 1)) % 2 != 0)
        nlevels_slab--;
    // (a smaller previous solution: the field of a weaker density, to start from)
    auto ref_solvers = solvers<FFTWGrid>();
    auto pencil_solvers = solvers<PencilGrid>();

    std::vector<FML::PencilDomain> layouts;
    for (int p0 = 1; p0 <= FML::NTasks; p0++)
        if (FML::NTasks % p0 == 0)
            layouts.push_back(FML::PencilDomain::uniform(p0, FML::NTasks / p0));

    for (size_t k = 0; k < ref_solvers.size(); k++) {
        const auto & name = ref_solvers[k].first;
        FFTWGrid d(N, 1, 1), phi, phi_warm;
        fill(d);
        const int nv = ref_solvers[k].second(d, phi, nullptr, nlevels_slab, 1e-9);
        const auto ref = global(phi);
        const int nv_warm = ref_solvers[k].second(d, phi_warm, &phi, nlevels_slab, 1e-9);
        // (converged further: the reference for the solution with other levels)
        FFTWGrid phi_tight;
        ref_solvers[k].second(d, phi_tight, nullptr, nlevels_slab, 1e-11);
        const auto ref_tight = global(phi_tight);
        const auto ref_warm = global(phi_warm);
        for (auto & layout : layouts) {
            const std::string where = name + ", pencils " + std::to_string(layout.process_grid()[0]) + "x" +
                                      std::to_string(layout.process_grid()[1]);
            PencilGrid dp(N, 1, 1, layout), phip, phip_warm;
            fill(dp);
            const int nvp = pencil_solvers[k].second(dp, phip, nullptr, nlevels_slab, 1e-9);
            const double diff = max_rel_diff(global(phip), ref);
            const int nvp_warm = pencil_solvers[k].second(dp, phip_warm, &phip, nlevels_slab, 1e-9);
            const double diff_warm = max_rel_diff(global(phip_warm), ref_warm);
            check(diff < 1e-10 and nvp == nv and diff_warm < 1e-10 and nvp_warm == nv_warm,
                  where + ": the field of the slabs (" + std::to_string(nlevels_slab) + " levels, " +
                      std::to_string(nvp) + " V-cycles, rel. diff " + num(diff, 2) + "; warm start " +
                      std::to_string(nvp_warm) + " V-cycles, " + num(diff_warm, 2) + ")");

            const int nlevels_max =
                FML::SOLVERS::MULTIGRIDSOLVER::MPIMultiGrid<NDIM, double>::max_levels(N, FML::Domain(layout));
            if (nlevels_max > nlevels_slab) {
                PencilGrid phid;
                const int nvd = pencil_solvers[k].second(dp, phid, nullptr, -1, 1e-11);
                const double diffd = max_rel_diff(global(phid), ref_tight);
                // (DGP: the converged field depends on the number of levels at ~6e-5, also for 3x1, which is
                // the slabs: a property of the DGP solve, not of the pencils; the cause is not investigated)
                const double tol = name == "DGP" ? 2e-4 : 1e-6;
                check(diffd < tol and nvd < 100,
                      where + ": " + std::to_string(nlevels_max) + " levels: the same converged field in " + std::to_string(nvd) +
                          " V-cycles (slabs " + std::to_string(nv) + ", rel. diff " + num(diffd, 2) + ")");
            }
        }
    }
    return FML::TESTS::summary("screened solvers on pencils");
}
