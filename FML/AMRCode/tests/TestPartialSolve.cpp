#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>

#include <cmath>

//==========================================================================================
// Test of the partial solve of AMRMultiGridSolver (solve(eq, min_active_level)), used for
// subcycling in AMRNBody:
//  1. Full solve of D^2 phi = delta on a grid refined around a peak (levels 4-7).
//  2. Scramble phi on the leaves at level >= L and do a partial solve: the fixed (coarser)
//     leaves must be untouched and the result must equal the full solution.
//  3. The partial solve must take about as many V-cycles as the full solve.
// Run: mpirun -np N ./testpartial   (the checksums must not depend on N)
//==========================================================================================

using namespace FML::GRID;
using namespace FML::SOLVERS::AMRMULTIGRIDSOLVER;

int main() {
    constexpr int NDIM = 3;
    using Grid = AdaptiveMPIGrid<NDIM, double>;
    const int levelmin = 4, levelmax = 7;
    const std::array<double, 3> center{0.43, 0.51, 0.56};
    const double sigma = 0.04;
    auto r2 = [&](const std::array<double, 3> & x) {
        double s = 0.0;
        for (int d = 0; d < 3; d++) {
            double dx = x[d] - center[d];
            dx -= std::round(dx);
            s += dx * dx;
        }
        return s;
    };

    Grid grid(levelmin, levelmax, {true, true, true});
    for (int pass = 0; pass < levelmax - levelmin; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            const auto & c = grid.get_cell(i);
            const double radius = 0.25 / (1 << (c.level - levelmin));
            grid.set_flag(i, std::sqrt(r2(grid.get_cell_center(c))) < radius ? Grid::REFINE : Grid::KEEP);
        }
        grid.adapt();
        grid.load_balance();
    }

    // A compensated source with zero volume-weighted mean
    const IndexInt n = grid.get_n_local();
    std::vector<double> delta(n);
    double mean = 0.0;
    for (IndexInt i = 0; i < n; i++) {
        const auto & c = grid.get_cell(i);
        const double q = r2(grid.get_cell_center(c)) / (2 * sigma * sigma);
        delta[i] = std::exp(-q) * (3.0 - 2.0 * q);
        mean += delta[i] * std::ldexp(1.0, -NDIM * c.level);
    }
    FML::SumOverTasks(&mean);
    for (auto & d : delta)
        d -= mean;

    AMRMultiGridSolver<NDIM> solver(grid, -1, false);
    int id = solver.add_external_field(delta);
    AMREquation<NDIM> eq = [id](const AMRMultiGridSolver<NDIM> & sol, int level, IndexInt i) {
        auto [lap, dlap] = sol.get_laplacian(level, i);
        return std::pair<double, double>{lap - sol.get_external_field(id, level, i), dlap};
    };
    // Absolute residual criterion: the partial solves start from a scrambled guess with a large residual
    solver.set_convergence_criterion_residual(true);
    solver.set_epsilon(1e-11);
    solver.set_maxsteps(50);
    solver.set_subtract_mean_residual(true);
    auto rfull = solver.solve(eq);
    const std::vector<double> phi_full = solver.get_solution();

    // Mean residual reduction per V-cycle
    auto rate = [](const SolveResult & r) { return std::pow(r.residual / r.initial_residual, 1.0 / r.iterations); };
    int nfail = 0;
    for (int L = levelmin + 1; L <= levelmax; L++) {
        // Scramble the unknowns
        std::vector<double> guess = phi_full;
        long long nactive = 0;
        for (IndexInt i = 0; i < n; i++)
            if (grid.get_cell(i).level >= L) {
                guess[i] += 0.3 * std::sin(17.0 * i + L);
                nactive++;
            }
        FML::SumOverTasks(&nactive);
        solver.set_initial_guess(guess);
        auto r = solver.solve(eq, L);
        const auto & phi = solver.get_solution();
        double maxerr_active = 0.0, maxchange_fixed = 0.0, maxphi = 0.0;
        for (IndexInt i = 0; i < n; i++) {
            maxphi = std::max(maxphi, std::abs(phi_full[i]));
            if (grid.get_cell(i).level >= L)
                maxerr_active = std::max(maxerr_active, std::abs(phi[i] - phi_full[i]));
            else
                maxchange_fixed = std::max(maxchange_fixed, std::abs(phi[i] - guess[i]));
        }
        FML::MaxOverTasks(&maxerr_active);
        FML::MaxOverTasks(&maxchange_fixed);
        FML::MaxOverTasks(&maxphi);
        const bool ok = r.converged() and maxchange_fixed == 0.0 and maxerr_active < 1e-9 * maxphi and rate(r) < 0.25;
        nfail += not ok;
        if (FML::ThisTask == 0)
            std::cout << "Partial solve L = " << L << " (" << nactive << " unknown leaves): " << r.iterations
                       << " V-cycles, reduction " << rate(r) << " per cycle (full solve " << rate(rfull) << "), max|phi - phi_full| / max|phi| = "
                      << maxerr_active / maxphi << ", fixed cells changed by " << maxchange_fixed
                      << (ok ? "  OK" : "  FAIL") << "\n";
    }
    double checksum = 0.0;
    for (IndexInt i = 0; i < n; i++)
        checksum += phi_full[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    FML::SumOverTasks(&checksum);
    const auto nglobal = grid.get_n_global(); // collective
    if (FML::ThisTask == 0)
        std::cout << "Leaves: " << nglobal << "  checksum " << std::setprecision(12) << checksum << "\n"
                  << (nfail ? "FAILED\n" : "ALL OK\n");
    return nfail;
}
