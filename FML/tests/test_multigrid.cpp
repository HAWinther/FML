//==========================================================
// Tests of the multigrid solver (MultigridSolver/MultiGridSolver.h), run on 1, 2 and 4 tasks:
//   - the finite difference helpers get_Gradient / get_Gradient2 (orders 1-4) against analytic derivatives
//   - Poisson's equation with a white-noise source: the residual is reduced by > 5 per V-cycle (2+2 sweeps)
//   - a DGP-type equation (mixed second derivatives, i.e. the 3^3 cells around a cell, and an operator that
//     only determines the field up to a constant): converges with the cube ordering of the sweeps and the
//     mean of the residual subtracted
//   - FofrSolverCosmology (with FFTW): the cubic of the sqrt method, the sqrt and log methods give the same fifth
//     force, and the sqrt method converges for |f_R0| = 1e-6 at a = 0.5 (where the log method gives NaN) and for n = 2
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/MultiGridSolver.h>
#ifdef USE_FFTW
#include <FML/MultigridSolver/FofrSolver.h>
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::SOLVERS::MULTIGRIDSOLVER;
constexpr int NDIM = 3;
using Solver = MultiGridSolver<NDIM, double>;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << std::setprecision(3) << x;
    return os.str();
}

// Deterministic noise in [-1,1] from the global coordinate (the same for any number of tasks)
double noise(const std::array<int, NDIM> & c) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < NDIM; i++) {
        h ^= uint64_t(c[i] + 1000003 * i);
        h *= 1099511628211ull;
        h ^= h >> 29;
    }
    h *= 0x9E3779B97F4A7C15ull;
    return double(h >> 11) / double(1ull << 52) - 1.0;
}

// A zero-mean field: a smooth peak plus noise, restricted to all levels
void make_source(MPIMultiGrid<NDIM, double> & rho, int N, double amp) {
    auto & g = rho.get_grid(0);
    double sum = 0.0;
    for (IndexInt i = 0; i < g.get_NtotLocal(); i++) {
        auto c = g.globalcoord_from_index(i);
        double r2 = 0;
        for (int k = 0; k < NDIM; k++) {
            const double dx = (c[k] + 0.5) / N - 0.4;
            r2 += dx * dx;
        }
        g[i] = 10.0 * std::exp(-r2 / 0.01) + 0.5 * noise(c);
        sum += g[i];
    }
    FML::SumOverTasks(&sum);
    const double mean = sum / std::pow(double(N), NDIM);
    for (IndexInt i = 0; i < g.get_NtotLocal(); i++)
        g[i] = amp * (g[i] - mean);
    rho.restrict_down_all();
}

int main() {
    const int N = 32;

    // 1. Derivatives
    for (int order = 1; order <= 4; order++) {
        Solver s(N, -1, false, true, 4, 4);
        s.set_initial_guess([](std::array<double, NDIM> & x) {
            return std::sin(2 * M_PI * x[0]) + std::sin(4 * M_PI * x[1]) + std::cos(2 * M_PI * x[2]);
        });
        s.get_grid(0).communicate_boundaries();
        double e1 = 0, e2 = 0;
        for (IndexInt i = 0; i < s.get_NtotLocal(0); i++) {
            auto x = s.get_Coordinate(0, i);
            auto g = s.get_Gradient(0, i, order);
            auto g2 = s.get_Gradient2(0, i, order);
            const double d1[3] = {2 * M_PI * std::cos(2 * M_PI * x[0]), 4 * M_PI * std::cos(4 * M_PI * x[1]),
                                  -2 * M_PI * std::sin(2 * M_PI * x[2])};
            const double d2[3] = {-4 * M_PI * M_PI * std::sin(2 * M_PI * x[0]),
                                  -16 * M_PI * M_PI * std::sin(4 * M_PI * x[1]),
                                  -4 * M_PI * M_PI * std::cos(2 * M_PI * x[2])};
            for (int k = 0; k < 3; k++) {
                e1 = std::max(e1, std::abs(g[k] - d1[k]) / (4 * M_PI));
                e2 = std::max(e2, std::abs(g2[k] - d2[k]) / (16 * M_PI * M_PI));
            }
        }
        FML::MaxOverTasks(&e1);
        FML::MaxOverTasks(&e2);
        // Truncation error ~ (k h)^(2 order) with k h = 4 pi / 32
        const double expected = std::pow(4 * M_PI / N, 2 * order);
        check(e1 < expected and e2 < expected,
              "get_Gradient / get_Gradient2 of order " + std::to_string(order) + ": relative errors " + num(e1) +
                  ", " + num(e2) + " (truncation error ~" + num(expected) + ")");
    }

    // 2. Poisson
    {
        MPIMultiGrid<NDIM, double> rho(N, -1, true, 1, 1);
        make_source(rho, N, 1e3);
        Solver s(N, -1, false, true, 1, 1);
        s.set_ngs_sweeps(2, 2, 2);
        s.set_initial_guess(0.0);
        std::vector<double> hist;
        MultiGridConvCrit crit = [&](double r, double, int step) {
            hist.push_back(r);
            return step >= 8;
        };
        MultiGridFunction<NDIM, double> eq = [&](Solver * sol, int level, IndexInt index) {
            auto il = sol->get_neighbor_gridindex(level, index);
            return std::pair<double, double>{sol->get_Laplacian(level, il) - rho.get_y(level, index),
                                             sol->get_derivLaplacian(level, il)};
        };
        s.solve(eq, crit);
        const double factor = std::pow(hist.back() / hist[1], 1.0 / (hist.size() - 2));
        check(factor < 0.2, "Poisson: residual reduced by a factor " + num(factor) + " per V-cycle (2+2 sweeps)");
    }

    // 3. DGP-type equation: D^2 phi - 0.5 alpha (sqrt(1 + 4 (2/3) sigma / alpha^2) - 1) / (2/3) = 0
    //    with sigma = (traceless part of phi_ij)^2 + alpha delta
    {
        MPIMultiGrid<NDIM, double> rho(N, -1, true, 1, 1);
        make_source(rho, N, 0.3);
        for (int l = 0; l < rho.get_Nlevel(); l++)
            rho.get_grid(l).communicate_boundaries();
        Solver s(N, -1, false, true, 1, 1);
        s.set_ngs_sweeps(2, 2, 10);
        s.set_cube_ordering(true);
        s.set_subtract_mean_residual(true);
        s.set_epsilon(1e-8);
        s.set_maxsteps(30);
        s.set_initial_guess(0.0);
        MultiGridConvCrit crit = [&](double r, double r0, int step) { return s.ConvergenceCriterionResidual(r, r0, step); };
        const double alpha = 3.0;
        MultiGridFunction<NDIM, double> eq = [&](Solver * sol, int level, IndexInt index) {
            const double h = sol->get_Gridspacing(level);
            auto cl = sol->get_cube_gridindex(level, index);
            auto F = [&](int k) { return sol->get_Field(level, cl[k]); };
            const int M = 13;
            const double phi = F(M);
            const double pxx = (F(M + 1) + F(M - 1) - 2 * phi) / (h * h);
            const double pyy = (F(M + 3) + F(M - 3) - 2 * phi) / (h * h);
            const double pzz = (F(M + 9) + F(M - 9) - 2 * phi) / (h * h);
            const double pxy = (F(M + 4) + F(M - 4) - F(M + 2) - F(M - 2)) * 0.25 / (h * h);
            const double pyz = (F(M + 12) + F(M - 12) - F(M + 6) - F(M - 6)) * 0.25 / (h * h);
            const double pzx = (F(M + 10) + F(M - 10) - F(M + 8) - F(M - 8)) * 0.25 / (h * h);
            const double lap = pxx + pyy + pzz;
            const double cross = pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pyz * pyz + pzx * pzx);
            const double sigma = cross - lap * lap / 3.0 + alpha * rho.get_y(level, index);
            const double disc = std::max(1.0 + 4.0 * (2.0 / 3.0) * sigma / (alpha * alpha), 0.0);
            return std::pair<double, double>{lap - 0.75 * alpha * (std::sqrt(disc) - 1.0), -6.0 / (h * h)};
        };
        s.solve(eq, crit);
        check(s.get_residual() < 1e-8 and s.get_number_of_vcycles() < 30,
              "DGP-type equation: residual " + num(s.get_residual()) + " after " +
                  std::to_string(s.get_number_of_vcycles()) + " V-cycles (cube ordering, mean subtracted)");
    }

#ifdef USE_FFTW
    // 4. f(R)
    {
        // The positive root of the cubic u^3 + p u + q (q < 0) for the three cases (one real root with p > 0 and
        // p < 0, three real roots)
        double maxerr = 0.0;
        for (double p : {3.0, 1e-3, -0.5, -3.0, -30.0})
            for (double q : {-1e-6, -0.1, -2.0, -50.0}) {
                const double u = FofrSolverCosmology<NDIM>::positive_cubic_root(p, q);
                maxerr = std::max(maxerr, std::abs(u * u * u + p * u + q) / (std::abs(q) + std::abs(p * u)));
                if (not(u > 0.0))
                    maxerr = 1.0;
            }
        check(maxerr < 1e-12, "f(R): positive root of the cubic (relative error " + num(maxerr) + ")");

        // A clustered density field
        FML::GRID::FFTWGrid<NDIM> delta(N, 1, 1);
        double sum = 0.0;
        for (int islice = 0; islice < delta.get_local_nx(); islice++)
            for (auto && ri : delta.get_real_range(islice, islice + 1)) {
                auto c = delta.get_coord_from_index(ri);
                c[0] += delta.get_local_x_start();
                double r2 = 0;
                for (int k = 0; k < NDIM; k++) {
                    const double dx = (c[k] + 0.5) / N - 0.4;
                    r2 += dx * dx;
                }
                const double r = std::sqrt(r2) * 20;
                const double d = 20.0 / (r + 0.1) / (1 + r) / (1 + r) + 0.3 * noise(c);
                delta.set_real_from_index(ri, d);
                sum += d;
            }
        FML::SumOverTasks(&sum);
        const double mean = sum / std::pow(double(N), NDIM);
        for (int islice = 0; islice < delta.get_local_nx(); islice++)
            for (auto && ri : delta.get_real_range(islice, islice + 1))
                delta.set_real_from_index(ri, delta.get_real_from_index(ri) / mean - 1.0);

        // Solve and return the fifth force (finite difference gradient) and the number of V-cycles
        auto solve = [&](std::string method, double fofr0, double n, double a, FML::GRID::FFTWGrid<NDIM> & phi,
                         double eps = 1e-8) {
            FML::GRID::FFTWGrid<NDIM> d = delta;
            for (int islice = 0; islice < d.get_local_nx(); islice++)
                for (auto && ri : d.get_real_range(islice, islice + 1))
                    d.set_real_from_index(ri, d.get_real_from_index(ri) * a);
            FofrSolverCosmology<NDIM> solver(0.3, n, fofr0, 256.0 / 2997.92458, false);
            solver.set_method(method);
            solver.set_ngs_steps(10, 10, 20);
            solver.set_epsilon(eps);
            solver.set_maxsteps(50);
            solver.solve(a, d, phi);
            return solver.get_number_of_vcycles();
        };
        // The rms of the y-derivative and of its difference between two solutions
        auto force_difference = [&](FML::GRID::FFTWGrid<NDIM> & p1, FML::GRID::FFTWGrid<NDIM> & p2) {
            double s2 = 0.0, d2 = 0.0;
            for (int islice = 0; islice < p1.get_local_nx(); islice++)
                for (auto && ri : p1.get_real_range(islice, islice + 1)) {
                    auto c = p1.get_coord_from_index(ri);
                    // The y-derivative (no extra slices needed with MPI)
                    auto cp = c, cm = c;
                    cp[1] = (c[1] + 1) % N;
                    cm[1] = (c[1] - 1 + N) % N;
                    const double f1 = p1.get_real(cp) - p1.get_real(cm), f2 = p2.get_real(cp) - p2.get_real(cm);
                    s2 += f1 * f1;
                    d2 += (f1 - f2) * (f1 - f2);
                }
            FML::SumOverTasks(&s2);
            FML::SumOverTasks(&d2);
            return std::sqrt(d2 / s2);
        };

        FML::GRID::FFTWGrid<NDIM> phi_log, phi_sqrt;
        solve("log", 1e-5, 1.0, 1.0, phi_log);
        const int nv = solve("sqrt", 1e-5, 1.0, 1.0, phi_sqrt);
        const double diff = force_difference(phi_log, phi_sqrt);
        check(diff < 0.01 and nv < 20,
              "f(R) |f_R0| = 1e-5, a = 1: sqrt and log methods agree (rms fifth force difference " + num(diff) +
                  "; sqrt " + std::to_string(nv) + " V-cycles)");

        // Hard cases for the log method (it gives NaN here): check that sqrt converges to a tight tolerance and is
        // the same as with a looser one
        for (double n : {1.0, 2.0}) {
            FML::GRID::FFTWGrid<NDIM> phi1, phi2;
            const int nv1 = solve("sqrt", 1e-6, n, 0.5, phi1, 1e-14);
            solve("sqrt", 1e-6, n, 0.5, phi2, 1e-7);
            const double dd = force_difference(phi1, phi2);
            check(nv1 > 0 and dd < 1e-4,
                  "f(R) |f_R0| = 1e-6, n = " + num(n) + ", a = 0.5 with the sqrt method: the solution with residual < "
                  "1e-7 is within " + num(dd) + " of one with < 1e-14 (" + std::to_string(nv1) + " V-cycles)");
        }
    }
#endif

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All multigrid tests passed\n" : "Some multigrid tests FAILED\n");
    return fails == 0 ? 0 : 1;
}
