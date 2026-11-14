#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>

#include <cstring>

//==========================================================================================
// Multigrid on an adaptively refined grid: the Poisson equation D^2 phi = delta for a
// Gaussian potential phi (periodic box) and a non-linear equation D^2 u - lambda u^3 = s
// (Dirichlet boundary), both with known solutions. The grid is refined around the peak.
//
// Run with any number of tasks: mpirun -np 4 ./amrpoisson [2|3]
// The checksums printed must be the same (to round-off) for any number of tasks.
//==========================================================================================

using namespace FML::GRID;
using namespace FML::SOLVERS::AMRMULTIGRIDSOLVER;

static int nfail = 0;

template <int NDIM>
std::array<double, NDIM> filled(double v) {
    std::array<double, NDIM> a;
    a.fill(v);
    return a;
}

template <int NDIM>
struct Gaussian {
    std::array<double, NDIM> center;
    double sigma;
    double r2(const std::array<double, NDIM> & x) const {
        double r2 = 0.0;
        for (int d = 0; d < NDIM; d++)
            r2 += (x[d] - center[d]) * (x[d] - center[d]);
        return r2;
    }
    double phi(const std::array<double, NDIM> & x) const { return std::exp(-r2(x) / (2 * sigma * sigma)); }
    double laplacian(const std::array<double, NDIM> & x) const {
        const double s2 = sigma * sigma;
        return phi(x) * (r2(x) / (s2 * s2) - NDIM / s2);
    }
};

// Refine every cell within a distance radius of the peak (one level per pass) and load balance
template <int NDIM>
AdaptiveMPIGrid<NDIM, double> make_grid(int levelmin, int levelmax, bool periodic, const Gaussian<NDIM> & g, double radius) {
    AdaptiveMPIGrid<NDIM, double> grid(levelmin, levelmax, periodic);
    for (int pass = 0; pass < levelmax - levelmin; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            const auto & c = grid.get_cell(i);
            const double dist = std::sqrt(g.r2(grid.get_cell_center(c)));
            grid.set_flag(i, dist < radius ? AdaptiveMPIGrid<NDIM, double>::REFINE : AdaptiveMPIGrid<NDIM, double>::KEEP);
        }
        grid.adapt();
        grid.load_balance();
    }
    return grid;
}

template <int NDIM>
struct Errors {
    double rms, max, checksum;
};

// Error of phi (up to a constant if subtract_mean) weighted by volume
template <int NDIM>
Errors<NDIM> errors(const AdaptiveMPIGrid<NDIM, double> & grid,
                    const std::vector<double> & phi,
                    const Gaussian<NDIM> & g,
                    bool subtract_mean) {
    double sum[3] = {0.0, 0.0, 0.0}; // mean numerical, mean exact, checksum
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        const double vol = std::ldexp(1.0, -NDIM * c.level);
        sum[0] += vol * phi[i];
        sum[1] += vol * g.phi(grid.get_cell_center(c));
        sum[2] += vol * phi[i] * phi[i];
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, sum, 3, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    const double shift = subtract_mean ? sum[1] - sum[0] : 0.0;
    double err2 = 0.0, errmax = 0.0;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        const double vol = std::ldexp(1.0, -NDIM * c.level);
        const double e = phi[i] + shift - g.phi(grid.get_cell_center(c));
        err2 += vol * e * e;
        errmax = std::max(errmax, std::abs(e));
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &err2, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
    MPI_Allreduce(MPI_IN_PLACE, &errmax, 1, MPI_DOUBLE, MPI_MAX, FML::communicator);
#endif
    return {std::sqrt(err2), errmax, sum[2]};
}

//==========================================================================================
// Poisson: D^2 phi = delta in a periodic box, delta = D^2 phi_exact minus its mean
//==========================================================================================
template <int NDIM>
Errors<NDIM> run_poisson(int levelmin, int levelmax, double radius, bool verbose) {
    const Gaussian<NDIM> g{filled<NDIM>(0.5), 0.04};
    auto grid = make_grid<NDIM>(levelmin, levelmax, true, g, radius);
    if (verbose)
        grid.info();

    // The source with zero volume weighted mean (the solution is only fixed up to a constant)
    std::vector<double> delta(grid.get_n_local());
    double mean = 0.0;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        delta[i] = g.laplacian(grid.get_cell_center(c));
        mean += std::ldexp(1.0, -NDIM * c.level) * delta[i];
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &mean, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    for (auto & d : delta)
        d -= mean;

    AMRMultiGridSolver<NDIM> solver(grid, -1, verbose);
    const int id = solver.add_external_field(delta);
    AMREquation<NDIM> equation = [id](const AMRMultiGridSolver<NDIM> & sol, int level, IndexInt i) {
        auto [lap, dlap] = sol.get_laplacian(level, i);
        return std::pair<double, double>{lap - sol.get_external_field(id, level, i), dlap};
    };
    solver.set_epsilon(1e-10);
    auto result = solver.solve(equation);
    if (not result.converged())
        nfail++;
    if (FML::ThisTask == 0 and verbose)
        std::cout << "Converged in " << result.iterations << " V-cycles, residual reduced by "
                  << result.residual / result.initial_residual << "\n";
    return errors<NDIM>(grid, solver.get_solution(), g, true);
}

//==========================================================================================
// Non-linear: D^2 u - lambda u^3 = s, Dirichlet boundary u = u_exact on the box faces
//==========================================================================================
template <int NDIM>
Errors<NDIM> run_nonlinear(int levelmin, int levelmax, double radius, bool verbose) {
    const double lambda = 500.0;
    Gaussian<NDIM> g{filled<NDIM>(0.4), 0.05};
    g.center[0] = 0.55;
    auto grid = make_grid<NDIM>(levelmin, levelmax, false, g, radius);
    if (verbose)
        grid.info();

    std::vector<double> s(grid.get_n_local());
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        auto x = grid.get_cell_center(grid.get_cell(i));
        s[i] = g.laplacian(x) - lambda * std::pow(g.phi(x), 3);
    }

    AMRMultiGridSolver<NDIM> solver(grid, -1, verbose);
    const int id = solver.add_external_field(s);
    solver.set_boundary_value([&](const std::array<double, NDIM> & x) { return g.phi(x); });
    AMREquation<NDIM> equation = [id, lambda](const AMRMultiGridSolver<NDIM> & sol, int level, IndexInt i) {
        auto [lap, dlap] = sol.get_laplacian(level, i);
        const double u = sol.get_field(level, i);
        return std::pair<double, double>{lap - lambda * u * u * u - sol.get_external_field(id, level, i),
                                         dlap - 3 * lambda * u * u};
    };
    solver.set_epsilon(1e-10);
    auto result = solver.solve(equation);
    if (not result.converged())
        nfail++;
    if (FML::ThisTask == 0 and verbose)
        std::cout << "Converged in " << result.iterations << " V-cycles, residual reduced by "
                  << result.residual / result.initial_residual << "\n";
    return errors<NDIM>(grid, solver.get_solution(), g, false);
}

template <int NDIM>
void run_all(int lmin, int lmax) {
    auto print = [](std::string name, Errors<NDIM> e) {
        if (FML::ThisTask == 0)
            std::cout << std::left << std::setw(24) << name << " rms error " << std::scientific << std::setprecision(3)
                      << e.rms << "  max error " << e.max << "  checksum " << std::setprecision(12) << e.checksum
                      << std::defaultfloat << "\n";
    };
    auto levels = [](int a, int b) { return "(levels " + std::to_string(a) + "-" + std::to_string(b) + ")"; };

    // Compare the AMR solution to the uniform grids at the coarsest and finest level and check that it
    // converges at second order when every level is refined by one
    auto compare = [&](auto run, double radius) {
        auto e_amr = run(lmin, lmax, radius, true);
        if (FML::ThisTask == 0)
            std::cout << "\n";
        auto e_coarse = run(lmin, lmin, radius, false);
        auto e_fine = run(lmax, lmax, radius, false);
        auto e_amr2 = run(lmin + 1, lmax + 1, radius, false);
        print("Uniform " + levels(lmin, lmin), e_coarse);
        print("Uniform " + levels(lmax, lmax), e_fine);
        print("AMR     " + levels(lmin, lmax), e_amr);
        print("AMR     " + levels(lmin + 1, lmax + 1), e_amr2);
        const double ratio = e_amr.rms / e_amr2.rms;
        if (FML::ThisTask == 0)
            std::cout << "Error ratio when refining every level by one: rms " << ratio << " max "
                      << e_amr.max / e_amr2.max << " (4 for second order)\n";
        if (e_amr.max > 1.5 * e_fine.max or ratio < 3.0)
            nfail++;
    };

    if (FML::ThisTask == 0)
        std::cout << "\n====== " << NDIM << "D Poisson D^2 phi = delta (periodic) ======\n";
    compare([](int a, int b, double r, bool v) { return run_poisson<NDIM>(a, b, r, v); }, 0.2);

    if (FML::ThisTask == 0)
        std::cout << "\n====== " << NDIM << "D D^2 u - lambda u^3 = s (Dirichlet) ======\n";
    compare([](int a, int b, double r, bool v) { return run_nonlinear<NDIM>(a, b, r, v); }, 0.25);
}

int main(int argc, char ** argv) {
    const int ndim = argc > 1 ? std::atoi(argv[1]) : 2;
    if (ndim == 2)
        run_all<2>(4, 8);
    else
        run_all<3>(3, 6);
    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail == 0 ? "All tests passed" : std::to_string(nfail) + " tests FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}
