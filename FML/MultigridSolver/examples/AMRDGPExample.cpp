#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRDGPSolver.h>

#include <iomanip>

//==========================================================================================
// The DGP fifth force on an adaptively refined grid (AMRDGPSolverCosmology) for a spherical,
// compensated overdensity, where the equation has an exact solution (Vainshtein screening):
//
//   M(r) = int_0^r delta r'^2 dr' = A r^3 exp(-r^2 / 2s^2)   (zero total mass: periodic box is fine)
//   delta(r) = A exp(-r^2 / 2s^2) (3 - r^2 / s^2)
//   r^2 phi' + (2/alpha) r phi'^2 = OmegaM a M / beta
//   => phi'(r) = (alpha r / 4) [ sqrt(1 + 8 OmegaM a M / (alpha beta r^3)) - 1 ]
//
// The linear (unscreened) force is OmegaM a M / (beta r^2). The test compares the radial fifth force
// of the AMR solution with the exact one, with the uniform grids at the coarsest and finest level,
// and checks second order convergence. Run: mpirun -np 4 ./amrdgp
//==========================================================================================

using namespace FML::GRID;
constexpr int NDIM = 3;
using Grid = AdaptiveMPIGrid<NDIM, double>;
using Vec = std::array<double, NDIM>;

static int nfail = 0;

// Model and profile
// (OmegaM a / (beta alpha) < 1/3 as for DGP in a cosmology: with delta >= -1 the discriminant is then positive)
const double OmegaM = 1.0, a_scale = 1.0, beta = 2.2, rcH0 = 2.0, H0Box = 1.0;
const double alpha = 3.0 * beta * std::pow(a_scale, 4) / (rcH0 * rcH0);
const double A = 5.0, s = 0.05;
const Vec center{0.5, 0.5, 0.5};

double radius(const Vec & x) {
    double r2 = 0.0;
    for (int d = 0; d < NDIM; d++)
        r2 += (x[d] - center[d]) * (x[d] - center[d]);
    return std::sqrt(r2);
}
double delta_of_r(double r) { return A * std::exp(-r * r / (2 * s * s)) * (3.0 - r * r / (s * s)); }
double force_exact(double r) {
    return alpha * r / 4.0 * (std::sqrt(1.0 + 8.0 * OmegaM * a_scale * A * std::exp(-r * r / (2 * s * s)) / (alpha * beta)) - 1.0);
}
double force_linear(double r) { return OmegaM * a_scale / beta * A * r * std::exp(-r * r / (2 * s * s)); }

struct Result {
    double rms, max, linear_rms, checksum;
    int vcycles;
};

Result run(int levelmin, int levelmax, double refine_radius, bool verbose) {
    Grid grid(levelmin, levelmax, true);
    for (int pass = 0; pass < levelmax - levelmin; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++)
            grid.set_flag(i, radius(grid.get_cell_center(grid.get_cell(i))) < refine_radius ? Grid::REFINE : Grid::KEEP);
        grid.adapt();
        grid.load_balance();
    }
    if (verbose)
        grid.info();

    std::vector<double> delta(grid.get_n_local());
    for (IndexInt i = 0; i < grid.get_n_local(); i++)
        delta[i] = delta_of_r(radius(grid.get_cell_center(grid.get_cell(i))));

    AMRDGPSolverCosmology<NDIM> dgp(OmegaM, rcH0, beta, H0Box, verbose);
    dgp.set_epsilon(1e-9);
    dgp.set_relative_criterion(true);
    std::vector<double> phi;
    std::vector<Vec> force;
    auto result = dgp.solve(a_scale, grid, delta, phi, &force);
    if (not result.converged())
        nfail++;

    // Radial force (of the field phi = potential / 0.5) vs the exact one, within 3 s of the center
    const double fmax = force_exact(s); // about the peak
    double err2 = 0.0, errmax = 0.0, vol = 0.0, lin2 = 0.0, checksum = 0.0;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        const auto x = grid.get_cell_center(c);
        const double r = radius(x);
        const double v = std::ldexp(1.0, -NDIM * c.level);
        checksum += v * phi[i] * phi[i];
        if (r > 3 * s or r == 0.0)
            continue;
        double fr = 0.0;
        for (int d = 0; d < NDIM; d++)
            fr += force[i][d] / 0.5 * (x[d] - center[d]) / r;
        const double e = (fr - force_exact(r)) / fmax;
        err2 += v * e * e;
        lin2 += v * std::pow((force_linear(r) - force_exact(r)) / fmax, 2);
        vol += v;
        errmax = std::max(errmax, std::abs(e));
    }
    FML::SumOverTasks(&err2);
    FML::SumOverTasks(&vol);
    FML::MaxOverTasks(&errmax);
    FML::SumOverTasks(&checksum);
    FML::SumOverTasks(&lin2);
    return {std::sqrt(err2 / vol), errmax, std::sqrt(lin2 / vol), checksum, result.iterations};
}

int main() {
    auto print = [](std::string name, Result r) {
        if (FML::ThisTask == 0)
            std::cout << std::left << std::setw(22) << name << " force error rms " << std::scientific
                      << std::setprecision(3) << r.rms << "  max " << r.max << "  V-cycles " << r.vcycles
                      << "  checksum " << std::setprecision(12) << r.checksum << std::defaultfloat << "\n";
    };
    const double refine_radius = 0.2;
    const int lmin = 3, lmax = 6;

    auto amr = run(lmin, lmax, refine_radius, true);
    auto coarse = run(lmin, lmin, refine_radius, false);
    auto fine = run(lmax, lmax, refine_radius, false);
    auto amr2 = run(lmin + 1, lmax + 1, refine_radius, false);

    if (FML::ThisTask == 0)
        std::cout << "\nScreening: exact / linear force at r = s/2 is " << force_exact(0.5 * s) / force_linear(0.5 * s)
                  << "; the linear force is off by " << amr.linear_rms << " (rms, in units of the peak force)\n";
    print("Uniform (level " + std::to_string(lmin) + ")", coarse);
    print("Uniform (level " + std::to_string(lmax) + ")", fine);
    print("AMR (levels " + std::to_string(lmin) + "-" + std::to_string(lmax) + ")", amr);
    print("AMR (levels " + std::to_string(lmin + 1) + "-" + std::to_string(lmax + 1) + ")", amr2);
    const double order = std::log2(amr.rms / amr2.rms);
    if (FML::ThisTask == 0)
        std::cout << "Convergence order when refining every level by one: " << order << "\n";
    if (amr.rms > 1.5 * fine.rms or order < 1.7 or amr.rms > 0.02)
        nfail++;

    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail == 0 ? "All tests passed" : std::to_string(nfail) + " tests FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}
