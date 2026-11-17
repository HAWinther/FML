#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRDilatonSolver.h>

#include <iomanip>

//==========================================================================================
// The dilaton on an adaptively refined grid (AMRDilatonSolverCosmology) for a spherical, compensated
// overdensity delta = A exp(-r^2 / 2s^2) (3 - r^2 / s^2) (zero total mass, so the field is the cosmic
// background far away, also in a periodic box; delta >= -0.985 for any A <= 6).
//
// The reference is the same equation for a spherically symmetric field,
//   w'' + (2/r) w' = norm w [(1 + delta) a^-3 - T(ln w)],  w'(0) = 0, w(R) = 1
// solved with Newton on a fine radial grid (independent of the 3D code). The test compares w and the
// radial fifth force D(PhiF) of the AMR solution with it, against uniform grids at the coarsest and
// finest level, and checks second order convergence, for the original dilaton (r = 3/2, the limit p -> 0)
// and r = 1. On one task it also checks that on a uniform grid the solution is that of
// DilatonSolverCosmology (DilatonSolver.h).
//
// Run: mpirun -np 4 ./amrdilaton [quick]   (the checksums must not depend on the number of tasks)
//==========================================================================================

using namespace FML::GRID;
constexpr int NDIM = 3;
using Grid = AdaptiveMPIGrid<NDIM, double>;
using Vec = std::array<double, NDIM>;
using Solver = AMRDilatonSolverCosmology<NDIM>;

static int nfail = 0;

// Model: a 100 Mpc/h box, xi = 1e-3 (range c/H0 xi = 3 Mpc/h, about 0.03 box), a = 1. With A2 = 5e5 (s = 1.35) the
// centre (1 + delta = 19) is screened: w ~ 0.1
const double OmegaM = 0.3, A2 = 5e5, beta0 = 0.5, xi = 1e-3, box = 100.0, a_scale = 1.0;
const double H0Box = box / FML::UTILS::SI::c_over_H0_mpch;
// Profile
const double A = 6.0, s = 0.05;
const Vec center{0.5, 0.5, 0.5};

double radius(const Vec & x) {
    double r2 = 0.0;
    for (int d = 0; d < NDIM; d++)
        r2 += (x[d] - center[d]) * (x[d] - center[d]);
    return std::sqrt(r2);
}
double delta_of_r(double r) { return A * std::exp(-r * r / (2 * s * s)) * (3.0 - r * r / (s * s)); }

//==========================================================================================
// Reference: the radial equation with (damped) Newton on n cells of [0, R] (conservative spherical Laplacian,
// w'(0) = 0, w = 1 at R). Returns w at the cell centres (r_j = (j + 1/2) R / n)
//==========================================================================================
struct Reference {
    double R{0.5}, dr{0.0};
    std::vector<double> w;
    double operator()(double r) const {
        const double u = r / dr - 0.5;
        const int j = std::max(0, std::min(int(w.size()) - 2, int(std::floor(u))));
        const double t = u - j;
        return (1 - t) * w[j] + t * w[j + 1];
    }
    double derivative(double r) const {
        const double u = r / dr - 0.5;
        const int j = std::max(0, std::min(int(w.size()) - 2, int(std::floor(u))));
        return (w[j + 1] - w[j]) / dr;
    }
};

Reference reference(const Solver & dil, int n) {
    Reference ref;
    ref.dr = ref.R / n;
    const double dr = ref.dr, norm = dil.get_norm(a_scale), ainv3 = std::pow(a_scale, -3);
    const double Ea = dil.get_Ea(a_scale), sp = dil.get_s(), p = dil.get_p();
    std::vector<double> w(n, 1.0), res(n), lo(n), di(n), up(n), dx(n);
    for (int it = 0; it < 500; it++) {
        double rmax = 0.0;
        for (int j = 0; j < n; j++) {
            const double r = (j + 0.5) * dr, rm = j * dr, rp = (j + 1) * dr;
            const double wm = j > 0 ? w[j - 1] : w[j];
            const double wp = j + 1 < n ? w[j + 1] : 1.0;
            const double lap = (rp * rp * (wp - w[j]) - rm * rm * (w[j] - wm)) / (r * r * dr * dr);
            double T, dT;
            Solver::Uniform::potential_term(std::log(w[j]), Ea, sp, p, T, dT);
            const double S = (1 + delta_of_r(r)) * ainv3 - T;
            res[j] = lap - norm * w[j] * S;
            lo[j] = j > 0 ? rm * rm / (r * r * dr * dr) : 0.0;
            up[j] = j + 1 < n ? rp * rp / (r * r * dr * dr) : 0.0;
            di[j] = -(rp * rp + (j > 0 ? rm * rm : 0.0)) / (r * r * dr * dr) - norm * (S - dT);
            rmax = std::max(rmax, std::abs(res[j]));
        }
        if (rmax < 1e-10 * norm)
            break;
        std::vector<double> c(n), g(n);
        c[0] = up[0] / di[0];
        g[0] = -res[0] / di[0];
        for (int j = 1; j < n; j++) {
            const double m = di[j] - lo[j] * c[j - 1];
            c[j] = up[j] / m;
            g[j] = (-res[j] - lo[j] * g[j - 1]) / m;
        }
        dx[n - 1] = g[n - 1];
        for (int j = n - 2; j >= 0; j--)
            dx[j] = g[j] - c[j] * dx[j + 1];
        // Damped: w changes by at most a factor ~1.5
        double maxrel = 0.0;
        for (int j = 0; j < n; j++)
            maxrel = std::max(maxrel, std::abs(dx[j]) / w[j]);
        const double damp = std::min(1.0, 0.5 / maxrel);
        for (int j = 0; j < n; j++)
            w[j] += damp * dx[j];
    }
    ref.w = w;
    return ref;
}

struct Result {
    double err_w, max_w, err_force, checksum;
    int vcycles;
    long long leaves;
};

Grid make_grid(int levelmin, int levelmax, double refine_radius) {
    Grid grid(levelmin, levelmax, {true, true, true});
    for (int pass = 0; pass < levelmax - levelmin; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++)
            grid.set_flag(i, radius(grid.get_cell_center(grid.get_cell(i))) < refine_radius ? Grid::REFINE : Grid::KEEP);
        grid.adapt();
        grid.load_balance();
    }
    return grid;
}

Result run(double r_model, int levelmin, int levelmax, double refine_radius, const Reference & ref, bool verbose) {
    auto grid = make_grid(levelmin, levelmax, refine_radius);
    std::vector<double> delta(grid.get_n_local());
    for (IndexInt i = 0; i < grid.get_n_local(); i++)
        delta[i] = delta_of_r(radius(grid.get_cell_center(grid.get_cell(i))));

    Solver dil(OmegaM, A2, beta0, r_model, xi, H0Box, verbose);
    dil.set_epsilon(1e-9);
    dil.set_relative_criterion(true);
    std::vector<double> phi;
    std::vector<Vec> force;
    auto result = dil.solve(a_scale, grid, delta, phi, &force);
    if (not result.converged())
        nfail++;
    const auto & w = dil.get_field();

    // w and the radial fifth force D(PhiF) = 2 PhiF_bar w w' against the reference, within 4 s of the centre
    const double phifbar = dil.get_phif_background(a_scale);
    auto fref = [&](double r) { return 2.0 * phifbar * ref(r) * ref.derivative(r); };
    double fpeak = 0.0;
    for (double r = 0.001; r < 4 * s; r += 0.001)
        fpeak = std::max(fpeak, std::abs(fref(r)));
    double e2 = 0.0, emax = 0.0, ef2 = 0.0, vol = 0.0, checksum = 0.0;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        const auto x = grid.get_cell_center(c);
        const double r = radius(x);
        const double v = std::ldexp(1.0, -NDIM * c.level);
        checksum += v * w[i] * w[i];
        if (r > 4 * s or r == 0.0)
            continue;
        const double e = w[i] - ref(r);
        double fr = 0.0;
        for (int d = 0; d < NDIM; d++)
            fr += force[i][d] * (x[d] - center[d]) / r;
        const double fe = (fr - fref(r)) / fpeak;
        e2 += v * e * e;
        ef2 += v * fe * fe;
        vol += v;
        emax = std::max(emax, std::abs(e));
    }
    FML::SumOverTasks(&e2);
    FML::SumOverTasks(&ef2);
    FML::SumOverTasks(&vol);
    FML::MaxOverTasks(&emax);
    FML::SumOverTasks(&checksum);
    return {std::sqrt(e2 / vol), emax, std::sqrt(ef2 / vol), checksum, result.iterations, grid.get_n_global()};
}

// On a uniform grid (one task): AMRDilatonSolverCosmology against DilatonSolverCosmology (max relative difference of
// PhiF). A strongly screened case (A2 = 5e6, at a = 0.5) with the field spanning a few e-folds
double compare_with_uniform(int level) {
    const double A2u = 5e6, au = 0.5;
    const int N = 1 << level;
    auto grid = make_grid(level, level, 0.0);
    std::vector<double> delta(grid.get_n_local());
    FFTWGrid<NDIM> dgrid(N), phigrid(N);
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        delta[i] = delta_of_r(radius(grid.get_cell_center(c)));
        std::array<int, NDIM> coord;
        for (int d = 0; d < NDIM; d++)
            coord[d] = int(c.anchor[d] >> (Grid::LMAX - c.level));
        dgrid.set_real(coord, delta[i]);
    }
    Solver amr(OmegaM, A2u, beta0, 1.0, xi, H0Box, false);
    amr.set_epsilon(1e-11);
    amr.set_relative_criterion(true);
    std::vector<double> phi;
    amr.solve(au, grid, delta, phi);

    DilatonSolverCosmology<NDIM> uni(OmegaM, A2u, beta0, 1.0, xi, H0Box, false);
    uni.set_epsilon(1e-11);
    uni.set_relative_criterion(true);
    uni.solve(au, dgrid, phigrid);

    double maxdiff = 0.0, umin = 1e100, umax = -1e100;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        std::array<int, NDIM> coord;
        for (int d = 0; d < NDIM; d++)
            coord[d] = int(c.anchor[d] >> (Grid::LMAX - c.level));
        const double pu = phigrid.get_real(coord);
        maxdiff = std::max(maxdiff, std::abs(phi[i] / pu - 1.0));
        umin = std::min(umin, std::log(amr.get_field()[i]));
        umax = std::max(umax, std::log(amr.get_field()[i]));
    }
    std::cout << "Uniform " << N << "^3 grid (A2 = " << A2u << ", a = " << au << ", ln w in [" << umin << ", " << umax
              << "], " << amr.get_number_of_vcycles() << " V-cycles AMR, " << uni.get_number_of_vcycles()
              << " uniform): max |PhiF_AMR / PhiF_uniform - 1| = " << maxdiff << "\n";
    return maxdiff;
}

//==========================================================================================
// A first solve from the background w = 1 in a strongly screened, dense object (models D4 and C4 of Brax et al.
// 2012 at the redshift where AMRCode starts them; a narrow peak with 1 + delta ~ 2e4 at the centre, as a halo in an
// N-body run), resolved at level 9 while a small region elsewhere goes to level 10, so the peak's leaves are relaxed
// on a coarse multigrid level: w drops from 1 to ~e^-40 or below there in the first V-cycle. That value used to come
// back to the leaves as the correction 1 + (e^-40 - 1) = 0 (exactly, in double precision) and the solve gave NaN; the
// cells that exist on both levels now get the coarse value itself
//==========================================================================================
struct StrongResult {
    bool converged;
    long long nbad;
    double umin;
    int vcycles;
};
StrongResult strong_screening(double r_model, double xi_model, double a) {
    const double box_s = 256.0, H0Box_s = box_s / FML::UTILS::SI::c_over_H0_mpch;
    const double Ap = 7000.0, sp = 0.004;
    const Vec other{0.2, 0.2, 0.2};
    Grid grid(6, 10, {true, true, true});
    for (int pass = 0; pass < 4; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            const auto & c = grid.get_cell(i);
            const auto x = grid.get_cell_center(c);
            double r2 = 0.0;
            for (int d = 0; d < NDIM; d++)
                r2 += (x[d] - other[d]) * (x[d] - other[d]);
            const bool refine = (radius(x) < 0.03 and c.level < 9) or r2 < 0.01 * 0.01 * 16;
            grid.set_flag(i, refine ? Grid::REFINE : Grid::KEEP);
        }
        grid.adapt();
        grid.load_balance();
    }
    std::vector<double> delta(grid.get_n_local());
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const double r = radius(grid.get_cell_center(grid.get_cell(i)));
        delta[i] = Ap * std::exp(-r * r / (2 * sp * sp)) * (3.0 - r * r / (sp * sp));
    }
    Solver dil(OmegaM, 1e5, beta0, r_model, xi_model, H0Box_s, false);
    dil.set_epsilon(1e-6);
    dil.set_relative_criterion(true);
    std::vector<double> phi;
    const auto result = dil.solve(a, grid, delta, phi);
    const auto & w = dil.get_field();
    long long nbad = 0;
    double umin = 1e300;
    for (auto x : w) {
        if (not(x > 0.0 and std::isfinite(x)))
            nbad++;
        else
            umin = std::min(umin, std::log(x));
    }
    FML::SumOverTasks(&nbad);
    FML::MinOverTasks(&umin);
    return {result.converged(), nbad, umin, result.iterations};
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 and std::string(argv[1]) == "quick";
    auto print = [](std::string name, Result r) {
        if (FML::ThisTask == 0)
            std::cout << std::left << std::setw(24) << name << std::right << std::setw(9) << r.leaves << " leaves  w error rms "
                      << std::scientific << std::setprecision(3) << r.err_w << "  max " << r.max_w
                      << "  force error rms " << r.err_force << "  V-cycles " << r.vcycles << "  checksum "
                      << std::setprecision(12) << r.checksum << std::defaultfloat << "\n";
    };
    const double refine_radius = 0.2;
    const int lmin = 4, lmax = quick ? 6 : 7;

    if (FML::NTasks == 1) {
        if (compare_with_uniform(5) > 1e-8)
            nfail++;
    }

    for (double r_model : {1.5, 1.0}) {
        Solver dil(OmegaM, A2, beta0, r_model, xi, H0Box, false);
        const auto ref = reference(dil, 20000);
        if (FML::ThisTask == 0)
            std::cout << "\nr = " << r_model << ": s = " << dil.get_s() << ", w at the centre = " << ref(0.0)
                      << " (screened), range = " << dil.get_range(a_scale) << " (box units)\n";

        auto coarse = run(r_model, lmin, lmin, refine_radius, ref, false);
        auto amr = run(r_model, lmin, lmax, refine_radius, ref, false);
        auto fine = run(r_model, lmax, lmax, refine_radius, ref, false);
        auto amr2 = run(r_model, lmin + 1, lmax + 1, refine_radius, ref, false);
        print("Uniform (level " + std::to_string(lmin) + ")", coarse);
        print("Uniform (level " + std::to_string(lmax) + ")", fine);
        print("AMR (levels " + std::to_string(lmin) + "-" + std::to_string(lmax) + ")", amr);
        print("AMR (levels " + std::to_string(lmin + 1) + "-" + std::to_string(lmax + 1) + ")", amr2);
        const double order = std::log2(amr.err_w / amr2.err_w);
        if (FML::ThisTask == 0)
            std::cout << "Convergence order of w when refining every level by one: " << order
                      << " (force: " << std::log2(amr.err_force / amr2.err_force) << ")\n";
        // The AMR solution is as accurate as the uniform grid at its finest level, and converges at second order
        if (amr.err_w > 1.5 * fine.err_w or order < 1.7)
            nfail++;
    }

    // A first solve in a strongly screened peak resolved below the finest level (D4: r = 1, xi = 3e-3 at z = 1.2;
    // C4: r = 0.4, xi = 1e-3 at z = 3.2)
    if (FML::ThisTask == 0)
        std::cout << "\nFirst solve from w = 1 in a peak with 1 + delta = 2e4 at level 9 (levels 6-10, 256 Mpc/h):\n";
    for (auto [name, r_model, xi_model, a] : {std::tuple{"D4", 1.0, 3e-3, 0.46}, std::tuple{"C4", 0.4, 1e-3, 0.24}}) {
        const auto res = strong_screening(r_model, xi_model, a);
        if (FML::ThisTask == 0)
            std::cout << "  " << name << ": " << (res.converged ? "converged" : "NOT converged") << " in " << res.vcycles
                      << " V-cycles, " << res.nbad << " leaves with w not positive and finite, min ln w " << res.umin
                      << "\n";
        if (not res.converged or res.nbad > 0)
            nfail++;
    }

    // The cell solve (DilatonSolverCosmology::solve_cell_affine) for the inputs of a cell of a coarse multigrid level in
    // a C4 N-body run (64^3 in 64 Mpc/h, z = 1.4), started next to its root with the bracket [-700, 1.86]: it used to
    // bisect to u ~ -350 and run out of iterations (Newton steps of ~+1) at u = -153.3 instead of the root -2.59. The
    // multigrid then did not converge (and depended on the number of tasks)
    {
        const double uroot = DilatonSolverCosmology<NDIM>::solve_cell_affine(-2.59184989785675457, 6.03550812143566436e+05, 1.0,
                                                                            6.29145600000000000e+06, 2.53893772930385632e+05,
                                                                            6.96864368637928067, -2.73972283981011788, 0.27,
                                                                            -2.2, -700.0, 1.86245243947600470);
        double T, dT;
        DilatonSolverCosmology<NDIM>::potential_term(uroot, -2.73972283981011788, 0.27, -2.2, T, dT);
        const double F = 6.03550812143566436e+05 * std::exp(-uroot) - 6.29145600000000000e+06 -
                         6.96864368637928067 * (2.53893772930385632e+05 - T);
        const bool ok = std::abs(uroot + 2.59185) < 1e-4 and std::abs(F) < 1e-6 * 6.29145600000000000e+06;
        if (FML::ThisTask == 0)
            std::cout << "\nCell solve far from the bracket edge: u = " << std::setprecision(10) << uroot
                      << " (root -2.59185), F(u) = " << F << (ok ? "  OK" : "  FAILED") << std::defaultfloat << "\n";
        if (not ok)
            nfail++;
    }

    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail == 0 ? "All tests passed" : std::to_string(nfail) + " tests FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}
