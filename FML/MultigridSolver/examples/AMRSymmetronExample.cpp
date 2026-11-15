#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRSymmetronSolver.h>

#include <iomanip>

//==========================================================================================
// The symmetron on an adaptively refined grid (AMRSymmetronSolverCosmology) for a spherical,
// compensated overdensity delta = A exp(-r^2 / 2s^2) (3 - r^2 / s^2) (zero total mass, so the field
// is the cosmic background far away, also in a periodic box). With A = 6 the centre is screened
// ((a_ssb/a)^3 (1 + delta) > 1: the field goes to ~0) and delta >= -1 everywhere.
//
// The reference is the same equation for a spherically symmetric field,
//   chi'' + (2/r) chi' = norm [chi^(N-1) (fac (1 + delta) - 1) + chi^(M-1)],  chi'(0) = 0, chi(R) = chi_bar
// solved with Newton on a fine radial grid (independent of the 3D code). The test compares chi and the
// radial fifth force D(PhiF) of the AMR solution with it, against uniform grids at the coarsest and
// finest level, and checks second order convergence, for (N, M) = (2, 4) and (2, 6).
//
// Run: mpirun -np 4 ./amrsymmetron   (the checksums must not depend on the number of tasks)
//==========================================================================================

using namespace FML::GRID;
constexpr int NDIM = 3;
using Grid = AdaptiveMPIGrid<NDIM, double>;
using Vec = std::array<double, NDIM>;
using Solver = AMRSymmetronSolverCosmology<NDIM>;

static int nfail = 0;

// Model: a 100 Mpc/h box and L = 2 Mpc/h (the range in units of the box is about 0.02), a = 1
const double OmegaM = 0.3, assb = 0.7, beta = 1.0, L_mpch = 2.0, box = 100.0, a_scale = 1.0;
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
// Reference: the radial equation with Newton on n cells of [0, R] (conservative spherical Laplacian,
// chi'(0) = 0, chi = chi_bar at R). Returns chi at the cell centres (r_j = (j + 1/2) R / n)
//==========================================================================================
struct Reference {
    double R{0.5}, dr{0.0};
    std::vector<double> chi;
    double operator()(double r) const {
        const double u = r / dr - 0.5;
        const int j = std::max(0, std::min(int(chi.size()) - 2, int(std::floor(u))));
        const double w = u - j;
        return (1 - w) * chi[j] + w * chi[j + 1];
    }
    double derivative(double r) const {
        const double u = r / dr - 0.5;
        const int j = std::max(0, std::min(int(chi.size()) - 2, int(std::floor(u))));
        return (chi[j + 1] - chi[j]) / dr;
    }
};

Reference reference(const Solver & sym, double Nd, double Md, int n) {
    Reference ref;
    ref.dr = ref.R / n;
    const double dr = ref.dr, f0 = sym.get_phi_background(a_scale), norm = sym.get_norm(a_scale);
    const double fac = std::pow(assb / a_scale, 3);
    const int iN = int(Nd), iM = int(Md);
    auto ipow = [](double x, int k) {
        double r = 1.0;
        for (int i = 0; i < k; i++)
            r *= x;
        return r;
    };
    std::vector<double> chi(n, f0), res(n), lo(n), di(n), up(n), dx(n);
    for (int it = 0; it < 200; it++) {
        double rmax = 0.0;
        for (int j = 0; j < n; j++) {
            const double r = (j + 0.5) * dr, rm = j * dr, rp = (j + 1) * dr;
            const double cm = j > 0 ? chi[j - 1] : chi[j];      // chi'(0) = 0 (rm = 0 anyway)
            const double cp = j + 1 < n ? chi[j + 1] : f0;      // Dirichlet at R + dr/2
            const double lap = (rp * rp * (cp - chi[j]) - rm * rm * (chi[j] - cm)) / (r * r * dr * dr);
            const double f = chi[j], d = delta_of_r(r);
            const double src = norm * (ipow(f, iN - 1) * (fac * (1 + d) - 1) + ipow(f, iM - 1));
            const double dsrc = norm * ((Nd - 1) * ipow(f, iN - 2) * (fac * (1 + d) - 1) + (Md - 1) * ipow(f, iM - 2));
            res[j] = lap - src;
            lo[j] = j > 0 ? rm * rm / (r * r * dr * dr) : 0.0;
            up[j] = j + 1 < n ? rp * rp / (r * r * dr * dr) : 0.0;
            di[j] = -(rp * rp + (j > 0 ? rm * rm : 0.0)) / (r * r * dr * dr) - dsrc;
            rmax = std::max(rmax, std::abs(res[j]));
        }
        if (rmax < 1e-10 * norm)
            break;
        // Solve J dx = -res (Thomas algorithm), damped step
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
        double maxdx = 0.0;
        for (auto x : dx)
            maxdx = std::max(maxdx, std::abs(x));
        const double damp = std::min(1.0, 0.2 / maxdx);
        for (int j = 0; j < n; j++)
            chi[j] += damp * dx[j];
    }
    ref.chi = chi;
    return ref;
}

struct Result {
    double err_chi, max_chi, err_force, checksum;
    int vcycles;
    long long leaves;
};

Result run(double Nd, double Md, int levelmin, int levelmax, double refine_radius, const Reference & ref, bool verbose) {
    Grid grid(levelmin, levelmax, {true, true, true});
    for (int pass = 0; pass < levelmax - levelmin; pass++) {
        for (IndexInt i = 0; i < grid.get_n_local(); i++)
            grid.set_flag(i, radius(grid.get_cell_center(grid.get_cell(i))) < refine_radius ? Grid::REFINE : Grid::KEEP);
        grid.adapt();
        grid.load_balance();
    }

    std::vector<double> delta(grid.get_n_local());
    for (IndexInt i = 0; i < grid.get_n_local(); i++)
        delta[i] = delta_of_r(radius(grid.get_cell_center(grid.get_cell(i))));

    Solver sym(OmegaM, assb, beta, L_mpch, H0Box, verbose, Nd, Md);
    sym.set_epsilon(1e-9);
    sym.set_relative_criterion(true);
    std::vector<double> phi;
    std::vector<Vec> force;
    auto result = sym.solve(a_scale, grid, delta, phi, &force);
    if (not result.converged())
        nfail++;
    const auto & chi = sym.get_field();

    // chi and the radial fifth force against the reference, within 4 s of the centre (volume weighted rms)
    const double forcenorm = sym.get_forcenorm(a_scale);
    double fpeak = 0.0;
    for (double r = 0.001; r < 4 * s; r += 0.001)
        fpeak = std::max(fpeak, std::abs(forcenorm * Nd * std::pow(ref(r), Nd - 1) * ref.derivative(r)));
    double e2 = 0.0, emax = 0.0, ef2 = 0.0, vol = 0.0, checksum = 0.0;
    for (IndexInt i = 0; i < grid.get_n_local(); i++) {
        const auto & c = grid.get_cell(i);
        const auto x = grid.get_cell_center(c);
        const double r = radius(x);
        const double v = std::ldexp(1.0, -NDIM * c.level);
        checksum += v * chi[i] * chi[i];
        if (r > 4 * s or r == 0.0)
            continue;
        const double e = chi[i] - ref(r);
        double fr = 0.0;
        for (int d = 0; d < NDIM; d++)
            fr += force[i][d] * (x[d] - center[d]) / r;
        const double fe = (fr - forcenorm * Nd * std::pow(ref(r), Nd - 1) * ref.derivative(r)) / fpeak;
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

int main(int argc, char ** argv) {
    const bool quick = argc > 1 and std::string(argv[1]) == "quick";
    auto print = [](std::string name, Result r) {
        if (FML::ThisTask == 0)
            std::cout << std::left << std::setw(24) << name << std::right << std::setw(9) << r.leaves << " leaves  chi error rms "
                      << std::scientific << std::setprecision(3) << r.err_chi << "  max " << r.max_chi
                      << "  force error rms " << r.err_force << "  V-cycles " << r.vcycles << "  checksum "
                      << std::setprecision(12) << r.checksum << std::defaultfloat << "\n";
    };
    const double refine_radius = 0.2;
    const int lmin = 4, lmax = quick ? 6 : 7;

    for (auto [Nd, Md] : {std::pair<double, double>{2, 4}, std::pair<double, double>{2, 6}}) {
        Solver sym(OmegaM, assb, beta, L_mpch, H0Box, false, Nd, Md);
        const auto ref = reference(sym, Nd, Md, 20000);
        if (FML::ThisTask == 0)
            std::cout << "\n(N, M) = (" << Nd << ", " << Md << "): background chi = " << sym.get_phi_background(a_scale)
                      << ", chi at the centre = " << ref(0.0) << " (screened), range ~ "
                      << 1.0 / std::sqrt(sym.get_norm(a_scale) * (Md - Nd) * std::pow(sym.get_phi_background(a_scale), Md - 2))
                      << " (box units)\n";

        auto coarse = run(Nd, Md, lmin, lmin, refine_radius, ref, false);
        auto amr = run(Nd, Md, lmin, lmax, refine_radius, ref, false);
        auto fine = run(Nd, Md, lmax, lmax, refine_radius, ref, false);
        auto amr2 = run(Nd, Md, lmin + 1, lmax + 1, refine_radius, ref, false);
        print("Uniform (level " + std::to_string(lmin) + ")", coarse);
        print("Uniform (level " + std::to_string(lmax) + ")", fine);
        print("AMR (levels " + std::to_string(lmin) + "-" + std::to_string(lmax) + ")", amr);
        print("AMR (levels " + std::to_string(lmin + 1) + "-" + std::to_string(lmax + 1) + ")", amr2);
        const double order = std::log2(amr.err_chi / amr2.err_chi);
        if (FML::ThisTask == 0)
            std::cout << "Convergence order of chi when refining every level by one: " << order
                      << " (force: " << std::log2(amr.err_force / amr2.err_force) << ")\n";
        // The AMR solution is as accurate as the uniform grid at its finest level, and converges at second order
        if (amr.err_chi > 1.5 * fine.err_chi or order < 1.7)
            nfail++;
    }

    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail == 0 ? "All tests passed" : std::to_string(nfail) + " tests FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}
