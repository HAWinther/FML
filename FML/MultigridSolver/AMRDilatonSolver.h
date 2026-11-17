#ifndef AMRDILATONSOLVER_HEADER
#define AMRDILATONSOLVER_HEADER

#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>
#include <FML/MultigridSolver/DilatonSolver.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

//======================================================================
///
/// The (generalised) dilaton fifth-force equation on an adaptively refined grid (AdaptiveMPIGrid), the AMR version of
/// DilatonSolverCosmology (DilatonSolver.h; Brax, Davis, Li, Winther & Zhao 2012, arXiv:1206.3568): the same
/// parameters (A2, beta0, r <= 3/2, xi = H0 / m0), equation, units and returned potential. With w = phi / phi_bar(a)
/// = e^u (box units, D^2 the flux Laplacian of AMRMultiGridSolver)
///
///   D^2 w = norm w [ (1 + delta) a^-3 - T(u) ],   T = (1 + p y)^(-3/p),   y = (a^p - 1)/p + u / s
///
/// norm = 3 A2 OmegaM a^2 (H0 Box / c)^2, s = 9 OmegaM A2 xi^2, p = 2r - 3. The returned fifth-force potential is
/// PhiF = a^2 beta(a)^2 w^2 / (2 A2 (H0 Box)^2) (total force D(PhiGR + PhiF) with PhiGR = a^2 Phi / (H0 Box)^2).
///
/// As in DilatonSolverCosmology the multigrid works with w (linear Laplacian: the coarse-grid corrections are averages
/// of phi) while the Gauss-Seidel update solves the equation in the cell for u = ln w (DilatonSolverCosmology::
/// solve_cell_affine, set as the custom update of AMRMultiGridSolver): monotone after dividing by w, so the root is
/// unique and w > 0 even where w spans many orders of magnitude. That needs a positive sum over the neighbours of a
/// cell: the coarse gradient in the coarse/fine ghost values is limited (AMRMultiGridSolver::set_coarse_fine_limiter;
/// unlimited, a ghost next to a screened halo can be negative, and the cell then has no positive solution). The root
/// is bracketed by the minima of the effective potential for the smallest and the largest density (maximum principle;
/// widened a little).
///
///   AMRDilatonSolverCosmology<3> dil(OmegaM, A2, beta0, r, xi, H0Box, verbose);
///   std::vector<double> phi;                   // per local leaf of the grid
///   dil.solve(a, grid, delta, phi);            // delta per local leaf of the grid
///   dil.solve(a, grid, delta, phi, &force);    // also the fifth force D(PhiF) = 2 PhiF_bar w D(w) per leaf
///   dil.get_field();                           // w of the last solve (it also starts the next solve on the same
///                                              // leaves, see set_previous_field)
///
/// For a code that keeps its own AMRMultiGridSolver (e.g. AMRCode, with partial solves) the equation and the update are
/// available as get_equation / get_update (the density as an external field of that solver; also call
/// set_coarse_fine_limiter(true) on that solver).
///
//======================================================================
template <int NDIM>
class AMRDilatonSolverCosmology {
  public:
    using Solver = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMRMultiGridSolver<NDIM>;
    using Equation = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMREquation<NDIM>;
    using Update = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMRUpdateFunction<NDIM>;
    using SolveResult = FML::SOLVERS::AMRMULTIGRIDSOLVER::SolveResult;
    using SolveStatus = FML::SOLVERS::AMRMULTIGRIDSOLVER::SolveStatus;
    using IndexInt = FML::GRID::IndexInt;
    using Vec = std::array<double, NDIM>;
    /// The uniform-grid solver: the model (background, units) and the cell solve
    using Uniform = DilatonSolverCosmology<NDIM, double>;

  private:
    /// Matter density parameter today
    double OmegaM{0.3};
    /// The parameters of the model (see DilatonSolverCosmology)
    double A2{1e6};
    double beta0{1.0};
    double r{1.0};
    double xi{1e-3};
    /// The dimensionless number H0 * Box / c
    double H0Box{};
    bool verbose{true};
    Uniform model;

    int ngs_pre{4};
    int ngs_post{4};
    int ngs_first{0};
    int maxsteps{100};
    int coarsest_level{-1};

    /// Converged when residual < epsilon, or (relative) residual < epsilon * rms(linear source norm a^-3 delta)
    double epsilon{1e-6};
    bool relative_criterion{false};

    /// w of the last solve, and a field to start the next solve from (per leaf of the same grid)
    std::vector<double> field;
    std::vector<double> previous_field;

    SolveResult last_result{};

  public:
    /// The bounds on u = ln w are widened by this (see the class description)
    static constexpr double bound_margin = 1.0;

    AMRDilatonSolverCosmology(double OmegaM, double A2, double beta0, double r, double xi, double H0Box, bool verbose)
        : OmegaM(OmegaM), A2(A2), beta0(beta0), r(r), xi(xi), H0Box(H0Box), verbose(verbose),
          model(OmegaM, A2, beta0, r, xi, H0Box, false) {}

    /// Gauss-Seidel sweeps before and after the coarse grid correction (and on level 0 before the first V-cycle)
    void set_ngs_steps(int pre, int post, int first = 0) {
        ngs_pre = pre;
        ngs_post = post;
        ngs_first = first;
    }
    void set_epsilon(double eps) { epsilon = eps; }
    /// Converged when the rms residual is below epsilon (fiducial) or, with relative = true, below epsilon times the
    /// rms of the linear source norm a^-3 delta
    void set_relative_criterion(bool relative) { relative_criterion = relative; }
    void set_maxsteps(int steps) { maxsteps = steps; }
    /// The refinement level of the coarsest multigrid level (default 2)
    void set_coarsest_level(int level) { coarsest_level = level; }
    int get_number_of_vcycles() const { return last_result.iterations; }
    SolveResult get_solve_result() const { return last_result; }

    /// The field w = phi / phi_bar of the last solve (per local leaf)
    const std::vector<double> & get_field() const { return field; }
    /// Start the next solve from this field w (per local leaf of the grid of the next solve; the default is the field
    /// of the last solve if the number of leaves matches, else the background w = 1)
    void set_previous_field(const std::vector<double> & w) { previous_field = w; }

    /// s = 9 OmegaM A2 xi^2 and p = 2r - 3
    double get_s() const { return model.get_s(); }
    double get_p() const { return model.get_p(); }
    /// The coupling in the background beta(a) = A2 phi_bar(a) / M_pl
    double get_beta_background(double a) const { return model.get_beta_background(a); }
    /// The background fifth-force potential PhiF_bar = a^2 beta(a)^2 / (2 A2 (H0Box)^2) (PhiF = PhiF_bar w^2)
    double get_phif_background(double a) const { return model.get_phif_background(a); }
    /// The factor norm in the equation
    double get_norm(double a) const { return 3.0 * A2 * OmegaM * a * a * H0Box * H0Box; }
    /// The range of the field in the background (box units): xi / H0Box a^(r - 1)
    double get_range(double a) const { return xi / H0Box * std::pow(a, r - 1.0); }
    /// (a^p - 1)/p (ln a for p = 0)
    double get_Ea(double a) const {
        const double p = get_p();
        return std::abs(p) < 1e-12 ? std::log(a) : std::expm1(p * std::log(a)) / p;
    }

    /// The bracket [ulo, uhi] of u = ln w for the densities 1 + delta in [rho_min, rho_max] (the minima of the
    /// effective potential, widened by bound_margin, within +-700)
    std::pair<double, double> get_u_bounds(double a, double rho_min, double rho_max) const {
        const double ainv3 = 1.0 / (a * a * a), Ea = get_Ea(a), s = get_s(), p = get_p();
        const double ulo = Uniform::u_local_minimum(rho_max * ainv3, Ea, s, p) - bound_margin;
        const double uhi = Uniform::u_local_minimum(rho_min * ainv3, Ea, s, p) + bound_margin;
        return {std::max(ulo, -700.0), std::min(uhi, 700.0)};
    }

    /// L(w) = D^2 w - norm w [(1 + delta) a^-3 - T(ln w)] and dL/dw, with delta the external field id_delta of the
    /// solver (values below -1 are taken as -1)
    Equation get_equation(double a, int id_delta) const {
        const double norm = get_norm(a), ainv3 = 1.0 / (a * a * a), Ea = get_Ea(a), s = get_s(), p = get_p();
        return [=](const Solver & sol, int level, IndexInt i) {
            auto [lap, dlap] = sol.get_laplacian(level, i);
            const double w = sol.get_field(level, i);
            const double d = std::max(sol.get_external_field(id_delta, level, i), -1.0);
            // The cell update keeps w > 0; a value w <= 0 (a transient after a coarse-grid correction) gets T = 0
            // (log(0) = -inf would give 0 * inf = NaN in the source)
            double T = 0.0, dT = 0.0;
            if (w > 0.0)
                Uniform::potential_term(std::log(w), Ea, s, p, T, dT);
            const double S = (1.0 + d) * ainv3 - T;
            return std::pair<double, double>{lap - norm * w * S, dlap - norm * (S - dT)};
        };
    }

    /// The Gauss-Seidel update: solve L(w) = source in the cell for u = ln w in [ulo, uhi] (with the Laplacian
    /// D^2 w' = lap + dlap (w' - w), affine in the cell's value)
    Update get_update(double a, int id_delta, double ulo, double uhi) const {
        const double norm = get_norm(a), ainv3 = 1.0 / (a * a * a), Ea = get_Ea(a), s = get_s(), p = get_p();
        return [=](const Solver & sol, int level, IndexInt i, double source) {
            auto [lap, dlap] = sol.get_laplacian(level, i);
            const double w = sol.get_field(level, i);
            const double d = std::max(sol.get_external_field(id_delta, level, i), -1.0);
            const double u = w > 0.0 ? std::log(w) : ulo;
            const double c = lap - dlap * w - source;
            const double unew =
                Uniform::solve_cell_affine(u, c, 1.0, -dlap, (1.0 + d) * ainv3, norm, Ea, s, p, ulo, uhi);
            // No root inside the bracket (a coarse level, where the FAS source can leave the cell equation without a
            // positive solution: c <= 0): keep the cell as it is rather than set it to the bound. The bracket is
            // wider than any solution of the fine equation, so a cell at a bound is never right
            if ((unew <= ulo or unew >= uhi) and w > 0.0 and std::isfinite(w))
                return w;
            return std::exp(unew);
        };
    }

    /// Solve for the fifth-force potential PhiF = PhiF_bar w^2 (per local leaf of grid) given the overdensity delta
    /// (per local leaf). If force is given it gets the fifth force D(PhiF) = 2 PhiF_bar w D(w) (box units)
    template <class T>
    SolveResult solve(double a,
                      const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                      const std::vector<double> & delta,
                      std::vector<double> & phi,
                      std::vector<Vec> * force = nullptr);
};

template <int NDIM>
template <class T>
typename AMRDilatonSolverCosmology<NDIM>::SolveResult
AMRDilatonSolverCosmology<NDIM>::solve(double a,
                                       const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                                       const std::vector<double> & delta,
                                       std::vector<double> & phi,
                                       std::vector<Vec> * force) {
    const IndexInt n = grid.get_n_local();
    assert_mpi(IndexInt(delta.size()) == n, "AMRDilatonSolverCosmology::solve: delta has the wrong size");

    if (FML::ThisTask == 0 and verbose) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Running AMR multigrid solver for the dilaton\n";
        std::cout << "# a           : " << a << "\n";
        std::cout << "# OmegaM      : " << OmegaM << "\n";
        std::cout << "# A2          : " << A2 << "\n";
        std::cout << "# beta0       : " << beta0 << "\n";
        std::cout << "# r           : " << r << "\n";
        std::cout << "# xi          : " << xi << "\n";
        std::cout << "# s           : " << get_s() << "\n";
        std::cout << "# H0Box       : " << H0Box << "\n";
        std::cout << "# beta(a)     : " << get_beta_background(a) << "\n";
        std::cout << "# range(a)    : " << get_range(a) << " (box units)\n";
        std::cout << "# Convergence : residual < " << epsilon << (relative_criterion ? " * rms(linear source)" : "") << "\n";
        std::cout << "#=====================================================\n";
    }

    // The density range (for the bracket of u) and the rms of the linear source
    double rho_min = std::numeric_limits<double>::max(), rho_max = -rho_min, source_rms = 0.0;
    for (IndexInt i = 0; i < n; i++) {
        const double rho = 1.0 + std::max(delta[i], -1.0);
        rho_min = std::min(rho_min, rho);
        rho_max = std::max(rho_max, rho);
        source_rms += delta[i] * delta[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    }
    FML::MinOverTasks(&rho_min);
    FML::MaxOverTasks(&rho_max);
    FML::SumOverTasks(&source_rms);
    source_rms = get_norm(a) / (a * a * a) * std::sqrt(source_rms);
    const auto [ulo, uhi] = get_u_bounds(a, rho_min, rho_max);

    Solver sol(grid, coarsest_level, verbose);
    const int id_delta = sol.add_external_field(delta);
    sol.set_ngs_sweeps(ngs_pre, ngs_post, ngs_first);
    sol.set_maxsteps(maxsteps);
    sol.set_convergence_criterion_residual(true);
    sol.set_epsilon(relative_criterion and source_rms > 0.0 ? epsilon * source_rms : epsilon);

    // Start from a previous field (given, or the last one on a grid with as many leaves), else the background w = 1
    const std::vector<double> & start = previous_field.size() == size_t(n) ? previous_field : field;
    if (start.size() == size_t(n)) {
        std::vector<double> guess(n);
        for (IndexInt i = 0; i < n; i++)
            guess[i] = start[i] > 0.0 and std::isfinite(start[i]) ? std::exp(std::clamp(std::log(start[i]), ulo, uhi)) : 1.0;
        sol.set_initial_guess(guess);
    } else {
        sol.set_initial_guess(1.0);
    }
    previous_field.clear();

    // Ghost values between positive values: the neighbour sum of the cell solve stays positive (see the class
    // description)
    sol.set_coarse_fine_limiter(true);
    sol.set_custom_update(get_update(a, id_delta, ulo, uhi));
    last_result = sol.solve(get_equation(a, id_delta));

    // The fifth-force potential (and force)
    field = sol.get_solution();
    const double phifbar = get_phif_background(a);
    phi.resize(n);
    double umin = std::numeric_limits<double>::max(), umax = -umin, umean = 0.0;
    long long nbad = 0;
    for (IndexInt i = 0; i < n; i++) {
        const double w = field[i];
        phi[i] = phifbar * w * w;
        if (not(w > 0.0 and std::isfinite(phi[i]))) {
            nbad++;
            continue;
        }
        const double u = std::log(w);
        umin = std::min(umin, u);
        umax = std::max(umax, u);
        umean += u * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    }
    if (force) {
        *force = sol.get_solution_gradient();
        for (IndexInt i = 0; i < n; i++)
            for (auto & x : (*force)[i])
                x *= 2.0 * phifbar * field[i];
    }
    FML::SumOverTasks(&nbad);
    FML::MinOverTasks(&umin);
    FML::MaxOverTasks(&umax);
    FML::SumOverTasks(&umean);
    if (nbad > 0) {
        last_result.status = SolveStatus::NonFinite;
        if (FML::ThisTask == 0)
            std::cerr << "[AMRDilatonSolverCosmology] The dilaton solution is not finite or not positive in " << nbad
                      << " leaves (a = " << a << ")\n";
    }
    if (FML::ThisTask == 0 and verbose) {
        std::cout << "#=====================================================\n";
        std::cout << "# V-cycles: " << last_result.iterations
                  << (last_result.converged() ? " (converged)" : " (NOT converged)") << "  residual "
                  << last_result.residual << "\n";
        std::cout << "# The minimum value of ln(phi/phi_bar) : " << umin << "\n";
        std::cout << "# The mean value of ln(phi/phi_bar)    : " << umean << "\n";
        std::cout << "# The maximum value of ln(phi/phi_bar) : " << umax << "\n";
        std::cout << "#=====================================================\n\n";
    }
    return last_result;
}

#endif
