#ifndef DILATONSOLVER_HEADER
#define DILATONSOLVER_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIGrid/ConvertMPIGridFFTWGrid.h>
#include <FML/MultigridSolver/MultiGridSolver.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

//======================================================================
/// Solve the (generalised) dilaton equation in a cosmological setting (Brax, Davis, Li, Winther & Zhao 2012,
/// arXiv:1206.3568, Sec. IV; the theory is FML/Cosmology/Perturbations/QuasiStaticGravity_Dilaton.h):
///   A(phi) = 1 + A2 phi^2 / (2 M_pl^2),   m(a) = m0 a^-r,   xi = H0 / m0,   s = 9 OmegaM A2 xi^2,   p = 2r - 3
///   beta(a) = A2 phi_bar(a) / M_pl = beta0 exp[ s (a^p - 1) / p ]   (r <= 3/2; r = 3/2 is the limit p -> 0)
/// With w = phi / phi_bar(a) = e^u (w = 1 in the background) the equation (box units, x in [0,1), D^2 the comoving
/// Laplacian) is
///
///   D^2 w = norm w [ (1 + delta) a^-3 - T(u) ],   T = (1 + p y)^(-3/p),   y = (a^p - 1)/p + u / s
///
/// with norm = 3 A2 OmegaM a^2 (H0 Box / c)^2 (T -> exp(-3 y) for p -> 0; for p < 0 T = 0 when 1 + p y <= 0, i.e.
/// phi above its value at delta = -1). The equation does not depend on beta0.
///
/// The multigrid solver works with w and the standard Laplacian (linear, so the coarse-grid corrections are averages
/// of phi), while the Gauss-Seidel update solves the equation in a cell (neighbors fixed) for u = ln w (solve_cell):
/// divided by w it is strictly decreasing in u on the fine level, so the root is unique and w > 0. The field spans
/// many orders of magnitude at early times (u ~ s/a times the change of (1 + delta)^(1/3)): with 64^3 particles on a
/// 128^3 grid at z = 30 the empty cells of the particle lattice have u ~ +40 and the full ones u ~ -40.
/// (The paper solves for u with D[e^u Du] instead (its eq. 69, 76-78); that form, and the same form divided by e^u,
/// did not converge for such fields: the Newton steps go the wrong way in underdense cells, and the coarse-grid
/// corrections of u ask for jumps of 1e6 in u.)
///
/// The solver returns PhiF = a^2 ln A(phi) / (H0 Box)^2 = a^2 beta(a)^2 w^2 / (2 A2 (H0 Box)^2) so that in a
/// cosmological setting where PhiGR = a^2 Phi / (H0 Box)^2 we simply have the total force = D(PhiGR + PhiF)
/// (the geodesic equation 12: the potential is Phi + ln A). Linear theory: -k^2 PhiF / (1.5 OmegaM a) =
/// 2 beta^2 k^2 / (k^2 + a^2 m^2) delta.
/// This class (kind of) assumes a LCDM cosmology (matter going as a^-3)
//======================================================================
template <int NDIM, typename SolverType = double>
class DilatonSolverCosmology {
  private:
    // Type aliases used below (kept inside the class so they don't leak into user code)
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    template <int N, class T>
    using MPIMultiGrid = FML::SOLVERS::MULTIGRIDSOLVER::MPIMultiGrid<N, T>;
    template <int N, class T>
    using MultiGridSolver = FML::SOLVERS::MULTIGRIDSOLVER::MultiGridSolver<N, T>;
    using MultiGridConvCrit = FML::SOLVERS::MULTIGRIDSOLVER::MultiGridConvCrit;
    template <int N, class T>
    using MultiGridFunction = FML::SOLVERS::MULTIGRIDSOLVER::MultiGridFunction<N, T>;
    template <int N, class T>
    using MultiGridUpdateFunction = FML::SOLVERS::MULTIGRIDSOLVER::MultiGridUpdateFunction<N, T>;
    using IndexInt = FML::SOLVERS::MULTIGRIDSOLVER::IndexInt;

    int Nmesh{};
    int Nlevels{-1};
    int nleft{1};
    int nright{1};

    int ngs_fine{10};
    int ngs_coarse{10};
    int ngs_first{40};
    int maxsteps{1000};

    /// Convergence criterion: residual < epsilon
    double epsilon{1e-6};
    /// Converged when residual < epsilon * rms(linear source) instead of residual < epsilon
    bool relative_criterion{false};

    /// Matter density parameter today
    double OmegaM{0.3};
    /// The parameters of the model (see the class description)
    double A2{1e6};
    double beta0{1.0};
    double r{1.0};
    double xi{1e-3};
    /// The dimensionless number H0 * Box / c
    double H0Box{};

    /// Print info as we go along
    bool verbose{true};

    /// The box is periodic?
    bool periodic{true};

    /// The number of V-cycles of the last solve
    int last_nvcycles{0};

    /// A previous solution (PhiF in real space, as returned by solve) and its scale factor: the starting point
    FFTWGrid<NDIM> * previous_solution{nullptr};
    double a_previous{-1.0};

    /// (a^p - 1)/p as a function of x = ln a (ln a for p = 0) and ln(1 + p y)/p (y for p = 0)
    static double expm1_over_p(double pp, double x) { return std::abs(pp) < 1e-12 ? x : std::expm1(pp * x) / pp; }
    static double log1p_over_p(double pp, double y) { return std::abs(pp) < 1e-12 ? y : std::log1p(pp * y) / pp; }

  public:
    /// The potential term T(u) = (1 + p y)^(-3/p), y = Ea + u / s, and dT/du (both 0 for p < 0 and 1 + p y <= 0)
    static void potential_term(double u, double Ea, double s, double p, double & T, double & dT) {
        const double y = Ea + u / s;
        const double onepy = 1.0 + p * y;
        T = 0.0;
        dT = 0.0;
        if (onepy > 0.0) {
            T = std::exp(-3.0 * log1p_over_p(p, y));
            dT = -3.0 * T / (s * onepy);
        }
    }

    /// The field u = ln(phi / phi_bar) at the minimum of the effective potential for the density
    /// rho / rho_bar = rhoa3 a^3 (the solution for a uniform density): T(u) = rhoa3, i.e.
    /// y = -(1/3) ((rhoa3)^(-p/3) - 1) / (-p/3), u = s (y - Ea). For p < 0 and rho = 0 it is the largest value u can
    /// have, s (-1/p - Ea); for p = 0 and rho = 0 it is +infinity
    static double u_local_minimum(double rhoa3, double Ea, double s, double p) {
        const double y = -expm1_over_p(-p / 3.0, std::log(rhoa3)) / 3.0;
        return s * (y - Ea);
    }

    /// Solve the equation in a cell for u = ln w with the neighbors fixed: L(w) = f with
    /// L(w) = (sum_n w_n - 2 NDIM w) / h^2 - norm w [rhoa3 - T(ln w)] (f = 0 on the fine level, the FAS source on the
    /// coarse ones). Divided by w,
    ///   F(u) = c e^-u / h^2 - 2 NDIM / h^2 - norm [rhoa3 - T(u)],   c = sum_n w_n - f h^2,
    /// which for c >= 0 (always on the fine level) is strictly decreasing, so the root is unique. The root is
    /// restricted to [ulo, uhi]: the maximum principle (at the largest w of the grid D^2 w <= 0, so T(u) >= rhoa3 >=
    /// rho_min a^-3: u is at most the local minimum for the smallest density; likewise at least the one for the
    /// largest) makes these bounds inactive on the fine level, while on the coarse levels they keep the correction
    /// sane. Newton steps in u keeping a bracket from the signs of F; a step that leaves the bracket is replaced by
    /// bisection, and while the upper side is open (uhi = infinity for p = 0) the steps are at most 1, 2, 4, ...
    /// (For c < 0, only possible on the coarse levels, F need not be monotone; a root in the bracket is returned)
    static double solve_cell(double u,
                             double sum_wnb,
                             double fas,
                             double h2,
                             double rhoa3,
                             double norm,
                             double Ea,
                             double s,
                             double p,
                             double ulo,
                             double uhi,
                             int ndim = NDIM) {
        const double c = sum_wnb - fas * h2;
        auto eval = [&](double x, double & F, double & dF, double & Fscale) {
            double T, dT;
            potential_term(x, Ea, s, p, T, dT);
            const double ce = c * std::exp(-x) / h2;
            F = ce - 2.0 * ndim / h2 - norm * (rhoa3 - T);
            dF = -ce + norm * dT;
            Fscale = std::abs(ce) + 2.0 * ndim / h2 + norm * (std::abs(rhoa3) + T);
        };
        double F, dF, Fscale;
        eval(ulo, F, dF, Fscale);
        if (F <= 0.0)
            return ulo;
        double lo = ulo, hi = uhi;
        if (std::isfinite(uhi)) {
            eval(uhi, F, dF, Fscale);
            if (F >= 0.0)
                return uhi;
        }
        double x = std::isfinite(u) ? std::clamp(u, ulo, uhi) : ulo;
        double maxstep = 1.0;
        for (int iter = 0; iter < 200; iter++) {
            eval(x, F, dF, Fscale);
            // Zero to round-off
            if (std::abs(F) <= 1e-16 * Fscale)
                return x;
            if (F > 0.0)
                lo = x;
            else
                hi = x;
            double xnew = x - F / dF;
            if (std::isinf(hi)) {
                // The upper side open: steps of at most maxstep (doubling)
                xnew = x + std::min((std::isfinite(xnew) and xnew > x) ? xnew - x : maxstep, maxstep);
                maxstep *= 2.0;
            }
            if (not(xnew > lo and xnew < hi))
                xnew = 0.5 * (lo + hi);
            if (std::abs(xnew - x) <= 1e-15 * (1.0 + std::abs(x)))
                return xnew;
            x = xnew;
        }
        return x;
    }

    /// The number of V-cycles of the last solve
    int get_number_of_vcycles() const { return last_nvcycles; }

    DilatonSolverCosmology(double OmegaM, double A2, double beta0, double r, double xi, double H0Box, bool verbose)
        : OmegaM(OmegaM), A2(A2), beta0(beta0), r(r), xi(xi), H0Box(H0Box), verbose(verbose) {
        if (not(A2 > 0.0 and xi > 0.0 and r <= 1.5))
            throw std::invalid_argument("[DilatonSolverCosmology] Need A2 > 0, xi > 0 and r <= 3/2");
    }

    /// Set how many sweeps over the grid to use when solving a given level
    void set_ngs_steps(int _ngs_fine, int _ngs_coarse, int _ngs_first) {
        ngs_fine = _ngs_fine;
        ngs_coarse = _ngs_coarse;
        ngs_first = _ngs_first;
    }
    /// The maximum number of V-cycles
    void set_maxsteps(int _maxsteps) { maxsteps = _maxsteps; }

    /// Set convergenc criterion
    void set_epsilon(double _epsilon) { epsilon = _epsilon; }
    /// Converged when the rms residual is below epsilon (fiducial) or, with relative = true, below epsilon times the
    /// rms of the linear source of the equation (norm a^-3 delta)
    void set_relative_criterion(bool relative) { relative_criterion = relative; }

    /// Start the next solve from a previous solution (PhiF, as returned by solve, at the scale factor a_prev) with the
    /// same phi / phi_bar instead of from the background. The grid must outlive the solve
    void set_previous_solution(FFTWGrid<NDIM> & solution, double a_prev) {
        if (solution.get_nmesh() > 0 and a_prev > 0.0) {
            previous_solution = &solution;
            a_previous = a_prev;
        }
    }

    /// s = 9 OmegaM A2 xi^2 and p = 2r - 3
    double get_s() const { return 9.0 * OmegaM * A2 * xi * xi; }
    double get_p() const { return 2.0 * r - 3.0; }
    /// The coupling in the background beta(a) = A2 phi_bar(a) / M_pl
    double get_beta_background(double a) const {
        return beta0 * std::exp(get_s() * expm1_over_p(get_p(), std::log(a)));
    }
    /// The background value of the fifth-force potential PhiF = a^2 beta(a)^2 / (2 A2 (H0Box)^2)
    double get_phif_background(double a) const {
        const double b = get_beta_background(a);
        return a * a * b * b / (2.0 * A2 * H0Box * H0Box);
    }

    /// Do the solving and return the fifth-force potential PhiF = a^2 beta(a)^2 w^2 / (2 A2 (H0Box)^2) in real space
    void solve(double a, FFTWGrid<NDIM> & overdensity_real, FFTWGrid<NDIM> & fifth_force_potential_real) {

        Nmesh = overdensity_real.get_nmesh();
        const double s = get_s();
        const double p = get_p();

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Running multigridsolver for the dilaton\n";
            std::cout << "# Nmesh       : " << Nmesh << "\n";
            std::cout << "# a           : " << a << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# A2          : " << A2 << "\n";
            std::cout << "# beta0       : " << beta0 << "\n";
            std::cout << "# r           : " << r << "\n";
            std::cout << "# xi          : " << xi << "\n";
            std::cout << "# s           : " << s << "\n";
            std::cout << "# H0Box       : " << H0Box << "\n";
            std::cout << "# beta(a)     : " << get_beta_background(a) << "\n";
            std::cout << "# Convergence : residual < " << epsilon << "\n";
            std::cout << "# Ngs_sweeps  : " << ngs_fine << " (fine) , " << ngs_coarse << " (coarse)\n";
            std::cout << "# Ngs_sweeps  : " << ngs_first << " (first step)\n";
            std::cout << "#=====================================================\n";
        }

        // Set up multigrid for density
        MPIMultiGrid<NDIM, SolverType> density_multigrid(Nmesh, Nlevels, periodic, nleft, nright);
        auto & grid = density_multigrid.get_grid();

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType irow = 0; irow < overdensity_real.get_local_nrows(); irow++) {
            for (auto && real_index : overdensity_real.get_real_range_of_rows(irow, irow + 1)) {
                // (delta < -1 from round-off, e.g. of a density from Fourier space, is set to -1)
                auto density_in_cell = std::max(overdensity_real.get_real_from_index(real_index), FML::GRID::FloatType(-1));
                auto coord = overdensity_real.get_coord_from_index(real_index);
                auto density_grid_index = grid.index_from_coord(coord);
                grid.set_y(density_grid_index, density_in_cell);
            }
        }
        density_multigrid.restrict_down_all();

        // Factors to define the equation
        const double norm = 3.0 * A2 * OmegaM * a * a * H0Box * H0Box;
        const double ainv3 = 1.0 / (a * a * a);
        const double Ea = expm1_over_p(p, std::log(a));

        // The bounds on u = ln w from the maximum principle (see solve_cell): the local minima for the largest and
        // the smallest density (within +-700 so that w is finite)
        double rho_min = std::numeric_limits<double>::max(), rho_max = -std::numeric_limits<double>::max();
        {
            auto & g0 = density_multigrid.get_grid(0);
            const IndexInt NtotLocal = g0.get_NtotLocal();
            for (IndexInt i = 0; i < NtotLocal; i++) {
                rho_min = std::min(rho_min, 1.0 + double(g0[i]));
                rho_max = std::max(rho_max, 1.0 + double(g0[i]));
            }
            FML::MinOverTasks(&rho_min);
            FML::MaxOverTasks(&rho_max);
        }
        const double ulo = std::max(u_local_minimum(rho_max * ainv3, Ea, s, p), -700.0);
        const double uhi = std::min(u_local_minimum(rho_min * ainv3, Ea, s, p), 700.0);

        // Set up solver
        MultiGridSolver<NDIM, SolverType> g(Nmesh, Nlevels, verbose, periodic, nleft, nright);
        g.set_epsilon(epsilon);
        g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
        g.set_maxsteps(maxsteps);

        // The initial guess: the background (w = 1) or the previous solution's phi / phi_bar
        g.set_initial_guess(SolverType(1.0));
        if (previous_solution and previous_solution->get_nmesh() == Nmesh) {
            const double phifbar_previous = get_phif_background(a_previous);
            FML::GRID::MPIGrid<NDIM, SolverType> guess;
            ConvertToMPIGrid(*previous_solution, guess);
            const IndexInt NtotLocal = guess.get_NtotLocal();
            SolverType * y = guess.get_y();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (IndexInt i = 0; i < NtotLocal; i++) {
                const double ratio = double(y[i]) / phifbar_previous;
                const double u = (ratio > 0.0 and std::isfinite(ratio)) ? 0.5 * std::log(ratio) : 0.0;
                y[i] = SolverType(std::exp(std::clamp(u, ulo, uhi)));
            }
            g.set_initial_guess(y);
        }

        //======================================================================
        // The convergence criterion (ConvergenceCriterionResidual is rms-residual < epsilon; stop if not finite)
        //======================================================================
        const double source_rms = norm * ainv3 * density_multigrid.get_grid(0).norm();
        MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
            if (not std::isfinite(rms_residual))
                return true;
            if (relative_criterion and source_rms > 0.0)
                return rms_residual < epsilon * source_rms;
            return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
        };

        //======================================================================
        // L(w) = D^2 w - norm w [(1 + delta) a^-3 - T(ln w)] and dL/dw (for the residual: the sweeps use Update)
        //======================================================================
        MultiGridFunction<NDIM, SolverType> Equation =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index) {
                auto index_list = sol->get_neighbor_gridindex(level, index);
                const double w = sol->get_Field(level, index);
                const double delta = density_multigrid.get_y(level, index);
                const double kinetic = sol->get_Laplacian(level, index_list);
                const double dkinetic = sol->get_derivLaplacian(level, index_list);
                double T, dT;
                potential_term(std::log(w), Ea, s, p, T, dT);
                const double S = (1.0 + delta) * ainv3 - T;
                const double source = norm * w * S;
                const double dsource = norm * (S - dT);
                return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
            };

        // The Gauss-Seidel update: solve the equation in the cell for u = ln w (see solve_cell)
        MultiGridUpdateFunction<NDIM, SolverType> Update =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index, SolverType fas_source) {
                auto index_list = sol->get_neighbor_gridindex(level, index);
                double sum_wnb = 0.0;
                for (int k = 1; k < 2 * NDIM + 1; k++)
                    sum_wnb += sol->get_Field(level, index_list[k]);
                const double h2 = std::pow(sol->get_Gridspacing(level), 2);
                const double delta = density_multigrid.get_y(level, index);
                const double w = sol->get_Field(level, index);
                const double u = w > 0.0 ? std::log(w) : ulo;
                return SolverType(std::exp(
                    solve_cell(u, sum_wnb, fas_source, h2, (1.0 + delta) * ainv3, norm, Ea, s, p, ulo, uhi)));
            };

        // Solve the equation and fetch the solution
        g.set_custom_update(Update);
        g.solve(Equation, ConvergenceCriterion);
        last_nvcycles = g.get_number_of_vcycles();
        auto sol = g.get_grid(0);
        ConvertToFFTWGrid(sol, fifth_force_potential_real);

        // Convert to the fifth-force potential PhiF = PhiF_bar w^2
        const double phifbar = get_phif_background(a);
        double umin = std::numeric_limits<double>::max();
        double umax = -std::numeric_limits<double>::max();
        double umean = 0.0;
        long long nbad = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : umax) reduction(min : umin) reduction(+ : umean, nbad)
#endif
        for (FML::IndexIntType irow = 0; irow < fifth_force_potential_real.get_local_nrows(); irow++) {
            for (auto && real_index : fifth_force_potential_real.get_real_range_of_rows(irow, irow + 1)) {
                const double w = fifth_force_potential_real.get_real_from_index(real_index);
                const double value = phifbar * w * w;
                fifth_force_potential_real.set_real_from_index(real_index, value);
                if (not(std::isfinite(value) and w > 0.0)) {
                    nbad++;
                    continue;
                }
                const double u = std::log(w);
                umean += u;
                umax = std::max(umax, u);
                umin = std::min(umin, u);
            }
        }
        FML::SumOverTasks(&umean);
        FML::SumOverTasks(&nbad);
        FML::MinOverTasks(&umin);
        FML::MaxOverTasks(&umax);
        umean /= std::pow(double(Nmesh), NDIM);

        // A non-finite solution would silently ruin the simulation
        if (nbad > 0) {
            const std::string msg = "[DilatonSolverCosmology] The dilaton solution is not finite or not positive in " +
                                    std::to_string(nbad) + " cells (a = " + std::to_string(a) + ")\n";
            if (FML::ThisTask == 0)
                std::cerr << msg << std::flush;
            FML::abort_mpi(1);
        }

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "#=====================================================\n";
            std::cout << "# V-cycles: " << last_nvcycles << "\n";
            std::cout << "# The minimum value of ln(phi/phi_bar): " << umin << "\n";
            std::cout << "# The mean value of ln(phi/phi_bar): " << umean << "\n";
            std::cout << "# The maximum value of ln(phi/phi_bar): " << umax << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
