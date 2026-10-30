#ifndef FOFRSOLVER_HEADER
#define FOFRSOLVER_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIGrid/ConvertMPIGridFFTWGrid.h>
#include <FML/MultigridSolver/MultiGridSolver.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

//======================================================================
/// Solve the Hu-Sawicki f(R) gravity equation in a cosmological setting
/// The solver returns PhiF = a^2 |f_R| / (2 (H0 Box)^2) so that in a
/// cosmological setting where PhiGR = a^2 Phi / (H0 Box)^2 we simply
/// have the total force = D(PhiGR + PhiF)
/// This class (kind of) assumes a LCDM cosmology
///
/// The equation (box units, x in [0,1)) is
///
///   D^2 PhiF = (1/3) 1.5 OmegaM a delta - prefac [ (PhiF_bar / PhiF)^(1/(n+1)) - 1 ]
///
/// with prefac = 0.5 OmegaM a^4 (a^-3 + 4 OmegaL / OmegaM). Two field variables (set_method):
///
///  "sqrt" (fiducial): PhiF = u^2 (Bose, Hellwing & Li 2017) with D^2 PhiF the standard Laplacian of PhiF. For n = 1
///         the equation in a cell (neighbors fixed) is a cubic u^3 + p u + q = 0 with q < 0 which has exactly one
///         positive root, so the Gauss-Seidel update solves it exactly and PhiF > 0 on every level. For n != 1 the
///         equation in a cell is monotone in u and is solved exactly with a safeguarded Newton-bisection.
///  "log": PhiF = e^f with D^2 PhiF = D[ e^f Df ] (Oyaizu 2008; the original method here). The Newton steps can
///         overshoot and give NaN when the field is far from its background value, e.g. for |f_R0| = 1e-6 at
///         a <~ 0.5 (also with many sweeps on the first step).
///
/// The two discretize the kinetic term differently; the fifth force agrees to ~0.1-0.5% where both converge. If
/// the solution is not finite the solver aborts with a message.
//======================================================================
template <int NDIM, typename SolverType = double>
class FofrSolverCosmology {
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

    /// Maximum number of V-cycles
    int maxsteps{1000};

    /// Convergence criterion: residual < epsilon
    double epsilon{1e-6};

    /// The field we solve for: "sqrt" (PhiF = u^2) or "log" (PhiF = e^f)
    std::string method{"sqrt"};

    /// Matter density parameter today
    double OmegaM{0.3};
    /// The f(R) n-index
    double nfofr{1.0};
    /// The f(R) f_R0 value
    double fofr0{1e-6};
    /// The dimensionless number H0 * Box / c
    double H0Box{};

    /// Print info as we go along
    bool verbose{true};

    /// The box is periodic?
    bool periodic{true};

    /// The number of V-cycles of the last solve
    int last_nvcycles{0};

    /// A previous solution (PhiF, real space) and its scale factor to start from (warm start)
    FFTWGrid<NDIM> * previous_solution{nullptr};
    double a_previous{0.0};

  public:
    FofrSolverCosmology(double OmegaM, double nfofr, double fofr0, double H0Box, bool verbose)
        : OmegaM(OmegaM), nfofr(nfofr), fofr0(fofr0), H0Box(H0Box), verbose(verbose) {}

    /// Set how many sweeps over the grid to use when solving a given level
    void set_ngs_steps(int _ngs_fine, int _ngs_coarse, int _ngs_first) {
        ngs_fine = _ngs_fine;
        ngs_coarse = _ngs_coarse;
        ngs_first = _ngs_first;
    }

    /// Set convergence criterion
    void set_epsilon(double _epsilon) { epsilon = _epsilon; }

    /// Set the maximum number of V-cycles
    void set_maxsteps(int _maxsteps) { maxsteps = _maxsteps; }

    /// The field we solve for: "sqrt" (fiducial) or "log"
    void set_method(const std::string & _method) {
        if (not(_method == "sqrt" or _method == "log"))
            throw std::runtime_error("[FofrSolverCosmology] Unknown method " + _method + " (use sqrt or log)");
        method = _method;
    }

    /// The number of V-cycles of the last solve
    int get_number_of_vcycles() const { return last_nvcycles; }

    /// Start the next solve from a previous solution PhiF (real space, same grid) at scale factor a_prev, rescaled by
    /// the change of the background (PhiF / PhiF_bar is kept). The grid must stay alive until solve() is called
    void set_previous_solution(FFTWGrid<NDIM> & previous_potential_real, double a_prev) {
        previous_solution = &previous_potential_real;
        a_previous = a_prev;
    }

    /// The cosmological background value |f_R|(a)
    double get_fofr_background(double a) const {
        const double fac1 = 1.0 + 4.0 * (1.0 - OmegaM) / OmegaM;
        const double fac2 = 1.0 / (a * a * a) + 4.0 * (1.0 - OmegaM) / OmegaM;
        return fofr0 * std::pow(fac1 / fac2, nfofr + 1);
    }

    /// The positive real root of u^3 + p u + q = 0 for q < 0 (there is exactly one)
    static double positive_cubic_root(double p, double q) {
        double u;
        const double D = 0.25 * q * q + p * p * p / 27.0;
        if (D >= 0.0) {
            // One real root u = A + B with A = cbrt(-q/2 + sqrt(D)) > 0 and B = -p/(3A)
            const double A = std::cbrt(-0.5 * q + std::sqrt(D));
            if (p > 0.0) {
                // Avoid the cancellation in A - p/(3A): u = (A^3 + B^3) / (A^2 - AB + B^2)
                u = -q / (A * A + p / 3.0 + p * p / (9.0 * A * A));
            } else {
                u = A - p / (3.0 * A);
            }
        } else {
            // Three real roots (p < 0): the largest is the positive one
            const double r = std::sqrt(-p / 3.0);
            const double phi = std::acos(std::clamp(-q / (2.0 * r * r * r), -1.0, 1.0));
            u = 2.0 * r * std::cos(phi / 3.0);
        }
        // One Newton step to polish the root
        const double denom = 3.0 * u * u + p;
        if (denom > 0.0) {
            const double unew = u - (u * u * u + p * u + q) / denom;
            if (unew > 0.0)
                u = unew;
        }
        return u;
    }

    /// The positive root u of g(u) = (sumnb - 2NDIM u^2)/h2 + prefac ((f0/u)^alpha - 1) - rhs (the equation in a
    /// cell for the sqrt method). g is strictly decreasing from +inf (u -> 0) to -inf, so the root is unique. Newton
    /// steps from the current value u, keeping a bracket [lo, hi] from the signs of g: a step that leaves the bracket
    /// is replaced by bisection in log u (or by a factor of 2 while one side of the bracket is still open). Usually
    /// 2-3 iterations
    static double
    solve_cell_general_n(double u, double sumnb, double h2, double rhs, double prefac, double f0, double alpha) {
        if (not(u > 0.0) or not std::isfinite(u))
            u = f0;
        double lo = 0.0, hi = std::numeric_limits<double>::infinity();
        double x = u;
        for (int iter = 0; iter < 50; iter++) {
            const double r = alpha == 1.0 ? f0 / x : std::pow(f0 / x, alpha);
            const double gx = (sumnb - 2.0 * NDIM * x * x) / h2 + prefac * (r - 1.0) - rhs;
            const double dgx = -4.0 * NDIM * x / h2 - prefac * alpha * r / x;
            // Stop when g is zero to round-off (with u much smaller than its neighbors the terms cancel to many
            // digits, so a relative step of 1e-12 is not always reachable)
            const double gscale = (sumnb + 2.0 * NDIM * x * x) / h2 + prefac * (r + 1.0) + std::abs(rhs);
            if (std::abs(gx) <= 1e-14 * gscale)
                return x;
            if (gx > 0.0)
                lo = x;
            else
                hi = x;
            double xnew = x - gx / dgx;
            if (not(xnew > lo and xnew < hi)) {
                if (lo == 0.0)
                    xnew = 0.5 * hi;
                else if (std::isinf(hi))
                    xnew = 2.0 * lo;
                else
                    xnew = std::sqrt(lo * hi);
            }
            if (std::abs(xnew - x) < 1e-12 * x)
                return xnew;
            x = xnew;
        }
        return x;
    }

    /// Solve and return the fifth-force potential PhiF = a^2 |f_R| / (2 (H0Box)^2) in real space
    void solve(double a, FFTWGrid<NDIM> & overdensity_real, FFTWGrid<NDIM> & fifth_force_potential_real) {

        Nmesh = overdensity_real.get_nmesh();
        auto Local_nx = overdensity_real.get_local_nx();
        const bool use_sqrt = (method == "sqrt");

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Running multigridsolver for f(R)\n";
            std::cout << "# Nmesh       : " << Nmesh << "\n";
            std::cout << "# a           : " << a << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# nfofr       : " << nfofr << "\n";
            std::cout << "# H0Box       : " << H0Box << "\n";
            std::cout << "# f_R0        : " << fofr0 << "\n";
            std::cout << "# f_R(a)      : " << get_fofr_background(a) << "\n";
            std::cout << "# Method      : " << method << (use_sqrt ? " (PhiF = u^2)" : " (PhiF = e^f)") << "\n";
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
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && real_index : overdensity_real.get_real_range(islice, islice + 1)) {
                // Fetch the value in the cell
                auto density_in_cell = overdensity_real.get_real_from_index(real_index);
                // Fetch the coordinates of the cell (ix,iy,iz) and index
                auto coord = overdensity_real.get_coord_from_index(real_index);
                auto density_grid_index = grid.index_from_coord(coord);
                // Set the value
                grid.set_y(density_grid_index, density_in_cell);
            }
        }
        density_multigrid.restrict_down_all();

        // Factors to define the equation (the first term is GR times 1/3)
        const double poisson_norm = (1.0 / 3.0) * (1.5 * OmegaM * a);
        const double prefac = 0.5 * OmegaM * a * a * a * a * (1.0 / (a * a * a) + 4.0 * (1.0 - OmegaM) / OmegaM);

        // The background value of PhiF and of the field we solve for
        const double phibar = a * a * get_fofr_background(a) / 2.0 / std::pow(H0Box, 2);
        const double f0 = use_sqrt ? std::sqrt(phibar) : std::log(phibar);
        const bool n_is_one = std::abs(nfofr - 1.0) < 1e-12;
        const double exponent = use_sqrt ? 2.0 / (1.0 + nfofr) : 1.0 / (1.0 + nfofr);

        // Set up solver
        MultiGridSolver<NDIM, SolverType> g(Nmesh, Nlevels, verbose, periodic, nleft, nright);

        // Set some options
        g.set_epsilon(epsilon);
        g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
        g.set_maxsteps(maxsteps);

        // Set the initial guess: the background value, or a previous solution with the same PhiF / PhiF_bar
        g.set_initial_guess(f0);
        if (previous_solution and previous_solution->get_nmesh() == Nmesh) {
            const double phibar_previous =
                a_previous * a_previous * get_fofr_background(a_previous) / 2.0 / std::pow(H0Box, 2);
            const double scale = phibar / phibar_previous;
            FML::GRID::MPIGrid<NDIM, SolverType> guess;
            ConvertToMPIGrid(*previous_solution, guess);
            const IndexInt NtotLocal = guess.get_NtotLocal();
            SolverType * y = guess.get_y();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (IndexInt i = 0; i < NtotLocal; i++) {
                const double value = double(y[i]) * scale;
                const bool ok = value > 0.0 and std::isfinite(value);
                y[i] = SolverType(ok ? (use_sqrt ? std::sqrt(value) : std::log(value)) : f0);
            }
            g.set_initial_guess(y);
        }

        //======================================================================
        // Set the convergence criterion
        // (ConvergenceCriterionResidual is rms-residual < epsilon; stop if it is not finite)
        //======================================================================
        MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
            if (not std::isfinite(rms_residual))
                return true;
            return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
        };

        //======================================================================
        // PhiF = e^f: D[ e^f Df ] - source(f) = 0
        //======================================================================
        MultiGridFunction<NDIM, SolverType> EquationLog =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index) {
                auto index_list = sol->get_neighbor_gridindex(level, index);
                const double f = sol->get_Field(level, index);
                const double delta = density_multigrid.get_y(level, index);

                // The kinetic term D[ b[f] Df ] with b = exp(f) (the face values of b are the averages)
                const double h = sol->get_Gridspacing(level);
                const double bc = std::exp(f);
                double kinetic = 0.0, dkinetic = 0.0;
                for (int k = 1; k < 2 * NDIM + 1; k++) {
                    const double fnb = sol->get_Field(level, index_list[k]);
                    const double bface = 0.5 * (std::exp(fnb) + bc);
                    kinetic += bface * (fnb - f);
                    dkinetic += 0.5 * bc * (fnb - f) - bface;
                }
                kinetic /= h * h;
                dkinetic /= h * h;

                // The right hand side source and the derivative of it
                const double e = std::exp((f0 - f) * exponent);
                const double source = poisson_norm * delta - prefac * (e - 1.0);
                const double dsource = prefac * e * exponent;
                return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
            };

        //======================================================================
        // PhiF = u^2: D^2 (u^2) - source(u) = 0
        //======================================================================
        auto sum_of_neighbors_squared = [](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index) {
            auto index_list = sol->get_neighbor_gridindex(level, index);
            double sum = 0.0;
            for (int k = 1; k < 2 * NDIM + 1; k++) {
                const double unb = sol->get_Field(level, index_list[k]);
                sum += unb * unb;
            }
            return sum;
        };
        MultiGridFunction<NDIM, SolverType> EquationSqrt =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index) {
                const double u = sol->get_Field(level, index);
                const double h = sol->get_Gridspacing(level);
                const double delta = density_multigrid.get_y(level, index);
                const double sumnb = sum_of_neighbors_squared(sol, level, index);
                const double lap = (sumnb - 2.0 * NDIM * u * u) / (h * h);
                const double dlap = -4.0 * NDIM * u / (h * h);
                const double ratio = n_is_one ? f0 / u : std::pow(f0 / u, exponent);
                const double source = poisson_norm * delta - prefac * (ratio - 1.0);
                const double dsource = prefac * exponent * ratio / u;
                return std::pair<SolverType, SolverType>{lap - source, dlap - dsource};
            };

        // The Gauss-Seidel update: solve the equation in the cell for u (neighbors fixed). For n = 1:
        // (sumnb - 2NDIM u^2)/h^2 - poisson_norm delta + prefac (f0/u - 1) - s = 0, times -u h^2/(2NDIM), is
        // u^3 + p u + q = 0 with q < 0. Otherwise the left hand side is strictly decreasing in u > 0 (from +inf to
        // -inf) and we find its root with a safeguarded Newton-bisection (see solve_cell_general_n)
        MultiGridUpdateFunction<NDIM, SolverType> UpdateSqrt =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index, SolverType fas_source) {
                const double u = sol->get_Field(level, index);
                const double h2 = std::pow(sol->get_Gridspacing(level), 2);
                const double delta = density_multigrid.get_y(level, index);
                const double sumnb = sum_of_neighbors_squared(sol, level, index);
                return SolverType(
                    solve_cell_general_n(u, sumnb, h2, poisson_norm * delta + fas_source, prefac, f0, exponent));
            };

        // Solve the equation and fetch the solution
        if (use_sqrt) {
            g.set_custom_update(UpdateSqrt);
            g.solve(EquationSqrt, ConvergenceCriterion);
        } else {
            g.solve(EquationLog, ConvergenceCriterion);
        }
        last_nvcycles = g.get_number_of_vcycles();
        auto sol = g.get_grid(0);
        ConvertToFFTWGrid(sol, fifth_force_potential_real);

        // Convert to fifth-force potential a^2 |f_R| / (2 (H0Box)^2)
        double fmin = std::numeric_limits<double>::max();
        double fmax = -std::numeric_limits<double>::max();
        double fmean = 0.0;
        long long nbad = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : fmax) reduction(min : fmin) reduction(+ : fmean, nbad)
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && real_index : fifth_force_potential_real.get_real_range(islice, islice + 1)) {
                double value = fifth_force_potential_real.get_real_from_index(real_index);
                value = use_sqrt ? value * value : std::exp(value);
                fifth_force_potential_real.set_real_from_index(real_index, value);
                if (not std::isfinite(value))
                    nbad++;
                fmean += value;
                fmax = std::max(fmax, value);
                fmin = std::min(fmin, value);
            }
        }
        FML::SumOverTasks(&fmean);
        FML::SumOverTasks(&nbad);
        FML::MinOverTasks(&fmin);
        FML::MaxOverTasks(&fmax);
        fmean /= std::pow(Nmesh, NDIM);

        // A non-finite solution would silently ruin the simulation
        if (nbad > 0) {
            const std::string msg = "[FofrSolverCosmology] The f(R) solution is not finite in " + std::to_string(nbad) +
                                    " cells (a = " + std::to_string(a) + ", method = " + method +
                                    "). With method log try method sqrt or more sweeps on the first step\n";
            if (FML::ThisTask == 0)
                std::cerr << msg << std::flush;
            FML::abort_mpi(1);
        }

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "#=====================================================\n";
            std::cout << "# V-cycles: " << last_nvcycles << "\n";
            std::cout << "# The minimum value of f_R / f_R(a): " << fmin / phibar << "\n";
            std::cout << "# The mean value of f_R / f_R(a): " << fmean / phibar << "\n";
            std::cout << "# The maximum value of f_R / f_R(a): " << fmax / phibar << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
