#ifndef AMRDGPSOLVER_HEADER
#define AMRDGPSOLVER_HEADER

#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

//======================================================================
///
/// The DGP (normal branch) fifth-force equation on an adaptively refined grid (AdaptiveMPIGrid), the AMR version of
/// DGPSolverCosmology (DGPSolver.h): the same equation, parameters, units and returned potential.
///
///   D^2 phi + [(D^2 phi)^2 - (D_i D_j phi)^2] / alpha = OmegaM a delta / beta,   alpha = 3 beta a^4 / (rcH0)^2
///
/// with derivatives in units of the box ([0,1)^3) and the returned fifth-force potential 0.5 * phi so that, with
/// PhiGR = a^2 Phi / (H0 Box)^2, the total force is D(PhiGR + PhiF). As in the uniform solver the equation is solved
/// for D^2 phi with the operator splitting w = 1/3 (the square root of the discriminant).
///
/// The second derivatives on the adaptive grid are AMRMultiGridSolver::get_hessian: the diagonal is the flux
/// (conservative) Laplacian split by direction, the mixed ones are derivatives of the gradient field (on a uniform
/// grid exactly the stencil of DGPSolverCosmology). The mixed terms are lagged within a Gauss-Seidel sweep.
///
///   AMRDGPSolverCosmology<3> dgp(OmegaM, rcH0, beta, H0Box, verbose);
///   std::vector<double> phi;                          // per local leaf of the grid
///   dgp.solve(a, grid, delta, phi);                   // delta per local leaf of the grid
///   dgp.solve(a, grid, delta, phi, &force);           // also the fifth force D(phi) per leaf
///
//======================================================================
template <int NDIM>
class AMRDGPSolverCosmology {
    static_assert(NDIM == 3, "AMRDGPSolverCosmology: the DGP equation (operator splitting w = 1/3) is for NDIM = 3");

  public:
    using Solver = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMRMultiGridSolver<NDIM>;
    using Equation = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMREquation<NDIM>;
    using SolveResult = FML::SOLVERS::AMRMULTIGRIDSOLVER::SolveResult;
    using IndexInt = FML::GRID::IndexInt;
    using Vec = std::array<double, NDIM>;

  private:
    /// Matter density parameter today
    double OmegaM{0.3};
    /// The cross-over scale rcH0/c
    double rcH0_DGP{1.0};
    /// The beta-function (coupling strength = 1/3beta)
    double beta_DGP{1e-6};
    /// The dimensionless number H0 * Box / c (only printed: the equation is in box units)
    double H0Box{};
    bool verbose{true};

    int ngs_pre{2};
    int ngs_post{2};
    int ngs_first{0};
    int maxsteps{100};
    int coarsest_level{-1};

    /// Converged when residual < epsilon, or (relative) residual < epsilon * rms(linear source)
    double epsilon{1e-6};
    bool relative_criterion{false};

    /// Allow a negative discriminant (set to zero there) instead of stopping, see DGPSolverCosmology
    bool allow_negative_discriminant{false};
    long long last_n_negative_discriminant{0};

    /// A previous solution (the returned potential, per leaf of the same grid) to start from
    std::vector<double> previous_solution;

    SolveResult last_result{};

    // The returned potential is forcenorm times the field we solve for
    static constexpr double forcenorm = 0.5;

  public:
    AMRDGPSolverCosmology(double OmegaM, double rcH0_DGP, double beta_DGP, double H0Box, bool verbose)
        : OmegaM(OmegaM), rcH0_DGP(rcH0_DGP), beta_DGP(beta_DGP), H0Box(H0Box), verbose(verbose) {}

    void set_ngs_steps(int pre, int post, int first = 0) {
        ngs_pre = pre;
        ngs_post = post;
        ngs_first = first;
    }
    void set_epsilon(double eps) { epsilon = eps; }
    void set_relative_criterion(bool relative) { relative_criterion = relative; }
    void set_maxsteps(int steps) { maxsteps = steps; }
    /// The refinement level of the coarsest multigrid level (default 2)
    void set_coarsest_level(int level) { coarsest_level = level; }
    void set_allow_negative_discriminant(bool allow) { allow_negative_discriminant = allow; }
    long long get_number_of_negative_discriminant_cells() const { return last_n_negative_discriminant; }
    int get_number_of_vcycles() const { return last_result.iterations; }
    SolveResult get_solve_result() const { return last_result; }

    /// Start the next solve from a previous solution (the returned potential on the same grid, same leaves)
    void set_previous_solution(const std::vector<double> & phi) { previous_solution = phi; }

    /// Solve for the fifth-force potential phi (per local leaf of grid) given the overdensity delta (per local leaf).
    /// If force is given it gets the fifth force D(phi) (box units) in every leaf
    template <class T>
    SolveResult solve(double a,
                      const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                      const std::vector<double> & delta,
                      std::vector<double> & phi,
                      std::vector<Vec> * force = nullptr);
};

template <int NDIM>
template <class T>
typename AMRDGPSolverCosmology<NDIM>::SolveResult
AMRDGPSolverCosmology<NDIM>::solve(double a,
                                   const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                                   const std::vector<double> & delta,
                                   std::vector<double> & phi,
                                   std::vector<Vec> * force) {
    const IndexInt n = grid.get_n_local();
    assert_mpi(IndexInt(delta.size()) == n, "AMRDGPSolverCosmology::solve: delta has the wrong size");

    if (FML::ThisTask == 0 and verbose) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Running AMR multigrid solver for DGP\n";
        std::cout << "# a           : " << a << "\n";
        std::cout << "# OmegaM      : " << OmegaM << "\n";
        std::cout << "# rcH0/c      : " << rcH0_DGP << "\n";
        std::cout << "# beta_DGP    : " << beta_DGP << "\n";
        std::cout << "# H0Box       : " << H0Box << "\n";
        std::cout << "# Convergence : residual < " << epsilon << (relative_criterion ? " * rms(source)" : "") << "\n";
        std::cout << "#=====================================================\n";
    }

    bool periodic = true;
    for (int d = 0; d < NDIM; d++)
        periodic = periodic and grid.get_periodic()[d];

    Solver sol(grid, coarsest_level, verbose);
    const int id_delta = sol.add_external_field(delta);
    sol.set_gradient_field(true);
    // L(phi + c) = L(phi) on a periodic grid and the discretized non-linear term does not have zero mean
    sol.set_subtract_mean_residual(periodic);
    sol.set_ngs_sweeps(ngs_pre, ngs_post, ngs_first);
    sol.set_maxsteps(maxsteps);

    // The linear source OmegaM a delta / beta (volume weighted rms) for the relative criterion
    double source_rms = 0.0;
    for (IndexInt i = 0; i < n; i++)
        source_rms += delta[i] * delta[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    FML::SumOverTasks(&source_rms);
    source_rms = OmegaM * a / beta_DGP * std::sqrt(source_rms);
    sol.set_convergence_criterion_residual(true);
    sol.set_epsilon(relative_criterion and source_rms > 0.0 ? epsilon * source_rms : epsilon);

    if (previous_solution.size() == size_t(n)) {
        std::vector<double> guess(n);
        for (IndexInt i = 0; i < n; i++)
            guess[i] = previous_solution[i] / forcenorm;
        sol.set_initial_guess(guess);
    } else {
        sol.set_initial_guess(0.0);
    }

    // The equation (as in DGPSolverCosmology)
    const double alpha = 3.0 * beta_DGP * (a * a * a * a) / (rcH0_DGP * rcH0_DGP);
    const double source_factor = OmegaM * a * alpha / beta_DGP;
    bool count_negative = false;
    long long n_negative = 0;
    Equation equation = [&](const Solver & s, int level, IndexInt i) {
        double dlap = 0.0;
        const auto H = s.get_hessian(level, i, &dlap);
        const double w = 1.0 / 3.0;
        double laplacian = 0.0, cross = 0.0;
        for (int p = 0; p < NDIM; p++) {
            laplacian += H[p][p];
            for (int q = 0; q < NDIM; q++)
                cross += H[p][q] * H[p][q];
        }
        const double sigma = cross + source_factor * s.get_external_field(id_delta, level, i) - w * laplacian * laplacian;
        double disc = 1.0 + 4.0 * (1.0 - w) * sigma / (alpha * alpha);
        if (disc < 0.0) {
            if (count_negative)
                n_negative++;
            disc = 0.0;
        }
        const double eq = laplacian - 0.5 * alpha * (std::sqrt(disc) - 1.0) / (1.0 - w);
        return std::pair<double, double>{eq, dlap};
    };

    last_result = sol.solve(equation);

    // The solution is only real if the discriminant is non-negative everywhere on the grid
    sol.get_residual(); // updates the gradient field
    count_negative = true;
    for (IndexInt i = 0; i < n; i++)
        equation(sol, 0, i);
    count_negative = false;
    FML::SumOverTasks(&n_negative);
    last_n_negative_discriminant = n_negative;
    if (n_negative > 0 and not allow_negative_discriminant) {
        if (FML::ThisTask == 0)
            std::cerr << "[AMRDGPSolverCosmology] The discriminant of the DGP equation is negative in " << n_negative
                      << " cells at a = " << a << ": no real solution (check the density field)\n"
                      << std::flush;
        FML::abort_mpi(1);
    }

    // The fifth-force potential (and force)
    const auto & f = sol.get_solution();
    phi.resize(n);
    double fmin = std::numeric_limits<double>::max(), fmax = -fmin, fmean = 0.0;
    for (IndexInt i = 0; i < n; i++) {
        phi[i] = forcenorm * f[i];
        fmin = std::min(fmin, phi[i]);
        fmax = std::max(fmax, phi[i]);
        fmean += phi[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    }
    if (force) {
        *force = sol.get_solution_gradient();
        for (auto & g : *force)
            for (auto & x : g)
                x *= forcenorm;
    }
    FML::MinOverTasks(&fmin);
    FML::MaxOverTasks(&fmax);
    FML::SumOverTasks(&fmean);
    if (FML::ThisTask == 0 and verbose) {
        std::cout << "#=====================================================\n";
        std::cout << "# V-cycles: " << last_result.iterations << (last_result.converged() ? " (converged)" : " (NOT converged)")
                  << "  residual " << last_result.residual << "\n";
        std::cout << "# The minimum value of phi_DGP : " << fmin - fmean << "\n";
        std::cout << "# The mean value of phi_DGP    : " << fmean << "\n";
        std::cout << "# The maximum value phi_DGP    : " << fmax - fmean << "\n";
        std::cout << "#=====================================================\n\n";
    }
    return last_result;
}

#endif
