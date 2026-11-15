#ifndef AMRSYMMETRONSOLVER_HEADER
#define AMRSYMMETRONSOLVER_HEADER

#include <FML/AdaptiveMPIGrid/AdaptiveMPIGrid.h>
#include <FML/Global/Global.h>
#include <FML/MultigridSolver/AMRMultiGridSolver.h>
#include <FML/Units/Units.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

//======================================================================
///
/// The symmetron fifth-force equation on an adaptively refined grid (AdaptiveMPIGrid), the AMR version of
/// SymmetronSolverCosmology (SymmetronSolver.h): the same equation, parameters, units and returned potential.
/// Generalised with the powers (N, M) of the potential (Brax, Davis, Li, Winther & Zhao 2012, arXiv:1206.3568;
/// even integers M > N >= 2, (2, 4) is the original symmetron). With chi = phi / phi_* and xi = H0 L / c:
///
///   D^2 chi = norm [ chi^(N-1) ((a_ssb/a)^3 (1 + delta) - 1) + chi^(M-1) ],   norm = a^2 (H0 Box)^2 / ((M - N) xi^2)
///
/// (box units, [0,1)^NDIM), background chi = (1 - (a_ssb/a)^3)^(1/(M-N)) (0 before symmetry breaking). The returned
/// fifth-force potential is PhiF = forcenorm chi^N, forcenorm = 3 (M - N)/N OmegaM a^2 beta^2 (xi / H0Box)^2 / a_ssb^3,
/// so that with PhiGR = a^2 Phi / (H0 Box)^2 the total force is D(PhiGR + PhiF). The equation is local in chi (no
/// derivatives beyond the Laplacian), so it uses the flux Laplacian of AMRMultiGridSolver and a Newton update per cell.
///
///   AMRSymmetronSolverCosmology<3> sym(OmegaM, assb, beta, L_mpch, H0Box, verbose);   // (N, M) = (2, 4)
///   std::vector<double> phi;                          // per local leaf of the grid
///   sym.solve(a, grid, delta, phi);                   // delta per local leaf of the grid
///   sym.solve(a, grid, delta, phi, &force);           // also the fifth force D(PhiF) per leaf
///   sym.get_field();                                  // chi of the last solve (it also starts the next solve on the
///                                                     // same leaves, see set_previous_field)
///
/// The field is started from the background value (or a previous field), so it stays on the positive branch: the
/// equation has the symmetric solution -chi too, and domain walls between the two are not followed unless they are
/// in the initial field.
///
//======================================================================
template <int NDIM>
class AMRSymmetronSolverCosmology {
  public:
    using Solver = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMRMultiGridSolver<NDIM>;
    using Equation = FML::SOLVERS::AMRMULTIGRIDSOLVER::AMREquation<NDIM>;
    using SolveResult = FML::SOLVERS::AMRMULTIGRIDSOLVER::SolveResult;
    using IndexInt = FML::GRID::IndexInt;
    using Vec = std::array<double, NDIM>;

  private:
    /// Matter density parameter today
    double OmegaM{0.3};
    /// The symmetry breaking scalefactor
    double assb{0.5};
    /// The coupling strength
    double beta{1.0};
    /// The range in vacuum in Mpc/h
    double L_mpch{1.0};
    /// The dimensionless number H0 * Box / c
    double H0Box{};
    /// The powers of the potential (2, 4: the original symmetron)
    double N{2.0};
    double M{4.0};
    bool verbose{true};

    int ngs_pre{4};
    int ngs_post{4};
    int ngs_first{0};
    int maxsteps{100};
    int coarsest_level{-1};

    /// Converged when residual < epsilon, or (relative) residual < epsilon * rms(linear source)
    double epsilon{1e-6};
    bool relative_criterion{false};

    /// chi of the last solve, and a field to start the next solve from (per leaf of the same grid)
    std::vector<double> field;
    std::vector<double> previous_field;

    SolveResult last_result{};

    /// x^n for an integer n >= 0 (x can be negative)
    static double ipow(double x, int n) {
        double r = 1.0;
        for (int i = 0; i < n; i++)
            r *= x;
        return r;
    }

  public:
    AMRSymmetronSolverCosmology(double OmegaM,
                                double assb,
                                double beta,
                                double L_mpch,
                                double H0Box,
                                bool verbose,
                                double N = 2.0,
                                double M = 4.0)
        : OmegaM(OmegaM), assb(assb), beta(beta), L_mpch(L_mpch), H0Box(H0Box), N(N), M(M), verbose(verbose) {
        auto even = [](double x) { return x == std::round(x) and std::fmod(x, 2.0) == 0.0; };
        if (not(even(N) and even(M) and N >= 2.0 and M > N))
            throw std::invalid_argument("[AMRSymmetronSolverCosmology] Need even integers M > N >= 2");
    }

    /// Gauss-Seidel sweeps before and after the coarse grid correction (and on level 0 before the first V-cycle)
    void set_ngs_steps(int pre, int post, int first = 0) {
        ngs_pre = pre;
        ngs_post = post;
        ngs_first = first;
    }
    void set_epsilon(double eps) { epsilon = eps; }
    /// Converged when the rms residual is below epsilon (fiducial) or, with relative = true, below epsilon times the
    /// rms of the linear source norm (a_ssb/a)^3 chi_bar^(N-1) delta (absolute before symmetry breaking)
    void set_relative_criterion(bool relative) { relative_criterion = relative; }
    void set_maxsteps(int steps) { maxsteps = steps; }
    /// The refinement level of the coarsest multigrid level (default 2)
    void set_coarsest_level(int level) { coarsest_level = level; }
    int get_number_of_vcycles() const { return last_result.iterations; }
    SolveResult get_solve_result() const { return last_result; }

    /// The field chi = phi / phi_* of the last solve (per local leaf)
    const std::vector<double> & get_field() const { return field; }
    /// Start the next solve from this field chi (per local leaf of the grid of the next solve; the default is the
    /// field of the last solve if the number of leaves matches, else the background value)
    void set_previous_field(const std::vector<double> & chi) { previous_field = chi; }

    /// The cosmological background value chi = phi / phi_*
    double get_phi_background(double a) const {
        return a < assb ? 0.0 : std::pow(1.0 - (assb * assb * assb) / (a * a * a), 1.0 / (M - N));
    }
    /// The factor forcenorm in the returned PhiF = forcenorm chi^N
    double get_forcenorm(double a) const {
        const double H0L = L_mpch / FML::UTILS::SI::c_over_H0_mpch;
        return 3.0 * (M - N) / N * OmegaM * a * a * beta * beta * (H0L / H0Box) * (H0L / H0Box) /
               (assb * assb * assb);
    }
    /// The factor norm in the equation
    double get_norm(double a) const {
        const double H0L = L_mpch / FML::UTILS::SI::c_over_H0_mpch;
        return a * a * (H0Box * H0Box) / ((M - N) * H0L * H0L);
    }

    /// Solve for the fifth-force potential PhiF = forcenorm chi^N (per local leaf of grid) given the overdensity delta
    /// (per local leaf). If force is given it gets the fifth force D(PhiF) = forcenorm N chi^(N-1) D(chi) (box units)
    template <class T>
    SolveResult solve(double a,
                      const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                      const std::vector<double> & delta,
                      std::vector<double> & phi,
                      std::vector<Vec> * force = nullptr);
};

template <int NDIM>
template <class T>
typename AMRSymmetronSolverCosmology<NDIM>::SolveResult
AMRSymmetronSolverCosmology<NDIM>::solve(double a,
                                         const FML::GRID::AdaptiveMPIGrid<NDIM, T> & grid,
                                         const std::vector<double> & delta,
                                         std::vector<double> & phi,
                                         std::vector<Vec> * force) {
    const IndexInt n = grid.get_n_local();
    assert_mpi(IndexInt(delta.size()) == n, "AMRSymmetronSolverCosmology::solve: delta has the wrong size");

    const double f0 = get_phi_background(a);
    const double norm = get_norm(a);
    const double fac = (assb * assb * assb) / (a * a * a);
    const int iN = int(std::lround(N)), iM = int(std::lround(M));

    if (FML::ThisTask == 0 and verbose) {
        std::cout << "\n";
        std::cout << "#=====================================================\n";
        std::cout << "# Running AMR multigrid solver for the symmetron\n";
        std::cout << "# a           : " << a << "\n";
        std::cout << "# OmegaM      : " << OmegaM << "\n";
        std::cout << "# assb        : " << assb << "\n";
        std::cout << "# beta        : " << beta << "\n";
        std::cout << "# L_mpch      : " << L_mpch << "\n";
        std::cout << "# (N, M)      : (" << N << ", " << M << ")\n";
        std::cout << "# H0Box       : " << H0Box << "\n";
        std::cout << "# phi/phi0(a) : " << f0 << "\n";
        std::cout << "# Convergence : residual < " << epsilon << (relative_criterion ? " * rms(linear source)" : "") << "\n";
        std::cout << "#=====================================================\n";
    }

    Solver sol(grid, coarsest_level, verbose);
    const int id_delta = sol.add_external_field(delta);
    sol.set_ngs_sweeps(ngs_pre, ngs_post, ngs_first);
    sol.set_maxsteps(maxsteps);

    // The linear source norm fac chi_bar^(N-1) delta (volume weighted rms) for the relative criterion
    double source_rms = 0.0;
    for (IndexInt i = 0; i < n; i++)
        source_rms += delta[i] * delta[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    FML::SumOverTasks(&source_rms);
    source_rms = norm * fac * ipow(f0, iN - 1) * std::sqrt(source_rms);
    sol.set_convergence_criterion_residual(true);
    sol.set_epsilon(relative_criterion and source_rms > 0.0 ? epsilon * source_rms : epsilon);

    // Start from a previous field (given, or the last one on a grid with as many leaves), else the background
    const std::vector<double> & start = previous_field.size() == size_t(n) ? previous_field : field;
    if (start.size() == size_t(n) and a >= assb)
        sol.set_initial_guess(start);
    else
        sol.set_initial_guess(f0);
    previous_field.clear();

    // D^2 chi - norm [chi^(N-1) (fac (1 + delta) - 1) + chi^(M-1)] = 0 (as SymmetronSolverCosmology)
    Equation equation = [&](const Solver & s, int level, IndexInt i) {
        auto [lap, dlap] = s.get_laplacian(level, i);
        const double f = s.get_field(level, i);
        const double d = s.get_external_field(id_delta, level, i);
        const double fN2 = ipow(f, iN - 2);
        const double fM2 = ipow(f, iM - 2);
        const double source = norm * (fN2 * f * (fac * (1.0 + d) - 1.0) + fM2 * f);
        const double dsource = norm * ((N - 1.0) * fN2 * (fac * (1.0 + d) - 1.0) + (M - 1.0) * fM2);
        return std::pair<double, double>{lap - source, dlap - dsource};
    };

    last_result = sol.solve(equation);

    // The fifth-force potential (and force)
    field = sol.get_solution();
    const double forcenorm = get_forcenorm(a);
    phi.resize(n);
    double fmin = std::numeric_limits<double>::max(), fmax = -fmin, fmean = 0.0;
    for (IndexInt i = 0; i < n; i++) {
        phi[i] = forcenorm * ipow(field[i], iN);
        fmin = std::min(fmin, field[i]);
        fmax = std::max(fmax, field[i]);
        fmean += field[i] * std::ldexp(1.0, -NDIM * grid.get_cell(i).level);
    }
    if (force) {
        *force = sol.get_solution_gradient();
        for (IndexInt i = 0; i < n; i++) {
            const double dphidchi = forcenorm * N * ipow(field[i], iN - 1);
            for (auto & x : (*force)[i])
                x *= dphidchi;
        }
    }
    FML::MinOverTasks(&fmin);
    FML::MaxOverTasks(&fmax);
    FML::SumOverTasks(&fmean);
    if (FML::ThisTask == 0 and verbose) {
        std::cout << "#=====================================================\n";
        std::cout << "# V-cycles: " << last_result.iterations
                  << (last_result.converged() ? " (converged)" : " (NOT converged)") << "  residual "
                  << last_result.residual << "\n";
        std::cout << "# The minimum value of phi/phi_* : " << fmin << "\n";
        std::cout << "# The mean value of phi/phi_*    : " << fmean << "\n";
        std::cout << "# The maximum value phi/phi_*    : " << fmax << "\n";
        std::cout << "# Background field value phi(a)  : " << f0 << "\n";
        std::cout << "#=====================================================\n\n";
    }
    return last_result;
}

#endif
