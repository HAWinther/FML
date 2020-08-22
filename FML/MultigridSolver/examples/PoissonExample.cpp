#include <FML/MultigridSolver/MultiGridSolver.h>
#include <cmath>
#include <iostream>

//======================================================================
// Solve the Poisson equation D^2 f = source
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

void PoissonSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 128;
    const int Nlevels = -1;
    const int Ndim = 2;
    const int nleft = 1;
    const int nright = 1;
    const bool verbose = true;
    const bool periodic = true;
    const double epsilon = 1e-7;
    const int ngs_fine = 10;
    const int ngs_coarse = 10;
    const int ngs_first = 10;

    //======================================================================
    // The overdensity field from a formula (as peak in the center of the box)
    // NB: solution only defined up to a constant. The mean in the initial
    // guess will be this constant
    //======================================================================
    auto sol_analytic = [=](std::array<double, Ndim> & x) -> SolverType {
        SolverType sol = 1.0;
        for (int idim = 0; idim < Ndim; idim++)
            sol *= sin(4 * 2.0 * M_PI * x[idim]);
        return sol;
        // Test complex source in Ndim = 1 (SolverType = std::complex<double>)
        // sol = std::exp( std::complex<double>(0,1) * 2.0 * M_PI * x[0]);
    };
    auto source_analytic = [&](std::array<double, Ndim> & x) -> SolverType {
        return (-4.0 * M_PI * M_PI) * 16 * Ndim * sol_analytic(x);
        // Test complex source in Ndim=1
        // return -4.0*M_PI*M_PI*std::exp( std::complex<double>(0,1) * 2.0 * M_PI * x[0]);
    };

    //======================================================================
    // Set up the solver
    //======================================================================
    MultiGridSolver<Ndim, SolverType> g(N, Nlevels, verbose, periodic, nleft, nright);

    // Set some options
    g.set_epsilon(epsilon);
    g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
    g.set_epsilon(epsilon);

    // Set the initial guess
    g.set_initial_guess(SolverType(0.0));

    //======================================================================
    // Set the convergence criterion
    // (ConvergenceCriterionResidual is rms-residual < epsilon)
    //======================================================================
    MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
        return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
    };

    //======================================================================
    // Implement the equation to be solved
    // Poisson equation: L = D^2f - source and derivative dL/df
    //======================================================================

    MultiGridFunction<Ndim, SolverType> Equation =
        [&](MultiGridSolver<Ndim, SolverType> * sol, int level, IndexInt index) {
            auto index_list = sol->get_neighbor_gridindex(level, index);
            auto coordinate = sol->get_Coordinate(level, index);
            auto L = sol->get_Laplacian(level, index_list) - source_analytic(coordinate);
            auto dL = sol->get_derivLaplacian(level, index_list);
            return std::pair<SolverType, SolverType>{L, dL};
        };

    // Solve the equation and fetch the solution
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    // Print the solution together with analytical solution
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto pos = sol.get_pos(index);
        if (FML::ThisTask == 0) {
            if (index % N == 0 and index > 0)
                std::cout << "\n";
            for (int idim = 0; idim < Ndim; idim++)
                std::cout << pos[idim] << " ";
            std::cout << sol[index] << " " << sol_analytic(pos) << "\n";
        }
    }
}

int main() { PoissonSolver(); }
