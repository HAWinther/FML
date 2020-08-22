#include <FML/MultigridSolver/MultiGridSolver.h>
#include <cmath>
#include <iostream>

//======================================================================
// Solve the continuity equation delta' + D((1+delta)v) = 0
// with the linear regime approx delta' = f H delta and v = Dphi but
// keeping the rest non-linear
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

void ContinuitySolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 256;      // Number of cells per dimension
    const int Nlevels = -1; // -1 = let the solver pick this
    const int Ndim = 1;
    const int nleft = 1;
    const int nright = 1;
    const bool verbose = true;
    const bool periodic = true;
    const double epsilon = 1e-7;
    const int ngs_fine = 10;   // Number of NGS sweeps on the main grid (more is better, but slower)
    const int ngs_coarse = 10; // Number of NGS sweeps on coarser grids (more is better, but slower)
    const int ngs_first = 10;  // Number of NGS sweeps the very first time (if it fails try to increase this)

    //======================================================================
    // Parameters defining the equation
    //======================================================================
    const double _omegam = 0.3;
    const double _aexp = 1.0;
    const double _box = 1000.0;
    const double c_kms = 299792.458;
    const double H_invMpch = 100.0 / c_kms * sqrt(_omegam / (_aexp * _aexp * _aexp) + 1.0 - _omegam);
    const double f = pow(_omegam / (_aexp * _aexp * _aexp) / (_omegam / (_aexp * _aexp * _aexp) + 1.0 - _omegam), 0.55);

    //======================================================================
    // The overdensity field (just a sine wave)
    //======================================================================
    auto delta_analytic = [](const std::array<double, Ndim> & x) -> double {
        return 0.9 * std::sin(2.0 * M_PI * x[0] + M_PI / 4);
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
    g.set_initial_guess(0.0);

    //======================================================================
    // Set the convergence criterion
    // (ConvergenceCriterionResidual is rms-residual < epsilon)
    //======================================================================
    MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
        return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
    };

    //======================================================================
    // Implement the equation to be solved
    //======================================================================

    MultiGridFunction<Ndim, double> Equation = [&](MultiGridSolver<Ndim, SolverType> * sol, int level, IndexInt index) {
        //======================================================================
        // Get a list of the index of the closest cells needed to compute
        // gradients and the laplacian
        //======================================================================
        auto index_list = sol->get_neighbor_gridindex(level, index);

        //======================================================================
        // Position of the cell in the global box
        //======================================================================
        const auto coordinate = sol->get_Coordinate(level, index);

        //======================================================================
        // Set the (over) density field from formula
        //======================================================================
        double delta = delta_analytic(coordinate);

        //======================================================================
        // Calculate the kinetic term: D[ b[f] Df ] with b = exp(x)
        // (Faster to not use the built in functions and code this directly
        // as we do a lot of double work, but whatever)
        //======================================================================
        std::function<SolverType(int, IndexInt)> b = [&](int lev, IndexInt ind) -> SolverType {
            auto coord = sol->get_Coordinate(lev, ind);
            return std::max((1.0 + delta_analytic(coord)), 0.1);
        };
        std::function<SolverType(int, IndexInt)> db = [&]([[maybe_unused]] int lev,
                                                          [[maybe_unused]] IndexInt ind) -> SolverType { return 0.0; };
        auto kinetic = sol->get_BLaplacian(level, index_list, b);
        auto dkinetic = sol->get_derivBLaplacian(level, index_list, b, db);

        // Fully linear
        // auto kinetic  = sol->get_Laplacian(level, index_list);
        // auto dkinetic = sol->get_derivLaplacian(level, index_list);

        //======================================================================
        // The right hand side source and the derivative of it
        //======================================================================
        double source = -H_invMpch * _box * f * delta;
        double dsource = 0.0;

        //======================================================================
        // Returns the equation and the derivative to the solver
        //======================================================================
        return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
    };

    // Solve the equation and fetch the solution
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    // Print the solution (x,y,...,vx,vy,..., phi, delta)
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto pos = sol.get_pos(index);
        auto vel = sol.get_gradient(index);
        if (FML::ThisTask == 0) {
            if (index % N == 0 and index > 0)
                std::cout << "\n";
            for (int idim = 0; idim < Ndim; idim++)
                std::cout << pos[idim] << " ";
            for (int idim = 0; idim < Ndim; idim++)
                std::cout << vel[idim] * c_kms << " ";
            std::cout << " " << sol[index] << " " << delta_analytic(pos) << " ";
            std::cout << "\n";
        }
    }
}

int main() { ContinuitySolver(); }
