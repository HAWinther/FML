#include <FML/MultigridSolver/MultiGridSolver.h>
#include <cmath>
#include <iostream>

//======================================================================
// Solve the f(R) gravity equation D^2[b(u)u] = f(rho,u)
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

void FofrSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 64;       // Number of cells per dimension
    const int Nlevels = -1; // -1 = let the solver pick this
    const int Ndim = 2;
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
    const double _nfofr = 1.0;
    const double _fofr0 = 1e-6;
    const double _omegam = 0.3;
    const double _aexp = 1.0;
    const double _box = 10.0;

    // Deried factors needed to define the equation
    const double coverH0_Mpch = 2997.92458;
    const double _fac1 = 1.0 + 4.0 * _aexp * _aexp * _aexp * (1.0 - _omegam) / _omegam;
    const double _fac2 =
        (1.0 + 4.0 * (1.0 - _omegam) / _omegam) * _aexp * _aexp * _aexp * pow(_fofr0 * _aexp, 1.0 / (_nfofr + 1.0));
    const double _prefac = pow(_box / coverH0_Mpch, 2) * _omegam * _aexp;

    //======================================================================
    // The cosmological background value (used to set the IC)
    //======================================================================
    const double _fofr_background = _fofr0 * pow(_fac2 / _fac1, _nfofr + 1);

    //======================================================================
    // The overdensity field from a formula (as peak in the center of the box)
    //======================================================================
    auto delta_analytic = [](const std::array<double, Ndim> & x) -> double {
        double r2 = 0.0;
        for (int idim = 0; idim < Ndim; idim++) {
            r2 += (x[idim] - 0.5) * (x[idim] - 0.5);
        }
        const double sigma = 1.0 / double(N), s2 = sigma * sigma;
        const double delta = 1.0 / sqrt(2.0 * M_PI * s2) * std::exp(-0.5 * r2 / s2) - 1.0;
        return delta;
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
    g.set_initial_guess(log(_fofr_background));

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
        // The solution in the current cell
        //======================================================================
        const double f = sol->get_Field(level, index);

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
        // (Faster to not use the built in function and code this directly
        // as we do a lot of double work)
        //======================================================================
        std::function<SolverType(int, IndexInt)> b = [&](int lev, IndexInt ind) -> double {
            return std::exp(sol->get_Field(lev, ind));
        };
        std::function<SolverType(int, IndexInt)> db = [&](int lev, IndexInt ind) -> double {
            return std::exp(sol->get_Field(lev, ind));
        };
        auto kinetic = sol->get_BLaplacian(level, index_list, b);
        auto dkinetic = sol->get_derivBLaplacian(level, index_list, b, db);

        //======================================================================
        // The right hand side source and the derivative of it
        //======================================================================
        double source = _prefac * (delta + _fac1 - _fac2 * std::exp(-f / (_nfofr + 1.0)));
        double dsource = _prefac * (_fac2 * std::exp(-f / (_nfofr + 1.0))) / (1.0 + _nfofr);

        //======================================================================
        // Returns the equation and the derivative to the solver
        //======================================================================
        return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
    };

    // Solve the equation and fetch the solution
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    // Print the solution (x,y,...,r, f_R/f_R0, delta)
    std::array<double, Ndim> center;
    center.fill(0.5);
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto pos = sol.get_pos(index);
        if (FML::ThisTask == 0) {
            if (index % N == 0 and index > 0)
                std::cout << "\n";
            for (int idim = 0; idim < Ndim; idim++)
                std::cout << pos[idim] << " ";
            double r = sol.get_radial_distance(index, center);
            std::cout << r << " " << std::exp(sol[index]) / _fofr0 << " " << delta_analytic(pos) << "\n";
        }
    }
}

int main() { FofrSolver(); }
