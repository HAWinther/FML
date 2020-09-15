#include <FML/MultigridSolver/MultiGridSolver.h>
#include <cmath>
#include <iostream>

//======================================================================
// Solve the symmetron gravity equation
// D^2phi = 0.5 a^2(B/L)^2[ (assb/a)^3(1+delta) phi - phi + phi^3  ]
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

void SymmetronSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 128;      // Number of cells per dimension
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
    const double L = 0.1;     // Range of field in Mpc/h
    const double assb = 0.33; // Symmetry breaking scalefactor
    const double aexp = 1.0;  // Scalefactor today
    const double box = 100.0; // Boxsize in Mpc/h

    //======================================================================
    // The cosmological background value (used to set the IC)
    //======================================================================
    const double phi_background = aexp > assb ? std::sqrt(1.0 - std::pow(assb / aexp, 3)) : 0.0;

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
    g.set_initial_guess(phi_background);

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
        const auto h = sol->get_Gridspacing(level);

        //======================================================================
        // Get a list of the index of the closest cells needed to compute
        // gradients and the laplacian
        //======================================================================
        auto index_list = sol->get_neighbor_gridindex(level, index);

        //======================================================================
        // The solution in the current cell
        //======================================================================
        const double phi = sol->get_Field(level, index);

        //======================================================================
        // Position of the cell in the global box
        //======================================================================
        const auto coordinate = sol->get_Coordinate(level, index);

        //======================================================================
        // Set the (over) density field from formula
        //======================================================================
        double delta = delta_analytic(coordinate);

        //======================================================================
        // The equation D^2 phi - source(phi) = 0  and the derivative of it
        //======================================================================
        const auto laplacian = sol->get_Laplacian(level, index_list);
        const double dlaplacian = -2.0 * Ndim / (h * h);
        const double source = 0.5 * aexp * aexp * (box / L) *
                              (box / L) * ((assb / aexp) * (assb / aexp) * (assb / aexp) * (1 + delta) * phi -
                                                phi + phi * phi * phi);
        const double dsource = 0.5 * aexp * aexp * (box / L) * (box / L) *
                               ((assb / aexp) * (assb / aexp) * (assb / aexp) * (1 + delta) - 1.0 + 3.0 * phi * phi);

        //======================================================================
        // Returns the equation and the derivative to the solver
        //======================================================================
        return std::pair<SolverType, SolverType>{laplacian - source, dlaplacian - dsource};
    };

    // Solve the equation and fetch the solution
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    // Print the solution (r, phi(r), delta(r))
    std::array<double, Ndim> center;
    center.fill(0.5);
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto pos = sol.get_pos(index);
        if (FML::ThisTask == 0) {
            if (index % N == 0 and index > 0)
                std::cout << "\n";
            double r = sol.get_radial_distance(index, center);
            std::cout << r << " " << sol[index] << " " << delta_analytic(pos) << "\n";
        }
    }
}

int main() { SymmetronSolver(); }
