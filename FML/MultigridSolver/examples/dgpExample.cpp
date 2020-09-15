#include <FML/MultigridSolver/MultiGridSolver.h>
#include <cmath>
#include <iostream>

//======================================================================
// Solve the nDGP equation
// D^2phi + A((D^2phi)^2 - (Dij phi)^2) = B delta
// Convergence of this kind of equation is not always smooth
// What usually happens is that the residual saturates at some value
// So experiment with what is a realistic convergence criterion
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

void DGPSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 128;      // Number of cells per dimension
    const int Nlevels = -1; // -1 = let the solver pick this
    const int Ndim = 3;
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
    const double rcH0 = 1.0;   // Crossover scale
    const double aexp = 1.0;   // Scalefactor today
    const double OmegaM = 0.3; // Density parameter today
    const double HoverH0 = std::sqrt(OmegaM / (aexp * aexp * aexp) + 1.0 - OmegaM);
    const double beta = 1.0 + 2.0 * rcH0 * HoverH0 * (1.0 - OmegaM / (2.0 * aexp * aexp * aexp * HoverH0 * HoverH0));
    const double alpha = 3.0 * beta * (aexp * aexp * aexp * aexp) / (rcH0 * rcH0);

    //======================================================================
    // The cosmological background value (used to set the IC)
    //======================================================================
    const double phi_background = 0.0;

    //======================================================================
    // The overdensity field: tophat in the center of the box
    //======================================================================
    const double rsphere = 0.05;
    const double vol_in = 4.0 * M_PI / 3.0 * rsphere * rsphere * rsphere;
    const double vol_out = 1.0 - vol_in;
    const double rho_out = 0.1;
    const double rho_in = (1.0 - rho_out * vol_out) / vol_in;

    if (FML::ThisTask == 0)
        std::cout << "Density of the sphere: " << rho_in << "\n";
    auto delta_analytic = [=](const std::array<double, Ndim> & x) -> double {
        double r2 = 0.0;
        for (int idim = 0; idim < Ndim; idim++) {
            r2 += (x[idim] - 0.5) * (x[idim] - 0.5);
        }
        return std::sqrt(r2) < rsphere ? rho_in - 1.0 : rho_out - 1.0;
    };

    //======================================================================
    // Set up the solver
    //======================================================================
    MultiGridSolver<Ndim, SolverType> g(N, Nlevels, verbose, periodic, nleft, nright);

    // Set some options
    g.set_epsilon(epsilon);
    g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
    g.set_epsilon(epsilon);

    // Set maximum number of steps
    g.set_maxsteps(10);

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
        // Get a list of the index of the closest 27 cells
        //======================================================================
        auto cube_index_list = sol->get_cube_gridindex(level, index);
        double phi = sol->get_Field(level, index);

        //======================================================================
        // Second derivatives (as written this only works for NDIM=3)
        //======================================================================
        double phi_xy =
            (sol->get_Field(level, cube_index_list[13 + 3 + 1]) + sol->get_Field(level, cube_index_list[13 - 3 - 1]) -
             sol->get_Field(level, cube_index_list[13 + 3 - 1]) - sol->get_Field(level, cube_index_list[13 - 3 + 1])) *
            0.25 / (h * h);

        double phi_yz =
            (sol->get_Field(level, cube_index_list[13 + 9 + 3]) + sol->get_Field(level, cube_index_list[13 - 9 - 3]) -
             sol->get_Field(level, cube_index_list[13 + 9 - 3]) - sol->get_Field(level, cube_index_list[13 - 9 + 3])) *
            0.25 / (h * h);

        double phi_zx =
            (sol->get_Field(level, cube_index_list[13 + 9 + 1]) + sol->get_Field(level, cube_index_list[13 - 9 - 1]) -
             sol->get_Field(level, cube_index_list[13 + 9 - 1]) - sol->get_Field(level, cube_index_list[13 - 9 + 1])) *
            0.25 / (h * h);

        double phi_xx = (sol->get_Field(level, cube_index_list[13 + 1]) +
                         sol->get_Field(level, cube_index_list[13 - 1]) - 2 * phi) /
                        (h * h);

        double phi_yy = (sol->get_Field(level, cube_index_list[13 + 3]) +
                         sol->get_Field(level, cube_index_list[13 - 3]) - 2 * phi) /
                        (h * h);

        double phi_zz = (sol->get_Field(level, cube_index_list[13 + 9]) +
                         sol->get_Field(level, cube_index_list[13 - 9]) - 2 * phi) /
                        (h * h);

        //======================================================================
        // Position of the cell in the global box
        //======================================================================
        const auto coordinate = sol->get_Coordinate(level, index);

        //======================================================================
        // Set the (over) density field from formula
        //======================================================================
        const double delta = delta_analytic(coordinate);

        //======================================================================
        // The equation D^2 phi + fac*(Sum phi_iiphi_jj - phi_ij^2) - source = 0
        // and the derivative of it. Here we use the operator splitting method to make it stable
        //======================================================================
        const double w = 1.0 / 3.0;
        const double laplacian = phi_xx + phi_yy + phi_zz;
        const double cross = phi_xx * phi_xx + phi_yy * phi_yy + phi_zz * phi_zz +
                             2.0 * (phi_xy * phi_xy + phi_yz * phi_yz + phi_zx * phi_zx);
        const double sigma =
            (cross + alpha / beta * OmegaM * aexp * delta - w * laplacian * laplacian) / (alpha * alpha);
        const double disc = 1.0 + 4.0 * (1 - w) * sigma;
        const double eq = laplacian - 0.5 * alpha * (std::sqrt(disc) - 1.0) / (1.0 - w);
        const double deq = -2.0 * Ndim / (h * h);

        // Check for fatal error
        if (disc < 0.0) {
            std::cout << disc << " is negative! Solution fails to remain real" << std::endl;
            exit(1);
        }

        //======================================================================
        // Returns the equation and the derivative to the solver
        //======================================================================
        return std::pair<SolverType, SolverType>{eq, deq};
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

int main() { DGPSolver(); }
