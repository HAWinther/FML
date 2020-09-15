#include <FML/MultigridSolver/MultiGridSolver.h>
#include <array>
#include <cmath>
#include <iostream>

//======================================================================
// A genertic type implementing arithmetic operations
// as an example on how to use it for nonstandard types
// Needed if we for example want to solve system of N equations
//======================================================================

struct MyType {
    static const int N = 1;  // 2 components/equation
    std::array<double, N> y; // The solution variables
    // We need to initialize the variable from a double
    MyType(double x = 0.0) {
        for (int i = 0; i < N; i++)
            y[i] = x;
    }
    MyType(const MyType & rhs) {
        for (int i = 0; i < N; i++)
            y[i] = rhs.y[i];
    }
    // We need to define all the arithmetic functions
#define OPS(OP)                                                                                                        \
    auto operator OP(const MyType & rhs)->MyType & {                                                                   \
        for (int i = 0; i < N; i++) {                                                                                  \
            this->y[i] OP rhs.y[i];                                                                                    \
        }                                                                                                              \
        return *this;                                                                                                  \
    }
    OPS(*=);
    OPS(/=);
    OPS(+=);
    OPS(-=);
#undef OPS
};

// We need to define all the arithmetic functions
#define OPS(OP)                                                                                                        \
    auto operator OP(const MyType lhs, const MyType & rhs)->MyType {                                                   \
        MyType res;                                                                                                    \
        for (int i = 0; i < res.N; i++) {                                                                              \
            res.y[i] = lhs.y[i] OP rhs.y[i];                                                                           \
        }                                                                                                              \
        return res;                                                                                                    \
    }
OPS(*);
OPS(/);
OPS(+);
OPS(-);
#undef OPS

// Finally we need to define the absolute value. Can be any norm
// This is what is used to compute rms_residual and determine convergence
double AbsoluteValue(MyType & rhs) {
    double norm2 = 0.0;
    for (int i = 0; i < rhs.N; i++)
        norm2 += rhs.y[i] * rhs.y[i];
    return std::sqrt(norm2);
}

// For printing the type to screen
std::ostream & operator<<(std::ostream & os, const MyType & x) {
    for (int i = 0; i < x.N; i++)
        os << x.y[i] << " ";
    return os;
}

//======================================================================
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

//======================================================================
// Solve the Poisson equation D^2 f = source for an analytical source
//======================================================================
void TestMultiGridSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = MyType;   // Can be any type (you can change to double, std::complex<double>, etc)
                                 // as long as it implements normal arithmetic operations
    const int N = 128;           // Gridcells per dimension
    const int Nlevels = -1;      // -1 = let the solver determine this
    const int Ndim = 2;          // Number of dimensions in the underlying domain
    const int nleft = 1;         // Extra grid-slices on the left (for MPI comm) Minimum 1
    const int nright = 1;        // Extra grid-slices on the right (for MPI comm) Minimum 1
    const bool verbose = true;   // Print info as we go along
    const bool periodic = true;  // Periodic box
    const int max_steps = 100;   // Maximum number of V-cycles before giving up
    const double epsilon = 1e-8; // Convergence criterion for residual
    const int ngs_fine = 10;     // Number of NGS sweeps on the main grid (more is better, but slower)
    const int ngs_coarse = 10;   // Number of NGS sweeps on coarser grids (more is better, but slower)
    const int ngs_first = 10;    // Number of NGS sweeps the very first time (if it fails try to increase this)

    //======================================================================
    // The analytical solution and corresponding source this corresponds to
    //======================================================================
    auto sol_analytic = [=](std::array<double, Ndim> & x) -> double {
        double sol = 1.0;
        for (int idim = 0; idim < Ndim; idim++)
            sol *= sin(4 * 2.0 * M_PI * x[idim]);
        return sol;
    };
    auto source_analytic = [&](std::array<double, Ndim> & x) -> double {
        return (-4.0 * M_PI * M_PI) * 16 * Ndim * sol_analytic(x);
    };

    //======================================================================
    // Set up the solver
    //======================================================================
    MultiGridSolver<Ndim, SolverType> g(N, Nlevels, verbose, periodic, nleft, nright);
    
    // Show info about the solver
    g.info();

    // Set options (optional; for finer control)
    g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
    g.set_epsilon(epsilon);
    g.set_maxsteps(max_steps);

    // Set the initial guess (if we use a mask then the value on boundary cells
    // we provide here will act as boundary conditions)
    g.set_initial_guess(SolverType(0.0));

    // Set it from a function
    // std::function<SolverType(std::array<double,Ndim>&)> func = [&](std::array<double,Ndim> &x){
    //   ...
    // };
    // g.set_initial_guess( func );

    //======================================================================
    // Implement the equation to be solved on the form L(f) = 0
    //======================================================================
    MultiGridFunction<Ndim, SolverType> Equation =
        [&](MultiGridSolver<Ndim, SolverType> * sol, int level, IndexInt index) {
            //======================================================================
            // Fetch index to all 2NDIM+1 neighboring cells (including own cell at 0)
            //======================================================================
            const auto index_list = sol->get_neighbor_gridindex(level, index);

            //======================================================================
            // Gridspacing h (we assume a square box so only fetch dx)
            //======================================================================
            const auto h = sol->get_Gridspacing(level);

            //======================================================================
            // Compute the laplacian [D^2 f]
            //======================================================================
            const auto laplacian = sol->get_Laplacian(level, index_list);

            //======================================================================
            // The position of the current cell in the grid
            //======================================================================
            auto coordinate = sol->get_Coordinate(level, index);

            //======================================================================
            // Set the right hand side of D^2f = source
            //======================================================================
            auto source = source_analytic(coordinate);

            //======================================================================
            // The full equation L(f) = 0 i.e. D^2f - source = 0
            //======================================================================
            const auto l = laplacian - source;

            //======================================================================
            // The derivtive dL/df
            //======================================================================
            const auto dl = SolverType(-2.0 * Ndim / (h * h));

            //======================================================================
            // Returns the equation and the derivative to the solver
            //======================================================================
            return std::pair<SolverType, SolverType>{l, dl};
        };

    //======================================================================
    // Implement a convergence criterion (here we use a fiducial one
    // rms_residual < epsilon )
    //======================================================================
    std::function<bool(double, double, int)> ConvergenceCriterion =
        [&](double rms_residual, double rms_residual_ini, int step_number) {
            return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
        };

    //======================================================================
    // Solve the equation and fetch the solution
    //======================================================================
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    // Show info about the grid
    sol.info();

    //======================================================================
    // Fetch solution and compare to analytical solution
    //======================================================================
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto coord = sol.coord_from_index(index);
        auto pos = sol.get_pos(index);
        if (FML::ThisTask == 0) {
            if (index % N == 0 and index > 0)
                std::cout << "\n";
            double anal = sol_analytic(pos);
            for (int idim = 0; idim < Ndim; idim++)
                std::cout << coord[idim] << " ";
            std::cout << sol[index] << " " << anal << "\n";
        }
    }
}

int main() { TestMultiGridSolver(); }
