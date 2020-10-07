#include <FML/ODESolver/ODESolver.h>
#include <cmath>

// using namespace FML::ODESOLVER;
using DVector = FML::SOLVERS::ODESOLVER::DVector;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;

int main() {

    //=========================================
    // Example on how to solve coupled ODEs
    //=========================================

    // Make an x-array to store the solution on
    const int npts = 10;
    const double xmin = 0.0;
    const double xmax = 2.0 * M_PI;
    DVector x_array(npts);
    for (int i = 0; i < npts; i++)
        x_array[i] = xmin + (xmax - xmin) * i / double(npts);

    // The RHS of the ODE y'' + y = 0
    // We write it as y0' = y1 ; y1' = -y0
    ODEFunction deriv = [&](double x, const double * y, double * dydx) {
        (void)x;
        dydx[0] = y[1];
        dydx[1] = -y[0];
        return GSL_SUCCESS;
    };

    // The initial conditions
    DVector y_initial{1.0, 0.0};

    // The anaytical solution
    auto function = [](double x) -> double { return std::cos(x); };

    // Solve the ODE from xmin = x[0] till xmax = x[npts-1] and store the
    // result at every x in x_array
    ODESolver ode;
    ode.solve(deriv, x_array, y_initial);

    // Fetch solution (y0 = y)
    auto y_array = ode.get_data_by_component(0);

    // Output
    std::cout << "# x     y       y_true      error\n";
    for (int i = 0; i < npts; i++) {
        double x = x_array[i];
        double y = y_array[i];
        double ytrue = function(x);
        std::cout << x << " " << y << " " << ytrue << " Error: " << y - ytrue << "\n";
    }
}
