#include <FML/Plotting/Matplotlibcpp.h>
#include <vector>

namespace plt = FML::UTILS::Matplotlib;
using DVector = std::vector<double>;

int main() {

    //========================================================
    // Example of how to use matplotlib to make plots from C++
    // With MPI only call this from the first task
    //========================================================

    const char * filename = "plot.png";

    // A test function to plot
    auto function = [](double x) -> double { return 2.0 + sin(2.0 * M_PI * x); };

    // Make an x-array
    const int npts = 100;
    const double xmin = 0.0;
    const double xmax = 1.0;
    DVector x_array(npts);
    for (int i = 0; i < npts; i++)
        x_array[i] = xmin + (xmax - xmin) * i / double(npts);

    // Fill the y-array
    DVector y_array(npts);
    for (int i = 0; i < npts; i++)
        y_array[i] = function(x_array[i]);

    // Call matplotlib
    plt::xlim(0.0, 1.0);
    plt::plot(x_array, y_array);
    plt::title("$y = 2 + \\sin(2 \\pi x)$");

    // Save plot to file
    plt::save(filename);

    // Show figure
    // plt::show();
}
