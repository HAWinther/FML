#include <cmath>
#include <iostream>
#include <random>

#include <FML/Spline/Spline.h>

// using namespace FML::SPLINE;
using Spline = FML::INTERPOLATION::SPLINE::Spline;
using DVector = FML::INTERPOLATION::SPLINE::DVector;

int main() {
    std::mt19937 generator;
    auto udist = std::uniform_real_distribution<double>(0.0, 1.0);

    //======================================
    // Make a spline and use it
    //======================================

    // A test function to spline
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

    // Make a spline
    Spline y_spline(x_array, y_array, "Name of spline for better error messages");

    // Output the error at random points close to the ones we sampled
    for (int i = 0; i < npts; i++) {
        const double x = x_array[i] + udist(generator) / double(npts);
        const double y = y_spline(x);
        const double ytrue = function(x);
        std::cout << x << " " << y << " " << ytrue << " Error %: " << abs(y - ytrue) / abs(ytrue) * 100. << "\n";
    }
}
