#ifndef FML_MATH_GRIDS_H
#define FML_MATH_GRIDS_H
#include <cmath>
#include <stdexcept>
#include <vector>

namespace FML { namespace MATH {
    /// Linear grid with inclusive endpoints; num <= 0 gives empty, num = 1 gives {xmin}.
    inline std::vector<double> linspace(double xmin, double xmax, int num) {
        if (num <= 0)
            return std::vector<double>();
        if (num == 1)
            return std::vector<double>{xmin};
        std::vector<double> res(num);
        double delta_x = (xmax - xmin) / double(num - 1);
        for (int i = 0; i < num - 1; i++) {
            res[i] = xmin + delta_x * i;
        }
        // Just to make sure its exactly xmax at the endpoint
        res[num - 1] = xmax;
        return res;
    }

    /// Grid between base^xmin and base^xmax (exponent endpoints).
    inline std::vector<double> logspace(double xmin, double xmax, int num, double base = 10.0) {
        if (base <= 0.0) throw std::runtime_error("logspace::Cannot have base <= 0.0\n");
        std::vector<double> res = linspace(xmin, xmax, num);
        for (int i = 0; i < num; i++) {
            res[i] = std::pow(base, res[i]);
        }
        return res;
    }

    /// Logarithmic spacing between positive physical endpoints (not exponents).
    /// Both endpoints are included; requires num >= 2.
    inline std::vector<double> geomspace(double xmin, double xmax, int num) {
        if (num < 2) throw std::invalid_argument("geomspace: need num >= 2");
        if (xmin <= 0.0 || xmax <= 0.0)
            throw std::invalid_argument("geomspace: limits must be > 0");
        auto x = linspace(std::log(xmin), std::log(xmax), num);
        for (auto & xi : x) xi = std::exp(xi);
        x.front() = xmin;
        x.back() = xmax;
        return x;
    }
} }
#endif
