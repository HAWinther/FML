#ifndef FML_FIELDLEVEL_TEST_GRADIENT_CONVERGENCE_H
#define FML_FIELDLEVEL_TEST_GRADIENT_CONVERGENCE_H
// Test-only step sweep. Require a resolved interval, not one fortuitous step.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
namespace FML::FIELDLEVEL::TESTING {
    struct DerivativeSweep {
        std::vector<double> errors;
        double best = INFINITY;
        bool resolved() const {
            bool adjacent = false;
            for (size_t i = 1; i < errors.size(); ++i)
                adjacent |= errors[i - 1] < 2e-6 && errors[i] < 2e-6;
            return adjacent && best < 1e-6 && (errors.front() < 1e-8 || best < .2 * errors.front());
        }
    };
    template <class Function>
    DerivativeSweep derivative_sweep(Function value, double analytic, double first_step, const std::string & label) {
        DerivativeSweep result;
        std::cout << "DERIVATIVE " << label << " analytic=" << analytic << '\n';
        for (int i = 0; i < 10; ++i) {
            const double h = first_step * std::pow(.25, i);
            const double numeric = (value(h) - value(-h)) / (2 * h);
            const double error = std::abs(analytic - numeric) / std::max({1., std::abs(analytic), std::abs(numeric)});
            result.errors.push_back(error);
            result.best = std::min(result.best, error);
            std::cout << "  h=" << h << " numeric=" << numeric << " relative_error=" << error << '\n';
        }
        return result;
    }
} // namespace FML::FIELDLEVEL::TESTING
#endif
