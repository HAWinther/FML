#ifndef FML_NEWTONIAN_SOFTENING_HEADER
#define FML_NEWTONIAN_SOFTENING_HEADER
#include <cmath>
#include <stdexcept>
namespace FML::SHORT_RANGE_GRAVITY {
    /// Positive Newtonian spline Green function (1/r beyond support h), before source mass and coupling.
    inline double newtonian_spline_green(double r, double h) {
        if (!std::isfinite(h) || h <= 0)
            throw std::invalid_argument("Invalid gravitational softening support");
        if (!std::isfinite(r) || r < 0)
            throw std::invalid_argument("[ShortRangeGravity] Separation must be finite and nonnegative");
        const double u = r / h;
        if (u >= 1)
            return 1 / r;
        if (u < .5)
            return (14.0 / 5 - 16.0 / 3 * u * u + 48.0 / 5 * std::pow(u, 4) - 32.0 / 5 * std::pow(u, 5)) / h;
        return (16.0 / 5 - 1 / (15 * u) - 32.0 / 3 * u * u + 16 * u * u * u - 48.0 / 5 * std::pow(u, 4) +
                32.0 / 15 * std::pow(u, 5)) /
               h;
    }
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
