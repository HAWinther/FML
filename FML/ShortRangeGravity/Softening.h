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
        const double u2 = u * u, u4 = u2 * u2;
        if (u < .5)
            return (14.0 / 5 - 16.0 / 3 * u2 + 48.0 / 5 * u4 - 32.0 / 5 * u4 * u) / h;
        return (16.0 / 5 - 1 / (15 * u) - 32.0 / 3 * u2 + 16 * u2 * u - 48.0 / 5 * u4 + 32.0 / 15 * u4 * u) / h;
    }
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
