#ifndef FML_SPH_KERNEL_HEADER
#define FML_SPH_KERNEL_HEADER
#include <array>
#include <cmath>
namespace FML::SPH {
    template <int D>
    using Vector = std::array<double, D>;
    /// Cubic B-spline: h is the smoothing scale; compact support is 2h.
    template <int D>
    struct CubicSpline {
        static_assert(D >= 1 && D <= 3);
        static double normalization() {
            return D == 1 ? 2. / 3 : D == 2 ? 10. / (7 * std::acos(-1.)) : 1 / std::acos(-1.);
        }
        struct Value {
            double w, dr, dh;
        };
        Value operator()(double r, double h) const {
            const double q = r / h;
            if (q >= 2)
                return {0, 0, 0};
            double f, df;
            if (q < 1) {
                f = 1 - 1.5 * q * q + .75 * q * q * q;
                df = -3 * q + 2.25 * q * q;
            } else {
                const double t = 2 - q;
                f = .25 * t * t * t;
                df = -.75 * t * t;
            }
            const double scale = normalization() / std::pow(h, D);
            return {scale * f, scale * df / h, -scale * (D * f + q * df) / h};
        }
    };
} // namespace FML::SPH
#endif
