#ifndef FML_SHORT_RANGE_YUKAWA_HEADER
#define FML_SHORT_RANGE_YUKAWA_HEADER
#include <array>
#include <cmath>
#include <limits>

namespace FML::SHORT_RANGE_GRAVITY {
    struct Newtonian {};
    /// Phi=-Gm/r [1+alpha exp(-r/lambda)] outside the softening support. lambda shares the coordinate unit.
    struct NewtonYukawa {
        double alpha{0};
        double lambda{1};
    };
    namespace detail {
        /// Scaled complementary error function for x>=0, without exp(x*x) overflow.
        inline double erfcx_positive(double x) {
            const double inverse_sqrt_pi = 1 / std::sqrt(std::acos(-1.0));
            if (x < 26)
                return std::exp(x * x) * std::erfc(x);
            double term = 1, sum = 1;
            for (int n = 1; n < 20; ++n) {
                term *= -(2 * n - 1) * .5 / x / x;
                sum += term;
                if (std::abs(term) < std::numeric_limits<double>::epsilon() * std::abs(sum))
                    break;
            }
            return inverse_sqrt_pi / x * sum;
        }
        /// log((1+x) exp(-x)); the small-x series retains the quadratic correction.
        inline double log_yukawa_force_window(double x) {
            if (x < .01) {
                double term = -.5 * x * x, sum = term;
                for (int n = 3; n <= 12; ++n) {
                    term *= -x * (n - 1) / n;
                    sum += term;
                }
                return sum;
            }
            return std::isinf(x) ? -std::numeric_limits<double>::infinity() : std::log1p(x) - x;
        }
        inline double yukawa_force_window(double x) { return std::exp(log_yukawa_force_window(x)); }
        inline double exponential_remainder(double x) {
            if (std::abs(x) < .1) {
                double term = .5 * x * x, sum = term;
                for (int n = 3; n <= 16; ++n) {
                    term *= x / n;
                    sum += term;
                }
                return sum;
            }
            return std::expm1(x) - x;
        }
        inline double combined_yukawa_factor(double alpha, double exponent) {
            return exponent > -.5 ? (1 + alpha) + alpha * std::expm1(exponent) : 1 + alpha * std::exp(exponent);
        }
        /// Analytic Ewald short-range Yukawa Green function and radial force coefficient, before mass/coupling.
        /// q=r/(2rs), z=rs/lambda. Only used at positive r, outside the softening support.
        inline std::array<double, 2> yukawa_short(double r, double rs, double lambda) {
            const double q = r / (2 * rs), z = rs / lambda, t = r / lambda;
            if (std::isinf(t))
                return {0, 0};
            // exp(+t) erfc(q+z) = exp(-q*q-z*z) erfcx(q+z).
            const double common = std::exp(-q * q - z * z);
            const double a = q >= z ? common * erfcx_positive(q - z) : std::exp(-t) * std::erfc(q - z);
            const double b = common * erfcx_positive(q + z);
            const double green = .5 * (a + b) / r;
            const double coefficient =
                .5 * (a + b + t * (a - b) + 4 * q / std::sqrt(std::acos(-1.0)) * common) / (r * r * r);
            return {green, coefficient};
        }
        /// Gauss-Legendre rule for positive heat-kernel integrals; initialized once, never during an OMP force loop.
        struct YukawaQuadrature {
            std::array<double, 48> node{}, weight{};
            YukawaQuadrature() {
                const double pi = std::acos(-1.0);
                constexpr int n = 48;
                for (int i = 0; i < n / 2; ++i) {
                    double x = std::cos(pi * (i + .75) / (n + .5)), derivative = 0;
                    for (int iteration = 0; iteration < 30; ++iteration) {
                        double p = 1, previous = 0;
                        for (int j = 1; j <= n; ++j) {
                            const double next = ((2 * j - 1) * x * p - (j - 1) * previous) / j;
                            previous = p;
                            p = next;
                        }
                        derivative = n * (x * p - previous) / (x * x - 1);
                        const double change = p / derivative;
                        x -= change;
                        if (std::abs(change) < 2 * std::numeric_limits<double>::epsilon())
                            break;
                    }
                    const double w = 1 / ((1 - x * x) * derivative * derivative);
                    node[i] = .5 * (1 - x);
                    node[n - 1 - i] = .5 * (1 + x);
                    weight[i] = weight[n - 1 - i] = w;
                }
            }
        };
        inline const YukawaQuadrature & yukawa_quadrature() {
            static const YukawaQuadrature rule;
            return rule;
        }
        /// Positive integral representation avoids subtracting singular full/short forces near r=0.
        /// Returns Gaussian-Ewald long-range Green function and radial force coefficient (without coupling).
        inline std::array<double, 2> yukawa_long(double r, double rs, double lambda) {
            const auto & rule = yukawa_quadrature();
            const double q = r / (2 * rs), z = rs / lambda;
            // Here the long-range part is at least an order-unity fraction of the full force,
            // so full-minus-short is well conditioned and avoids quadrature of narrow interior peaks.
            if (q >= 1 && q >= z) {
                const auto short_part = yukawa_short(r, rs, lambda);
                const double t = r / lambda;
                return {std::exp(-t) / r - short_part[0], yukawa_force_window(t) / (r * r * r) - short_part[1]};
            }
            // Resolve the narrow v~z endpoint analytically when z is small. A fixed rule would miss
            // the O(z) potential offset and eventually the O(z^3) force contribution.
            if (z < .5) {
                if (q >= 1) {
                    const auto short_part = yukawa_short(r, rs, lambda);
                    const double t = r / lambda;
                    return {std::exp(-t) / r - short_part[0], yukawa_force_window(t) / (r * r * r) - short_part[1]};
                }
                const double pi = std::acos(-1.0), e = std::exp(-z * z);
                double moment = e - std::sqrt(pi) * z * std::erfc(z);
                double term = 1, potential = 0, force = 0;
                for (int n = 0; n < 32; ++n) {
                    const double next = (e - 2 * z * z * moment) / (2 * n + 3);
                    potential += term * moment;
                    force += term * next;
                    term *= -q * q / (n + 1);
                    moment = next;
                }
                return {potential / (std::sqrt(pi) * rs), force / (2 * std::sqrt(pi) * rs * rs * rs)};
            }
            double potential = 0, force = 0;
            for (std::size_t i = 0; i < rule.node.size(); ++i) {
                const double v = rule.node[i], t = z / v;
                const double integrand = rule.weight[i] * std::exp(-q * q * v * v - t * t);
                potential += integrand;
                force += v * v * integrand;
            }
            const double inverse_sqrt_pi = 1 / std::sqrt(std::acos(-1.0));
            return {potential * inverse_sqrt_pi / rs, force * inverse_sqrt_pi / (2 * rs * rs * rs)};
        }
        /// Newtonian minus Yukawa Ewald short part. Positive integral/series retains accuracy when lambda is large.
        inline std::array<double, 2> yukawa_short_difference(double r, double rs, double lambda) {
            const double q = r / (2 * rs), z = rs / lambda, pi = std::acos(-1.0);
            if (z < .5) {
                double potential = 0, force = 0;
                if (q < 1) {
                    double moment = std::sqrt(pi) / (2 * q) * std::erfc(q);
                    double term = z * z;
                    for (int n = 1; n <= 18; ++n) {
                        const double next = (std::exp(-q * q) - 2 * q * q * moment) / (2 * n - 1);
                        force += term * moment;
                        potential += term * next;
                        term *= -z * z / (n + 1);
                        moment = next;
                    }
                } else {
                    const auto & rule = yukawa_quadrature();
                    for (std::size_t i = 0; i < rule.node.size(); ++i) {
                        const double u = rule.node[i], u2 = u * u;
                        const double f = rule.weight[i] * std::exp(-q * q / u2) * (-std::expm1(-z * z * u2));
                        potential += f / u2;
                        force += f / (u2 * u2);
                    }
                }
                return {potential / (std::sqrt(pi) * rs), force / (2 * std::sqrt(pi) * rs * rs * rs)};
            }
            const auto y = yukawa_short(r, rs, lambda);
            const double n_p = std::erfc(q) / r;
            const double n_f = (std::erfc(q) + 2 * q / std::sqrt(pi) * std::exp(-q * q)) / (r * r * r);
            if (q >= 1)
                return {n_p - y[0], n_f - y[1]};
            // At small q, full-minus-long is well conditioned for this difference.
            const auto yl = yukawa_long(r, rs, lambda);
            const double nl_p = std::erf(q) / r;
            double nl_f;
            if (q < .05) {
                const double t = q * q;
                nl_f = (1.0 / 6 + t * (-1.0 / 10 + t * (1.0 / 28 + t * (-1.0 / 108 + t / 528)))) /
                       (std::sqrt(pi) * rs * rs * rs);
            } else
                nl_f = (std::erf(q) - 2 * q / std::sqrt(pi) * std::exp(-q * q)) / (r * r * r);
            const double t = r / lambda;
            return {-std::expm1(-t) / r - (nl_p - yl[0]),
                    -std::expm1(log_yukawa_force_window(t)) / (r * r * r) - (nl_f - yl[1])};
        }
        /// Newtonian minus Yukawa Ewald long part, without cancellation of nearly equal component forces.
        inline std::array<double, 2>
        yukawa_long_difference(double r, double rs, double lambda, bool subtract_mu = false) {
            const double q = r / (2 * rs), z = rs / lambda, pi = std::acos(-1.0);
            if (z < .5) {
                if (q >= 1) {
                    const auto ds = yukawa_short_difference(r, rs, lambda);
                    const double t = r / lambda;
                    return {(subtract_mu ? -exponential_remainder(-t) : -std::expm1(-t)) / r - ds[0],
                            -std::expm1(log_yukawa_force_window(t)) / (r * r * r) - ds[1]};
                }
                const double one_minus_e = -std::expm1(-z * z);
                double moment = one_minus_e + std::sqrt(pi) * z * std::erfc(z);
                double term = 1, potential = 0, force = 0;
                for (int n = 0; n < 32; ++n) {
                    const double next = (one_minus_e + 2 * z * z * (1.0 / (2 * n + 1) - moment)) / (2 * n + 3);
                    potential +=
                        term * (subtract_mu && n == 0 ? one_minus_e - std::sqrt(pi) * z * std::erf(z) : moment);
                    force += term * next;
                    term *= -q * q / (n + 1);
                    moment = next;
                }
                return {potential / (std::sqrt(pi) * rs), force / (2 * std::sqrt(pi) * rs * rs * rs)};
            }
            const auto yl = yukawa_long(r, rs, lambda);
            const double np = q == 0 ? 1 / (std::sqrt(pi) * rs) : std::erf(q) / r;
            double nf;
            if (q < .05) {
                const double t = q * q;
                nf = (1.0 / 6 + t * (-1.0 / 10 + t * (1.0 / 28 + t * (-1.0 / 108 + t / 528)))) /
                     (std::sqrt(pi) * rs * rs * rs);
            } else
                nf = (std::erf(q) - 2 * q / std::sqrt(pi) * std::exp(-q * q)) / (r * r * r);
            return {np - yl[0] - (subtract_mu ? 1 / lambda : 0), nf - yl[1]};
        }
    } // namespace detail
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
