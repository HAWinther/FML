#ifndef FML_TEST_CORRELATION_REFERENCE_H
#define FML_TEST_CORRELATION_REFERENCE_H
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
namespace correlation_reference {
    using Vector = std::vector<double>;
    constexpr long double pi = 3.141592653589793238462643383279502884L;
    inline long double bessel(int ell, long double x) {
        if (std::abs(x) < .7L) {
            long double term = 1;
            for (int j = 1; j <= ell; ++j)
                term *= x / (2 * j + 1);
            long double sum = term;
            for (int n = 1; n < 40; ++n) {
                term *= -x * x / (2 * n * (2 * ell + 2 * n + 1));
                sum += term;
                if (std::abs(term) < 1e-25L * std::max(1.L, std::abs(sum)))
                    break;
            }
            return sum;
        }
        long double a = std::sin(x) / x;
        if (!ell)
            return a;
        long double b = (a - std::cos(x)) / x;
        for (int l = 1; l < ell; ++l) {
            long double c = (2 * l + 1) * b / x - a;
            a = b;
            b = c;
        }
        return b;
    }
    // Independent composite Simpson integral, splitting at EVERY input interpolation knot.
    // No FFT, Mellin coefficients, cached output grid or production interpolation helper.
    inline std::array<Vector, 3> direct(const Vector & k,
                                        const std::array<Vector, 3> & power,
                                        const Vector & s,
                                        double taper,
                                        double damping,
                                        double phase_step = .1) {
        std::array<Vector, 3> result;
        for (auto & row : result)
            row.assign(s.size(), 0);
        const long double lo = std::log((long double)k.front()), span = std::log((long double)k.back()) - lo;
        std::vector<std::array<long double, 4>> nodes;
        for (size_t i = 1; i < k.size(); ++i) {
            const long double a = std::log((long double)k[i - 1]), b = std::log((long double)k[i]);
            int steps = std::max(
                2,
                int(std::ceil(std::max<long double>((b - a) / .004L, (k[i] - k[i - 1]) * s.back() / phase_step))));
            if (steps % 2)
                ++steps;
            for (int j = 0; j <= steps; ++j) {
                const long double t = (long double)j / steps, logq = (1 - t) * a + t * b, q = std::exp(logq);
                long double window = std::exp(-q * q * damping * damping);
                if (taper > 0) {
                    const long double edge = std::min(logq - lo, lo + span - logq) / (span * taper);
                    if (edge < 1)
                        window *= edge - std::sin(2 * pi * edge) / (2 * pi);
                }
                const long double weight = (b - a) / steps / 3 *
                                           (j == 0 || j == steps ? 1 :
                                            j % 2                ? 4 :
                                                                   2) *
                                           q * q * q * window / (2 * pi * pi);
                std::array<long double, 4> node{q, 0, 0, 0};
                for (int il = 0; il < 3; ++il)
                    node[il + 1] = weight * ((1 - t) * power[il][i - 1] + t * power[il][i]);
                nodes.push_back(node);
            }
        }
        for (int il = 0; il < 3; ++il)
            for (size_t i = 0; i < s.size(); ++i) {
                long double sum = 0;
                for (const auto & n : nodes)
                    sum += n[il + 1] * bessel(2 * il, n[0] * s[i]);
                result[il][i] = double((il == 1 ? -1 : 1) * sum);
            }
        return result;
    }
} // namespace correlation_reference
#endif
