#ifndef FML_SHORT_RANGE_GRAVITY_TESTS_PAIRWISE_EWALD_REFERENCE_H
#define FML_SHORT_RANGE_GRAVITY_TESTS_PAIRWISE_EWALD_REFERENCE_H
#include <array>
#include <cmath>
// Test-only pairwise periodic Ewald force of a unit point mass (box 1, inverse split 6, images -2..2, modes -12..12).
// Independent of the production kernels. Used by the SimulationCode short-range tests and the clustered-force benchmark.
namespace FML::SHORT_RANGE_GRAVITY::TESTS::PAIRWISE {
    using Vector = std::array<double, 3>;
    inline double length(Vector x) { return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); }
    inline Vector ewald(Vector x) {
        Vector result{};
        constexpr double alpha = 6;
        for (int a = -2; a <= 2; ++a)
            for (int b = -2; b <= 2; ++b)
                for (int c = -2; c <= 2; ++c) {
                    Vector r{x[0] + a, x[1] + b, x[2] + c};
                    double v = length(r);
                    if (v == 0)
                        continue;
                    double factor =
                        (std::erfc(alpha * v) + 2 * alpha * v / std::sqrt(M_PI) * std::exp(-alpha * alpha * v * v)) /
                        (v * v * v);
                    for (int d = 0; d < 3; ++d)
                        result[d] -= factor * r[d];
                }
        for (int a = -12; a <= 12; ++a)
            for (int b = -12; b <= 12; ++b)
                for (int c = -12; c <= 12; ++c) {
                    double n2 = a * a + b * b + c * c;
                    if (n2 == 0)
                        continue;
                    Vector k{2 * M_PI * a, 2 * M_PI * b, 2 * M_PI * c};
                    double phase = k[0] * x[0] + k[1] * x[1] + k[2] * x[2];
                    double factor = -std::exp(-M_PI * M_PI * n2 / (alpha * alpha)) / (M_PI * n2) * std::sin(phase);
                    for (int d = 0; d < 3; ++d)
                        result[d] += factor * k[d];
                }
        return result;
    }
} // namespace FML::SHORT_RANGE_GRAVITY::TESTS::PAIRWISE
#endif
