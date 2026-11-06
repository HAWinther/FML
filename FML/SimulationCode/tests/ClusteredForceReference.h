#ifndef SIMCODE_CLUSTERED_FORCE_REFERENCE_H
#define SIMCODE_CLUSTERED_FORCE_REFERENCE_H
// Test-only periodic Newtonian oracle. No production force-kernel calls.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <vector>
namespace SIMCODE::CLUSTER_REFERENCE {
    using Vector = std::array<double, 3>;
    struct Source {
        Vector x;
        double mass;
    };
    inline double norm(Vector x) { return std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); }
    inline Vector displacement(Vector x, Vector y) {
        for (int d = 0; d < 3; ++d) {
            x[d] -= y[d];
            x[d] -= std::round(x[d]);
        }
        return x;
    }
    // Enclosed mass of W(r)=8/(pi h^3) f(r/h), integrated with four-point
    // Gauss-Legendre quadrature (exact for each polynomial piece, degree <=5).
    inline double softened_coefficient(double r, double h) {
        if (r >= h)
            return 1 / (r * r * r);
        if (r == 0)
            return 32 / (3 * h * h * h);
        constexpr double nodes[]{-.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526};
        constexpr double weights[]{.3478548451374538, .6521451548625461, .6521451548625461, .3478548451374538};
        const double u = r / h;
        double enclosed = 0;
        for (int piece = 0; piece < 2; ++piece) {
            const double low = piece == 0 ? 0 : .5, high = std::min(u, piece == 0 ? .5 : 1.);
            if (high <= low)
                continue;
            for (int j = 0; j < 4; ++j) {
                const double q = (low + high) / 2 + (high - low) / 2 * nodes[j];
                const double shape = piece == 0 ? 1 - 6 * q * q + 6 * q * q * q : 2 * std::pow(1 - q, 3);
                enclosed += 32 * (high - low) / 2 * weights[j] * q * q * shape;
            }
        }
        return enclosed / (r * r * r);
    }
    struct Mode {
        Vector k;
        double real, imag;
    };
    class Ewald {
        const std::vector<Source> & sources;
        double alpha;
        int images;
        std::vector<Mode> modes;

      public:
        Ewald(const std::vector<Source> & s, double split, int nmax, int image_extent = 1)
            : sources(s), alpha(split), images(image_extent) {
            const double pi = std::acos(-1.);
            for (int nx = -nmax; nx <= nmax; ++nx)
                for (int ny = -nmax; ny <= nmax; ++ny)
                    for (int nz = -nmax; nz <= nmax; ++nz) {
                        // Half lattice; its conjugate contributes the same real force.
                        if (!(nx > 0 || (nx == 0 && ny > 0) || (nx == 0 && ny == 0 && nz > 0)))
                            continue;
                        Vector k{2 * pi * nx, 2 * pi * ny, 2 * pi * nz};
                        const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                        double real = 0, imag = 0;
                        for (const auto & p : sources) {
                            const double phase = k[0] * p.x[0] + k[1] * p.x[1] + k[2] * p.x[2];
                            real += p.mass * std::cos(phase);
                            imag -= p.mass * std::sin(phase);
                        }
                        const double factor = 8 * pi * std::exp(-k2 / (4 * alpha * alpha)) / k2;
                        modes.push_back({k, real * factor, imag * factor});
                    }
        }
        Vector force(size_t target, double h) const {
            const Vector x = sources[target].x;
            Vector result{};
            const double sqrtpi = std::sqrt(std::acos(-1.));
            for (size_t j = 0; j < sources.size(); ++j) {
                if (j == target)
                    continue; // Self images cancel in symmetric pairs.
                const auto dx = displacement(x, sources[j].x);
                for (int a = -images; a <= images; ++a)
                    for (int b = -images; b <= images; ++b)
                        for (int c = -images; c <= images; ++c) {
                            Vector r{dx[0] + a, dx[1] + b, dx[2] + c};
                            const double length = norm(r), q = alpha * length;
                            const double window = std::erfc(q) + 2 * q / sqrtpi * std::exp(-q * q);
                            double coefficient = window / (length * length * length);
                            if (a == 0 && b == 0 && c == 0 && length < h)
                                coefficient =
                                    softened_coefficient(length, h) + (window - 1) / (length * length * length);
                            for (int d = 0; d < 3; ++d)
                                result[d] -= sources[j].mass * coefficient * r[d];
                        }
            }
            for (const auto & m : modes) {
                const double phase = m.k[0] * x[0] + m.k[1] * x[1] + m.k[2] * x[2];
                const double amplitude = m.real * std::sin(phase) + m.imag * std::cos(phase);
                for (int d = 0; d < 3; ++d)
                    result[d] -= m.k[d] * amplitude;
            }
            return result;
        }
    };
} // namespace SIMCODE::CLUSTER_REFERENCE
#endif
