#ifndef FML_WAVE_TIME_FACTORS_H
#define FML_WAVE_TIME_FACTORS_H
#include <FML/WaveDarkMatter/ComplexGrid.h>
#include <FML/Units/Units.h>
#include <array>
#include <cmath>
#include <functional>
namespace FML::WAVEDM {
    struct Physics {
        double eta{1}; // hbar/m in chosen code length^2/time.
        double G{1};   // Chosen code length^3/(mass*time^2).
        static Physics from_si(double mass_ev, double length_m, double time_s, double mass_kg) {
            const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
            if (!positive(mass_ev) || !positive(length_m) || !positive(time_s) || !positive(mass_kg))
                throw std::invalid_argument("[WaveDM] Invalid axion mass or unit scales");
            using namespace FML::UTILS::SI;
            Physics p{hbar * c * c / (mass_ev * eV) * time_s / length_m / length_m,
                      FML::UTILS::SI::G * mass_kg / length_m * (time_s / length_m) * (time_s / length_m)};
            if (!positive(p.eta) || !positive(p.G))
                throw std::invalid_argument("[WaveDM] Unit conversion overflow/underflow");
            return p;
        }
    };
    struct TimeFactors {
        double dt{}, kinetic{}, potential{}, a_begin{1}, a_end{1}; // Integral dt, dt/a^2, dt/a.
        static TimeFactors fixed(double dt, double a = 1) { return {dt, dt / a / a, dt / a, a, a}; }
    };
    // Independent 16-point Gauss-Legendre panels, sampling H once per node for all three integrals.
    inline TimeFactors
    expansion_factors(double a0, double a1, const std::function<double(double)> & H, int panels = 4) {
        detail::same(a0, "[WaveDM] Initial scale factor differs between ranks");
        detail::same(a1, "[WaveDM] Final scale factor differs between ranks");
        detail::same(panels, "[WaveDM] Background panel count differs between ranks");
        detail::require(a0 > 0 && a1 > 0 && std::isfinite(a0) && std::isfinite(a1) && bool(H) && panels > 0,
                        "[WaveDM] Invalid background integration interval");
        constexpr std::array<double, 8> nodes{.09501250983763744,
                                              .2816035507792589,
                                              .4580167776572274,
                                              .6178762444026438,
                                              .7554044083550030,
                                              .8656312023878318,
                                              .9445750230732326,
                                              .9894009349916499};
        constexpr std::array<double, 8> weights{.1894506104550685,
                                                .1826034150449236,
                                                .1691565193950025,
                                                .1495959888165767,
                                                .1246289712555339,
                                                .09515851168249278,
                                                .06225352393864789,
                                                .02715245941175409};
        std::array<long double, 3> sum{};
        bool good = true;
        try {
            for (int panel = 0; panel < panels; ++panel) {
                const double lo = a0 + (a1 - a0) * (double(panel) / panels),
                             hi = a0 + (a1 - a0) * (double(panel + 1) / panels), half = (hi - lo) * .5, mid = lo + half;
                for (size_t i = 0; i < nodes.size(); ++i)
                    for (double sign : {-1., 1.}) {
                        const double a = mid + sign * half * nodes[i], h = H(a);
                        good &= std::isfinite(h) && h > 0;
                        const long double w = static_cast<long double>(half) * weights[i] / a / h;
                        sum[0] += w;
                        sum[1] += w / a / a;
                        sum[2] += w / a;
                    }
            }
        } catch (const std::exception &) {
            good = false;
        }
        for (auto x : sum)
            good &= std::isfinite(double(x));
        detail::require(good, "[WaveDM] Invalid background H(a) or time factors");
        TimeFactors result{double(sum[0]), double(sum[1]), double(sum[2]), a0, a1};
        for (double x : {result.dt, result.kinetic, result.potential})
            detail::same(x, "[WaveDM] Background factors differ between ranks");
        return result;
    }
    struct EnergyBalance {
        bool initialized{false};
        double initial{}, previous_a{}, previous_integrand{}, integral{};
        // For an isolated wave system in comoving coordinates: d(K+W)/dt + H(2K+W)=0.
        // Integrates in log(a) with a trapezoid; convergence of this diagnostic is itself second order.
        double observe(double a, double K, double W) {
            if (!(a > 0) || !std::isfinite(a) || !std::isfinite(K) || !std::isfinite(W))
                throw std::invalid_argument("[WaveDM] Invalid energy observation");
            if (!initialized) {
                initialized = true;
                initial = K + W;
                previous_a = a;
                previous_integrand = 2 * K + W;
                return 0;
            }
            integral += (previous_integrand + 2 * K + W) * .5 * std::log(a / previous_a);
            previous_a = a;
            previous_integrand = 2 * K + W;
            return K + W - initial + integral;
        }
    };
} // namespace FML::WAVEDM
#endif
