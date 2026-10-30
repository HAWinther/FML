#ifndef FML_ONE_LOOP_HEADER
#define FML_ONE_LOOP_HEADER
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace FML::PERTURBATION {
    using Vector = std::vector<double>;
    using Spectrum = std::function<double(double)>;
    struct Prediction {
        Vector k, linear, p22, p13, loop, total;
        std::string method, kernels;
        Vector validation_k, reference_p22, reference_p13;
        double max_reference_relative_error = 0;
    };
    struct DirectOptions {
        int order = 12;
        double log_step = .15;
    };
    struct Kernels {
        std::function<double(double, double, double)> f2;           // F2(q,p,cos(q,p)), fully symmetrized.
        std::function<double(double, double, double)> f3_collapsed; // F3(k,q,-q; cos(k,q)), fully symmetrized.
        std::string name = "EdS";
    };
    namespace detail {
        constexpr double pi = 3.141592653589793238462643383279502884;
        inline void require(bool ok, const char * msg) {
            if (!ok)
                throw std::invalid_argument(msg);
        }
        inline double finite(double value) {
            if (!std::isfinite(value))
                throw std::overflow_error("Nonfinite one-loop result/integrand");
            return value;
        }
        inline std::pair<Vector, Vector> gauss(int n) {
            require(n >= 2 && n <= 64, "Gauss order must be in [2,64]");
            Vector x(n), w(n);
            for (int i = 0; i < (n + 1) / 2; ++i) {
                double z = std::cos(pi * (i + .75) / (n + .5)), derivative = 0;
                for (int it = 0; it < 100; ++it) {
                    double p = 1, prev = 0;
                    for (int j = 1; j <= n; ++j) {
                        double old = p;
                        p = ((2 * j - 1) * z * p - (j - 1) * prev) / j;
                        prev = old;
                    }
                    derivative = n * (z * p - prev) / (z * z - 1);
                    double next = z - p / derivative;
                    if (std::abs(next - z) < 4e-16) {
                        z = next;
                        break;
                    }
                    z = next;
                }
                x[i] = -z;
                x[n - 1 - i] = z;
                w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * derivative * derivative);
            }
            return {x, w};
        }
        template <class Function>
        double integrate_log(double low,
                             double high,
                             const Function & f,
                             const DirectOptions & options,
                             const std::pair<Vector, Vector> & gl) {
            if (high <= low)
                return 0;
            double a = std::log(low), b = std::log(high);
            double segments = std::ceil((b - a) / options.log_step);
            require(segments <= 1000000, "Excessive direct integration range");
            int count = std::max(1, int(segments));
            long double sum = 0;
            for (int s = 0; s < count; ++s) {
                double lo = a + (b - a) * s / count, hi = a + (b - a) * (s + 1) / count;
                for (size_t j = 0; j < gl.first.size(); ++j) {
                    double q = std::exp((lo + hi) / 2 + (hi - lo) / 2 * gl.first[j]);
                    sum += gl.second[j] * (hi - lo) / 2 * q * finite(f(q));
                }
            }
            return finite(double(sum));
        }
    } // namespace detail
    inline double eds_f2(double q, double p, double mu) {
        return 5. / 7 + .5 * mu * (q / p + p / q) + 2. / 7 * mu * mu;
    }
    /// Stable angle-integrated EdS P13 kernel K(r)=r^2 Z(r). K(1)=-88.
    inline double p13_kernel(double r) {
        detail::require(r > 0 && std::isfinite(r), "P13 ratio must be positive and finite");
        if (r < .05) {
            double t = r * r;
            return -168 + t * (928. / 5 + t * (-4512. / 35 + t * (416. / 21 + t * (2656. / 1155 + t * 3232. / 5005))));
        }
        if (r > 20) {
            double t = 1 / r / r;
            return -488. / 5 +
                   t * (96. / 5 + t * (-160. / 21 + t * (-1376. / 1155 + t * (-1952. / 5005 - t * 2528. / 15015))));
        }
        if (r == 1)
            return -88;
        const long double x = r, t = x * x, logarithm = std::log1p((x < 1 ? 2 * x : 2) / (std::abs(x - 1)));
        return double(12 / t - 158 + 100 * t - 42 * t * t +
                      3 / (x * t) * (t - 1) * (t - 1) * (t - 1) * (7 * t + 2) * logarithm);
    }
    /// Local direct quadrature. P is explicitly zero outside [qmin,qmax].
    /// Custom kernels are never routed through the EdS FFT approximation.
    inline Prediction direct_one_loop(const Vector & output,
                                      const Spectrum & power,
                                      double qmin,
                                      double qmax,
                                      const DirectOptions & options = {},
                                      const Kernels & kernels = {}) {
        detail::require(bool(power) && qmin > 0 && qmax > qmin && std::isfinite(qmax) && std::isfinite(qmax / qmin) &&
                            options.log_step > 0 && std::isfinite(options.log_step),
                        "Invalid direct spectrum/range/options");
        detail::require((!kernels.f2 && !kernels.f3_collapsed && kernels.name == "EdS") ||
                            (kernels.f2 && kernels.f3_collapsed && !kernels.name.empty() && kernels.name != "EdS"),
                        "Custom predictions require both symmetrized kernels and an explicit name");
        for (double k : output)
            detail::require(k > 0 && std::isfinite(k) && std::isfinite(qmax / k) && qmin / k > 0,
                            "Invalid output wavenumber/range");
        auto gl = detail::gauss(options.order);
        auto P = [&](double k) {
            if (k < qmin || k > qmax)
                return 0.;
            double value = power(k);
            detail::require(value >= 0 && std::isfinite(value), "Power must be finite and nonnegative");
            return value;
        };
        Prediction out;
        out.k = output;
        out.method = "direct Gauss quadrature";
        out.kernels = kernels.name;
        for (double k : output) {
            const double pk = P(k);
            std::vector<double> breaks{qmin, qmax};
            for (double x : {k, k - qmin, k + qmin, qmax - k, k - qmax})
                if (x > qmin && x < qmax)
                    breaks.push_back(x);
            std::sort(breaks.begin(), breaks.end());
            breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
            long double p22 = 0, p13 = 0;
            for (size_t b = 1; b < breaks.size(); ++b) {
                p22 += detail::integrate_log(
                    breaks[b - 1],
                    breaks[b],
                    [&](double q) {
                        double lower = std::max(qmin, std::abs(k - q)), upper = std::min(qmax, k + q);
                        if (lower >= upper)
                            return 0.;
                        return q * P(q) *
                               detail::integrate_log(
                                   lower,
                                   upper,
                                   [&](double p) {
                                       double mu = std::clamp((k * k - q * q - p * p) / (2 * q * p), -1., 1.);
                                       double f = kernels.f2 ? kernels.f2(q, p, mu) : eds_f2(q, p, mu);
                                       return p * P(p) * f * f;
                                   },
                                   options,
                                   gl);
                    },
                    options,
                    gl);
                p13 += detail::integrate_log(
                    breaks[b - 1],
                    breaks[b],
                    [&](double q) {
                        if (!kernels.f3_collapsed)
                            return P(q) * p13_kernel(q / k);
                        long double angular = 0;
                        for (size_t j = 0; j < gl.first.size(); ++j)
                            angular += gl.second[j] * detail::finite(kernels.f3_collapsed(k, q, gl.first[j]));
                        return double(angular) * q * q * P(q);
                    },
                    options,
                    gl);
            }
            out.linear.push_back(pk);
            out.p22.push_back(detail::finite(double(p22) / (2 * detail::pi * detail::pi * k)));
            out.p13.push_back(detail::finite(double(p13) * pk *
                                             (kernels.f3_collapsed ? 3. / (2 * detail::pi * detail::pi) :
                                                                     k * k / (1008 * detail::pi * detail::pi))));
            out.loop.push_back(out.p22.back() + out.p13.back());
            out.total.push_back(detail::finite(pk + out.loop.back()));
        }
        return out;
    }
} // namespace FML::PERTURBATION
#endif
