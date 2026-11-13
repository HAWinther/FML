#ifndef FML_PERTURBATION_CORRELATION_MULTIPOLES_H
#define FML_PERTURBATION_CORRELATION_MULTIPOLES_H
#include <FML/FFTLog/SphericalBesselTransform.h>
#include <algorithm>
#include <array>
#include <limits>

// FFTLog Hankel transforms: Hamilton (2000), arXiv:astro-ph/9905191.
// FML finite-band, taper and unit conventions: CORRELATION_MULTIPOLES.md.
namespace FML::PERTURBATION {
    struct CorrelationOptions {
        int fft_samples = 4096;
        // Fraction of ln(kmax/kmin) smoothly tapered at EACH end. Zero disables tapering.
        double taper_fraction = 0.1;
        // Explicit exp[-(k R)^2] regulator. R has separation units; zero disables it.
        double damping_radius = 0;
    };
    struct CorrelationMultipoles {
        std::vector<double> s;
        std::array<std::vector<double>, 3> xi; // ell=0,2,4 including i^ell: +,-,+.
    };
    /// Local, reusable finite-band transform:
    /// xi_l(s) = i^l/(2 pi^2) int_{kmin}^{kmax} dk k^2 P_l(k) W(k) j_l(ks).
    /// Signed P is interpolated linearly versus ln k; no extrapolation. One object per concurrent caller.
    /// Construction/destruction creates/destroys FFTW plans and must be serialized with other FFTW planning.
    class CorrelationWorkspace {
      public:
        using Vector = std::vector<double>;
        CorrelationWorkspace(const Vector & k, const Vector & separations, CorrelationOptions options = {})
            : input_k(k), s(separations), opt(options) {
            if (k.size() < 4 || s.empty() || opt.fft_samples < 64 || opt.fft_samples > (1 << 20) ||
                opt.fft_samples % 2 || !std::isfinite(opt.taper_fraction) || opt.taper_fraction < 0 ||
                opt.taper_fraction > .5 || !std::isfinite(opt.damping_radius) || opt.damping_radius < 0)
                throw std::invalid_argument("Correlation multipoles: invalid grid/options");
            auto increasing_positive = [](const Vector & v) {
                for (size_t i = 0; i < v.size(); ++i)
                    if (!std::isfinite(v[i]) || !(v[i] > 0) || (i && !(v[i] > v[i - 1])))
                        return false;
                return true;
            };
            if (!increasing_positive(k) || !increasing_positive(s))
                throw std::invalid_argument("Correlation multipoles: k and s must be positive, finite and increasing");
            Vector logk(k.size());
            for (size_t i = 0; i < k.size(); ++i)
                logk[i] = std::log(k[i]);
            for (size_t i = 1; i < logk.size(); ++i)
                if (!(logk[i] > logk[i - 1]))
                    throw std::invalid_argument("Correlation multipoles: k knots indistinguishable in log space");
            const double span = logk.back() - logk.front();
            if (!(span > 0) || !std::isfinite(span))
                throw std::invalid_argument("Correlation multipoles: degenerate logarithmic k support");
            Vector grid(opt.fft_samples);
            window.resize(grid.size());
            for (size_t i = 0; i < grid.size(); ++i) {
                const double t = double(i) / (grid.size() - 1);
                const double logq = (1 - t) * logk.front() + t * logk.back();
                grid[i] = std::exp(logq);
                if (i == 0)
                    grid[i] = k.front();
                if (i + 1 == grid.size())
                    grid[i] = k.back();
                to_fft.push_back(interpolation(logk, logq));
                double taper = 1;
                if (opt.taper_fraction > 0) {
                    const double edge = std::min(t, 1 - t) / opt.taper_fraction;
                    if (edge < 1)
                        taper = edge - std::sin(2 * pi * edge) / (2 * pi);
                }
                const double kr = grid[i] * opt.damping_radius;
                window[i] = taper * std::exp(-kr * kr);
            }
            // The underlying transform requires representable log spacing and reciprocal grids.
            if (!increasing_positive(grid) || !std::isfinite(grid.back() / grid.front()) ||
                !std::isfinite(grid.front() * grid.back()) || grid.front() * grid.back() == 0)
                throw std::invalid_argument("Correlation multipoles: unrepresentable FFT grid");
            transform = SOLVERS::FFTLog::SphericalBesselTransform(grid, 5, true, true, true);
            for (int il = 0; il < 3; ++il) {
                const auto & native = transform.y(2 * il);
                if (!increasing_positive(native))
                    throw std::invalid_argument("Correlation multipoles: unrepresentable reciprocal grid");
                Vector logs(native.size());
                for (size_t i = 0; i < native.size(); ++i)
                    logs[i] = std::log(native[i]);
                for (size_t i = 1; i < logs.size(); ++i)
                    if (!(logs[i] > logs[i - 1]))
                        throw std::invalid_argument(
                            "Correlation multipoles: reciprocal grid indistinguishable in log space");
                for (double value : s) {
                    if (value < native.front() || value > native.back())
                        throw std::invalid_argument(
                            "Correlation multipoles: separation outside the shared FFT output range");
                    from_fft[il].push_back(interpolation(logs, std::log(value)));
                }
            }
            fft_values.resize(grid.size());
            fft_result.resize(grid.size());
        }
        CorrelationWorkspace(const CorrelationWorkspace &) = delete;
        CorrelationWorkspace & operator=(const CorrelationWorkspace &) = delete;
        CorrelationWorkspace(CorrelationWorkspace &&) noexcept = default;
        CorrelationWorkspace & operator=(CorrelationWorkspace &&) noexcept = default;

        const Vector & k() const { return input_k; }
        const Vector & separations() const { return s; }
        const CorrelationOptions & options() const { return opt; }
        size_t evaluations() const { return completed; }

        CorrelationMultipoles evaluate(const Vector & p0, const Vector & p2, const Vector & p4) {
            return evaluate_fields({&p0, &p2, &p4});
        }
        CorrelationMultipoles evaluate(const std::array<Vector, 3> & power) {
            return evaluate(power[0], power[1], power[2]);
        }
        /// Accepts the k/p0/p2/p4 result structs of LPTRedshiftSpace and EPTRedshiftSpace.
        template <class PowerMultipoles>
        CorrelationMultipoles evaluate_spectrum(const PowerMultipoles & power) {
            if (power.k != input_k)
                throw std::invalid_argument("Correlation multipoles: result k grid differs from workspace");
            return evaluate(power.p0, power.p2, power.p4);
        }

      private:
        static constexpr double pi = 3.141592653589793238462643383279502884;
        struct Interpolation {
            size_t left;
            double weight;
        };
        Vector input_k, s, window, fft_values, fft_result;
        CorrelationOptions opt;
        SOLVERS::FFTLog::SphericalBesselTransform transform;
        std::vector<Interpolation> to_fft;
        std::array<std::vector<Interpolation>, 3> from_fft;
        size_t completed = 0;
        static Interpolation interpolation(const Vector & x, double value) {
            const size_t right =
                std::clamp(size_t(std::upper_bound(x.begin(), x.end(), value) - x.begin()), size_t(1), x.size() - 1);
            return {right - 1, std::clamp((value - x[right - 1]) / (x[right] - x[right - 1]), 0., 1.)};
        }
        CorrelationMultipoles evaluate_fields(const std::array<const Vector *, 3> & power) {
            if (input_k.empty() || fft_values.empty())
                throw std::logic_error("Correlation multipoles: workspace was moved from");
            for (const auto * p : power) {
                if (p->size() != input_k.size())
                    throw std::invalid_argument("Correlation multipoles: power and k lengths differ");
                for (double value : *p)
                    if (!std::isfinite(value))
                        throw std::invalid_argument("Correlation multipoles: nonfinite power");
            }
            CorrelationMultipoles result;
            result.s = s;
            for (int il = 0; il < 3; ++il) {
                const auto & p = *power[il];
                for (size_t i = 0; i < to_fft.size(); ++i) {
                    const auto map = to_fft[i];
                    fft_values[i] = ((1 - map.weight) * p[map.left] + map.weight * p[map.left + 1]) * window[i];
                }
                transform.transform(2 * il, fft_values.data(), fft_result.data());
                auto & values = result.xi[il];
                values.resize(s.size());
                for (size_t i = 0; i < s.size(); ++i) {
                    const auto map = from_fft[il][i];
                    values[i] = (il == 1 ? -1 : 1) *
                                ((1 - map.weight) * fft_result[map.left] + map.weight * fft_result[map.left + 1]);
                    if (!std::isfinite(values[i]))
                        throw std::overflow_error("Correlation multipoles: nonfinite transform result");
                }
            }
            ++completed;
            return result;
        }
    };
} // namespace FML::PERTURBATION
#endif
