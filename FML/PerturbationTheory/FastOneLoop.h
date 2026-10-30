#ifndef FML_FAST_ONE_LOOP_HEADER
#define FML_FAST_ONE_LOOP_HEADER
#include <FML/FFTLog/FFTLog.h>
#include <FML/PerturbationTheory/OneLoop.h>
#include <climits>
#include <memory>
namespace FML::PERTURBATION {
    struct FastOptions {
        size_t padding = std::numeric_limits<size_t>::max(); // automatic: half an input length on each side
        // Fraction of the logarithmic input range at each end over which the spectrum is smoothly tapered to zero
        // for the P22 transforms (P13 always uses the untapered spectrum). The FFTLog transforms of the
        // k^{n+1.5} P(k) inputs (n = -2..2) of an abruptly truncated spectrum ring, and P22 is a cancellation
        // among them: for a LCDM spectrum on 1e-5..50 h/Mpc, untapered, P22 was wrong by factors 5 at k = 0.1 and
        // 1e3-1e9 below 0.02 h/Mpc; with 0.1 the error is < 1e-4 for k >= 0.04 and 0.1% at 0.02 h/Mpc. 0: no taper.
        double taper = .1;
        size_t reference_points = 3;
        double relative_tolerance = .02, absolute_tolerance = 0;
        DirectOptions reference;
    };
    /// A local serial workspace. FFTW planning/destruction must be serialized.
    class FastEdS {
        FML::SOLVERS::FFTLog::Workspace hankel;
        struct Plans {
            int n;
            fftw_complex *a = nullptr, *b = nullptr;
            fftw_plan forward = nullptr, reverse = nullptr;
            explicit Plans(int size) : n(size) {
                a = fftw_alloc_complex(n);
                b = fftw_alloc_complex(n);
                if (!a || !b) {
                    fftw_free(a);
                    fftw_free(b);
                    throw std::bad_alloc();
                }
                forward = fftw_plan_dft_1d(n, a, a, FFTW_FORWARD, FFTW_ESTIMATE);
                reverse = fftw_plan_dft_1d(n, b, b, FFTW_BACKWARD, FFTW_ESTIMATE);
                if (!forward || !reverse) {
                    if (forward)
                        fftw_destroy_plan(forward);
                    if (reverse)
                        fftw_destroy_plan(reverse);
                    fftw_free(a);
                    fftw_free(b);
                    throw std::runtime_error("One-loop FFT planning failed");
                }
            }
            ~Plans() {
                fftw_destroy_plan(forward);
                fftw_destroy_plan(reverse);
                fftw_free(a);
                fftw_free(b);
            }
        };
        std::unique_ptr<Plans> plans;
        Vector cached_kernel;
        double cached_step = 0;
        size_t cached_n = 0;
        size_t builds = 0;
        std::pair<Vector, Vector> xi(int ell, int exponent, const Vector & k, const Vector & pk) {
            FML::SOLVERS::FFTLog::CVector input(k.size());
            for (size_t i = 0; i < k.size(); ++i)
                input[i] = std::pow(k[i], exponent + 1.5) * pk[i];
            auto transformed = FML::SOLVERS::FFTLog::DiscreteHankelTransform(k, input, hankel, ell + .5, 0, 1, false);
            Vector values(k.size());
            for (size_t i = 0; i < k.size(); ++i)
                values[i] = detail::finite(transformed.second[i].real() *
                                           std::pow(2 * detail::pi * transformed.first[i], -1.5));
            return {std::move(transformed.first), std::move(values)};
        }
        Vector p13_fft(const Vector & k, const Vector & pk, double step) {
            const size_t n = k.size();
            size_t size = 1;
            while (size < 3 * n - 2) {
                detail::require(size <= size_t(INT_MAX) / 2, "One-loop convolution too large");
                size *= 2;
            }
            if (!plans || plans->n != int(size)) {
                plans = std::make_unique<Plans>(int(size));
                cached_n = 0;
                ++builds;
            }
            auto & p = *plans;
            if (cached_n != n || cached_step != step) {
                for (int i = 0; i < p.n; ++i)
                    p.a[i][0] = p.a[i][1] = 0;
                for (size_t i = 0; i < 2 * n - 1; ++i) {
                    double r = std::exp((double(n - 1) - i) * step);
                    p.a[i][0] = p13_kernel(r);
                }
                fftw_execute(p.forward);
                cached_kernel.resize(2 * size);
                for (size_t i = 0; i < size; ++i) {
                    cached_kernel[2 * i] = p.a[i][0];
                    cached_kernel[2 * i + 1] = p.a[i][1];
                }
                cached_n = n;
                cached_step = step;
            }
            for (int i = 0; i < p.n; ++i)
                p.a[i][0] = p.a[i][1] = 0;
            for (size_t i = 0; i < n; ++i)
                p.a[i][0] = k[i] * pk[i] * (i == 0 || i + 1 == n ? .5 : 1);
            fftw_execute(p.forward);
            for (size_t i = 0; i < size; ++i) {
                p.b[i][0] = p.a[i][0] * cached_kernel[2 * i] - p.a[i][1] * cached_kernel[2 * i + 1];
                p.b[i][1] = p.a[i][0] * cached_kernel[2 * i + 1] + p.a[i][1] * cached_kernel[2 * i];
            }
            fftw_execute(p.reverse);
            Vector out(n);
            for (size_t i = 0; i < n; ++i)
                out[i] = detail::finite(p.b[i + n - 1][0] / size * step * k[i] * k[i] * pk[i] /
                                        (1008 * detail::pi * detail::pi));
            return out;
        }

      public:
        size_t convolution_plan_builds() const { return builds; }
        size_t fft_buffer_bytes() const {
            return hankel.buffer_bytes() + (plans ? size_t(plans->n) * 2 * sizeof(fftw_complex) : 0);
        }
        /// Uniform log-k input, zero outside its finite support. Padding adds zeros,
        /// not a power-law extrapolation. Trust only an interior range checked by convergence.
      private:
        Prediction raw(const Vector & k, const Vector & pk, size_t padding, double taper) {
            const size_t n = k.size();
            detail::require(n >= 64 && n == pk.size() && n <= size_t(INT_MAX) / 4 &&
                                padding <= (size_t(INT_MAX) - n) / 2,
                            "Invalid fast spectrum size/padding");
            const double step = std::log(k.back() / k.front()) / (n - 1);
            detail::require(k.front() > 0 && std::isfinite(step) && step > 0, "Invalid fast logarithmic grid");
            for (size_t i = 0; i < n; ++i) {
                detail::require(k[i] > 0 && (i == 0 || k[i] > k[i - 1]) && std::isfinite(k[i]) && pk[i] >= 0 &&
                                    std::isfinite(pk[i]),
                                "Invalid fast spectrum values");
                detail::require(std::abs(std::log(k[i] / k.front()) - i * step) < 1e-8 * std::max(1., i * step),
                                "Fast one-loop requires uniform log spacing");
            }
            const size_t length = n + 2 * padding;
            Vector padded_k(length), padded_p(length);
            for (size_t i = 0; i < length; ++i) {
                padded_k[i] = k.front() * std::exp((double(i) - padding) * step);
                detail::require(padded_k[i] > 0 && std::isfinite(padded_k[i]), "Padded logarithmic grid overflows");
                if (i >= padding && i < padding + n)
                    padded_p[i] = pk[i - padding];
            }
            // Taper both ends (window x - sin(2 pi x)/(2 pi) of FAST-PT, McEwen et al. 2016) for the P22 transforms
            if (taper > 0) {
                const double range = std::log(k.back() / k.front());
                for (size_t i = padding; i < padding + n; ++i) {
                    const double x = std::log(padded_k[i] / k.front()) / range;
                    for (double t : {x / taper, (1 - x) / taper})
                        if (t < 1)
                            padded_p[i] *= t - std::sin(2 * detail::pi * t) / (2 * detail::pi);
                }
            }
            // All multipoles use kcrc=1 without per-order no-ringing shifts, so products share exactly one r grid.
            auto a = xi(0, 0, padded_k, padded_p);
            const Vector r = std::move(a.first);
            Vector correlation(length);
            for (size_t i = 0; i < length; ++i)
                correlation[i] = 1219. / 735 * a.second[i] * a.second[i];
            Vector().swap(a.second);
            auto product = [&](int ell, int left, int right, double coefficient) {
                auto x = xi(ell, left, padded_k, padded_p);
                if (left == right) {
                    for (size_t i = 0; i < length; ++i)
                        correlation[i] += coefficient * x.second[i] * x.second[i];
                } else {
                    auto y = xi(ell, right, padded_k, padded_p);
                    for (size_t i = 0; i < length; ++i)
                        correlation[i] += coefficient * x.second[i] * y.second[i];
                }
            };
            product(0, -2, 2, 1. / 3);
            product(1, -1, 1, -124. / 35);
            product(2, 0, 0, 1342. / 1029);
            product(2, -2, 2, 2. / 3);
            product(3, -1, 1, -16. / 35);
            product(4, 0, 0, 64. / 1715);
            auto back = xi(0, 0, r, correlation);
            Prediction out;
            out.k = k;
            out.linear = pk;
            out.p13 = p13_fft(k, pk, step);
            out.method = "FFTLog P22 / linear FFT P13";
            out.kernels = "EdS";
            for (size_t i = 0; i < n; ++i) {
                out.p22.push_back(detail::finite(back.second[i + padding] * 8 * detail::pi * detail::pi * detail::pi));
                out.loop.push_back(out.p22.back() + out.p13[i]);
                out.total.push_back(detail::finite(pk[i] + out.loop.back()));
            }
            return out;
        }

      public:
        /// Return only requested input-grid modes. Mandatory direct spot checks
        /// fail explicitly on inaccurate tails; they are not an error bound between checks.
        Prediction
        evaluate(const Vector & k, const Vector & pk, double kmin, double kmax, const FastOptions & options = {}) {
            detail::require(k.size() >= 64 && k.size() == pk.size() && kmin > 0 && kmax > kmin && kmin >= k.front() &&
                                kmax <= k.back(),
                            "Invalid fast output range");
            detail::require(options.reference_points >= 3 && options.relative_tolerance > 0 &&
                                std::isfinite(options.relative_tolerance) && options.absolute_tolerance >= 0 &&
                                std::isfinite(options.absolute_tolerance),
                            "Fast EdS requires at least three finite direct-reference checks");
            size_t padding = options.padding == std::numeric_limits<size_t>::max() ? k.size() / 2 : options.padding;
            detail::require(options.taper >= 0 && options.taper < .5, "Fast EdS taper must be in [0, 0.5)");
            auto full = raw(k, pk, padding, options.taper);
            Prediction out;
            out.method = full.method;
            out.kernels = full.kernels;
            for (size_t i = 0; i < k.size(); ++i)
                if (k[i] >= kmin && k[i] <= kmax) {
                    out.k.push_back(k[i]);
                    out.linear.push_back(pk[i]);
                    out.p22.push_back(full.p22[i]);
                    out.p13.push_back(full.p13[i]);
                    out.loop.push_back(full.loop[i]);
                    out.total.push_back(full.total[i]);
                }
            detail::require(out.k.size() >= 3, "Fast output range must contain at least three grid modes");
            auto interpolated = [&](double q) {
                if (q < k.front() || q > k.back())
                    return 0.;
                auto hi = std::lower_bound(k.begin(), k.end(), q);
                if (hi == k.begin())
                    return pk.front();
                if (hi == k.end())
                    return pk.back();
                size_t j = size_t(hi - k.begin());
                double fraction = std::log(q / k[j - 1]) / std::log(k[j] / k[j - 1]);
                return std::max(0., pk[j - 1] + fraction * (pk[j] - pk[j - 1]));
            };
            const size_t checks = std::min(out.k.size(), options.reference_points);
            std::vector<size_t> indices;
            for (size_t i = 0; i < checks; ++i) {
                size_t j = i * (out.k.size() - 1) / (checks - 1);
                indices.push_back(j);
                out.validation_k.push_back(out.k[j]);
            }
            auto reference = direct_one_loop(out.validation_k, interpolated, k.front(), k.back(), options.reference);
            out.reference_p22 = reference.p22;
            out.reference_p13 = reference.p13;
            for (size_t i = 0; i < checks; ++i)
                for (const auto & pair : {std::make_pair(out.p22[indices[i]], reference.p22[i]),
                                          std::make_pair(out.p13[indices[i]], reference.p13[i])}) {
                    double error = std::abs(pair.first - pair.second), scale = std::abs(pair.second);
                    out.max_reference_relative_error = std::max(out.max_reference_relative_error,
                                                                scale > 0  ? error / scale :
                                                                error == 0 ? 0 :
                                                                             std::numeric_limits<double>::max());
                    if (error > options.absolute_tolerance + options.relative_tolerance * scale)
                        throw std::runtime_error("Fast EdS failed direct-reference checks: adjust "
                                                 "support/resolution/padding or use direct_one_loop");
                }
            return out;
        }
    };
} // namespace FML::PERTURBATION
#endif
