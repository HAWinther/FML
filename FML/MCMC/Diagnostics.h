#ifndef FML_MCMC_DIAGNOSTICS_HEADER
#define FML_MCMC_DIAGNOSTICS_HEADER
// Rank-normalized split/folded Rhat and bulk/tail ESS (Vehtari et al. 2021).
// Standalone C++17; see README.md for input, degeneracy and stopping contracts.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace FML::MCMC {
    enum class DiagnosticStatus {
        Ok,
        InsufficientChains,
        InsufficientDraws,
        NonFinite,
        ConstantChain,
        NumericalFailure
    };
    inline const char * diagnostic_status_name(DiagnosticStatus status) {
        switch (status) {
            case DiagnosticStatus::Ok:
                return "ok";
            case DiagnosticStatus::InsufficientChains:
                return "insufficient_chains";
            case DiagnosticStatus::InsufficientDraws:
                return "insufficient_draws";
            case DiagnosticStatus::NonFinite:
                return "nonfinite_draw";
            case DiagnosticStatus::ConstantChain:
                return "constant_chain";
            default:
                return "numerical_failure";
        }
    }
    struct ChainDiagnostics {
        DiagnosticStatus status = DiagnosticStatus::InsufficientChains;
        std::size_t chains = 0, draws_per_chain = 0, draws_used = 0;
        double rhat = std::numeric_limits<double>::quiet_NaN();
        double rhat_bulk = std::numeric_limits<double>::quiet_NaN();
        double rhat_folded = std::numeric_limits<double>::quiet_NaN();
        double ess_bulk = std::numeric_limits<double>::quiet_NaN();
        double ess_tail = std::numeric_limits<double>::quiet_NaN();
        double ess_lower = std::numeric_limits<double>::quiet_NaN();
        double ess_upper = std::numeric_limits<double>::quiet_NaN();
        bool valid() const { return status == DiagnosticStatus::Ok; }
    };

    namespace diagnostic_detail {
        using Chains = std::vector<std::vector<double>>;
        // Invert the normal CDF by bisection of erfc. Ranking is O(S log S), so
        // these fixed 64 iterations do not change its asymptotic complexity.
        inline double normal_quantile(double p) {
            double lo = -40, hi = 40;
            for (int i = 0; i < 64; ++i) {
                const double mid = (lo + hi) / 2;
                if (.5 * std::erfc(-mid / std::sqrt(2.)) < p)
                    lo = mid;
                else
                    hi = mid;
            }
            return (lo + hi) / 2;
        }
        inline std::vector<double> ranks(const std::vector<long double> & values) {
            std::vector<std::size_t> order(values.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](auto a, auto b) { return values[a] < values[b]; });
            std::vector<double> result(values.size());
            for (std::size_t first = 0; first < order.size();) {
                std::size_t end = first + 1;
                while (end < order.size() && values[order[end]] == values[order[first]])
                    ++end;
                const double rank = (double(first) + double(end) + 1) / 2;
                const double z = normal_quantile((rank - .375) / (double(values.size()) + .25));
                for (auto j = first; j < end; ++j)
                    result[order[j]] = z;
                first = end;
            }
            return result;
        }
        inline Chains split(const Chains & input) {
            const auto half = input.front().size() / 2;
            Chains result;
            result.reserve(2 * input.size());
            for (const auto & chain : input)
                result.emplace_back(chain.begin(), chain.begin() + half);
            for (const auto & chain : input)
                result.emplace_back(chain.end() - half, chain.end());
            return result;
        }
        inline Chains ranked(const Chains & input, bool fold = false) {
            std::vector<long double> values;
            for (const auto & c : input)
                values.insert(values.end(), c.begin(), c.end());
            if (fold) {
                auto sorted = values;
                std::sort(sorted.begin(), sorted.end());
                const auto n = sorted.size();
                const bool scale = std::max(std::abs(sorted.front()), std::abs(sorted.back())) >
                                   std::numeric_limits<double>::max() / 4;
                const double factor = scale ? .5 : 1.;
                const double right = double(sorted[n / 2]) * factor;
                const double left = double(sorted[(n - 1) / 2]) * factor;
                const double median = n % 2 ? right : (left + right) / 2;
                // Use double median/deviations to match reference tie handling;
                // common scaling prevents overflow near the limits of double.
                for (auto & v : values)
                    v = std::abs(double(v) * factor - median);
            }
            const auto scores = ranks(values);
            Chains output(input.size(), std::vector<double>(input.front().size()));
            std::size_t i = 0;
            for (auto & c : output)
                for (auto & v : c)
                    v = scores[i++];
            return output;
        }
        struct Variances {
            double within, between_means, pooled;
        };
        inline Variances variances(const Chains & input) {
            const double n = double(input.front().size()), m = double(input.size());
            std::vector<double> means;
            double within = 0;
            for (const auto & c : input) {
                const bool constant = std::all_of(c.begin(), c.end(), [&](double x) { return x == c.front(); });
                const double mean = constant ? c.front() : std::accumulate(c.begin(), c.end(), 0.) / n;
                means.push_back(mean);
                for (double v : c)
                    within += (v - mean) * (v - mean) / ((n - 1) * m);
            }
            const bool constant_means =
                std::all_of(means.begin(), means.end(), [&](double x) { return x == means.front(); });
            const double mean = constant_means ? means.front() : std::accumulate(means.begin(), means.end(), 0.) / m;
            double between = 0;
            for (double v : means)
                between += (v - mean) * (v - mean) / (m - 1);
            return {within, between, (n - 1) / n * within + between};
        }
        inline double rhat(const Chains & input, bool constant_fold = false) {
            const auto v = variances(input);
            if (v.within == 0) {
                if (v.between_means > 0)
                    return std::numeric_limits<double>::infinity();
                return constant_fold ? 1. : std::numeric_limits<double>::quiet_NaN();
            }
            return std::sqrt(v.pooled / v.within);
        }
        // Local radix-2 FFT: diagnostics do not require FFTW or FML setup.
        inline void fft(std::vector<std::complex<double>> & a, bool inverse) {
            const auto n = a.size();
            for (std::size_t i = 1, j = 0; i < n; ++i) {
                auto bit = n >> 1;
                for (; j & bit; bit >>= 1)
                    j ^= bit;
                j ^= bit;
                if (i < j)
                    std::swap(a[i], a[j]);
            }
            for (std::size_t len = 2; len <= n;) {
                const auto step = std::polar(1., (inverse ? 2 : -2) * std::acos(-1.) / double(len));
                for (std::size_t first = 0; first < n; first += len) {
                    std::complex<double> w(1);
                    for (std::size_t j = 0; j < len / 2; ++j) {
                        const auto u = a[first + j], v = a[first + j + len / 2] * w;
                        a[first + j] = u + v;
                        a[first + j + len / 2] = u - v;
                        w *= step;
                    }
                }
                if (len == n)
                    break;
                len *= 2;
            }
            if (inverse)
                for (auto & v : a)
                    v /= double(n);
        }
        inline std::vector<double> autocovariance(const std::vector<double> & chain) {
            const auto n = chain.size();
            if (n > std::numeric_limits<std::size_t>::max() / 2)
                throw std::length_error("diagnostics FFT too large");
            std::size_t size = 1;
            while (size < 2 * n) {
                if (size > std::numeric_limits<std::size_t>::max() / 2)
                    throw std::length_error("diagnostics FFT too large");
                size *= 2;
            }
            const double mean = std::accumulate(chain.begin(), chain.end(), 0.) / double(n);
            std::vector<std::complex<double>> scratch(size);
            for (std::size_t i = 0; i < n; ++i)
                scratch[i] = chain[i] - mean;
            fft(scratch, false);
            for (auto & v : scratch)
                v = std::norm(v);
            fft(scratch, true);
            std::vector<double> result(n);
            for (std::size_t i = 0; i < n; ++i)
                result[i] = scratch[i].real() / double(n);
            return result;
        }
        inline double ess(const Chains & input) {
            const auto n = input.front().size();
            const auto v = variances(input);
            if (!(v.pooled > 0))
                return std::numeric_limits<double>::quiet_NaN();
            std::vector<double> cov(n, 0.);
            for (const auto & c : input) {
                const auto ac = autocovariance(c);
                for (std::size_t i = 0; i < n; ++i)
                    cov[i] += ac[i] / double(input.size());
            }
            const auto rho = [&](std::size_t lag) { return 1 - (v.within - cov[lag]) / v.pooled; };
            std::vector<double> correlations(n, 0.);
            correlations[0] = 1;
            correlations[1] = rho(1);
            double even = 1, odd = correlations[1];
            std::ptrdiff_t last = 1;
            // Initial positive paired sequence, then enforce monotonicity.
            while (last < std::ptrdiff_t(n) - 3 && even + odd > 0) {
                even = rho(std::size_t(last + 1));
                odd = rho(std::size_t(last + 2));
                if (even + odd >= 0) {
                    correlations[std::size_t(last + 1)] = even;
                    correlations[std::size_t(last + 2)] = odd;
                }
                last += 2;
            }
            last -= 2;
            if (even > 0)
                correlations[std::size_t(last + 1)] = even;
            for (std::ptrdiff_t j = 1; j <= last - 2; j += 2) {
                const double previous = correlations[std::size_t(j - 1)] + correlations[std::size_t(j)];
                if (correlations[std::size_t(j + 1)] + correlations[std::size_t(j + 2)] > previous)
                    correlations[std::size_t(j + 1)] = correlations[std::size_t(j + 2)] = previous / 2;
            }
            double tau = -1 + correlations[std::size_t(last + 1)];
            for (std::ptrdiff_t j = 0; j <= last; ++j)
                tau += 2 * correlations[std::size_t(j)];
            const double total = double(input.size()) * double(n);
            // Anticorrelated chains can legitimately have ESS > draw count.
            return total / std::max(tau, 1 / std::log10(total));
        }
        inline double quantile(std::vector<double> values, double p) {
            std::sort(values.begin(), values.end());
            const double position = p * double(values.size() - 1);
            const auto lower = std::size_t(position);
            const double fraction = position - double(lower);
            return (1 - fraction) * values[lower] + fraction * values[std::min(lower + 1, values.size() - 1)];
        }
    } // namespace diagnostic_detail

    /// Equal-length, unweighted post-warmup scalar chains, including rejected-step repeats.
    /// At least two chains of eight draws. Odd lengths discard each central draw on splitting.
    /// Shape errors throw; insufficient/constant/nonfinite data return a status and NaN metrics.
    inline ChainDiagnostics diagnose_chains(const std::vector<std::vector<double>> & input) {
        using namespace diagnostic_detail;
        ChainDiagnostics result;
        result.chains = input.size();
        if (input.empty())
            return result;
        result.draws_per_chain = input.front().size();
        for (const auto & c : input)
            if (c.size() != result.draws_per_chain)
                throw std::invalid_argument("diagnostics require equal chain lengths");
        if (input.size() < 2)
            return result;
        if (result.draws_per_chain < 8) {
            result.status = DiagnosticStatus::InsufficientDraws;
            return result;
        }
        bool constant = false;
        std::vector<double> pooled;
        for (const auto & c : input) {
            for (double v : c) {
                if (!std::isfinite(v)) {
                    result.status = DiagnosticStatus::NonFinite;
                    return result;
                }
                pooled.push_back(v);
            }
            constant = constant || std::all_of(c.begin(), c.end(), [&](double v) { return v == c.front(); });
        }
        if (constant) {
            result.status = DiagnosticStatus::ConstantChain;
            return result;
        }
        const auto halves = split(input);
        result.draws_used = input.size() * 2 * (result.draws_per_chain / 2);
        const auto scores = ranked(halves);
        result.rhat_bulk = rhat(scores);
        result.rhat_folded = rhat(ranked(halves, true), true);
        result.rhat = std::max(result.rhat_bulk, result.rhat_folded);
        result.ess_bulk = ess(scores);
        auto indicators = input;
        const auto tail = [&](double p) {
            const double q = quantile(pooled, p);
            for (std::size_t i = 0; i < input.size(); ++i)
                for (std::size_t j = 0; j < input[i].size(); ++j)
                    indicators[i][j] = input[i][j] <= q ? 1. : 0.;
            return ess(split(indicators));
        };
        result.ess_lower = tail(.05);
        result.ess_upper = tail(.95);
        if (std::isfinite(result.ess_lower) && std::isfinite(result.ess_upper))
            result.ess_tail = std::min(result.ess_lower, result.ess_upper);
        result.status = std::isfinite(result.rhat) && std::isfinite(result.ess_bulk) &&
                                std::isfinite(result.ess_lower) && std::isfinite(result.ess_upper) ?
                            DiagnosticStatus::Ok :
                            DiagnosticStatus::NumericalFailure;
        return result;
    }
} // namespace FML::MCMC
#endif
