#ifndef FML_FIELD_LEVEL_FIXED_IC_H
#define FML_FIELD_LEVEL_FIXED_IC_H
#include <FML/FieldLevel/Framework.h>
#include <FML/Global/MPICounts.h>
#include <array>
#include <random>
#include <optional>

namespace FML::FIELDLEVEL::FRAMEWORK {
    namespace FIXED_IC_DETAIL {
        inline void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        inline void replicated(const std::vector<double> & values) {
            double n = double(values.size()), lo = n, hi = n;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, "[FixedIC] Replicated dimensions differ");
            for (double x : values) {
                require(std::isfinite(x), "[FixedIC] Nonfinite replicated control");
                lo = hi = x;
                FML::MinOverTasks(&lo);
                FML::MaxOverTasks(&hi);
                require(lo == hi, "[FixedIC] Controls differ across ranks");
            }
        }
        // Full-support shrinkage is used ONLY for convex scalar negative log densities.
        // The linear Gaussian likelihood plus a bounded uniform/Gaussian bias prior has connected slices.
        template <class Potential>
        double convex_slice(double current, double lower, double upper, Potential potential, std::mt19937_64 & rng) {
            std::uniform_real_distribution<double> uniform(0., 1.);
            double u = 0.;
            while (u == 0.)
                u = uniform(rng);
            const double initial = potential(current), level = initial - std::log(u);
            if (!std::isfinite(initial))
                throw std::runtime_error("[FixedIC] Nonfinite conditional slice start");
            for (int attempt = 0; attempt < 10000; ++attempt) {
                const double proposal = lower + (upper - lower) * uniform(rng);
                const double value = potential(proposal);
                if (std::isfinite(value) && value <= level)
                    return proposal;
                if (proposal < current)
                    lower = proposal;
                else
                    upper = proposal;
            }
            throw std::runtime_error("[FixedIC] Conditional slice exhausted its budget");
        }
    } // namespace FIXED_IC_DETAIL

    // Normalized, unbounded Gaussian proposal in PHYSICAL bias coordinates.
    // It is proposal geometry only; every finite-support/scalar prior remains in the MH target.
    template <size_t N>
    class FixedICGaussianProposal {
        static_assert(N > 0 && N <= 64, "Use a small bias proposal");
        std::array<double, N> mean_{};
        std::array<double, N * N> lower_{}; // Physical precision = lower * lower^T.
        double log_normalization_{};
        static auto cholesky(const std::array<double, N * N> & precision) {
            std::array<double, N * N> lower{};
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j) {
                    if (!std::isfinite(precision[i * N + j]) ||
                        std::abs(precision[i * N + j] - precision[j * N + i]) >
                            64 * N * std::numeric_limits<double>::epsilon() *
                                std::max({1., std::abs(precision[i * N + j]), std::abs(precision[j * N + i])}))
                        throw std::invalid_argument("[FixedIC] Nonfinite/asymmetric proposal precision");
                }
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j <= i; ++j) {
                    long double v = precision[i * N + j];
                    for (size_t k = 0; k < j; ++k)
                        v -= (long double)lower[i * N + k] * lower[j * N + k];
                    if (i == j) {
                        if (!(v > 0) || !std::isfinite(v))
                            throw UnrepresentableState("[FixedIC] Gaussian proposal is not positive definite");
                        lower[i * N + j] = std::sqrt(double(v));
                    } else
                        lower[i * N + j] = double(v) / lower[j * N + j];
                }
            return lower;
        }
        FixedICGaussianProposal(std::array<double, N> mean,
                                const std::array<double, N * N> & scaled_lower,
                                const std::array<double, N> & scales)
            : mean_(mean) {
            log_normalization_ = -.5 * N * std::log(2 * M_PI);
            for (size_t i = 0; i < N; ++i) {
                if (!std::isfinite(mean_[i]) || !(scales[i] > 0) || !std::isfinite(scales[i]))
                    throw UnrepresentableState("[FixedIC] Invalid Gaussian proposal scale/mean");
                for (size_t j = 0; j <= i; ++j) {
                    lower_[i * N + j] = scaled_lower[i * N + j] / scales[i];
                    if (!std::isfinite(lower_[i * N + j]))
                        throw UnrepresentableState("[FixedIC] Unrepresentable Gaussian proposal precision");
                }
                log_normalization_ += std::log(scaled_lower[i * N + i]) - std::log(scales[i]);
            }
        }

      public:
        static FixedICGaussianProposal from_precision(std::array<double, N> mean,
                                                      const std::array<double, N * N> & precision) {
            std::array<double, N> scales;
            scales.fill(1.);
            return {mean, cholesky(precision), scales};
        }
        static FixedICGaussianProposal from_scaled_quadratic(const std::array<double, N> & reference,
                                                             std::array<double, N> scaled_gradient,
                                                             const std::array<double, N * N> & scaled_precision,
                                                             const std::array<double, N> & scales) {
            const auto lower = cholesky(scaled_precision);
            // Solve H_s x = gradient_s, then mean_b = reference_b - S x.
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < i; ++j)
                    scaled_gradient[i] -= lower[i * N + j] * scaled_gradient[j];
                scaled_gradient[i] /= lower[i * N + i];
            }
            for (size_t i = N; i-- > 0;) {
                for (size_t j = i + 1; j < N; ++j)
                    scaled_gradient[i] -= lower[j * N + i] * scaled_gradient[j];
                scaled_gradient[i] /= lower[i * N + i];
            }
            auto mean = reference;
            for (size_t i = 0; i < N; ++i)
                mean[i] -= scales[i] * scaled_gradient[i];
            return {mean, lower, scales};
        }
        const auto & mean() const { return mean_; }
        double log_density(const std::array<double, N> & b) const {
            long double quadratic = 0;
            for (size_t i = 0; i < N; ++i) {
                long double v = 0;
                for (size_t j = i; j < N; ++j)
                    v += lower_[j * N + i] * ((long double)b[j] - mean_[j]);
                quadratic += v * v;
            }
            return log_normalization_ - .5 * double(quadratic);
        }
        std::array<double, N> draw(std::mt19937_64 & rng, std::normal_distribution<double> & normal) const {
            std::array<double, N> b;
            for (double & x : b)
                x = normal(rng);
            for (size_t i = N; i-- > 0;) {
                for (size_t j = i + 1; j < N; ++j)
                    b[i] -= lower_[j * N + i] * b[j];
                b[i] /= lower_[i * N + i];
            }
            for (size_t i = 0; i < N; ++i)
                b[i] += mean_[i];
            return b;
        }
    };

    // Exact sufficient statistics for a fixed-noise Fourier likelihood linear in N real bias coefficients.
    // All inputs are local half-complex modes; weights include Hermitian multiplicity.
    // Construction is collective; subsequent evaluation is replicated O(N^2), without an FFT or MPI reduction.
    template <size_t N>
    class FixedNoiseLinearBiasLikelihood {
        static_assert(N > 0 && N <= 64, "Use a small linear bias basis");
        std::array<double, N> reference_{}, gradient_{};
        std::array<double, N * N> gram_{};
        double value_{};

      public:
        FixedNoiseLinearBiasLikelihood(const std::array<Modes, N> & columns,
                                       const Modes & data,
                                       const std::vector<double> & weights,
                                       double variance,
                                       std::array<double, N> reference)
            : reference_(reference) {
            bool valid = data.size() == weights.size() && std::isfinite(variance) && variance > 0;
            for (const auto & c : columns)
                valid &= c.size() == data.size();
            for (double x : reference_)
                valid &= std::isfinite(x);
            for (double x : weights)
                valid &= std::isfinite(x) && x >= 0;
            for (auto x : data)
                valid &= std::isfinite(x.real()) && std::isfinite(x.imag());
            for (const auto & c : columns)
                for (auto x : c)
                    valid &= std::isfinite(x.real()) && std::isfinite(x.imag());
            FIXED_IC_DETAIL::require(valid, "[FixedIC] Invalid linear likelihood input");
            std::vector<double> controls(reference_.begin(), reference_.end());
            controls.push_back(variance);
            FIXED_IC_DETAIL::replicated(controls);
            std::array<long double, 1 + N + N * N> local{};
            for (size_t k = 0; k < data.size(); ++k) {
                Complex residual = -data[k];
                for (size_t a = 0; a < N; ++a)
                    residual += reference_[a] * columns[a][k];
                const long double w = weights[k], inverse = w / variance;
                local[0] += .5L * (inverse * std::norm(residual) + w * std::log(variance));
                for (size_t a = 0; a < N; ++a) {
                    local[1 + a] += inverse * std::real(std::conj(columns[a][k]) * residual);
                    for (size_t b = 0; b < N; ++b)
                        local[1 + N + a * N + b] += inverse * std::real(std::conj(columns[a][k]) * columns[b][k]);
                }
            }
            std::array<double, 1 + N + N * N> reduced{};
            for (size_t j = 0; j < local.size(); ++j)
                reduced[j] = double(local[j]);
            FML::SumArrayOverTasks(reduced.data(), reduced.size());
            valid = true;
            for (double x : reduced)
                valid &= std::isfinite(x);
            if (!valid)
                throw UnrepresentableState("[FixedIC] Unrepresentable linear likelihood statistics");
            value_ = reduced[0];
            for (size_t a = 0; a < N; ++a) {
                gradient_[a] = reduced[1 + a];
                for (size_t b = 0; b < N; ++b)
                    gram_[a * N + b] = reduced[1 + N + a * N + b];
            }
        }
        double value(const std::array<double, N> & bias) const {
            long double result = value_;
            for (size_t a = 0; a < N; ++a) {
                const long double da = (long double)bias[a] - reference_[a];
                result += da * gradient_[a];
                for (size_t b = 0; b < N; ++b)
                    result += .5L * da * gram_[a * N + b] * ((long double)bias[b] - reference_[b]);
            }
            return double(result);
        }
        FixedICGaussianProposal<N> gaussian_proposal(const std::array<Parameter, N> & priors) const {
            std::array<double, N> scales{}, centers{}, gradient{};
            std::array<double, N * N> precision{};
            for (size_t i = 0; i < N; ++i) {
                const auto & p = priors[i];
                scales[i] = p.prior == Prior::Gaussian ? p.sigma : (p.upper - p.lower) / std::sqrt(12.);
                centers[i] = p.prior == Prior::Gaussian ? p.mean : p.lower + .5 * (p.upper - p.lower);
                if (!(scales[i] > 0) || !std::isfinite(scales[i]) || !std::isfinite(centers[i]))
                    throw std::invalid_argument("[FixedIC] Invalid proposal prior scales");
            }
            double largest = 0;
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j) {
                    precision[i * N + j] = gram_[i * N + j] * scales[i] * scales[j];
                    if (!std::isfinite(precision[i * N + j]))
                        throw UnrepresentableState("[FixedIC] Unrepresentable scaled proposal precision");
                    if (i == j)
                        largest = std::max(largest, precision[i * N + j]);
                }
            // Proper proposal even for a rank-deficient likelihood. Uniform priors use a Gaussian
            // approximation with the same midpoint/variance, ONLY in q. No target penalty is added.
            const double ridge = std::max(1., 64 * std::numeric_limits<double>::epsilon() * N * largest);
            for (size_t i = 0; i < N; ++i) {
                precision[i * N + i] += ridge;
                gradient[i] = scales[i] * gradient_[i] + ridge * (reference_[i] - centers[i]) / scales[i];
                if (!std::isfinite(precision[i * N + i]) || !std::isfinite(gradient[i]))
                    throw UnrepresentableState("[FixedIC] Unrepresentable regularized proposal geometry");
            }
            return FixedICGaussianProposal<N>::from_scaled_quadratic(reference_, gradient, precision, scales);
        }
        std::array<double, N> gradient(const std::array<double, N> & bias) const {
            auto result = gradient_;
            for (size_t a = 0; a < N; ++a)
                for (size_t b = 0; b < N; ++b)
                    result[a] += gram_[a * N + b] * (bias[b] - reference_[b]);
            return result;
        }
    };

    struct FixedICOptions {
        int warmup{128}, samples{1024}, bias_sweeps{8};
        double amplitude_step{.07}; // Frozen; warmup is discarded, with no tuning on retained draws.
        uint64_t seed{1};
    };
    template <size_t N>
    struct FixedICResult {
        std::vector<std::array<double, N + 1>> draws;
        std::vector<double> potential;
        int amplitude_attempts{}, amplitude_accepted{}, forward_evaluations{}, support_rejections{};
        double sampling_acceptance{};
    };
    // Exact Metropolis-within-Gibbs in (alpha, beta_j=alpha^power_j b_j).
    // The alpha proposal is symmetric with reflection in its finite positive prior bounds;
    // its target includes +sum(power_j)*ln(alpha). Bias conditionals remain in physical b.
    // CacheProvider(alpha) must collectively return the SAME replicated linear likelihood on all ranks.
    // The fixed initial realization lives in that provider and is never changed by this sampler.
    template <bool Joint, size_t N, class CacheProvider>
    FixedICResult<N> sample_fixed_ic_amplitude_impl(CacheProvider provider,
                                                    Parameter amplitude,
                                                    std::array<Parameter, N> bias_priors,
                                                    std::array<double, N> powers,
                                                    std::array<double, N + 1> initial,
                                                    FixedICOptions options = {}) {
        using FIXED_IC_DETAIL::require;
        auto valid_prior = [](const Parameter & p) {
            return std::isfinite(p.lower) && std::isfinite(p.upper) && p.lower < p.upper &&
                   std::isfinite(p.upper - p.lower) && std::isfinite(p.mean) && std::isfinite(p.sigma) && p.sigma > 0 &&
                   (p.prior == Prior::Gaussian || p.prior == Prior::Uniform);
        };
        bool valid = valid_prior(amplitude) && amplitude.lower > 0 && options.warmup >= 0 && options.samples > 0 &&
                     options.warmup <= INT_MAX - options.samples &&
                     (Joint ? options.bias_sweeps >= 0 : options.bias_sweeps > 0) &&
                     std::isfinite(options.amplitude_step) && options.amplitude_step > 0 &&
                     initial[0] >= amplitude.lower && initial[0] <= amplitude.upper;
        for (size_t j = 0; j < N; ++j)
            valid &= valid_prior(bias_priors[j]) && std::isfinite(powers[j]) && std::isfinite(initial[j + 1]) &&
                     initial[j + 1] >= bias_priors[j].lower && initial[j + 1] <= bias_priors[j].upper;
        require(valid, "[FixedIC] Invalid sampler priors, controls or initial state");
        std::vector<double> controls = {double(Joint),
                                        double(options.warmup),
                                        double(options.samples),
                                        double(options.bias_sweeps),
                                        options.amplitude_step,
                                        double(options.seed >> 32),
                                        double(options.seed & 0xffffffffu)};
        for (auto p : bias_priors)
            controls.insert(controls.end(), {p.lower, p.upper, p.mean, p.sigma, double(p.prior)});
        controls.insert(controls.end(),
                        {amplitude.lower, amplitude.upper, amplitude.mean, amplitude.sigma, double(amplitude.prior)});
        controls.insert(controls.end(), powers.begin(), powers.end());
        controls.insert(controls.end(), initial.begin(), initial.end());
        FIXED_IC_DETAIL::replicated(controls);
        std::array<double, N> bias{};
        std::copy(initial.begin() + 1, initial.end(), bias.begin());
        double alpha = initial[0], power_sum = 0;
        for (double x : powers)
            power_sum += x;
        std::mt19937_64 rng(options.seed);
        std::normal_distribution<double> normal(0., 1.);
        std::uniform_real_distribution<double> uniform(0., 1.);
        auto cache = provider(alpha);
        struct Empty {};
        std::conditional_t<Joint, std::optional<FixedICGaussianProposal<N>>, Empty> geometry;
        if constexpr (Joint)
            geometry = cache.gaussian_proposal(bias_priors);
        FixedICResult<N> result;
        result.forward_evaluations = 1;
        FML::COMMUNICATION::collective_try(
            [&] {
                result.draws.reserve(options.samples);
                result.potential.reserve(options.samples);
            },
            "[FixedIC] Allocate bounded scalar output");
        auto physical_value = [&](const auto & c, double a, const auto & b) {
            double u = c.value(b) + amplitude.penalty(a);
            for (size_t j = 0; j < N; ++j)
                u += bias_priors[j].penalty(b[j]);
            return u;
        };
        const double span = amplitude.upper - amplitude.lower;
        require(std::isfinite(2 * span) && std::isfinite(power_sum), "[FixedIC] Unrepresentable proposal scaling");
        int retained_accept = 0;
        for (int i = 0; i < options.warmup + options.samples; ++i) {
            for (int sweep = 0; sweep < options.bias_sweeps; ++sweep)
                for (size_t j = 0; j < N; ++j) {
                    auto potential = [&](double b) {
                        auto trial = bias;
                        trial[j] = b;
                        return physical_value(cache, alpha, trial);
                    };
                    bias[j] = FIXED_IC_DETAIL::convex_slice(bias[j],
                                                            bias_priors[j].lower,
                                                            bias_priors[j].upper,
                                                            potential,
                                                            rng);
                }
            const double raw = (alpha - amplitude.lower) + options.amplitude_step * normal(rng);
            double wrapped = std::fmod(raw, 2 * span);
            if (wrapped < 0)
                wrapped += 2 * span;
            const double proposal = amplitude.lower + (wrapped <= span ? wrapped : 2 * span - wrapped);
            ++result.amplitude_attempts;
            if constexpr (Joint) {
                ++result.forward_evaluations;
                try {
                    auto next = provider(proposal);
                    auto next_geometry = next.gaussian_proposal(bias_priors);
                    const auto trial = next_geometry.draw(rng, normal); // Exactly ONE draw; no clipping/resampling.
                    valid = true;
                    for (size_t j = 0; j < N; ++j)
                        valid &= std::isfinite(trial[j]) && trial[j] >= bias_priors[j].lower &&
                                 trial[j] <= bias_priors[j].upper;
                    if (valid) {
                        const double change = physical_value(next, proposal, trial) -
                                              physical_value(cache, alpha, bias) + next_geometry.log_density(trial) -
                                              geometry->log_density(bias);
                        double u = 0.;
                        while (u == 0.)
                            u = uniform(rng);
                        if (std::isfinite(change) && std::log(u) < -change) {
                            alpha = proposal;
                            bias = trial;
                            cache = std::move(next);
                            geometry = std::move(next_geometry);
                            ++result.amplitude_accepted;
                            if (i >= options.warmup)
                                ++retained_accept;
                        }
                    } else
                        ++result.support_rejections;
                } catch (const UnrepresentableState &) {
                    // Numerical proposal failure rejects the joint move.
                }
            } else {
                auto trial = bias;
                valid = true;
                for (size_t j = 0; j < N; ++j) {
                    trial[j] *= std::pow(alpha / proposal, powers[j]);
                    valid &=
                        std::isfinite(trial[j]) && trial[j] >= bias_priors[j].lower && trial[j] <= bias_priors[j].upper;
                }
                if (valid) {
                    ++result.forward_evaluations;
                    try {
                        auto next = provider(proposal);
                        const double change = physical_value(next, proposal, trial) -
                                              physical_value(cache, alpha, bias) +
                                              power_sum * std::log(proposal / alpha);
                        double u = 0.;
                        while (u == 0.)
                            u = uniform(rng);
                        if (std::isfinite(change) && std::log(u) < -change) {
                            alpha = proposal;
                            bias = trial;
                            cache = std::move(next);
                            ++result.amplitude_accepted;
                            if (i >= options.warmup)
                                ++retained_accept;
                        }
                    } catch (const UnrepresentableState &) {
                        // A numerical proposal failure rejects alpha; malformed contracts still propagate.
                    }
                }
                if (!valid)
                    ++result.support_rejections;
            }
            if (i >= options.warmup) {
                std::array<double, N + 1> draw{};
                draw[0] = alpha;
                std::copy(bias.begin(), bias.end(), draw.begin() + 1);
                result.draws.push_back(draw);
                result.potential.push_back(physical_value(cache, alpha, bias));
            }
        }
        result.sampling_acceptance = double(retained_accept) / options.samples;
        return result;
    }
    // Historical rescaled-bias proposal, preserving its default behavior.
    template <size_t N, class CacheProvider>
    FixedICResult<N> sample_fixed_ic_amplitude(CacheProvider provider,
                                               Parameter amplitude,
                                               std::array<Parameter, N> bias_priors,
                                               std::array<double, N> powers,
                                               std::array<double, N + 1> initial,
                                               FixedICOptions options = {}) {
        return sample_fixed_ic_amplitude_impl<false, N>(std::move(provider),
                                                        amplitude,
                                                        bias_priors,
                                                        powers,
                                                        initial,
                                                        options);
    }
    // Joint proposal in physical (alpha,b) coordinates. Uses the normalized unbounded Gaussian q
    // from each cache; finite prior bounds are enforced by one-shot rejection and the exact MH ratio.
    template <size_t N, class CacheProvider>
    FixedICResult<N> sample_fixed_ic_joint_amplitude(CacheProvider provider,
                                                     Parameter amplitude,
                                                     std::array<Parameter, N> bias_priors,
                                                     std::array<double, N + 1> initial,
                                                     FixedICOptions options = {}) {
        return sample_fixed_ic_amplitude_impl<true, N>(std::move(provider),
                                                       amplitude,
                                                       bias_priors,
                                                       {},
                                                       initial,
                                                       options);
    }
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
