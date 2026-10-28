#ifndef FML_SHORT_RANGE_GRAVITY_HEADER
#define FML_SHORT_RANGE_GRAVITY_HEADER

#include <FML/ShortRangeGravity/Yukawa.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace FML::SHORT_RANGE_GRAVITY {
    using Vector = std::array<double, 3>;

    /// Positions in [0, boxsize), masses in the caller's mass unit. IDs must be unique among sources.
    struct Particle {
        Vector position{};
        double mass{1};
        std::uint64_t id{0};
    };

    /// All lengths share the coordinate unit. softening_support is h, NOT Plummer-equivalent epsilon (h=2.8 epsilon).
    struct Options {
        double boxsize{1};
        double split_scale{0.02};
        double softening_support{0.005};
        double cutoff_tolerance{1e-6}; // Residual pair force relative to unsoftened Newtonian force at the cutoff.
        double coupling{1};            // G in physical coordinates, or the explicitly derived simulation prefactor.
        std::size_t max_cells{1000000};
    };

    inline double short_range_window(double q) {
        return std::erfc(q) + 2.0 / std::sqrt(std::acos(-1.0)) * q * std::exp(-q * q);
    }

    /// Immutable, validated spatial kernel. Acceleration on a target is +m_source * coefficient(r) * (source-target).
    class Kernel {
        Options options_;
        double cutoff_;
        bool yukawa_model_{false};
        double alpha_{0}, lambda_{1};
        double inverse_sqrt_pi_{1.0 / std::sqrt(std::acos(-1.0))};

        Kernel(Options options, double alpha, double lambda, bool yukawa_model)
            : options_(options), yukawa_model_(yukawa_model), alpha_(alpha), lambda_(lambda) {
            if (!std::isfinite(alpha) || !std::isfinite(lambda) || lambda <= 0)
                throw std::invalid_argument("[ShortRangeGravity] Require finite alpha and finite positive lambda");
            const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
            if (!positive(options.boxsize) || !positive(options.split_scale) || !positive(options.softening_support) ||
                !std::isfinite(options.coupling) || options.coupling < 0 || !positive(options.cutoff_tolerance) ||
                options.cutoff_tolerance >= 1 || options.max_cells == 0)
                throw std::invalid_argument("[ShortRangeGravity] Invalid length, coupling, tolerance or cell budget");
            const auto exceeds_tolerance = [&](double q) {
                if (alpha_ == 0)
                    return short_range_window(q) > options.cutoff_tolerance;
                // |F_N,short+alpha F_Y,short| <= (1+|alpha|) F_N,short outside softening.
                const double log_window = -q * q + std::log(detail::erfcx_positive(q) + 2 * q * inverse_sqrt_pi_);
                return log_window + std::log1p(std::abs(alpha_)) > std::log(options.cutoff_tolerance);
            };
            double lo = 0, hi = 1;
            while (exceeds_tolerance(hi))
                hi *= 2;
            for (int iteration = 0; iteration < 80; ++iteration) {
                const double mid = (lo + hi) * .5;
                if (exceeds_tolerance(mid))
                    lo = mid;
                else
                    hi = mid;
            }
            cutoff_ = 2 * options.split_scale * hi;
            if (!(cutoff_ < .5 * options.boxsize) || options.softening_support > cutoff_)
                throw std::invalid_argument(
                    "[ShortRangeGravity] Require softening support <= cutoff < half the periodic box");
            // Catch unusable scales before entering a parallel force loop.
            if (!positive(std::pow(options.softening_support, -3)) || !positive(std::pow(options.split_scale, -3)))
                throw std::invalid_argument("[ShortRangeGravity] Length scale outside representable force range");
            if (alpha_ != 0)
                (void)detail::yukawa_quadrature();
        }

      public:
        explicit Kernel(Options options = {}) : Kernel(options, 0, 1, false) {}
        Kernel(Options options, Newtonian) : Kernel(options) {}
        Kernel(Options options, NewtonYukawa model) : Kernel(options, model.alpha, model.lambda, true) {}
        bool has_yukawa() const { return alpha_ != 0; }
        double yukawa_alpha() const { return alpha_; }
        double yukawa_lambda() const { return lambda_; }
        std::array<double, 3> model_parameters() const {
            return {yukawa_model_ ? 1.0 : 0.0, alpha_, yukawa_model_ ? lambda_ : 0.0};
        }
        bool tree_scale_resolved(double diameter, double opening_angle) const {
            return diameter <
                   opening_angle * (has_yukawa() ? std::min(options_.split_scale, lambda_) : options_.split_scale);
        }
        const Options & options() const { return options_; }
        double cutoff() const { return cutoff_; }
        double long_range_filter(double k_squared) const {
            if (!std::isfinite(k_squared) || k_squared < 0)
                throw std::invalid_argument("[ShortRangeGravity] k squared must be finite and nonnegative");
            const double filter = std::exp(-k_squared * options_.split_scale * options_.split_scale);
            if (!has_yukawa() || k_squared == 0)
                return filter;
            const double x = std::sqrt(k_squared) * lambda_;
            const double ratio = x > 1 ? 1 / (1 + (1 / x) * (1 / x)) : x * x / (1 + x * x);
            const double z = options_.split_scale / lambda_;
            if (alpha_ < 0) {
                const double complement = x > 1 ? ((1 / x) * (1 / x)) / (1 + (1 / x) * (1 / x)) : 1 / (1 + x * x);
                const double difference = complement + ratio * (-std::expm1(-z * z));
                return filter * ((1 + alpha_) - alpha_ * difference);
            }
            return filter * (1 + alpha_ * ratio * std::exp(-z * z));
        }
        /// Positive spline-softened Newtonian Green function, before coupling/mass.
        double newtonian_green(double r) const {
            if (!std::isfinite(r) || r < 0)
                throw std::invalid_argument("[ShortRangeGravity] Separation must be finite and nonnegative");
            const double h = options_.softening_support, u = r / h;
            if (u >= 1)
                return 1 / r;
            if (u < .5)
                return (14.0 / 5 - 16.0 / 3 * u * u + 48.0 / 5 * std::pow(u, 4) - 32.0 / 5 * std::pow(u, 5)) / h;
            return (16.0 / 5 - 1 / (15 * u) - 32.0 / 3 * u * u + 16 * u * u * u - 48.0 / 5 * std::pow(u, 4) +
                    32.0 / 15 * std::pow(u, 5)) /
                   h;
        }
        /// Full softened pair potential per unit source mass, including coupling. Numerical softening uses
        /// effective separation s=1/newtonian_green(r): Phi_Y=-G exp(-s/lambda)/s, exactly Yukawa beyond h.
        double potential(double r) const {
            const double g = newtonian_green(r);
            return -options_.coupling * g *
                   (has_yukawa() ? detail::combined_yukawa_factor(alpha_, -1 / (g * lambda_)) : 1);
        }
        double full_coefficient(double r) const {
            (void)newtonian_green(r); // Validate radius before entering the low-level Newtonian formula.
            const double base = softened_coefficient(r);
            if (!has_yukawa())
                return options_.coupling * base;
            const double s = 1 / newtonian_green(r);
            return options_.coupling * base *
                   (detail::combined_yukawa_factor(alpha_, detail::log_yukawa_force_window(s / lambda_)));
        }
        double long_range_potential(double r) const {
            (void)newtonian_green(r);
            const double rs = options_.split_scale;
            const double n = r == 0 ? inverse_sqrt_pi_ / rs : std::erf(r / (2 * rs)) / r;
            if (alpha_ < 0)
                return -options_.coupling *
                       ((1 + alpha_) * n - alpha_ * detail::yukawa_long_difference(r, rs, lambda_)[0]);
            return -options_.coupling * (n + (has_yukawa() ? alpha_ * detail::yukawa_long(r, rs, lambda_)[0] : 0));
        }
        double untruncated_potential(double r) const {
            if (r < options_.softening_support) {
                if (alpha_ < 0) {
                    const double g = newtonian_green(r), rs = options_.split_scale;
                    const double nl = r == 0 ? inverse_sqrt_pi_ / rs : std::erf(r / (2 * rs)) / r;
                    const double x = 1 / (g * lambda_);
                    const double difference =
                        rs / lambda_ < .5 ? -g * detail::exponential_remainder(-x) -
                                                detail::yukawa_long_difference(r, rs, lambda_, true)[0] :
                                            -g * std::expm1(-x) - detail::yukawa_long_difference(r, rs, lambda_)[0];
                    return -options_.coupling * ((1 + alpha_) * (g - nl) - alpha_ * difference);
                }
                return potential(r) - long_range_potential(r);
            }
            (void)newtonian_green(r);
            const double n = std::erfc(r / (2 * options_.split_scale)) / r;
            if (alpha_ < 0)
                return -options_.coupling *
                       ((1 + alpha_) * n -
                        alpha_ * detail::yukawa_short_difference(r, options_.split_scale, lambda_)[0]);
            return -options_.coupling *
                   (n + (has_yukawa() ? alpha_ * detail::yukawa_short(r, options_.split_scale, lambda_)[0] : 0));
        }
        /// Coefficient of the full spline-softened Newtonian acceleration, without coupling or source mass.
        double softened_coefficient(double r) const {
            const double h = options_.softening_support;
            const double u = r / h;
            if (u >= 1)
                return 1 / (r * r * r);
            if (u < .5)
                return (32.0 / 3 + u * u * (32 * u - 192.0 / 5)) / (h * h * h);
            return (64.0 / 3 - 48 * u + 192.0 / 5 * u * u - 32.0 / 3 * u * u * u - 1 / (15 * u * u * u)) / (h * h * h);
        }
        /// Gaussian long-range acceleration coefficient. Series avoids cancellation close to r=0.
        double newtonian_long_range_coefficient(double r) const {
            const double rs = options_.split_scale;
            const double q = r / (2 * rs);
            if (q < .05) {
                const double t = q * q;
                return inverse_sqrt_pi_ / (rs * rs * rs) *
                       (1.0 / 6 + t * (-1.0 / 10 + t * (1.0 / 28 + t * (-1.0 / 108 + t / 528))));
            }
            return (std::erf(q) - 2 * inverse_sqrt_pi_ * q * std::exp(-q * q)) / (r * r * r);
        }
        /// Complete long-range radial force coefficient, before coupling/mass.
        double long_range_coefficient(double r) const {
            (void)newtonian_green(r);
            const double n = newtonian_long_range_coefficient(r);
            if (alpha_ < 0)
                return (1 + alpha_) * n - alpha_ * detail::yukawa_long_difference(r, options_.split_scale, lambda_)[1];
            return n + (has_yukawa() ? alpha_ * detail::yukawa_long(r, options_.split_scale, lambda_)[1] : 0);
        }
        /// Untruncated short-range coefficient; public for split/softening validation.
        double untruncated_coefficient(double r) const {
            if (!std::isfinite(r) || r < 0)
                throw std::invalid_argument("[ShortRangeGravity] Separation must be finite and nonnegative");
            double n;
            if (r < options_.softening_support)
                n = softened_coefficient(r) - newtonian_long_range_coefficient(r);
            else
                n = short_range_window(r / (2 * options_.split_scale)) / (r * r * r);
            if (!has_yukawa())
                return options_.coupling * n;
            if (alpha_ < 0) {
                double difference;
                if (r < options_.softening_support) {
                    const double separation = 1 / newtonian_green(r);
                    const double full_difference =
                        softened_coefficient(r) * (-std::expm1(detail::log_yukawa_force_window(separation / lambda_)));
                    difference = full_difference - detail::yukawa_long_difference(r, options_.split_scale, lambda_)[1];
                } else
                    difference = detail::yukawa_short_difference(r, options_.split_scale, lambda_)[1];
                return options_.coupling * ((1 + alpha_) * n - alpha_ * difference);
            }
            double y;
            if (r < options_.softening_support) {
                const double s = 1 / newtonian_green(r);
                y = softened_coefficient(r) * detail::yukawa_force_window(s / lambda_) -
                    detail::yukawa_long(r, options_.split_scale, lambda_)[1];
            } else
                y = detail::yukawa_short(r, options_.split_scale, lambda_)[1];
            return options_.coupling * (n + alpha_ * y);
        }
        double coefficient(double r) const { return r >= cutoff_ ? 0 : untruncated_coefficient(r); }
    };

    /// For FML's unit-box Poisson equation laplacian_x Phi=C delta, return accelerations -grad_x Phi.
    /// Positions here use a box of side L, masses sum globally to M; hence coupling=C L^2/(4 pi M).
    /// The caller applies the existing cosmological kick integral once, with a PLUS sign for this acceleration.
    inline double fml_acceleration_coupling(double poisson_normalization, double total_mass, double boxsize = 1) {
        if (!std::isfinite(poisson_normalization) || poisson_normalization < 0 || !std::isfinite(total_mass) ||
            total_mass <= 0 || !std::isfinite(boxsize) || boxsize <= 0)
            throw std::invalid_argument("[ShortRangeGravity] Invalid FML normalization");
        return poisson_normalization / (4 * std::acos(-1.0)) * boxsize * boxsize / total_mass;
    }

    struct Statistics {
        std::uint64_t candidate_pairs{0}; // Directed, nonself candidate interactions, before spherical cutoff.
        std::uint64_t accepted_pairs{0};
        std::size_t cells{0};
    };
    struct Result {
        std::vector<Vector> acceleration;
        Statistics statistics;
    };

    namespace detail {
        inline void validate_particles(const std::vector<Particle> & particles, double boxsize, bool unique_ids) {
            std::vector<std::uint64_t> ids;
            if (unique_ids)
                ids.reserve(particles.size());
            for (const auto & p : particles) {
                if (!std::isfinite(p.mass) || p.mass < 0)
                    throw std::invalid_argument("[ShortRangeGravity] Mass must be finite and nonnegative");
                for (double x : p.position)
                    if (!std::isfinite(x) || x < 0 || x >= boxsize)
                        throw std::invalid_argument("[ShortRangeGravity] Position outside [0, boxsize)");
                if (unique_ids)
                    ids.push_back(p.id);
            }
            if (unique_ids) {
                std::sort(ids.begin(), ids.end());
                if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
                    throw std::invalid_argument("[ShortRangeGravity] Duplicate source IDs");
            }
        }
        inline Vector displacement(const Vector & target, const Vector & source, double boxsize) {
            Vector dx;
            for (int d = 0; d < 3; ++d) {
                dx[d] = source[d] - target[d];
                if (dx[d] > .5 * boxsize)
                    dx[d] -= boxsize;
                else if (dx[d] < -.5 * boxsize)
                    dx[d] += boxsize;
            }
            return dx;
        }
        inline bool
        add_pair(Vector & acceleration, const Particle & target, const Particle & source, const Kernel & k) {
            const auto dx = displacement(target.position, source.position, k.options().boxsize);
            const double r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if (r2 >= k.cutoff() * k.cutoff())
                return false;
            const double factor = source.mass * k.coefficient(std::sqrt(r2));
            for (int d = 0; d < 3; ++d)
                acceleration[d] += factor * dx[d];
            return true;
        }
    } // namespace detail

    /// Sources contain each owned/ghost particle exactly once; matching IDs exclude self interactions.
    inline Result
    direct_sum(const std::vector<Particle> & targets, const std::vector<Particle> & sources, const Kernel & kernel) {
        detail::validate_particles(targets, kernel.options().boxsize, false);
        detail::validate_particles(sources, kernel.options().boxsize, true);
        Result result;
        result.acceleration.resize(targets.size());
        std::uint64_t candidates = 0, accepted = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : candidates, accepted)
#endif
        for (std::size_t i = 0; i < targets.size(); ++i) {
            auto & acceleration = result.acceleration[i];
            for (const auto & source : sources) {
                if (targets[i].id == source.id)
                    continue;
                ++candidates;
                accepted += detail::add_pair(acceleration, targets[i], source, kernel);
            }
        }
        result.statistics.candidate_pairs = candidates;
        result.statistics.accepted_pairs = accepted;
        return result;
    }

    /// Contiguous offset/index cell list. References the source vector: rebuild after any source change or migration.
    class CellList {
        const std::vector<Particle> & sources_;
        Kernel kernel_;
        int nside_{1};
        std::vector<std::size_t> offsets_, indices_;
        std::array<int, 3> cell(const Vector & x) const {
            std::array<int, 3> c;
            for (int d = 0; d < 3; ++d)
                c[d] = std::min(nside_ - 1, int(x[d] / kernel_.options().boxsize * nside_));
            return c;
        }
        std::size_t index(const std::array<int, 3> & c) const {
            return (std::size_t(c[0]) * nside_ + c[1]) * nside_ + c[2];
        }

      public:
        CellList(const std::vector<Particle> & sources, const Kernel & kernel) : sources_(sources), kernel_(kernel) {
            detail::validate_particles(sources, kernel.options().boxsize, true);
            // Bound metadata independently of cutoff or particle count. Coarser cells preserve correctness.
            const auto limit = kernel.options().max_cells;
            while (nside_ < std::numeric_limits<int>::max() - 1) {
                const std::size_t next = std::size_t(nside_) + 1;
                if (next > limit / next / next || next > kernel.options().boxsize / kernel.cutoff())
                    break;
                ++nside_;
            }
            offsets_.resize(std::size_t(nside_) * nside_ * nside_ + 1);
            for (const auto & source : sources)
                ++offsets_[index(cell(source.position)) + 1];
            std::partial_sum(offsets_.begin(), offsets_.end(), offsets_.begin());
            auto cursor = offsets_;
            indices_.resize(sources.size());
            for (std::size_t i = 0; i < sources.size(); ++i)
                indices_[cursor[index(cell(sources[i].position))]++] = i;
        }
        CellList(std::vector<Particle> &&, const Kernel &) = delete;
        Result compute(const std::vector<Particle> & targets) const {
            detail::validate_particles(targets, kernel_.options().boxsize, false);
            Result result;
            result.acceleration.resize(targets.size());
            result.statistics.cells = offsets_.size() - 1;
            std::uint64_t candidates = 0, accepted = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : candidates, accepted)
#endif
            for (std::size_t i = 0; i < targets.size(); ++i) {
                const auto centre = cell(targets[i].position);
                // nside=1/2 has periodic aliases: enumerate each neighbour cell once.
                const int count = std::min(3, nside_);
                for (int a = 0; a < count; ++a)
                    for (int b = 0; b < count; ++b)
                        for (int c = 0; c < count; ++c) {
                            const auto ci = index({(centre[0] + a + nside_ - 1) % nside_,
                                                   (centre[1] + b + nside_ - 1) % nside_,
                                                   (centre[2] + c + nside_ - 1) % nside_});
                            for (auto j = offsets_[ci]; j < offsets_[ci + 1]; ++j) {
                                const auto & source = sources_[indices_[j]];
                                if (targets[i].id == source.id)
                                    continue;
                                ++candidates;
                                accepted += detail::add_pair(result.acceleration[i], targets[i], source, kernel_);
                            }
                        }
            }
            result.statistics.candidate_pairs = candidates;
            result.statistics.accepted_pairs = accepted;
            return result;
        }
    };
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
