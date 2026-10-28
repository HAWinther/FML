#ifndef FML_SHORT_RANGE_GRAVITY_HEADER
#define FML_SHORT_RANGE_GRAVITY_HEADER

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
        double inverse_sqrt_pi_{1.0 / std::sqrt(std::acos(-1.0))};

      public:
        explicit Kernel(Options options = {}) : options_(options) {
            const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
            if (!positive(options.boxsize) || !positive(options.split_scale) || !positive(options.softening_support) ||
                !std::isfinite(options.coupling) || options.coupling < 0 || !positive(options.cutoff_tolerance) ||
                options.cutoff_tolerance >= 1 || options.max_cells == 0)
                throw std::invalid_argument("[ShortRangeGravity] Invalid length, coupling, tolerance or cell budget");
            double lo = 0, hi = 1;
            while (short_range_window(hi) > options.cutoff_tolerance)
                hi *= 2;
            for (int iteration = 0; iteration < 80; ++iteration) {
                const double mid = (lo + hi) * .5;
                if (short_range_window(mid) > options.cutoff_tolerance)
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
        }
        const Options & options() const { return options_; }
        double cutoff() const { return cutoff_; }
        double long_range_filter(double k_squared) const {
            if (!std::isfinite(k_squared) || k_squared < 0)
                throw std::invalid_argument("[ShortRangeGravity] k squared must be finite and nonnegative");
            return std::exp(-k_squared * options_.split_scale * options_.split_scale);
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
        double long_range_coefficient(double r) const {
            const double rs = options_.split_scale;
            const double q = r / (2 * rs);
            if (q < .05) {
                const double t = q * q;
                return inverse_sqrt_pi_ / (rs * rs * rs) *
                       (1.0 / 6 + t * (-1.0 / 10 + t * (1.0 / 28 + t * (-1.0 / 108 + t / 528))));
            }
            return (std::erf(q) - 2 * inverse_sqrt_pi_ * q * std::exp(-q * q)) / (r * r * r);
        }
        /// Untruncated short-range coefficient; public for split/softening validation.
        double untruncated_coefficient(double r) const {
            if (!std::isfinite(r) || r < 0)
                throw std::invalid_argument("[ShortRangeGravity] Separation must be finite and nonnegative");
            if (r < options_.softening_support)
                return options_.coupling * (softened_coefficient(r) - long_range_coefficient(r));
            return options_.coupling * short_range_window(r / (2 * options_.split_scale)) / (r * r * r);
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
