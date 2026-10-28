#ifndef FML_HOD_HEADER
#define FML_HOD_HEADER
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <unordered_set>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif
namespace FML::HOD {
    using Vector = std::array<double, 3>;
    using Stream = FML::RANDOM::COUNTER_BASED::Stream;
    struct Member {
        std::uint64_t id = 0;
        Vector position{}, velocity{};
    };
    struct Halo {
        std::uint64_t id = 0;
        double mass = 1e13, radius = 1, concentration = 5, velocity_dispersion = 0;
        Vector position{}, velocity{};
        std::vector<Member> members; // Optional particle phase space, IDs stable within the halo.
    };
    struct Galaxy {
        Vector position{}, velocity{};
        std::uint64_t halo_id = 0, ordinal = 0; // Collision-free identity pair; central=0, satellites=1,2,...
        bool central = false;
        double* get_pos() { return position.data(); }
        double* get_vel() { return velocity.data(); }
        constexpr int get_ndim() const { return 3; }
    };
    struct OccupationMeans {
        double central_probability = 0, satellites_if_central = 0;
    };
    struct Zheng07 {
        double log10_mmin = 13, sigma_log_mass = .4, m0 = 1e12, m1 = 1e14, alpha = 1;
        OccupationMeans operator()(const Halo& halo) const {
            if (!std::isfinite(log10_mmin) || !(sigma_log_mass > 0) || !std::isfinite(sigma_log_mass) || !(m0 >= 0) ||
                !std::isfinite(m0) || !(m1 > 0) || !std::isfinite(m1) || !(alpha > 0) || !std::isfinite(alpha) ||
                !(halo.mass > 0) || !std::isfinite(halo.mass))
                throw std::invalid_argument("Invalid Zheng07 occupation parameters or halo mass");
            return {.5 * (1 + std::erf((std::log10(halo.mass) - log10_mmin) / sigma_log_mass)),
                    halo.mass > m0 ? std::pow((halo.mass - m0) / m1, alpha) : 0};
        }
    };
    enum class SatelliteProfile { NFW, Particles };
    struct Options {
        std::uint64_t seed = 1234;
        std::uint32_t stream = 0;
        double boxsize = 0; // >0: wrap positions in the periodic box; 0: unwrapped Cartesian output.
        SatelliteProfile profile = SatelliteProfile::NFW;
        bool require_central = true;
        double satellite_concentration_factor = 1;
        double central_velocity_bias = 0, satellite_velocity_bias = 1;
        std::uint64_t max_satellites_per_halo = 1000000; // Fail explicitly; never truncate a draw.
    };
    struct OccupationDraw {
        std::uint64_t halo_id = 0, satellites = 0;
        bool central = false;
    };
    struct Mock {
        std::vector<Galaxy> galaxies;
        std::vector<OccupationDraw> occupations;
    };
    namespace detail {
        inline void require(bool ok, const char* message) {
            if (!ok)
                throw std::invalid_argument(message);
        }
        inline bool finite(const Vector& x) {
            return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
        }
        /// Hörmann's transformed-rejection (PTRS) method for large means; product method below 10.
        /// Same mathematical algorithm used by NumPy; independent Philox-driven implementation.
        inline std::uint64_t poisson(double mean, Stream& rng) {
            require(std::isfinite(mean) && mean >= 0 && mean <= 1e8, "Poisson mean must be finite and in [0,1e8]");
            if (mean == 0)
                return 0;
            if (mean < 10) {
                const double stop = std::exp(-mean);
                double product = 1;
                std::uint64_t n = 0;
                do {
                    product *= rng.uniform();
                    ++n;
                } while (product > stop);
                return n - 1;
            }
            const double scale = .931 + 2.53 * std::sqrt(mean);
            const double shape = -.059 + .02483 * scale;
            const double inverse = 1.1239 + 1.1328 / (scale - 3.4);
            const double quick = .9277 - 3.6224 / (scale - 2);
            while (true) {
                const double centered = rng.uniform() - .5, draw = rng.uniform();
                const double margin = .5 - std::abs(centered);
                const double proposed = std::floor((2 * shape / margin + scale) * centered + mean + .43);
                if (proposed < 0 || proposed > 0x1p53)
                    continue; // Check range before any integer conversion.
                if (margin >= .07 && draw <= quick)
                    return std::uint64_t(proposed);
                if (margin < .013 && draw > margin)
                    continue;
                const double log_accept = std::log(draw * inverse / (shape / (margin * margin) + scale));
                const double log_mass = -mean + proposed * std::log(mean) - std::lgamma(proposed + 1);
                if (log_accept <= log_mass)
                    return std::uint64_t(proposed);
            }
        }
        inline double nfw_small_factor(double x) {
            double sum = 0;
            for (int n = 12; n >= 2; --n)
                sum = sum * x + (n % 2 ? -1 : 1) * (n - 1.0) / n;
            return sum;
        }
        inline double nfw_mass(double x) {
            if (x < 1e-3) {
                double power = x * x, sum = 0;
                for (int n = 2; n <= 12; ++n) {
                    sum += (n % 2 ? -1 : 1) * (n - 1.0) / n * power;
                    power *= x;
                }
                return sum;
            }
            return std::log1p(x) - x / (1 + x);
        }
        inline double wrap(double x, double box) {
            if (box <= 0)
                return x;
            x = std::fmod(x, box);
            if (x < 0)
                x += box;
            return x >= box ? 0 : x;
        }
    } // namespace detail
    /// Occupations without allocating any galaxy phase space.
    inline OccupationDraw sample_occupation(std::uint64_t halo_id, const OccupationMeans& mean,
                                            const Options& options = {}) {
        detail::require(mean.central_probability >= 0 && mean.central_probability <= 1 &&
                            std::isfinite(mean.central_probability) && mean.satellites_if_central >= 0 &&
                            std::isfinite(mean.satellites_if_central),
                        "Invalid HOD occupation means");
        Stream central(options.seed, halo_id, 0x43454E54, options.stream);
        Stream satellite(options.seed, halo_id, 0x5341544E, options.stream);
        OccupationDraw result;
        result.halo_id = halo_id;
        result.central = central.uniform() < mean.central_probability;
        const double lambda = options.require_central ? (result.central ? mean.satellites_if_central : 0)
                                                      : mean.central_probability * mean.satellites_if_central;
        detail::require(lambda <= options.max_satellites_per_halo, "HOD mean exceeds per-halo resource limit");
        result.satellites = detail::poisson(lambda, satellite);
        if (result.satellites > options.max_satellites_per_halo)
            throw std::length_error("HOD draw exceeds per-halo resource limit");
        return result;
    }
    /// Inverse CDF of the NFW mass profile truncated at the supplied halo radius.
    inline double nfw_radius_fraction(double probability, double concentration) {
        detail::require(probability >= 0 && probability <= 1 && concentration > 0 && std::isfinite(concentration),
                        "Invalid NFW quantile or concentration");
        if (probability == 0 || probability == 1)
            return probability;
        if (concentration < 1e-3) {
            // Cancel the c^2 factor analytically, avoiding underflow in the small-c limit.
            const double denominator = detail::nfw_small_factor(concentration);
            double lo = 0, hi = 1;
            for (int j = 0; j < 64; ++j) {
                const double fraction = .5 * (lo + hi);
                const double cdf =
                    fraction * fraction * detail::nfw_small_factor(concentration * fraction) / denominator;
                if (cdf < probability)
                    lo = fraction;
                else
                    hi = fraction;
            }
            return .5 * (lo + hi);
        }
        // Solve in log(1+c*r/R), so a wide concentration range is resolved uniformly.
        const double target = probability * detail::nfw_mass(concentration);
        double lo = 0, hi = std::log1p(concentration);
        for (int j = 0; j < 64; ++j) {
            const double y = .5 * (lo + hi);
            const double x = y < 1e-3 ? std::expm1(y) : 0;
            const double mass = y < 1e-3 ? x * x * detail::nfw_small_factor(x) : y + std::expm1(-y);
            if (mass < target)
                lo = y;
            else
                hi = y;
        }
        return std::min(1.0, std::expm1(.5 * (lo + hi)) / concentration);
    }
    /// Own and validate halo data once; reuse for occupation/seed scans. Move inputs to avoid a copy.
    class PreparedCatalog {
        std::vector<Halo> data;
        std::vector<std::vector<size_t>> member_order;

      public:
        explicit PreparedCatalog(std::vector<Halo> halos) : data(std::move(halos)), member_order(data.size()) {
            std::unordered_set<std::uint64_t> ids;
            for (size_t i = 0; i < data.size(); ++i) {
                const auto& h = data[i];
                detail::require(ids.insert(h.id).second, "Halo IDs must be unique within a prepared catalogue");
                detail::require(h.mass > 0 && std::isfinite(h.mass) && h.radius > 0 && std::isfinite(h.radius) &&
                                    h.concentration > 0 && std::isfinite(h.concentration) &&
                                    h.velocity_dispersion >= 0 && std::isfinite(h.velocity_dispersion) &&
                                    detail::finite(h.position) && detail::finite(h.velocity),
                                "Invalid halo mass, radius, concentration or phase space");
                auto& order = member_order[i];
                order.resize(h.members.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](size_t a, size_t b) { return h.members[a].id < h.members[b].id; });
                for (size_t j = 0; j < order.size(); ++j) {
                    const auto& p = h.members[order[j]];
                    detail::require(detail::finite(p.position) && detail::finite(p.velocity) &&
                                        (!j || p.id != h.members[order[j - 1]].id),
                                    "Invalid/duplicate halo member phase-space ID");
                }
            }
        }
        const std::vector<Halo>& halos() const { return data; }
        const std::vector<size_t>& ordered_members(size_t index) const { return member_order.at(index); }
    };
    /// Local operation: no hidden gathers/reductions. Global halo IDs must be unique across MPI ranks.
    /// A custom occupation callable returns central probability and the conditional satellite mean.
    template <class Occupation>
    Mock populate(const PreparedCatalog& prepared, const Occupation& occupation, const Options& options = {}) {
        detail::require(
            std::isfinite(options.boxsize) && options.boxsize >= 0 && options.satellite_concentration_factor > 0 &&
                std::isfinite(options.satellite_concentration_factor) && options.central_velocity_bias >= 0 &&
                std::isfinite(options.central_velocity_bias) && options.satellite_velocity_bias >= 0 &&
                std::isfinite(options.satellite_velocity_bias) &&
                (options.profile == SatelliteProfile::NFW || options.profile == SatelliteProfile::Particles),
            "Invalid HOD options");
        const auto& halos = prepared.halos();
        Mock result;
        result.occupations.resize(halos.size());
        std::vector<size_t> offsets(halos.size() + 1);
        // Call user models and perform all allocations/resource checks outside the parallel phase.
        for (size_t i = 0; i < halos.size(); ++i) {
            const auto& h = halos[i];
            const auto mean = occupation(h);
            result.occupations[i] = sample_occupation(h.id, mean, options);
            const auto& counts = result.occupations[i];
            const double c = h.concentration * options.satellite_concentration_factor;
            if (options.profile == SatelliteProfile::NFW && counts.satellites)
                detail::require(std::isfinite(c) && c > 0 && (options.boxsize == 0 || h.radius < .5 * options.boxsize),
                                "Invalid NFW concentration or radius >= half periodic box");
            if (options.profile == SatelliteProfile::Particles && counts.satellites)
                detail::require(!h.members.empty(), "Particle satellite profile requires halo members");
            for (int d = 0; d < 3; ++d) {
                detail::require(
                    std::isfinite(std::abs(h.position[d]) + h.radius) &&
                        std::isfinite(std::abs(h.velocity[d]) +
                                      10 * h.velocity_dispersion *
                                          std::max(options.central_velocity_bias, options.satellite_velocity_bias)),
                    "HOD phase space exceeds finite arithmetic range");
            }
            if (options.profile == SatelliteProfile::Particles && counts.satellites)
                for (const auto& member : h.members)
                    for (int d = 0; d < 3; ++d)
                        detail::require(std::isfinite(h.velocity[d] + options.satellite_velocity_bias *
                                                                          (member.velocity[d] - h.velocity[d])),
                                        "Particle velocity bias exceeds finite arithmetic");
            const size_t room = result.galaxies.max_size() - offsets[i];
            if (counts.satellites > room || size_t(counts.central) > room - size_t(counts.satellites))
                throw std::length_error("HOD catalogue exceeds addressable size");
            offsets[i + 1] = offsets[i] + size_t(counts.satellites) + size_t(counts.central);
        }
        result.galaxies.resize(offsets.back());
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < halos.size(); ++i) {
            const auto& h = halos[i];
            const auto counts = result.occupations[i];
            size_t out = offsets[i];
            Stream position_rng(options.seed, h.id, 0x504F534E, options.stream),
                velocity_rng(options.seed, h.id, 0x56454C53, options.stream),
                central_rng(options.seed, h.id, 0x56454C43, options.stream),
                particle_rng(options.seed, h.id, 0x50415254, options.stream);
            if (counts.central) {
                Galaxy g;
                g.halo_id = h.id;
                g.central = true;
                g.position = h.position;
                g.velocity = h.velocity;
                for (int d = 0; d < 3; ++d) {
                    g.position[d] = detail::wrap(g.position[d], options.boxsize);
                    g.velocity[d] += options.central_velocity_bias * h.velocity_dispersion * central_rng.normal();
                }
                result.galaxies[out++] = g;
            }
            for (std::uint64_t j = 0; j < counts.satellites; ++j) {
                Galaxy g;
                g.halo_id = h.id;
                g.ordinal = j + 1;
                if (options.profile == SatelliteProfile::NFW) {
                    const double r =
                        h.radius * nfw_radius_fraction(position_rng.uniform(),
                                                       h.concentration * options.satellite_concentration_factor);
                    const double mu = 2 * position_rng.uniform() - 1,
                                 phi = 6.2831853071795864769 * position_rng.uniform();
                    const double transverse = std::sqrt(std::max(0.0, 1 - mu * mu));
                    const Vector direction{transverse * std::cos(phi), transverse * std::sin(phi), mu};
                    for (int d = 0; d < 3; ++d) {
                        g.position[d] = h.position[d] + r * direction[d];
                        g.velocity[d] = h.velocity[d] +
                                        options.satellite_velocity_bias * h.velocity_dispersion * velocity_rng.normal();
                    }
                } else {
                    const auto& order = prepared.ordered_members(i);
                    const auto& member = h.members[order[size_t(particle_rng.bounded(order.size()))]];
                    g.position = member.position;
                    for (int d = 0; d < 3; ++d)
                        g.velocity[d] =
                            h.velocity[d] + options.satellite_velocity_bias * (member.velocity[d] - h.velocity[d]);
                }
                for (auto& x : g.position)
                    x = detail::wrap(x, options.boxsize);
                result.galaxies[out++] = g;
            }
        }
        return result;
    }
} // namespace FML::HOD
#endif
