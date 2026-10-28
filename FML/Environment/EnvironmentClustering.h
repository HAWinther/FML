#ifndef FML_ENVIRONMENT_CLUSTERING_HEADER
#define FML_ENVIRONMENT_CLUSTERING_HEADER
#include <FML/CountsInCells/SpatialStatistics.h>
#include <FML/PairCounting/PairCount.h>
#include <map>
#include <numeric>
#include <set>
namespace FML::ENVIRONMENT {
    using Point = std::array<double, 3>;
    using Count = unsigned long long;
    struct Sample {
        Point position{};
        std::uint64_t id = 0;
    };
    struct Environments {
        std::vector<Count> counts;
        std::vector<double> delta;
        double radius = 0, boxsize = 0;
        bool exclude_same_id = false;
    };
    struct Quantiles {
        std::vector<size_t> label, counts;
        std::vector<double> minimum, maximum;
    };
    struct Correlation {
        std::vector<double> radius, pairs, xi;
    };
    struct DensitySplit {
        Quantiles quantiles;
        std::vector<Correlation> cross;
        Correlation parent;
    };
    struct MarkedCorrelation {
        Correlation unmarked;
        std::vector<double> marked_pairs, weighted_xi, mark_ratio, pair_normalized_ratio;
        double mean_mark = 0, finite_pair_baseline = 0;
    };
    namespace detail {
        inline void require(bool valid, const char* message) {
            int ok = valid;
            FML::MinOverTasks(&ok);
            if (!ok)
                throw std::invalid_argument(message);
        }
        inline void local(bool valid, const char* message) {
            if (!valid)
                throw std::invalid_argument(message);
        }
        inline void word(Count& h, Count value) {
            h ^= value;
            h *= 1099511628211ULL;
        }
        inline void real(Count& h, double value) {
            std::uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            word(h, bits);
        }
        inline void same(Count h) {
            Count lo = h, hi = h;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, "Environment catalogues/options differ across MPI ranks");
        }
        inline bool box(double L) { return L > 0 && std::isfinite(L * L * L) && L * L * L > 0; }
        inline bool catalogue(const std::vector<Sample>& points, double L) {
            std::set<std::uint64_t> ids;
            for (const auto& p : points) {
                if (!ids.insert(p.id).second)
                    return false;
                for (double x : p.position)
                    if (!std::isfinite(x) || x < 0 || x >= L)
                        return false;
            }
            return true;
        }
        inline void hash(Count& h, const std::vector<Sample>& points) {
            word(h, points.size());
            for (const auto& p : points) {
                word(h, p.id);
                for (double x : p.position)
                    real(h, x);
            }
        }
        inline void hash(Count& h, const std::vector<double>& values) {
            word(h, values.size());
            for (double x : values)
                real(h, x);
        }
        inline void schema(const std::vector<Sample>& a, const std::vector<Sample>& b, double L, double low,
                           double high, int bins) {
            require(box(L) && std::isfinite(low) && std::isfinite(high) && low > 0 && high > low && high <= L / 2 &&
                        bins > 0 && catalogue(a, L) && catalogue(b, L),
                    "Invalid periodic catalogue, unique IDs or radial bins (0 < rmin < rmax <= L/2 required)");
            bool shells = true;
            for (int b = 0; b < bins; ++b) {
                const double a = low / L + (high / L - low / L) * b / bins,
                             z = low / L + (high / L - low / L) * (b + 1) / bins;
                shells &= z * z * z - a * a * a > 0;
            }
            require(shells, "Unrepresentable periodic shell volume");
            Count h = 1469598103934665603ULL;
            hash(h, a);
            hash(h, b);
            real(h, L);
            real(h, low);
            real(h, high);
            word(h, Count(bins));
            same(h);
        }
        struct Particle {
            Point position{};
            double mark = 1;
            size_t quantile = 0;
            double* get_pos() { return position.data(); }
            const double* get_pos() const { return position.data(); }
            constexpr int get_ndim() const { return 3; }
        };
        inline std::vector<Particle> particles(const std::vector<Sample>& points, double L) {
            std::vector<Particle> out(points.size());
            for (size_t i = 0; i < points.size(); ++i)
                for (int d = 0; d < 3; ++d)
                    out[i].position[d] = std::min(std::nextafter(1., 0.), points[i].position[d] / L);
            return out;
        }
        inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }
    } // namespace detail
    /// Local pure ranking operation: equal-count ranks, deterministic (density, ID) ties.
    /// Equal densities may be split; IDs must be unique and density values finite.
    inline Quantiles density_quantiles(const std::vector<double>& density, const std::vector<std::uint64_t>& ids,
                                       size_t number) {
        detail::local(density.size() == ids.size() && number > 0 && number <= density.size(),
                      "Invalid density-quantile dimensions");
        std::set<std::uint64_t> unique;
        for (size_t i = 0; i < ids.size(); ++i)
            detail::local(std::isfinite(density[i]) && unique.insert(ids[i]).second,
                          "Nonfinite density or duplicate quantile ID");
        std::vector<size_t> order(ids.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return density[a] < density[b] || (density[a] == density[b] && ids[a] < ids[b]);
        });
        Quantiles out;
        out.label.resize(ids.size());
        out.counts.resize(number);
        out.minimum.resize(number);
        out.maximum.resize(number);
        size_t start = 0;
        for (size_t q = 0; q < number; ++q) {
            size_t count = ids.size() / number + (q < ids.size() % number);
            out.counts[q] = count;
            out.minimum[q] = density[order[start]];
            out.maximum[q] = density[order[start + count - 1]];
            for (size_t i = start; i < start + count; ++i)
                out.label[order[i]] = q;
            start += count;
        }
        return out;
    }
    /// Collective. Replicated catalogues in physical [0,L)^3. Queries are divided
    /// across ranks; only the counts, not neighbour lists, are reduced.
    inline Environments top_hat_environment(const std::vector<Sample>& tracers, const std::vector<Sample>& centers,
                                            double L, double radius, bool exclude_same_id = false) {
        detail::require(detail::box(L) && radius > 0 && radius <= L / 2 && std::isfinite(radius) && !tracers.empty() &&
                            detail::catalogue(tracers, L) && detail::catalogue(centers, L),
                        "Invalid periodic top-hat catalogue or radius");
        Count h = 1469598103934665603ULL;
        detail::hash(h, tracers);
        detail::hash(h, centers);
        detail::real(h, L);
        detail::real(h, radius);
        detail::word(h, exclude_same_id);
        detail::same(h);
        const double fraction = 4.1887902047863909846 * std::pow(radius / L, 3);
        detail::require(fraction > 0, "Top-hat volume underflows");
        std::map<std::uint64_t, size_t> indices;
        FML::KDTREE::pointVec points;
        for (size_t i = 0; i < tracers.size(); ++i) {
            indices.emplace(tracers[i].id, i);
            points.emplace_back(tracers[i].position.begin(), tracers[i].position.end());
        }
        bool valid = true;
        std::vector<bool> excluded(centers.size());
        for (size_t i = 0; i < centers.size(); ++i)
            if (exclude_same_id) {
                auto found = indices.find(centers[i].id);
                if (found != indices.end()) {
                    excluded[i] = true;
                    valid &= centers[i].position == tracers[found->second].position && tracers.size() > 1;
                }
            }
        detail::require(valid, "Self exclusion requires matching ID and position and at least one remaining tracer");
        FML::COUNTS_IN_CELLS::SpatialStatistics tree(points, L);
        std::vector<FML::COUNTS_IN_CELLS::Query> queries;
        std::vector<size_t> owned;
        for (size_t i = size_t(FML::ThisTask); i < centers.size(); i += size_t(FML::NTasks)) {
            FML::COUNTS_IN_CELLS::Query query;
            query.position = centers[i].position;
            if (excluded[i])
                query.exclude = indices.at(centers[i].id);
            queries.push_back(query);
            owned.push_back(i);
        }
        auto counts = tree.sphere_counts(queries, {radius});
        Environments out;
        out.radius = radius;
        out.boxsize = L;
        out.exclude_same_id = exclude_same_id;
        out.counts.resize(centers.size());
        out.delta.resize(centers.size());
        for (size_t i = 0; i < owned.size(); ++i)
            out.counts[owned[i]] = counts[0][i];
#ifdef USE_MPI
        for (size_t offset = 0; offset < out.counts.size();) {
            int n = int(std::min(size_t(INT_MAX), out.counts.size() - offset));
            MPI_Allreduce(MPI_IN_PLACE, out.counts.data() + offset, n, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                          FML::communicator);
            offset += size_t(n);
        }
#endif
        valid = true;
        for (size_t i = 0; i < centers.size(); ++i) {
            out.delta[i] = double(out.counts[i]) / (double(tracers.size() - size_t(excluded[i])) * fraction) - 1;
            valid &= std::isfinite(out.delta[i]);
        }
        detail::require(valid, "Top-hat overdensity overflows");
        return out;
    }
    /// Collective periodic monopole cross-correlations of quantiles with tracers.
    /// Centers must be uniform volume samples for a volume-weighted density split.
    inline DensitySplit density_split_correlation(const std::vector<Sample>& centers,
                                                  const std::vector<double>& density,
                                                  const std::vector<Sample>& tracers, size_t number, double L,
                                                  double rmin, double rmax, int bins) {
        detail::schema(centers, tracers, L, rmin, rmax, bins);
        bool valid = !tracers.empty() && density.size() == centers.size() && number > 0 && number <= centers.size();
        for (double x : density)
            valid &= std::isfinite(x);
        detail::require(valid, "Invalid density split dimensions/values");
        Count h = number;
        detail::hash(h, density);
        detail::same(h);
        std::vector<std::uint64_t> ids;
        for (const auto& p : centers)
            ids.push_back(p.id);
        detail::require(number < size_t(INT_MAX) && size_t(bins) <= std::vector<double>().max_size() / (number + 1),
                        "Density-split histogram exceeds supported dimensions");
        DensitySplit out;
        out.quantiles = density_quantiles(density, ids, number);
        auto all = detail::particles(centers, L);
        const auto target = detail::particles(tracers, L);
        for (size_t i = 0; i < all.size(); ++i)
            all[i].quantile = out.quantiles.label[i];
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra;
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::ExtraQuantitiesToBinFunction<detail::Particle> by_quantile =
            [](const double*, double, double, const detail::Particle& center, const detail::Particle&, double* sums) {
                sums[center.quantile] += 1;
            };
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::RadialCorrelationFunctionBox(
            all.data(), all.size(), target.data(), target.size(), rmin / L, rmax / L, bins, out.parent.radius,
            out.parent.pairs, false, out.parent.xi, true, false, int(number), extra, by_quantile);
        for (double& r : out.parent.radius)
            r *= L;
        out.cross.resize(number);
        for (size_t q = 0; q < number; ++q) {
            auto& correlation = out.cross[q];
            correlation.radius = out.parent.radius;
            correlation.pairs.resize(bins);
            correlation.xi.resize(bins);
            for (int b = 0; b < bins; ++b) {
                const double lo = rmin / L + (rmax / L - rmin / L) * b / bins,
                             hi = rmin / L + (rmax / L - rmin / L) * (b + 1) / bins;
                const double fraction = 4.1887902047863909846 * (hi * hi * hi - lo * lo * lo);
                correlation.pairs[b] = extra[b][q];
                correlation.xi[b] =
                    correlation.pairs[b] / (double(out.quantiles.counts[q]) * tracers.size() * fraction) - 1;
                valid &= std::isfinite(correlation.xi[b]) && std::isfinite(out.parent.xi[b]);
            }
        }
        detail::require(valid, "Periodic density-split correlation overflows");
        return out;
    }
    /// Local mark recipe ((1+delta_s)/(1+delta_s+delta))^p. No shot-noise correction.
    inline std::vector<double> inverse_density_marks(const std::vector<double>& delta, double delta_s = 1,
                                                     double power = 1) {
        detail::local(delta_s > 0 && delta_s <= 1e100 && std::isfinite(delta_s) && power >= 0 && std::isfinite(power),
                      "Invalid inverse-density mark parameters");
        std::vector<double> out;
        out.reserve(delta.size());
        for (double d : delta) {
            detail::local(std::isfinite(d) && d >= -1, "Marks require finite overdensity >= -1");
            double m = std::exp(power * (std::log1p(delta_s) - std::log(delta_s + (1 + d))));
            detail::local(std::isfinite(m) && m >= 0 && m <= 1e100, "Inverse-density mark exceeds supported range");
            out.push_back(m);
        }
        return out;
    }
    /// Collective marked auto-pairs in one pass. Marks are normalized by their
    /// sample mean; the optional finite-population pair normalization is separate.
    inline MarkedCorrelation marked_correlation(const std::vector<Sample>& tracers, const std::vector<double>& marks,
                                                double L, double rmin, double rmax, int bins) {
        detail::schema(tracers, {}, L, rmin, rmax, bins);
        bool valid = tracers.size() >= 2 && marks.size() == tracers.size();
        long double total = 0;
        for (double m : marks) {
            valid &= std::isfinite(m) && m >= 0 && m <= 1e100;
            total += m;
        }
        detail::require(valid && total > 0,
                        "Marks must be finite/nonnegative, have positive mean, and match at least two tracers");
        Count h = 0;
        detail::hash(h, marks);
        detail::same(h);
        MarkedCorrelation out;
        const long double mean = total / tracers.size();
        out.mean_mark = double(mean);
        auto particles = detail::particles(tracers, L);
        long double prefix = 0, pair_total = 0;
        for (size_t i = 0; i < marks.size(); ++i) {
            const long double m = marks[i] / mean;
            valid &= m <= 1e100;
            particles[i].mark = double(m);
            pair_total += prefix * m;
            prefix += m;
        }
        detail::require(valid, "Normalized marks exceed supported range");
        const double norm = double(tracers.size()) * double(tracers.size() - 1) / 2;
        out.finite_pair_baseline = double(pair_total / norm);
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::DVector2D extra;
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::ExtraQuantitiesToBinFunction<detail::Particle> weighted =
            [](const double*, double, double, const detail::Particle& a, const detail::Particle& b, double* sum) {
                sum[0] += a.mark * b.mark;
            };
        FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::RadialCorrelationFunctionBox(
            particles.data(), particles.size(), rmin / L, rmax / L, bins, out.unmarked.radius, out.unmarked.pairs,
            false, out.unmarked.xi, true, false, 1, extra, weighted);
        out.marked_pairs.resize(bins);
        for (int b = 0; b < bins; ++b)
            out.marked_pairs[b] = extra[b][0];
        out.mark_ratio.resize(bins);
        out.pair_normalized_ratio.resize(bins);
        out.weighted_xi.resize(bins);
        for (int b = 0; b < bins; ++b) {
            const double low = rmin / L + (rmax / L - rmin / L) * b / bins,
                         high = rmin / L + (rmax / L - rmin / L) * (b + 1) / bins;
            const double fraction = 4.1887902047863909846 * (high * high * high - low * low * low);
            out.weighted_xi[b] = out.marked_pairs[b] / (norm * fraction) - 1;
            out.mark_ratio[b] = out.unmarked.pairs[b] > 0 ? out.marked_pairs[b] / out.unmarked.pairs[b] : detail::nan();
            out.pair_normalized_ratio[b] =
                out.finite_pair_baseline > 0 ? out.mark_ratio[b] / out.finite_pair_baseline : detail::nan();
            valid &= std::isfinite(out.weighted_xi[b]) && std::isfinite(out.unmarked.xi[b]);
            out.unmarked.radius[b] *= L;
        }
        detail::require(valid, "Periodic marked correlation overflows");
        return out;
    }
} // namespace FML::ENVIRONMENT
#endif
