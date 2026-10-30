#ifndef FML_SPATIAL_STATISTICS_HEADER
#define FML_SPATIAL_STATISTICS_HEADER
#include <FML/KDTree/KDTree.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <array>
#include <climits>
#include <cstring>
#include <optional>
namespace FML::COUNTS_IN_CELLS {
    using Point = std::array<double, 3>;
    using Count = unsigned long long;
    enum class Reduction { Local, Global };
    struct Query {
        Point position{};
        size_t exclude = std::numeric_limits<size_t>::max();
    };
    struct Bounds {
        Point lower{}, upper{1, 1, 1};
        double volume() const { return (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]); }
    };
    struct KnnDistances {
        std::vector<size_t> orders;
        std::vector<std::vector<double>> distances;
    };
    struct KnnCDF {
        std::vector<size_t> orders;
        std::vector<double> radii;
        std::vector<std::vector<Count>> hits;
        std::vector<std::vector<double>> cdf;
        Count queries = 0, excluded_queries = 0;
        double boxsize = 0;
    };
    struct Options {
        Reduction reduction = Reduction::Global;
        size_t histogram_limit = 128; // Exact P(N=0..limit), plus one explicit overflow bin.
        unsigned factorial_order = 4;
        std::vector<double> delta_edges{-1, -.5, 0, .5, 1, 2, 4, 8};
        std::optional<double> number_density;
        std::optional<Bounds> window;       // Required for nonperiodic full-sphere measurements.
        bool allow_truncated_cells = false; // Explicitly opt into uncorrected clipped spheres.
    };
    struct CellStatistics {
        double radius = 0, volume = 0, reference_number_density = 0;
        Count queries = 0;
        std::vector<Count> histogram, density_histogram;
        std::vector<double> probability, density_probability, density_pdf;
        Count density_underflow = 0, density_overflow = 0, density_undefined = 0;
        double mean = 0, variance = 0, mean_delta = 0, poisson_subtracted_variance = 0,
               normalized_connected_variance = 0;
        std::vector<double> factorial_moments, normalized_factorial_moments; // Include order zero.
    };
    struct CountsInCells {
        Count excluded_queries = 0;
        double boxsize = 0;
        bool uncorrected_clipping = false;
        std::vector<CellStatistics> cells;
        std::vector<double> delta_edges;
        size_t histogram_limit = 0;
    };
    namespace detail {
        inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }
        inline bool finite_point(const Point & p) {
            for (double x : p)
                if (!std::isfinite(x) || std::abs(x) > 1e150)
                    return false;
            return true;
        }
        inline bool valid_bounds(const Bounds & b) {
            if (!finite_point(b.lower) || !finite_point(b.upper) || !(b.volume() > 0) || !std::isfinite(b.volume()))
                return false;
            for (int d = 0; d < 3; ++d)
                if (!(b.lower[d] < b.upper[d]))
                    return false;
            return true;
        }
        inline void check(bool good, Reduction reduction, const char * message) {
            int valid = good;
            if (reduction == Reduction::Global)
                FML::MinOverTasks(&valid);
            if (!valid)
                throw std::invalid_argument(message);
        }
        inline void hash_word(std::uint64_t & hash, std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        inline void hash_double(std::uint64_t & hash, double value) {
            std::uint64_t word;
            std::memcpy(&word, &value, sizeof(word));
            hash_word(hash, word);
        }
        inline void same(std::uint64_t fingerprint, Reduction reduction) {
            if (reduction == Reduction::Local)
                return;
            Count low = fingerprint, high = fingerprint;
            FML::MinOverTasks(&low);
            FML::MaxOverTasks(&high);
            check(low == high,
                  reduction,
                  "Global spatial statistics need identical replicated tracer catalogues and summary options");
        }
        template <class T>
        void sum(std::vector<T> & values, Reduction reduction) {
#ifdef USE_MPI
            if (reduction == Reduction::Global)
                for (size_t offset = 0; offset < values.size();) {
                    const int n = int(std::min(values.size() - offset, size_t(INT_MAX)));
                    MPI_Datatype type;
                    if constexpr (std::is_same_v<T, double>)
                        type = MPI_DOUBLE;
                    else {
                        static_assert(std::is_same_v<T, Count>);
                        type = MPI_UNSIGNED_LONG_LONG;
                    }
                    MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, n, type, MPI_SUM, FML::communicator);
                    offset += n;
                }
#else
            (void)values;
            (void)reduction;
#endif
        }
        inline int threads() {
#ifdef USE_OMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        }
        inline int thread() {
#ifdef USE_OMP
            return omp_get_thread_num();
#else
            return 0;
#endif
        }
        inline bool radii_valid(const std::vector<double> & radii, bool positive = false) {
            if (radii.empty())
                return false;
            for (size_t i = 0; i < radii.size(); ++i)
                if (!std::isfinite(radii[i] * radii[i]) || (positive ? radii[i] <= 0 : radii[i] < 0) ||
                    (i && radii[i] <= radii[i - 1]))
                    return false;
            return true;
        }
        inline bool edges_valid(const std::vector<double> & edges) {
            if (edges.empty())
                return true;
            if (edges.size() < 2)
                return false;
            for (size_t i = 0; i < edges.size(); ++i)
                if (!std::isfinite(edges[i]) || (i && edges[i] <= edges[i - 1]))
                    return false;
            return true;
        }
        struct Accumulator {
            Count n = 0, under = 0, over = 0, undefined = 0;
            double mean = 0, m2 = 0;
            std::vector<Count> histogram, density;
            std::vector<double> factorial;
            Accumulator(size_t bins, size_t dbins, unsigned order)
                : histogram(bins), density(dbins), factorial(order + 1) {}
            void add(size_t count, double expected, const Options & options) {
                ++n;
                const double delta = double(count) - mean;
                mean += delta / double(n);
                m2 += delta * (double(count) - mean);
                ++histogram[std::min(count, histogram.size() - 1)];
                double moment = 1;
                factorial[0] += 1;
                for (size_t q = 1; q < factorial.size(); ++q) {
                    moment *= count >= q ? double(count - q + 1) : 0;
                    factorial[q] += moment;
                }
                if (density.empty())
                    return;
                if (!(expected > 0)) {
                    ++undefined;
                    return;
                }
                const double contrast = double(count) / expected - 1;
                if (contrast < options.delta_edges.front())
                    ++under;
                else if (contrast >= options.delta_edges.back())
                    ++over;
                else
                    ++density[size_t(
                        std::upper_bound(options.delta_edges.begin(), options.delta_edges.end(), contrast) -
                        options.delta_edges.begin() - 1)];
            }
            void merge(const Accumulator & other) {
                if (other.n) {
                    const double total = double(n) + other.n, delta = other.mean - mean;
                    m2 += other.m2 + delta * delta * double(n) * double(other.n) / total;
                    mean += delta * double(other.n) / total;
                    n += other.n;
                }
                for (size_t i = 0; i < histogram.size(); ++i)
                    histogram[i] += other.histogram[i];
                for (size_t i = 0; i < density.size(); ++i)
                    density[i] += other.density[i];
                for (size_t i = 0; i < factorial.size(); ++i)
                    factorial[i] += other.factorial[i];
                under += other.under;
                over += other.over;
                undefined += other.undefined;
            }
        };
    } // namespace detail
    /// Query-ID-keyed uniform volume samples. Distribute disjoint global ID ranges across MPI ranks.
    inline std::vector<Query> uniform_queries(size_t number,
                                              const Bounds & bounds,
                                              std::uint64_t seed,
                                              std::uint64_t first_id = 0,
                                              std::uint32_t stream = 0) {
        if (!detail::valid_bounds(bounds) ||
            (number && number - 1 > std::numeric_limits<std::uint64_t>::max() - first_id))
            throw std::invalid_argument("Invalid query bounds or overflowing global query IDs");
        std::vector<Query> result(number);
        for (size_t i = 0; i < number; ++i) {
            FML::RANDOM::COUNTER_BASED::Stream rng(seed, first_id + i, 0x51555259, stream);
            for (int d = 0; d < 3; ++d)
                result[i].position[d] = std::min(std::nextafter(bounds.upper[d], bounds.lower[d]),
                                                 bounds.lower[d] + rng.uniform() * (bounds.upper[d] - bounds.lower[d]));
        }
        return result;
    }
    /// Own an existing KDTree representation. Tracers are replicated; only query samples are distributed.
    /// Construction/raw distance queries are local. Global summaries are collective, including empty query ranks.
    class SpatialStatistics {
        FML::KDTREE::KDTree tree;
        std::uint64_t fingerprint = 1469598103934665603ULL;
        Bounds tracer_extent{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
        double box = 0;
        void add_point(const Point & p) {
            if (!detail::finite_point(p))
                throw std::invalid_argument("Invalid tracer position");
            for (int d = 0; d < 3; ++d) {
                detail::hash_double(fingerprint, p[d]);
                tracer_extent.lower[d] = std::min(tracer_extent.lower[d], p[d]);
                tracer_extent.upper[d] = std::max(tracer_extent.upper[d], p[d]);
            }
        }
        void validate_box() {
            if (!std::isfinite(box) || box < 0 ||
                (box > 0 && (!(box * box * box > 0) || !std::isfinite(box * box * box))))
                throw std::invalid_argument("Invalid periodic box size");
            detail::hash_double(fingerprint, box);
        }
        bool valid_queries(const std::vector<Query> & queries) const {
            for (const auto & q : queries)
                if (!detail::finite_point(q.position) ||
                    (q.exclude != std::numeric_limits<size_t>::max() && q.exclude >= tree.size()))
                    return false;
            return true;
        }
        void
        begin_summary(bool valid, const std::vector<Query> & queries, std::uint64_t schema, Reduction reduction) const {
            detail::check(valid && valid_queries(queries),
                          reduction,
                          "Invalid spatial query, radius/order or summary options");
            detail::same(fingerprint, reduction);
            detail::same(schema, reduction);
            detail::check(reduction == Reduction::Local ||
                              queries.size() <= std::numeric_limits<Count>::max() / size_t(FML::NTasks),
                          reduction,
                          "Global query count can overflow");
        }
        std::vector<double> squared_distances(const Query & query, const std::vector<size_t> & orders) const {
            std::vector<double> result(orders.size(), INFINITY);
            const bool exclude = query.exclude != std::numeric_limits<size_t>::max();
            const size_t available = tree.size() - size_t(exclude);
            size_t wanted = 0;
            for (size_t k : orders)
                if (k <= available)
                    wanted = std::max(wanted, k);
            if (!wanted)
                return result;
            std::vector<double> d2;
            const FML::KDTREE::point_t point(query.position.begin(), query.position.end());
            const auto indices = tree.knearest_indices(point, std::min(tree.size(), wanted + size_t(exclude)), &d2);
            size_t used = 0;
            for (size_t i = 0; i < indices.size(); ++i)
                if (indices[i] != query.exclude) {
                    ++used;
                    for (size_t k = 0; k < orders.size(); ++k)
                        if (orders[k] == used)
                            result[k] = d2[i];
                }
            return result;
        }
        std::vector<double> sorted_neighbors(const Query & query, double radius) const {
            if (tree.size() == 0)
                return {};
            std::vector<size_t> ids;
            std::vector<double> d2;
            tree.neighborhood_indices_and_distances({query.position.begin(), query.position.end()}, radius, ids, d2);
            size_t keep = 0;
            for (size_t i = 0; i < ids.size(); ++i)
                if (ids[i] != query.exclude)
                    d2[keep++] = d2[i];
            d2.resize(keep);
            std::sort(d2.begin(), d2.end());
            return d2;
        }
        static bool valid_orders(const std::vector<size_t> & orders) {
            if (orders.empty())
                return false;
            for (size_t i = 0; i < orders.size(); ++i)
                if (!orders[i] || std::find(orders.begin(), orders.begin() + i, orders[i]) != orders.begin() + i)
                    return false;
            return true;
        }

      public:
        explicit SpatialStatistics(const FML::KDTREE::pointVec & tracers, double boxsize = 0, int leaf = 8)
            : box(boxsize) {
            validate_box();
            for (const auto & p : tracers) {
                if (p.size() != 3)
                    throw std::invalid_argument("Spatial statistics require 3D tracers");
                add_point({p[0], p[1], p[2]});
            }
            tree = FML::KDTREE::KDTree(tracers, box, leaf);
            detail::hash_word(fingerprint, tracers.size());
        }
        template <class Particle>
        SpatialStatistics(Particle * tracers, size_t number, double boxsize = 0, int leaf = 8) : box(boxsize) {
            validate_box();
            if (FML::PARTICLE::GetNDIM(Particle()) != 3 || (number && !tracers))
                throw std::invalid_argument("Spatial statistics require valid 3D particles");
            for (size_t i = 0; i < number; ++i) {
                auto p = FML::PARTICLE::GetPos(tracers[i]);
                add_point({double(p[0]), double(p[1]), double(p[2])});
            }
            tree = FML::KDTREE::KDTree(tracers, number, box, leaf);
            detail::hash_word(fingerprint, number);
        }
        size_t size() const { return tree.size(); }
        KnnDistances knn_distances(const std::vector<Query> & queries, const std::vector<size_t> & orders) const {
            detail::check(valid_queries(queries) && valid_orders(orders), Reduction::Local, "Invalid kNN query/orders");
            KnnDistances out;
            out.orders = orders;
            out.distances.assign(orders.size(), std::vector<double>(queries.size()));
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < queries.size(); ++i) {
                auto d = squared_distances(queries[i], orders);
                for (size_t k = 0; k < orders.size(); ++k)
                    out.distances[k][i] = std::sqrt(d[k]);
            }
            return out;
        }
        KnnCDF knn_cdf(const std::vector<Query> & queries,
                       const std::vector<size_t> & orders,
                       const std::vector<double> & radii,
                       Reduction reduction = Reduction::Global) const {
            std::uint64_t schema = 25;
            for (auto k : orders)
                detail::hash_word(schema, k);
            detail::hash_word(schema, orders.size());
            for (double r : radii)
                detail::hash_double(schema, r);
            detail::hash_word(schema, radii.size());
            begin_summary(valid_orders(orders) && detail::radii_valid(radii), queries, schema, reduction);
            KnnCDF out;
            out.orders = orders;
            out.radii = radii;
            std::vector<std::vector<std::vector<Count>>> hits(
                size_t(detail::threads()),
                std::vector<std::vector<Count>>(orders.size(), std::vector<Count>(radii.size())));
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < queries.size(); ++i) {
                const auto d = squared_distances(queries[i], orders);
                auto & counts = hits[detail::thread()];
                for (size_t k = 0; k < orders.size(); ++k)
                    for (size_t r = 0; r < radii.size(); ++r)
                        if (d[k] <= radii[r] * radii[r])
                            ++counts[k][r];
            }
            out.hits.assign(orders.size(), std::vector<Count>(radii.size()));
            for (const auto & h : hits)
                for (size_t k = 0; k < orders.size(); ++k)
                    for (size_t r = 0; r < radii.size(); ++r)
                        out.hits[k][r] += h[k][r];
            Count excluded = 0;
            for (const auto & q : queries)
                excluded += q.exclude != std::numeric_limits<size_t>::max();
            std::vector<Count> n{Count(queries.size()), excluded};
            detail::sum(n, reduction);
            out.queries = n[0];
            out.excluded_queries = n[1];
            out.boxsize = box;
            out.cdf.assign(orders.size(), std::vector<double>(radii.size()));
            for (size_t k = 0; k < orders.size(); ++k) {
                detail::sum(out.hits[k], reduction);
                for (size_t r = 0; r < radii.size(); ++r)
                    out.cdf[k][r] = out.queries ? double(out.hits[k][r]) / double(out.queries) : detail::nan();
            }
            return out;
        }
        /// Raw toroidal/Euclidean sphere counts, inclusive boundary. Any finite nonnegative radius is allowed.
        std::vector<std::vector<size_t>> sphere_counts(const std::vector<Query> & queries,
                                                       const std::vector<double> & radii) const {
            detail::check(valid_queries(queries) && detail::radii_valid(radii),
                          Reduction::Local,
                          "Invalid sphere-count query/radii");
            std::vector<std::vector<size_t>> result(radii.size(), std::vector<size_t>(queries.size()));
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < queries.size(); ++i) {
                const auto d = sorted_neighbors(queries[i], radii.back());
                for (size_t r = 0; r < radii.size(); ++r)
                    result[r][i] = std::upper_bound(d.begin(), d.end(), radii[r] * radii[r]) - d.begin();
            }
            return result;
        }
        CountsInCells counts_in_cells(const std::vector<Query> & queries,
                                      const std::vector<double> & radii,
                                      const Options & options = {}) const {
            bool valid = detail::radii_valid(radii, true) && detail::edges_valid(options.delta_edges) &&
                         options.factorial_order <= 16 &&
                         options.histogram_limit <= std::vector<Count>().max_size() - 2;
            if (options.number_density)
                valid &= *options.number_density >= 0 && std::isfinite(*options.number_density);
            if (options.window)
                valid &= detail::valid_bounds(*options.window);
            if (box > 0 && !radii.empty())
                valid &= radii.back() <= .5 * box;
            if (box == 0 && !options.allow_truncated_cells) {
                valid &= bool(options.window);
                if (options.window && !radii.empty())
                    for (const auto & q : queries)
                        for (int d = 0; d < 3; ++d)
                            valid &= q.position[d] - radii.back() >= options.window->lower[d] &&
                                     q.position[d] + radii.back() <= options.window->upper[d];
            }
            if (box == 0 && options.window && tree.size())
                for (int d = 0; d < 3; ++d)
                    valid &= tracer_extent.lower[d] >= options.window->lower[d] &&
                             tracer_extent.upper[d] <= options.window->upper[d];
            valid &= box > 0 || bool(options.window) || bool(options.number_density);
            for (double r : radii)
                valid &= (4.1887902047863909846 * r * r * r > 0) && std::isfinite(4.1887902047863909846 * r * r * r);
            std::uint64_t schema = 30;
            for (double r : radii)
                detail::hash_double(schema, r);
            detail::hash_word(schema, radii.size());
            detail::hash_word(schema, options.histogram_limit);
            detail::hash_word(schema, options.factorial_order);
            for (double edge : options.delta_edges)
                detail::hash_double(schema, edge);
            detail::hash_word(schema, options.delta_edges.size());
            detail::hash_word(schema, bool(options.number_density));
            if (options.number_density)
                detail::hash_double(schema, *options.number_density);
            detail::hash_word(schema, bool(options.window));
            if (options.window)
                for (int d = 0; d < 3; ++d) {
                    detail::hash_double(schema, options.window->lower[d]);
                    detail::hash_double(schema, options.window->upper[d]);
                }
            detail::hash_word(schema, options.allow_truncated_cells);
            begin_summary(valid, queries, schema, options.reduction);
            const double density = options.number_density ?
                                       *options.number_density :
                                       double(tree.size()) / (box > 0 ? box * box * box : options.window->volume());
            detail::check(std::isfinite(density),
                          options.reduction,
                          "Reference number density exceeds finite arithmetic");
            for (double r : radii)
                detail::check(std::isfinite(density * 4.1887902047863909846 * r * r * r),
                              options.reduction,
                              "Reference expected count exceeds finite arithmetic");
            CountsInCells result;
            std::vector<Count> excluded{0};
            for (const auto & q : queries)
                excluded[0] += q.exclude != std::numeric_limits<size_t>::max();
            detail::sum(excluded, options.reduction);
            result.excluded_queries = excluded[0];
            result.boxsize = box;
            result.uncorrected_clipping = box == 0 && options.allow_truncated_cells;
            result.delta_edges = options.delta_edges;
            result.histogram_limit = options.histogram_limit;
            const size_t dbins = options.delta_edges.empty() ? 0 : options.delta_edges.size() - 1;
            std::vector<std::vector<detail::Accumulator>> workers(static_cast<size_t>(detail::threads()));
            for (auto & worker : workers)
                for (size_t r = 0; r < radii.size(); ++r)
                    worker.emplace_back(options.histogram_limit + 2, dbins, options.factorial_order);
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
            for (size_t i = 0; i < queries.size(); ++i) {
                const auto neighbors = sorted_neighbors(queries[i], radii.back());
                auto & worker = workers[detail::thread()];
                for (size_t r = 0; r < radii.size(); ++r) {
                    const size_t count =
                        std::upper_bound(neighbors.begin(), neighbors.end(), radii[r] * radii[r]) - neighbors.begin();
                    worker[r].add(count, density * 4.1887902047863909846 * radii[r] * radii[r] * radii[r], options);
                }
            }
            for (size_t r = 0; r < radii.size(); ++r) {
                detail::Accumulator aggregate(options.histogram_limit + 2, dbins, options.factorial_order);
                for (const auto & worker : workers)
                    aggregate.merge(worker[r]);
                std::vector<Count> numbers{aggregate.n, aggregate.under, aggregate.over, aggregate.undefined};
                detail::sum(numbers, options.reduction);
                const Count n = numbers[0];
                std::vector<double> mean_sum{double(aggregate.n) * aggregate.mean};
                detail::sum(mean_sum, options.reduction);
                const double mean = n ? mean_sum[0] / double(n) : detail::nan();
                std::vector<double> m2{aggregate.m2 +
                                       (aggregate.n ? double(aggregate.n) * std::pow(aggregate.mean - mean, 2) : 0)};
                detail::sum(m2, options.reduction);
                detail::sum(aggregate.histogram, options.reduction);
                detail::sum(aggregate.density, options.reduction);
                detail::sum(aggregate.factorial, options.reduction);
                CellStatistics c;
                c.radius = radii[r];
                c.volume = 4.1887902047863909846 * c.radius * c.radius * c.radius;
                c.reference_number_density = density;
                c.queries = n;
                c.histogram = std::move(aggregate.histogram);
                c.density_histogram = std::move(aggregate.density);
                c.density_underflow = numbers[1];
                c.density_overflow = numbers[2];
                c.density_undefined = numbers[3];
                c.mean = mean;
                c.variance = n ? m2[0] / double(n) : detail::nan();
                c.mean_delta = density * c.volume > 0 ? mean / (density * c.volume) - 1 : detail::nan();
                c.poisson_subtracted_variance = c.variance - mean;
                c.normalized_connected_variance =
                    mean > 0 ? c.poisson_subtracted_variance / (mean * mean) : detail::nan();
                for (auto value : c.histogram)
                    c.probability.push_back(n ? double(value) / double(n) : detail::nan());
                for (size_t j = 0; j < c.density_histogram.size(); ++j) {
                    const double p = n ? double(c.density_histogram[j]) / double(n) : detail::nan();
                    c.density_probability.push_back(p);
                    c.density_pdf.push_back(p / (options.delta_edges[j + 1] - options.delta_edges[j]));
                }
                for (size_t q = 0; q < aggregate.factorial.size(); ++q) {
                    const double f = n ? aggregate.factorial[q] / double(n) : detail::nan();
                    c.factorial_moments.push_back(f);
                    c.normalized_factorial_moments.push_back(
                        q == 0 ? f : (mean > 0 ? f / std::pow(mean, double(q)) : detail::nan()));
                }
                result.cells.push_back(std::move(c));
            }
            return result;
        }
    };
} // namespace FML::COUNTS_IN_CELLS
#endif
