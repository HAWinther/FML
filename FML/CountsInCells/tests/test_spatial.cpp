#include <FML/CountsInCells/SpatialStatistics.h>
#include <FML/HOD/HOD.h>
#include <iostream>
using namespace FML::COUNTS_IN_CELLS;
int failures = 0;
void check(bool good, const std::string& label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double a, double b, double tolerance = 1e-12) {
    if (std::isnan(a) || std::isnan(b))
        return std::isnan(a) && std::isnan(b);
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}
std::vector<Query> local_queries(const std::vector<Query>& all, bool root = false) {
    std::vector<Query> q;
    for (size_t i = 0; i < all.size(); ++i)
        if ((root ? 0 : int(i % FML::NTasks)) == FML::ThisTask)
            q.push_back(all[i]);
    return q;
}
std::vector<double> brute(const FML::KDTREE::pointVec& points, const Query& q, double box) {
    std::vector<double> result;
    for (size_t i = 0; i < points.size(); ++i)
        if (i != q.exclude) {
            double r2 = 0;
            for (int d = 0; d < 3; ++d) {
                double dx = points[i][d] - q.position[d];
                if (box > 0)
                    dx -= box * std::round(dx / box);
                r2 += dx * dx;
            }
            result.push_back(r2);
        }
    std::sort(result.begin(), result.end());
    return result;
}
void test_exact() {
    const FML::KDTREE::pointVec tracers{{0, 0, 0},       {.5, 0, 0},   {.999, .1, .2}, {.25, .25, .25},
                                        {.25, .25, .25}, {.7, .4, .9}, {.1, .7, .8},   {.3, .2, .6}};
    const std::vector<Query> all{{{0, 0, 0}},          {{0, 0, 0}, 0}, {{1.001, .1, .2}},
                                 {{.25, .25, .25}, 3}, {{.5, .5, .5}}, {{.25, .25, .25}}};
    const std::vector<size_t> orders{4, 1, 2, 100};
    const std::vector<double> radii{0, .05, .1, .25, .5, .9};
    for (double box : {0.0, 1.0}) {
        SpatialStatistics statistics(tracers, box);
        auto queries = local_queries(all);
        auto batch = statistics.knn_distances(queries, orders);
        const auto spheres = statistics.sphere_counts(queries, radii);
        bool exact = true;
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto ref = brute(tracers, queries[i], box);
            for (size_t k = 0; k < orders.size(); ++k) {
                const double expected = orders[k] <= ref.size() ? std::sqrt(ref[orders[k] - 1]) : INFINITY;
                exact &=
                    std::isinf(expected) ? std::isinf(batch.distances[k][i]) : near(batch.distances[k][i], expected);
            }
            for (size_t r = 0; r < radii.size(); ++r)
                exact &= spheres[r][i] ==
                         size_t(std::upper_bound(ref.begin(), ref.end(), radii[r] * radii[r]) - ref.begin());
        }
        check(exact, std::string("batched distances and inclusive sphere counts match brute force: ") +
                         (box ? "periodic" : "nonperiodic"));
        auto cdf = statistics.knn_cdf(queries, orders, radii);
        bool fractions = cdf.queries == all.size() && cdf.excluded_queries == 2 && cdf.boxsize == box;
        for (size_t k = 0; k < orders.size(); ++k)
            for (size_t r = 0; r < radii.size(); ++r) {
                Count expected = 0;
                for (const auto& q : all) {
                    auto d2 = brute(tracers, q, box);
                    expected += orders[k] <= d2.size() && d2[orders[k] - 1] <= radii[r] * radii[r];
                }
                fractions &= cdf.hits[k][r] == expected && near(cdf.cdf[k][r], double(expected) / all.size());
            }
        check(fractions, "global kNN CDFs retain missing-neighbor queries in the denominator");
    }
    SpatialStatistics statistics(tracers, 1);
    Options options;
    options.histogram_limit = 2;
    options.factorial_order = 5;
    const std::vector<double> radii2{.05, .1, .25, .5};
    auto summary = statistics.counts_in_cells(local_queries(all, true), radii2, options);
    const auto cdf = statistics.knn_cdf(local_queries(all), {1, 2, 3}, radii2);
    bool exact = summary.excluded_queries == 2 && !summary.uncorrected_clipping, relation = true;
    for (size_t r = 0; r < radii2.size(); ++r) {
        const auto& actual = summary.cells[r];
        double sum = 0, square = 0;
        std::vector<double> factorial(6);
        std::vector<Count> histogram(4), density(7);
        Count under = 0, over = 0;
        for (const auto& q : all) {
            auto d2 = brute(tracers, q, 1);
            const size_t count = std::upper_bound(d2.begin(), d2.end(), radii2[r] * radii2[r]) - d2.begin();
            sum += count;
            square += double(count) * count;
            ++histogram[std::min(size_t(3), count)];
            double falling = 1;
            factorial[0] += 1;
            for (size_t order = 1; order < factorial.size(); ++order) {
                falling *= count >= order ? double(count - order + 1) : 0;
                factorial[order] += falling;
            }
            const double delta = count / (double(tracers.size()) * 4 * M_PI / 3 * std::pow(radii2[r], 3)) - 1;
            if (delta < options.delta_edges.front())
                ++under;
            else if (delta >= options.delta_edges.back())
                ++over;
            else
                ++density[size_t(std::upper_bound(options.delta_edges.begin(), options.delta_edges.end(), delta) -
                                 options.delta_edges.begin() - 1)];
        }
        const double mean = sum / all.size(), variance = square / all.size() - mean * mean;
        exact &= actual.queries == all.size() && actual.histogram == histogram && actual.density_histogram == density &&
                 actual.density_underflow == under && actual.density_overflow == over && near(actual.mean, mean) &&
                 near(actual.variance, variance) && near(actual.poisson_subtracted_variance, variance - mean);
        for (size_t q = 0; q < factorial.size(); ++q) {
            const double expected = factorial[q] / all.size();
            exact &= near(actual.factorial_moments[q], expected) &&
                     near(actual.normalized_factorial_moments[q],
                          q ? (mean > 0 ? expected / std::pow(mean, double(q)) : NAN) : 1);
        }
        double density_mass = double(under + over) / all.size();
        for (size_t b = 0; b < density.size(); ++b) {
            exact &= near(actual.density_probability[b], double(density[b]) / all.size());
            density_mass += actual.density_pdf[b] * (options.delta_edges[b + 1] - options.delta_edges[b]);
        }
        exact &= near(density_mass, 1) && near(actual.probability[3], double(histogram[3]) / all.size());
        relation &= near(actual.probability[0], 1 - cdf.cdf[0][r]) &&
                    near(actual.probability[1], cdf.cdf[0][r] - cdf.cdf[1][r]) &&
                    near(actual.probability[2], cdf.cdf[1][r] - cdf.cdf[2][r]);
    }
    check(exact, "count/density PDFs, overflow bins, stable variance and factorial moments match direct counts");
    check(relation, "counts-in-cells probabilities equal consecutive kNN-CDF differences");
    const auto duplicates = statistics.sphere_counts({{{.25, .25, .25}, 3}}, {0});
    check(duplicates[0][0] == 1, "self-exclusion removes only the indexed tracer, retaining coincident other tracers");
#ifdef USE_OMP
    auto q = uniform_queries(1000, Bounds{}, 42);
    omp_set_num_threads(1);
    auto one = statistics.counts_in_cells(q, {.1, .2}, options);
    omp_set_num_threads(4);
    auto four = statistics.counts_in_cells(q, {.1, .2}, options);
    bool stable = true;
    for (size_t r = 0; r < one.cells.size(); ++r)
        stable &= one.cells[r].histogram == four.cells[r].histogram &&
                  near(one.cells[r].variance, four.cells[r].variance) &&
                  near(one.cells[r].factorial_moments[3], four.cells[r].factorial_moments[3]);
    check(stable, "thread-local summaries preserve exact histograms and stable floating moments");
    omp_set_num_threads(2);
#endif
}
void test_windows_and_empty() {
    SpatialStatistics empty(FML::KDTREE::pointVec{}, 1);
    const std::vector<Query> q{{{.3, .4, .5}}};
    auto cdf = empty.knn_cdf(local_queries(q), {1, 4}, {0, .1, .5});
    auto summary = empty.counts_in_cells(local_queries(q), {.1});
    check(cdf.queries == 1 && cdf.cdf[0][2] == 0 && summary.cells[0].probability[0] == 1 &&
              summary.cells[0].density_undefined == 1 && std::isnan(summary.cells[0].normalized_factorial_moments[1]),
          "empty tracer catalogue gives zero counts and explicitly undefined normalized density");
    auto noquery = empty.counts_in_cells({}, {.1});
    auto nocdf = empty.knn_cdf({}, {1}, {.1});
    check(noquery.cells[0].queries == 0 && std::isnan(noquery.cells[0].mean) && std::isnan(nocdf.cdf[0][0]),
          "zero global query sample is reported as undefined, not a zero-probability measurement");
    SpatialStatistics bounded(FML::KDTREE::pointVec{{.5, .5, .5}, {.6, .5, .5}}, 0);
    Options options;
    options.window = Bounds{};
    auto full = bounded.counts_in_cells(local_queries(q), {.1}, options);
    check(full.cells[0].queries == 1 && near(full.cells[0].reference_number_density, 2),
          "nonperiodic full-sphere summaries use an explicit rectangular observation window");
    options.allow_truncated_cells = true;
    const std::vector<Query> edge{{{.01, .01, .01}}};
    auto clipped = bounded.counts_in_cells(local_queries(edge), {.2}, options);
    check(clipped.cells[0].queries == 1 && clipped.uncorrected_clipping,
          "boundary-clipped spheres require an explicit uncorrected opt-in");
    const auto whole = uniform_queries(41, Bounds{}, 919, 100), a = uniform_queries(13, Bounds{}, 919, 100),
               b = uniform_queries(28, Bounds{}, 919, 113);
    bool sequence = true;
    for (size_t i = 0; i < whole.size(); ++i) {
        const auto& part = i < a.size() ? a[i] : b[i - a.size()];
        sequence &= whole[i].position == part.position;
        for (double x : whole[i].position)
            sequence &= x >= 0 && x < 1;
    }
    check(sequence, "uniform query IDs replay exactly across batch/MPI partitions");
}
void test_binomial() {
    const int n = 8000, nq = 40000;
    FML::KDTREE::pointVec points;
    points.reserve(n);
    auto uniform = uniform_queries(n, Bounds{}, 671);
    for (const auto& q : uniform)
        points.push_back({q.position.begin(), q.position.end()});
    SpatialStatistics statistics(points, 1);
    const size_t start = size_t(nq) * FML::ThisTask / FML::NTasks,
                 stop = size_t(nq) * (FML::ThisTask + 1) / FML::NTasks;
    const auto q = uniform_queries(stop - start, Bounds{}, 913, start);
    const double lambda = 2, r = std::cbrt(lambda / (n * 4 * M_PI / 3));
    Options options;
    auto counts = statistics.counts_in_cells(q, {r}, options);
    const auto cdf = statistics.knn_cdf(q, {1, 2}, {r});
    const auto& c = counts.cells[0];
    bool moments = std::abs(c.mean - lambda) < .06 && std::abs(c.normalized_connected_variance + 1.0 / n) < .04;
    for (size_t order = 2; order <= 4; ++order) {
        double expected = 1;
        for (size_t j = 0; j < order; ++j)
            expected *= 1 - double(j) / n;
        moments &= std::abs(c.normalized_factorial_moments[order] - expected) < (.05 * order);
    }
    const double p = lambda / n, p0 = std::pow(1 - p, n), p1 = n * p * std::pow(1 - p, n - 1);
    check(moments, "uniform binomial-process count moments approach the Poisson intensity moments");
    check(std::abs(cdf.cdf[0][0] - (1 - p0)) < .025 && std::abs(cdf.cdf[1][0] - (1 - p0 - p1)) < .025,
          "random-query kNN CDF agrees with the analytic binomial/Poisson limit");
}
void test_mock_integration() {
    std::vector<FML::HOD::Halo> halos;
    for (int i = 0; i < 20; ++i) {
        FML::HOD::Halo h;
        h.id = i;
        h.radius = .01;
        h.position = {.02 + .04 * i, .4, .5};
        halos.push_back(h);
    }
    FML::HOD::Options options;
    options.boxsize = 1;
    auto mock = FML::HOD::populate(
        FML::HOD::PreparedCatalog(std::move(halos)),
        [](const auto&) {
            return FML::HOD::OccupationMeans{1, 2};
        },
        options);
    SpatialStatistics stats(mock.galaxies.data(), mock.galaxies.size(), 1);
    const auto q = local_queries({{{.2, .4, .5}}, {{.5, .5, .5}}});
    const auto cdf = stats.knn_cdf(q, {1}, {.1});
    const auto counts = stats.counts_in_cells(q, {.1});
    check(cdf.queries == 2 && counts.cells[0].queries == 2 && near(1 - cdf.cdf[0][0], counts.cells[0].probability[0]),
          "HOD galaxies feed kNN and counts-in-cells through the standard particle interface");
}
void test_local_only() {
    bool local = true;
    if (FML::ThisTask == 0) {
        SpatialStatistics s(FML::KDTREE::pointVec{{.1, .2, .3}}, 1);
        const std::vector<Query> q{{{.1, .2, .3}, 0}};
        Options o;
        o.reduction = Reduction::Local;
        const auto c = s.counts_in_cells(q, {.1}, o);
        const auto f = s.knn_cdf(q, {1}, {.1}, Reduction::Local);
        local = c.cells[0].queries == 1 && c.excluded_queries == 1 && c.cells[0].probability[0] == 1 &&
                f.queries == 1 && f.cdf[0][0] == 0;
    }
    check(local, "local summaries can run on rank zero alone without hidden MPI collectives");
}
void test_invalid() {
    auto rejects = [](auto operation) {
        try {
            operation();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    SpatialStatistics stats(FML::KDTREE::pointVec{{.1, .2, .3}}, 1);
    auto q = local_queries({{{.2, .3, .4}}});
    check(rejects([&] { stats.knn_cdf(q, {0}, {.1}); }) && rejects([&] {
              stats.knn_cdf(q, {1, 1}, {.1});
          }),
          "zero/duplicate kNN orders are rejected collectively");
    check(rejects([&] { stats.counts_in_cells(q, {.6}); }) && rejects([&] { stats.counts_in_cells(q, {0}); }),
          "density cell radii must have positive standard spherical volume without torus overlap");
    if (FML::ThisTask == 0)
        q.push_back({{NAN, .2, .3}});
    check(rejects([&] { stats.knn_cdf(q, {1}, {.1}); }), "invalid query on one rank fails collectively");
    SpatialStatistics nonperiodic(FML::KDTREE::pointVec{{.1, .2, .3}});
    const std::vector<Query> edge{{{.01, .01, .01}}};
    Options options;
    options.window = Bounds{};
    check(rejects([&] { nonperiodic.counts_in_cells(edge, {.1}, options); }) &&
              rejects([&] { nonperiodic.counts_in_cells(edge, {.1}); }),
          "unconfigured or partially clipped nonperiodic windows are rejected");
    if (FML::NTasks > 1) {
        Options mismatch;
        mismatch.histogram_limit = 10 + FML::ThisTask;
        check(rejects([&] { stats.counts_in_cells({}, {.1}, mismatch); }),
              "different summary schemas across ranks fail before reducing arrays");
        SpatialStatistics different(FML::KDTREE::pointVec{{.1 + .01 * FML::ThisTask, .2, .3}}, 1);
        check(rejects([&] { different.knn_cdf({}, {1}, {.1}); }),
              "global reductions reject differing tracer catalogue fingerprints");
    }
}
int main() {
    test_exact();
    test_windows_and_empty();
    test_binomial();
    test_mock_integration();
    test_local_only();
    test_invalid();
    if (FML::ThisTask == 0)
        std::cout << (failures ? "Spatial statistics tests FAILED\n" : "All spatial statistics tests passed\n");
    return failures ? 1 : 0;
}
