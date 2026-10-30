#include <FML/Environment/EnvironmentClustering.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <iostream>
using namespace FML::ENVIRONMENT;
int failures = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double x, double y, double tolerance = 2e-10) {
    return std::abs(x - y) <= tolerance * std::max({1., std::abs(x), std::abs(y)});
}
double distance(const Sample & a, const Sample & b, double L) {
    double r2 = 0;
    for (int d = 0; d < 3; ++d) {
        double x = std::abs(a.position[d] - b.position[d]);
        x = std::min(x, L - x);
        r2 += x * x;
    }
    return std::sqrt(r2);
}
std::vector<double> pairs(const std::vector<Sample> & a,
                          const std::vector<Sample> & b,
                          double L,
                          double low,
                          double high,
                          int bins,
                          bool auto_pairs = false,
                          const std::vector<double> & mark = {}) {
    std::vector<double> out(bins);
    double mean = 1;
    if (!mark.empty())
        mean = std::accumulate(mark.begin(), mark.end(), 0.) / mark.size();
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = auto_pairs ? i + 1 : 0; j < b.size(); ++j) {
            double r = distance(a[i], b[j], L);
            if (r < low || r >= high || r == 0)
                continue;
            int bin = std::min(bins - 1, int((r - low) / (high - low) * bins));
            out[bin] += mark.empty() ? 1 : mark[i] * mark[j] / (mean * mean);
        }
    return out;
}
std::vector<Sample> random_points(size_t n, double L, std::uint64_t seed) {
    std::vector<Sample> p(n);
    for (size_t i = 0; i < n; ++i) {
        FML::RANDOM::COUNTER_BASED::Stream rng(seed, i);
        p[i].id = i + 100;
        for (double & x : p[i].position)
            x = L * rng.uniform();
    }
    return p;
}
void test_quantiles() {
    const std::vector<double> d{2, 1, 1, 3, 1, 0, 2};
    const std::vector<std::uint64_t> ids{10, 9, 8, 7, 6, 5, 4};
    auto q = density_quantiles(d, ids, 3);
    check(q.counts == std::vector<size_t>({3, 2, 2}) && q.label == std::vector<size_t>({2, 1, 0, 2, 0, 0, 1}) &&
              q.minimum == std::vector<double>({0, 1, 2}) && q.maximum == std::vector<double>({1, 2, 3}),
          "quantile sizes and deterministic density/ID tie ranking match hand calculation");
    auto rd = d;
    auto ri = ids;
    std::reverse(rd.begin(), rd.end());
    std::reverse(ri.begin(), ri.end());
    auto r = density_quantiles(rd, ri, 3);
    std::reverse(r.label.begin(), r.label.end());
    check(r.label == q.label, "quantile assignment is independent of catalogue input order");
    auto tied = density_quantiles({1, 1, 1, 1}, {4, 2, 3, 1}, 2);
    check(tied.label == std::vector<size_t>({1, 0, 1, 0}), "equal-density ties are explicitly split by stable ID");
    bool rejected = false;
    try {
        density_quantiles({0, 1}, {2, 2}, 2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "duplicate quantile IDs rejected");
}
void test_density() {
    std::vector<Sample> p{{{.2, 2, 2}, 10}, {{9.8, 2, 2}, 11}, {{2, 2, 2}, 12}, {{5, 2, 2}, 13}};
    std::vector<Sample> q{{{0, 2, 2}, 100}, {{9.9, 2, 2}, 101}, {{2.5, 2, 2}, 102}};
    auto result = top_hat_environment(p, q, 10, .5);
    double f = 4 * M_PI / 3 * std::pow(.5 / 10, 3);
    check(result.counts == std::vector<Count>({2, 2, 1}) && near(result.delta[0], 2 / (4 * f) - 1) &&
              near(result.delta[2], 1 / (4 * f) - 1),
          "periodic top-hat counts, inclusive radius and overdensity match hand calculation");
    auto self = top_hat_environment(p, p, 10, .5, true);
    check(self.counts == std::vector<Count>({1, 1, 0, 0}) && near(self.delta[0], 1 / (3 * f) - 1),
          "same-ID self exclusion also uses the N-1 reference number density");
    auto a = random_points(43, 10, 1), b = random_points(37, 10, 2);
    auto measured = top_hat_environment(a, b, 10, 1.7);
    bool ok = true;
    for (size_t i = 0; i < b.size(); ++i) {
        size_t n = 0;
        for (const auto & point : a)
            n += distance(point, b[i], 10) <= 1.7;
        ok &= measured.counts[i] == n;
    }
    check(ok, "KD-tree environments agree with independent minimum-image direct counts");
    auto empty = top_hat_environment(a, {}, 10, 1.7);
    check(empty.counts.empty() && empty.delta.empty(), "empty query catalogue handled collectively");
    auto wrong = p;
    wrong[0].position[0] += .01;
    bool rejected = false;
    try {
        top_hat_environment(p, wrong, 10, .5, true);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "self-exclusion ID/position mismatch rejected");
}
void test_marked() {
    auto p = random_points(39, 10, 3);
    std::vector<double> m(p.size());
    for (size_t i = 0; i < m.size(); ++i)
        m[i] = .5 + double(i % 7);
    auto measured = marked_correlation(p, m, 10, .1, 5, 9);
    auto raw = pairs(p, p, 10, .1, 5, 9, true), weighted = pairs(p, p, 10, .1, 5, 9, true, m);
    bool ok = true;
    double norm = p.size() * (p.size() - 1) / 2.;
    for (int b = 0; b < 9; ++b) {
        double lo = (.1 + 4.9 * b / 9) / 10, hi = (.1 + 4.9 * (b + 1) / 9) / 10,
               volume = 4 * M_PI / 3 * (hi * hi * hi - lo * lo * lo);
        ok &= near(measured.unmarked.pairs[b], raw[b]) && near(measured.marked_pairs[b], weighted[b]) &&
              near(measured.unmarked.xi[b], raw[b] / (norm * volume) - 1) &&
              near(measured.weighted_xi[b], weighted[b] / (norm * volume) - 1);
        if (raw[b] > 0)
            ok &= near(measured.mark_ratio[b], weighted[b] / raw[b]) &&
                  near(measured.mark_ratio[b], (1 + measured.weighted_xi[b]) / (1 + measured.unmarked.xi[b]));
        else
            ok &= std::isnan(measured.mark_ratio[b]);
    }
    check(ok, "one-pass marked/unmarked pairs, xi and mark ratio match independent direct sums");
    auto constant = marked_correlation(p, std::vector<double>(p.size(), 7), 10, .1, 5, 9);
    auto rescaled = m;
    for (auto & x : rescaled)
        x *= 11;
    auto scale = marked_correlation(p, rescaled, 10, .1, 5, 9);
    ok = true;
    for (int b = 0; b < 9; ++b)
        if (raw[b] > 0)
            ok &= near(constant.mark_ratio[b], 1) && near(constant.pair_normalized_ratio[b], 1) &&
                  near(scale.mark_ratio[b], measured.mark_ratio[b]);
    check(ok, "unit/constant marks and global mark rescaling preserve normalization");
    std::vector<Sample> line{{{.05, .1, .1}, 1}, {{.15, .1, .1}, 2}, {{.3, .1, .1}, 3}, {{.45, .1, .1}, 4}};
    auto finite = marked_correlation(line, {1, 2, 3, 4}, 1, .001, .5, 1);
    check(near(finite.mean_mark, 2.5) && near(finite.mark_ratio[0], 35 / (6 * 6.25)) &&
              near(finite.finite_pair_baseline, 35 / (6 * 6.25)) && near(finite.pair_normalized_ratio[0], 1),
          "sample-mean and all-pair finite-population normalizations remain distinct");
    auto sparse = marked_correlation(line, {1, 0, 0, 0}, 1, .001, .5, 2);
    check(sparse.finite_pair_baseline == 0 && std::isnan(sparse.pair_normalized_ratio[0]),
          "single nonzero mark makes all-pair normalization explicitly undefined");
    std::vector<Sample> edges{{{0, .1, .1}, 1}, {{.125, .1, .1}, 2}, {{.25, .1, .1}, 3}, {{.5, .1, .1}, 4}};
    auto edge_result = marked_correlation(edges, {1, 2, 3, 4}, 1, .125, .5, 3);
    check(edge_result.unmarked.pairs == std::vector<double>({2, 2, 1}),
          "radial lower edges are included and final upper edge is excluded");
    auto enlarged = edges;
    for (auto & sample : enlarged)
        for (auto & x : sample.position)
            x *= 10;
    auto enlarged_result = marked_correlation(enlarged, {1, 2, 3, 4}, 10, 1.25, 5, 3);
    bool length_ok = true;
    for (int bin = 0; bin < 3; ++bin)
        length_ok &= near(enlarged_result.unmarked.radius[bin], 10 * edge_result.unmarked.radius[bin]) &&
                     near(enlarged_result.mark_ratio[bin], edge_result.mark_ratio[bin]) &&
                     near(enlarged_result.unmarked.xi[bin], edge_result.unmarked.xi[bin]);
    check(length_ok, "physical length rescaling changes radius while preserving dimensionless statistics");
    auto empty_bins = marked_correlation(line, {1, 1, 1, 1}, 1, .001, .02, 2);
    check(empty_bins.unmarked.pairs == std::vector<double>({0, 0}) &&
              empty_bins.unmarked.xi == std::vector<double>({-1, -1}) && std::isnan(empty_bins.mark_ratio[0]) &&
              std::isnan(empty_bins.pair_normalized_ratio[1]),
          "empty pair bins have xi=-1 and undefined mark ratios");
    auto recipe = inverse_density_marks({-1, 0, 1}, 1, 2);
    check(near(recipe[0], 4) && near(recipe[1], 1) && near(recipe[2], 4. / 9),
          "inverse-density mark recipe matches analytic values");
    auto original = p;
    std::reverse(p.begin(), p.end());
    std::reverse(m.begin(), m.end());
    auto reversed = marked_correlation(p, m, 10, .1, 5, 9);
    ok = true;
    for (int b = 0; b < 9; ++b)
        ok &= near(reversed.marked_pairs[b], measured.marked_pairs[b]);
    check(ok, "marked statistic is invariant under catalogue/mark reordering");
    bool rejected = false;
    try {
        marked_correlation(original, std::vector<double>(original.size(), 0), 10, .1, 5, 9);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "zero-mean marks rejected");
}
void test_split() {
    auto tracers = random_points(29, 10, 4), centers = random_points(41, 10, 5);
    auto env = top_hat_environment(tracers, centers, 10, 2);
    auto split = density_split_correlation(centers, env.delta, tracers, 5, 10, .1, 5, 7);
    bool ok = true;
    for (size_t q = 0; q < 5; ++q) {
        std::vector<Sample> selected;
        for (size_t i = 0; i < centers.size(); ++i)
            if (split.quantiles.label[i] == q)
                selected.push_back(centers[i]);
        auto direct = pairs(selected, tracers, 10, .1, 5, 7);
        for (int b = 0; b < 7; ++b) {
            double lo = (.1 + 4.9 * b / 7) / 10, hi = (.1 + 4.9 * (b + 1) / 7) / 10,
                   volume = 4 * M_PI / 3 * (hi * hi * hi - lo * lo * lo);
            ok &= near(split.cross[q].pairs[b], direct[b]) &&
                  near(split.cross[q].xi[b], direct[b] / (selected.size() * tracers.size() * volume) - 1);
        }
    }
    auto direct = pairs(centers, tracers, 10, .1, 5, 7);
    for (int b = 0; b < 7; ++b)
        ok &= near(split.parent.pairs[b], direct[b]);
    check(ok, "quantile cross-correlations and recombined parent counts match direct enumeration");
    double nonzero = 0;
    ok = true;
    for (int b = 0; b < 7; ++b) {
        double combined = 0;
        for (size_t q = 0; q < 5; ++q)
            combined += split.quantiles.counts[q] * split.cross[q].xi[b] / centers.size();
        ok &= near(combined, split.parent.xi[b]);
        nonzero += std::abs(combined);
    }
    check(ok && nonzero > 0, "weighted quantile sum equals the finite-query parent, without falsely forcing zero");
    std::vector<Sample> two{{{.1, .1, .1}, 1}, {{.8, .8, .8}, 2}};
    auto small = density_split_correlation(two, {0, 1}, two, 2, 1, .01, .5, 2);
    check(small.quantiles.counts == std::vector<size_t>({1, 1}),
          "one-center quantiles work with more MPI ranks than centers");
}
void test_failures() {
    auto p = random_points(8, 10, 6);
    std::vector<double> marks(p.size(), 1);
    bool rejected = false;
    try {
        marked_correlation(p, marks, 10, 0, 5, 4);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "zero separation bins rejected explicitly");
    rejected = false;
    try {
        marked_correlation(p, marks, 10, .1, 5.1, 4);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "shells beyond half the periodic box rejected");
    auto bad = p;
    if (!FML::ThisTask)
        bad[0].position[0] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try {
        marked_correlation(bad, marks, 10, .1, 5, 4);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "one-rank nonfinite catalogue fails collectively");
    if (FML::NTasks > 1) {
        auto divergent = p;
        divergent[0].position[0] += .001 * FML::ThisTask;
        rejected = false;
        try {
            top_hat_environment(divergent, p, 10, 1);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "different replicated catalogues rejected across ranks");
        auto density = marks;
        density[0] += FML::ThisTask;
        rejected = false;
        try {
            density_split_correlation(p, density, p, 2, 10, .1, 5, 4);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "different quantile densities rejected across ranks");
    }
}
int main() {
    test_quantiles();
    test_density();
    test_marked();
    test_split();
    test_failures();
    if (!FML::ThisTask)
        std::cout << (failures ? "Environment tests FAILED" : "All environment tests passed") << '\n';
    return failures ? 1 : 0;
}
