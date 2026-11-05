#include <FML/ThreePointCorrelation/ThreePointCorrelation.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
using namespace FML::THREEPOINT;
using FML::TESTS::check;
using FML::TESTS::throws;
std::vector<Sample> points(size_t n, std::uint64_t seed) {
    std::vector<Sample> p(n);
    for (size_t i = 0; i < n; ++i) {
        p[i].id = i;
        FML::RANDOM::COUNTER_BASED::Stream rng(seed, i);
        for (double & x : p[i].position)
            x = rng.uniform();
    }
    return p;
}
Point vector(const Sample & x, const Sample & y, double L) {
    Point u;
    for (int d = 0; d < 3; ++d) {
        u[d] = y.position[d] - x.position[d];
        u[d] -= L * std::round(u[d] / L);
    }
    return u;
}
double length(Point u) { return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); }
double legendre(int l, double x) {
    if (l == 0)
        return 1;
    double a = 1, b = x;
    for (int k = 2; k <= l; ++k) {
        double c = ((2 * k - 1) * x * b - (k - 1) * a) / k;
        a = b;
        b = c;
    }
    return b;
}
// Independent O(N^3) enumeration: no harmonics and no factorized self-pair correction.
std::vector<double> direct(const std::vector<Sample> & D, const std::vector<Sample> & R, const Result & shape) {
    std::array<const std::vector<Sample> *, 2> cat{&D, &R};
    const auto & o = shape.options;
    std::vector<double> out(shape.moments.size());
    for (int c = 0; c < 2; ++c)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (size_t i = 0; i < cat[c]->size(); ++i)
                    for (size_t j = 0; j < cat[a]->size(); ++j)
                        for (size_t k = 0; k < cat[b]->size(); ++k) {
                            if ((c == a && i == j) || (c == b && i == k) || (a == b && j == k))
                                continue;
                            auto u = vector((*cat[c])[i], (*cat[a])[j], o.boxsize),
                                 v = vector((*cat[c])[i], (*cat[b])[k], o.boxsize);
                            double r = length(u), s = length(v);
                            if (r < o.edges.front() || r >= o.edges.back() || s < o.edges.front() ||
                                s >= o.edges.back())
                                continue;
                            size_t x = std::upper_bound(o.edges.begin(), o.edges.end(), r) - o.edges.begin() - 1;
                            size_t y = std::upper_bound(o.edges.begin(), o.edges.end(), s) - o.edges.begin() - 1;
                            double mu = std::clamp((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / r / s, -1., 1.);
                            for (int l = 0; l <= o.lmax; ++l)
                                out[(c * 4 + a * 2 + b) * shape.zeta.size() + shape.index(l, x, y)] += legendre(l, mu);
                        }
    return out;
}
bool near(const std::vector<double> & a, const std::vector<double> & b, double eps = 2e-10) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > eps * std::max({1., std::abs(a[i]), std::abs(b[i])}))
            return false;
    return true;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    auto D = points(21, 71), R = points(29, 72);
    Options o{1, {.025, .13, .24, .41}, 16};
    auto result = periodic(D, R, o);
    auto ref = direct(D, R, result);
    check(near(result.moments, ref),
          "all eight ordered D/R triplet moments agree with explicit enumeration through ell=16");
    std::vector<double> z(result.zeta.size());
    const size_t nb = o.edges.size() - 1;
    for (int c = 0; c < 2; ++c)
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                int nr = c + a + b;
                double norm = 1;
                for (int i = 0; i < 3 - nr; ++i)
                    norm *= D.size() - i;
                for (int i = 0; i < nr; ++i)
                    norm *= R.size() - i;
                for (int l = 0; l <= o.lmax; ++l)
                    for (size_t x = 0; x < nb; ++x)
                        for (size_t y = 0; y < nb; ++y) {
                            auto volume = [&](size_t t) {
                                return 4 * M_PI / 3 * (std::pow(o.edges[t + 1], 3) - std::pow(o.edges[t], 3));
                            };
                            size_t i = result.index(l, x, y);
                            z[i] += (nr % 2 ? -1 : 1) * (2 * l + 1) * ref[(c * 4 + a * 2 + b) * z.size() + i] / norm /
                                    volume(x) / volume(y);
                        }
            }
    check(near(z, result.zeta),
          "signed factorial-normalized estimator and (2ell+1)/shell-volume factors match direct reference");
    bool symmetric = true;
    for (int l = 0; l <= o.lmax; ++l)
        for (size_t a = 0; a < nb; ++a)
            for (size_t b = 0; b < nb; ++b)
                symmetric &= std::abs(result.zeta[result.index(l, a, b)] - result.zeta[result.index(l, b, a)]) < 1e-9;
    check(symmetric, "radial-leg exchange symmetry including odd angular multipoles");
    auto reordered = D;
    std::reverse(reordered.begin(), reordered.end());
    check(near(periodic(reordered, R, o).zeta, result.zeta), "catalogue permutation preserves the estimator");
    auto rotated = D, rr = R;
    for (auto * p : {&rotated, &rr})
        for (auto & x : *p) {
            double a = x.position[0];
            x.position[0] = x.position[1];
            x.position[1] = 1 - a;
        }
    check(near(periodic(rotated, rr, o).zeta, result.zeta), "rigid box rotation preserves all isotropic multipoles");
    auto translated = D, tr = R;
    for (auto * p : {&translated, &tr})
        for (auto & x : *p)
            for (double & q : x.position)
                q = std::fmod(q + .27, 1.);
    check(near(periodic(translated, tr, o).zeta, result.zeta),
          "periodic boundary crossing and global translation preserve moments");
    std::vector<Sample> line{{{{.1, .1, .1}}, 0}, {{{.2, .1, .1}}, 1}, {{{.4, .1, .1}}, 2}};
    auto exact = periodic(line, R, {1, {.05, .45}, 16});
    bool ok = true;
    for (int l = 0; l <= 16; ++l)
        ok &= std::abs(exact.moment(0, 0, 0, l, 0, 0) - (l % 2 ? 2 : 6)) < 1e-10;
    check(ok, "analytic collinear triangle: six ordered monopoles and correct odd/even moments through ell=16");
    auto triangle = line;
    triangle[2].position = {.1, .2, .1};
    auto right = periodic(triangle, R, {1, {.05, .2}, 2});
    check(std::abs(right.moment(0, 0, 0, 1, 0, 0) - 2 * std::sqrt(2.)) < 1e-11,
          "right triangle addition theorem normalization matches analytic dipole");
    // Only two data objects can be neighbors in the selected shell; no data triple exists.
    auto pair = line;
    pair[2].position = {.7, .7, .7};
    auto self = periodic(pair, R, {1, {.05, .15}, 6});
    ok = true;
    for (int l = 0; l <= 6; ++l)
        ok &= std::abs(self.moment(0, 0, 0, l, 0, 0)) < 1e-11;
    check(ok, "same-bin neighbor self-products are removed for every ell");
    auto edge = line;
    edge[0].position = {.125, .125, .125};
    edge[1].position = {.25, .125, .125};
    edge[2].position = {.5, .125, .125};
    auto boundary = periodic(edge, R, {1, {.125, .25, .375}, 4});
    check(near(boundary.moments, direct(edge, R, boundary)),
          "half-open shell edges, inner inclusion and outer exclusion match direct counts");
    auto coincident = line;
    coincident[1].position = coincident[0].position;
    auto zero = periodic(coincident, R, o);
    check(near(zero.moments, direct(coincident, R, zero)),
          "distinct IDs at coincident positions are outside positive-radius shells");
    auto onebin = periodic(D, R, {1, {.025, .41}, 0});
    double total = 0;
    for (size_t a = 0; a < nb; ++a)
        for (size_t b = 0; b < nb; ++b)
            total += result.moment(0, 0, 0, 0, a, b);
    check(std::abs(total - onebin.moment(0, 0, 0, 0, 0, 0)) < 1e-10,
          "monopole shell partitions add to the unpartitioned ordered count");
    check(throws([&] {
              auto p = D;
              p[1].id = p[0].id;
              periodic(p, R, o);
          }),
          "duplicate IDs are rejected");
    check(throws([&] {
              auto p = D;
              p[0].position[0] = NAN;
              periodic(p, R, o);
          }),
          "nonfinite positions are rejected");
    check(throws([&] { periodic({}, R, o); }), "undersized data catalogue is rejected collectively");
    check(throws([&] {
              auto p = o;
              p.edges = {0, .2};
              periodic(D, R, p);
          }),
          "zero minimum radius is rejected");
    check(throws([&] {
              auto p = o;
              p.edges = {.1, .1};
              periodic(D, R, p);
          }),
          "nonincreasing edges are rejected");
    check(throws([&] {
              auto p = o;
              p.edges = {.1, .5};
              periodic(D, R, p);
          }),
          "half-box maximum is rejected to keep analytic shells unambiguous");
    check(throws([&] {
              auto p = o;
              p.lmax = 17;
              periodic(D, R, p);
          }),
          "unsupported harmonic order is rejected");
    check(throws([&] {
              auto p = D;
              p[0].position[0] = 1;
              periodic(p, R, o);
          }),
          "unwrapped/out-of-box input is rejected");
    check(throws([&] {
              auto p = o;
              p.boxsize = 1e200;
              periodic(D, R, p);
          }),
          "unrepresentable KD-tree squared distances are rejected");
    check(throws([&] {
              auto p = o;
              p.edges = {1e-150, 2e-150};
              periodic(D, R, p);
          }),
          "unrepresentable shell normalization is rejected");
    check(throws([&] { result.moment(2, 0, 0, 0, 0, 0); }), "result accessors check catalogue indices");
    if (FML::NTasks > 1) {
        check(throws([&] {
                  auto p = o;
                  if (FML::ThisTask == 0)
                      p.lmax = 3;
                  periodic(D, R, p);
              }),
              "rank-dependent options fail collectively before reduction");
        check(throws([&] {
                  auto p = D;
                  if (FML::ThisTask == 0)
                      p[0].position[0] += .001;
                  periodic(p, R, o);
              }),
              "rank-dependent valid catalogues fail collectively");
        check(throws([&] {
                  auto p = D;
                  if (FML::ThisTask == 0)
                      p.clear();
                  periodic(p, R, o);
              }),
              "one-rank invalid input fails on every rank");
    }
    // Independent uniform mocks: test both the disconnected baseline and contrast cancellation.
    constexpr int mocks = 48;
    std::array<double, 3> sum{}, sum2{};
    double rawmean = 0;
    Options u{1, {.04, .32}, 2};
    for (int m = 0; m < mocks; ++m) {
        auto q = periodic(points(100, 1000 + 2 * m), points(160, 1001 + 2 * m), u);
        for (int l = 0; l < 3; ++l) {
            double x = q.zeta[q.index(l, 0, 0)];
            sum[l] += x;
            sum2[l] += x * x;
        }
        rawmean += q.moment(0, 0, 0, 0, 0, 0) / (100. * 99. * 98.);
    }
    for (int l = 0; l < 3; ++l) {
        double mean = sum[l] / mocks,
               se = std::sqrt(std::max(0., (sum2[l] - sum[l] * sum[l] / mocks) / (mocks - 1) / mocks));
        check(std::abs(mean) < 5 * se + .005,
              "independent uniform mocks have zero contrast multipole within measured ensemble error ell=" +
                  std::to_string(l));
        if (FML::ThisTask == 0)
            std::cout << "UNIFORM ell " << l << " mean " << mean << " SEM " << se << '\n';
    }
    double f = 4 * M_PI / 3 * (std::pow(.32, 3) - std::pow(.04, 3));
    check(std::abs(rawmean / mocks - f * f) < .08 * f * f,
          "uniform raw DDD monopole retains its nonzero analytic disconnected baseline");
    return FML::TESTS::summary("periodic 3PCF");
}
