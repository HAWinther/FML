// =============================================================================
// Tests of the Latin hypercube sampler (LatinHyperCube.h). Run with `make run` in this folder.
//
//   * Centered discrepancy against scipy.stats.qmc.discrepancy, and the O(n d) swap update against a full recompute
//   * Strength 1: one point per stratum in every dimension, cell centres without scrambling, range [0,1)
//   * Strength 2: the orthogonal-array property in every 2D projection; invalid n and d are rejected
//   * Seeds, reset and fast_forward
//   * random-cd lowers the discrepancy and keeps the LHS; lloyd spreads the points and stays in [0,1)
//   * scale (and its inverse) and integers
// =============================================================================

#include <FML/LatinHyperCube/LatinHyperCube.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <set>
#include <stdexcept>

using namespace FML::SAMPLING;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::throws;

// Every dimension has exactly one point in each of the n strata [k/n, (k+1)/n)
static bool is_latin_hypercube(const Sample & s) {
    const size_t n = s.size();
    for (size_t m = 0; m < s[0].size(); m++) {
        std::vector<int> count(n, 0);
        for (const auto & x : s) {
            const long k = long(std::floor(x[m] * double(n)));
            if (k < 0 or k >= long(n) or ++count[size_t(k)] > 1)
                return false;
        }
    }
    return true;
}

static bool in_unit_cube(const Sample & s) {
    for (const auto & x : s)
        for (double v : x)
            if (not(v >= 0.0 and v < 1.0))
                return false;
    return true;
}

static double min_distance(const Sample & s) {
    double best = INFINITY;
    for (size_t i = 0; i < s.size(); i++)
        for (size_t j = i + 1; j < s.size(); j++) {
            double d2 = 0.0;
            for (size_t m = 0; m < s[i].size(); m++)
                d2 += (s[i][m] - s[j][m]) * (s[i][m] - s[j][m]);
            best = std::min(best, std::sqrt(d2));
        }
    return best;
}

int main() {

    //-------------------------------------------------------------------------
    // Discrepancy: the example in the scipy.stats.qmc.discrepancy documentation
    //-------------------------------------------------------------------------
    {
        const Sample space{{1, 3}, {2, 6}, {3, 2}, {4, 5}, {5, 1}, {6, 4}};
        const Sample unit = scale(space, {0.5, 0.5}, {6.5, 6.5}, true);
        check_close(centered_discrepancy(unit), 0.008142039609053464, 1e-14, "CD equals scipy's value");

        // The swap update against the full recompute
        LatinHyperCube lhs(4, true, 1, LHSOptimization::none, 7);
        Sample s = lhs.random(25);
        double worst = 0.0;
        detail::PortableRNG rng(3);
        for (int t = 0; t < 50; t++) {
            const size_t i1 = rng.below(25), i2 = rng.below(25), k = rng.below(4);
            const double before = centered_discrepancy(s);
            const double change = detail::cd_swap_change(s, i1, i2, k);
            std::swap(s[i1][k], s[i2][k]);
            worst = std::max(worst, std::abs(centered_discrepancy(s) - before - change));
        }
        check(worst < 1e-14, "CD swap update equals the full recompute (max error " + FML::TESTS::num(worst) + ")");
    }

    //-------------------------------------------------------------------------
    // Strength 1
    //-------------------------------------------------------------------------
    {
        LatinHyperCube lhs(5, true, 1, LHSOptimization::none, 42);
        const Sample s = lhs.random(37);
        check(s.size() == 37 and s[0].size() == 5, "strength 1: shape n x d");
        check(is_latin_hypercube(s), "strength 1: one point per stratum in every dimension");
        check(in_unit_cube(s), "strength 1: points in [0,1)");
        check(lhs.get_num_generated() == 37, "num_generated counts the points");

        LatinHyperCube centred(3, false, 1, LHSOptimization::none, 1);
        bool at_centres = true;
        for (const auto & x : centred.random(10))
            for (double v : x)
                at_centres = at_centres and std::abs(v * 10.0 - (std::floor(v * 10.0) + 0.5)) < 1e-12;
        check(at_centres, "scramble = false: points at the cell centres");

        check(lhs.random(0).empty(), "n = 0 gives an empty sample");
        check(is_latin_hypercube(LatinHyperCube(1, true, 1, LHSOptimization::none, 5).random(100)), "d = 1 works");

        // The mean of many uniform samples
        LatinHyperCube big(2, true, 1, LHSOptimization::none, 11);
        const Sample b = big.random(10000);
        double mean = 0.0;
        for (const auto & x : b)
            mean += x[0];
        check_close(mean / 1e4, 0.5, 1e-4, "strength 1: mean of a coordinate (stratified: error ~ 1/n)");
    }

    //-------------------------------------------------------------------------
    // Strength 2
    //-------------------------------------------------------------------------
    {
        const int p = 7;
        for (bool scr : {true, false}) {
            LatinHyperCube lhs(p + 1, scr, 2, LHSOptimization::none, 99);
            const Sample s = lhs.random(p * p);
            bool orthogonal = true;
            for (int a = 0; a < p + 1; a++)
                for (int b = a + 1; b < p + 1; b++) {
                    std::set<std::pair<int, int>> cells;
                    for (const auto & x : s)
                        cells.insert({int(std::floor(x[a] * p)), int(std::floor(x[b] * p))});
                    orthogonal = orthogonal and cells.size() == size_t(p * p);
                }
            const std::string tag = scr ? " (scrambled)" : " (centred)";
            check(orthogonal, "strength 2: one point per p x p cell in every 2D projection" + tag);
            check(is_latin_hypercube(s), "strength 2: is also a Latin hypercube" + tag);
            check(in_unit_cube(s), "strength 2: points in [0,1)" + tag);
        }
        LatinHyperCube lhs(3, true, 2, LHSOptimization::none, 1);
        check(throws<std::invalid_argument>([&] { lhs.random(50); }), "strength 2: n not a square is rejected");
        check(throws<std::invalid_argument>([&] { lhs.random(16); }), "strength 2: n = 4^2 (not prime) is rejected");
        LatinHyperCube too_many(9, true, 2, LHSOptimization::none, 1);
        check(throws<std::invalid_argument>([&] { too_many.random(49); }), "strength 2: d > sqrt(n) + 1 is rejected");
    }

    //-------------------------------------------------------------------------
    // Seeds, reset, fast_forward
    //-------------------------------------------------------------------------
    {
        LatinHyperCube a(3, true, 1, LHSOptimization::random_cd, 2024), b(3, true, 1, LHSOptimization::random_cd, 2024);
        LatinHyperCube c(3, true, 1, LHSOptimization::random_cd, 2025);
        const Sample sa = a.random(20);
        check(sa == b.random(20), "the same seed gives the same points");
        check(sa != c.random(20), "another seed gives other points");
        const Sample next = a.random(15);
        a.reset();
        check(a.random(20) == sa and a.get_num_generated() == 20, "reset gives the same points again");
        b.reset();
        b.fast_forward(20);
        check(b.random(15) == next, "fast_forward(n) skips the next n points");
        LatinHyperCube e(2);
        LatinHyperCube f(2, true, 1, LHSOptimization::none, e.get_seed());
        check(e.random(5) == f.random(5), "without a seed: get_seed() reproduces the points");
    }

    //-------------------------------------------------------------------------
    // Optimizations
    //-------------------------------------------------------------------------
    {
        LatinHyperCube plain(3, true, 1, LHSOptimization::none, 5);
        LatinHyperCube cd(3, true, 1, LHSOptimization::random_cd, 5); // starts from the same LHS
        const Sample s0 = plain.random(30), s1 = cd.random(30);
        const double d0 = centered_discrepancy(s0), d1 = centered_discrepancy(s1);
        check(d1 < d0, "random-cd lowers the discrepancy (" + FML::TESTS::num(d0) + " -> " + FML::TESTS::num(d1) + ")");
        check(is_latin_hypercube(s1), "random-cd keeps the LHS property");

        LatinHyperCube plain2(2, true, 1, LHSOptimization::none, 8);
        LatinHyperCube lloyd(2, true, 1, LHSOptimization::lloyd, 8);
        const Sample t0 = plain2.random(50), t1 = lloyd.random(50);
        const double m0 = min_distance(t0), m1 = min_distance(t1);
        check(m1 > m0, "lloyd increases the minimum distance (" + FML::TESTS::num(m0) + " -> " + FML::TESTS::num(m1) +
                           ")");
        check(in_unit_cube(t1), "lloyd: points in [0,1)");
        check(throws<std::invalid_argument>([] { LatinHyperCube(1, true, 1, LHSOptimization::lloyd); }),
              "lloyd with d = 1 is rejected");
    }

    //-------------------------------------------------------------------------
    // scale and integers
    //-------------------------------------------------------------------------
    {
        LatinHyperCube lhs(2, true, 1, LHSOptimization::none, 3);
        const Sample s = lhs.random(10);
        const Sample scaled = scale(s, {-1.0, 10.0}, {1.0, 20.0});
        bool inside = true;
        for (const auto & x : scaled)
            inside = inside and x[0] >= -1.0 and x[0] < 1.0 and x[1] >= 10.0 and x[1] < 20.0;
        check(inside, "scale maps to [l, u)");
        const Sample back = scale(scaled, {-1.0, 10.0}, {1.0, 20.0}, true);
        double err = 0.0;
        for (size_t i = 0; i < s.size(); i++)
            for (int m = 0; m < 2; m++)
                err = std::max(err, std::abs(back[i][m] - s[i][m]));
        check(err < 1e-14, "scale with reverse = true inverts it");
        check(throws<std::invalid_argument>([&] { scale(s, {1.0, 0.0}, {0.0, 1.0}); }), "scale: l >= u is rejected");

        // An LHS with n points scaled to n integers is a permutation in every dimension
        LatinHyperCube ints(3, true, 1, LHSOptimization::none, 4);
        const auto v = ints.integers({5, -3, 0}, {15, 7, 10}, 10, false);
        bool permutation = true;
        for (int m = 0; m < 3; m++) {
            const long long lo = std::vector<long long>{5, -3, 0}[m];
            std::set<long long> seen;
            for (const auto & x : v)
                seen.insert(x[m]);
            permutation = permutation and seen.size() == 10 and *seen.begin() == lo and *seen.rbegin() == lo + 9;
        }
        check(permutation, "integers: n points in n integers are a permutation in each dimension");
        const auto w = ints.integers({0, 0, 0}, {4, 4, 4}, 5, true);
        bool endpoint = true;
        for (const auto & x : w)
            for (long long k : x)
                endpoint = endpoint and k >= 0 and k <= 4;
        check(endpoint, "integers with endpoint = true: in [l, u]");
    }

    check(throws<std::invalid_argument>([] { LatinHyperCube(0); }), "d = 0 is rejected");
    check(throws<std::invalid_argument>([] { LatinHyperCube(2, true, 3); }), "strength 3 is rejected");

    return FML::TESTS::summary("LatinHyperCube");
}
