//==========================================================
// Tests for FML::CORRELATIONFUNCTIONS::PAIRCOUNTS (PairCounting/PairCount.h) against brute force:
//   - auto and cross pair counts in 1D, 2D and 3D, periodic and not, with weights, rmax up to 0.45
//   - cross pairs of catalogues of very different size (e.g. data and many more randoms) in both orders
//   - (r, mu) counts with the observer at infinity: auto pairs symmetric in mu, = brute force folded in |mu|
//   - the correlation function of uniform random points ~ 0 in 1D, 2D and 3D
//   - extra quantities binned with the pairs, a pair at r just below rmax, multipoles (with one ell)
//   - the survey correlation function (Landy-Szalay) of uniform points and its D1R2 counts
//   - particles with default member initializers; the header in two translation units (test_paircount_odr.cpp)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PairCounting/PairCount.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <random>
#include <string>
#include <vector>

using namespace FML::CORRELATIONFUNCTIONS::PAIRCOUNTS;

int paircount_odr_check(); // defined in test_paircount_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << std::setprecision(5) << x;
    return os.str();
}

// A particle with a weight (and a default member initializer, which must compile)
template <int N>
struct Particle {
    double pos[N];
    double w{1.0};
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return N; }
    double get_weight() const { return w; }
};

std::mt19937_64 rng(1234);
std::uniform_real_distribution<double> uniform(0.0, 1.0);

template <int N>
std::vector<Particle<N>> make(size_t n, bool weights) {
    std::vector<Particle<N>> v(n);
    for (auto & p : v) {
        for (int d = 0; d < N; d++)
            p.pos[d] = uniform(rng);
        p.w = weights ? 0.5 + uniform(rng) : 1.0;
    }
    return v;
}

// Brute-force weighted counts in (|mu|, r) bins (mu wrt the last axis), r in [0, rmax)
template <int N>
DVector2D brute(const std::vector<Particle<N>> & a,
                const std::vector<Particle<N>> & b,
                bool autopairs,
                bool periodic,
                double rmax,
                int nr,
                int nmu) {
    DVector2D c(nmu, DVector(nr, 0.0));
    for (size_t i = 0; i < a.size(); i++)
        for (size_t j = autopairs ? i + 1 : 0; j < b.size(); j++) {
            double d[N], r2 = 0.0;
            for (int k = 0; k < N; k++) {
                d[k] = a[i].pos[k] - b[j].pos[k];
                if (periodic and d[k] > 0.5)
                    d[k] -= 1.0;
                if (periodic and d[k] < -0.5)
                    d[k] += 1.0;
                r2 += d[k] * d[k];
            }
            if (r2 >= rmax * rmax or r2 == 0.0)
                continue;
            const double r = std::sqrt(r2);
            const int imu = std::min(int(std::fabs(d[N - 1]) / r * nmu), nmu - 1);
            c[imu][std::min(int(r / rmax * nr), nr - 1)] += a[i].w * b[j].w;
        }
    return c;
}

// Pair counts (r bins, or |mu| folded) from the library against brute force
template <int N>
void compare_counts(std::string label,
                    const std::vector<Particle<N>> & a,
                    const std::vector<Particle<N>> * b,
                    bool periodic,
                    double rmax,
                    int nmu_folded = 1) {
    const int nr = 8;
    const int nmu = 2 * nmu_folded;
    DVector r_array, mu_array, observer(N, 0.5);
    observer[N - 1] = -effective_infinity;
    DVector2D pc, xi;
    if (b)
        AngularCorrelationFunctionBox<Particle<N>, Particle<N>>(a.data(),
                                                                a.size(),
                                                                b->data(),
                                                                b->size(),
                                                                0.0,
                                                                rmax,
                                                                nr,
                                                                -1.0,
                                                                1.0,
                                                                nmu_folded > 1 ? nmu : 1,
                                                                r_array,
                                                                mu_array,
                                                                pc,
                                                                false,
                                                                xi,
                                                                observer,
                                                                periodic,
                                                                false);
    else
        AngularCorrelationFunctionBox<Particle<N>>(a.data(),
                                                   a.size(),
                                                   0.0,
                                                   rmax,
                                                   nr,
                                                   -1.0,
                                                   1.0,
                                                   nmu_folded > 1 ? nmu : 1,
                                                   r_array,
                                                   mu_array,
                                                   pc,
                                                   false,
                                                   xi,
                                                   observer,
                                                   periodic,
                                                   false);
    const auto bf = brute<N>(a, b ? *b : a, b == nullptr, periodic, rmax, nr, nmu_folded);
    double maxdiff = 0.0, total = 0.0;
    for (int m = 0; m < nmu_folded; m++)
        for (int i = 0; i < nr; i++) {
            double count = pc[0][i];
            if (nmu_folded > 1)
                count = pc[nmu_folded + m][i] + pc[nmu_folded - 1 - m][i];
            maxdiff = std::max(maxdiff, std::fabs(count - bf[m][i]) / std::max(1.0, bf[m][i]));
            total += bf[m][i];
        }
    check(maxdiff < 1e-10, label + ": " + num(total) + " (weighted) pairs, max relative difference " + num(maxdiff));
}

// The old linear scan is an independent reference for the exact assignment/order contract.
void test_rank_assignment() {
    using Costs = std::vector<std::pair<double, size_t>>;
    const Costs tied{{8., 0}, {8., 1}, {4., 2}, {4., 3}, {4., 4}, {4., 5}, {0., 6}};
    check(detail::assign_cells_to_rank(tied, 2, 0) == std::vector<size_t>({0, 2, 4, 6}) &&
              detail::assign_cells_to_rank(tied, 2, 1) == std::vector<size_t>({1, 3, 5}),
          "rank assignment breaks equal-load ties by the lowest rank");
    Costs generated;
    std::mt19937_64 schedule_rng(92847);
    for (size_t i = 0; i < 1024; ++i)
        generated.emplace_back(double(schedule_rng() % 97) * .1, i);
    std::sort(generated.begin(), generated.end(), [](const auto &a, const auto &b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    // Includes empty schedules, idle ranks, repeated ties, zero increments, skewed loads and rounding.
    for (const auto &cost : {Costs{}, tied, Costs{{1., 7}}, Costs{{0., 0}, {0., 1}, {0., 2}},
                             Costs{{0x1p100, 0}, {1., 1}, {1., 2}, {.5, 3}, {.5, 4}},
                             Costs{{0x1p100, 0}, {0x1p100, 1}, {1., 2}, {1., 3}, {.5, 4}}, generated}) {
        for (int nranks : {1, 2, 3, 4, 7, 16, 65}) {
            std::vector<double> load(nranks, 0.);
            std::vector<std::vector<size_t>> expected(nranks);
            for (const auto &cell : cost) {
                int rank = int(std::min_element(load.begin(), load.end()) - load.begin());
                expected[rank].push_back(cell.second);
                load[rank] += cell.first;
            }
            bool same = true;
            for (int rank = 0; rank < nranks; ++rank)
                same &= detail::assign_cells_to_rank(cost, nranks, rank) == expected[rank];
            if (!same) {
                check(false, "heap assignment matches the legacy scan, including each rank's cell order");
                return;
            }
        }
    }
    check(true, "heap assignment matches the legacy scan, including each rank's cell order");
}

int main() {
    test_rank_assignment();
    check(paircount_odr_check() == 1, "PairCount.h can be included in several translation units");

    // Pair counts against brute force
    {
        auto a3 = make<3>(2500, false), b3 = make<3>(3000, true), aw = make<3>(2000, true);
        compare_counts<3>("3D auto, periodic, rmax 0.1", a3, nullptr, true, 0.1);
        compare_counts<3>("3D auto, not periodic, rmax 0.1", a3, nullptr, false, 0.1);
        compare_counts<3>("3D auto, periodic, rmax 0.45", a3, nullptr, true, 0.45);
        compare_counts<3>("3D auto, not periodic, rmax 0.7 (> half the box)", a3, nullptr, false, 0.7);
        compare_counts<3>("3D cross, weights, periodic", aw, &b3, true, 0.15);
        compare_counts<3>("3D cross, weights, not periodic", aw, &b3, false, 0.15);
        compare_counts<3>("3D auto (r, |mu|), 4 |mu| bins", a3, nullptr, true, 0.1, 4);
        compare_counts<3>("3D cross (r, |mu|), 4 |mu| bins", aw, &b3, true, 0.1, 4);
        auto small = make<3>(150, false), large = make<3>(150000, false);
        compare_counts<3>("3D cross, n2 = 1000 n1 (grids 10 / 42), periodic", small, &large, true, 0.02);
        compare_counts<3>("3D cross, n2 = 1000 n1, not periodic", small, &large, false, 0.02);
        compare_counts<3>("3D cross, n1 = 1000 n2, periodic", large, &small, true, 0.02);
        auto a2 = make<2>(3000, true);
        auto a1 = make<1>(3000, false);
        compare_counts<2>("2D auto, weights, periodic", a2, nullptr, true, 0.1);
        compare_counts<2>("2D auto, weights, not periodic", a2, nullptr, false, 0.1);
        compare_counts<1>("1D auto, periodic", a1, nullptr, true, 0.05);
    }

    // Auto pairs: the counts are symmetric in mu
    {
        auto a = make<3>(3000, false);
        DVector r_array, mu_array, observer{0.5, 0.5, -effective_infinity};
        DVector2D pc, xi;
        AngularCorrelationFunctionBox<Particle<
            3>>(a.data(), a.size(), 0.0, 0.1, 4, -1.0, 1.0, 6, r_array, mu_array, pc, false, xi, observer, true, false);
        double asym = 0.0, total = 0.0;
        for (int m = 0; m < 3; m++)
            for (int i = 0; i < 4; i++) {
                asym = std::max(asym, std::fabs(pc[3 + m][i] - pc[2 - m][i]));
                total += pc[3 + m][i] + pc[2 - m][i];
            }
        check(asym < 1e-10,
              "auto pairs symmetric in mu (max |N(mu) - N(-mu)| = " + num(asym) + " of " + num(total) + ")");
    }

    // The correlation function of uniform random points is ~ 0
    {
        auto run = [&](auto ndim_tag) {
            constexpr int N = decltype(ndim_tag)::value;
            auto a = make<N>(20000, false);
            DVector r_array, pc, xi;
            RadialCorrelationFunctionBox<
                Particle<N>>(a.data(), a.size(), 0.0, 0.05, 5, r_array, pc, true, xi, true, false);
            double mx = 0.0;
            for (auto x : xi)
                mx = std::max(mx, std::fabs(x));
            check(mx < 0.05, "xi of uniform random points in " + std::to_string(N) + "D: max |xi| = " + num(mx));
            // (r, mu) in 2D: the mu bins are not uniform in angle
            if constexpr (N >= 2) {
                DVector mu_array, observer(N, 0.5);
                observer[N - 1] = -effective_infinity;
                DVector2D pc2, xi2;
                AngularCorrelationFunctionBox<Particle<N>>(a.data(),
                                                           a.size(),
                                                           0.0,
                                                           0.05,
                                                           3,
                                                           -1.0,
                                                           1.0,
                                                           4,
                                                           r_array,
                                                           mu_array,
                                                           pc2,
                                                           true,
                                                           xi2,
                                                           observer,
                                                           true,
                                                           false);
                double mx2 = 0.0;
                for (auto & row : xi2)
                    for (auto x : row)
                        mx2 = std::max(mx2, std::fabs(x));
                check(mx2 < 0.1,
                      "xi(r, mu) of uniform random points in " + std::to_string(N) + "D: max |xi| = " + num(mx2));
            }
        };
        run(std::integral_constant<int, 1>{});
        run(std::integral_constant<int, 2>{});
        run(std::integral_constant<int, 3>{});
    }

    // Extra quantities: the number of pairs and the sum of r per bin
    {
        auto a = make<3>(3000, false);
        DVector r_array, pc, xi;
        DVector2D extra;
        ExtraQuantitiesToBinFunction<Particle<3>, Particle<3>> f =
            [](const double *, double r, double, const Particle<3> &, const Particle<3> &, double * storage) {
                storage[0] += 1.0;
                storage[1] += r;
            };
        RadialCorrelationFunctionBox<
            Particle<3>>(a.data(), a.size(), 0.0, 0.1, 5, r_array, pc, false, xi, true, false, 2, extra, f);
        bool ok = true;
        for (int i = 0; i < 5; i++) {
            const double rmean = extra[i][1] / extra[i][0];
            ok = ok and extra[i][0] == pc[i] and rmean > 0.02 * i and rmean < 0.02 * (i + 1);
        }
        check(ok, "extra quantities (pair count = paircounts, mean r inside the bins)");
    }

    // A pair at r just below rmax where (r - rmin) / (rmax - rmin) * nbins rounds to nbins
    {
        double rmin = 0.0, rmax = 0.0, r = 0.0;
        int nb = 0;
        bool found = false;
        for (int t = 0; t < 200000 and not found; t++) {
            rmin = 0.001 * (t % 97);
            rmax = rmin + 0.01 + 0.0037 * (t % 101);
            nb = 1 + t % 50;
            r = std::nextafter(rmax, 0.0);
            found = int((r - rmin) / (rmax - rmin) * nb) == nb;
        }
        std::vector<Particle<3>> a(2);
        for (auto & p : a)
            p.pos[1] = p.pos[2] = 0.1;
        a[0].pos[0] = 0.0;
        a[1].pos[0] = r;
        DVector r_array, pc, xi;
        RadialCorrelationFunctionBox<Particle<3>>(a.data(), 2, rmin, rmax, nb, r_array, pc, false, xi, false, false);
        check(found and pc.back() == 1.0, "a pair at r just below rmax goes in the last bin");
    }

    // Survey: uniform data and randoms (not periodic): Landy-Szalay xi ~ 0, and D1R2 = the box cross counts
    {
        auto data = make<3>(8000, true);
        auto randoms = make<3>(40000, true);
        DVector r_array, dd, dr, rd, rr, xi, observer{0.5, 0.5, 0.5};
        RadialCorrelationFunctionSurvey<Particle<3>>(data.data(),
                                                     data.size(),
                                                     randoms.data(),
                                                     randoms.size(),
                                                     data.data(),
                                                     data.size(),
                                                     randoms.data(),
                                                     randoms.size(),
                                                     0.02,
                                                     0.1,
                                                     4,
                                                     r_array,
                                                     dd,
                                                     dr,
                                                     rd,
                                                     rr,
                                                     xi,
                                                     observer,
                                                     false,
                                                     false,
                                                     0,
                                                     empty_2D_array,
                                                     empty_2D_array,
                                                     empty_2D_array,
                                                     empty_2D_array,
                                                     {},
                                                     "LS");
        double mx = 0.0;
        for (auto x : xi)
            mx = std::max(mx, std::fabs(x));
        check(mx < 0.05, "survey Landy-Szalay xi of uniform points ~ 0 (max |xi| = " + num(mx) + ")");
        DVector pc, xib;
        RadialCorrelationFunctionBox<Particle<3>, Particle<3>>(data.data(),
                                                               data.size(),
                                                               randoms.data(),
                                                               randoms.size(),
                                                               0.02,
                                                               0.1,
                                                               4,
                                                               r_array,
                                                               pc,
                                                               true,
                                                               xib,
                                                               false,
                                                               false);
        double maxdiff = 0.0;
        for (size_t i = 0; i < pc.size(); i++)
            maxdiff = std::max(maxdiff, std::fabs(pc[i] - dr[i]) / pc[i]);
        check(maxdiff < 1e-12 and rd == dr, "survey D1R2 = box cross pair counts (normalized), R1D2 = D1R2 for auto");
    }

#ifdef USE_GSL
    // Multipoles of xi = 1 (only ell = 0)
    {
        DVector r{0.1, 0.2, 0.3, 0.4, 0.5}, mu{-1.0, -0.5, 0.0, 0.5, 1.0};
        DVector2D z(5, DVector(5, 1.0));
        Spline2D xi_spline(mu, r, z);
        std::vector<DVector> multipoles(1);
        FromAngularCorrelationToMultipoles(r, xi_spline, multipoles);
        check(std::fabs(multipoles[0][2] - 1.0) < 1e-10, "monopole of xi = 1 (one multipole)");
    }
#endif

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All pair count tests passed\n" :
                                   std::to_string(fails) + " pair count test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
