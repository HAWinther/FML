//==========================================================
// Tests for FML::KDTREE (KDTree/KDTree.h) against brute force:
//   - nearest neighbour and points within a radius, 2D and 3D, with and without a periodic box
//   - periodic: queries near the boundaries and outside the box, radius > boxsize/2, duplicate points, empty tree
//   - the k nearest neighbours (and the distances) if available
//   - DensityProfileBinner (OpenMP + MPI) against a brute-force pair count
// The header is also included in test_kdtree_odr.cpp (the header must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/KDTree/KDTree.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace FML::KDTREE;

int kdtree_odr_check(); // defined in test_kdtree_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

double pdist2(const point_t & a, const point_t & b, double boxsize) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double dx = a[i] - b[i];
        if (boxsize > 0.0)
            dx -= boxsize * std::round(dx / boxsize);
        d += dx * dx;
    }
    return d;
}

struct Point3 {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    check(kdtree_odr_check() == 1, "KDTree.h can be included in several translation units");

    // Nearest neighbour and radius queries against brute force
    for (double boxsize : {0.0, 1.0, 2.5}) {
        for (int ndim : {2, 3}) {
            const int npts = 2000;
            const double L = boxsize > 0.0 ? boxsize : 1.0;
            pointVec pts(npts, point_t(ndim));
            for (auto & p : pts)
                for (auto & x : p)
                    x = L * uniform(rng);
            KDTree tree(pts, boxsize);
            int bad_nearest = 0, bad_radius = 0;
            for (int q = 0; q < 300; q++) {
                point_t pt(ndim);
                for (auto & x : pt)
                    x = L * uniform(rng);
                if (q % 3 == 0) // near the corners
                    for (auto & x : pt)
                        x = uniform(rng) < 0.5 ? 0.01 * L * uniform(rng) : L * (1.0 - 0.01 * uniform(rng));
                double best = 1e100;
                for (auto & p : pts)
                    best = std::min(best, pdist2(p, pt, boxsize));
                if (std::fabs(pdist2(pts[tree.nearest_index(pt)], pt, boxsize) - best) > 1e-14 * L * L)
                    bad_nearest++;
                const double r = 0.07 * L;
                std::set<size_t> truth;
                for (size_t i = 0; i < pts.size(); i++)
                    if (pdist2(pts[i], pt, boxsize) <= r * r)
                        truth.insert(i);
                auto found = tree.neighborhood_indices(pt, r);
                if (std::set<size_t>(found.begin(), found.end()) != truth or found.size() != truth.size())
                    bad_radius++;
            }
            const std::string label = "box " + std::to_string(boxsize).substr(0, 3) + ", " + std::to_string(ndim) + "D: ";
            check(bad_nearest == 0, label + "nearest neighbour (" + std::to_string(bad_nearest) + " of 300 wrong)");
            check(bad_radius == 0, label + "points within r (" + std::to_string(bad_radius) + " of 300 wrong)");
        }
    }

    // Periodic edge cases
    {
        const double boxsize = 1.0;
        pointVec pts(3000, point_t(3));
        for (auto & p : pts)
            for (auto & x : p)
                x = uniform(rng);
        KDTree tree(pts, boxsize);
        for (double r : {0.5, 0.7, 0.9}) {
            point_t pt{0.3, 0.6, 0.95};
            size_t truth = 0;
            for (auto & p : pts)
                truth += pdist2(p, pt, boxsize) <= r * r;
            auto found = tree.neighborhood_indices(pt, r);
            std::set<size_t> unique(found.begin(), found.end());
            check(found.size() == truth and unique.size() == truth,
                  "periodic r = " + std::to_string(r).substr(0, 3) + " (>= boxsize/2): " + std::to_string(found.size()) +
                      " of " + std::to_string(truth) + ", each once");
        }
        point_t outside{1.02, -0.03, 0.5}, wrapped{0.02, 0.97, 0.5};
        auto a = tree.neighborhood_indices(outside, 0.1), b = tree.neighborhood_indices(wrapped, 0.1);
        check(std::set<size_t>(a.begin(), a.end()) == std::set<size_t>(b.begin(), b.end()) and a.size() > 0,
              "periodic query outside the box = wrapped query (" + std::to_string(a.size()) + " points)");
        check(tree.nearest_index(outside) == tree.nearest_index(wrapped), "periodic nearest outside the box");

        pointVec same(50, point_t{0.0, 0.0, 0.0});
        KDTree tree_same(same, boxsize);
        check(tree_same.neighborhood_indices(point_t{0.999, 0.0, 0.001}, 0.5).size() == 50,
              "50 identical points on the boundary, r = boxsize/2: each found once");
        KDTree empty(pointVec{}, boxsize);
        check(empty.neighborhood_indices(point_t{0.5, 0.5, 0.5}, 0.2).empty(), "empty tree");
    }

    // Tree from particles + DensityProfileBinner against brute force (periodic, unit box)
    {
        const int n1 = 40, n2 = 4000, nbins = 8;
        const double rmin = 0.0, rmax = 0.2;
        std::vector<Point3> t1(n1), t2(n2);
        for (auto & p : t1)
            for (auto & x : p.pos)
                x = uniform(rng);
        for (auto & p : t2)
            for (auto & x : p.pos)
                x = uniform(rng);
        DVector rcenter;
        std::vector<DVector> profiles;
        DensityProfileBinner(t1.data(), n1, t2.data(), n2, rmin, rmax, nbins, true, rcenter, profiles);
        double maxerr = 0.0;
        for (int i = 0; i < n1; i++) {
            std::vector<double> count(nbins, 0.0);
            for (auto & p : t2) {
                const double d2 = pdist2(point_t(t1[i].pos, t1[i].pos + 3), point_t(p.pos, p.pos + 3), 1.0);
                if (d2 < rmax * rmax and d2 >= rmin * rmin)
                    count[std::min(int((std::sqrt(d2) - rmin) / (rmax - rmin) * nbins), nbins - 1)] += 1.0;
            }
            for (int b = 0; b < nbins; b++) {
                const double r0 = rmin + (rmax - rmin) * b / nbins, r1 = rmin + (rmax - rmin) * (b + 1) / nbins;
                const double expected = n2 * 4.0 * M_PI / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
                maxerr = std::max(maxerr, std::fabs(profiles[i][b] - count[b] / expected));
            }
        }
        check(maxerr < 1e-12, "DensityProfileBinner = brute force (max difference " + std::to_string(maxerr) + ")");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All kd-tree tests passed\n" : std::to_string(fails) + " kd-tree test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
