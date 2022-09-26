#include <FML/KDTree/KDTree.h>
#include <iostream>
#include <vector>

using point_t = FML::KDTREE::point_t;
using pointVec = FML::KDTREE::pointVec;
using KDTree = FML::KDTREE::KDTree;

void minimal_kdtree_example() {

    // Make point set (here in 2D)
    pointVec points;
    points.push_back({0.0, 0.0});
    points.push_back({0.0, 1.0});
    points.push_back({1.0, 0.0});
    points.push_back({1.0, 1.0});

    // Make tree
    KDTree tree(points);

    // Find closest point in set to a given point
    point_t pt = {0.25, 0.25};
    auto pt_nearest = tree.nearest_point(pt);

    // Find all points within some radius of a given point
    pt = {0.9, 0.9};
    double radius = 0.25;
    auto nbor_points = tree.neighborhood_points(pt, radius);

    // Find index (in points) of all points within some radius of a given point
    auto nbor_index = tree.neighborhood_indices(pt, radius);
}


// Test the kdtree for periodic or non periodic boxes
void test(int ndim, bool periodic) {
    pointVec points;
    point_t pt(ndim);

    std::cout << "\n";
    std::cout << "#=====================================================\n";
    if (periodic)
        std::cout << "# Running test of periodic kdTree for ndim = " << ndim << "\n";
    else
        std::cout << "# Running test of kdTree for ndim = " << ndim << "\n";
    std::cout << "#=====================================================\n";

    // Set boxsize
    double boxsize = 1.0;
    if (not periodic)
        boxsize = 0.0;

    // Make pointset in [0,boxsize]^ndim
    const int n = 1000;
    for (int i = 0; i < n - 1; i++) {
        pt = point_t(ndim);
        for (int idim = 0; idim < ndim; idim++) {
            pt[idim] = boxsize * (rand() % RAND_MAX) / double(RAND_MAX);
        }
        points.push_back(pt);
    }
    // Add origin (0,0,0,...)
    points.push_back(point_t(ndim, 0.0));

    // Make tree
    KDTree tree(points, boxsize);

    // Find closest point to our points (which should be themselves)
    for (size_t i = 0; i < points.size(); i++) {
        pt = points[i];
        auto pt_closest = tree.nearest_point(pt);
        for (int idim = 0; idim < ndim; idim++) {
            assert(std::fabs(pt[idim] - pt_closest[idim]) < 1e-10);
        }
    }

    // Find closest point to (boxsize,boxsize,...)
    // which should be the origin if periodic
    if (periodic) {
        pt = point_t(ndim, boxsize);
        auto pt_closest = tree.nearest_point(pt);
        for (int idim = 0; idim < ndim; idim++) {
            assert(pt_closest[idim] == 0.0);
        }
    }

    // Get neihborhood points and check that they all are within the distance
    pt = point_t(ndim, 0.0);
    double radius = 0.1;
    auto nbor_points = tree.neighborhood_points(pt, radius);
    for (point_t pt_close : nbor_points) {
        double dist = 0.0;
        for (int idim = 0; idim < ndim; idim++) {
            double dx = pt_close[idim] - pt[idim];
            if (periodic) {
                if (dx >= boxsize / 2)
                    dx -= boxsize;
                if (dx < -boxsize / 2)
                    dx += boxsize;
            }
            dist += dx * dx;
        }
        dist = std::sqrt(dist);
        assert(dist <= radius);
    }

    // Count how many points we should find
    size_t count = 0;
    for (size_t i = 0; i < points.size(); i++) {
        double dist = 0.0;
        for (int idim = 0; idim < ndim; idim++) {
            double dx = points[i][idim] - pt[idim];
            if (periodic) {
                if (dx >= boxsize / 2)
                    dx -= boxsize;
                if (dx < -boxsize / 2)
                    dx += boxsize;
            }
            dist += dx * dx;
        }
        dist = std::sqrt(dist);
        if (dist <= radius)
            count++;
    }
    assert(count == nbor_points.size());

    std::cout << "All test passed for ndim = " << ndim << "\n";
}

template <int N>
class Particle {
  public:
    double pos[N];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return N; }
};

template <int N>
void periodic_kdtree_from_particles() {
    const int nparts = 1000;
    const double boxsize = 1.0;
    const double radius = 0.01 * N * N * N;

    std::cout << "\n#=====================================================\n";
    std::cout << "# Periodic kdTree (from particles) ndim = " << N << "\n";
    std::cout << "#=====================================================\n";

    // Generate particles with positions (or pointset)
    std::vector<Particle<N>> parts(nparts);
    for (auto & p : parts) {
        auto pos = FML::PARTICLE::GetPos(p);
        for (int i = 0; i < N; i++) {
            pos[i] = (rand() % RAND_MAX) / double(RAND_MAX);
        }
    }
    KDTree tree(parts.data(), parts.size(), boxsize);

    // Find nearest point to origin
    point_t pt(N, 0.0);
    auto res1 = tree.nearest_point(pt);
    std::cout << "Closest point to origin: (";
    for (auto x : res1)
        std::cout << std::setw(15) << x << " ";
    std::cout << ")\n";

    // Find points within radius of origin
    auto nbor_points = tree.neighborhood_points(pt, radius);
    auto nbor_index = tree.neighborhood_indices(pt, radius);

    double expected = std::pow(M_PI, N / 2.0) * std::pow(radius, N) / std::tgamma(N / 2.0 + 1.0) * parts.size();
    std::cout << "All points within radius = " << radius << " of origin:\n";
    std::cout << "Expecting ~" << expected << " found " << nbor_points.size() << "\n";
    for (auto & point : nbor_points) {
        for (auto x : point)
            std::cout << std::setw(15) << x << " ";
        std::cout << "\n";
    }
}

void kdtree_from_points(int ndim) {
    const int nparts = 1000;
    const double radius = 0.01 * ndim * ndim * ndim;

    std::cout << "\n#=====================================================\n";
    std::cout << "# kdTree (from pointset) ndim = " << ndim << "\n";
    std::cout << "#=====================================================\n";

    // Generate pointset
    pointVec points(nparts, point_t(ndim, 0.0));
    for (auto & pt : points) {
        for (int i = 0; i < ndim; i++) {
            pt[i] = (rand() % RAND_MAX) / double(RAND_MAX);
        }
    }

    // Make tree
    KDTree tree(points);

    // Find nearst point to origin
    point_t pt(ndim, 0.0);
    auto res1 = tree.nearest_point(pt);
    std::cout << "Closest point to origin: (";
    for (auto x : res1)
        std::cout << std::setw(15) << x << " ";
    std::cout << ")\n";

    // Find points within radius of origin
    auto nbor_points = tree.neighborhood_points(pt, radius);
    auto nbor_index = tree.neighborhood_indices(pt, radius);

    double expected =
        std::pow(M_PI, ndim / 2.0) * std::pow(radius / 2.0, ndim) / std::tgamma(ndim / 2.0 + 1.0) * points.size();
    std::cout << "All points within radius = " << radius << " of origin:\n";
    std::cout << "Expecting ~" << expected << " found " << nbor_points.size() << "\n";
    for (size_t i = 0; i < nbor_points.size(); i++) {
        std::cout << "Index: " << std::setw(5) << nbor_index[i] << " Pos: ";
        for (auto x : nbor_points[i]) {
            std::cout << std::setw(15) << x << " ";
        }
        std::cout << "\n";
    }
}

int main() {

    // Test the periodic kdTree
    test(1, true);
    test(2, true);
    test(3, true);

    // Test the non-periodic kdTree
    test(1, false);
    test(2, false);
    test(3, false);

    // Example periodic kdtree from particles (with positions)
    periodic_kdtree_from_particles<1>();
    periodic_kdtree_from_particles<2>();
    periodic_kdtree_from_particles<3>();

    // Example normal kdtree from simple points
    kdtree_from_points(1);
    kdtree_from_points(2);
    kdtree_from_points(3);

    // Minimal example
    minimal_kdtree_example();

    return 0;
}
