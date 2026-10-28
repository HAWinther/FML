// Regression tests for periodic multi-slab ghosts and spherical void catalogues.
// The brute-force reference deliberately replicates the small test catalogue.
#include <FML/VoidFinders/SphericalVoidFinder.h>
#include <iomanip>
#include <iostream>
#include <set>

using namespace FML::VOIDFINDER;
int fails = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    if (FML::ThisTask == 0) std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
    fails += bad;
}
struct Record { double x; int id; };

// Independent image-intersection predicate, rather than the transport's distance function.
bool included(double x, double lo, double hi, double buffer) {
    if (lo == hi) return false;
    for (int shift = -2; shift <= 2; ++shift)
        if (x + shift >= lo - buffer && x + shift <= hi + buffer) return true;
    return false;
}

void test_exchange() {
    std::vector<Record> all;
    for (int i = 0; i < 128; ++i) all.push_back({i / 128.0, i});
    all.push_back({std::nextafter(1.0, 0.0), 128});
    auto run = [&](double buffer, bool sparse, const std::string & label) {
        std::vector<Record> local;
        std::set<int> expected;
        for (const auto & r : all) {
            if (sparse && r.x >= .5) continue; // empty source ranks
            const bool owned = r.x >= FML::xmin_domain && r.x < FML::xmax_domain;
            if (owned) local.push_back(r);
            if (owned || included(r.x, FML::xmin_domain, FML::xmax_domain, buffer)) expected.insert(r.id);
        }
        // Rank-dependent limits force multiple whole-record rounds and zero-length edges.
        FML::COMMUNICATION::append_periodic_slab_ghosts(local, buffer,
            [](const Record & r) { return r.x; }, sizeof(Record) * (1 + FML::ThisTask % 3));
        std::set<int> actual;
        bool unchanged = true;
        for (const auto & r : local) {
            actual.insert(r.id);
            unchanged &= r.x == all[r.id].x;
        }
        check(actual == expected && actual.size() == local.size() && unchanged,
              "exact ghost IDs, no duplicate images, unchanged coordinates: " + label);
        return local.size();
    };
    run(0, false, "zero buffer and exact boundaries");
    run(.03125, false, "narrow periodic buffer");
    const auto retained = run(.2 + 1.0 / 16, false, "buffer spans multiple thin slabs");
    check(FML::NTasks < 4 || retained < all.size(), "wide buffer avoids full replication on thin slabs");
    run(.49, false, "overlapping left and right images");
    run(1.2, false, "box-wide buffer retains each particle once");
    run(.26, true, "empty source ranks");
    run(.01 + .03 * FML::ThisTask, false, "rank-dependent radii");
    std::vector<Record> empty;
    FML::COMMUNICATION::append_periodic_slab_ghosts(empty, .3, [](const Record & r) { return r.x; }, sizeof(Record));
    check(empty.empty(), "all ranks empty");
    const double saved_lo = FML::xmin_domain, saved_hi = FML::xmax_domain;
    // Nonuniform domains, including an empty final slab for MPI builds.
    auto boundary = [](int task) {
        if (FML::NTasks == 1) return double(task);
        const double t = std::min(task, FML::NTasks - 1) / double(FML::NTasks - 1);
        return t * t;
    };
    FML::xmin_domain = boundary(FML::ThisTask);
    FML::xmax_domain = boundary(FML::ThisTask + 1);
    run(.17, false, "nonuniform and empty slabs");
    FML::xmin_domain = saved_lo;
    FML::xmax_domain = saved_hi;
}

struct Particle {
    double pos[3];
    double mass;
    double * get_pos() { return pos; }
    double get_mass() const { return mass; }
    constexpr int get_ndim() const { return 3; }
};

std::vector<Void> brute_force(const std::vector<Particle> & all, std::vector<Particle> & local,
                              const SphericalOptions & options) {
    FML::GRID::FFTWGrid<3> grid;
    void_density_field(local.data(), local.size(), options.grid, grid);
    const int n = grid.get_nmesh();
    std::vector<double> density(size_t(n) * n * n, 0);
    for (int ix = 0; ix < grid.get_local_nx(); ++ix)
        for (int iy = 0; iy < n; ++iy)
            for (int iz = 0; iz < n; ++iz)
                density[(size_t(ix + grid.get_local_x_start()) * n + iy) * n + iz] = grid.get_real({ix, iy, iz});
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, density.data(), int(density.size()), MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    double total_mass = 0;
    for (const auto & p : all) total_mass += p.mass;
    const double separation = std::cbrt(1.0 / all.size());
    const double rmax = std::min(.2, options.max_radius_in_mean_separation * separation);
    const double rmin = options.min_radius_in_mean_separation * separation;
    const double max_drift = options.recentering_steps > 0 ? .5 * rmax : 0;
    const double threshold = (1 + options.delta_threshold) * total_mass * 4 * M_PI / 3;
    using Sphere = detail::SphereCandidate;
    auto radius = [&](const double * x, double & mass, long long & count) {
        std::vector<std::pair<double, double>> distances;
        for (const auto & p : all) {
            double d2 = 0;
            for (int d = 0; d < 3; ++d) {
                double dx = p.pos[d] - x[d];
                dx -= std::round(dx);
                d2 += dx * dx;
            }
            if (d2 <= rmax * rmax) distances.push_back({std::sqrt(d2), p.mass});
        }
        std::sort(distances.begin(), distances.end());
        mass = 0; count = 0;
        for (const auto & dm : distances) {
            if (mass + dm.second > threshold * dm.first * dm.first * dm.first) return dm.first;
            mass += dm.second;
            ++count;
        }
        return 0.0;
    };
    std::vector<Sphere> spheres;
    for (int ix = 0; ix < n; ++ix)
        for (int iy = 0; iy < n; ++iy)
            for (int iz = 0; iz < n; ++iz) {
                const size_t index = (size_t(ix) * n + iy) * n + iz;
                if (density[index] >= options.candidate_density_cut) continue;
                Sphere c{};
                c.index = index;
                c.candidate_density = density[index];
                c.center[0] = c.candidate[0] = double(ix) / n;
                c.center[1] = c.candidate[1] = double(iy) / n;
                c.center[2] = c.candidate[2] = double(iz) / n;
                c.radius = radius(c.center, c.mass_inside, c.n_inside);
                if (c.radius > 0 && options.recentering_steps > 0) {
                    std::mt19937_64 rng(1234567ULL + 7919ULL * c.index);
                    std::uniform_real_distribution<double> uniform(-1, 1);
                    for (int step = 0; step < options.recentering_steps; ++step) {
                        double u[3], u2;
                        do {
                            u2 = 0;
                            for (auto & v : u) { v = uniform(rng); u2 += v * v; }
                        } while (u2 > 1);
                        double trial[3], drift2 = 0;
                        for (int d = 0; d < 3; ++d) {
                            trial[d] = c.center[d] + .25 * c.radius * u[d];
                            double dx = trial[d] - c.candidate[d];
                            dx -= std::round(dx);
                            drift2 += dx * dx;
                            trial[d] -= std::floor(trial[d]);
                            if (trial[d] >= 1) trial[d] = 0;
                        }
                        if (drift2 > max_drift * max_drift) continue;
                        double mass;
                        long long count;
                        const double r = radius(trial, mass, count);
                        if (r > c.radius) {
                            std::copy(trial, trial + 3, c.center);
                            c.radius = r; c.mass_inside = mass; c.n_inside = count;
                        }
                    }
                }
                if (c.radius > 0 && c.radius >= rmin) spheres.push_back(c);
            }
    std::sort(spheres.begin(), spheres.end(), [](const Sphere & a, const Sphere & b) {
        if (a.radius != b.radius) return a.radius > b.radius;
        if (a.candidate_density != b.candidate_density) return a.candidate_density < b.candidate_density;
        return a.index < b.index;
    });
    std::vector<Void> result;
    for (const auto & c : spheres) {
        bool overlap = false;
        for (const auto & kept : result) {
            double d2 = 0;
            for (int d = 0; d < 3; ++d) {
                double dx = c.center[d] - kept.pos[d];
                dx -= std::round(dx);
                d2 += dx * dx;
            }
            const double dmin = (1 - options.max_overlap) * (c.radius + kept.radius);
            if (d2 < dmin * dmin) { overlap = true; break; }
        }
        if (overlap) continue;
        Void v;
        std::copy(c.center, c.center + 3, v.pos.begin());
        v.pos_barycenter = v.pos;
        std::copy(c.candidate, c.candidate + 3, v.pos_min.begin());
        v.radius = c.radius;
        v.volume = 4 * M_PI / 3 * c.radius * c.radius * c.radius;
        v.density_min = c.candidate_density;
        v.density_avg = c.mass_inside / v.volume / total_mass;
        v.nmembers = c.n_inside;
        v.id = result.size();
        result.push_back(v);
    }
    return result;
}

void test_catalogue() {
    std::vector<Particle> all, local;
    for (int ix = 0; ix < 12; ++ix)
        for (int iy = 0; iy < 12; ++iy)
            for (int iz = 0; iz < 12; ++iz) {
                Particle p{{(ix + .37) / 12, (iy + .41) / 12, (iz + .29) / 12}, 1};
                bool hole = false;
                for (const auto & center : {std::array<double, 3>{.99, .5, .5}, {.5, .25, .7}}) {
                    double d2 = 0;
                    for (int d = 0; d < 3; ++d) {
                        double dx = p.pos[d] - center[d]; dx -= std::round(dx); d2 += dx * dx;
                    }
                    hole |= d2 < .14 * .14;
                }
                if (!hole) all.push_back(p);
            }
    for (bool weighted : {false, true}) {
        for (size_t i = 0; i < all.size(); ++i) all[i].mass = weighted ? (i % 3 ? .5 : 2) : 1;
        local.clear();
        for (const auto & p : all)
            if (p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain) local.push_back(p);
        for (double max_radius : {8.0, .6}) for (int drift : {0, 6}) {
            SphericalOptions options;
            options.grid.Nmesh = 16;
            options.grid.smoothing_scale = 0;
            options.grid.density_assignment_method = "NGP";
            options.max_radius_in_mean_separation = max_radius;
            options.min_radius_in_mean_separation = 0;
            options.recentering_steps = drift;
            const auto expected = brute_force(all, local, options);
            const auto actual = SphericalVoidFinder(local.data(), local.size(), options);
            bool same = !expected.empty() && expected.size() == actual.size();
            if (same) for (size_t i = 0; i < actual.size(); ++i) {
                const auto & a = actual[i]; const auto & b = expected[i];
                same &= a.id == b.id && a.nmembers == b.nmembers;
                same &= a.pos == b.pos && a.pos_min == b.pos_min && a.pos_barycenter == b.pos_barycenter;
                same &= std::abs(a.radius - b.radius) < 1e-13 && std::abs(a.volume - b.volume) < 1e-13;
                same &= std::abs(a.density_min - b.density_min) < 1e-12 && std::abs(a.density_avg - b.density_avg) < 1e-12;
            }
            check(same, std::string("brute-force catalogue including masses, counts and overlap: ") +
                        (weighted ? "weighted" : "unit mass") + ", recenter=" + std::to_string(drift) +
                        ", max separation=" + std::to_string(max_radius));
        }
    }
}
int main(int argc, char ** argv) {
    test_exchange();
    if (argc == 1 || std::string(argv[1]) != "exchange") test_catalogue();
    if (FML::ThisTask == 0) std::cout << (fails ? "Spherical void tests FAILED\n" : "All spherical void tests passed\n");
    return fails ? 1 : 0;
}
