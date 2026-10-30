#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <set>
using Halo = FML::FOF::SOHalo<3>;
using Point = std::array<double, 3>;
struct Particle {
    Point pos{}, vel{};
    double mass = 1;
    std::uint64_t id = 0;
    std::vector<double> payload;
    inline static bool fail_decode = false;
    bool fail_pack = false;
    double * get_pos() { return pos.data(); }
    double * get_vel() { return vel.data(); }
    double get_mass() { return mass; }
    constexpr int get_ndim() const { return 3; }
    int get_particle_byte_size() {
        return int(2 * sizeof(Point) + sizeof(mass) + 2 * sizeof(std::uint64_t) + payload.size() * sizeof(double));
    }
    void append_to_buffer(char * b) {
        if (fail_pack)
            throw std::runtime_error("injected serializer failure");
        std::uint64_t n = payload.size();
        auto put = [&](const void * data, size_t bytes) {
            std::memcpy(b, data, bytes);
            b += bytes;
        };
        put(pos.data(), sizeof(pos));
        put(vel.data(), sizeof(vel));
        put(&mass, sizeof(mass));
        put(&id, sizeof(id));
        put(&n, sizeof(n));
        if (n)
            put(payload.data(), n * sizeof(double));
    }
    void assign_from_buffer(char * b) {
        if (fail_decode)
            throw std::runtime_error("injected decoder failure");
        std::uint64_t n = 0;
        auto get = [&](void * data, size_t bytes) {
            std::memcpy(data, b, bytes);
            b += bytes;
        };
        get(pos.data(), sizeof(pos));
        get(vel.data(), sizeof(vel));
        get(&mass, sizeof(mass));
        get(&id, sizeof(id));
        get(&n, sizeof(n));
        payload.resize(size_t(n));
        if (n)
            get(payload.data(), n * sizeof(double));
    }
};
int failures = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
double wrap(double x) { return x - std::floor(x); }
double distance(const Point & x, const Point & y) {
    double value = 0;
    for (int d = 0; d < 3; ++d) {
        double dx = x[d] - y[d];
        dx -= std::round(dx);
        value += dx * dx;
    }
    return value;
}
std::vector<Particle> local(const std::vector<Particle> & all) {
    std::vector<Particle> out;
    for (const auto & p : all)
        if (p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain)
            out.push_back(p);
    return out;
}
std::vector<Particle> sample() {
    std::vector<Particle> all;
    std::mt19937_64 rng(387);
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < 256; ++i) {
        Particle p;
        p.id = i;
        p.pos = {u(rng), u(rng), u(rng)};
        p.vel = {.125 * i, .25 * (i % 7), -.5 * (i % 3)};
        p.mass = 1 + .25 * (i % 5);
        p.payload.assign(1 + i % 5, 1000 + i * .5);
        all.push_back(p);
    }
    all[0].pos[0] = 0;
    all[1].pos[0] = std::nextafter(1., 0.);
    return all;
}
// Explicit periodic image intersection; independent of the transport's distance-to-slab implementation.
bool needed(double x, double lo, double hi, double width) {
    if (lo == hi)
        return false;
    for (int image = -2; image <= 2; ++image)
        if (x + image >= lo - width && x + image <= hi + width)
            return true;
    return false;
}
void test_exchange(const std::vector<Particle> & all) {
    for (double width : {0., .03125, .18, .49, .8}) {
        auto owned = local(all);
        auto original = owned;
        double radius = width + .002 * FML::ThisTask;
        auto ghosts = FML::FOF::fetch_boundary_particles(owned.data(),
                                                         owned.size(),
                                                         radius,
                                                         128 + 32 * (FML::ThisTask % 3),
                                                         false);
        std::set<std::uint64_t> expected, actual;
        bool data = true;
        for (const auto & p : all)
            if (!(p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain) &&
                needed(p.pos[0], FML::xmin_domain, FML::xmax_domain, radius))
                expected.insert(p.id);
        for (const auto & p : ghosts) {
            actual.insert(p.id);
            const auto & reference = all.at(size_t(p.id));
            data &= p.pos == reference.pos && p.vel == reference.vel && p.mass == reference.mass &&
                    p.payload == reference.payload;
        }
        for (size_t i = 0; i < owned.size(); ++i)
            data &= owned[i].pos == original[i].pos && owned[i].payload == original[i].payload;
        check(actual == expected && actual.size() == ghosts.size() && data,
              "selective multi-slab IDs, variable serialization and unchanged wrapped coordinates, radius=" +
                  std::to_string(width));
        if (FML::NTasks >= 4 && width == .18)
            check(owned.size() + ghosts.size() < all.size(),
                  "buffer wider than thin slabs still avoids full particle replication");
        auto unwrapped = FML::FOF::fetch_boundary_particles(owned.data(), owned.size(), radius, 160, true);
        data = unwrapped.size() == ghosts.size();
        for (const auto & p : unwrapped) {
            const auto & q = all.at(size_t(p.id));
            double best = 10;
            for (int shift = -2; shift <= 2; ++shift)
                best =
                    std::min(best,
                             std::max({FML::xmin_domain - q.pos[0] - shift, q.pos[0] + shift - FML::xmax_domain, 0.}));
            double actual_distance = std::max({FML::xmin_domain - p.pos[0], p.pos[0] - FML::xmax_domain, 0.});
            data &= std::abs(wrap(p.pos[0]) - q.pos[0]) < 2e-15 && std::abs(actual_distance - best) < 2e-15 &&
                    p.payload == q.payload;
        }
        check(data, "unwrapped ghosts choose one nearest periodic image with all metadata preserved");
    }
    std::vector<Particle> empty;
    auto none = FML::FOF::fetch_boundary_particles(empty.data(), 0, .3, 128, false);
    check(none.empty(), "empty global particle catalogue participates safely");
}
// Independent all-particle reference, intentionally replicated only for the small test fixture.
Halo brute(std::vector<Particle> & all,
           Point center,
           double radius,
           const std::vector<double> & thresholds,
           double search,
           int minimum) {
    auto within = [&](double r) {
        std::vector<std::pair<double, Particle *>> out;
        for (auto & p : all) {
            double r2 = distance(p.pos, center);
            if (r2 <= r * r)
                out.emplace_back(r2, &p);
        }
        return out;
    };
    size_t initial = 0;
    radius = std::min(radius, search);
    for (int iteration = 0; iteration < 200; ++iteration) {
        auto inside = within(radius);
        if (!iteration)
            initial = inside.size();
        if (inside.empty() || inside.size() < std::max(size_t(minimum), initial / 10))
            break;
        Point shift{};
        double mass = 0;
        for (const auto & entry : inside) {
            const auto & p = *entry.second;
            mass += p.mass;
            for (int d = 0; d < 3; ++d) {
                double dx = p.pos[d] - center[d];
                dx -= std::round(dx);
                shift[d] += p.mass * dx;
            }
        }
        for (int d = 0; d < 3; ++d)
            center[d] = wrap(center[d] + shift[d] / mass);
        radius *= .9;
    }
    Halo out;
    out.pos = center;
    out.radius.resize(thresholds.size());
    out.mass.resize(thresholds.size());
    out.npart.resize(thresholds.size());
    auto inside = within(search);
    std::sort(inside.begin(), inside.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    double total = 0;
    for (const auto & p : all)
        total += p.mass;
    std::vector<double> cumulative;
    double enclosed = 0;
    for (auto entry : inside) {
        enclosed += entry.second->mass;
        cumulative.push_back(enclosed);
    }
    for (size_t t = 0; t < thresholds.size(); ++t) {
        size_t n = inside.size();
        while (n > 0 &&
               cumulative[n - 1] / (4 * M_PI / 3 * std::pow(std::sqrt(inside[n - 1].first), 3)) < thresholds[t] * total)
            --n;
        if (n == inside.size() && n) {
            out.truncated = true;
            out.radius[t] = search;
            out.mass[t] = cumulative.back();
            out.npart[t] = n;
        } else {
            out.mass[t] = n ? cumulative[n - 1] : 0;
            out.npart[t] = n;
            out.radius[t] = std::cbrt(out.mass[t] / (4 * M_PI / 3 * thresholds[t] * total));
        }
    }
    if (!thresholds.empty()) {
        double mass = 0;
        for (const auto & entry : inside)
            if (entry.first <= out.radius[0] * out.radius[0]) {
                const auto & p = *entry.second;
                mass += p.mass;
                for (int d = 0; d < 3; ++d) {
                    out.vel[d] += p.mass * p.vel[d];
                    out.vel_rms[d] += p.mass * p.vel[d] * p.vel[d];
                }
            }
        if (mass)
            for (int d = 0; d < 3; ++d) {
                out.vel[d] /= mass;
                out.vel_rms[d] = std::sqrt(std::max(0., out.vel_rms[d] / mass - out.vel[d] * out.vel[d]));
            }
    }
    return out;
}
bool same(const Halo & a, const Halo & b) {
    bool ok = a.npart == b.npart && a.truncated == b.truncated;
    for (int d = 0; d < 3; ++d) {
        double dx = a.pos[d] - b.pos[d];
        dx -= std::round(dx);
        ok &= std::abs(dx) < 1e-12 && std::abs(a.vel[d] - b.vel[d]) < 1e-10 &&
              std::abs(a.vel_rms[d] - b.vel_rms[d]) < 1e-9;
    }
    for (size_t i = 0; i < a.radius.size(); ++i)
        ok &= std::abs(a.radius[i] - b.radius[i]) < 1e-12 && std::abs(a.mass[i] - b.mass[i]) < 1e-10;
    return ok;
}
void compare(std::vector<Particle> all,
             const std::vector<Point> & candidates,
             double search,
             bool recenter,
             const std::string & name) {
    auto owned = local(all);
    std::vector<Point> centers;
    for (auto c : candidates)
        if (c[0] >= FML::xmin_domain && c[0] < FML::xmax_domain)
            centers.push_back(c);
    double total = 0;
    for (const auto & p : all)
        total += p.mass;
    std::vector<double> thresholds{2., .1, 100.};
    std::vector<double> initial(centers.size(), search * .8);
    int minimum = recenter ? 3 : 1000000;
    auto measured = FML::FOF::SphericalOverdensity<Particle, 3>(owned.data(),
                                                                owned.size(),
                                                                total,
                                                                centers,
                                                                initial,
                                                                thresholds,
                                                                search,
                                                                minimum,
                                                                160);
    bool correct = measured.size() == centers.size();
    size_t truncated = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        auto expected = brute(all, centers[i], initial[i], thresholds, search, minimum);
        correct &= same(measured[i], expected);
        truncated += expected.truncated;
    }
    FML::SumOverTasks(&truncated);
    correct &= truncated == FML::FOF::SO_number_of_truncated_halos;
    check(correct, name + ": center, radii, mass, count, velocity, dispersion and truncation match direct reference");
}
void test_measurements(std::vector<Particle> all) {
    std::vector<Point> centers{{.001, .5, .5}, {.249, .3, .7}, {.501, .7, .3}, {.751, .2, .6}, {.999, .8, .4}};
    compare(all, centers, .3, false, "wide multi-slab sphere");
    compare(all, centers, .3, true, "wide sphere with shrinking-sphere centering");
    compare(all, centers, .49, true, "overlapping periodic image regions");
    std::vector<Particle> sparse;
    for (auto p : all)
        if (p.pos[0] < .45)
            sparse.push_back(p);
    compare(sparse, centers, .32, true, "empty particle ranks");
    double boundary = 1.;
    std::array<double, 2> extent{FML::xmin_domain, FML::xmax_domain};
    for (const auto & slab : FML::GatherFromTasks(&extent))
        if (slab[1] > 0 && slab[1] < 1)
            boundary = std::min(boundary, slab[1]);
    std::vector<Particle> drift;
    for (int i = 0; i < 40; ++i) {
        Particle p;
        p.id = i;
        p.pos = {wrap(boundary + .025 + .0007 * (i % 5)), .5 + .0003 * (i % 7), .5 + .0004 * (i % 3)};
        p.mass = 1 + (i % 3) * .25;
        p.vel = {double(i % 5), double(i % 7), -double(i % 3)};
        p.payload = {double(i)};
        drift.push_back(p);
    }
    Point guess{wrap(boundary - .003), .5, .5};
    compare(drift, {guess}, .08, true, "centroid migrates beyond its owner slab");
    if (FML::NTasks > 1)
        check(FML::FOF::SO_number_of_buffer_refreshes > 0,
              "recentered growth sphere triggers a collective selective buffer refresh");
}
// Also exercise native fixed-size serialization and optional mass/velocity methods.
struct FloatParticle {
    float pos[3]{};
    std::uint64_t id = 0;
    float * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
void test_float_particles() {
    std::vector<FloatParticle> owned;
    std::vector<Particle> reference;
    std::set<std::uint64_t> expected;
    for (int i = 0; i < 32; ++i) {
        FloatParticle p{{float((i + .25) / 32), .5f, .5f}, std::uint64_t(i)};
        Particle q;
        q.pos = {p.pos[0], p.pos[1], p.pos[2]};
        reference.push_back(q);
        if (p.pos[0] >= FML::xmin_domain && p.pos[0] < FML::xmax_domain)
            owned.push_back(p);
        else if (needed(p.pos[0], FML::xmin_domain, FML::xmax_domain, .3))
            expected.insert(p.id);
    }
    const auto ghosts = FML::FOF::fetch_boundary_particles(owned.data(), owned.size(), .3, 128, false);
    std::set<std::uint64_t> actual;
    bool ok = true;
    for (const auto & p : ghosts) {
        actual.insert(p.id);
        ok &= p.id < reference.size();
        if (p.id < reference.size())
            for (int d = 0; d < 3; ++d)
                ok &= p.pos[d] == reference[p.id].pos[d];
    }
    check(ok && actual == expected && actual.size() == ghosts.size(),
          "float-position fixed-size records preserve coordinates and IDs");
    std::vector<Point> centers;
    const Point wrapped{.125, .5, .5};
    if (wrapped[0] >= FML::xmin_domain && wrapped[0] < FML::xmax_domain)
        centers.push_back({2.125, -1.5, 3.5});
    const std::vector<double> thresholds{2., 10.};
    auto halos = FML::FOF::SphericalOverdensity<FloatParticle, 3>(owned.data(),
                                                                  owned.size(),
                                                                  32.,
                                                                  centers,
                                                                  std::vector<double>(centers.size(), .2),
                                                                  thresholds,
                                                                  .3,
                                                                  1000000,
                                                                  128);
    ok = halos.size() == centers.size();
    for (const auto & h : halos)
        ok &= same(h, brute(reference, wrapped, .2, thresholds, .3, 1000000));
    check(ok,
          "float positions, unit-mass fallback, absent velocities and unwrapped seed centers match direct reference");
}
int main() {
    test_float_particles();
    auto all = sample();
    test_exchange(all);
    test_measurements(all);
    if (FML::NTasks > 1) {
        double lo = FML::xmin_domain, hi = FML::xmax_domain;
        std::vector<double> edges(FML::NTasks + 1);
        for (int t = 0; t <= FML::NTasks; ++t)
            edges[t] = double(t) / FML::NTasks;
        if (FML::NTasks >= 3) {
            edges[1] = edges[0];
            edges[2] = .2;
        }
        FML::xmin_domain = edges[FML::ThisTask];
        FML::xmax_domain = edges[FML::ThisTask + 1];
        test_exchange(all);
        test_measurements(all);
        FML::xmin_domain = lo;
        FML::xmax_domain = hi;
    }
    auto owned = local(all);
    if (FML::NTasks > 1) {
        auto throwing = owned;
        if (!FML::ThisTask)
            throwing.front().fail_pack = true;
        bool caught = false;
        try {
            FML::FOF::fetch_boundary_particles(throwing.data(), throwing.size(), .8, 160, false);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        check(caught, "one-rank serializer failure is collective");
        Particle::fail_decode = FML::ThisTask == 0;
        caught = false;
        try {
            FML::FOF::fetch_boundary_particles(owned.data(), owned.size(), .8, 160, false);
        } catch (const std::invalid_argument &) {
            caught = true;
        }
        Particle::fail_decode = false;
        check(caught, "one-rank decoder failure is collective");
    }
    bool rejected = false;
    try {
        FML::FOF::fetch_boundary_particles(owned.data(), owned.size(), .2, 8, false);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "undersized message budgets fail collectively");
    rejected = false;
    try {
        FML::FOF::fetch_boundary_particles<Particle>(nullptr, FML::ThisTask == 0 ? 1 : 0, .2, 160, false);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "one-rank null owned storage fails collectively before access");
    rejected = false;
    std::vector<Point> c{{.1, .2, .3}};
    try {
        FML::FOF::SphericalOverdensity<Particle, 3>(owned.data(), owned.size(), 1, c, {}, {2.}, .2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "invalid center/radius shapes fail collectively before indexing");
    if (!FML::ThisTask)
        std::cout << (failures ? "Multi-slab SO tests FAILED" : "All multi-slab SO tests passed") << '\n';
    return failures ? 1 : 0;
}
