#include <FML/PhaseSpaceHaloFinder/Distributed.h>
#include <FML/PhaseSpaceHaloFinder/Adapters.h>
#include "Plummer.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
using namespace FML::PHASE_SPACE_HALOS;
int failures = 0;
void check(bool valid, const char * label) {
    int bad = !valid;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "FAIL " : "PASS ") << label << '\n';
}
bool near(double a, double b, double rel = 1e-10, double floor = 1) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= rel * std::max({floor, std::abs(a), std::abs(b)});
}
std::vector<Particle> cloud(std::size_t n,
                            Vector centre,
                            Vector bulk,
                            double radius,
                            double dispersion,
                            ID first = 0,
                            double particle_mass = 1) {
    std::mt19937_64 rng(first + 123);
    std::normal_distribution<double> normal(0, 1);
    std::vector<Particle> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i].id = first + i;
        p[i].mass = particle_mass;
        for (int d = 0; d < 3; ++d) {
            p[i].position[d] = detail::wrap(centre[d] + radius * normal(rng), 1);
            p[i].velocity[d] = bulk[d] + dispersion * normal(rng);
        }
    }
    return p;
}
std::vector<Particle> overlap() {
    auto p = cloud(200, {.5, .5, .5}, {-20, 0, 0}, .01, .1);
    auto q = cloud(200, {.5, .5, .5}, {20, 0, 0}, .01, .1, 200);
    p.insert(p.end(), q.begin(), q.end());
    return p;
}
Options options() {
    Options o;
    o.gravitational_constant = .01;
    o.potential_opening_angle = 0;
    return o;
}
bool same_membership(const Catalogue & a, const Catalogue & b) {
    if (a.halos.size() != b.halos.size() || a.assignments.size() != b.assignments.size())
        return false;
    for (std::size_t i = 0; i < a.halos.size(); ++i)
        if (a.halos[i].id != b.halos[i].id || a.halos[i].particle_ids != b.halos[i].particle_ids ||
            a.halos[i].host != b.halos[i].host)
            return false;
    for (std::size_t i = 0; i < a.assignments.size(); ++i)
        if (a.assignments[i].particle != b.assignments[i].particle || a.assignments[i].halo != b.assignments[i].halo ||
            a.assignments[i].fof != b.assignments[i].fof)
            return false;
    return true;
}
void potential() {
    auto p = cloud(300, {.5, .5, .5}, {0, 0, 0}, .03, .01);
    p[1].position = p[0].position;
    p[0].mass = 2;
    const auto exact = BindingPotentialTree(p, .01, .7, 0).evaluate(),
               approx = BindingPotentialTree(p, .01, .7, .2).evaluate();
    bool good = true;
    double maximum = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        double expected = 0;
        for (std::size_t j = 0; j < p.size(); ++j)
            if (i != j) {
                auto dx = detail::difference(p[i].position, p[j].position);
                const double r = std::sqrt(detail::dot(dx, dx));
                expected -= .7 * p[j].mass * FML::SHORT_RANGE_GRAVITY::newtonian_spline_green(r, .01);
            }
        good &= near(exact[i], expected, 3e-14);
        maximum = std::max(maximum, std::abs(approx[i] - expected) / std::abs(expected));
    }
    check(good && maximum < 1e-3, "isolated unsplit binding potentials exclude self terms and converge to direct sums");
    std::vector<Particle> pair{{0, {0, 0, 0}, {}, 2}, {1, {1, 0, 0}, {}, 3}};
    const auto phi = BindingPotentialTree(pair, .1, 1, 0).evaluate();
    check(phi[0] == -3 && phi[1] == -2, "analytic unequal-mass pair binding potential");
}
void recovery() {
    auto p = overlap();
    auto o = options();
    const auto original = p;
    const auto c = find(p, o);
    bool good = c.halos.size() == 2;
    if (FML::ThisTask == 0) {
        std::cout << "OVERLAP halos=" << c.halos.size() << " refinements=" << c.statistics.refinement_nodes
                  << " merges=" << c.statistics.seed_merges << '\n';
        for (const auto & h : c.halos)
            std::cout << " halo " << h.id << " n=" << h.particle_ids.size() << " vx=" << h.velocity[0] << '\n';
    }
    for (const auto & h : c.halos) {
        std::size_t left = 0, right = 0;
        for (auto id : h.particle_ids)
            (id < 200 ? left : right)++;
        good &= std::max(left, right) >= 180 && std::min(left, right) < 10 && h.bound_mass >= 180 && h.vmax > 0;
    }
    check(good && c.statistics.refinement_nodes > 1,
          "6D hierarchy separates spatially overlapping velocity populations");
    std::reverse(p.begin(), p.end());
    const auto reversed = find(p, o);
    check(same_membership(c, reversed), "particle permutation preserves halo IDs and membership");
    p = original;
    for (auto & x : p) {
        for (auto & v : x.velocity)
            v += 1000;
        for (auto & v : x.position)
            v = detail::wrap(v + .49, 1);
    }
    const auto shifted = find(p, o);
    check(same_membership(c, shifted), "uniform velocity boosts and periodic translations preserve membership");
#ifdef USE_OMP
    const int threads = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto serial = find(original, o);
    omp_set_num_threads(threads);
    check(same_membership(c, serial), "thread-count invariant membership");
#endif
    auto single = cloud(300, {.3, .4, .5}, {0, 0, 0}, .015, .1, 1000);
    const auto one = find(single, o);
    if (FML::ThisTask == 0)
        std::cout << "SINGLE halos=" << one.halos.size() << '\n';
    check(one.halos.size() == 1 && one.halos.front().particle_ids.size() >= 270,
          "single smooth bound halo is retained without spurious resolved fragments");
}
void substructure() {
    auto host = cloud(800, {.5, .5, .5}, {0, 0, 0}, .04, .1, 0);
    auto sub = cloud(150, {.54, .5, .5}, {5, 0, 0}, .005, .05, 10000);
    auto all = host;
    all.insert(all.end(), sub.begin(), sub.end());
    for (int i = 0; i < 5; ++i) {
        auto tail = sub[i];
        tail.id = 20000 + i;
        tail.velocity[0] = 50 + 5 * i;
        all.push_back(tail);
    }
    const auto c = find(all, options());
    const Halo * main = nullptr;
    const Halo * satellite = nullptr;
    for (const auto & h : c.halos) {
        std::size_t n_host = 0, n_sub = 0;
        for (auto id : h.particle_ids) {
            n_host += id < 800;
            n_sub += id >= 10000 && id < 10150;
        }
        if (n_host > 600)
            main = &h;
        if (n_sub > 130)
            satellite = &h;
    }
    bool good = main && satellite;
    if (good)
        good = satellite->host == main->id &&
               near(main->inclusive_bound_mass, main->bound_mass + satellite->bound_mass) &&
               satellite->binding_performed && satellite->unbinding_converged;
    for (const auto & a : c.assignments)
        if (a.particle >= 20000)
            good &= a.halo == no_halo;
    if (FML::ThisTask == 0)
        std::cout << "SUBSTRUCTURE halos=" << c.halos.size() << " host=" << (main ? main->bound_mass : 0)
                  << " sub=" << (satellite ? satellite->bound_mass : 0) << '\n';
    check(good, "compact subhalo survives host overlap while high-energy tidal debris is removed");
}
void unbinding() {
    auto p = cloud(150, {.4, .5, .6}, {0, 0, 0}, .01, .05);
    Particle fast = p[0];
    fast.id = 10000;
    fast.velocity = {100, 0, 0};
    p.push_back(fast);
    auto o = options();
    auto c = find(p, o);
    bool rejected = false;
    for (const auto & a : c.assignments)
        if (a.particle == fast.id)
            rejected = a.halo == no_halo;
    check(rejected && !c.halos.empty(), "high-energy interloper is excluded from bound membership");
    auto limited = o;
    limited.max_unbinding_iterations = 1;
    bool failed = false;
    try {
        find(p, limited);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed, "unbinding iteration cap reports nonconvergence explicitly");
    // Pair energy has a known boundary; bypass refinement by making all particles coincident in phase-space scale.
    std::vector<Particle> pair{{0, {.49, .5, .5}, {-.1, 0, 0}, 1}, {1, {.51, .5, .5}, {.1, 0, 0}, 1}};
    o.minimum_particles = o.minimum_seed_particles = 2;
    o.linking_length = .4;
    o.minimum_bound_fraction = 0;
    o.gravitational_constant = 1;
    const auto bound = find(pair, o);
    pair[0].velocity[0] = -20;
    pair[1].velocity[0] = 20;
    const auto escaping = find(pair, o);
    check(bound.halos.size() == 1 && escaping.halos.empty(), "analytic two-particle bound/unbound classification");
    check(bound.halos.size() == 1 && near(bound.halos[0].bound_mass, 2) && near(bound.halos[0].rmax, .01) &&
              near(bound.halos[0].vmax, std::sqrt(200.)),
          "bound mass and equal-radius circular-velocity shells match analytic values");
    o.hubble_velocity_gradient = 200;
    pair[0].velocity = {0, 0, 0};
    pair[1].velocity = {0, 0, 0};
    o.gravitational_constant = .01;
    const auto hubble = find(pair, o);
    o.hubble_velocity_gradient = 0;
    const auto no_hubble = find(pair, o);
    check(hubble.halos.empty() && no_hubble.halos.size() == 1,
          "optional Hubble flow enters kinetic energy in explicit velocity/length units");
}
void unequal_mass_unbinding() {
    // Removing two massive interlopers must also remove their tree-approximation error.
    // Exact subtraction from an approximate monopole leaves a positive potential in
    // the left cloud and used to discard this entirely cold, self-bound remnant.
    std::vector<Particle> p;
    for (int i = 0; i < 32; ++i)
        p.push_back({ID(i), {.3 + 1e-4 * i, .5, .5}, {}, 1});
    for (int i = 0; i < 32; ++i)
        p.push_back({ID(32 + i), {.48 + .04 * i / 31., .5, .5}, {}, (i == 0 || i == 31) ? 1e6 : 1});
    p[32].velocity = {1e6, 0, 0};
    p[63].velocity = {-1e6, 0, 0};
    Options o;
    o.minimum_seed_particles = o.minimum_particles = 62;
    o.minimum_bound_fraction = 0;
    o.linking_length = 1;
    o.potential_opening_angle = 0;
    const auto exact = find(p, o);
    bool good = exact.halos.size() == 1 && exact.halos.front().particle_ids.size() == 62 &&
                exact.statistics.unbound_particles == 2;
    for (double angle : {.3, .7}) {
        o.potential_opening_angle = angle;
        const auto approximate = find(p, o);
        good &= same_membership(exact, approximate) && approximate.statistics.unbound_particles == 2;
    }
    check(good, "approximate unbinding removes massive interlopers without losing the cold bound remnant");
}
void invariants() {
    auto p = overlap();
    auto o = options();
    const auto base = find(p, o);
    auto rotated = p;
    const double angle = .37, c = std::cos(angle), s = std::sin(angle);
    for (auto & x : rotated) {
        const double px = x.position[0] - .5, py = x.position[1] - .5, vx = x.velocity[0], vy = x.velocity[1];
        x.position[0] = .5 + c * px - s * py;
        x.position[1] = .5 + s * px + c * py;
        x.velocity[0] = c * vx - s * vy;
        x.velocity[1] = s * vx + c * vy;
    }
    check(same_membership(base, find(rotated, o)),
          "rotation of positions and velocities preserves phase-space membership");
    for (auto & x : p) {
        for (auto & r : x.position)
            r *= 10;
        x.mass *= 5;
    }
    o.boxsize *= 10;
    o.softening_support *= 10;
    o.gravitational_constant *= 2; // G*m/r unchanged.
    const auto scaled = find(p, o);
    check(same_membership(base, scaled), "length/mass unit conversion preserves physical membership");
    check(find({}, options()).halos.empty(), "empty catalogue");
    bool invalid = false;
    try {
        auto bad = overlap();
        bad[1].id = bad[0].id;
        find(bad, options());
    } catch (const std::invalid_argument &) {
        invalid = true;
    }
    check(invalid, "duplicate IDs are rejected");
    auto limit = options();
    limit.max_neighbor_visits = 1;
    invalid = false;
    try {
        find(overlap(), limit);
    } catch (const std::length_error &) {
        invalid = true;
    }
    check(invalid, "work budget failure is explicit rather than silently truncating neighbours");
}
void distributed() {
    auto all = overlap();
    auto extra = cloud(120, {.99, .2, .2}, {0, 4, 0}, .006, .05, 10000);
    all.insert(all.end(), extra.begin(), extra.end());
    auto o = options();
    o.max_message_bytes = 256;
    const auto reference = find(all, o);
    std::vector<Particle> local;
    for (const auto & p : all)
        if (p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain)
            local.push_back(p);
    const auto c = find_distributed(local, o);
    bool good = true;
    std::uint64_t halos = c.halos.size();
    FML::SumOverTasks(&halos);
    good &= halos == reference.halos.size();
    for (const auto & h : c.halos) {
        const auto it =
            std::find_if(reference.halos.begin(), reference.halos.end(), [&](const auto & x) { return x.id == h.id; });
        good &= it != reference.halos.end();
        if (it != reference.halos.end())
            good &= it->particle_ids == h.particle_ids && it->host == h.host && near(it->bound_mass, h.bound_mass);
    }
    for (const auto & a : c.assignments) {
        const auto it = std::lower_bound(reference.assignments.begin(),
                                         reference.assignments.end(),
                                         a.particle,
                                         [](const auto & x, ID id) { return x.particle < id; });
        good &=
            it != reference.assignments.end() && it->particle == a.particle && it->halo == a.halo && it->fof == a.fof;
    }
    check(good && c.assignments.size() == local.size(),
          "MPI complete-parent routing and local membership match serial, including periodic/empty slabs and chunks");
    bool rejected = true;
    if (FML::NTasks > 1) {
        auto mismatch = o;
        mismatch.retained_fraction = FML::ThisTask == 0 ? .6 : .7;
        try {
            find_distributed(local, mismatch);
            rejected = false;
        } catch (const std::runtime_error &) {
        }
    }
    check(rejected, "MPI options mismatch is rejected collectively");
    auto duplicate = local;
    if (FML::NTasks > 1) {
        Particle x;
        x.id = 777777;
        x.position = {(FML::xmin_domain + FML::xmax_domain) / 2, .8, .8};
        duplicate.push_back(x);
        rejected = false;
        try {
            find_distributed(duplicate, o);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
    }
    check(rejected, "distributed duplicate-ID validation does not replicate particle payloads");
    std::vector<Particle> chain;
    for (ID i = 0; i < 16; ++i)
        chain.push_back({i, {.1 + .018 * double(i), .2, .2}, {}, 1});
    Options wide = options();
    wide.minimum_particles = wide.minimum_seed_particles = 2;
    wide.linking_length = .9;
    wide.unbind = false;
    wide.max_message_bytes = 128;
    const auto wide_reference = find(chain, wide);
    std::vector<Particle> chain_local;
    for (const auto & p : chain)
        if (p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain)
            chain_local.push_back(p);
    const auto wide_distributed = find_distributed(chain_local, wide);
    good = true;
    for (const auto & a : wide_distributed.assignments) {
        const auto it = std::find_if(wide_reference.assignments.begin(),
                                     wide_reference.assignments.end(),
                                     [&](const auto & b) { return b.particle == a.particle; });
        good &= it != wide_reference.assignments.end() && it->fof == a.fof && it->halo == a.halo;
    }
    check(good, "FoF buffers spanning several slabs converge without truncated parents");
    check(find_distributed({}, o).assignments.empty(), "globally empty distributed catalogue");
    auto capped = o;
    capped.max_parent_particles = 30;
    rejected = false;
    try {
        find_distributed(local, capped);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "parent-size limit fails collectively before returning a partial catalogue");
    const auto routed = so_seeds_by_slab(c, o);
    good = true;
    for (const auto & seed : routed)
        good &= seed.position[0] >= FML::xmin_domain && seed.position[0] < FML::xmax_domain;
    std::uint64_t nseeds = routed.size();
    FML::SumOverTasks(&nseeds);
    check(good && nseeds == halos, "SO seed adapter restores spatial ownership after parent-owner analysis");
}
struct FMLParticle {
    std::uint64_t id = 0;
    Vector x{}, v{};
    double mass = 1;
    double * get_pos() { return x.data(); }
    double * get_vel() { return v.data(); }
    constexpr int get_ndim() const { return 3; }
    std::uint64_t get_id() const { return id; }
    double get_mass() const { return mass; }
};
void adapters() {
    FMLParticle imported;
    imported.id = 83;
    imported.x = {.1, .2, .3};
    imported.v = {1, 2, 3};
    imported.mass = 4;
    const auto converted = import_particles(&imported, 1, 10, 2, 3);
    auto c = find(overlap(), options());
    bool good = !c.halos.empty() && converted.size() == 1 && converted[0].id == 83 && converted[0].position[0] == 1 &&
                converted[0].velocity[0] == 2 && converted[0].mass == 12;
    const auto seeds = so_seeds(c, 1);
    good &= seeds.size() == c.halos.size();
    if (good) {
        const auto & h = c.halos[0];
        const auto hod = hod_halo(h, 1e12, .2, 8);
        good &= hod.mass == 1e12 && hod.radius == .2 && hod.id == h.id;
    }
    FML::HALO_HISTORY::Metadata metadata;
    metadata.mass_definition = "exclusive_bound";
    metadata.id_namespace = "phase-test";
    const auto snapshot = history_snapshot(c, 0, 1, 1, metadata);
    good &= snapshot.halos.size() == c.halos.size() && snapshot.membership == FML::HALO_HISTORY::Membership::Exclusive;
    check(good, "SO seed, explicit HOD mass/radius and exclusive halo-history adapters");
}
void plummer_recovery() {
    for (std::size_t n : {1000, 4000}) {
        auto p = phase_test::plummer(n);
        Options o;
        o.softening_support = .0005;
        o.potential_opening_angle = .2;
        const auto c = find(p, o);
        const auto largest = std::max_element(c.halos.begin(), c.halos.end(), [](const auto & a, const auto & b) {
            return a.bound_mass < b.bound_mass;
        });
        const double expected_v = std::sqrt(2 / (3 * std::sqrt(3.) * .015));
        bool good = largest != c.halos.end();
        if (good) {
            good = largest->bound_mass > .9 && std::abs(largest->vmax / expected_v - 1) < .15;
            if (FML::ThisTask == 0)
                std::cout << "PLUMMER n=" << n << " halos=" << c.halos.size() << " mass=" << largest->bound_mass
                          << " vmax=" << largest->vmax << " expected=" << expected_v << '\n';
        }
        check(good, "Plummer mass and circular velocity recover the analytic profile at two resolutions");
        std::vector<Particle> local;
        for (const auto & particle : p)
            if (particle.position[0] >= FML::xmin_domain && particle.position[0] < FML::xmax_domain)
                local.push_back(particle);
        const auto distributed = find_distributed(local, o);
        good = true;
        for (const auto & h : distributed.halos) {
            const auto it = std::find_if(c.halos.begin(), c.halos.end(), [&](const auto & r) { return r.id == h.id; });
            good &= it != c.halos.end();
            if (it != c.halos.end())
                good &= h.particle_ids == it->particle_ids;
        }
        std::uint64_t count = distributed.halos.size();
        FML::SumOverTasks(&count);
        check(good && count == c.halos.size(),
              "Plummer distributed particle memberships agree with the complete serial catalogue");
    }
}
// Planted Plummer haloes (G = 1) with a portable generator (std:: distributions differ between libraries)
struct PortableRandom {
    std::uint64_t state;
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    double normal() {
        const double a = std::max(1e-300, uniform()), b = uniform();
        return std::sqrt(-2 * std::log(a)) * std::cos(2 * std::acos(-1.0) * b);
    }
};
void planted_plummer(std::vector<Particle> & p,
                     PortableRandom & rng,
                     std::size_t n,
                     double particle_mass,
                     double scale,
                     Vector centre,
                     Vector bulk) {
    const double total = n * particle_mass, cut = 10 * scale,
                 umax = std::pow(cut / std::sqrt(cut * cut + scale * scale), 3);
    for (std::size_t i = 0; i < n; ++i) {
        Particle q;
        q.id = p.size();
        q.mass = particle_mass;
        const double u = std::max(1e-15, rng.uniform()) * umax, r = scale / std::sqrt(std::pow(u, -2. / 3) - 1);
        Vector d{rng.normal(), rng.normal(), rng.normal()};
        double norm = std::sqrt(detail::dot(d, d));
        for (int k = 0; k < 3; ++k)
            q.position[k] = centre[k] + r * d[k] / norm;
        double x;
        do {
            x = rng.uniform();
        } while (.1 * rng.uniform() > x * x * std::pow(1 - x * x, 3.5));
        const double v = x * std::sqrt(2 * total / std::sqrt(r * r + scale * scale));
        d = {rng.normal(), rng.normal(), rng.normal()};
        norm = std::sqrt(detail::dot(d, d));
        for (int k = 0; k < 3; ++k)
            q.velocity[k] = bulk[k] + v * d[k] / norm;
        p.push_back(q);
    }
}
void planted_subhalo() {
    // Host (8000 particles), two field haloes, a 800-particle subhalo on a circular orbit at 2.5 host scale radii
    // and 20000 background particles. Seeds were merged with a symmetric, transitive union of all Poisson-consistent
    // pairs: a small host fragment between host and subhalo was consistent with both and merged the subhalo into
    // the host. Over 48 such realizations (subhalo radius .03-.08, 400 or 800 particles) the subhalo was found in
    // 34 before and 40 after making each seed merge into at most one larger seed.
    PortableRandom rng{4};
    std::vector<Particle> p;
    const double mp = 1e-4, r0 = .05, a0 = .02, host = 8000 * mp;
    const double vc = std::sqrt(host * std::pow(r0, 2) / std::pow(r0 * r0 + a0 * a0, 1.5));
    planted_plummer(p, rng, 8000, mp, a0, {.3, .3, .3}, {0, 0, 0});
    planted_plummer(p, rng, 2400, mp, .012, {.7, .3, .5}, {0, 0, 0});
    planted_plummer(p, rng, 800, mp, .008, {.5, .7, .7}, {0, 0, 0});
    const ID sub_first = p.size();
    planted_plummer(p, rng, 800, mp, .004, {.3 + r0, .3, .3}, {0, vc, 0});
    const ID sub_end = p.size();
    for (int i = 0; i < 20000; ++i) {
        Particle q;
        q.id = p.size();
        q.mass = mp;
        q.position = {rng.uniform(), rng.uniform(), rng.uniform()};
        q.velocity = {.05 * rng.normal(), .05 * rng.normal(), .05 * rng.normal()};
        p.push_back(q);
    }
    Options o;
    o.softening_support = .002;
    const auto catalogue = find(p, o);
    bool found = false;
    for (const auto & h : catalogue.halos) {
        std::size_t sub = 0;
        for (auto id : h.particle_ids)
            sub += id >= sub_first && id < sub_end;
        if (sub < .9 * (sub_end - sub_first))
            continue;
        const auto host_halo = std::find_if(catalogue.halos.begin(), catalogue.halos.end(), [&](const auto & x) {
            return x.id == h.host;
        });
        found = host_halo != catalogue.halos.end() && host_halo->particle_ids.size() > 6000 &&
                std::abs(h.position[0] - (.3 + r0)) < .005;
    }
    check(catalogue.halos.size() == 4 && found,
          "planted subhalo is not merged into its host through a small host fragment (one-way seed merging)");
    // Default Vmax as ROCKSTAR: the host's members plus its subhalo's, radii floored at the softening, inside the
    // radius where the mean enclosed density exceeds the assignment density. Against a direct evaluation, and above
    // the exclusive estimate (vmax_inclusive = false), which leaves the memberships unchanged.
    const auto largest = std::max_element(catalogue.halos.begin(), catalogue.halos.end(), [](auto & a, auto & b) {
        return a.bound_mass < b.bound_mass;
    });
    std::vector<double> r;
    for (const auto & h : catalogue.halos)
        if (h.id == largest->id || h.host == largest->id)
            for (auto id : h.particle_ids) {
                const auto dx = detail::periodic_difference(p[id].position, largest->position, 1);
                r.push_back(std::max(std::sqrt(detail::dot(dx, dx)), o.softening_support));
            }
    std::sort(r.begin(), r.end());
    double v2 = 0;
    for (std::size_t i = 0; i < r.size(); ++i)
        if ((i + 1) * mp > catalogue.resolved_assignment_density * 4 * std::acos(-1.0) / 3 * std::pow(r[i], 3) &&
            (i + 1 == r.size() || r[i + 1] != r[i]))
            v2 = std::max(v2, o.gravitational_constant * (i + 1) * mp / r[i]);
    o.vmax_inclusive = false;
    const auto exclusive = find(p, o);
    const auto host_exclusive = std::find_if(exclusive.halos.begin(), exclusive.halos.end(), [&](const auto & x) {
        return x.id == largest->id;
    });
    check(std::abs(largest->vmax / std::sqrt(v2) - 1) < 1e-12 && host_exclusive != exclusive.halos.end() &&
              host_exclusive->vmax < largest->vmax && same_membership(catalogue, exclusive),
          "default Vmax as ROCKSTAR (host and subhaloes, softening floor, density limit) = direct evaluation");
}
void benchmark() {
    for (std::size_t n : {1000, 4000, 16000}) {
        auto all = phase_test::plummer(n);
        std::vector<Particle> local;
        for (const auto & p : all)
            if (p.position[0] >= FML::xmin_domain && p.position[0] < FML::xmax_domain)
                local.push_back(p);
        Options o;
        o.softening_support = .0005;
        o.potential_opening_angle = .3;
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        const auto start = std::chrono::steady_clock::now();
        const auto c = find_distributed(local, o);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        FML::MaxOverTasks(&seconds);
        std::uint64_t halos = c.halos.size(), visits = c.statistics.potential_pair_visits,
                      ghosts = c.statistics.ghost_particles, max_group = c.statistics.routed_particles;
        FML::SumOverTasks(&halos);
        FML::SumOverTasks(&visits);
        FML::MaxOverTasks(&ghosts);
        FML::MaxOverTasks(&max_group);
        double peak = FML::get_system_memory_use().second / (1024 * 1024);
        FML::MaxOverTasks(&peak);
        if (FML::ThisTask == 0)
            std::cout << "BENCH n=" << n << " ranks=" << FML::NTasks << " seconds_max=" << seconds << " halos=" << halos
                      << " potential_visits=" << visits << " ghosts_max=" << ghosts
                      << " owner_particles_max=" << max_group << " peak_rss_mib_max=" << peak << '\n';
    }
}
int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        benchmark();
        return 0;
    }
    potential();
    plummer_recovery();
    recovery();
    substructure();
    planted_subhalo();
    unbinding();
    unequal_mass_unbinding();
    invariants();
    distributed();
    adapters();
    if (FML::ThisTask == 0)
        std::cout << "Phase-space halo failures: " << failures << '\n';
    return failures ? 1 : 0;
}
