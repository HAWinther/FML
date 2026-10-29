#include <FML/Global/Global.h>
#include <FML/HOD/HOD.h>
#include <iostream>
#include <map>
using namespace FML::HOD;
int failures = 0;
void check(bool good, const std::string & label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool close(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}
bool same(const Galaxy & a, const Galaxy & b) {
    return a.halo_id == b.halo_id && a.ordinal == b.ordinal && a.central == b.central && a.position == b.position &&
           a.velocity == b.velocity;
}
void test_rng() {
    using namespace FML::RANDOM::COUNTER_BASED;
    check(philox4x32({0, 0, 0, 0}, {0, 0}) ==
              std::array<std::uint32_t, 4>{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
          "shared Philox zero known-answer vector");
    check(philox4x32({~0u, ~0u, ~0u, ~0u}, {~0u, ~0u}) ==
              std::array<std::uint32_t, 4>{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu},
          "shared Philox all-ones known-answer vector");
    check(philox4x32({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}) ==
              std::array<std::uint32_t, 4>{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u},
          "shared Philox pi known-answer vector");
    check(uniform_open(0, 0) > 0 && uniform_open(~0u, ~0u) < 1, "uniform endpoints are strictly open");
    Stream a(123, 1), b(123, 1), c(123, (std::uint64_t(1) << 63) | 1);
    bool repeat = true, different = false;
    for (int i = 0; i < 100; ++i) {
        const auto x = a.next64();
        repeat &= x == b.next64();
        different |= x != c.next64();
    }
    check(repeat && different, "stream replay preserves 64-bit object IDs");
    Stream bounded(123, 4);
    bool valid = true;
    for (int i = 0; i < 10000; ++i)
        valid &= bounded.bounded(7) < 7;
    check(valid, "bounded integer draws respect non-power-of-two ranges");
}
void test_occupations() {
    Zheng07 model;
    Halo h;
    h.mass = std::pow(10, model.log10_mmin);
    check(close(model(h).central_probability, .5) &&
              close(model(h).satellites_if_central, (h.mass - model.m0) / model.m1),
          "Zheng07 transition and conditional satellite means");
    h.mass = model.m0;
    check(model(h).satellites_if_central == 0, "satellite cutoff at M0");
    const int n = 40000;
    Options options;
    options.seed = 8934;
    for (bool conditional : {false, true}) {
        options.require_central = conditional;
        double nc = 0, ns = 0, square = 0, cross = 0;
        bool valid = true;
        for (int i = 0; i < n; ++i) {
            auto draw = sample_occupation(std::uint64_t(i) + (std::uint64_t(1) << 40), {.35, 2}, options);
            nc += draw.central;
            ns += draw.satellites;
            square += double(draw.satellites) * draw.satellites;
            cross += double(draw.central) * draw.satellites;
            if (conditional)
                valid &= draw.central || draw.satellites == 0;
        }
        nc /= n;
        ns /= n;
        square = square / n - ns * ns;
        cross = cross / n - nc * ns;
        const double expected_mean = .7, expected_variance = conditional ? 1.61 : .7,
                     expected_covariance = conditional ? .455 : 0;
        check(valid && std::abs(nc - .35) < .02 && std::abs(ns - expected_mean) < .04 &&
                  std::abs(square - expected_variance) < .1 && std::abs(cross - expected_covariance) < .04,
              std::string("Bernoulli/Poisson moments and central-satellite covariance: ") +
                  (conditional ? "conditional" : "independent"));
    }
    for (double lambda : {.2, 4.0, 10.0, 100.0, 1000.0}) {
        double mean = 0, m2 = 0;
        for (int i = 0; i < n; ++i) {
            Stream rng(91, std::uint64_t(i), 44);
            const double x = double(detail::poisson(lambda, rng));
            const double d = x - mean;
            mean += d / (i + 1);
            m2 += d * (x - mean);
        }
        check(std::abs(mean - lambda) < 8 * std::sqrt(lambda / n) &&
                  std::abs(m2 / n - lambda) < 8 * std::sqrt((lambda + 2 * lambda * lambda) / n),
              "Poisson mean/variance across product and transformed-rejection regimes, lambda=" +
                  std::to_string(lambda));
    }
    check(sample_occupation(1, {0, 100}, options).satellites == 0 &&
              sample_occupation(1, {1, 0}, options).satellites == 0,
          "zero occupation limits");
}
void test_population() {
    std::vector<Halo> halos;
    for (int i = 0; i < 200; ++i) {
        Halo h;
        h.id = (std::uint64_t(1) << 48) + i;
        h.mass = 1e14;
        h.radius = .2;
        h.position = {.01 + double(i % 5), .3, .7};
        h.velocity = {20, 30, 40};
        h.velocity_dispersion = 100;
        halos.push_back(h);
    }
    PreparedCatalog prepared(halos);
    Options options;
    options.seed = 44;
    options.boxsize = 10;
    options.central_velocity_bias = .2;
    auto occupation = [](const Halo &) { return OccupationMeans{.7, 3}; };
    auto full = populate(prepared, occupation, options);
    std::map<std::pair<std::uint64_t, std::uint64_t>, Galaxy> reference;
    for (const auto & g : full.galaxies)
        reference[{g.halo_id, g.ordinal}] = g;
    std::reverse(halos.begin(), halos.end());
    auto reversed = populate(PreparedCatalog(halos), occupation, options);
    bool replay = full.galaxies.size() == reversed.galaxies.size();
    for (const auto & g : reversed.galaxies)
        replay &= same(g, reference.at({g.halo_id, g.ordinal}));
    check(replay && reference.size() == full.galaxies.size(),
          "halo reordering preserves phase space and collision-free identity pairs");
    std::vector<Halo> local;
    for (const auto & h : halos)
        if (h.id % FML::NTasks == std::uint64_t(FML::ThisTask))
            local.push_back(h);
    auto partial = populate(PreparedCatalog(local), occupation, options);
    bool matched = true;
    for (const auto & g : partial.galaxies)
        matched &= same(g, reference.at({g.halo_id, g.ordinal}));
    unsigned long long count = partial.galaxies.size();
    FML::SumOverTasks(&count);
    check(matched && count == full.galaxies.size(), "MPI redistribution of halos preserves the complete realization");
#ifdef USE_OMP
    omp_set_num_threads(1);
    auto one = populate(prepared, occupation, options);
    omp_set_num_threads(4);
    auto four = populate(prepared, occupation, options);
    bool threads = one.galaxies.size() == four.galaxies.size();
    for (size_t i = 0; i < one.galaxies.size(); ++i)
        threads &= same(one.galaxies[i], four.galaxies[i]);
    check(threads, "population is exactly reproducible across OpenMP thread counts");
    omp_set_num_threads(2);
#endif
    auto varied_options = options;
    varied_options.satellite_velocity_bias = 0;
    auto no_vel = populate(prepared, occupation, varied_options);
    bool isolated = true;
    for (size_t i = 0; i < full.galaxies.size(); ++i)
        isolated &= full.galaxies[i].position == no_vel.galaxies[i].position &&
                    full.galaxies[i].ordinal == no_vel.galaxies[i].ordinal;
    check(isolated, "velocity changes do not consume occupation or position randomness");
    varied_options = options;
    varied_options.stream = 1;
    auto changed = populate(prepared, occupation, varied_options);
    bool differs = changed.galaxies.size() != full.galaxies.size();
    for (size_t i = 0; i < std::min(changed.galaxies.size(), full.galaxies.size()); ++i)
        differs |= !same(changed.galaxies[i], full.galaxies[i]);
    check(differs, "user stream selects a different mock");
}
double reference_nfw_mass(long double x) { return double(std::log1p(x) - x / (1 + x)); }
void test_profiles() {
    for (double c : {.0001, 1.0, 5.0, 20.0, 1e100}) {
        bool inverse = true;
        for (double u : {0.0, 1e-8, .1, .5, .9, 1.0}) {
            const double r = nfw_radius_fraction(u, c);
            inverse &= r >= 0 && r <= 1 && close(reference_nfw_mass(c * r) / reference_nfw_mass(c), u, 1e-8);
        }
        check(inverse, "truncated NFW inverse CDF, concentration=" + std::to_string(c));
    }
    bool small_limit = true;
    for (double u : {1e-12, .1, .5, .99})
        small_limit &= close(nfw_radius_fraction(u, 1e-200), std::sqrt(u), 1e-12);
    check(small_limit, "NFW inverse CDF remains normalized when the unscaled mass would underflow");
    Halo h;
    h.id = 891;
    h.radius = .4;
    h.concentration = 7;
    h.position = {.02, .97, .5};
    h.velocity = {200, -40, 60};
    h.velocity_dispersion = 120;
    Options options;
    options.boxsize = 1;
    options.satellite_velocity_bias = .8;
    const auto mock = populate(PreparedCatalog({h}), [](const Halo &) { return OccupationMeans{1, 6000}; }, options);
    std::vector<double> uniforms;
    double velocity_mean = 0, velocity_square = 0, cos2 = 0;
    bool support = true;
    for (const auto & g : mock.galaxies)
        if (!g.central) {
            double r2 = 0;
            Vector offset;
            for (int d = 0; d < 3; ++d) {
                support &= g.position[d] >= 0 && g.position[d] < 1;
                offset[d] = g.position[d] - h.position[d];
                offset[d] -= std::round(offset[d]);
                r2 += offset[d] * offset[d];
            }
            const double r = std::sqrt(r2);
            support &= r <= h.radius;
            uniforms.push_back(reference_nfw_mass(h.concentration * r / h.radius) /
                               reference_nfw_mass(h.concentration));
            cos2 += offset[2] * offset[2] / r2;
            const double v = g.velocity[0] - h.velocity[0];
            velocity_mean += v;
            velocity_square += v * v;
        }
    std::sort(uniforms.begin(), uniforms.end());
    double ks = 0;
    for (size_t i = 0; i < uniforms.size(); ++i)
        ks = std::max(ks,
                      std::max(std::abs(uniforms[i] - double(i) / uniforms.size()),
                               std::abs(uniforms[i] - double(i + 1) / uniforms.size())));
    const double n = uniforms.size(), sigma = options.satellite_velocity_bias * h.velocity_dispersion;
    check(support && ks < 6 / std::sqrt(n) && std::abs(cos2 / n - 1.0 / 3) < .03,
          "NFW radial CDF, isotropy and periodic boundary wrapping");
    check(std::abs(velocity_mean / n) < 8 * sigma / std::sqrt(n) &&
              std::abs(velocity_square / n - sigma * sigma) < 8 * sigma * sigma * std::sqrt(2 / n),
          "satellite Gaussian velocities have the supplied biased 1D dispersion");
    auto concentrated_options = options;
    concentrated_options.satellite_concentration_factor = 2;
    const auto concentrated =
        populate(PreparedCatalog({h}), [](const Halo &) { return OccupationMeans{1, 6000}; }, concentrated_options);
    bool concentration = concentrated.galaxies.size() == mock.galaxies.size();
    for (size_t i = 0; concentration && i < mock.galaxies.size(); ++i)
        if (!mock.galaxies[i].central) {
            double old_r2 = 0, new_r2 = 0;
            for (int d = 0; d < 3; ++d) {
                double a = mock.galaxies[i].position[d] - h.position[d],
                       b = concentrated.galaxies[i].position[d] - h.position[d];
                a -= std::round(a);
                b -= std::round(b);
                old_r2 += a * a;
                new_r2 += b * b;
            }
            const double old_u = reference_nfw_mass(h.concentration * std::sqrt(old_r2) / h.radius) /
                                 reference_nfw_mass(h.concentration);
            const double new_u = reference_nfw_mass(2 * h.concentration * std::sqrt(new_r2) / h.radius) /
                                 reference_nfw_mass(2 * h.concentration);
            concentration &=
                close(old_u, new_u, 1e-10) && mock.galaxies[i].velocity == concentrated.galaxies[i].velocity;
        }
    check(concentration, "concentration bias changes radial mapping while preserving ranks and velocity streams");
    h.members = {{90, {.9, .1, .3}, {10, 20, 30}}, {3, {.1, .2, .3}, {50, 60, 70}}, {72, {.3, .4, .5}, {90, 100, 110}}};
    options.profile = SatelliteProfile::Particles;
    auto members = populate(PreparedCatalog({h}), [](const Halo &) { return OccupationMeans{1, 30}; }, options);
    std::reverse(h.members.begin(), h.members.end());
    auto reordered = populate(PreparedCatalog({h}), [](const Halo &) { return OccupationMeans{1, 30}; }, options);
    bool selected = members.galaxies.size() == reordered.galaxies.size();
    for (size_t i = 0; i < members.galaxies.size(); ++i) {
        const auto & g = members.galaxies[i];
        selected &= same(g, reordered.galaxies[i]);
        if (g.central)
            continue;
        bool found = false;
        for (const auto & p : h.members)
            if (g.position == p.position) {
                found = true;
                for (int d = 0; d < 3; ++d)
                    found &= close(g.velocity[d], h.velocity[d] + .8 * (p.velocity[d] - h.velocity[d]));
            }
        selected &= found;
    }
    check(selected, "particle satellites use member phase space and replay after member reordering");
    // Without replacement: with at least as many members as satellites no member is used twice
    h.members.clear();
    for (std::uint64_t m = 0; m < 50; ++m)
        h.members.push_back({1000 + m, {.2 + .001 * m, .3, .4}, {0, 0, 0}});
    bool distinct = true, enough = true;
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        options.seed = seed;
        Halo hs = h;
        hs.id = seed;
        auto draw = populate(PreparedCatalog({hs}), [](const Halo &) { return OccupationMeans{1, 30}; }, options);
        std::vector<double> xs;
        for (const auto & g : draw.galaxies)
            if (!g.central)
                xs.push_back(g.position[0]);
        enough &= draw.occupations[0].satellites <= 50 && xs.size() == draw.occupations[0].satellites;
        std::sort(xs.begin(), xs.end());
        distinct &= std::adjacent_find(xs.begin(), xs.end()) == xs.end();
    }
    check(enough && distinct, "particle satellites are drawn without replacement when there are enough members");
}
void test_invalid() {
    auto rejects = [](auto operation) {
        try {
            operation();
        } catch (const std::exception &) {
            return true;
        }
        return false;
    };
    Halo h;
    check(rejects([&] { PreparedCatalog p({h, h}); }), "duplicate halo IDs are rejected");
    check(rejects([&] { sample_occupation(0, {-1, 2}); }) && rejects([&] { sample_occupation(0, {1, INFINITY}); }),
          "invalid occupation probabilities/means are rejected");
    Options cap;
    cap.max_satellites_per_halo = 1;
    check(rejects([&] { sample_occupation(0, {1, 2}, cap); }),
          "resource limit fails instead of truncating an occupation");
    Options particle;
    particle.profile = SatelliteProfile::Particles;
    check(rejects(
              [&] { populate(PreparedCatalog({h}), [](const Halo &) { return OccupationMeans{1, 100}; }, particle); }),
          "particle profile rejects missing member data");
    bool draw_failed = false;
    for (std::uint64_t id = 0; id < 100 && !draw_failed; ++id) {
        try {
            (void)sample_occupation(id, {1, 1}, cap);
        } catch (const std::length_error &) {
            draw_failed = true;
        }
    }
    check(draw_failed, "realized Poisson draws above a resource limit fail instead of being clipped");
    check(populate(PreparedCatalog({}), Zheng07{}).galaxies.empty(), "empty halo catalogues are valid local inputs");
}
int main() {
    test_rng();
    test_occupations();
    test_population();
    test_profiles();
    test_invalid();
    if (FML::ThisTask == 0)
        std::cout << (failures ? "HOD tests FAILED\n" : "All HOD tests passed\n");
    return failures ? 1 : 0;
}
