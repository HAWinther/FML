#include <FML/Global/Global.h>
#include <FML/HOD/AssemblyBias.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <map>
using namespace FML::HOD;
using FML::TESTS::check;

template <class Function>
bool rejects(Function function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}
void test_means() {
    bool conserved = true, bounded = true, direction = true, neutral = true;
    for (double split : {.001, .1, .3, .5, .8, .999})
        for (double p : {0., 1e-12, .1, .4, .8, 1.})
            for (double mu : {0., 1e-12, .8, 4., 1e5})
                for (double ac : {-1., -.4, 0., .7, 1.})
                    for (double as : {-1., -.3, 0., .6, 1.}) {
                        HeavisideAssemblyBias bias{split, ac, as};
                        const auto low = bias.decorate_marginal({p, mu}, 0), high = bias.decorate_marginal({p, mu}, 1);
                        const long double pc = static_cast<long double>(split) * low.central_probability +
                                               (1.0L - split) * high.central_probability;
                        const long double ns =
                            static_cast<long double>(split) * low.satellite_mean + (1.0L - split) * high.satellite_mean;
                        conserved &= std::abs(pc - p) < 64 * std::numeric_limits<double>::epsilon() &&
                                     std::abs(ns - mu) <= 64 * std::numeric_limits<double>::epsilon() * mu;
                        bounded &= low.central_probability >= 0 && low.central_probability <= 1 &&
                                   high.central_probability >= 0 && high.central_probability <= 1 &&
                                   low.satellite_mean >= 0 && high.satellite_mean >= 0 &&
                                   std::isfinite(low.satellite_mean) && std::isfinite(high.satellite_mean);
                        direction &= ac * (high.central_probability - low.central_probability) >= 0 &&
                                     as * (high.satellite_mean - low.satellite_mean) >= 0;
                        if (ac == 0)
                            neutral &= low.central_probability == p && high.central_probability == p;
                        if (as == 0)
                            neutral &= low.satellite_mean == mu && high.satellite_mean == mu;
                    }
    check(conserved, "decorated central and marginal satellite means conserve fixed-mass population averages");
    check(bounded && direction, "decorated probabilities/means are bounded with the declared correlation sign");
    check(neutral, "zero strengths preserve each baseline marginal mean exactly");
    HeavisideAssemblyBias bias{.3, .7, -.4};
    const auto lower = bias.decorate_marginal({.4, 2}, .3),
               upper = bias.decorate_marginal({.4, 2}, std::nextafter(.3, 1.));
    check(lower.central_probability == bias.decorate_marginal({.4, 2}, 0).central_probability &&
              upper.central_probability == bias.decorate_marginal({.4, 2}, 1).central_probability,
          "percentile equal to split belongs to the low population");
    const auto adapted = bias(OccupationMeans{.4, 5}, .8), direct = bias.decorate_marginal({.4, 2}, .8);
    check(adapted.central_probability == direct.central_probability && adapted.satellite_mean == direct.satellite_mean,
          "legacy conditional-rate means are converted to marginal satellites before decoration");
    const double tiny = std::numeric_limits<double>::denorm_min();
    const auto rare = HeavisideAssemblyBias{tiny, 0, -1}.decorate_marginal({.5, tiny}, 0);
    check(rare.satellite_mean == 1, "rare-population satellite scaling avoids intermediate underflow");
    check(
        HeavisideAssemblyBias{.5, 0, 0}.decorate_marginal({.5, std::numeric_limits<double>::max()}, 0).satellite_mean ==
            std::numeric_limits<double>::max(),
        "zero satellite strength avoids an unused overflowing extreme");
}
void test_sampling() {
    constexpr int samples = 120000;
    HeavisideAssemblyBias bias{.3, .5, -.3};
    for (bool conditional : {false, true})
        for (bool high : {false, true}) {
            const auto mean = bias.decorate_marginal({.4, 1.2}, high ? 1 : 0);
            Options options;
            options.seed = 17435;
            options.require_central = conditional;
            long double moments[4]{};
            unsigned long long count = 0;
            int valid = 1;
            for (int i = 0; i < samples; ++i)
                if (i % FML::NTasks == FML::ThisTask) {
                    const auto draw = sample_marginal_occupation((std::uint64_t(1) << 60) + i, mean, options);
                    const long double c = draw.central, s = draw.satellites;
                    moments[0] += c;
                    moments[1] += s;
                    moments[2] += s * s;
                    moments[3] += c * s;
                    ++count;
                    valid &= !conditional || draw.central || draw.satellites == 0;
                }
            FML::SumArrayOverTasks(moments, 4);
            FML::SumOverTasks(&count);
            FML::MinOverTasks(&valid);
            const long double p = mean.central_probability, mu = mean.satellite_mean, rate = conditional ? mu / p : mu;
            const long double s2 = (conditional ? p : 1) * (rate * rate + rate);
            const long double s4 =
                (conditional ? p : 1) * (std::pow(rate, 4) + 6 * std::pow(rate, 3) + 7 * rate * rate + rate);
            const long double cross = conditional ? mu : p * mu, cross2 = conditional ? s2 : p * s2;
            const long double expected[4]{p, mu, s2, cross}, second[4]{p, s2, s4, cross2};
            bool matched = count == samples && valid;
            for (int j = 0; j < 4; ++j)
                matched &= std::abs(moments[j] / count - expected[j]) <
                           8 * std::sqrt(std::max(0.0L, second[j] - expected[j] * expected[j]) / count) + 1e-12;
            check(matched,
                  std::string("decorated Bernoulli/Poisson first, second and cross moments: ") +
                      (conditional ? "conditional " : "independent ") + (high ? "high" : "low"));
        }
    Options options;
    options.require_central = false;
    const auto absent = sample_marginal_occupation(0, {0, 10}, options);
    check(!absent.central && absent.satellites > 0,
          "independent marginal satellites are supported at zero central probability");
    bool replay = true;
    for (bool conditional : {false, true}) {
        options.require_central = conditional;
        for (std::uint64_t id = 0; id < 1000; ++id) {
            const auto legacy = sample_occupation(id, {.5, 4}, options),
                       marginal = sample_marginal_occupation(id, {.5, 2}, options);
            replay &= legacy.central == marginal.central && legacy.satellites == marginal.satellites;
        }
    }
    check(replay, "equivalent representable marginal/conditional means use the existing separated Philox streams");
}
void test_population() {
    std::vector<Halo> halos;
    std::map<std::uint64_t, double> percentiles;
    for (int i = 0; i < 120; ++i) {
        Halo h;
        h.id = (std::uint64_t(1) << 50) + i;
        h.mass = 1e13;
        h.radius = .1;
        h.position = {.01 * i, .2, .3};
        h.velocity = {10, 20, 30};
        h.velocity_dispersion = 30;
        halos.push_back(h);
        percentiles[h.id] = (i + .5) / 120;
    }
    const HeavisideAssemblyBias bias{.3, .5, .6};
    auto model = [&](const Halo & h) { return bias(OccupationMeans{.4, 3}, percentiles.at(h.id)); };
    Options options;
    options.seed = 91;
    options.boxsize = 2;
    options.central_velocity_bias = .2;
    const auto reference = populate(PreparedCatalog(halos), model, options);
    std::map<std::pair<std::uint64_t, std::uint64_t>, Galaxy> galaxies;
    for (const auto & g : reference.galaxies)
        galaxies[{g.halo_id, g.ordinal}] = g;
    auto equal = [&](const Mock & mock) {
        bool same = true;
        for (const auto & g : mock.galaxies) {
            const auto & expected = galaxies.at({g.halo_id, g.ordinal});
            same &= g.position == expected.position && g.velocity == expected.velocity && g.central == expected.central;
        }
        return same;
    };
    std::reverse(halos.begin(), halos.end());
    const auto reverse = populate(PreparedCatalog(halos), model, options);
    check(reverse.galaxies.size() == reference.galaxies.size() && equal(reverse),
          "decorated phase-space realization is invariant under halo reordering");
    std::vector<Halo> local;
    for (const auto & h : halos)
        if (h.id % FML::NTasks == std::uint64_t(FML::ThisTask))
            local.push_back(h);
    const auto part = populate(PreparedCatalog(local), model, options);
    unsigned long long n = part.galaxies.size();
    FML::SumOverTasks(&n);
    check(n == reference.galaxies.size() && equal(part),
          "fixed calibrated percentiles preserve the decorated realization across MPI ownership");
#ifdef USE_OMP
    omp_set_num_threads(1);
    const auto one = populate(PreparedCatalog(halos), model, options);
    omp_set_num_threads(4);
    const auto four = populate(PreparedCatalog(halos), model, options);
    check(one.galaxies.size() == four.galaxies.size() && equal(one) && equal(four),
          "decorated phase-space fill is replayable across OpenMP thread counts");
    omp_set_num_threads(2);
#endif
    options.require_central = false;
    const auto independent = populate(
        PreparedCatalog(halos),
        [](const Halo &) {
            return MarginalOccupationMeans{0, 2};
        },
        options);
    bool no_centrals = true;
    for (const auto & g : independent.galaxies)
        no_centrals &= !g.central && g.ordinal > 0;
    check(!independent.galaxies.empty() && no_centrals,
          "populate accepts independent marginal models with satellites and no centrals");
}
void test_invalid() {
    bool valid = true;
    for (int scenario = 0; scenario < 9; ++scenario) {
        HeavisideAssemblyBias bias;
        double percentile = .8;
        if (scenario == 0)
            bias.split = 0;
        if (scenario == 1)
            bias.split = 1;
        if (scenario == 2)
            bias.split = NAN;
        if (scenario == 3)
            bias.central_strength = 1.01;
        if (scenario == 4)
            bias.satellite_strength = -1.01;
        if (scenario == 5)
            bias.central_strength = NAN;
        if (scenario == 6)
            percentile = -.1;
        if (scenario == 7)
            percentile = 1.1;
        if (scenario == 8)
            percentile = NAN;
        valid &= rejects([&] { bias.decorate_marginal({.5, 1}, percentile); });
    }
    check(valid, "invalid assembly-bias parameters and percentiles are rejected");
    check(rejects([&] {
              HeavisideAssemblyBias{}.decorate_marginal({-1, 2}, .2);
          }) &&
              rejects([&] {
                  HeavisideAssemblyBias{}.decorate_marginal({.5, INFINITY}, .2);
              }) &&
              rejects([&] {
                  HeavisideAssemblyBias{}(OccupationMeans{.5, -1}, .2);
              }),
          "invalid baseline means are rejected before decoration");
    check(rejects([&] {
              HeavisideAssemblyBias{.5, 0, 1}.decorate_marginal({.5, std::numeric_limits<double>::max()}, 1);
          }),
          "unrepresentable decorated means fail explicitly");
    const auto incompatible = HeavisideAssemblyBias{.5, 1, 0}.decorate_marginal({.4, 1}, 0);
    check(incompatible.central_probability == 0 && rejects([&] { sample_marginal_occupation(0, incompatible); }),
          "conditional sampling rejects incompatible central and satellite decorations");
    Options cap;
    cap.max_satellites_per_halo = 1;
    check(rejects([&] {
              sample_marginal_occupation(0, {1e-6, 1}, cap);
          }) &&
              rejects([&] {
                  sample_marginal_occupation(0, {0, 1});
              }),
          "marginal conditional rates are validated before the random central draw");
    cap.require_central = false;
    check(rejects([&] {
              sample_marginal_occupation(0, {.2, 2}, cap);
          }) &&
              rejects([&] {
                  sample_marginal_occupation(0, {NAN, 1}, cap);
              }),
          "invalid marginal independent sampling fails without clipping");
}
int export_means(const std::string & path) {
    bool good = true;
    if (!FML::ThisTask) {
        std::ofstream out(path);
        out << std::setprecision(17) << "p,mu,split,ac,as,q,central,satellites\n";
        for (double p : {0., 1e-12, .01, .2, .5, .8, .99, 1.})
            for (double mu : {0., .01, 1., 3.})
                for (double split : {.1, .3, .5, .8, .9})
                    for (double ac : {-1., -.6, 0., .4, 1.})
                        for (double as : {-1., -.5, 0., .7, 1.})
                            for (double q : {0., split, std::nextafter(split, 1.), 1.}) {
                                const auto means = HeavisideAssemblyBias{split, ac, as}.decorate_marginal({p, mu}, q);
                                out << p << ',' << mu << ',' << split << ',' << ac << ',' << as << ',' << q << ','
                                    << means.central_probability << ',' << means.satellite_mean << '\n';
                            }
        out.close();
        good = bool(out);
    }
    check(good, "export reference means");
    return FML::TESTS::summary("assembly-bias export");
}
int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--export")
        return export_means(argv[2]);
    test_means();
    test_sampling();
    test_population();
    test_invalid();
    return FML::TESTS::summary("HOD assembly bias");
}
