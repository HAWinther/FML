#include "Fixture.h"
#include "Metric.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
// Exact normalized Fourier projections of this fixture, not a full-white jump measure.
std::array<double, 6> project(const GN::Field & v) {
    std::array<double, 6> result{};
    const std::array<std::array<int, 3>, 3> modes = {{{1, 0, 0}, {1, 1, 0}, {0, 0, 1}}};
    for (size_t a = 0; a < 3; ++a)
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y)
                for (int z = 0; z < 4; ++z) {
                    const double angle = 2 * M_PI * (modes[a][0] * x + modes[a][1] * y + modes[a][2] * z) / 4;
                    result[2 * a] += std::sqrt(2.) / 8 * v[(x * 4 + y) * 4 + z] * std::cos(angle);
                    result[2 * a + 1] += std::sqrt(2.) / 8 * v[(x * 4 + y) * 4 + z] * std::sin(angle);
                }
    return result;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 4 || FML::NTasks != 1)
            throw std::invalid_argument("gn_probe FIXTURE METRIC_DIRECTORY OUTPUT.csv, serial only");
        GN::Fixture fixture(argv[1]);
        auto model = GN::model();
        auto work = model.workspace();
        auto target = GN::FF::fixed_noise_posterior(model, fixture.data, 1000.);
        std::array<GN::Metric, 4> metrics;
        for (size_t e = 0; e < 4; ++e)
            metrics[e] =
                GN::Metric::read((std::filesystem::path(argv[2]) / ("ensemble" + std::to_string(e) + ".txt")).string());
        std::ofstream out(argv[3]);
        out << std::setprecision(17) << "ensemble,draw,trial,case,acceptance,divergences,gradient_calls,delta_H";
        for (auto name : {"cos100", "sin100", "cos110", "sin110", "cos001", "sin001"})
            out << ',' << name;
        out << '\n';
        for (const auto & f : fixture.frames)
            for (unsigned trial = 0; trial < 8; ++trial)
                for (unsigned variant = 0; variant < 2; ++variant) {
                    const auto & metric = metrics[f.ensemble];
                    if (metric.dimension != 64)
                        throw std::invalid_argument("Wrong metric dimension");
                    FML::MCMC::HMCOptions o;
                    o.warmup = 0;
                    o.samples = 1;
                    o.step_size = .075;
                    o.n_leapfrog = 16;
                    o.jitter = false;
                    o.adapt_mass = false;
                    o.adapt_step_size = false;
                    o.seed = UINT64_C(0x6100000000000000) + f.ensemble * UINT64_C(1048576) + f.draw * 16 + trial;
                    FML::MCMC::HMCMass mass;
                    mass.n_custom = 64;
                    double first_k = 0, last_k = 0, first_u = 0, last_u = 0;
                    bool first_velocity = true;
                    uint64_t calls = 0;
                    mass.apply_sqrt = [&](auto & v) {
                        v = variant ? metric.apply(v, 1., .5) : target.apply_field_metric(v, f.frozen, work, .5);
                    };
                    mass.apply_inverse = [&](auto & v) {
                        const auto p = v;
                        v = variant ? metric.apply(v, 1., -1.) : target.apply_field_metric(v, f.frozen, work, -1.);
                        last_k = .5 * GN::Metric::dot(p, v);
                        if (first_velocity) {
                            first_k = last_k;
                            first_velocity = false;
                        }
                    };
                    auto potential = [&](const auto & v, auto & g) {
                        std::vector<double> gp;
                        last_u = target.evaluate(v, f.parameters, work, g, gp);
                        if (!calls)
                            first_u = last_u;
                        ++calls;
                        return last_u;
                    };
                    const auto result = FML::MCMC::run_hmc(potential,
                                                           f.white,
                                                           mass,
                                                           o,
                                                           nullptr,
                                                           GN::FF::Execution<GN::Model>::hmc(model));
                    const double dh = std::isfinite(last_u) ? last_u + last_k - first_u - first_k : HUGE_VAL;
                    if (std::abs((std::isfinite(dh) ? std::min(1., std::exp(-dh)) : 0.) - result.acceptance) > 1e-10)
                        throw std::runtime_error("Independent HMC energy check failed");
                    auto difference = result.last;
                    for (size_t j = 0; j < 64; ++j)
                        difference[j] -= f.white[j];
                    const auto jump = project(difference);
                    out << f.ensemble << ',' << f.draw << ',' << trial << ',' << (variant ? "gauss_newton" : "baseline")
                        << ',' << result.acceptance << ',' << result.divergences << ',' << calls << ',' << dh;
                    for (double x : jump)
                        out << ',' << x * x;
                    out << '\n';
                }
        out.close();
        if (!out)
            throw std::runtime_error("Cannot save probe");
        std::cout << "PASS: 192 actual native HMC trajectories and energy-decision controls\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
