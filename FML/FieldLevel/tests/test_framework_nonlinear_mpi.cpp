// Bounded same-model nonlinear inference validation; not cosmological calibration.
#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/tests/TestHelpers.h>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
static void run(FF::Evolution evolution) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .10;
    c.likelihood_cutoff = .11;
    c.growth_rate = .6;
    c.evolution = evolution;
    auto spectrum = FF::Spectrum::fixed([](double) { return 2000.; });
    FF::DistributedLPTModel model(c, spectrum);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[0] = 1.4;
    p[1] = .3;
    p[2] = -.2;
    p[3] = 5;
    p.back() = std::log(5.);
    const int N = c.mesh;
    for (int mock = 0; mock < 2; ++mock) {
        const auto truth = model.white_noise(901 + mock);
        auto data = model.mock(truth, p, 1001 + mock, w);
        FF::DistributedLPTPosterior posterior(model, data);
        auto lc = c;
        lc.evolution = FF::Evolution::Linear;
        FF::DistributedLinearModel linear(lc, spectrum);
        auto lw = linear.workspace();
        auto lp = linear.initial_parameters();
        lp[0] = p[0];
        lp.back() = p.back();
        const auto lin = linear.predict(truth, lp, lw), nonlinear = model.predict(truth, p, w);
        double e = 0, norm = 0;
        for (size_t i = 0; i < lin.size(); ++i) {
            e += model.modes()[i].weight * std::norm(lin[i] - nonlinear[i]);
            norm += model.modes()[i].weight * std::norm(nonlinear[i]);
        }
        check_local(std::sqrt(model.sum(e) / model.sum(norm)) > .01,
                    "fixture has resolved nonlinear signal beyond linear RSD");
        std::vector<FML::MCMC::HMCResult> chains;
        Field mean(model.cells(), 0);
        for (int chain = 0; chain < 2; ++chain) {
            FML::MCMC::HMCOptions opt;
            opt.warmup = 150;
            opt.samples = 300;
            opt.n_leapfrog = 8;
            opt.target_acceptance = .85;
            opt.seed = 4001 + 11 * mock + chain;
            std::vector<std::vector<double>> summaries;
            auto sink = [&](const Field & field, const auto &, size_t) {
                double a = 0, b = 0, power = 0;
                for (size_t q = 0; q < field.size(); ++q) {
                    const int x = model.layout().local_start + int(q / (N * N));
                    const double phase = 2 * M_PI * x / N;
                    a += field[q] * std::cos(phase);
                    b += field[q] * std::sin(phase);
                    power += field[q] * field[q];
                    mean[q] += field[q] / (2 * opt.samples);
                }
                const double scale = std::sqrt(2. / model.layout().global_cells);
                summaries.push_back(
                    {model.sum(a) * scale, model.sum(b) * scale, model.sum(power) / model.layout().global_cells});
            };
            auto start = model.white_noise(5001 + 17 * mock + chain);
            for (auto & x : start)
                x *= chain ? 2. : .5;
            auto result = FF::sample(posterior, start, p, opt, false, sink);
            check_local(result.divergences == 0 && result.acceptance > .7,
                        "nonlinear field HMC acceptance and zero sampling divergences");
            result.recorded = std::move(summaries);
            chains.push_back(std::move(result));
        }
        for (size_t j = 0; j < 3; ++j) {
            auto d = FML::MCMC::diagnose_hmc_chains(chains, j);
            if (FML::ThisTask == 0)
                std::cout << "NONLINEAR evolution=" << int(evolution) << " mock=" << mock << " column=" << j
                          << " Rhat=" << d.rhat << " bulk=" << d.ess_bulk << " tail=" << d.ess_tail << '\n';
            check_local(d.valid() && d.rhat < 1.05 && d.ess_bulk > 60 && d.ess_tail > 40,
                        "dispersed nonlinear chains mix for field summaries");
        }
        const auto estimate = model.reconstruct(mean, p, w, 1.).initial_density;
        const auto target = model.reconstruct(truth, p, w, 1.).initial_density;
        double err = 0, base = 0;
        for (size_t q = 0; q < mean.size(); ++q) {
            err += std::pow(estimate[q] - target[q], 2);
            base += target[q] * target[q];
        }
        const double ratio = std::sqrt(model.sum(err) / model.sum(base));
        check_local(ratio < .5, "posterior mean initial density improves over prior-zero reconstruction");
        double average = 0, variance = 0;
        for (const auto & chain : chains)
            for (const auto & row : chain.recorded)
                average += row[0] / 600;
        for (const auto & chain : chains)
            for (const auto & row : chain.recorded)
                variance += std::pow(row[0] - average, 2) / 599;
        check_local(variance < .25, "data constrain a physical initial-field mode below its unit prior variance");
        if (FML::ThisTask == 0)
            std::cout << "RECOVERY evolution=" << int(evolution) << " mock=" << mock << " rmse_ratio=" << ratio
                      << " mode_mean=" << average << " mode_variance=" << variance << '\n';
    }
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(15) << std::unitbuf;
    run(FF::Evolution::Zeldovich);
    run(FF::Evolution::LPT2);
    return FML::TESTS::summary("bounded nonlinear MPI inference");
}
