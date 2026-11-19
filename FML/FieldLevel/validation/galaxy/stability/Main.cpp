#include "Probe.h"
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using Model = FF::DistributedEulerianLPTModel;
using Field = FML::FIELDLEVEL::Field;
struct Case {
    std::string name;
    int current;
    double step;
    int leaps;
};
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 3 || FML::NTasks != 1)
            throw std::invalid_argument("stability_probe INPUT OUTPUT_PREFIX, serial only");
        std::ifstream in(argv[1]);
        std::string tag;
        unsigned mid, nm, nframes, trials, ncases;
        if (!(in >> tag >> mid >> nm >> nframes >> trials >> ncases) || tag != "FML_STABILITY_1" || mid < 1 ||
            mid > 65535 || nm < 1 || nm > 64 || nframes < 1 || nframes > 48 || trials < 1 || trials > 16 ||
            ncases < 1 || ncases > 8)
            throw std::invalid_argument("Invalid bounded probe controls");
        std::vector<Case> cases(ncases);
        for (auto & c : cases)
            if (!(in >> c.name >> c.current >> c.step >> c.leaps) || (c.current != 0 && c.current != 1) ||
                !std::isfinite(c.step) || c.step <= 0 || c.leaps < 1 || c.leaps > 64)
                throw std::invalid_argument("Invalid probe case");
        FF::Modes data(nm);
        for (auto & v : data) {
            double re, im;
            if (!(in >> re >> im) || !std::isfinite(re) || !std::isfinite(im))
                throw std::invalid_argument("Invalid data");
            v = {re, im};
        }
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        c.product_mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        const double kf = 2 * M_PI / 100;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> bias = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                              {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), bias);
        if (model.modes().size() != nm)
            throw std::invalid_argument("Input/model mode layout differs");
        auto target = FF::fixed_noise_posterior(model, data, 1000.);
        auto work = model.workspace();
        const std::string prefix = argv[2];
        std::ofstream out(prefix + ".csv"), checks(prefix + ".checks.csv");
        out << std::setprecision(17)
            << "mock,ensemble,draw,trial,case,step,leaps,current_metric,acceptance,divergences,delta_H,gradient_"
               "evaluations,jump_rms\n";
        checks << std::setprecision(17)
               << "mock,ensemble,draw,prediction_error,quadratic_error,gradient_checks,skipped_bounds,gradient_best_"
                  "max_error,gradient_sweeps_passed,native_matches\n";
        for (unsigned frame = 0; frame < nframes; ++frame) {
            unsigned e, draw;
            if (!(in >> e >> draw) || e > 3 || draw > 2047)
                throw std::invalid_argument("Invalid frame identity");
            std::vector<double> frozen(5), p(5);
            Field white(64);
            FF::Modes saved(nm);
            double savedq;
            for (auto * v : {&frozen, &p, &white})
                for (auto & x : *v)
                    if (!(in >> x) || !std::isfinite(x))
                        throw std::invalid_argument("Invalid frame vector");
            for (auto & v : saved) {
                double re, im;
                if (!(in >> re >> im))
                    throw std::invalid_argument("Missing prediction");
                v = {re, im};
            }
            if (!(in >> savedq) || !std::isfinite(savedq))
                throw std::invalid_argument("Missing quadratic");
            const auto prediction = model.predict(white, p, work);
            double pe = 0, q = 0;
            for (size_t j = 0; j < nm; ++j) {
                pe = std::max(pe, std::abs(prediction[j] - saved[j]) / std::max(1., std::abs(saved[j])));
                q += .5 * target.likelihood().weights()[j] * std::norm(prediction[j] - data[j]) * 1e3;
            }
            const double qe = std::abs(q - savedq) / std::max(1., std::abs(q));
            if (pe > 1e-10 || qe > 1e-11)
                throw std::runtime_error("Saved-state forward reproduction failed");
            Field gradient;
            std::vector<double> gp;
            target.evaluate(white, p, work, gradient, gp);
            auto reference = target.value_reference(white, p, work);
            double worst = 0;
            int checked = 0, skipped = 0;
            bool allpass = true;
            for (int direction = 0; direction < 8; ++direction) {
                Field d(64, 0);
                double analytic = 0, scale = 1;
                if (direction < 3) {
                    double norm = 0;
                    for (size_t j = 0; j < 64; ++j) {
                        d[j] = std::cos(double(j + 1) * (direction + 2));
                        norm += d[j] * d[j];
                    }
                    for (size_t j = 0; j < 64; ++j) {
                        d[j] /= std::sqrt(norm);
                        analytic += gradient[j] * d[j];
                    }
                } else {
                    const int j = direction - 3;
                    analytic = gp[j];
                    scale = std::max(1., std::abs(p[j]));
                    if (std::min(p[j] - model.parameters()[j].lower, model.parameters()[j].upper - p[j]) < 1e-10) {
                        ++skipped;
                        continue;
                    }
                }
                std::vector<double> errors;
                for (double baseh : {1e-3, 1e-4, 1e-5}) {
                    double h = baseh * scale;
                    auto plus = white, minus = white;
                    auto pp = p, pm = p;
                    if (direction < 3) {
                        for (size_t j = 0; j < 64; ++j) {
                            plus[j] += h * d[j];
                            minus[j] -= h * d[j];
                        }
                    } else {
                        const int j = direction - 3;
                        h = std::min(
                            h,
                            .25 * std::min(p[j] - model.parameters()[j].lower, model.parameters()[j].upper - p[j]));
                        pp[j] += h;
                        pm[j] -= h;
                    }
                    const double fd = (target.value_change(plus, pp, work, white, reference).change -
                                       target.value_change(minus, pm, work, white, reference).change) /
                                      (2 * h);
                    errors.push_back(std::abs(fd - analytic) / std::max({1., std::abs(fd), std::abs(analytic)}));
                }
                const double best = *std::min_element(errors.begin(), errors.end());
                worst = std::max(worst, best);
                ++checked;
                allpass &= std::isfinite(best) &&
                           ((errors[0] < 3e-5 && errors[1] < 3e-5) || (errors[1] < 3e-5 && errors[2] < 3e-5));
            }
            int matches = 0;
            for (const auto & setting : cases) {
                const auto & metric = setting.current ? p : frozen;
                for (unsigned trial = 0; trial < trials; ++trial) {
                    FML::MCMC::HMCOptions options;
                    options.warmup = 0;
                    options.samples = 1;
                    options.n_leapfrog = setting.leaps;
                    options.step_size = setting.step;
                    options.jitter = false;
                    options.adapt_mass = false;
                    options.adapt_step_size = false;
                    options.seed = UINT64_C(0x5600000000000000) | (uint64_t(mid) << 32) | (uint64_t(e) << 30) |
                                   (uint64_t(draw) << 8) | trial;
                    auto result = STABILITY::trajectory(target, white, p, metric, options, work);
                    if (trial == 0) {
                        auto native = FF::DETAIL::sample_run(target, white, p, options, false, {}, metric);
                        if (native.last != result.hmc.last || native.acceptance != result.hmc.acceptance ||
                            native.divergences != result.hmc.divergences)
                            throw std::runtime_error("Instrumented/native trajectory differs");
                        ++matches;
                    }
                    double jump = 0;
                    for (size_t j = 0; j < 64; ++j)
                        jump += std::pow(result.hmc.last[j] - white[j], 2);
                    jump = std::sqrt(jump / 64);
                    out << mid << ',' << e << ',' << draw << ',' << trial << ',' << setting.name << ',' << setting.step
                        << ',' << setting.leaps << ',' << setting.current << ',' << result.hmc.acceptance << ','
                        << result.hmc.divergences << ',' << result.energy_error << ',' << result.evaluations << ','
                        << jump << '\n';
                }
            }
            checks << mid << ',' << e << ',' << draw << ',' << pe << ',' << qe << ',' << checked << ',' << skipped
                   << ',' << worst << ',' << allpass << ',' << matches << '\n';
            out.flush();
            checks.flush();
        }
        std::string extra;
        if (in >> extra)
            throw std::invalid_argument("Extra probe inputs");
        out.close();
        checks.close();
        if (!out || !checks)
            throw std::runtime_error("Cannot publish probe results");
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
