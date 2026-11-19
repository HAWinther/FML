#include "Native.h"
#include <FML/FieldLevel/validation/galaxy/gauss_newton/Fixture.h>
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 4 || FML::NTasks != 1)
            throw std::invalid_argument("donor_probe POOL SCREEN_INPUT OUTPUT.csv, serial only");
        DONOR_DRIVER::load(argv[1]);
        if (!DONOR_DRIVER::enabled || DONOR_DRIVER::raw.size() != 12)
            throw std::invalid_argument("Twelve frozen donors required");
        std::ifstream in(argv[2]);
        std::string tag;
        unsigned frames, modes, trials;
        if (!(in >> tag >> frames >> modes >> trials) || tag != "FML_DONOR_SCREEN_1" || frames != 12 || modes != 17 ||
            trials != 64)
            throw std::invalid_argument("Wrong bounded screen input");
        GN::FF::Modes data(modes);
        for (auto & z : data) {
            double re, im;
            if (!(in >> re >> im))
                throw std::invalid_argument("Missing data");
            z = {re, im};
        }
        auto model = GN::model();
        auto work = model.workspace();
        ACTIVE_SUPPORT::InitialSupport<GN::Model> support(model);
        DONOR_DRIVER::prepare(support);
        auto target = GN::FF::fixed_noise_posterior(model, data, 1000.);
        std::vector<GN::FF::Modes> predictions(frames);
        std::vector<double> values(frames), nullnorm(frames);
        for (size_t i = 0; i < frames; ++i) {
            unsigned e, d;
            double q;
            if (!(in >> e >> d >> q) || e != DONOR_DRIVER::raw[i].ensemble || d != DONOR_DRIVER::raw[i].draw)
                throw std::invalid_argument("Reference identity differs");
            auto & saved = predictions[i];
            saved.resize(modes);
            for (auto & z : saved) {
                double re, im;
                if (!(in >> re >> im))
                    throw std::invalid_argument("Missing saved prediction");
                z = {re, im};
            }
            const auto & s = DONOR_DRIVER::raw[i];
            const auto predicted = model.predict(s.white, s.parameters, work);
            double rebuilt = 0;
            for (size_t j = 0; j < modes; ++j) {
                if (std::abs(predicted[j] - saved[j]) > 1e-10)
                    throw std::runtime_error("Reference prediction changed");
                rebuilt += .5 * target.likelihood().weights()[j] * std::norm(predicted[j] - data[j]) * 1000.;
            }
            if (std::abs(rebuilt - q) > 1e-11 * std::max(1., std::abs(q)))
                throw std::runtime_error("Reference Q changed");
            values[i] = target.value(s.white, s.parameters, work);
            const auto active = support.project(s.white);
            for (size_t j = 0; j < 64; ++j)
                nullnorm[i] += std::pow(s.white[j] - active[j], 2);
        }
        std::string extra;
        if (in >> extra)
            throw std::invalid_argument("Extra screen fields");
        double endpoint_error = 0, prior_error = 0;
        for (size_t j = 0; j < frames; ++j)
            for (size_t i = 0; i < frames; ++i)
                if (i != j) {
                    const auto & s = DONOR_DRIVER::raw[j];
                    const auto p = DONOR_DRIVER::pool->translate(s.white, s.parameters, i, j, 1.);
                    const auto pred = model.predict(p.white, p.parameters, work);
                    for (size_t k = 0; k < modes; ++k)
                        endpoint_error = std::max(endpoint_error, std::abs(pred[k] - predictions[i][k]));
                    const double u = target.value(p.white, p.parameters, work),
                                 expected = .5 * (nullnorm[j] - nullnorm[i]);
                    prior_error =
                        std::max(prior_error, std::abs(u - values[i] - expected) / std::max(1., std::abs(expected)));
                }
        if (endpoint_error > 5e-10 || prior_error > 1e-9)
            throw std::runtime_error("Endpoint transfer/preserved-null target check failed");
        std::ofstream out(argv[3]);
        out << std::setprecision(17)
            << "frame,trial,first,second,scale,cross_ensemble_pair,support_rejection,value_calls,acceptance_"
               "probability,physical_jump_squared\n";
        for (size_t k = 0; k < frames; ++k)
            for (unsigned t = 0; t < trials; ++t) {
                const auto & s = DONOR_DRIVER::raw[k];
                std::mt19937_64 rng(944001 + k * 128 + t);
                const auto p = DONOR_DRIVER::pool->draw(s.white, s.parameters, rng);
                const bool valid = model.in_support(p.parameters);
                double next = HUGE_VAL;
                if (valid) {
                    try {
                        next = target.value(p.white, p.parameters, work);
                    } catch (const GN::FF::UnrepresentableState &) {
                    }
                }
                const double probability = std::isfinite(next) ? std::min(1., std::exp(values[k] - next)) : 0.;
                const auto a = support.project(s.white), b = support.project(p.white);
                double physical = 0;
                for (size_t i = 0; i < 64; ++i)
                    physical += std::pow(p.parameters[0] * b[i] - s.parameters[0] * a[i], 2);
                out << k << ',' << t << ',' << p.first << ',' << p.second << ',' << p.scale << ','
                    << (DONOR_DRIVER::raw[p.first].ensemble != DONOR_DRIVER::raw[p.second].ensemble) << ',' << !valid
                    << ',' << valid << ',' << probability << ',' << physical << '\n';
            }
        out.close();
        if (!out)
            throw std::runtime_error("Cannot save screen");
        std::cout << std::setprecision(17) << "PASS: twelve saved targets, 132 endpoint transfers, errors "
                  << endpoint_error << ' ' << prior_error << "; 768 fixed proposals\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
