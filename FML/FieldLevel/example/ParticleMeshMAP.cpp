#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/Frontend.h>
#include <iostream>
#include <iomanip>
int main() {
    namespace FF = FML::FIELDLEVEL::FRAMEWORK;
    try {
        // Small dimensionless demonstration, not a production cosmology run.
        FML::FIELDLEVEL::FRONTEND::initialize(4, 8);
        FF::Configuration c;
        c.mesh = 4; c.density_mesh = 8; c.boxsize = 1;
        c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = 10.; c.evolution = FF::Evolution::Linear;
        FF::Spectrum power;
        power.parameters = {{"log_amplitude", .1, 0., .6, -2., 2.}, {"tilt", -.1, 0., .5, -1.5, 1.5}};
        power.evaluate = [](double k, const auto & p) {
            const double logk = std::log(k / (2 * M_PI));
            return FF::Power{.003 * std::exp(p[0] + p[1] * logk), {1., logk}};
        };
        FF::PMSettings controls; controls.scale_factors = {.2, .3, .4, .5, .6};
        const double log_noise = std::log(1e-4);
        FF::Parameter noise{"log_noise_power", log_noise, log_noise, 1., -15., 0.};
        using Grid = FML::FIELDLEVEL::FRONTEND::Grid;
        FF::PeriodicPMModelT<Grid> model(c, power, controls, noise);
        auto w = model.workspace(); const auto p = model.initial_parameters();
        const auto data = model.mock(model.white_noise(91), p, 92, w);
        const auto truth = model.reconstruction(w).initial_density;
        FF::PeriodicPMPosteriorT<Grid> posterior(model, data);
        auto start = model.white_noise(95); for (auto & x : start) x *= .001;
        const auto fit = FF::map(posterior, start, p, false, 200, 2e-5);
        model.predict(fit.white, fit.parameters, w);
        const auto & reconstructed = model.reconstruction(w).initial_density;
        double squared_error = 0, squared_truth = 0;
        for (size_t q = 0; q < truth.size(); ++q) {
            squared_error += std::pow(reconstructed[q] - truth[q], 2); squared_truth += truth[q] * truth[q];
        }
        const double relative_error = model.sum(squared_error) / model.sum(squared_truth);
        if (FML::ThisTask == 0)
            std::cout << std::setprecision(12) << "PM MAP status (0=converged): " << int(fit.status)
                      << "\nIterations / gradient norm: " << fit.iterations << " / " << fit.gradient_norm
                      << "\nPosterior potential: " << fit.potential
                      << "\nInitial-density squared error / truth norm: " << relative_error << '\n';
        return fit.status == FF::MAPStatus::Converged ? 0 : 1;
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0) std::cerr << e.what() << '\n';
        return 1;
    }
}
