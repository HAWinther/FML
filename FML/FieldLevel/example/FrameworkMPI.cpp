#include <FML/FieldLevel/DistributedLinear.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    FF::Configuration c;
    c.evolution = FF::Evolution::Linear;
    c.mesh = 8;
    c.boxsize = 100;
    c.initial_cutoff = .15;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .11;
    c.growth_rate = .6;
    FF::DistributedLinearModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto work = model.workspace();
    const auto parameters = model.initial_parameters();                        // b1, log noise power
    const auto data = model.mock(model.white_noise(10), parameters, 11, work); // LOCAL modes
    FF::DistributedLinearPosterior posterior(model, data);
    const auto exact = model.linear_conditional(data, parameters, work); // LOCAL mean field
    const auto fit = FF::map(posterior, model.white_noise(12), parameters, false);
    double error = 0;
    for (size_t q = 0; q < model.cells(); ++q)
        error = std::max(error, std::abs(fit.white[q] - exact.mean[q]));
    error = model.maximum(error); // Every rank participates; only root prints.
    FML::MCMC::HMCOptions options;
    options.warmup = 100;
    options.samples = 200;
    options.n_leapfrog = 6;
    options.seed = 13;
    double average_norm = 0;
    const auto chain = FF::sample(posterior,
                                  fit.white,
                                  parameters,
                                  options,
                                  false,
                                  [&](const auto & local_white, const auto &, size_t) {
                                      double norm = 0;
                                      for (double x : local_white)
                                          norm += x * x;
                                      average_norm += model.sum(norm) / options.samples;
                                  });
    if (FML::ThisTask == 0)
        std::cout << "Linear MPI MAP maximum error: " << error << "\nField HMC acceptance: " << chain.acceptance
                  << " divergences: " << chain.divergences << "\nMean squared white-field norm: " << average_norm
                  << '\n';
    return fit.status == FF::MAPStatus::Converged ? 0 : 1;
}
