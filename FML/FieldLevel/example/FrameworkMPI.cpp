#include <FML/FieldLevel/DistributedLPT.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
template <class Forward>
int demonstrate(const Forward & model) {
    auto work = model.workspace();
    const auto parameters = model.initial_parameters();                        // Model-specific scalar parameters
    const auto data = model.mock(model.white_noise(10), parameters, 11, work); // LOCAL modes
    FF::BasicPosterior<Forward> posterior(model, data);
    const auto fit = FF::map(posterior, model.white_noise(12), parameters, false);
    if constexpr (std::is_same_v<Forward, FF::DistributedLinearModel>) {
        const auto exact = model.linear_conditional(data, parameters, work); // LOCAL mean field
        double error = 0;
        for (size_t q = 0; q < model.cells(); ++q)
            error = std::max(error, std::abs(fit.white[q] - exact.mean[q]));
        error = model.maximum(error); // Every rank participates; only root prints.
        if (FML::ThisTask == 0)
            std::cout << "Linear MPI MAP maximum error: " << error << '\n';
    } else if (FML::ThisTask == 0) {
        std::cout << (model.configuration().evolution == FF::Evolution::LPT2 ? "2LPT" : "1LPT")
                  << " MPI MAP gradient norm: " << fit.gradient_norm << '\n';
    }
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
        std::cout << "Field HMC acceptance: " << chain.acceptance << " divergences: " << chain.divergences
                  << "\nMean squared white-field norm: " << average_norm << '\n';
    return fit.status == FF::MAPStatus::Converged ? 0 : 1;
}

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    const bool first = argc == 2 && std::string(argv[1]) == "1lpt";
    const bool second = argc == 2 && std::string(argv[1]) == "2lpt";
    const bool lpt = first || second;
    if (argc > 2 || (argc == 2 && !lpt)) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: framework_mpi [1lpt|2lpt]\n";
        return 1;
    }
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = lpt ? .115 : .15;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .11;
    c.growth_rate = .6;
    auto spectrum = FF::Spectrum::fixed([](double) { return 200.; });
    if (lpt) {
        c.evolution = second ? FF::Evolution::LPT2 : FF::Evolution::Zeldovich;
        FF::DistributedLPTModel model(c, spectrum);
        return demonstrate(model);
    }
    c.evolution = FF::Evolution::Linear;
    FF::DistributedLinearModel model(c, spectrum);
    return demonstrate(model);
}
