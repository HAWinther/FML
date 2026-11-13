// End-to-end masked Cartesian survey inference with fixed physics/nuisance values.
#include <FML/FieldLevel/SurveyLikelihood.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
template <class Likelihood>
int reconstruct(const FF::DistributedLPTModel & model, Likelihood likelihood, const std::vector<double> & p) {
    FF::BasicPosterior<FF::DistributedLPTModel, Likelihood> posterior(model, std::move(likelihood));
    const auto initial = model.white_noise(82);
    auto w = model.workspace();
    Field gradient;
    std::vector<double> gp;
    const double before = posterior.evaluate(initial, p, w, gradient, gp);
    const auto fit = FF::map(posterior, initial, p, false, 80, 1e-5);
    FML::MCMC::HMCOptions o;
    o.warmup = 50;
    o.samples = 60;
    o.n_leapfrog = 6;
    o.seed = 91;
    o.target_acceptance = .85;
    const auto chain = FF::sample(posterior, fit.white, p, o, false);
    double norm = 0;
    for (double x : chain.last)
        norm += x * x;
    const double rms = std::sqrt(model.sum(norm) / model.layout().global_cells);
    if (FML::ThisTask == 0)
        std::cout << "Masked survey MAP: " << before << " -> " << fit.potential << " (status " << int(fit.status)
                  << ")\nField HMC acceptance " << chain.acceptance << " divergences " << chain.divergences
                  << " final white RMS " << rms << '\n';
    return std::isfinite(fit.potential) && fit.potential < before && chain.divergences == 0 ? 0 : 1;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    const std::string kind = argc > 1 ? argv[1] : "gaussian";
    if (argc > 2 || (kind != "gaussian" && kind != "poisson")) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: survey_mpi [gaussian|poisson]\n";
        return 1;
    }
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    FF::DistributedLPTModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto p = model.initial_parameters();
    p[0] = 1.4;
    p[1] = .2;
    p[2] = -.1;
    auto window = FF::make_survey_window(model, {-50, -50, -50}, [](auto x) {
        const double r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        return FF::SurveyVoxel{r2 < 45 * 45 && x[2] > -30, .9 * std::exp(-r2 / (2 * 40 * 40))};
    });
    auto w = model.workspace();
    const auto prediction = model.predict(model.white_noise(80), p, w);
    FF::CartesianSurveyObservation observation(model, window);
    const auto delta = observation.synthesize(prediction);
    Field data(model.cells()), variance(model.cells(), 1), exposure(model.cells(), 3);
    std::vector<std::uint64_t> counts(model.cells());
    const int N = c.mesh;
    for (size_t q = 0; q < model.cells(); ++q) {
        FML::RANDOM::COUNTER_BASED::Stream rng(81, size_t(model.layout().local_start) * N * N + q, 0x53555256u);
        data[q] =
            window.selection[q] * delta[q] + std::sqrt(std::exp(p.back()) / observation.cell_volume()) * rng.normal();
        // Knuth's independent Poisson generator is adequate for this toy lambda~3.
        const double lambda = exposure[q] * window.selection[q] * std::exp(delta[q]);
        double product = 1;
        std::uint64_t n = 0;
        do {
            ++n;
            product *= rng.uniform();
        } while (product > std::exp(-lambda));
        counts[q] = n - 1;
    }
    // Counts do not use the Gaussian-noise scalar. Fix all scalars here; exposure
    // carries the expected count normalization. Joint bias/spectrum inference is
    // available, with the unused noise scalar retaining its independent prior.
    if (kind == "gaussian")
        return reconstruct(model,
                           FF::SurveyGaussianLikelihood(model, std::move(window), std::move(data), std::move(variance)),
                           p);
    return reconstruct(model,
                       FF::SurveyPoissonLikelihood(model, std::move(window), std::move(counts), std::move(exposure)),
                       p);
}
