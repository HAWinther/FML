// Restart a fixed-parameter 2LPT survey chain using immutable analytic toy inputs.
#include <FML/FieldLevel/SurveyCheckpoint.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
template <class Likelihood>
static int run(const FF::DistributedLPTModel & model,
               Likelihood like,
               const std::vector<double> & p,
               const std::string & kind,
               int argc,
               char ** argv) {
    FF::BasicPosterior<FF::DistributedLPTModel, Likelihood> posterior(model, std::move(like));
    FML::MCMC::HMCOptions options;
    options.warmup = 60;
    options.samples = 80;
    options.n_leapfrog = 4;
    options.seed = 71;
    options.target_acceptance = .85;
    FF::CheckpointOptions io;
    io.directory = argv[1];
    io.problem_id = "FML-survey-restart-demo-v1:2LPT:P200:" + kind;
    io.resume = std::string(argv[3]) == "resume";
    io.every = 20;
    io.max_iterations = argc == 5 ? std::stoi(argv[4]) : 40;
    const auto result = io.resume ? FF::sample_checkpointed(posterior, {}, {}, options, io, false) :
                                    FF::sample_checkpointed(posterior, model.white_noise(73), p, options, io, false);
    if (FML::ThisTask == 0)
        std::cout << "Completed trajectories " << result.completed_iterations << " / 140; sampling draws "
                  << result.completed_samples << " / 80; new rows " << result.recorded.size()
                  << "\nCumulative acceptance " << result.acceptance << " divergences " << result.divergences << '\n';
    return 0;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (argc < 4 || argc > 5 || (std::string(argv[2]) != "gaussian" && std::string(argv[2]) != "poisson") ||
        (std::string(argv[3]) != "start" && std::string(argv[3]) != "resume")) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: survey_checkpoint_mpi DIRECTORY gaussian|poisson start|resume [trajectory_budget]\n";
        return 1;
    }
    try {
        FF::Configuration c;
        c.mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        c.initial_cutoff = .115;
        c.bias_cutoff = .08;
        c.likelihood_cutoff = .10;
        c.growth_rate = .6;
        FF::DistributedLPTModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
        const auto p = model.initial_parameters();
        auto window = FF::make_survey_window(model, {-50, -50, -50}, [](auto x) {
            return FF::SurveyVoxel{x[0] + x[1] + x[2] < 20, .7 + .002 * x[2]};
        });
        FML::FIELDLEVEL::Field data(model.cells()), variance(model.cells(), 1), exposure(model.cells(), 2.5);
        std::vector<std::uint64_t> counts(model.cells());
        const int N = c.mesh;
        // Recreate the same observations in every process; this is a restart demo,
        // not a scientific mock-recovery experiment. Real applications reload data.
        for (size_t q = 0; q < model.cells(); ++q) {
            const size_t id = size_t(model.layout().local_start) * N * N + q;
            data[q] =
                .05 * std::cos(2 * M_PI * double(id / (N * N)) / N) + .02 * std::sin(2 * M_PI * double(id % N) / N);
            counts[q] = id % 3;
        }
        const std::string kind = argv[2];
        if (kind == "gaussian")
            return run(model,
                       FF::SurveyGaussianLikelihood(model, std::move(window), std::move(data), std::move(variance)),
                       p,
                       kind,
                       argc,
                       argv);
        return run(model,
                   FF::SurveyPoissonLikelihood(model, std::move(window), std::move(counts), std::move(exposure)),
                   p,
                   kind,
                   argc,
                   argv);
    } catch (const std::exception & error) {
        if (FML::ThisTask == 0)
            std::cerr << error.what() << '\n';
        return 1;
    }
}
