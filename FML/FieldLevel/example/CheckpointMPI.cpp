#include <FML/FieldLevel/Checkpoint.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (argc < 2 || argc > 4 || (argc >= 3 && std::string(argv[2]) != "start" && std::string(argv[2]) != "resume")) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: checkpoint_mpi DIRECTORY [start|resume] [trajectory_budget]\n";
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
        c.evolution = FF::Evolution::LPT2;
        FF::DistributedLPTModel model(c, FF::Spectrum::fixed([](double) { return 200.; }));
        auto work = model.workspace();
        const auto p = model.initial_parameters();
        const auto data = model.mock(model.white_noise(10), p, 11, work);
        FF::DistributedLPTPosterior posterior(model, data);
        FML::MCMC::HMCOptions hmc;
        hmc.warmup = 120;
        hmc.samples = 300;
        hmc.n_leapfrog = 4;
        hmc.step_size = .01;
        hmc.seed = 13;
        FF::CheckpointOptions checkpoint;
        checkpoint.directory = argv[1];
        checkpoint.problem_id = "FML-example-v1:EdS2LPT:P200:mock10-noise11";
        checkpoint.resume = argc >= 3 && std::string(argv[2]) == "resume";
        checkpoint.every = 25;
        checkpoint.max_iterations = argc == 4 ? std::stoi(argv[3]) : 60;
        // The same full HMC schedule is used on every invocation; the budget limits this job only.
        auto result = checkpoint.resume ? FF::sample_checkpointed(posterior, {}, {}, hmc, checkpoint) :
                                          FF::sample_checkpointed(posterior, model.white_noise(12), p, hmc, checkpoint);
        FML::FIELDLEVEL::Field white(result.last.begin(), result.last.begin() + model.cells());
        std::vector<double> parameters(result.last.begin() + model.cells(), result.last.end());
        const auto fields = model.reconstruct(white, parameters, work, 1.); // Numerical toy velocity conversion.
        double sum = 0;
        for (double x : fields.matter_density)
            sum += x * x;
        const double rms = std::sqrt(model.sum(sum) / model.layout().global_cells);
        if (FML::ThisTask == 0)
            std::cout << "Completed trajectories: " << result.completed_iterations << " / " << hmc.warmup + hmc.samples
                      << "\nCompleted sampling draws: " << result.completed_samples
                      << "; new rows this invocation: " << result.recorded.size()
                      << "\nCumulative acceptance: " << result.acceptance << "; divergences: " << result.divergences
                      << "\nCurrent-state real-space matter RMS: " << rms << '\n';
    } catch (const std::exception & error) {
        if (FML::ThisTask == 0)
            std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
