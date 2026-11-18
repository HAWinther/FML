#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
static void array(std::ostream & out, const std::vector<double> & values) {
    out << '[';
    for (size_t j = 0; j < values.size(); ++j) {
        if (j)
            out << ',';
        out << values[j];
    }
    out << ']';
}
// The draw index is a stable key: replay after a process failure overwrites any
// uncommitted record with the same deterministic draw. Readers use summary.json's
// committed count, never the number of files present in draws/.
static void record(const FS::path & root,
                   const Field & white,
                   const std::vector<double> & p,
                   const FF::Modes & prediction,
                   size_t draw) {
    const auto final =
        root / "draws" / ("draw-" + std::to_string(draw) + ".rank-" + std::to_string(FML::ThisTask) + ".json");
    const auto temporary = FS::path(final.string() + ".tmp");
    std::ofstream out(temporary);
    out << std::setprecision(17) << "{\"draw\":" << draw << ",\"rank\":" << FML::ThisTask << ",\"white\":";
    array(out, white);
    out << ",\"parameters\":";
    array(out, p);
    out << ",\"prediction\":[";
    for (size_t j = 0; j < prediction.size(); ++j) {
        if (j)
            out << ',';
        out << '[' << prediction[j].real() << ',' << prediction[j].imag() << ']';
    }
    out << ']';
    out << "}\n";
    out.close();
    if (!out)
        throw std::runtime_error("Cannot write draw record");
    FS::rename(temporary, final);
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        FF::CheckpointOptions io;
        io.every = 64;
        io.max_iterations = 64;
        int chain = 0;
        FML::COMMUNICATION::collective_try(
            [&] {
                if (argc < 4 || argc > 5)
                    throw std::invalid_argument("Usage: joint_pilot DIRECTORY start|resume CHAIN [trajectory_budget]");
                io.directory = argv[1];
                const std::string phase = argv[2];
                if (phase != "start" && phase != "resume")
                    throw std::invalid_argument("Expected start or resume");
                io.resume = phase == "resume";
                size_t chain_used;
                chain = std::stoi(argv[3], &chain_used);
                if (chain_used != std::string(argv[3]).size() || chain < 0 || chain > 3)
                    throw std::invalid_argument("Invalid chain index");
                if (argc == 5) {
                    size_t used;
                    io.max_iterations = std::stoi(argv[4], &used);
                    if (used != std::string(argv[4]).size() || io.max_iterations < 0)
                        throw std::invalid_argument("Invalid trajectory budget");
                }
            },
            "Eulerian checkpoint command line");
        const auto root = FS::absolute(io.directory).lexically_normal();
        CP::same_string(root.string());
        CP::same_string(std::to_string(io.resume) + ":" + std::to_string(io.max_iterations));
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    if (!io.resume && FS::exists(root) && !FS::is_empty(root))
                        throw std::invalid_argument("Start requires a new or empty directory");
                    FS::create_directories(root / "draws");
                }
            },
            "Eulerian example output directory");
        FML::FIELDLEVEL::FRONTEND::initialize(4, 16);
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        c.product_mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        const double kf = 2 * M_PI / c.boxsize;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> bias = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                              {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        using Model = FF::DistributedEulerianLPTModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
        Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), bias);
        auto work = model.workspace();
        const auto parameters = model.initial_parameters();
        const auto data = model.mock(model.white_noise(4101), parameters, 1000., 4102, work);
        const auto target = FF::fixed_noise_posterior(model, data, 1000.);
        // Must be changed when callback physics/source or external data conventions change.
        io.problem_id =
            "FML-joint-pilot-1:plan-sha256:a2bac875657c6ce24a8bd1b3ce8b81cdba48e7ed8f5412ee53298a7ad0098986:chain=" +
            std::to_string(chain);
        FML::MCMC::HMCOptions options;
        options.warmup = 256;
        options.samples = 1024;
        options.n_leapfrog = 12;
        options.step_size = .05;
        options.jitter = true;
        options.adapt_step_size = true;
        options.target_acceptance = .9;
        options.adapt_mass = true;
        options.seed = 4110 + chain;
        auto sink = [&](const Field & white, const auto & p, size_t draw) {
            FML::COMMUNICATION::collective_try([&] { record(root, white, p, model.predict(white, p, work), draw); },
                                               "Eulerian draw publication");
        };
        const std::vector<std::vector<double>> starts = {{.8, 1.2, -.3, .2, -20},
                                                         {1.2, 2, .8, -.6, 30},
                                                         {.9, 2.4, -.8, .6, 50},
                                                         {1.1, .8, 1, -1, -40}};
        const auto initial = starts[chain];
        if (FML::NTasks != 1)
            throw std::invalid_argument("This declared pilot uses independent serial chains");
        if (!io.resume) {
            std::ofstream out(root / "target.json");
            out << std::setprecision(17) << "{\"truth_white\":";
            array(out, model.white_noise(4101));
            out << ",\"data\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << '[' << data[j].real() << ',' << data[j].imag() << ']';
            }
            out << "],\"k\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << model.modes()[j].k;
            }
            out << "],\"weights\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << model.modes()[j].weight;
            }
            out << "]}\n";
            out.close();
            if (!out)
                throw std::runtime_error("Cannot write target");
        }
        const auto result =
            io.resume ?
                FF::sample_checkpointed(target, {}, {}, options, io, true, sink) :
                FF::sample_checkpointed(target, model.white_noise(4103 + chain), initial, options, io, true, sink);
        const auto generation = CP::current_generation(root);
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    const auto final = root / "summary.json", temporary = root / "summary.json.tmp";
                    std::ofstream out(temporary);
                    out << std::setprecision(17)
                        << "{\"synthetic_joint_pilot\":true,\"mesh\":4,\"product_mesh\":8,\"density_mesh\":16,"
                           "\"ranks\":"
                        << FML::NTasks << ",\"generation\":\"" << generation
                        << "\",\"completed_iterations\":" << result.completed_iterations
                        << ",\"completed_samples\":" << result.completed_samples << ",\"warmup\":" << options.warmup
                        << ",\"samples\":" << options.samples << ",\"acceptance\":" << result.acceptance
                        << ",\"step_size\":" << result.step_size << ",\"divergences\":" << result.divergences << "}\n";
                    out.close();
                    if (!out)
                        throw std::runtime_error("Cannot publish summary");
                    FS::rename(temporary, final);
                }
            },
            "Eulerian summary publication");
        if (FML::ThisTask == 0)
            std::cout << "Completed trajectories: " << result.completed_iterations << '/'
                      << options.warmup + options.samples << "; committed draws: " << result.completed_samples
                      << "; new rows: " << result.recorded.size() << '\n';
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0)
            std::cerr << e.what() << '\n';
        return 1;
    }
}
