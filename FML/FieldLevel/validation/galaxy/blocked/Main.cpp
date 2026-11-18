#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <FML/FieldLevel/FixedIC.h>
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
struct BlockState {
    Field white;
    std::vector<double> parameters;
    int next = 0;
    double accept_sum = 0;
    uint64_t divergences = 0, scalar_accepted = 0, scalar_attempts = 0, support_rejections = 0, cache_builds = 0;
};
static void save(const FS::path & root, const std::string & identity, const BlockState & s) {
    CP::Codec::Encoder e;
    e.text("FML serial blocked pilot state 1");
    e.text(identity);
    e.integer(s.next);
    e.vector(s.white);
    e.vector(s.parameters);
    e.real(s.accept_sum);
    for (auto x : {s.divergences, s.scalar_accepted, s.scalar_attempts, s.support_rejections, s.cache_builds})
        e.integer(x);
    CP::Codec::write_file((root / "CURRENT.tmp").string(), e.bytes);
    FS::rename(root / "CURRENT.tmp", root / "CURRENT");
}
template <class Model>
static BlockState load(const FS::path & root, const std::string & identity, const Model & model) {
    const auto bytes = CP::Codec::read_file((root / "CURRENT").string(), 16384);
    CP::Codec::Decoder d{bytes};
    if (d.text(64) != "FML serial blocked pilot state 1" || d.text() != identity)
        throw std::runtime_error("Blocked target/plan/chain changed");
    BlockState s;
    s.next = int(d.size(1280));
    s.white = d.vector(model.cells());
    s.parameters = d.vector(5);
    s.accept_sum = d.real();
    s.divergences = d.integer();
    s.scalar_accepted = d.integer();
    s.scalar_attempts = d.integer();
    s.support_rejections = d.integer();
    s.cache_builds = d.integer();
    d.end();
    const uint64_t draws = std::max(0, s.next - 256);
    bool valid = std::isfinite(s.accept_sum) && s.accept_sum >= 0 && s.accept_sum <= draws && s.divergences <= draws &&
                 s.scalar_attempts == 4 * draws && s.scalar_accepted <= s.scalar_attempts &&
                 s.support_rejections <= s.scalar_attempts && s.cache_builds == 5 * uint64_t(s.next);
    for (double x : s.white)
        valid &= std::isfinite(x);
    valid &= model.in_support(s.parameters);
    if (!valid)
        throw std::runtime_error("Invalid blocked checkpoint state");
    return s;
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
                    throw std::invalid_argument(
                        "Usage: blocked_pilot DIRECTORY start|resume CHAIN [trajectory_budget]");
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
        const std::string problem_id =
            "FML-blocked-pilot-1:plan-sha256:38edf59b51100daf4d6f1d7b9a70a66e88fb5c2efdf80c4ae8a99434e52db721:CHAIN=" +
            std::to_string(chain);
        const auto identity = CP::identity(target, problem_id);
        const std::vector<std::vector<double>> starts = {{.8, 1.2, -.3, .2, -20},
                                                         {1.2, 2, .8, -.6, 30},
                                                         {.9, 2.4, -.8, .6, 50},
                                                         {1.1, .8, 1, -1, -40}};
        const auto initial = starts[chain];
        const auto frozen = initial;
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
        BlockState state;
        if (io.resume)
            state = load(root, identity, model);
        else {
            state.white = model.white_noise(4103 + chain);
            state.parameters = initial;
        }
        auto scalar_cache = [&](const Field & field, double alpha, const std::vector<double> & p) {
            auto trial = p;
            trial[0] = alpha;
            model.predict(field, trial, work);
            const auto columns = model.bias_modes(work);
            return FF::FixedNoiseLinearBiasLikelihood<4>(columns,
                                                         data,
                                                         target.likelihood().weights(),
                                                         1000. / target.likelihood().volume(),
                                                         {0, 0, 0, 0});
        };
        std::array<FF::Parameter, 4> priors;
        std::copy(model.parameters().begin() + 1, model.parameters().end(), priors.begin());
        if (!io.resume && io.max_iterations == 0) {
            double worst = 0;
            for (int seedcheck : {4103, 4104}) {
                const auto field = model.white_noise(seedcheck);
                const auto reference = scalar_cache(field, initial[0], initial);
                const std::array<double, 4> refbias = {initial[1], initial[2], initial[3], initial[4]};
                const double refgap = target.value(field, initial, work) - reference.value(refbias);
                for (double alpha : {.7, 1., 1.3}) {
                    const auto cache = scalar_cache(field, alpha, initial);
                    for (const auto & b :
                         std::vector<std::array<double, 4>>{{.2, -3, 2, 70}, {2, .5, -.3, 7}, {3, -1, -2, -60}}) {
                        const std::vector<double> trial = {alpha, b[0], b[1], b[2], b[3]};
                        const double full = target.value(field, trial, work), cached = cache.value(b);
                        worst = std::max(worst,
                                         std::abs((full - cached) - refgap) /
                                             std::max({1., std::abs(full), std::abs(cached)}));
                        const auto gradient = cache.gradient(b);
                        for (size_t j = 0; j < 4; ++j) {
                            auto plus = trial, minus = trial;
                            plus[j + 1] += 1e-5;
                            minus[j + 1] -= 1e-5;
                            const double fd =
                                (target.value(field, plus, work) - target.value(field, minus, work)) / 2e-5;
                            if (std::abs(fd - gradient[j]) > 1e-5 * std::max({1., std::abs(fd), std::abs(gradient[j])}))
                                throw std::runtime_error("Cached scalar gradient differs from full target");
                        }
                    }
                }
            }
            if (worst > 1e-10)
                throw std::runtime_error("Cached scalar target differs from full posterior");
            std::ofstream verified(root / "cache-verification.json");
            verified << std::setprecision(17)
                     << "{\"passed\":true,\"points\":18,\"bias_derivatives\":72,\"max_scaled_gap_error\":" << worst
                     << "}\n";
            verified.close();
            if (!verified)
                throw std::runtime_error("Cannot write cache verification");
        }
        const int stop = state.next + std::min(io.max_iterations, 1280 - state.next);
        for (; state.next < stop;) {
            const uint64_t seed = 53000 + chain * 100000 + 2 * state.next;
            FML::MCMC::HMCOptions field_options;
            field_options.warmup = 0;
            field_options.samples = 1;
            field_options.n_leapfrog = 8;
            field_options.step_size = .15;
            field_options.jitter = false;
            field_options.adapt_mass = false;
            field_options.adapt_step_size = false;
            field_options.seed = seed;
            const auto field_result =
                FF::DETAIL::sample_run(target, state.white, state.parameters, field_options, false, {}, frozen);
            state.white = field_result.last;
            FF::FixedICOptions scalar_options;
            scalar_options.warmup = 0;
            scalar_options.samples = 4;
            scalar_options.bias_sweeps = 8;
            scalar_options.amplitude_step = .07;
            scalar_options.seed = seed + 1;
            std::array<double, 5> point;
            std::copy(state.parameters.begin(), state.parameters.end(), point.begin());
            auto provider = [&](double alpha) { return scalar_cache(state.white, alpha, state.parameters); };
            const auto scalar_result =
                FF::sample_fixed_ic_joint_amplitude<4>(provider, model.parameters()[0], priors, point, scalar_options);
            state.parameters.assign(scalar_result.draws.back().begin(), scalar_result.draws.back().end());
            state.cache_builds += scalar_result.forward_evaluations;
            if (state.next >= 256) {
                state.accept_sum += field_result.acceptance;
                state.divergences += field_result.divergences;
                state.scalar_accepted += scalar_result.amplitude_accepted;
                state.scalar_attempts += scalar_result.amplitude_attempts;
                state.support_rejections += scalar_result.support_rejections;
                record(root,
                       state.white,
                       state.parameters,
                       model.predict(state.white, state.parameters, work),
                       state.next - 256);
            }
            ++state.next;
            if (state.next % 64 == 0)
                save(root, identity, state);
        }
        save(root, identity, state);
        const int draws = std::max(0, state.next - 256);
        const auto final = root / "summary.json", temporary = root / "summary.json.tmp";
        std::ofstream out(temporary);
        out << std::setprecision(17)
            << "{\"synthetic_blocked_pilot\":true,\"mesh\":4,\"product_mesh\":8,\"density_mesh\":16,\"ranks\":1,"
               "\"completed_iterations\":"
            << state.next << ",\"completed_samples\":" << draws
            << ",\"warmup\":256,\"samples\":1024,\"acceptance\":" << (draws ? state.accept_sum / draws : 0)
            << ",\"step_size\":0.15,\"divergences\":" << state.divergences
            << ",\"scalar_accepted\":" << state.scalar_accepted << ",\"scalar_attempts\":" << state.scalar_attempts
            << ",\"support_rejections\":" << state.support_rejections << ",\"cache_builds\":" << state.cache_builds
            << "}\n";
        out.close();
        if (!out)
            throw std::runtime_error("Cannot publish blocked summary");
        FS::rename(temporary, final);
        std::cout << "Completed blocked sweeps: " << state.next << "/1280; retained " << draws << '\n';
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0)
            std::cerr << e.what() << '\n';
        return 1;
    }
}
