#define FML_FIELDLEVEL_CHECKPOINT_TEST_FAILURE
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::TESTS::check_local;
static void write_reference(const FS::path & path, const FML::MCMC::HMCResult & r) {
    C::Encoder e;
    e.vector(r.last);
    e.vector(r.diagonal);
    e.vector(r.potential);
    e.integer(r.recorded.size());
    for (const auto & row : r.recorded)
        e.vector(row);
    e.real(r.acceptance);
    e.real(r.step_size);
    e.integer(r.divergences);
    C::write_file(path.string(), e.bytes);
}
static FML::MCMC::HMCResult read_reference(const FS::path & path, size_t n, size_t np, int samples) {
    const auto bytes = C::read_file(path.string(), C::size_bound(n, np) + size_t(samples) * (np + 1) * 16);
    C::Decoder d{bytes};
    FML::MCMC::HMCResult r;
    r.last = d.vector(n);
    r.diagonal = d.vector(np);
    r.potential = d.vector(samples);
    if (d.size(samples) != size_t(samples))
        throw std::runtime_error("Wrong reference sample count");
    for (int i = 0; i < samples; ++i)
        r.recorded.push_back(d.vector(np));
    r.acceptance = d.real();
    r.step_size = d.real();
    r.divergences = long(d.size(LONG_MAX));
    d.end();
    return r;
}
template <class Forward>
static void exercise(const FS::path & root, const std::string & phase, FF::Configuration config) {
    const std::string name = config.evolution == FF::Evolution::Linear    ? "linear" :
                             config.evolution == FF::Evolution::Zeldovich ? "1lpt" :
                                                                            "2lpt";
    auto spectrum = FF::Spectrum::fixed([](double) { return 200.; });
    Forward model(config, spectrum);
    auto work = model.workspace();
    auto p = model.initial_parameters();
    const auto data = model.mock(model.white_noise(11), p, 12, work);
    FF::BasicPosterior<Forward> posterior(model, data);
    FML::MCMC::HMCOptions options;
    options.warmup = 120;
    options.samples = 30;
    options.n_leapfrog = 4;
    options.step_size = .01;
    options.target_acceptance = .9;
    options.seed = 37;
    FF::CheckpointOptions io;
    io.directory = (root / name).string();
    io.problem_id = "fixture-v1-fixed-spectrum-200";
    io.every = 17;
    const auto reference = root / (name + "-reference.rank" + std::to_string(FML::ThisTask) + ".hmc");
    if (phase == "rank-error") {
        io.resume = true;
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " restart rejects changed MPI rank count");
        return;
    }
    if (phase == "prepare" || phase == "all") {
        const auto uninterrupted = FF::sample(posterior, model.white_noise(13), p, options, true);
        FML::COMMUNICATION::collective_try([&] { write_reference(reference, uninterrupted); },
                                           "Checkpoint test reference");
        std::vector<std::vector<double>> prefix;
        std::vector<double> values;
        std::vector<size_t> indices;
        auto sink = [&](const auto &, const auto &, size_t i) { indices.push_back(i); };
        for (int budget : {27, 33, 55, 10}) {
            io.max_iterations = budget;
            auto chunk = io.resume ?
                             FF::sample_checkpointed(posterior, {}, {}, options, io, true, sink) :
                             FF::sample_checkpointed(posterior, model.white_noise(13), p, options, io, true, sink);
            prefix.insert(prefix.end(), chunk.recorded.begin(), chunk.recorded.end());
            values.insert(values.end(), chunk.potential.begin(), chunk.potential.end());
            io.resume = true;
        }
        check_local(
            prefix == std::vector<std::vector<double>>(uninterrupted.recorded.begin(),
                                                       uninterrupted.recorded.begin() + 5) &&
                values == std::vector<double>(uninterrupted.potential.begin(), uninterrupted.potential.begin() + 5) &&
                indices == std::vector<size_t>({0, 1, 2, 3, 4}),
            name + " warmup-window disk resumes reproduce uninterrupted prefix and global sink indices bitwise");
        const auto saved =
            CP::load(io.directory, CP::identity(posterior, io.problem_id), model.cells(), p.size(), true);
        check_local(saved.frozen == p && saved.sampler.next_iteration == 125 && saved.sampler.window == 2,
                    name + " original field metric, iteration and adaptation window preserved");
        check_local(std::vector<double>(saved.sampler.position.begin() + model.cells(), saved.sampler.position.end()) !=
                        p,
                    name + " joint scalar state moved away from frozen metric parameters");
    }
    if (phase == "prepare" || phase == "all") {
        auto fixed_options = options;
        fixed_options.warmup = 20;
        fixed_options.samples = 6;
        fixed_options.seed = 87;
        const auto fixed_full = FF::sample(posterior, model.white_noise(13), p, fixed_options, false);
        auto fixed_io = io;
        fixed_io.directory = (root / (name + "-fixed")).string();
        fixed_io.resume = false;
        fixed_io.every = 0;
        fixed_io.max_iterations = 9;
        FF::sample_checkpointed(posterior, model.white_noise(13), p, fixed_options, fixed_io, false);
        fixed_io.resume = true;
        fixed_io.max_iterations = INT_MAX;
        const auto fixed = FF::sample_checkpointed(posterior, {}, {}, fixed_options, fixed_io, false);
        check_local(fixed.last == fixed_full.last && fixed.recorded == fixed_full.recorded &&
                        fixed.potential == fixed_full.potential && fixed.acceptance == fixed_full.acceptance &&
                        fixed.step_size == fixed_full.step_size,
                    name + " fixed-parameter disk restart preserves field trajectory exactly");
    }
    if (phase == "resume" || phase == "all") {
        FML::MCMC::HMCResult expected;
        FML::COMMUNICATION::collective_try(
            [&] { expected = read_reference(reference, model.cells() + p.size(), p.size(), options.samples); },
            "Checkpoint reference read");
        io.resume = true;
        io.max_iterations = INT_MAX;
        std::vector<size_t> indices;
        const auto suffix =
            FF::sample_checkpointed(posterior, {}, {}, options, io, true, [&](const auto &, const auto &, size_t i) {
                indices.push_back(i);
            });
        bool indexok = indices.size() == 25;
        for (size_t i = 0; i < indices.size(); ++i)
            indexok &= indices[i] == i + 5;
        check_local(suffix.last == expected.last &&
                        suffix.recorded ==
                            std::vector<std::vector<double>>(expected.recorded.begin() + 5, expected.recorded.end()) &&
                        suffix.potential ==
                            std::vector<double>(expected.potential.begin() + 5, expected.potential.end()) &&
                        indexok,
                    name + " fresh-process continuation matches full state, rows, potentials and sink indices bitwise");
        check_local(suffix.diagonal == expected.diagonal && suffix.step_size == expected.step_size &&
                        suffix.acceptance == expected.acceptance && suffix.divergences == expected.divergences &&
                        suffix.completed_iterations == 150 && suffix.completed_samples == 30,
                    name + " cumulative adaptation/acceptance/divergence results preserved exactly");
        io.max_iterations = 0;
        const auto generation = CP::current_generation(io.directory);
        if (FML::ThisTask == 0)
            CP::fail_shard_write_once = true;
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " one-rank shard write failure propagates collectively");
        check_local(CP::current_generation(io.directory) == generation,
                    name + " failed save leaves previous manifest unchanged");
        const auto intact = FF::sample_checkpointed(posterior, {}, {}, options, io);
        check_local(intact.last == expected.last && intact.recorded.empty(),
                    name + " prior checkpoint remains usable after interrupted save");
        auto badio = io;
        badio.problem_id = "different-version";
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, badio); }),
                    name + " changed caller target identity rejected");
        auto changed = options;
        ++changed.n_leapfrog;
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, changed, io); }),
                    name + " changed HMC controls rejected");
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io, false); }),
                    name + " changed joint/fixed sampling mode rejected");
        auto modified = data;
        if (FML::ThisTask == 0 && !modified.empty())
            modified[0] += FML::FIELDLEVEL::Complex(.001, 0);
        FF::BasicPosterior<Forward> changed_data(model, modified);
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(changed_data, {}, {}, options, io); }),
                    name + " one-rank data change rejected");
        Forward changed_model(config, FF::Spectrum::fixed([](double) { return 220.; }));
        FF::BasicPosterior<Forward> changed_target(changed_model, data);
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(changed_target, {}, {}, options, io); }),
                    name + " unversioned spectrum callback change caught by saved force/potential check");
        const auto original =
            CP::load(io.directory, CP::identity(posterior, io.problem_id), model.cells(), p.size(), true);
        auto poisoned = original;
        if (FML::ThisTask == 0)
            poisoned.sampler.gradient[0] += 1;
        CP::save(io.directory, poisoned);
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " rank-local saved force mismatch rejects collectively");
        CP::save(io.directory, original);
        const auto gen = CP::current_generation(io.directory);
        const auto file = CP::shard(io.directory, gen);
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    FS::copy_file(file, file.string() + ".generation-backup", FS::copy_options::overwrite_existing);
                    FS::copy_file(CP::shard(io.directory, generation), file, FS::copy_options::overwrite_existing);
                }
            },
            "Checkpoint mixed-generation fixture");
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " wrong-generation shard rejected even with matching sampler state");
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    FS::copy_file(file.string() + ".generation-backup", file, FS::copy_options::overwrite_existing);
                    FS::remove(file.string() + ".generation-backup");
                }
            },
            "Checkpoint generation fixture restoration");
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    FS::copy_file(file, file.string() + ".backup", FS::copy_options::overwrite_existing);
                    std::fstream f(file, std::ios::in | std::ios::out | std::ios::binary);
                    f.seekg(-1, std::ios::end);
                    char c;
                    f.read(&c, 1);
                    c ^= 1;
                    f.seekp(-1, std::ios::end);
                    f.write(&c, 1);
                }
            },
            "Checkpoint corruption fixture");
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " corrupted rank shard rejected collectively");
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    FS::copy_file(file.string() + ".backup", file, FS::copy_options::overwrite_existing);
                    FS::remove(file.string() + ".backup");
                    FS::rename(file, file.string() + ".missing");
                }
            },
            "Checkpoint missing fixture");
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " missing rank shard rejected collectively");
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0)
                    FS::rename(file.string() + ".missing", file);
            },
            "Checkpoint fixture restoration");
    }
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::string directory = argc > 1 ? argv[1] : "";
    const std::string phase = argc > 2 ? argv[2] : "all";
    if (directory.empty() && FML::ThisTask == 0)
        directory = (FS::temp_directory_path() /
                     ("fml-field-checkpoint-" +
                      std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())))
                        .string();
    directory = CP::broadcast_string(directory);
    FML::COMMUNICATION::collective_try(
        [&] {
            if (FML::ThisTask == 0)
                FS::create_directories(directory);
        },
        "Checkpoint fixture directory");
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = FF::Evolution::Linear;
    exercise<FF::DistributedLinearModel>(directory, phase, c);
    c.evolution = FF::Evolution::Zeldovich;
    exercise<FF::DistributedLPTModel>(directory, phase, c);
    c.evolution = FF::Evolution::LPT2;
    exercise<FF::DistributedLPTModel>(directory, phase, c);
    if (argc == 1)
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0)
                    FS::remove_all(directory);
            },
            "Checkpoint fixture cleanup");
    return FML::TESTS::summary("field-level checkpoint");
}
