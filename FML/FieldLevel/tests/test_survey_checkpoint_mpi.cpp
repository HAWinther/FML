#include <FML/FieldLevel/SurveyCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check_local;
struct Fixture {
    FF::SurveyWindow window;
    Field data, variance, exposure;
    std::vector<std::uint64_t> counts;
    double offset = .1;
};
template <class Forward>
static Fixture fixture(const Forward & m) {
    Fixture f;
    f.window = FF::full_survey_window(m);
    f.window.origin = {-50, -50, -50};
    f.data.resize(m.cells());
    f.variance.resize(m.cells());
    f.exposure.resize(m.cells());
    f.counts.resize(m.cells());
    const int N = m.configuration().mesh;
    for (size_t q = 0; q < m.cells(); ++q) {
        const size_t id = size_t(m.layout().local_start) * N * N + q;
        f.window.observed[q] = id % 5 != 4;
        f.window.selection[q] = .5 + .02 * (id % 7);
        f.data[q] = f.window.observed[q] ? .01 * std::cos(double(id)) : NAN;
        f.variance[q] = f.window.observed[q] ? .8 + .1 * (id % 3) : NAN;
        f.counts[q] = id % 4;
        f.exposure[q] = f.window.observed[q] ? 1.5 + .2 * (id % 5) : NAN;
    }
    return f;
}
template <bool Poisson, class Forward>
static auto likelihood(const Forward & m, Fixture f) {
    if constexpr (Poisson)
        return FF::SurveyPoissonLikelihood(m,
                                           std::move(f.window),
                                           std::move(f.counts),
                                           std::move(f.exposure),
                                           f.offset);
    else
        return FF::SurveyGaussianLikelihood(m, std::move(f.window), std::move(f.data), std::move(f.variance));
}
static void write_reference(const FS::path & file, const FML::MCMC::HMCResult & r) {
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
    C::write_file(file.string(), e.bytes);
}
static FML::MCMC::HMCResult read_reference(const FS::path & file, size_t n, size_t np, int samples, bool vary = true) {
    const auto bytes = C::read_file(file.string(), C::size_bound(n, np) + size_t(samples) * (np + 1) * 16);
    C::Decoder d{bytes};
    FML::MCMC::HMCResult r;
    r.last = d.vector(n);
    r.diagonal = d.vector(vary ? np : 0);
    r.potential = d.vector(samples);
    if (d.size(samples) != size_t(samples))
        throw std::runtime_error("Wrong reference rows");
    for (int i = 0; i < samples; ++i)
        r.recorded.push_back(d.vector(np));
    r.acceptance = d.real();
    r.step_size = d.real();
    r.divergences = long(d.size(LONG_MAX));
    d.end();
    return r;
}
static bool stats_equal(const FML::MCMC::HMCResult & a, const FML::MCMC::HMCResult & b) {
    return a.last == b.last && a.diagonal == b.diagonal && a.step_size == b.step_size && a.acceptance == b.acceptance &&
           a.divergences == b.divergences;
}
template <bool Poisson, class Forward>
static void exercise(const FS::path & root, const std::string & phase, FF::Configuration c) {
    const std::string name = (Poisson ? "poisson-" : "gaussian-") + std::to_string(int(c.evolution));
    Forward m(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto p = m.initial_parameters();
    const auto f = fixture(m);
    auto like = likelihood<Poisson>(m, f);
    FF::BasicPosterior<Forward, decltype(like)> posterior(m, std::move(like));
    FML::MCMC::HMCOptions options;
    options.warmup = 120;
    options.samples = 20;
    options.n_leapfrog = 3;
    options.seed = 91;
    options.target_acceptance = .9;
    FF::CheckpointOptions io;
    io.directory = (root / name).string();
    io.problem_id = "survey-fixture-v1:P200";
    io.every = 17;
    const auto reference = root / (name + ".reference.rank" + std::to_string(FML::ThisTask));
    auto fixed_options = options;
    fixed_options.warmup = 30;
    fixed_options.samples = 12;
    fixed_options.seed = 92;
    auto fixed_io = io;
    fixed_io.directory += "-fixed";
    fixed_io.every = 7;
    const auto fixed_reference = root / (name + ".fixed.reference.rank" + std::to_string(FML::ThisTask));
    if (phase == "rank-error") {
        io.resume = true;
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, options, io); }),
                    name + " changed rank count rejected");
        return;
    }
    if (phase == "prepare") {
        const auto full = FF::sample(posterior, m.white_noise(13), p, options, true);
        FML::COMMUNICATION::collective_try([&] { write_reference(reference, full); }, "Survey checkpoint reference");
        std::vector<std::vector<double>> rows;
        std::vector<double> values;
        std::vector<size_t> indices;
        auto sink = [&](const auto &, const auto &, size_t i) { indices.push_back(i); };
        for (int budget : {27, 33, 55, 10}) {
            io.max_iterations = budget;
            const auto part = io.resume ?
                                  FF::sample_checkpointed(posterior, {}, {}, options, io, true, sink) :
                                  FF::sample_checkpointed(posterior, m.white_noise(13), p, options, io, true, sink);
            rows.insert(rows.end(), part.recorded.begin(), part.recorded.end());
            values.insert(values.end(), part.potential.begin(), part.potential.end());
            io.resume = true;
        }
        check_local(rows == std::vector<std::vector<double>>(full.recorded.begin(), full.recorded.begin() + 5) &&
                        values == std::vector<double>(full.potential.begin(), full.potential.begin() + 5) &&
                        indices == std::vector<size_t>({0, 1, 2, 3, 4}),
                    name + " joint warmup splits reproduce prefix and global sink indices exactly");
        const auto state = CP::load(io.directory, CP::identity(posterior, io.problem_id), m.cells(), p.size(), true);
        check_local(state.frozen == p && state.sampler.next_iteration == 125 && state.sampler.window == 2,
                    name + " preserves frozen field metric and adaptation windows");
        const auto fixed = FF::sample(posterior, m.white_noise(14), p, fixed_options, false);
        FML::COMMUNICATION::collective_try([&] { write_reference(fixed_reference, fixed); },
                                           "Fixed survey checkpoint reference");
        fixed_io.max_iterations = 10;
        FF::sample_checkpointed(posterior, m.white_noise(14), p, fixed_options, fixed_io, false);
    } else if (phase == "resume") {
        FML::MCMC::HMCResult full, fixed;
        FML::COMMUNICATION::collective_try(
            [&] {
                full = read_reference(reference, m.cells() + p.size(), p.size(), options.samples);
                fixed = read_reference(fixed_reference, m.cells(), p.size(), fixed_options.samples, false);
            },
            "Read survey references");
        io.resume = true;
        std::vector<size_t> indices;
        const auto suffix =
            FF::sample_checkpointed(posterior, {}, {}, options, io, true, [&](const auto &, const auto &, size_t i) {
                indices.push_back(i);
            });
        bool indexok = indices.size() == 15;
        for (size_t i = 0; i < indices.size(); ++i)
            indexok &= indices[i] == i + 5;
        check_local(
            stats_equal(suffix, full) &&
                suffix.recorded == std::vector<std::vector<double>>(full.recorded.begin() + 5, full.recorded.end()) &&
                suffix.potential == std::vector<double>(full.potential.begin() + 5, full.potential.end()) && indexok &&
                suffix.completed_iterations == 140 && suffix.completed_samples == 20,
            name + " fresh-process joint continuation matches state, rows, stats, metric and sink indices bitwise");
        fixed_io.resume = true;
        const auto fixed_suffix = FF::sample_checkpointed(posterior, {}, {}, fixed_options, fixed_io, false);
        check_local(stats_equal(fixed_suffix, fixed) && fixed_suffix.recorded == fixed.recorded &&
                        fixed_suffix.potential == fixed.potential,
                    name + " fresh-process fixed-parameter continuation matches uninterrupted chain bitwise");
        io.max_iterations = 0;
        auto reject = [&](Fixture changed, const std::string & what) {
            auto altered = likelihood<Poisson>(m, std::move(changed));
            FF::BasicPosterior<Forward, decltype(altered)> target(m, std::move(altered));
            check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(target, {}, {}, options, io); }),
                        name + " rejects " + what);
        };
        auto changed = f;
        if (FML::ThisTask == 0)
            changed.window.observed[0] = 0;
        reject(changed, "one-rank mask change");
        changed = f;
        if (FML::ThisTask == 0)
            changed.window.selection[0] += .01;
        reject(changed, "one-rank selection change");
        changed = f;
        changed.window.origin[0] += 1;
        reject(changed, "replicated geometry relabeling");
        if constexpr (Poisson) {
            changed = f;
            if (FML::ThisTask == 0)
                ++changed.counts[0];
            reject(changed, "one-rank count change");
            changed = f;
            if (FML::ThisTask == 0)
                changed.exposure[0] *= 1.01;
            reject(changed, "one-rank exposure change");
            changed = f;
            changed.offset += .1;
            reject(changed, "count-link offset change");
        } else {
            changed = f;
            if (FML::ThisTask == 0)
                changed.data[0] += .001;
            reject(changed, "one-rank data change");
            changed = f;
            if (FML::ThisTask == 0)
                changed.variance[0] *= 1.01;
            reject(changed, "one-rank relative-variance change");
        }
        auto other = likelihood<!Poisson>(m, f);
        FF::BasicPosterior<Forward, decltype(other)> other_target(m, std::move(other));
        check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(other_target, {}, {}, options, io); }),
                    name + " rejects changed likelihood family");
        changed = f;
        for (size_t q = 0; q < m.cells(); ++q)
            if (!changed.window.observed[q]) {
                changed.data[q] = 123;
                changed.variance[q] = -7;
                changed.counts[q] = 99;
                changed.exposure[q] = NAN;
            }
        auto masked = likelihood<Poisson>(m, changed);
        FF::BasicPosterior<Forward, decltype(masked)> equivalent(m, std::move(masked));
        const auto same = FF::sample_checkpointed(equivalent, {}, {}, options, io);
        check_local(same.last == full.last && same.recorded.empty(),
                    name + " ignored masked values are canonicalized for identity");
    } else
        throw std::invalid_argument("Expected prepare, resume or rank-error");
}
static void prior_family_test(const FS::path & root) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.evolution = FF::Evolution::Linear;
    FF::Parameter b{"b1", 1, 1, 1, 0, 2, FF::Prior::Gaussian};
    auto spectrum = FF::Spectrum::fixed([](double) { return 200.; });
    FF::DistributedLinearModel gaussian(c, spectrum, {}, {b});
    b.prior = FF::Prior::Uniform;
    FF::DistributedLinearModel uniform(c, spectrum, {}, {b});
    FF::BasicPosterior gp(gaussian, FF::Modes(gaussian.modes().size(), 0)),
        up(uniform, FF::Modes(uniform.modes().size(), 0));
    Field s(gaussian.cells(), 0), gs;
    std::vector<double> grad;
    auto w = gaussian.workspace(), uw = uniform.workspace();
    const auto p = gaussian.initial_parameters();
    const double U = gp.evaluate(s, p, w, gs, grad), V = up.evaluate(s, p, uw, gs, grad);
    check_local(U == V, "prior-family fixture has equal cached potentials at the prior center");
    FF::CheckpointOptions io;
    io.directory = (root / "prior-family").string();
    io.problem_id = "prior-test";
    io.max_iterations = 0;
    FML::MCMC::HMCOptions options;
    options.warmup = 2;
    options.samples = 2;
    FF::sample_checkpointed(gp, s, p, options, io);
    io.resume = true;
    check_local(FML::TESTS::throws([&] { FF::sample_checkpointed(up, {}, {}, options, io); }),
                "identity rejects changed prior family even at an unchanged cached potential/force");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::unitbuf;
    if (argc != 3) {
        if (FML::ThisTask == 0)
            std::cerr << "Usage: test_survey_checkpoint_mpi DIRECTORY prepare|resume|rank-error\n";
        return 1;
    }
    C::Encoder encoded;
    CP::SurveyInputHash streamed;
    const Field values(2048, .123);
    encoded.vector(values);
    streamed.vector(values);
    const std::string text(9000, 'x');
    encoded.text(text);
    streamed.text(text);
    encoded.integer(UINT64_MAX);
    streamed.integer(UINT64_MAX);
    check_local(streamed.finish() == C::checksum(encoded.bytes) && streamed.finish() == C::checksum(encoded.bytes),
                "bounded streaming identity hash matches monolithic codec across chunk boundaries");
    FS::path root = argv[1];
    const std::string phase = argv[2];
    FML::COMMUNICATION::collective_try(
        [&] {
            if (FML::ThisTask == 0)
                FS::create_directories(root);
        },
        "Survey checkpoint test directory");
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = FF::Evolution::Linear;
    exercise<false, FF::DistributedLinearModel>(root, phase, c);
    exercise<true, FF::DistributedLinearModel>(root, phase, c);
    c.evolution = FF::Evolution::Zeldovich;
    exercise<false, FF::DistributedLPTModel>(root, phase, c);
    exercise<true, FF::DistributedLPTModel>(root, phase, c);
    c.evolution = FF::Evolution::LPT2;
    exercise<false, FF::DistributedLPTModel>(root, phase, c);
    exercise<true, FF::DistributedLPTModel>(root, phase, c);
    if (phase == "resume")
        prior_family_test(root);
    return FML::TESTS::summary("survey checkpoint");
}
