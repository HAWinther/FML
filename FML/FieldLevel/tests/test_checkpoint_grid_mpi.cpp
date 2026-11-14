#include <FML/PencilDomain/PencilGrid.h>
#include <FML/FieldLevel/SurveyCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
using Pencil = FML::GRID::PencilGrid<3>;
using PL = FF::DistributedLinearModelT<Pencil>;
using PN = FF::DistributedLPTModelT<Pencil>;
static std::vector<double> edges(int parts, int n) {
    std::vector<double> x(parts + 1);
    for (int i = 0; i <= parts; ++i)
        x[i] = double(i * n / parts) / n;
    return x;
}
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
static FML::MCMC::HMCResult read_reference(const FS::path & path, size_t n, size_t np, int samples, bool vary) {
    const auto bytes = C::read_file(path.string(), C::size_bound(n, np) + size_t(samples) * (np + 1) * 16);
    C::Decoder d{bytes};
    FML::MCMC::HMCResult r;
    r.last = d.vector(n);
    r.diagonal = d.vector(vary ? np : 0);
    r.potential = d.vector(samples);
    if (d.size(samples) != size_t(samples))
        throw std::runtime_error("Wrong reference row count");
    for (int i = 0; i < samples; ++i)
        r.recorded.push_back(d.vector(np));
    r.acceptance = d.real();
    r.step_size = d.real();
    r.divergences = long(d.size(LONG_MAX));
    d.end();
    return r;
}
static bool same(const FML::MCMC::HMCResult & a, const FML::MCMC::HMCResult & b) {
    return a.last == b.last && a.diagonal == b.diagonal && a.acceptance == b.acceptance && a.step_size == b.step_size &&
           a.divergences == b.divergences;
}
template <int Kind, class Model>
auto likelihood(const Model & m, bool changed = false, bool masked_change = false) {
    if constexpr (Kind == 0) {
        FF::Modes data(m.modes().size(), 0);
        if (changed)
            for (size_t i = 0; i < data.size(); ++i)
                if (m.modes()[i].n[0] == 1 && m.modes()[i].n[1] == 0 && m.modes()[i].n[2] == 1)
                    data[i] = .01;
        return FF::Execution<Model>::template likelihood<FF::GaussianFieldLikelihood>(m, std::move(data));
    } else {
        auto window = FF::full_survey_window(m);
        Field data(m.cells()), variance(m.cells()), exposure(m.cells());
        std::vector<std::uint64_t> counts(m.cells());
        for (size_t q = 0; q < m.cells(); ++q) {
            size_t id = m.layout().global_cell_id(q);
            window.observed[q] = id % 5 != 4;
            window.selection[q] = .5 + .02 * (id % 7);
            data[q] = window.observed[q] ? .01 * std::cos(double(id)) : NAN;
            variance[q] = window.observed[q] ? 1 + .1 * (id % 3) : NAN;
            counts[q] = id % 3;
            exposure[q] = window.observed[q] ? 1.5 + .2 * (id % 5) : NAN;
            if (changed && FML::ThisTask == 0 && q == 0)
                window.selection[q] += .01;
            if (masked_change && !window.observed[q]) {
                data[q] = 123;
                variance[q] = -7;
                counts[q] = 99;
                exposure[q] = NAN;
            }
        }
        if constexpr (Kind == 1)
            return FF::SurveyGaussianLikelihood(m, std::move(window), std::move(data), std::move(variance));
        else
            return FF::SurveyPoissonLikelihood(m, std::move(window), std::move(counts), std::move(exposure), .1);
    }
}
template <int Kind, class Model>
void exercise(const FS::path & root, const std::string & phase, FF::Configuration c, const FML::Domain & domain) {
    Model m(c, FF::Spectrum::fixed([](double) { return 200.; }), {}, {}, domain);
    auto l = likelihood<Kind>(m);
    FF::BasicPosterior<Model, decltype(l)> posterior(m, std::move(l));
    auto p = m.initial_parameters();
    const std::string label = std::to_string(int(c.evolution)) + "-" + std::to_string(Kind);
    FML::MCMC::HMCOptions o;
    o.warmup = (int(c.evolution) == 0 && Kind == 0) ? 120 : 12;
    o.samples = 5;
    o.n_leapfrog = 2;
    o.seed = 91;
    o.step_size = .01;
    o.adapt_mass = (o.warmup == 120);
    FF::CheckpointOptions io;
    io.directory = (root / label).string();
    io.problem_id = "pencil-fixture-v1:P200";
    io.every = 7;
    const auto reference = root / (label + ".reference.rank" + std::to_string(FML::ThisTask));
    auto fixed = o;
    fixed.warmup = 4;
    fixed.samples = 3;
    fixed.adapt_mass = false;
    fixed.adapt_step_size = false;
    fixed.seed = 92;
    auto fio = io;
    fio.directory += "-fixed";
    const auto fref = root / (label + ".fixed.rank" + std::to_string(FML::ThisTask));
    if (phase == "reject") {
        io.resume = true;
        io.max_iterations = 0;
        check(FML::TESTS::throws([&] { FF::sample_checkpointed(posterior, {}, {}, o, io); }),
              label + " rejects changed backend/process grid/domain edges/rank count");
        return;
    }
    if (phase == "prepare") {
        const auto full = FF::sample(posterior, m.white_noise(13), p, o, true);
        FML::COMMUNICATION::collective_try([&] { write_reference(reference, full); }, "Grid checkpoint reference");
        io.max_iterations = 5;
        auto prefix = FF::sample_checkpointed(posterior, m.white_noise(13), p, o, io, true);
        io.resume = true;
        io.max_iterations = o.warmup - 5 + 2;
        auto part = FF::sample_checkpointed(posterior, {}, {}, o, io, true);
        check(prefix.recorded.empty() &&
                  part.recorded == std::vector<std::vector<double>>(full.recorded.begin(), full.recorded.begin() + 2) &&
                  part.potential == Field(full.potential.begin(), full.potential.begin() + 2),
              label + " split warmup and sampling prefix are bitwise exact");
        const auto state = CP::load(io.directory, CP::identity(posterior, io.problem_id), m.cells(), p.size(), true);
        check(state.sampler.next_iteration == o.warmup + 2 && state.frozen == p,
              label + " checkpoint retains iteration and frozen metric parameters");
        const auto ffull = FF::sample(posterior, m.white_noise(14), p, fixed, false);
        FML::COMMUNICATION::collective_try([&] { write_reference(fref, ffull); }, "Fixed grid reference");
        fio.max_iterations = 2;
        FF::sample_checkpointed(posterior, m.white_noise(14), p, fixed, fio, false);
    } else if (phase == "resume") {
        FML::MCMC::HMCResult full, ffull;
        FML::COMMUNICATION::collective_try(
            [&] {
                full = read_reference(reference, m.cells() + p.size(), p.size(), o.samples, true);
                ffull = read_reference(fref, m.cells(), p.size(), fixed.samples, false);
            },
            "Grid checkpoint reference read");
        io.resume = true;
        std::vector<size_t> indices;
        const auto suffix =
            FF::sample_checkpointed(posterior, {}, {}, o, io, true, [&](const auto &, const auto &, size_t i) {
                indices.push_back(i);
            });
        check(same(suffix, full) &&
                  suffix.recorded == std::vector<std::vector<double>>(full.recorded.begin() + 2, full.recorded.end()) &&
                  suffix.potential == Field(full.potential.begin() + 2, full.potential.end()) &&
                  indices == std::vector<size_t>{2, 3, 4},
              label + " fresh-process joint continuation is bitwise exact (state/rows/stats/metric/sink)");
        fio.resume = true;
        const auto fsuffix = FF::sample_checkpointed(posterior, {}, {}, fixed, fio, false);
        check(same(fsuffix, ffull) && fsuffix.recorded == ffull.recorded && fsuffix.potential == ffull.potential,
              label + " fresh-process fixed-parameter continuation is bitwise exact");
        io.max_iterations = 0;
        auto changed = likelihood<Kind>(m, true);
        FF::BasicPosterior<Model, decltype(changed)> invalid(m, std::move(changed));
        check(FML::TESTS::throws([&] { FF::sample_checkpointed(invalid, {}, {}, o, io); }),
              label + " one-rank target change rejects collectively");
        if constexpr (Kind != 0) {
            auto equivalent = likelihood<Kind>(m, false, true);
            FF::BasicPosterior<Model, decltype(equivalent)> target(m, std::move(equivalent));
            auto same_target = FF::sample_checkpointed(target, {}, {}, o, io);
            check(same_target.last == full.last && same_target.recorded.empty(),
                  label + " masked ignored values remain compatible");
        }
    } else
        throw std::invalid_argument("Expected prepare/resume/reject");
}
template <class Linear, class Nonlinear>
void all(const FS::path & root, const std::string & phase, const FML::Domain & domain) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = FF::Evolution::Linear;
    exercise<0, Linear>(root, phase, c, domain);
    exercise<1, Linear>(root, phase, c, domain);
    exercise<2, Linear>(root, phase, c, domain);
    for (auto evolution : {FF::Evolution::Zeldovich, FF::Evolution::LPT2}) {
        c.evolution = evolution;
        exercise<0, Nonlinear>(root, phase, c, domain);
        exercise<1, Nonlinear>(root, phase, c, domain);
        exercise<2, Nonlinear>(root, phase, c, domain);
    }
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::unitbuf;
    if (argc < 3)
        return 1;
    // Replica validation must ignore local field lengths/values, but retain scalar tails/control state.
    CP::State probe;
    probe.sampler.n_custom = size_t(FML::ThisTask) + 1;
    probe.sampler.position.assign(probe.sampler.n_custom, double(FML::ThisTask));
    probe.sampler.gradient.assign(probe.sampler.n_custom, -double(FML::ThisTask));
    probe.sampler.position.push_back(.125);
    probe.sampler.gradient.push_back(.25);
    check(!FML::TESTS::throws([&] { CP::replicas(probe); }), "replica check accepts unequal local field extents");
    if (FML::NTasks > 1) {
        if (FML::ThisTask == 0)
            probe.sampler.position.back() += 1;
        check(FML::TESTS::throws([&] { CP::replicas(probe); }), "replica check still rejects unequal scalar tails");
        if (FML::ThisTask == 0) {
            probe.sampler.position.back() -= 1;
            probe.sampler.next_iteration += 1;
        }
        check(FML::TESTS::throws([&] { CP::replicas(probe); }), "replica check still rejects unequal sampler controls");
    }
    FS::path root = argv[1];
    std::string phase = argv[2], geometry = argc > 3 ? argv[3] : "normal";
    FML::COMMUNICATION::collective_try(
        [&] {
            if (FML::ThisTask == 0)
                FS::create_directories(root);
        },
        "Grid checkpoint directory");
    if (geometry == "slab")
        all<FF::DistributedLinearModel, FF::DistributedLPTModel>(root, phase, FML::Domain::uniform_slab(FML::NTasks));
    else {
        int px = FML::NTasks % 2 == 0 ? 2 : 1, py = FML::NTasks / px;
        if (geometry == "other-process-grid") {
            px = FML::NTasks;
            py = 1;
        }
        auto x = edges(px, 8), y = edges(py, 8);
        if (geometry == "other-y-edge" && py > 1)
            y[1] = .25;
        if (geometry == "uneven" && px == 2)
            x[1] = .375;
        all<PL, PN>(root, phase, FML::Domain::pencil(x, y));
    }
    return FML::TESTS::summary("pencil field/survey checkpoint");
}
