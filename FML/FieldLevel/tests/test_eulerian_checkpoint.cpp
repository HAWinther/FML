#define FML_FIELDLEVEL_CHECKPOINT_TEST_FAILURE
#include <FML/FieldLevel/EulerianCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <fstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static auto priors() {
    return std::array<FF::Parameter, 4>{{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                         {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
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
    e.integer(r.completed_iterations);
    e.integer(r.completed_samples);
    C::write_file(path.string(), e.bytes);
}
static FML::MCMC::HMCResult read_reference(const FS::path & path, size_t n, size_t np, int samples, bool vary) {
    auto bytes = C::read_file(path.string(), C::size_bound(n, np) + size_t(samples) * (np + 1) * 16);
    C::Decoder d{bytes};
    FML::MCMC::HMCResult r;
    r.last = d.vector(n);
    r.diagonal = d.vector(vary ? np : 0);
    r.potential = d.vector(samples);
    if (d.size(samples) != size_t(samples))
        throw std::runtime_error("Reference count mismatch");
    for (int j = 0; j < samples; ++j)
        r.recorded.push_back(d.vector(np));
    r.acceptance = d.real();
    r.step_size = d.real();
    r.divergences = long(d.size(LONG_MAX));
    r.completed_iterations = int(d.size(INT_MAX));
    r.completed_samples = int(d.size(INT_MAX));
    d.end();
    return r;
}
static bool identical(const Field & a, const Field & b) {
    C::Encoder x, y;
    x.vector(a);
    y.vector(b);
    return x.bytes == y.bytes;
}
static bool identical(const std::vector<Field> & a, const std::vector<Field> & b) {
    if (a.size() != b.size())
        return false;
    for (size_t j = 0; j < a.size(); ++j)
        if (!identical(a[j], b[j]))
            return false;
    return true;
}
static bool same(const FML::MCMC::HMCResult & a, const FML::MCMC::HMCResult & b) {
    C::Encoder x, y;
    for (const auto * result : {&a, &b}) {
        auto & e = result == &a ? x : y;
        e.vector(result->last);
        e.vector(result->diagonal);
        e.real(result->acceptance);
        e.real(result->step_size);
        e.integer(result->divergences);
        e.integer(result->completed_iterations);
        e.integer(result->completed_samples);
    }
    return x.bytes == y.bytes;
}
template <class Grid>
void exercise(const FS::path & root,
              const std::string & phase,
              const FML::Domain & domain,
              const std::string & label,
              bool compact,
              bool pswf) {
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    FF::EulerianLPTConfiguration c;
    c.mesh = compact ? 4 : 8;
    c.product_mesh = compact ? 8 : 0;
    c.density_mesh = 16;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 1.7 * kf;
    c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
    c.matter_assignment = pswf ? FF::MatterAssignment::PSWF : FF::MatterAssignment::Smooth;
    auto spectrum = FF::sigma8_spectrum([](double k) { return 500 * std::exp(-5 * k); });
    Model model(c, spectrum, priors(), domain);
    auto p = model.initial_parameters();
    FF::Modes data(model.modes().size(), 0);
    auto post = FF::fixed_noise_posterior(model, data, 1000);
    FML::MCMC::HMCOptions o;
    o.warmup = 6;
    o.samples = 5;
    o.n_leapfrog = 2;
    o.jitter = true;
    o.seed = 901;
    o.step_size = .002;
    o.adapt_mass = false;
    FF::CheckpointOptions io;
    io.directory = (root / label).string();
    io.problem_id = "Eulerian-restart-fixture:P500exp-v1";
    io.every = 2;
    const auto reference = root / (label + ".rank" + std::to_string(FML::ThisTask));
    const auto id = CP::identity(post, io.problem_id);
    if (phase == "prepare") {
        const auto full = FF::sample(post, model.white_noise(902), p, o);
        FML::COMMUNICATION::collective_try([&] { write_reference(reference, full); }, "Eulerian reference write");
        io.max_iterations = 2;
        const auto prefix = FF::sample_checkpointed(post, model.white_noise(902), p, o, io);
        check(prefix.recorded.empty() && prefix.completed_iterations == 2, label + " mid-warmup checkpoint");
        io.resume = true;
        io.max_iterations = o.warmup - 2 + 2;
        const auto part = FF::sample_checkpointed(post, {}, {}, o, io);
        check(identical(part.recorded, std::vector<Field>(full.recorded.begin(), full.recorded.begin() + 2)) &&
                  identical(part.potential, Field(full.potential.begin(), full.potential.begin() + 2)),
              label + " warmup continuation and first draws bitwise exact");
        auto state = CP::load(io.directory, id, model.cells(), p.size(), true);
        check(state.sampler.next_iteration == o.warmup + 2 && state.frozen == p,
              label + " compact field and frozen metric state persisted");
        if (compact && pswf) {
            const auto generation = CP::current_generation(io.directory);
            if (FML::ThisTask == FML::NTasks - 1)
                CP::fail_shard_write_once = true;
            check(FML::TESTS::throws([&] { CP::save(io.directory, state); }),
                  label + " one-rank write failure is collective");
            check(CP::current_generation(io.directory) == generation,
                  label + " failed generation never replaces CURRENT");
            auto retained = CP::load(io.directory, id, model.cells(), p.size(), true);
            check(retained.sampler.position == state.sampler.position,
                  label + " previous complete generation remains readable");
        }
        auto fixed = o;
        fixed.warmup = 3;
        fixed.samples = 3;
        fixed.adapt_step_size = false;
        fixed.jitter = false;
        const auto full_fixed = FF::sample(post, model.white_noise(903), p, fixed, false);
        FML::COMMUNICATION::collective_try([&] { write_reference(reference.string() + ".fixed", full_fixed); },
                                           "Fixed-scalar reference");
        auto fio = io;
        fio.directory += "-field-only";
        fio.resume = false;
        fio.max_iterations = 2;
        FF::sample_checkpointed(post, model.white_noise(903), p, fixed, fio, false);
    } else if (phase == "resume") {
        FML::MCMC::HMCResult full, full_fixed;
        FML::COMMUNICATION::collective_try(
            [&] {
                full = read_reference(reference, model.cells() + p.size(), p.size(), o.samples, true);
                full_fixed = read_reference(reference.string() + ".fixed", model.cells(), p.size(), 3, false);
            },
            "Eulerian reference read");
        io.resume = true;
        std::vector<size_t> indices;
        const auto suffix =
            FF::sample_checkpointed(post, {}, {}, o, io, true, [&](const auto &, const auto &, size_t j) {
                indices.push_back(j);
            });
        check(same(suffix, full) &&
                  identical(suffix.recorded, std::vector<Field>(full.recorded.begin() + 2, full.recorded.end())) &&
                  identical(suffix.potential, Field(full.potential.begin() + 2, full.potential.end())) &&
                  indices == std::vector<size_t>{2, 3, 4},
              label + " fresh-process joint suffix/state/metrics/stats/sink bitwise exact");
        auto fixed = o;
        fixed.warmup = 3;
        fixed.samples = 3;
        fixed.adapt_step_size = false;
        fixed.jitter = false;
        auto fio = io;
        fio.directory += "-field-only";
        const auto field_only = FF::sample_checkpointed(post, {}, {}, fixed, fio, false);
        check(same(field_only, full_fixed) && identical(field_only.recorded, full_fixed.recorded) &&
                  identical(field_only.potential, full_fixed.potential),
              label + " fresh-process field-only continuation bitwise exact");
        io.max_iterations = 0;
        const auto generation = CP::current_generation(io.directory);
        auto changed = io;
        changed.problem_id += "-new-callback";
        check(FML::TESTS::throws([&] { FF::sample_checkpointed(post, {}, {}, o, changed); }),
              label + " callback/problem version change rejected");
        auto seed = o;
        ++seed.seed;
        check(FML::TESTS::throws([&] { FF::sample_checkpointed(post, {}, {}, seed, io); }),
              label + " changed HMC controls rejected");
        auto noise = FF::fixed_noise_posterior(model, data, 1100);
        check(CP::identity(noise, io.problem_id) != id &&
                  FML::TESTS::throws([&] { FF::sample_checkpointed(noise, {}, {}, o, io); }),
              label + " fixed noise change rejected");
        auto conditional = FF::fixed_noise_posterior(model, data, 1000, FF::InitialFieldPrior::ConditionOnField);
        check(CP::identity(conditional, io.problem_id) != id &&
                  FML::TESTS::throws([&] { FF::sample_checkpointed(conditional, {}, {}, o, io); }),
              label + " field prior policy change rejected");
        auto observed = data;
        for (size_t j = 0; j < observed.size(); ++j) {
            const auto & n = model.modes()[j].n;
            if (n[0] == 0 && n[1] == 0 && n[2] == 1)
                observed[j] = .01;
        }
        auto different_data = FF::fixed_noise_posterior(model, observed, 1000);
        check(FML::TESTS::throws([&] { FF::sample_checkpointed(different_data, {}, {}, o, io); }),
              label + " changed observed modes rejected");
        if (compact && pswf) {
            for (int kind = 0; kind < 10; ++kind) {
                auto altered = c;
                auto bias = priors();
                switch (kind) {
                    case 0:
                        altered.mesh = 8;
                        break;
                    case 1:
                        altered.product_mesh = 12;
                        break;
                    case 2:
                        altered.density_mesh = 24;
                        break;
                    case 3:
                        altered.nufft_epsilon = 1e-10;
                        break;
                    case 4:
                        altered.matter_assignment = FF::MatterAssignment::Smooth;
                        break;
                    case 5:
                        altered.matter_band.geometry = FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere;
                        break;
                    case 6:
                        altered.likelihood_band.geometry = FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere;
                        break;
                    case 7:
                        altered.evolution = FF::Evolution::Zeldovich;
                        break;
                    case 8:
                        altered.initial_cutoff *= .99;
                        break;
                    case 9:
                        bias[1].lower = -3.;
                        break;
                }
                Model other(altered, spectrum, bias, domain);
                auto target = FF::fixed_noise_posterior(other, FF::Modes(other.modes().size(), 0), 1000);
                check(CP::identity(target, io.problem_id) != id &&
                          FML::TESTS::throws([&] { FF::sample_checkpointed(target, {}, {}, o, io); }),
                      label + " changed numerical target rejected " + std::to_string(kind));
            }
            const auto corrupt = FS::path(io.directory + "-corrupt");
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == 0)
                        FS::copy(io.directory, corrupt, FS::copy_options::recursive);
                },
                "Copy corruption fixture");
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == FML::NTasks - 1) {
                        std::fstream file(CP::shard(corrupt, generation),
                                          std::ios::in | std::ios::out | std::ios::binary);
                        if (!file)
                            throw std::runtime_error("Missing corruption fixture");
                        file.seekg(-1, std::ios::end);
                        char byte;
                        file.get(byte);
                        file.seekp(-1, std::ios::end);
                        file.put(byte ^ 1);
                        if (!file)
                            throw std::runtime_error("Corruption fixture write failed");
                    }
                },
                "Corrupt one shard");
            auto bad = io;
            bad.directory = corrupt.string();
            check(FML::TESTS::throws([&] { FF::sample_checkpointed(post, {}, {}, o, bad); }),
                  label + " one corrupt rank shard rejected collectively");
        }
        check(CP::current_generation(io.directory) == generation,
              label + " rejected restarts leave checkpoint untouched");
    } else
        throw std::invalid_argument("Expected prepare/resume");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 3)
            throw std::invalid_argument("ROOT prepare|resume required");
        const FS::path root = argv[1];
        const std::string phase = argv[2];
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0)
                    FS::create_directories(root);
            },
            "Checkpoint test root");
        for (bool compact : {false, true})
            for (bool pswf : {false, true}) {
                const auto label = std::string(compact ? "compact" : "shared") + (pswf ? "-PSWF" : "-smooth");
                exercise<FML::GRID::FFTWGrid<3>>(root,
                                                 phase,
                                                 FML::Domain::uniform_slab(FML::NTasks),
                                                 "slab-" + label,
                                                 compact,
                                                 pswf);
#ifdef USE_MPI
                const int px = FML::NTasks % 2 ? 1 : 2, py = FML::NTasks / px;
                std::vector<double> ex(px + 1), ey(py + 1);
                for (int j = 0; j <= px; ++j)
                    ex[j] = double(j) / px;
                for (int j = 0; j <= py; ++j)
                    ey[j] = double(j) / py;
                exercise<FML::GRID::PencilGrid<3>>(root,
                                                   phase,
                                                   FML::Domain::pencil(ex, ey),
                                                   "pencil-" + label,
                                                   compact,
                                                   pswf);
#endif
            }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return FML::TESTS::summary("Eulerian checkpoint");
}
