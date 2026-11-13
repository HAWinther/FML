#include <FML/MCMC/HMCCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#include <filesystem>
#include <fstream>
#include <chrono>
using namespace FML::MCMC;
using FML::TESTS::check_local;
int main() {
    namespace C = FML::MCMC::CHECKPOINT;
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("fml-hmc-checkpoint-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    auto potential = [](const auto & x, auto & g) {
        double U = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const double scale = .5 + i * .2;
            g[i] = (x[i] - .1 * i) / scale;
            U += .5 * (x[i] - .1 * i) * g[i];
        }
        return U;
    };
    HMCMass mass;
    mass.n_custom = 2;
    mass.apply_sqrt = [](auto &) {};
    mass.apply_inverse = [](auto &) {};
    HMCOptions opt;
    opt.warmup = 120;
    opt.samples = 30;
    opt.n_leapfrog = 5;
    opt.target_acceptance = .83;
    opt.seed = 982;
    opt.lower = {-.7, -.7, -.7};
    opt.upper = {.7, .7, .7};
    std::vector<double> initial(5, 0);
    const auto full = run_hmc(potential, initial, mass, opt);
    HMCState state;
    HMCResult last;
    std::vector<std::vector<double>> rows;
    std::vector<double> values;
    int callbacks = 0;
    for (int budget : {0, 27, 33, 55, 9, 26}) {
        HMCRunControl run;
        run.state = &state;
        run.max_iterations = budget;
        run.checkpoint_every = 7;
        run.checkpoint = [&](const auto & s) {
            check_local(s.next_iteration % 7 == 0, "checkpoint callback at trajectory boundary");
            ++callbacks;
        };
        last = run_hmc(potential, state.initialized ? state.position : initial, mass, opt, nullptr, {}, run);
        rows.insert(rows.end(), last.recorded.begin(), last.recorded.end());
        values.insert(values.end(), last.potential.begin(), last.potential.end());
        C::save((directory / "chain.hmc").string(), state);
        auto loaded = C::load((directory / "chain.hmc").string(), 5, 2);
        check_local(C::stream_text(loaded.rng) == C::stream_text(state.rng) &&
                        C::stream_text(loaded.gauss) == C::stream_text(state.gauss),
                    "RNG engine and cached normal variate survive disk round trip");
        state = std::move(loaded);
    }
    check_local(rows == full.recorded && values == full.potential && last.last == full.last,
                "chunked disk-resumed Gaussian chain is bitwise equal to uninterrupted chain");
    check_local(last.diagonal == full.diagonal && last.step_size == full.step_size &&
                    last.acceptance == full.acceptance && last.divergences == full.divergences,
                "adaptation, acceptance and divergence totals resume exactly");
    check_local(callbacks > 0 && state.next_iteration == 150 && state.window == 2,
                "checkpoints cover warmup windows and sampling phase");
    auto altered = opt;
    altered.n_leapfrog++;
    HMCRunControl resume;
    resume.state = &state;
    check_local(FML::TESTS::throws([&] { run_hmc(potential, state.position, mass, altered, nullptr, {}, resume); }),
                "changed restart controls rejected");
    auto changed = [&](const auto & x, auto & g) { return potential(x, g) + 1; };
    check_local(FML::TESTS::throws([&] { run_hmc(changed, state.position, mass, opt, nullptr, {}, resume); }),
                "changed restart potential rejected");
    auto bad = state;
    bad.gradient[0] += 1;
    resume.state = &bad;
    check_local(FML::TESTS::throws([&] { run_hmc(potential, bad.position, mass, opt, nullptr, {}, resume); }),
                "changed restart force rejected");
    // Corruption, truncation and inconsistent dimensions fail before decoding allocations.
    auto bytes = C::read_file((directory / "chain.hmc").string(), C::size_bound(5, 3));
    bytes[8] = '!';
    C::write_file((directory / "wrong.hmc").string(), bytes);
    check_local(FML::TESTS::throws([&] { C::load((directory / "wrong.hmc").string(), 5, 2); }),
                "incompatible build tag rejected");
    {
        std::fstream file(directory / "chain.hmc", std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(-1, std::ios::end);
        char c;
        file.read(&c, 1);
        c ^= 1;
        file.seekp(-1, std::ios::end);
        file.write(&c, 1);
    }
    check_local(FML::TESTS::throws([&] { C::load((directory / "chain.hmc").string(), 5, 2); }),
                "checksum corruption rejected");
    C::Encoder malformed;
    malformed.text(C::build_tag());
    malformed.integer(1);
    malformed.integer(2);
    C::options(malformed, state.options);
    malformed.integer(std::numeric_limits<std::uint64_t>::max());
    C::write_file((directory / "oversized.hmc").string(), malformed.bytes);
    check_local(FML::TESTS::throws([&] { C::load((directory / "oversized.hmc").string(), 5, 2); }),
                "oversized vector count rejected before allocation");
    std::filesystem::resize_file(directory / "chain.hmc", 10);
    check_local(FML::TESTS::throws([&] { C::load((directory / "chain.hmc").string(), 5, 2); }),
                "truncated checkpoint rejected");
    check_local(FML::TESTS::throws([&] { C::load((directory / "wrong.hmc").string(), 1, 2); }),
                "invalid expected custom dimension rejected");
    auto silent = opt;
    silent.warmup = 0;
    silent.samples = 3;
    silent.verbose_every = -1;
    silent.divergence_threshold = 0;
    HMCState compatibility;
    HMCRunControl short_run;
    short_run.state = &compatibility;
    short_run.max_iterations = 1;
    const auto silent_full = run_hmc(potential, initial, mass, silent);
    run_hmc(potential, initial, mass, silent, nullptr, {}, short_run);
    C::save((directory / "silent.hmc").string(), compatibility);
    compatibility = C::load((directory / "silent.hmc").string(), 5, 2);
    short_run.max_iterations = INT_MAX;
    const auto silent_tail = run_hmc(potential, compatibility.position, mass, silent, nullptr, {}, short_run);
    check_local(silent_tail.last == silent_full.last,
                "zero divergence threshold and legacy silent logging remain restartable");
    std::filesystem::remove_all(directory);
    return FML::TESTS::summary("HMC checkpoint");
}
