#ifndef FML_FIELD_LEVEL_CHECKPOINT_H
#define FML_FIELD_LEVEL_CHECKPOINT_H
#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/MCMC/HMCCheckpoint.h>
#include <filesystem>
#include <chrono>
#include <typeinfo>
#include <type_traits>

namespace FML::FIELDLEVEL::FRAMEWORK {
    struct CheckpointOptions {
        std::string directory;
        // Caller must version callback physics, code and any external target inputs.
        std::string problem_id;
        bool resume = false;
        int every = 0;                // Trajectories including warmup; final boundary always saved.
        int max_iterations = INT_MAX; // Per-call budget, not a new warmup/sample schedule.
    };
    namespace CHECKPOINT {
        namespace Codec = FML::MCMC::CHECKPOINT;
        namespace FS = std::filesystem;
        struct State {
            FML::MCMC::HMCState sampler;
            std::vector<double> frozen;
            bool vary = true;
            std::string identity;
        };
#ifdef FML_FIELDLEVEL_CHECKPOINT_TEST_FAILURE
        inline bool fail_shard_write_once = false;
#endif
        inline void same_string(const std::string & value) {
            std::string root = value;
#ifdef USE_MPI
            std::uint64_t n = FML::ThisTask == 0 ? root.size() : 0;
            MPI_Bcast(&n, 1, MPI_UINT64_T, 0, FML::communicator);
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (n > 16384)
                        throw std::runtime_error("[Field checkpoint] Oversized replicated metadata");
                    root.resize(size_t(n));
                },
                "Checkpoint replicated metadata");
            MPI_Bcast(root.data(), int(n), MPI_CHAR, 0, FML::communicator);
#endif
            FML::COMMUNICATION::require_count(value == root, "[Field checkpoint] Inconsistent replicated metadata");
        }
        inline std::string broadcast_string(std::string value) {
#ifdef USE_MPI
            std::uint64_t n = FML::ThisTask == 0 ? value.size() : 0;
            MPI_Bcast(&n, 1, MPI_UINT64_T, 0, FML::communicator);
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (n > 256)
                        throw std::runtime_error("[Field checkpoint] Oversized generation name");
                    value.resize(size_t(n));
                },
                "Checkpoint generation name");
            MPI_Bcast(value.data(), int(n), MPI_CHAR, 0, FML::communicator);
#endif
            return value;
        }
        // Unsupported likelihoods fail at compile time: never silently fingerprint
        // a custom target as the periodic Gaussian. Specializations append all
        // fixed target inputs and a versioned likelihood-law tag.
        template <class Likelihood>
        struct LikelihoodIdentity {
            static void append(Codec::Encoder &, const Likelihood &) {
                static_assert(!std::is_same_v<Likelihood, Likelihood>,
                              "Checkpoint likelihood requires a LikelihoodIdentity specialization; "
                              "include SurveyCheckpoint.h for built-in survey likelihoods");
            }
        };
        template <>
        struct LikelihoodIdentity<GaussianFieldLikelihood> {
            static void append(Codec::Encoder & e, const GaussianFieldLikelihood & likelihood) {
                e.text("FML Fourier Gaussian likelihood 1");
                e.real(likelihood.volume());
                e.vector(likelihood.weights());
                e.integer(likelihood.data().size());
                for (auto z : likelihood.data()) {
                    e.real(z.real());
                    e.real(z.imag());
                }
            }
        };
        template <class Forward, class Likelihood>
        std::string identity(const BasicPosterior<Forward, Likelihood> & posterior, const std::string & problem_id) {
            Codec::Encoder e;
            const auto & m = posterior.model();
            const auto & c = m.configuration();
            const auto l = m.layout();
            e.text("FML field target 2");
            e.text(typeid(Likelihood).name());
            e.text(Codec::build_tag());
            e.text(problem_id);
            e.text(typeid(Forward).name());
            for (int x : {FML::NTasks,
                          FML::ThisTask,
                          FML::NThreads,
                          int(sizeof(FML::GRID::FloatType)),
                          c.mesh,
                          c.density_mesh,
                          c.kernel_width,
                          int(c.evolution),
                          l.local_start,
                          l.local_width})
                e.integer(x);
            for (double x : {c.boxsize, c.initial_cutoff, c.bias_cutoff, c.likelihood_cutoff, c.growth_rate})
                e.real(x);
            e.integer(l.global_cells);
            e.integer(l.owned_cells);
            e.integer(m.parameters().size());
            for (const auto & p : m.parameters()) {
                e.text(p.name);
                for (double x : {p.initial, p.mean, p.sigma, p.lower, p.upper})
                    e.real(x);
                e.integer(int(p.prior));
            }
            e.integer(m.modes().size());
            for (const auto & mode : m.modes()) {
                for (int x : mode.n)
                    e.integer(std::uint64_t(std::int64_t(x)));
                e.real(mode.k);
                e.real(mode.weight);
            }
            e.vector(m.response());
            LikelihoodIdentity<Likelihood>::append(e, posterior.likelihood());
            return std::to_string(Codec::checksum(e.bytes));
        }
        inline void replicas(const State & state) {
            Codec::Encoder e;
            FML::COMMUNICATION::collective_try(
                [&] {
                    e.integer(state.vary);
                    e.vector(state.frozen);
                    Codec::state(e, state.sampler, false);
                },
                "Checkpoint replicated sampler state");
            same_string(std::to_string(Codec::checksum(e.bytes)));
        }
        inline FS::path shard(const FS::path & dir, const std::string & gen) {
            return dir / gen / ("rank-" + std::to_string(FML::ThisTask) + ".hmc");
        }
        inline std::string current_generation(const FS::path & dir) {
            std::string gen;
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == 0) {
                        const auto bytes = Codec::read_file((dir / "CURRENT").string(), 4096);
                        Codec::Decoder d{bytes};
                        if (d.text(64) != "FML field manifest 1" || d.size(INT_MAX) != size_t(FML::NTasks))
                            throw std::runtime_error("[Field checkpoint] Incompatible manifest/rank count");
                        gen = d.text(128);
                        d.end();
                        if (gen.rfind("g-", 0) != 0 || gen.find_first_not_of("g-0123456789") != std::string::npos)
                            throw std::runtime_error("[Field checkpoint] Invalid generation name");
                    }
                },
                "Checkpoint manifest read");
            return broadcast_string(gen);
        }
        inline void save(const FS::path & directory, const State & state) {
            replicas(state);
            std::string generation;
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == 0) {
                        FS::create_directories(directory);
                        const auto tick =
                            std::uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
                        for (unsigned attempt = 0; attempt < 1000; ++attempt) {
                            auto candidate = "g-" + std::to_string(tick) + "-" + std::to_string(attempt);
                            if (FS::create_directory(directory / candidate)) {
                                generation = candidate;
                                break;
                            }
                        }
                        if (generation.empty())
                            throw std::runtime_error("[Field checkpoint] Cannot create unique generation");
                    }
                },
                "Checkpoint generation creation");
            generation = broadcast_string(generation);
            FML::COMMUNICATION::collective_try(
                [&] {
#ifdef FML_FIELDLEVEL_CHECKPOINT_TEST_FAILURE
                    if (fail_shard_write_once) {
                        fail_shard_write_once = false;
                        throw std::runtime_error("Injected checkpoint write failure");
                    }
#endif
                    Codec::Encoder e;
                    e.text("FML field shard 1");
                    e.text(Codec::build_tag());
                    e.integer(FML::NTasks);
                    e.integer(FML::ThisTask);
                    e.text(generation);
                    e.text(state.identity);
                    e.integer(state.vary);
                    e.vector(state.frozen);
                    Codec::state(e, state.sampler);
                    const auto file = shard(directory, generation);
                    Codec::write_file(file.string() + ".tmp", e.bytes);
                    FS::rename(file.string() + ".tmp", file);
                },
                "Checkpoint shard write");
            // Publish only after every rank closed and renamed its complete shard.
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask == 0) {
                        Codec::Encoder e;
                        e.text("FML field manifest 1");
                        e.integer(FML::NTasks);
                        e.text(generation);
                        const auto temporary = directory / ("CURRENT-" + generation + ".tmp");
                        Codec::write_file(temporary.string(), e.bytes);
                        FS::rename(temporary, directory / "CURRENT");
                    }
                },
                "Checkpoint manifest publication");
        }
        inline State load(const FS::path & directory,
                          const std::string & expected_identity,
                          size_t cells,
                          size_t parameters,
                          bool vary) {
            const auto generation = current_generation(directory);
            State state;
            FML::COMMUNICATION::collective_try(
                [&] {
                    const size_t n = cells + (vary ? parameters : 0);
                    const auto bytes =
                        Codec::read_file(shard(directory, generation).string(), Codec::size_bound(n, parameters));
                    Codec::Decoder d{bytes};
                    if (d.text(64) != "FML field shard 1" || d.text() != Codec::build_tag() ||
                        d.size(INT_MAX) != size_t(FML::NTasks) || d.size(INT_MAX) != size_t(FML::ThisTask) ||
                        d.text(128) != generation)
                        throw std::runtime_error("[Field checkpoint] Incompatible shard/build/layout");
                    state.identity = d.text(128);
                    state.vary = d.boolean();
                    if (state.identity != expected_identity || state.vary != vary)
                        throw std::runtime_error("[Field checkpoint] Target/configuration/sampling mode changed");
                    state.frozen = d.vector(parameters);
                    state.sampler = Codec::state(d, n, cells);
                    d.end();
                },
                "Checkpoint shard read");
            replicas(state);
            return state;
        }
    } // namespace CHECKPOINT
    // Same rank count/layout/build/thread configuration; explicitly fingerprinted likelihood.
    // Returned rows/potentials are this invocation's suffix. Acceptance/divergences
    // are cumulative over all completed sampling trajectories, including previous calls.
    template <class Forward, class Likelihood>
    inline FML::MCMC::HMCResult sample_checkpointed(const BasicPosterior<Forward, Likelihood> & posterior,
                                                    Field white,
                                                    std::vector<double> parameters,
                                                    FML::MCMC::HMCOptions options,
                                                    CheckpointOptions io,
                                                    bool vary_parameters = true,
                                                    FieldSink sink = {}) {
        using namespace CHECKPOINT;
        const auto & model = posterior.model();
        FS::path directory;
        std::string target;
        FML::COMMUNICATION::collective_try(
            [&] {
                if (io.directory.empty() || io.problem_id.empty() || io.problem_id.size() > 4096 || io.every < 0 ||
                    io.max_iterations < 0)
                    throw std::invalid_argument(
                        "[Field checkpoint] Directory, versioned problem ID and valid controls required");
                directory = FS::absolute(io.directory).lexically_normal();
                target = identity(posterior, io.problem_id);
            },
            "Checkpoint target description");
        same_string(directory.string());
        same_string(io.problem_id);
        same_string(std::to_string(io.resume) + ":" + std::to_string(io.every) + ":" +
                    std::to_string(io.max_iterations) + ":" + std::to_string(vary_parameters));
        Execution<Forward>::hmc_controls(model, vary_parameters, options);
        State state;
        if (io.resume) {
            FML::COMMUNICATION::require_count(white.empty() && parameters.empty(),
                                              "[Field checkpoint] Resume takes empty initial field/parameters");
            state = load(directory, target, model.cells(), model.parameters().size(), vary_parameters);
            const size_t n = model.cells();
            FML::COMMUNICATION::collective_try(
                [&] {
                    white.assign(state.sampler.position.begin(), state.sampler.position.begin() + n);
                    parameters = vary_parameters ? std::vector<double>(state.sampler.position.begin() + n,
                                                                       state.sampler.position.end()) :
                                                   state.frozen;
                },
                "Checkpoint local state restoration");
        } else {
            state.identity = target;
            state.vary = vary_parameters;
            state.frozen = parameters;
        }
        Execution<Forward>::validate(model, white, parameters);
        if (!model.in_support(parameters) || !model.in_support(state.frozen))
            throw std::invalid_argument("[Field checkpoint] Initial/frozen parameters outside support");
        FML::MCMC::HMCRunControl control;
        control.state = &state.sampler;
        control.max_iterations = io.max_iterations;
        control.checkpoint_every = io.every;
        int last_saved = -1;
        control.checkpoint = [&](const auto &) {
            save(directory, state);
            last_saved = state.sampler.next_iteration;
        };
        auto result = DETAIL::sample_run(posterior,
                                         std::move(white),
                                         std::move(parameters),
                                         options,
                                         vary_parameters,
                                         std::move(sink),
                                         state.frozen,
                                         control);
        if (last_saved != state.sampler.next_iteration)
            save(directory, state);
        return result;
    }
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
