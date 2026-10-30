#include "BuildInfo.h"
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/Global/SystemMemory.h>
#include <FML/Performance/Counters.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <thread>
using Grid = FML::GRID::FFTWGrid<3>;
using Complex = FML::GRID::ComplexType;
using Count = FML::PERFORMANCE::Count;
using Counter = FML::PERFORMANCE::Counter;
namespace Perf = FML::PERFORMANCE;
struct Particle {
    double pos[3]{};
    Count id = 0;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
std::string quote(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << c;
        else if (c < 32)
            out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c) << std::dec;
        else
            out << c;
    }
    out << '"';
    return out.str();
}
void barrier() {
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
}
struct Options {
    std::string name, output;
    int mesh = 32, repeat = 5, warmup = 1, shells = 3;
    size_t message_bytes = 1024 * 1024;
    Count seed = 12345;
};
Count integer(const std::string & value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected unsigned integer");
    size_t end;
    auto result = std::stoull(value, &end);
    return result;
}
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc)
            throw std::invalid_argument("each option needs a value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--case")
            o.name = value;
        else if (key == "--output")
            o.output = value;
        else {
            auto n = integer(value);
            if (key == "--seed") {
                o.seed = n;
                continue;
            }
            if (n > Count(INT_MAX))
                throw std::invalid_argument("numeric option exceeds INT_MAX");
            if (key == "--mesh")
                o.mesh = int(n);
            else if (key == "--repeat")
                o.repeat = int(n);
            else if (key == "--warmup")
                o.warmup = int(n);
            else if (key == "--cache-shells")
                o.shells = int(n);
            else if (key == "--message-bytes")
                o.message_bytes = size_t(n);
            else
                throw std::invalid_argument("unknown option: " + key);
        }
    }
    const std::array<std::string, 6> cases{"fft_roundtrip",
                                           "grid_copy",
                                           "grid_clone",
                                           "resample",
                                           "migration",
                                           "polyspectrum"};
    if (std::find(cases.begin(), cases.end(), o.name) == cases.end())
        throw std::invalid_argument("unknown case: " + o.name);
    if (o.mesh < 8 || o.mesh > INT_MAX / 4 || o.mesh % FML::NTasks || o.repeat < 1 || o.repeat > 10000 ||
        o.warmup > 10000 || o.output.empty() || (o.shells && o.shells < 3) || o.message_bytes < sizeof(Complex))
        throw std::invalid_argument("invalid mesh/rank divisibility, repeat/warmup, cache size, output or message cap");
    return o;
}
std::array<int, 3> wave(const Grid & grid, ptrdiff_t index) {
    auto k = grid.get_fourier_coord_from_index(index);
    k[0] += grid.get_local_x_start();
    for (int d = 0; d < 2; ++d)
        if (k[d] > grid.get_nmesh() / 2)
            k[d] -= grid.get_nmesh();
    return k;
}
Complex coefficient(std::array<int, 3> k, Count seed) {
    Count code = seed % 127;
    for (int x : k)
        code = (code * 31 + Count(static_cast<long long>(x) * x)) % 127;
    return Complex(FML::GRID::FloatType(code + 1), 0);
}
struct Case {
    std::function<void()> prepare = [] {};
    std::function<void()> work;
    std::function<bool(const Perf::Snapshot &)> validate;
    std::function<Count()> shell_builds = [] { return Count(0); };
    int work_mesh = 0;
};
// Captured shared owners keep work/validation storage alive; no timed object-setup is hidden in the harness.
Case setup(const Options & o) {
    const double epsilon = std::numeric_limits<FML::GRID::FloatType>::epsilon();
    if (o.name == "fft_roundtrip") {
        auto field = std::make_shared<Grid>(o.mesh);
        field->fill_real_grid([&](std::array<double, 3> & x) {
            return std::cos(2 * M_PI * x[0]) + .25 * std::sin(4 * M_PI * x[1]) + double(o.seed % 17) / 32;
        });
        return {[] {},
                [field] {
                    field->fftw_r2c();
                    field->fftw_c2r();
                },
                [field, o, epsilon](const auto & s) {
                    double error = 0;
                    bool finite = true;
                    for (auto i : field->get_real_range()) {
                        auto x = field->get_real_position(field->get_coord_from_index(i));
                        double expected =
                            std::cos(2 * M_PI * x[0]) + .25 * std::sin(4 * M_PI * x[1]) + double(o.seed % 17) / 32;
                        double actual = double(field->get_real_from_index(i));
                        finite &= std::isfinite(actual);
                        error = std::max(error, std::abs(actual - expected));
                    }
                    return finite && error < 128 * epsilon * (o.repeat + o.warmup + 1) &&
                           s[Counter::ForwardFFTs] == Count(o.repeat) && s[Counter::InverseFFTs] == Count(o.repeat);
                }};
    }
    if (o.name == "grid_copy" || o.name == "grid_clone") {
        auto source = std::make_shared<Grid>(o.mesh), target = std::make_shared<Grid>();
        source->set_grid_status_real(false);
        for (auto i : source->get_fourier_range())
            source->set_fourier_from_index(i, coefficient(wave(*source, i), o.seed));
        bool copy = o.name == "grid_copy";
        return {[] {},
                [source, target, copy] { *target = copy ? Grid(*source) : source->clone_empty(); },
                [source, target, o, copy](const auto & s) {
                    bool good = s[Counter::GridAllocations] == Count(o.repeat);
                    Count bytes = Count(source->get_ntot_fourier_alloc()) * sizeof(Complex) * o.repeat;
                    good &= s[Counter::GridCopiedBytes] == (copy ? bytes : 0);
                    for (auto i : target->get_fourier_range())
                        good &= target->get_fourier_from_index(i) ==
                                (copy ? source->get_fourier_from_index(i) : Complex(0));
                    return good;
                }};
    }
    if (o.name == "resample") {
        auto source = std::make_shared<Grid>(o.mesh), target = std::make_shared<Grid>(2 * o.mesh);
        source->set_grid_status_real(false);
        for (auto i : source->get_fourier_range())
            source->set_fourier_from_index(i, coefficient(wave(*source, i), o.seed));
        auto plan = std::make_shared<FML::GRID::FourierResamplingPlan<3>>(*source, *target);
        std::vector<Count> starts(FML::NTasks), widths(FML::NTasks);
        Count start = target->get_local_x_start(), width = target->get_local_nx();
        starts = FML::GatherFromTasks(&start);
        widths = FML::GatherFromTasks(&width);
        Count remote = 0;
        for (auto i : source->get_fourier_range()) {
            auto k = wave(*source, i);
            bool keep = true;
            for (int x : k)
                keep &= 2LL * std::abs(x) < o.mesh;
            if (!keep)
                continue;
            int x = k[0] < 0 ? k[0] + 2 * o.mesh : k[0], owner = -1;
            for (int t = 0; t < FML::NTasks; ++t)
                if (Count(x) >= starts[t] && Count(x) < starts[t] + widths[t])
                    owner = t;
            if (owner < 0)
                throw std::logic_error("resample oracle has no owner");
            if (owner != FML::ThisTask)
                ++remote;
        }
        return {[] {},
                [source, target, plan, o] { plan->apply(*source, *target, o.message_bytes); },
                [target, o, remote](const auto & s) {
                    bool good = s[Counter::PayloadSentBytes] == remote * sizeof(Complex) * o.repeat;
                    good &= s[Counter::TransportPackedBytes] == s[Counter::PayloadSentBytes] &&
                            s[Counter::TransportUnpackedBytes] == s[Counter::PayloadReceivedBytes];
                    for (auto i : target->get_fourier_range()) {
                        auto k = wave(*target, i);
                        bool keep = true;
                        for (int x : k)
                            keep &= 2LL * std::abs(x) < o.mesh;
                        good &= target->get_fourier_from_index(i) == (keep ? coefficient(k, o.seed) : Complex(0));
                    }
                    return good;
                }};
    }
    if (o.name == "migration") {
        const Count total = Count(o.mesh) * o.mesh * o.mesh, local = total / FML::NTasks;
        std::vector<Particle> initial(size_t(local), Particle{});
        for (Count i = 0; i < local; ++i)
            initial[size_t(i)] = {{(FML::ThisTask + .5) / FML::NTasks, .25, .75}, Count(FML::ThisTask) * local + i};
        auto particles = std::make_shared<FML::PARTICLE::MPIParticles<Particle>>();
        particles->create(initial.data(), initial.size(), size_t(local + 3), FML::xmin_domain, FML::xmax_domain, false);
        return {[particles] {
                    for (auto & p : *particles) {
                        int step = p.id % 3 == 1 ? -1 : p.id % 3 == 2 ? 1 : 0;
                        int dest = (FML::ThisTask + step + FML::NTasks) % FML::NTasks;
                        p.pos[0] = (dest + .5) / FML::NTasks;
                    }
                },
                [particles, o] { particles->communicate_particles(o.message_bytes); },
                [particles, o, local](const auto & s) {
                    bool good = s[Counter::TransportPackedBytes] == s[Counter::PayloadSentBytes] &&
                                s[Counter::TransportUnpackedBytes] == s[Counter::PayloadReceivedBytes];
                    std::vector<Count> ids, expected;
                    const int steps = (o.warmup + o.repeat) % FML::NTasks;
                    for (auto & p : *particles) {
                        ids.push_back(p.id);
                        good &= p.pos[0] == (FML::ThisTask + .5) / FML::NTasks && p.pos[1] == .25 && p.pos[2] == .75;
                    }
                    for (int c = 0; c < 3; ++c) {
                        int sign = c == 1 ? -1 : c == 2 ? 1 : 0;
                        int source = (FML::ThisTask - sign * steps + FML::NTasks) % FML::NTasks;
                        for (Count id = Count(source) * local; id < Count(source + 1) * local; ++id)
                            if (id % 3 == Count(c))
                                expected.push_back(id);
                    }
                    std::sort(ids.begin(), ids.end());
                    std::sort(expected.begin(), expected.end());
                    good &= ids == expected;
                    return good;
                }};
    }
    if (o.name == "polyspectrum") {
        using Bins = FML::CORRELATIONFUNCTIONS::SparseBispectrumBinning<3>;
        auto input = std::make_shared<Grid>(o.mesh);
        input->fill_real_grid([&](std::array<double, 3> & x) {
            Count h = o.seed;
            for (double v : x)
                h = (h ^ Count(std::llround(v * o.mesh))) * 1099511628211ULL;
            double u = double(h % 1024) / 1024 - .5;
            return u + .2 * u * u;
        });
        input->fftw_r2c();
        auto bins = std::make_shared<Bins>(6, o.mesh);
        bins->set_configurations({{1, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 4}});
        bins->options.max_cached_shells = o.shells;
        auto builds = std::make_shared<Count>(0);
        return {[] {},
                [input, bins, builds] {
                    FML::CORRELATIONFUNCTIONS::compute_bispectrum(*input, *bins);
                    *builds += bins->statistics.geometry_shell_builds + bins->statistics.field_shell_builds;
                },
                [input, bins, o, epsilon](const auto & s) {
                    Bins full(6, o.mesh);
                    full.set_configurations({{1, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 4}});
                    FML::CORRELATIONFUNCTIONS::compute_bispectrum(*input, full);
                    bool good = true;
                    for (size_t i = 0; i < full.P123.size(); ++i) {
                        good &= std::abs(full.P123[i] - bins->P123[i]) <=
                                4096 * epsilon * std::max(1., std::abs(full.P123[i]));
                        good &= std::abs(full.N123[i] - bins->N123[i]) <=
                                4096 * epsilon * std::max(1., std::abs(full.N123[i]));
                    }
                    return good && s[Counter::ForwardFFTs] == 0;
                },
                [builds] { return *builds; },
                FML::CORRELATIONFUNCTIONS::detail::polyspectrum_work_mesh<3>(o.mesh, {bins->khigh[4]}, bins->dealias)};
    }
    throw std::invalid_argument("unknown case: " + o.name);
}
int main(int argc, char ** argv) {
    try {
        if (!Perf::enabled)
            throw std::invalid_argument("benchmark requires FML_ENABLE_PERFORMANCE_COUNTERS");
        const auto options = parse(argc, argv);
        // Check flattening/address arithmetic before constructing cases.
        FML::COMMUNICATION::Count cells = options.mesh;
        bool valid =
            FML::COMMUNICATION::multiply_count(cells, options.mesh, cells) &&
            FML::COMMUNICATION::multiply_count(cells, options.mesh, cells, size_t(PTRDIFF_MAX) / sizeof(Particle));
        FML::COMMUNICATION::require_count(valid, "benchmark mesh product exceeds supported storage");
        std::ofstream file;
        if (!FML::ThisTask)
            file.open(options.output);
        FML::COMMUNICATION::require_count(FML::ThisTask || bool(file), "cannot open benchmark output");
#ifdef USE_MPI
        constexpr bool mpi_enabled = true;
        char mpi_text[MPI_MAX_LIBRARY_VERSION_STRING]{};
        int mpi_length = 0;
        MPI_Get_library_version(mpi_text, &mpi_length);
        const std::string mpi_version(mpi_text, size_t(mpi_length));
#else
        constexpr bool mpi_enabled = false;
        const std::string mpi_version = "disabled";
#endif
#ifdef USE_OMP
        constexpr bool omp_enabled = true;
#else
        constexpr bool omp_enabled = false;
#endif
#ifdef SINGLE_PRECISION_FFTW
        const std::string fftw_library = fftwf_version;
#elif defined(LONG_DOUBLE_PRECISION_FFTW)
        const std::string fftw_library = fftwl_version;
#else
        const std::string fftw_library = fftw_version;
#endif
        std::vector<std::string> rank_hosts(FML::NTasks);
#ifdef USE_MPI
        std::array<char, MPI_MAX_PROCESSOR_NAME> host{};
        int host_length = 0;
        MPI_Get_processor_name(host.data(), &host_length);
        std::vector<std::array<char, MPI_MAX_PROCESSOR_NAME>> hosts(FML::NTasks);
        MPI_Allgather(host.data(),
                      MPI_MAX_PROCESSOR_NAME,
                      MPI_CHAR,
                      hosts.data(),
                      MPI_MAX_PROCESSOR_NAME,
                      MPI_CHAR,
                      FML::communicator);
        for (int rank = 0; rank < FML::NTasks; ++rank)
            rank_hosts[rank] = hosts[rank].data();
#else
        std::array<char, 256> host{};
        gethostname(host.data(), host.size() - 1);
        rank_hosts[0] = host.data();
#endif
        auto bench = setup(options);
        for (int i = 0; i < options.warmup; ++i) {
            bench.prepare();
            bench.work();
        }
        const Count warm_builds = bench.shell_builds();
        Perf::reset();
        std::vector<double> times;
        for (int i = 0; i < options.repeat; ++i) {
            bench.prepare();
            barrier();
            auto start = std::chrono::steady_clock::now();
            bench.work();
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            FML::MaxOverTasks(&elapsed);
            times.push_back(elapsed);
        }
        auto counters = Perf::snapshot();
        Count peak = getPeakRSS(), peak_sum = peak;
        int rss_known = peak != 0;
        FML::MaxOverTasks(&peak);
        FML::COMMUNICATION::require_count(peak <= std::numeric_limits<Count>::max() / Count(FML::NTasks),
                                          "RSS rank-sum upper bound exceeds supported range");
        FML::SumOverTasks(&peak_sum);
        FML::MinOverTasks(&rss_known);
        Count builds = bench.shell_builds() - warm_builds;
        bool good = !counters.overflow;
        if (options.name == "polyspectrum")
            good &= counters[Counter::InverseFFTs] == builds;
        good &= bench.validate(counters);
        FML::COMMUNICATION::require_count(good, "benchmark correctness/counter check failed");
        auto sums = counters.values, maxima = counters.values;
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE,
                      maxima.data(),
                      int(maxima.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX,
                      FML::communicator);
        bool sums_fit = true;
        for (auto value : maxima)
            sums_fit &= value <= std::numeric_limits<Count>::max() / Count(FML::NTasks);
        FML::COMMUNICATION::require_count(sums_fit, "rank-summed counter upper bound exceeds supported range");
        MPI_Allreduce(MPI_IN_PLACE, sums.data(), int(sums.size()), MPI_UNSIGNED_LONG_LONG, MPI_SUM, FML::communicator);
#endif
        FML::COMMUNICATION::require_count(sums[size_t(Counter::PayloadSentBytes)] ==
                                              sums[size_t(Counter::PayloadReceivedBytes)],
                                          "payload conservation failed");
        if (options.name == "migration") {
            Count moving = cells - (cells + 2) / 3;
            Count expected = FML::NTasks == 1 ? 0 : moving * sizeof(Particle) * options.repeat;
            FML::COMMUNICATION::require_count(sums[size_t(Counter::PayloadSentBytes)] == expected,
                                              "migration wire-byte oracle mismatch");
        }
        if (!FML::ThisTask) {
            auto sorted = times;
            std::sort(sorted.begin(), sorted.end());
            const size_t middle = sorted.size() / 2;
            const double median = sorted.size() % 2 ? sorted[middle] : .5 * (sorted[middle - 1] + sorted[middle]);
            file << std::setprecision(17) << "{\"schema_version\":1,\"case\":" << quote(options.name)
                 << ",\"mesh\":" << options.mesh << ",\"ranks\":" << FML::NTasks << ",\"omp_threads\":" << FML::NThreads
                 << ",\"fftw_threads\":" << FML::GRID::get_fftw_nthreads()
                 << ",\"precision_bytes\":" << sizeof(FML::GRID::FloatType) << ",\"seed\":" << options.seed
                 << ",\"warmup\":" << options.warmup << ",\"repeat\":" << options.repeat
                 << ",\"particle_record_bytes\":" << sizeof(Particle) << ",\"work_mesh\":" << bench.work_mesh
                 << ",\"cache_shells\":" << options.shells << ",\"message_bytes\":" << options.message_bytes
                 << ",\"compiler\":" << quote(__VERSION__) << ",\"revision\":" << quote(build_info::revision)
                 << ",\"tracked_dirty\":" << quote(build_info::dirty)
                 << ",\"source_sha256\":" << quote(build_info::source_sha256)
                 << ",\"build_cxx\":" << quote(build_info::cxx) << ",\"build_options\":" << quote(build_info::options)
                 << ",\"build_includes\":" << quote(build_info::includes)
                 << ",\"build_libraries\":" << quote(build_info::libraries)
                 << ",\"build_links\":" << quote(build_info::links)
                 << ",\"mpi_enabled\":" << (mpi_enabled ? "true" : "false")
                 << ",\"omp_enabled\":" << (omp_enabled ? "true" : "false") << ",\"mpi_library\":" << quote(mpi_version)
                 << ",\"fftw_library\":" << quote(fftw_library) << ",\"cxx_standard\":" << __cplusplus
                 << ",\"fftw_planner_flag\":" << FML::GRID::get_fftw_planner_flag()
                 << ",\"counter_scope\":{\"allocations\":\"FFTWGrid storage/backup and transport payload "
                    "buffers\",\"copies\":\"FFTWGrid storage and explicit backup/clone copies; packing is "
                    "separate\",\"payload\":\"resampling/migration payload arguments; excludes FFTW and control "
                    "collectives\"}"
                 << ",\"median_max_rank_seconds\":" << median << ",\"maximum_rank_seconds\":" << sorted.back()
                 << ",\"max_rank_peak_rss_bytes\":" << (rss_known ? std::to_string(peak) : "null")
                 << ",\"sum_rank_peak_rss_bytes\":" << (rss_known ? std::to_string(peak_sum) : "null")
                 << ",\"validation_passed\":true,\"samples_max_rank_seconds\":[";
            for (size_t i = 0; i < times.size(); ++i)
                file << (i ? "," : "") << times[i];
            for (int group = 0; group < 2; ++group) {
                file << (group ? "},\"counter_rank_maxima\":{" : "],\"counter_rank_sums\":{");
                for (size_t i = 0; i < Perf::number_of_counters; ++i)
                    file << (i ? "," : "") << quote(Perf::names[i]) << ':' << (group ? maxima[i] : sums[i]);
            }
            file << "},\"rank_hosts\":[";
            for (size_t i = 0; i < rank_hosts.size(); ++i)
                file << (i ? "," : "") << quote(rank_hosts[i]);
            file << "]}\n";
            file.flush();
            if (!file)
                throw std::runtime_error("benchmark report write failed");
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "benchmark: " << e.what() << '\n';
        return 2;
    }
}
