#include <FML/LPT/ArbitraryOrder.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <sys/resource.h>

using namespace FML::COSMOLOGY::LPT;
using Grid = FML::GRID::FFTWGrid<3>;
using Complex = FML::GRID::ComplexType;
using Real = FML::GRID::FloatType;
constexpr double comparison_tolerance = sizeof(Real) == sizeof(float) ? 2e-9 : 1e-15;
using FML::TESTS::check_local;

EdSLPTOptions settings(int order = 5, int cutoff = 2) {
    EdSLPTOptions options;
    options.max_order = order;
    options.input_mode_limit.fill(cutoff);
    options.max_owned_bytes_per_rank = 512ULL * 1024 * 1024;
    options.max_remap_message_bytes = 3 * sizeof(Complex); // Force transport chunking.
    return options;
}

Grid make_input(int mesh, const std::map<std::array<int,3>, Complex> & modes, int halos = 0) {
    Grid grid(mesh,halos,halos);
    std::fill(grid.get_real_grid_left(), grid.get_real_grid_left() + 2*grid.get_ntot_fourier_alloc(), Real(42));
    grid.set_grid_status_real(false);
    for (auto index : grid.get_fourier_range()) {
        const auto found = modes.find(eds_detail::mode_numbers(grid,index));
        grid.set_fourier_from_index(index, found == modes.end() ? Complex(0) : found->second);
    }
    return grid;
}

Grid fixture(int mesh = 16) {
    return make_input(mesh, {{{1,0,0},Complex(.12,.03)}, {{-1,0,0},Complex(.12,-.03)},
                             {{0,1,1},Complex(-.08,.04)}, {{1,-1,2},Complex(.06,-.02)}}, 1);
}

double difference(const EdSLPTVector & a, const EdSLPTVector & b) {
    double error = 0;
    for (int i = 0; i < 3; ++i)
        for (auto index : a[i].get_fourier_range())
            error = std::max(error, double(std::abs(a[i].get_fourier_from_index(index) - b[i].get_fourier_from_index(index))));
    FML::MaxOverTasks(&error);
    return error;
}

template <class Function> void collective_failure(Function function, const char * description) {
    int caught = 0;
    std::string message;
    try { function(); }
    catch (const std::exception & error) { caught = 1; message = error.what(); }
    check_local(caught == 1, description);
    // The next collective also proves every participant left the failed operation.
    unsigned long long hash = 1469598103934665603ULL;
    for (unsigned char c : message) hash = (hash ^ c) * 1099511628211ULL;
    auto minimum = hash, maximum = hash;
    FML::MinOverTasks(&minimum);
    FML::MaxOverTasks(&maximum);
    check_local(minimum == maximum, "Collective failure has a common error message");
}

void correctness_and_failures() {
    auto input = fixture();
    const std::vector<Real> before(input.get_real_grid_left(), input.get_real_grid_left()+2*input.get_ntot_fourier_alloc());
    auto options = settings();
    const auto original = compute_eds_lpt(input,options);
    options.workspace = EdSLPTWorkspace::RecomputeGradients;
    const auto recomputed = compute_eds_lpt(input,options);
    for (int n = 0; n < 5; ++n)
        check_local(difference(original.orders_fourier[n],recomputed.orders_fourier[n]) < comparison_tolerance,
                    "Recomputed gradients reproduce cached-gradient fields");
    check_local(recomputed.report.fft_count == recomputed.report.planned_fft_count &&
                original.report.maximum_accounted_peak_bytes <= original.report.maximum_owned_peak_bound_bytes,
                "Distributed transform and capacity accounting agrees with preflight");
    options.workspace = EdSLPTWorkspace::CacheGradients;
    int delivered = 0;
    double local_stream_error = 0;
    for_each_eds_lpt_order(input, options, [&](int n, const EdSLPTVector & fields) {
        ++delivered;
        for (int i = 0; i < 3; ++i)
            for (auto index : fields[i].get_fourier_range())
                local_stream_error = std::max(local_stream_error,
                    double(std::abs(fields[i].get_fourier_from_index(index) - original.orders_fourier[n-1][i].get_fourier_from_index(index))));
    });
    check_local(delivered == 5 && local_stream_error == 0, "Distributed streaming reproduces collected fields");
    const auto evaluated = original.evaluate(.7, options.max_owned_bytes_per_rank);
    const auto evaluated_recomputed = recomputed.evaluate(.7, options.max_owned_bytes_per_rank);
    check_local(difference(evaluated.displacement_fourier,evaluated_recomputed.displacement_fourier) < comparison_tolerance,
                "Distributed EdS evaluation agrees across workspace policies");
    const int target = FML::NTasks-1;
    collective_failure([&] {
        for_each_eds_lpt_order(input, options, [&](int n, const EdSLPTVector &) {
            if (n == 2 && FML::ThisTask == target) throw std::runtime_error("single-rank sink failure");
        });
    }, "One-rank sink failure reaches every rank");
    collective_failure([&] {
        EdSLPTOrderSink sink;
        if (FML::ThisTask != target) sink = [](int,const EdSLPTVector &) {};
        for_each_eds_lpt_order(input,options,sink);
    }, "Missing sink on one rank fails collectively");
    collective_failure([&] {
        auto bad = input;
        if (FML::ThisTask == target) bad.set_grid_status_real(true);
        compute_eds_lpt(bad,options);
    }, "One-rank status failure reaches every rank");
    collective_failure([&] {
        auto bad = input;
        if (FML::ThisTask == target) bad.set_fourier_from_index(0,Complex(std::numeric_limits<Real>::quiet_NaN()));
        compute_eds_lpt(bad,options);
    }, "One-rank nonfinite input fails collectively");
    collective_failure([&] {
        auto bad = input;
        if (FML::ThisTask == target) bad.set_fourier({0,0,3},Complex(.01));
        compute_eds_lpt(bad,options);
    }, "One-rank off-support coefficient fails collectively");
    collective_failure([&] {
        auto bad = input;
        // The negative-x conjugate of (1,0,0) belongs to the final slab.
        if (FML::ThisTask == target) bad.set_fourier({15-int(bad.get_local_x_start()),0,0},Complex(.2));
        compute_eds_lpt(bad,options);
    }, "Hermitian mismatch across slab owners fails collectively");
#ifdef FML_EDS_LPT_TEST_ALLOCATIONS
    collective_failure([&] {
        if (FML::ThisTask == target) eds_detail::fail_allocation_after = 3;
        compute_eds_lpt(input,options);
    }, "One-rank allocation failure reaches every rank before the next FFT");
    eds_detail::fail_allocation_after = -1;
#endif
#if defined(FML_FOURIER_RESAMPLING_TEST_ALLOCATIONS) && defined(USE_MPI)
    using Remap = FML::GRID::FourierResamplingPlan<3>;
    Grid output(24,0,0);
    collective_failure([&] {
        if (FML::ThisTask == target) Remap::fail_geometry_once = true;
        Remap plan(input,output);
    }, "One-rank remapping geometry allocation failure is recovered");
    Remap plan(input,output);
    collective_failure([&] {
        if (FML::ThisTask == target) Remap::fail_transport_once = true;
        plan.apply(input,output,3*sizeof(Complex));
    }, "One-rank remapping transport allocation failure is recovered");
    plan.apply(input,output,3*sizeof(Complex));
    check_local(true,"Remapping succeeds after a recovered transport allocation failure");
#endif
    if (FML::NTasks > 1) {
        collective_failure([&] {
            auto changed = options;
            if (FML::ThisTask == target) changed.max_order = 4;
            compute_eds_lpt(input,changed);
        }, "Cross-rank option mismatch fails collectively");
        collective_failure([&] {
            auto changed = options;
            if (FML::ThisTask == target) --changed.max_owned_bytes_per_rank;
            compute_eds_lpt(input,changed);
        }, "Cross-rank budget mismatch fails collectively");
        collective_failure([&] {
            original.evaluate(FML::ThisTask == target ? .8 : .7, options.max_owned_bytes_per_rank);
        }, "Cross-rank evaluation mismatch fails collectively");
    }
    auto budget = options;
    budget.max_owned_bytes_per_rank = original.report.maximum_owned_peak_bound_bytes;
    check_local(difference(compute_eds_lpt(input,budget).orders_fourier[4],original.orders_fourier[4]) == 0,
                "Maximum per-rank bound is an accepted common budget");
    --budget.max_owned_bytes_per_rank;
    collective_failure([&] { compute_eds_lpt(input,budget); }, "Common budget one byte below the worst rank fails");
    collective_failure([&] {
        auto copy = original;
        if (FML::ThisTask == target) copy.orders_fourier[1][0].set_grid_status_real(true);
        copy.evaluate(.7,options.max_owned_bytes_per_rank);
    }, "One-rank retained-field failure reaches every rank");
    check_local(std::equal(before.begin(),before.end(),input.get_real_grid_left()) && !input.get_grid_status_real(),
                "Distributed computations and failures preserve input and halos");
    auto after_failure = compute_eds_lpt(input,options);
    check_local(difference(after_failure.orders_fourier[4],original.orders_fourier[4]) == 0,
                "Successful computation follows recovered failures");
}

void export_or_benchmark(int argc, char ** argv) {
    const bool benchmark = std::string(argv[1]) == "--benchmark";
    if ((!benchmark && argc != 8) || (benchmark && argc != 4))
        throw std::invalid_argument("Expected --export input prefix mesh cutoff order cache|recompute, or --benchmark prefix cache|recompute");
    const int mesh = benchmark ? 32 : std::stoi(argv[4]);
    auto options = settings(benchmark ? 8 : std::stoi(argv[6]), benchmark ? 2 : std::stoi(argv[5]));
    options.max_remap_message_bytes = 64*1024;
    const std::string policy = benchmark ? argv[3] : argv[7];
    if (policy == "recompute") options.workspace = EdSLPTWorkspace::RecomputeGradients;
    else if (policy != "cache") throw std::invalid_argument("Unknown workspace policy");
    Grid input;
    if (benchmark) input = fixture(mesh);
    else {
        std::map<std::array<int,3>,Complex> modes;
        eds_detail::collective_try([&] {
            std::ifstream file(argv[2]);
            if (!file) throw std::runtime_error("Cannot read sparse fixture");
            std::array<int,3> mode;
            double re,im;
            while (file >> mode[0] >> mode[1] >> mode[2] >> re >> im)
                modes[mode] = Complex(Real(re),Real(im));
            if (!file.eof()) throw std::runtime_error("Malformed sparse fixture");
        }, "Read test fixture");
        input = make_input(mesh,modes);
    }
    const auto result = compute_eds_lpt(input,options);
    const std::string prefix = benchmark ? argv[2] : argv[3];
    if (!benchmark) {
        eds_detail::collective_try([&] {
            std::ofstream output(prefix+"."+std::to_string(FML::ThisTask));
            output << std::setprecision(17);
            for (int n = 0; n < options.max_order; ++n)
                for (int axis = 0; axis < 3; ++axis)
                    for (auto index : result.orders_fourier[n][axis].get_fourier_range()) {
                        const auto m = eds_detail::mode_numbers(input,index);
                        const auto value = result.orders_fourier[n][axis].get_fourier_from_index(index);
                        output << n+1 << ' ' << m[0] << ' ' << m[1] << ' ' << m[2] << ' ' << axis << ' '
                               << value.real() << ' ' << value.imag() << '\n';
                    }
            output.close();
            if (!output) throw std::runtime_error("Cannot write comparison fields");
        }, "Write test output");
        eds_detail::collective_try([&] {
            if (FML::ThisTask == 0) {
                std::ofstream metadata(prefix + ".report.json");
                metadata << "{\"scalar_bytes\":" << result.report.scalar_bytes
                         << ",\"ranks\":" << result.report.ranks << ",\"threads\":" << result.report.threads
                         << ",\"work_mesh\":" << result.report.work_mesh
                         << ",\"validated_max_order\":" << result.report.validated_max_order
                         << ",\"worst_compatibility_ratio\":" << result.report.worst_compatibility_ratio << "}\n";
                metadata.close();
                if (!metadata) throw std::runtime_error("Cannot write comparison metadata");
            }
        }, "Write test metadata");
    } else {
        rusage usage{};
        getrusage(RUSAGE_SELF,&usage);
        size_t rss = size_t(usage.ru_maxrss);
#ifndef __APPLE__
        rss *= 1024;
#endif
        size_t maximum, total;
        eds_detail::aggregate_bytes(rss,maximum,total);
        if (FML::ThisTask == 0) {
            std::ofstream output(prefix);
            output << "{\"ranks\":" << FML::NTasks << ",\"threads\":" << FML::NThreads
                   << ",\"workspace\":\"" << policy << "\",\"work_mesh\":" << result.report.work_mesh
                   << ",\"seconds\":" << result.report.maximum_elapsed_seconds
                   << ",\"planned_max_bytes\":" << result.report.maximum_owned_peak_bound_bytes
                   << ",\"accounted_max_bytes\":" << result.report.maximum_accounted_peak_bytes
                   << ",\"rss_max_bytes\":" << maximum << ",\"rss_sum_bytes\":" << total
                   << ",\"fft_count\":" << result.report.fft_count << "}\n";
        }
    }
}

int main(int argc,char ** argv) {
    if (argc > 1) { export_or_benchmark(argc,argv); return 0; }
    correctness_and_failures();
    return FML::TESTS::summary("Parallel arbitrary-order LPT");
}
