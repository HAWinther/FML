#include <FML/LPT/ArbitraryOrder.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <map>

using namespace FML::COSMOLOGY::LPT;
using Grid = FML::GRID::FFTWGrid<3>;
using Complex = FML::GRID::ComplexType;
using Real = FML::GRID::FloatType;
using FML::TESTS::check_local;
constexpr bool single_precision = sizeof(Real) == sizeof(float);
constexpr double epsilon = std::numeric_limits<Real>::epsilon();

template <class Function> bool throws(Function function) {
    try { function(); } catch (const std::exception &) { return true; }
    return false;
}

EdSLPTOptions settings(int order = 5, int cutoff = 2) {
    EdSLPTOptions options;
    options.max_order = order;
    options.input_mode_limit.fill(cutoff);
    options.max_owned_bytes_per_rank = size_t(512) * 1024 * 1024;
    return options;
}

Grid fixture() {
    Grid input(16, 1, 2);
    std::fill(input.get_real_grid_left(), input.get_real_grid_left() + 2 * input.get_ntot_fourier_alloc(), Real(42));
    for (auto index : input.get_fourier_range())
        input.set_fourier_from_index(index, Complex(0));
    input.set_fourier({1,0,0}, Complex(.12,.03));
    input.set_fourier({15,0,0}, Complex(.12,-.03));
    input.set_fourier({0,1,1}, Complex(-.08,.04));
    input.set_fourier({1,15,2}, Complex(.06,-.02));
    input.set_grid_status_real(false);
    return input;
}

double difference(const EdSLPTVector & left, const EdSLPTVector & right) {
    double error = 0;
    for (int i = 0; i < 3; ++i)
        for (auto index : left[i].get_fourier_range())
            error = std::max(error, double(std::abs(left[i].get_fourier_from_index(index) - right[i].get_fourier_from_index(index))));
    return error;
}

void contract_tests() {
    std::array<Real, 9> rank_one;
    rank_one.fill(Real(.137));
    Real envelope;
    check_local(eds_detail::cubic(rank_one, rank_one, rank_one, envelope) == 0 && envelope > 0,
                "Mixed determinant cancels aligned rank-one minors exactly");
    auto input = fixture();
    std::vector<Real> saved(input.get_real_grid_left(), input.get_real_grid_left() + 2 * input.get_ntot_fourier_alloc());
    auto options = settings();
    auto plan = plan_eds_lpt(input, options, EdSLPTOutput::RetainOrders);
    auto result = compute_eds_lpt(input, options);
    check_local(plan.work_mesh == 22 && result.orders_fourier.size() == 5, "EdS plan chooses full-bandwidth 22^3 work mesh");
    check_local(plan.owned_peak_bound_bytes >= result.report.accounted_owned_peak_bytes &&
                result.report.fft_count == 52 && result.report.remap_count == 16,
                "EdS owned-capacity and transform counts agree with preflight");
    check_local(result.report.max_input_correction == 0, "Exact band-limited seed needs no correction");
    const auto planner_flag = FML::GRID::get_fftw_planner_flag();
    FML::GRID::set_fftw_planner_flag(FFTW_MEASURE);
    EdSLPTResult measured;
    try {
        measured = compute_eds_lpt(input, settings(3));
    } catch (...) {
        FML::GRID::set_fftw_planner_flag(planner_flag);
        throw;
    }
    FML::GRID::set_fftw_planner_flag(planner_flag);
    check_local(difference(measured.orders_fourier[2], result.orders_fourier[2]) < (single_precision ? 2e-9 : 1e-14) &&
                measured.report.accounted_owned_peak_bytes <= measured.report.owned_peak_bound_bytes,
                "Non-ESTIMATE FFT planning preserves fields within the owned bound");
    int delivered = 0;
    auto streamed = for_each_eds_lpt_order(input, options, [&](int n, const EdSLPTVector & fields) {
        ++delivered;
        check_local(n == delivered && difference(fields, result.orders_fourier[n - 1]) == 0,
                    "Streamed order agrees with collected output");
    });
    check_local(streamed.output_bytes < result.report.output_bytes, "Streaming reduces cropped-output storage");
    check_local(throws([&] { for_each_eds_lpt_order(input, options, {}); }), "Empty stream sink is rejected");
    check_local(throws([&] { for_each_eds_lpt_order(input, options, [](int, const EdSLPTVector &) {
        throw std::runtime_error("intentional sink failure");
    }); }), "Stream sink failure propagates");
    auto exact_budget = options;
    exact_budget.max_owned_bytes_per_rank = plan.owned_peak_bound_bytes;
    auto budget_result = compute_eds_lpt(input, exact_budget);
    check_local(difference(result.orders_fourier[4], budget_result.orders_fourier[4]) == 0, "Exact preflight budget is accepted");
    --exact_budget.max_owned_bytes_per_rank;
    check_local(throws([&] { compute_eds_lpt(input, exact_budget); }), "Budget below preflight bound is rejected");
    const double g = .7, step = single_precision ? .01 : 1e-5;
    auto evaluated = result.evaluate(g, options.max_owned_bytes_per_rank);
    auto plus = result.evaluate(g + step, options.max_owned_bytes_per_rank);
    auto minus = result.evaluate(g - step, options.max_owned_bytes_per_rank);
    double error = 0, derivative_error = 0, amplitude = 0;
    for (int i = 0; i < 3; ++i)
        for (auto index : evaluated.displacement_fourier[i].get_fourier_range()) {
            Complex sum(0), slope(0);
            for (int n = 1; n <= 5; ++n) {
                auto term = Real(std::pow(g, n)) * result.orders_fourier[n - 1][i].get_fourier_from_index(index);
                sum += term;
                slope += Real(n) * term;
            }
            error = std::max(error, double(std::abs(sum - evaluated.displacement_fourier[i].get_fourier_from_index(index))));
            error = std::max(error, double(std::abs(slope - evaluated.ddisplacement_dln_a_fourier[i].get_fourier_from_index(index))));
            const auto numerical = Real(g / (2 * step)) * (plus.displacement_fourier[i].get_fourier_from_index(index) -
                                                           minus.displacement_fourier[i].get_fourier_from_index(index));
            derivative_error = std::max(derivative_error, double(std::abs(numerical - slope)));
            amplitude = std::max(amplitude, double(std::abs(slope)));
        }
    check_local(error < 32 * epsilon * amplitude, "EdS evaluation uses g^n and n g^n weights");
    check_local(derivative_error < (single_precision ? 2e-4 : 1e-8) * amplitude, "EdS time derivative agrees with finite differences");
    const size_t retained_bytes = result.orders_fourier.capacity() * sizeof(EdSLPTVector) + 15 * plan.output_grid_bytes;
    const size_t evaluation_budget = retained_bytes + 6 * plan.output_grid_bytes;
    check_local(!throws([&] { result.evaluate(g, evaluation_budget); }) &&
                throws([&] { result.evaluate(g, evaluation_budget - 1); }), "Evaluation checks retained and new-output capacity");
    check_local(throws([&] { result.evaluate(0, evaluation_budget); }) &&
                throws([&] { result.evaluate(std::numeric_limits<double>::infinity(), evaluation_budget); }) &&
                throws([&] { result.evaluate(std::numeric_limits<double>::max(), evaluation_budget); }),
                "Invalid or overflowing growth weights are rejected");
    check_local(std::equal(saved.begin(), saved.end(), input.get_real_grid_left()) && !input.get_grid_status_real(),
                "Planning, computation, sink failure and evaluation preserve input and halos");
    input.free();
    check_local(!throws([&] { result.evaluate(g, evaluation_budget); }), "Collected result owns fields after the input is released");

    for (auto & field : result.orders_fourier[0])
        check_local(!field.get_grid_status_real() && field.get_n_extra_slices_left() == 0 &&
                    field.get_n_extra_slices_right() == 0, "Returned Fourier fields have zero halos");
    result.orders_fourier[0][0].set_grid_status_real(true);
    check_local(throws([&] { result.evaluate(g, evaluation_budget); }), "Evaluation rejects altered field status");
}

void invalid_input_tests() {
    auto options = settings();
    auto input = fixture();
    auto rejected = [&](const char * label, const std::function<void(Grid &, EdSLPTOptions &)> & change) {
        auto changed = input;
        auto settings_copy = options;
        change(changed, settings_copy);
        check_local(throws([&] { compute_eds_lpt(changed, settings_copy); }), label);
    };
    rejected("Missing cutoff is rejected", [](Grid &, EdSLPTOptions & o) { o.input_mode_limit.fill(-1); });
    rejected("Nyquist cutoff is rejected", [](Grid &, EdSLPTOptions & o) { o.input_mode_limit.fill(8); });
    rejected("Missing budget is rejected", [](Grid &, EdSLPTOptions & o) { o.max_owned_bytes_per_rank = 0; });
    rejected("Zero order is rejected", [](Grid &, EdSLPTOptions & o) { o.max_order = 0; });
    rejected("Work-mesh overflow is rejected", [](Grid &, EdSLPTOptions & o) { o.max_order = std::numeric_limits<int>::max() - 1; });
    rejected("Real input is rejected", [](Grid & g, EdSLPTOptions &) { g.set_grid_status_real(true); });
    rejected("Input mean is rejected", [](Grid & g, EdSLPTOptions &) { g.set_fourier({0,0,0}, Complex(.01)); });
    rejected("Off-support input is rejected", [](Grid & g, EdSLPTOptions &) { g.set_fourier({3,0,1}, Complex(.01)); });
    rejected("Nyquist input is rejected", [](Grid & g, EdSLPTOptions &) { g.set_fourier({8,0,0}, Complex(.01)); });
    rejected("Non-Hermitian boundary is rejected", [](Grid & g, EdSLPTOptions &) { g.set_fourier({15,0,0}, Complex(.12,.03)); });
    rejected("NaN input is rejected", [](Grid & g, EdSLPTOptions &) { g.set_fourier({1,0,0}, Complex(std::numeric_limits<Real>::quiet_NaN())); });
    const double perturbation = single_precision ? 1e-8 : 1e-17;
    auto small = input;
    small.set_fourier({0,0,0}, Complex(Real(perturbation)));
    small.set_fourier({3,0,1}, Complex(Real(perturbation)));
    small.set_fourier({1,0,0}, small.get_fourier({1,0,0}) + Complex(0,Real(perturbation)));
    auto corrected = compute_eds_lpt(small, options);
    check_local(corrected.report.max_mean_correction > 0 && corrected.report.max_offband_correction > 0 &&
                corrected.report.max_hermitian_correction > 0, "Roundoff seed corrections are accepted and reported");
    Grid empty;
    check_local(throws([&] { compute_eds_lpt(empty, options); }), "Unallocated input is rejected");
    check_local(throws([&] { plan_eds_lpt(input, options, static_cast<EdSLPTOutput>(100)); }), "Invalid output mode is rejected");
    options.fft_workspace_allowance_bytes_per_rank = std::numeric_limits<size_t>::max();
    check_local(throws([&] { plan_eds_lpt(input, options, EdSLPTOutput::RetainOrders); }), "Report-total overflow is rejected");
    Grid zero(8,0,0);
    zero.set_grid_status_real(false);
    auto zero_result = compute_eds_lpt(zero, settings(5,0));
    double maximum = 0;
    for (const auto & fields : zero_result.orders_fourier)
        for (const auto & field : fields)
            for (auto index : field.get_fourier_range())
                maximum = std::max(maximum, double(std::abs(field.get_fourier_from_index(index))));
    check_local(maximum == 0, "Zero-support seed gives exactly zero fields");
    check_local(compute_eds_lpt(zero, settings(7,0)).report.beyond_validated_order, "Generic order seven is marked beyond current validation");
}

int export_fields(int argc, char ** argv) {
    if (argc != 7)
        throw std::invalid_argument("Usage: --export input output mesh cutoff order");
    const int mesh = std::stoi(argv[4]), cutoff = std::stoi(argv[5]), order = std::stoi(argv[6]);
    Grid input(mesh,0,0);
    input.set_grid_status_real(false);
    std::ifstream file(argv[2]);
    if (!file)
        throw std::runtime_error("Cannot read the reference fixture");
    std::array<int,3> mode;
    double re, im;
    while (file >> mode[0] >> mode[1] >> mode[2] >> re >> im) {
        if (mode[2] < 0)
            continue;
        for (auto & component : mode)
            component = (component % mesh + mesh) % mesh;
        input.set_fourier(mode, Complex(Real(re), Real(im)));
    }
    if (!file.eof())
        throw std::runtime_error("Malformed reference fixture");
    auto result = compute_eds_lpt(input, settings(order, cutoff));
    std::ofstream output(argv[3]);
    output << std::setprecision(17);
    for (int n = 1; n <= order; ++n)
        for (int axis = 0; axis < 3; ++axis)
            for (auto index : result.orders_fourier[n - 1][axis].get_fourier_range()) {
                auto m = eds_detail::mode_numbers(input, index);
                const auto value = result.orders_fourier[n - 1][axis].get_fourier_from_index(index);
                output << n << ' ' << m[0] << ' ' << m[1] << ' ' << m[2] << ' ' << axis << ' '
                       << value.real() << ' ' << value.imag() << '\n';
            }
    output.close();
    if (!output)
        throw std::runtime_error("Cannot write comparison fields");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1)
        return export_fields(argc, argv);
    contract_tests();
    invalid_input_tests();
    return FML::TESTS::summary("Arbitrary-order EdS LPT");
}
