#include <FML/LPT/ArbitraryOrder.h>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace FML::COSMOLOGY::LPT;
using Grid = FML::GRID::FFTWGrid<3>;
using Real = FML::GRID::FloatType;
using Complex = FML::GRID::ComplexType;

double rms(const EdSLPTVector & vector) {
    double sum = 0;
    for (const auto & field : vector)
        for (auto index : field.get_fourier_range()) {
            const auto z = field.get_fourier_coord_from_index(index)[2];
            const double multiplicity = z == 0 || 2 * z == field.get_nmesh() ? 1 : 2;
            sum += multiplicity * double(std::norm(field.get_fourier_from_index(index)));
        }
    FML::SumOverTasks(&sum);
    return std::sqrt(sum); // Normalized forward transform: Parseval in unit-box coordinates.
}
std::pair<double, unsigned long long> sampled_jacobian(const EdSLPTVector & sum) {
    std::array<Grid, 9> gradient;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            auto & field = gradient[3 * i + j];
            field = sum[i];
            for (auto index : field.get_fourier_range()) {
                std::array<double, 3> wave;
                double k2;
                field.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                field.set_fourier_from_index(index, Complex(0, Real(wave[j])) * field.get_fourier_from_index(index));
            }
            field.fftw_c2r();
        }
    double minimum = std::numeric_limits<double>::infinity();
    unsigned long long nonpositive = 0;
    for (auto index : gradient[0].get_real_range()) {
        std::array<double, 9> f;
        for (int i = 0; i < 9; ++i)
            f[i] = gradient[i].get_real_from_index(index) + (i % 4 == 0 ? 1 : 0);
        const double determinant = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[1] * (f[3] * f[8] - f[5] * f[6]) +
                                   f[2] * (f[3] * f[7] - f[4] * f[6]);
        minimum = std::min(minimum, determinant);
        nonpositive += determinant <= 0;
    }
    FML::MinOverTasks(&minimum);
    FML::SumOverTasks(&nonpositive);
    return {minimum, nonpositive};
}
int main(int argc, char ** argv) {
    int mesh = 16, order = 6;
    double growth = .5;
    std::string prefix = "eds_lpt";
    bool plane = false;
    FML::COMMUNICATION::collective_try(
        [&] {
            if (argc != 1 && argc != 5 && argc != 6)
                throw std::invalid_argument("Usage: eds_lpt [prefix order growth mesh [coupled|plane]]");
            if (argc >= 5) {
                prefix = argv[1];
                order = std::stoi(argv[2]);
                growth = std::stod(argv[3]);
                mesh = std::stoi(argv[4]);
                if (argc == 6) {
                    const std::string seed = argv[5];
                    if (seed != "plane" && seed != "coupled")
                        throw std::invalid_argument("Unknown example seed");
                    plane = seed == "plane";
                }
            }
            if (mesh < 4 || mesh > 64 || order < 1 || order > (mesh - 1) / 2 || !(growth > 0) || !std::isfinite(growth))
                throw std::invalid_argument("Need 1 <= order < mesh/2, 4 <= mesh <= 64 and finite positive growth");
        },
        "Read EdS example options");
    std::array<double, 4> lower{double(mesh), double(order), growth, double(plane)}, upper = lower;
    FML::MinArrayOverTasks(lower.data(), lower.size());
    FML::MaxArrayOverTasks(upper.data(), upper.size());
    FML::COMMUNICATION::collective_try(
        [&] {
            if (lower != upper)
                throw std::invalid_argument("Example options differ across ranks");
        },
        "Agree EdS example options");
    EdSLPTOptions options;
    options.max_order = order;
    options.input_mode_limit = {1, 1, 1};
    options.max_owned_bytes_per_rank = 512ULL * 1024 * 1024;
    Grid input(mesh, 0, 0);
    input.set_grid_status_real(false);
    // Reference-epoch density: three non-coplanar waves plus their conjugates.
    std::array<std::array<int, 3>, 3> modes{{{1, 0, 0}, {0, 1, 1}, {1, -1, 1}}};
    std::array<Complex, 3> values{Complex(.12, .03), Complex(-.08, .04), Complex(.06, -.02)};
    if (plane) {
        modes[0] = {1, 1, 1};
        values = {Complex(.1, .04), Complex(0), Complex(0)};
    }
    for (auto index : input.get_fourier_range()) {
        auto coordinate = input.get_fourier_coord_from_index(index);
        coordinate[0] += input.get_local_x_start();
        for (int d = 0; d < 2; ++d)
            if (coordinate[d] > mesh / 2)
                coordinate[d] -= mesh;
        Complex value(0);
        for (size_t i = 0; i < modes.size(); ++i) {
            if (coordinate == modes[i])
                value = values[i];
            auto opposite = modes[i];
            for (auto & v : opposite)
                v = -v;
            if (coordinate == opposite)
                value = std::conj(values[i]);
        }
        input.set_fourier_from_index(index, value);
    }
    auto result = compute_eds_lpt(input, options);
    auto evaluated = result.evaluate(growth, options.max_owned_bytes_per_rank);
    EdSLPTVector partial;
    for (auto & field : partial) {
        field = Grid(mesh, 0, 0);
        field.set_grid_status_real(false);
    }
    std::ofstream output;
    FML::COMMUNICATION::collective_try(
        [&] {
            if (!FML::ThisTask) {
                output.open(prefix + ".csv");
                output << std::setprecision(17)
                       << "order,growth,order_rms,weighted_order_rms,sum_rms,min_jacobian,nonpositive_samples\n";
                if (!output)
                    throw std::runtime_error("Cannot open example output");
            }
        },
        "Open EdS diagnostics");
    Real weight = 1;
    for (int n = 1; n <= order; ++n) {
        weight *= Real(growth);
        for (int axis = 0; axis < 3; ++axis)
            for (auto index : partial[axis].get_fourier_range())
                partial[axis].set_fourier_from_index(
                    index,
                    partial[axis].get_fourier_from_index(index) +
                        weight * result.orders_fourier[n - 1][axis].get_fourier_from_index(index));
        const auto jacobian = sampled_jacobian(partial);
        const double term = rms(result.orders_fourier[n - 1]), total = rms(partial);
        if (!FML::ThisTask)
            output << n << ',' << growth << ',' << term << ',' << double(weight) * term << ',' << total << ','
                   << jacobian.first << ',' << jacobian.second << '\n';
    }
    const double evaluated_norm = rms(evaluated.displacement_fourier);
    FML::COMMUNICATION::collective_try(
        [&] {
            if (!FML::ThisTask) {
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot write example diagnostics");
                std::cout << std::setprecision(17) << "scalar_bytes=" << result.report.scalar_bytes
                          << " work_mesh=" << result.report.work_mesh << " evaluated_rms=" << evaluated_norm
                          << " planned_bytes=" << result.report.maximum_owned_peak_bound_bytes
                          << " accounted_bytes=" << result.report.maximum_accounted_peak_bytes << '\n';
                std::cout << "Jacobian values are grid samples; positive values do not prove convergence or exclude "
                             "crossings between cells.\n"
                          << "The example's diagnostic copies/scratch are outside the solver's owned-memory report.\n";
            }
        },
        "Write EdS diagnostics");
}
