#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/FFTWGrid/FourierResampling.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/COLASolver/src/ParameterSchema.h>
#include <FML/COLASolver/src/InitialConditions.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <cstdint>
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace FML::RANDOM;
int failures = 0;
void check(bool ok, const std::string & label) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n";
    failures += !ok;
}
template <int N>
using Grid = FML::GRID::FFTWGrid<N>;
template <int N>
double difference(const Grid<N> & a, const Grid<N> & b) {
    double result = 0;
    for (auto i : a.get_fourier_range())
        result = std::max(result, double(std::abs(a.get_fourier_from_index(i) - b.get_fourier_from_index(i))));
    FML::MaxOverTasks(&result);
    return result;
}
template <int N>
void test_field(int mesh) {
    using namespace GAUSSIAN;
    RandomFieldOptions opts;
    opts.backend = RandomFieldBackend::Philox;
    opts.seed = 123456789;
    opts.stream = 7;
    auto power = [](double) { return 1.0; };
    Grid<N> field(mesh), fixed(mesh), other(mesh);
    generate_gaussian_random_field_fourier<N>(field, nullptr, power, false, opts);
    generate_gaussian_random_field_fourier<N>(fixed, nullptr, power, true, opts);
    double worst = 0;
    bool zeros = true;
    for (auto index : field.get_fourier_range()) {
        auto coord = field.get_fourier_coord_from_index(index);
        coord[0] += field.get_local_x_start();
        bool excluded = true, nyq = false;
        for (auto k : coord) {
            excluded = excluded && k == 0;
            nyq = nyq || (mesh % 2 == 0 && k == mesh / 2);
        }
        auto a = field.get_fourier_from_index(index), b = fixed.get_fourier_from_index(index);
        if (excluded || nyq)
            zeros = zeros && a == decltype(a)(0) && b == decltype(b)(0);
        else {
            worst = std::max(worst, double(std::abs(std::abs(b) - 1)));
            if (std::abs(a) > 0)
                worst = std::max(worst, double(std::abs(a / std::abs(a) - b)));
        }
    }
    FML::MaxOverTasks(&worst);
    int bad = !zeros;
    FML::MaxOverTasks(&bad);
    check(!bad && worst < 2e-14,
          "Philox DC/Nyquist and fixed-amplitude phase conventions, " + std::to_string(N) + "D mesh " +
              std::to_string(mesh));
    auto roundtrip = field;
    roundtrip.fftw_c2r();
    roundtrip.fftw_r2c();
    check(difference(field, roundtrip) < 2e-14, "Philox satisfies Hermitian symmetry through FFT round trip");
    opts.stream++;
    generate_gaussian_random_field_fourier<N>(other, nullptr, power, false, opts);
    check(difference(field, other) > 0.1, "independent stream changes the realization");
    opts.stream--;
#ifdef USE_OMP
    int old = omp_get_max_threads();
    omp_set_num_threads(1);
    generate_gaussian_random_field_fourier<N>(other, nullptr, power, false, opts);
    omp_set_num_threads(old);
    check(difference(field, other) == 0, "Philox output is bit-identical across thread counts");
#endif
    unsigned long long hash = 0;
    for (auto index : field.get_fourier_range()) {
        auto coord = field.get_fourier_coord_from_index(index);
        coord[0] += field.get_local_x_start();
        auto value = field.get_fourier_from_index(index);
        uint64_t h = 1469598103934665603ull;
        for (auto k : coord)
            h = (h ^ uint64_t(k)) * 1099511628211ull;
        for (double v : {double(value.real()), double(value.imag())}) {
            uint64_t bits;
            std::memcpy(&bits, &v, sizeof bits);
            h = (h ^ bits) * 1099511628211ull;
        }
        hash ^= h;
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, FML::communicator);
#endif
    if (FML::ThisTask == 0)
        std::cout << "FIELD_HASH dim=" << N << " mesh=" << mesh << " value=" << hash << "\n";
}
int main() {
    using COUNTER_BASED::philox4x32;
    check(philox4x32({0, 0, 0, 0}, {0, 0}) ==
              std::array<uint32_t, 4>{0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u},
          "Philox zero known-answer vector matches Random123");
    check(philox4x32({~0u, ~0u, ~0u, ~0u}, {~0u, ~0u}) ==
              std::array<uint32_t, 4>{0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu},
          "Philox all-ones known-answer vector matches Random123");
    check(philox4x32({0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}) ==
              std::array<uint32_t, 4>{0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u},
          "Philox pi known-answer vector matches Random123");
    test_field<3>(16);
    test_field<2>(16);
    if (FML::NTasks == 1)
        test_field<3>(15);
    using namespace GAUSSIAN;
    RandomFieldOptions legacy;
    Grid<3> a(16), b(16);
    auto power = [](double) { return 1.0; };
    for (bool fixed : {false, true}) {
        RandomGenerator one(1234), two(1234);
        generate_gaussian_random_field_fourier<3>(a, &one, power, fixed);
        generate_gaussian_random_field_fourier<3>(b, &two, power, fixed, legacy);
        check(difference(a, b) == 0 && one.generate_uniform() == two.generate_uniform(),
              "explicit N-GenIC backend preserves legacy field and RNG state");
#ifdef USE_GSL
        GSLRandomGenerator gsl_one(1234), gsl_two(1234);
        generate_gaussian_random_field_fourier<3>(a, &gsl_one, power, fixed);
        generate_gaussian_random_field_fourier<3>(b, &gsl_two, power, fixed, legacy);
        check(difference(a, b) == 0 && gsl_one.generate_uniform() == gsl_two.generate_uniform(),
              "explicit N-GenIC backend preserves the GSL/2LPTic path");
#endif
    }
    RandomFieldOptions philox;
    philox.backend = RandomFieldBackend::Philox;
    philox.seed = 897;
    Grid<3> small(16), large(32);
    generate_gaussian_random_field_fourier<3>(small, nullptr, power, true, philox);
    generate_gaussian_random_field_fourier<3>(large, nullptr, power, true, philox);
    auto down = FML::GRID::resize_fourier_grid(large, 16);
    check(difference(small, down) == 0, "common signed Fourier modes match exactly across mesh resolutions");
    Grid<3> sample(32);
    generate_gaussian_random_field_fourier<3>(sample, nullptr, power, false, philox);
    double re = 0, im = 0, variance = 0, count = 0;
    for (auto i : sample.get_fourier_range()) {
        auto c = sample.get_fourier_coord_from_index(i);
        c[0] += sample.get_local_x_start();
        if (c[0] == 16 || c[1] == 16 || c[2] == 0 || c[2] == 16)
            continue;
        auto z = sample.get_fourier_from_index(i);
        re += z.real();
        im += z.imag();
        variance += std::norm(z);
        count++;
    }
    FML::SumOverTasks(&re);
    FML::SumOverTasks(&im);
    FML::SumOverTasks(&variance);
    FML::SumOverTasks(&count);
    check(std::abs(re / count) < 0.04 && std::abs(im / count) < 0.04 && std::abs(variance / count - 1) < 0.05,
          "Philox complex Gaussian sample has the expected mean and power");
    Grid<3> ng(16), g(16);
    NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier<3>(ng,
                                                                    nullptr,
                                                                    power,
                                                                    true,
                                                                    0.0,
                                                                    "local",
                                                                    0.0,
                                                                    {},
                                                                    philox);
    generate_gaussian_random_field_fourier<3>(g, nullptr, power, true, philox);
    check(difference(ng, g) < 1e-13, "fNL=0 uses the selected Philox Gaussian backend");
    {
        COLASOLVER::ICSettings ic{};
        check(ic.random_field_backend == COLASOLVER::RandomFieldBackend::NGenIC,
              "direct ICSettings construction retains the legacy backend");
        ic.random_field_type = COLASOLVER::ICRandomFieldType::Gaussian;
        ic.random_field_backend = COLASOLVER::RandomFieldBackend::Philox;
        ic.random_seed = 897;
        ic.fix_amplitude = true;
        FML::INTERPOLATION::SPLINE::Spline flat(std::vector<double>{0, 100, 1000},
                                                std::vector<double>{1, 1, 1},
                                                "flat power");
        COLASOLVER::generate_initial_density_field<3>(ng, ic, 1.0, flat, flat, {}, {}, nullptr);
        check(difference(ng, g) < 1e-14, "COLASolver Gaussian IC path dispatches to Philox");
        ic.random_field_backend = COLASOLVER::RandomFieldBackend::NGenIC;
        RandomGenerator legacy_one(897), legacy_two(897);
        COLASOLVER::generate_initial_density_field<3>(ng, ic, 1.0, flat, flat, {}, {}, &legacy_one);
        generate_gaussian_random_field_fourier<3>(g, &legacy_two, power, true);
        check(difference(ng, g) == 0, "COLASolver explicit legacy IC path retains the original realization");
    }
    const auto & schema = COLASOLVER::parameter_schema<3>();
    auto option =
        std::find_if(schema.begin(), schema.end(), [](const auto & s) { return s.name == "ic_random_field_backend"; });
    check(option != schema.end() && option->need == COLASOLVER::Need::AlwaysOptional &&
              std::get<std::string>(option->default_value) == "ngenic" &&
              option->choices == std::vector<std::string>{"ngenic", "philox"},
          "COLASolver exposes both backends and defaults to N-GenIC compatibility");
    FML::SumOverTasks(&failures);
    return failures ? 1 : 0;
}
