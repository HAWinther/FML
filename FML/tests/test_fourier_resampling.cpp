#include <FML/FFTWGrid/FourierResampling.h>
#include <iostream>

using namespace FML::GRID;
int failures = 0;
void check(bool good, const std::string & label) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
    failures += bad;
}
template <size_t N>
ComplexType value(const std::array<int, N> & wave, int generation) {
    int code = generation * 1000, weight = 1;
    for (int k : wave) {
        code += weight * k;
        weight *= 23;
    }
    return ComplexType(FloatType(code + 1), FloatType(2 * code - 3));
}
template <int N>
auto wave_at(const FFTWGrid<N> & grid, ptrdiff_t index) {
    auto wave = grid.get_fourier_coord_from_index(index);
    wave[0] += int(grid.get_local_x_start());
    for (int d = 0; d < N - 1; ++d)
        if (wave[d] > grid.get_nmesh() / 2)
            wave[d] -= grid.get_nmesh();
    return wave;
}
template <int N>
void fill(FFTWGrid<N> & grid, int generation) {
    grid.set_grid_status_real(false);
    for (auto i : grid.get_fourier_range())
        grid.set_fourier_from_index(i, value(wave_at(grid, i), generation));
}
template <int N>
bool matches(const FFTWGrid<N> & output, int oldmesh, int generation) {
    bool good = true;
    for (auto i : output.get_fourier_range()) {
        auto wave = wave_at(output, i);
        bool keep = true;
        for (int k : wave)
            keep &= 2LL * std::abs(k) < std::min(oldmesh, output.get_nmesh());
        auto expected = keep ? value(wave, generation) : ComplexType(0);
        good &= output.get_fourier_from_index(i) == expected;
    }
    return good;
}
template <int N>
void test(int oldmesh, int newmesh) {
    FFTWGrid<N> input(oldmesh, 1, 2), output(newmesh, 1, 1);
    FourierResamplingPlan<N> plan(input, output);
    // Reuse after changing field values; change chunk boundaries on every invocation.
    for (int generation = 0; generation < 3; ++generation) {
        fill(input, generation);
        output.fill_fourier_grid(ComplexType(-987, 654));
        plan.apply(input,
                   output,
                   generation == 0 ? sizeof(ComplexType) * (3 + FML::ThisTask) + 1 :
                   generation == 1 ? sizeof(ComplexType) :
                                     64 * 1024 * 1024);
        bool good = matches(output, oldmesh, generation);
        for (auto i : input.get_fourier_range())
            good &= input.get_fourier_from_index(i) == value(wave_at(input, i), generation);
        ptrdiff_t plane = newmesh / 2 + 1;
        for (int d = 0; d < N - 2; ++d)
            plane *= newmesh;
        good &= output.get_fourier_grid()[-plane] == ComplexType(-987, 654) &&
                output.get_fourier_grid()[output.get_local_nx() * plane] == ComplexType(-987, 654);
        check(good,
              std::to_string(N) + "D " + std::to_string(oldmesh) + "->" + std::to_string(newmesh) +
                  " exact coefficients/zeros, input and halos preserved, generation " + std::to_string(generation));
    }
    auto resized = resize_fourier_grid(input, newmesh, 2, 1);
    check(matches(resized, oldmesh, 2) && resized.get_n_extra_slices_left() == 2 &&
              resized.get_n_extra_slices_right() == 1,
          "convenience resize retains output halo allocation and normalization");
    bool rejected = false;
    try {
        plan.apply(input, output, FML::ThisTask == 0 ? sizeof(ComplexType) - 1 : INT_MAX);
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    check(rejected && matches(output, oldmesh, 2), "one-rank undersized cap rejected collectively before mutation");
    FFTWGrid<N> wrong(newmesh + FML::NTasks);
    rejected = false;
    try {
        plan.apply(input, FML::ThisTask == 0 ? wrong : output);
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    check(rejected, "one-rank incompatible mesh rejects plan reuse collectively");
    FourierResamplingPlan<N> same(input, input);
    rejected = false;
    try {
        same.apply(input, input);
    } catch (const std::overflow_error &) {
        rejected = true;
    }
    bool unchanged = true;
    for (auto i : input.get_fourier_range())
        unchanged &= input.get_fourier_from_index(i) == value(wave_at(input, i), 2);
    check(rejected && unchanged, "aliased input/output rejected before zeroing input");
}
int main() {
    const int mesh = FML::NTasks == 3 ? 6 : 8;
    test<2>(mesh, 2 * mesh);
    test<2>(2 * mesh, mesh);
    test<3>(mesh, mesh);
    test<3>(mesh, 3 * mesh);
    test<3>(3 * mesh, mesh);
    test<4>(mesh, mesh);
    if (FML::NTasks == 1 || FML::NTasks == 3) {
        test<2>(9, 15);
        test<3>(15, 9);
        test<3>(9, 12);
        test<3>(12, 9);
    }
    if (!FML::ThisTask)
        std::cout << (failures ? "Fourier resampling tests FAILED\n" : "All Fourier resampling tests passed\n");
    return failures ? 1 : 0;
}
