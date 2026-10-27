// Offset deposition against the previous explicit-shift algorithm, including float rounding,
// periodic/slab boundaries, empty ranks, OpenMP chunking and particle position preservation.
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace I = FML::INTERPOLATION;
int failures = 0;
void check(bool ok, const std::string & what) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (FML::ThisTask == 0) std::cout << (bad ? "[FAIL] " : "[PASS] ") << what << '\n';
}

template <int N, class Position, int Weight = 0>
struct Particle {
    static constexpr int weight_kind = Weight;
    Position pos[N];
    double weight = 1;
    // Read-only access deliberately makes accidental writes in an interlacing routine a compile error.
    const Position * get_pos() const { return pos; }
    constexpr int get_ndim() const { return N; }
    template <int W = Weight, std::enable_if_t<W == 1, int> = 0>
    double get_mass() const { return weight; }
    template <int W = Weight, std::enable_if_t<W == 2, int> = 0>
    double get_deltaf_weight() const { return weight; }
};

template <int N, class P>
std::vector<P> particles(bool empty_rank) {
    std::mt19937 rng(234 + FML::ThisTask);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<P> p(empty_rank && FML::NTasks > 1 && FML::ThisTask == 0 ? 0 : 800);
    using Pos = std::remove_cv_t<std::remove_reference_t<decltype(P{}.pos[0])>>;
    for (size_t j = 0; j < p.size(); j++) {
        for (int d = 0; d < N; d++) {
            const Pos lo = d == 0 ? Pos(FML::xmin_domain) : Pos(0);
            const Pos hi = d == 0 ? Pos(FML::xmax_domain) : Pos(1);
            Pos x = Pos(double(lo) + (double(hi) - lo) * uniform(rng));
            if (j % 7 == 0) x = lo;
            if (j % 7 == 1) x = std::nextafter(hi, lo);
            if (x >= hi) x = std::nextafter(hi, lo);
            p[j].pos[d] = x;
        }
        p[j].weight = 0.1 + 0.001 * j;
        if constexpr (P::weight_kind == 2) p[j].weight -= 0.4;
    }
    return p;
}

template <int N, class P>
void explicit_shift(std::vector<P> & p, const std::array<double, N> & shift, bool wrap_x) {
    for (auto & particle : p)
        for (int d = 0; d < N; d++) {
            particle.pos[d] += shift[d];
            if (d > 0 || wrap_x)
                particle.pos[d] = FML::PARTICLE::wrap_coordinate_periodic(particle.pos[d]);
        }
}

template <int N>
double relative_difference(const FML::GRID::FFTWGrid<N> & a, const FML::GRID::FFTWGrid<N> & b, bool fourier) {
    double error = 0, scale = 0;
    if (fourier) {
        for (auto k : a.get_fourier_range()) {
            error = std::max(error, double(std::abs(a.get_fourier_from_index(k) - b.get_fourier_from_index(k))));
            scale = std::max(scale, double(std::abs(b.get_fourier_from_index(k))));
        }
    } else {
        for (auto k : a.get_real_range()) {
            error = std::max(error, std::abs(double(a.get_real_from_index(k) - b.get_real_from_index(k))));
            scale = std::max(scale, std::abs(double(b.get_real_from_index(k))));
        }
    }
    FML::MaxOverTasks(&error); FML::MaxOverTasks(&scale);
    return error / std::max(scale, 1e-30);
}

template <int N, class P>
void run_case(const std::string & label, bool empty_rank = false) {
    auto p = particles<N, P>(empty_rank);
    size_t total = p.size(); FML::SumOverTasks(&total);
    std::vector<unsigned char> original(p.size() * sizeof(P));
    if (!p.empty()) std::memcpy(original.data(), p.data(), original.size());
    const P * input = p.data();
    constexpr int nmesh = 24;
    // Use a wider slab for direct-deposition tests to exercise the non-atomic chunked path for every kernel.
    const int chunk_mesh = 48 * FML::NTasks;
    double worst_deposition = 0, worst_legacy = 0, worst_multi = 0;
    bool conserved = true;
    for (const std::string method : {"NGP", "CIC", "TSC", "PCS", "PQS"}) {
        auto halo = I::get_extra_slices_needed_for_density_assignment(method);
        std::array<double, N> offset; offset.fill(0.375 / chunk_mesh);
        FML::GRID::FFTWGrid<N> a(chunk_mesh, halo.first, halo.second + 1), b = a;
        auto shifted = p;
        explicit_shift<N>(shifted, offset, false);
        I::particles_to_grid<N, P>(input, p.size(), total, a, method, offset);
        I::particles_to_grid<N, P>(shifted.data(), shifted.size(), total, b, method);
        worst_deposition = std::max(worst_deposition, relative_difference(a, b, false));
        // A zero offset must take the exact original deposition path.
        I::particles_to_grid<N, P>(input, p.size(), total, a, method, std::array<double, N>{});
        I::particles_to_grid<N, P>(input, p.size(), total, b, method);
        check(relative_difference(a, b, false) == 0 || relative_difference(a, b, false) < 1e-13,
              label + " " + method + " zero offset agrees with ordinary deposition");

        // Reference for the old two-grid algorithm, including its original Nyquist phase convention.
        FML::GRID::FFTWGrid<N> reference(nmesh, halo.first, halo.second + 1), shifted_grid = reference;
        I::particles_to_grid<N, P>(input, p.size(), total, reference, method);
        offset.fill(0.5 / nmesh);
        shifted = p; explicit_shift<N>(shifted, offset, false);
        I::particles_to_grid<N, P>(shifted.data(), shifted.size(), total, shifted_grid, method);
        reference.fftw_r2c(); shifted_grid.fftw_r2c();
        for (auto k : reference.get_fourier_range()) {
            auto kv = reference.get_fourier_wavevector_from_index(k);
            double sum = kv[0]; for (int d = 1; d < N; d++) sum += kv[d];
            const auto phase = std::exp(std::complex<FML::GRID::FloatType>(0, 1) *
                                       FML::GRID::FloatType(sum * offset[0]));
            reference.set_fourier_from_index(k, (reference.get_fourier_from_index(k) +
                                                phase * shifted_grid.get_fourier_from_index(k)) /
                                                FML::GRID::FloatType(2));
        }
        FML::GRID::FFTWGrid<N> actual(nmesh);
        I::particles_to_fourier_grid<N, P>(input, p.size(), total, actual, method, true);
        worst_legacy = std::max(worst_legacy, relative_difference(actual, reference, true));

        // Reference for the old generalized algorithm: shift a copy, deposit and undo the Fourier phase.
        for (int order : {1, 2, 4, 8}) {
            reference.fill_fourier_grid(0.0); reference.set_grid_status_real(false);
            const auto shifts = I::generate_interlacing_shifts<N>(order, 1.0 / nmesh);
            for (const auto & shift : shifts) {
                shifted = p; explicit_shift<N>(shifted, shift, FML::NTasks == 1);
                I::particles_to_grid<N, P>(shifted.data(), shifted.size(), total, shifted_grid, method);
                shifted_grid.fftw_r2c();
                FML::GRID::shift_grid_fourier<N>(shifted_grid, shift, 1);
                for (auto k : reference.get_fourier_range())
                    reference.set_fourier_from_index(k, reference.get_fourier_from_index(k) +
                        shifted_grid.get_fourier_from_index(k) * FML::GRID::FloatType(1.0 / shifts.size()));
            }
            I::particles_to_fourier_grid_interlaced<N, P>(input, p.size(), total, actual, method, order);
            worst_multi = std::max(worst_multi, relative_difference(actual, reference, true));
        }
        // For number/mass particles the mean of delta is zero; delta-f particles carry their actual mean weight.
        double mean_weight = 0;
        if constexpr (FML::PARTICLE::has_get_deltaf_weight<P>()) {
            for (const auto & particle : p) mean_weight += particle.weight;
            FML::SumOverTasks(&mean_weight); mean_weight /= total;
        }
        double dc = FML::ThisTask == 0 ? double(std::abs(actual.get_fourier_from_index(0) - mean_weight)) : 0;
        FML::MaxOverTasks(&dc); conserved = conserved && dc < 1e-11;
    }
    check(worst_deposition < 2e-12, label + " shifted deposition matches explicit-shift reference");
    check(worst_legacy < 2e-12, label + " legacy two-grid Fourier field unchanged");
    check(worst_multi < 2e-12, label + " generalized 1/2/4/8-grid Fourier fields agree");
    check(conserved, label + " mass/weight is conserved");
    check(p.empty() || std::memcmp(original.data(), p.data(), original.size()) == 0,
          label + " all particle bytes preserved exactly");
    if (FML::ThisTask == 0)
        std::cout << "  max relative differences: deposition=" << worst_deposition
                  << " legacy=" << worst_legacy << " generalized=" << worst_multi << '\n';
}

int main(int argc, char ** argv) {
    static_assert(FML::PARTICLE::has_get_mass<Particle<3, double, 1>>());
    static_assert(!FML::PARTICLE::has_get_deltaf_weight<Particle<3, double, 1>>());
    static_assert(FML::PARTICLE::has_get_deltaf_weight<Particle<3, float, 2>>());
    static_assert(!FML::PARTICLE::has_get_mass<Particle<3, float, 2>>());
    const std::string mode = argc > 1 ? argv[1] : "";
    if (!mode.empty()) {
        using P = Particle<3, double>;
        auto p = particles<3, P>(false);
        auto halo = I::get_extra_slices_needed_for_density_assignment("CIC");
        FML::GRID::FFTWGrid<3> grid(24, halo.first, halo.second);
        std::array<double, 3> shift{}; shift[0] = mode == "negative" ? -0.1 : 0.5 / 24;
        I::particles_to_grid<3, P>(p.data(), p.size(), p.size() * FML::NTasks, grid, "CIC", shift);
        return 0;
    }
    run_case<2, Particle<2, double>>("2D double");
    run_case<3, Particle<3, double>>("3D double");
    run_case<3, Particle<3, float>>("3D float");
    run_case<3, Particle<3, double, 1>>("3D mass weighted", true);
    run_case<3, Particle<3, float, 2>>("3D delta-f weighted", true);
    if (FML::ThisTask == 0) std::cout << "Offset/interlacing failures: " << failures << '\n';
    return failures ? 1 : 0;
}
