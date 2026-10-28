//==========================================================
// Tests for FML::NBODY (NBody/NBody.h) and FML::PGD (PGD/PotentialGradientDecent.h):
//   - the discrete 4-point Green's function is minus one over the square of the 4-point gradient (it had the wrong
//     sign of the sin^2 term) and the Fourier force on a plane wave with the 4-point kernels is 1/D4(k)
//   - the finite difference force uses the chosen Green's function and deconvolution (it always used -1/k^2 and
//     never deconvolved): with stencil 2 (4) it equals the Fourier force with the 2-point (4-point) gradient
//   - DriftParticles with tasks without particles (deadlocked: the empty tasks returned before the collectives)
//   - the periodic wrap after the drift never gives 1.0 and handles displacements larger than the box
//   - PGD with tasks without particles (aborted) and the size of the PGD displacement
//   - NBody.h and PotentialGradientDecent.h can be included in several translation units (test_nbody_odr.cpp)
//   - NBodyInitialConditions applies the optional scale-dependent 2LPT velocity norm(k) at the k of the 2LPT field
//   - the force of the reflection of a density field is the reflection of the force on an even mesh (the continuous
//     gradient used +k_Ny for the Nyquist modes)
// Bad input that must stop with a clear message:
//   ./test_nbody fdslices : potential with fewer than stencil_order/2 extra slices (read out of bounds before)
//   ./test_nbody stencil  : stencil order that is not implemented (threw inside an OpenMP region: std::terminate)
// Run with 1, 2 and 4 MPI tasks (OMP_NUM_THREADS=2)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/PGD/PotentialGradientDecent.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace FML;
using FFTWGrid = FML::GRID::FFTWGrid<3>;
using FloatType = FML::GRID::FloatType;
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;

int nbody_odr_check(); // defined in test_nbody_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(6);
    os << x;
    return os.str();
}

struct Particle {
    double pos[3];
    double vel[3];
    double expected[3]; // Where the particle should end up
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return 3; }
};

// Deterministic white noise as a function of the (global) cell (the same for any number of tasks)
double noise(std::array<double, 3> & x, int Nmesh) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 3; i++) {
        h ^= uint64_t(std::lround(x[i] * Nmesh)) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= h >> 31;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
    }
    return double(h >> 11) / double(1ULL << 53) - 0.5;
}

// Deterministic uniform numbers in [0,1)
double uniform(uint64_t i) {
    uint64_t z = i * 0x9E3779B97F4A7C15ULL + 12345;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return double(z >> 11) / double(1ULL << 53);
}

NBODY::ForceOptions force_options(int laplace, int gradient, bool deconvolve) {
    return {static_cast<NBODY::GreensFunctionLaplaceOperatorKernels>(laplace),
            static_cast<NBODY::GradientKernels>(gradient),
            deconvolve};
}

// max |a - b| / max |b| over all components and tasks
double rel_diff(std::array<FFTWGrid, 3> & a, std::array<FFTWGrid, 3> & b) {
    double maxdiff = 0.0, maxval = 0.0;
    for (int idim = 0; idim < 3; idim++)
        for (auto && i : b[idim].get_real_range()) {
            maxdiff =
                std::max(maxdiff, double(std::fabs(a[idim].get_real_from_index(i) - b[idim].get_real_from_index(i))));
            maxval = std::max(maxval, double(std::fabs(b[idim].get_real_from_index(i))));
        }
    FML::MaxOverTasks(&maxdiff);
    FML::MaxOverTasks(&maxval);
    return maxval > 0.0 ? maxdiff / maxval : 1e100;
}

// Reference PGD pipeline with the previous buffered interpolation/update. Keep the
// stored FloatType forces so exact comparisons detect changes at precision boundaries.
template <int N, class T>
double buffered_pgd(std::vector<T> & particles, const std::string & method, double alpha) {
    using Grid = FML::GRID::FFTWGrid<N>;
    size_t total = particles.size();
    FML::SumOverTasks(&total);
    const auto halo = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(method);
    Grid density(24, halo.first, halo.second);
    FML::INTERPOLATION::particles_to_fourier_grid<N, T>(particles.data(),
                                                        particles.size(),
                                                        total,
                                                        density,
                                                        method,
                                                        false);
    std::function<double(double)> filter = [](double k2) {
        const double high = k2 / 10000.0 < 10.0 ? k2 / 10000.0 : 10.0;
        const double low = k2 / 4.0 < 0.01 ? 100.0 : 4.0 / k2;
        return std::exp(-high * high - low);
    };
    FML::GRID::custom_smoothing_filter_fourier_space(density, filter);
    std::array<Grid, N> force_grid;
    NBODY::compute_force_from_density_fourier<N>(density, force_grid, method, 1.5);
    density.free();
    for (auto & grid : force_grid)
        grid.communicate_boundaries();
    std::array<std::vector<FloatType>, N> force;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N, T>(force_grid,
                                                                            particles.data(),
                                                                            particles.size(),
                                                                            force,
                                                                            method);
    for (auto & grid : force_grid)
        grid.free();
    double maximum = 0;
    for (size_t i = 0; i < particles.size(); ++i)
        for (int d = 0; d < N; ++d) {
            const double disp = -force[d][i] * alpha;
            maximum = std::max(maximum, std::abs(disp));
            particles[i].pos[d] += disp;
            NBODY::periodic_wrap_unit_box(particles[i].pos[d]);
        }
    FML::MaxOverTasks(&maximum);
    return maximum;
}

template <int N, class Position>
struct PGDParticle {
    Position pos[N];
    double vel[N];
    uint64_t id;
    Position * get_pos() { return pos; }
    constexpr int get_ndim() const { return N; }
};

template <int N, class Position>
void check_streamed_pgd(const std::string & label) {
    using P = PGDParticle<N, Position>;
    // Include a completely empty task, exact slab edges, and neighboring floating-point values.
    std::vector<P> original(FML::NTasks > 1 && FML::ThisTask == 0 ? 0 : 512);
    for (size_t i = 0; i < original.size(); ++i) {
        for (int d = 0; d < N; ++d) {
            const Position lo = d == 0 ? Position(FML::xmin_domain) : Position(0);
            const Position hi = d == 0 ? Position(FML::xmax_domain) : Position(1);
            Position x = Position(double(lo) + (double(hi) - lo) * uniform(31 * i + d + 7));
            if (i % 7 == 0)
                x = lo;
            if (i % 7 == 1 || x >= hi)
                x = std::nextafter(hi, lo);
            original[i].pos[d] = x;
            original[i].vel[d] = uniform(19 * i + d);
        }
        original[i].id = i + 12345;
    }
    bool positions = true, attributes = true, diagnostics = true, wrapped = true;
    bool large_displacement = false;
    for (const std::string method : {"NGP", "CIC", "TSC", "PCS", "PQS"})
        for (const double alpha : {0.0, 0.03, -0.02, 10000.0}) {
            auto expected = original, actual = original;
#ifdef USE_OMP
            if (omp_get_max_threads() > 1) {
                // Thin slabs use atomic deposition. Two nonoverlapping clouds, each
                // with identical positions, make its sums repeatable without disabling
                // threading in the interpolation/update under test.
                for (size_t i = 0; i < expected.size(); ++i) {
                    for (int d = 0; d < N; ++d)
                        expected[i].pos[d] = original.front().pos[d];
                    expected[i].pos[0] = alpha == 0.03 ?
                                             std::nextafter(Position(FML::xmax_domain), Position(FML::xmin_domain)) :
                                             Position(FML::xmin_domain);
                    expected[i].pos[1] = i < expected.size() / 2 ? Position(0.2) : Position(0.6);
                }
                actual = expected;
            }
#endif
            const double maximum = buffered_pgd<N>(expected, method, alpha);
            large_displacement = large_displacement || maximum > 1;
            std::ostringstream output, reference;
            reference.copyfmt(std::cout);
            if (FML::ThisTask == 0)
                reference << "[potential_gradient_decent_method] Maximum displacement : " << maximum << " = "
                          << maximum * 24 << " grid-cells\n";
            auto * previous = std::cout.rdbuf(output.rdbuf());
            PGD::potential_gradient_decent_method<N,
                                                  P>(actual.data(), actual.size(), 2.0, 100.0, alpha, 1.5, 24, method);
            std::cout.rdbuf(previous);
            const std::string prefix = "[potential_gradient_decent_method] Maximum displacement : ";
            const auto start = output.str().find(prefix);
            diagnostics = diagnostics && (FML::ThisTask == 0 ? start != std::string::npos &&
                                                                   output.str().substr(start) == reference.str() :
                                                               output.str().find(prefix) == std::string::npos);
            for (size_t i = 0; i < actual.size(); ++i) {
                positions = positions && std::memcmp(actual[i].pos, expected[i].pos, sizeof(actual[i].pos)) == 0;
                attributes = attributes && actual[i].id == original[i].id &&
                             std::memcmp(actual[i].vel, original[i].vel, sizeof(actual[i].vel)) == 0;
                for (int d = 0; d < N; ++d)
                    wrapped = wrapped && actual[i].pos[d] >= 0 && actual[i].pos[d] < 1;
            }
        }
    int bad[] = {int(!positions), int(!attributes), int(!diagnostics), int(!wrapped), int(!large_displacement)};
    for (auto & value : bad)
        FML::MaxOverTasks(&value);
    check(!bad[0], label + ": PGD positions match buffered updates bit-for-bit for all assignment methods");
    check(!bad[1], label + ": PGD preserves velocities and IDs");
    check(!bad[2], label + ": PGD maximum-displacement diagnostic is unchanged");
    check(!bad[3] && !bad[4], label + ": PGD wraps displacements larger than a box");
}

void check_streamed_pgd_suite() {
    check_streamed_pgd<2, double>("2D double");
    check_streamed_pgd<3, double>("3D double");
    check_streamed_pgd<2, float>("2D float");
    check_streamed_pgd<3, float>("3D float");
}

void test_explicit_force_options() {
    FFTWGrid density(16);
    density.fill_real_grid([](const std::array<double, 3> & x) {
        return std::cos(2 * M_PI * 3 * x[0]) + 0.3 * std::cos(2 * M_PI * 5 * x[1]);
    });
    FFTWGrid real_density = density;
    density.fftw_r2c();
    const NBODY::ForceOptions standard{};
    const auto different =
        force_options(NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT, NBODY::DISCRETE_FOUR_POINT_GRADIENT, true);
    std::array<FFTWGrid, 3> first, second, repeated, defaults, from_real;
    NBODY::compute_force_from_density_fourier<3>(density, first, "CIC", 1.5, standard);
    NBODY::compute_force_from_density_fourier<3>(density, second, "CIC", 1.5, different);
    NBODY::compute_force_from_density_fourier<3>(density, repeated, "CIC", 1.5, standard);
    NBODY::compute_force_from_density_fourier<3>(density, defaults, "CIC", 1.5);
    NBODY::compute_force_from_density_real<3>(real_density, from_real, "CIC", 1.5, different);
    check(rel_diff(first, repeated) == 0 && rel_diff(first, defaults) == 0,
          "interleaved explicit force choices preserve independent calls and fixed defaults");
    check(rel_diff(first, second) > 1e-3, "explicit force options change the selected discretization");
    check(rel_diff(second, from_real) < 1e-12, "real-density wrapper forwards every force option");
    check(NBODY::laplace_kernel_from_name("fiducial") == standard.laplace_kernel &&
              NBODY::gradient_kernel_from_name("") == standard.gradient_kernel &&
              NBODY::laplace_kernel_from_name("discrete_2pt") == different.laplace_kernel &&
              NBODY::gradient_kernel_from_name("discrete_4pt") == different.gradient_kernel,
          "kernel names parse into typed choices with deterministic fiducial defaults");
    bool invalid_name = false, invalid_value = false;
    try {
        NBODY::gradient_kernel_from_name("missing");
    } catch (const std::exception &) {
        invalid_name = true;
    }
    try {
        auto invalid = standard;
        invalid.laplace_kernel = static_cast<NBODY::GreensFunctionLaplaceOperatorKernels>(99);
        NBODY::compute_force_from_density_fourier<3>(density, repeated, "CIC", 1.5, invalid);
    } catch (const std::exception &) {
        invalid_value = true;
    }
    check(invalid_name && invalid_value, "invalid force options are rejected before parallel force work");
}

//==========================================================
// Nyquist modes of the force: on an even mesh the Nyquist component n_i = Nmesh/2 is its own mirror image
// n_i = -Nmesh/2, so the gradient i k_i of it depends on the sign the code picks for k_i. The force of the
// reflection x_a -> -x_a of a density field must be the reflection of the force (the a-component odd, the others
// even), which holds on the grid only if the Nyquist component of the gradient is zero. The input is white noise
// (full power at the Nyquist frequency). Returns max |F_refl(x) - P F(R x)| / max |F| over all components/axes.
//==========================================================
template <int N>
std::vector<double> gather_all_cells(const FML::GRID::FFTWGrid<N> & g) {
    const int n = g.get_nmesh();
    std::vector<double> out(size_t(FML::power(n, N)), 0.0);
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && ri : g.get_real_range(islice, islice + 1)) {
            auto c = g.get_coord_from_index(ri);
            c[0] += g.get_local_x_start();
            size_t at = 0;
            for (int idim = 0; idim < N; idim++)
                at = at * n + c[idim];
            out[at] = g.get_real_from_index(ri);
        }
    FML::SumArrayOverTasks(out.data(), int(out.size()));
    return out;
}

template <int N>
double force_reflection_error(int Nmesh, const NBODY::ForceOptions & options, bool finite_difference = false) {
    auto hash = [](const std::array<int, N> & c) {
        uint64_t h = 0x9E3779B97F4A7C15ULL;
        for (int idim = 0; idim < N; idim++) {
            h ^= uint64_t(c[idim]) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
            h *= 0xBF58476D1CE4E5B9ULL;
            h ^= h >> 29;
        }
        return double(h >> 11) / double(1ULL << 53) - 0.5;
    };
    // The force of the density field reflected in axis (-1: not reflected), gathered on all tasks
    auto compute = [&](int axis) {
        FML::GRID::FFTWGrid<N> density(Nmesh, 2, 2);
        for (int islice = 0; islice < density.get_local_nx(); islice++)
            for (auto && ri : density.get_real_range(islice, islice + 1)) {
                auto c = density.get_coord_from_index(ri);
                c[0] += density.get_local_x_start();
                if (axis >= 0)
                    c[axis] = (Nmesh - c[axis]) % Nmesh;
                density.set_real_from_index(ri, hash(c));
            }
        density.fftw_r2c();
        std::array<FML::GRID::FFTWGrid<N>, N> force;
        if (finite_difference) {
            FML::GRID::FFTWGrid<N> potential;
            NBODY::compute_potential_real_from_density_fourier<N>(density, potential, 1.5, 2, options);
            NBODY::compute_force_from_potential_real<N>(potential, force, "CIC", 4, options);
        } else {
            NBODY::compute_force_from_density_fourier<N>(density, force, "CIC", 1.5, options);
        }
        std::array<std::vector<double>, N> out;
        for (int idim = 0; idim < N; idim++)
            out[idim] = gather_all_cells<N>(force[idim]);
        return out;
    };
    const auto original = compute(-1);
    const size_t ncells = size_t(FML::power(Nmesh, N));
    double maxval = 0.0, maxerr = 0.0;
    for (int axis = 0; axis < N; axis++) {
        const auto reflected = compute(axis);
        for (size_t at = 0; at < ncells; at++) {
            // The index of the mirror cell
            std::array<int, N> c;
            size_t rest = at;
            for (int idim = N - 1; idim >= 0; idim--) {
                c[idim] = int(rest % Nmesh);
                rest /= Nmesh;
            }
            c[axis] = (Nmesh - c[axis]) % Nmesh;
            size_t mirror = 0;
            for (int idim = 0; idim < N; idim++)
                mirror = mirror * Nmesh + c[idim];
            for (int idim = 0; idim < N; idim++) {
                const double sign = idim == axis ? -1.0 : 1.0;
                maxval = std::max(maxval, std::fabs(original[idim][at]));
                maxerr = std::max(maxerr, std::fabs(reflected[idim][at] - sign * original[idim][mirror]));
            }
        }
    }
    return maxerr / maxval;
}

int main(int argc, char ** argv) {
    std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "streamed_pgd") {
        check_streamed_pgd_suite();
        return fails == 0 ? 0 : 1;
    }
    const int Nmesh = 16;
    const double dx = 1.0 / Nmesh;

    // A density field (white noise, i.e. all modes) in Fourier space
    FFTWGrid density(Nmesh, 1, 1);
    density.fill_real_grid([&](std::array<double, 3> & x) -> FloatType { return noise(x, Nmesh); });
    density.fftw_r2c();

    test_explicit_force_options();

    //=====================================================
    // Bad input: fewer extra slices than the stencil needs
    //=====================================================
    if (mode == "fdslices") {
        FFTWGrid potential;
        std::array<FFTWGrid, 3> force;
        NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.0, 1);
        NBODY::compute_force_from_potential_real<3>(potential, force, "CIC", 4);
        if (FML::ThisTask == 0)
            std::cout << "[FAIL] fdslices: stencil_order = 4 with 1 extra slice did not give an error\n";
        return 1;
    }

    //=====================================================
    // Bad input: a stencil order that is not implemented
    //=====================================================
    if (mode == "stencil") {
        FFTWGrid potential;
        std::array<FFTWGrid, 3> force;
        NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.0, 3);
        NBODY::compute_force_from_potential_real<3>(potential, force, "CIC", 3);
        if (FML::ThisTask == 0)
            std::cout << "[FAIL] stencil: stencil_order = 3 did not give an error\n";
        return 1;
    }

    if (mode != "all") {
        if (FML::ThisTask == 0)
            std::cout << "Unknown mode " << mode << " (all, fdslices, stencil)\n";
        return 1;
    }

    check(nbody_odr_check() == 1, "NBody.h and PotentialGradientDecent.h can be included in several translation units");

    //=====================================================
    // 1. The 4-point Green's function is -1/D4^2
    //=====================================================
    {
        double maxerr = 0.0;
        const std::vector<std::array<int, 3>> modes{{1, 0, 0},
                                                    {3, 2, 1},
                                                    {5, -7, 2},
                                                    {8, 8, -3},
                                                    {-6, 1, 4},
                                                    {7, 0, -3}};
        for (auto & n : modes) {
            std::array<double, 3> kvec;
            double kmag2 = 0.0, sumD2 = 0.0;
            for (int i = 0; i < 3; i++) {
                kvec[i] = 2.0 * M_PI * n[i];
                kmag2 += kvec[i] * kvec[i];
                const double D = NBODY::gradient_kernel_fourier(kvec[i], Nmesh, NBODY::DISCRETE_FOUR_POINT_GRADIENT);
                sumD2 += D * D;
            }
            const double G =
                NBODY::greens_function_laplace_operator_fourier<3>(kmag2,
                                                                   kvec,
                                                                   Nmesh,
                                                                   NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT);
            maxerr = std::max(maxerr, std::fabs(G * sumD2 + 1.0));
        }
        check(maxerr < 1e-12,
              "discrete_4pt Green's function = -1 / (4-point gradient)^2 (max error " + num(maxerr) + ")");
        // The 4-point gradient vanishes when all components are 0 or the Nyquist frequency: no (infinite) potential
        std::array<double, 3> knyq{M_PI * Nmesh, 0.0, -M_PI * Nmesh};
        const double Gnyq =
            NBODY::greens_function_laplace_operator_fourier<3>(2.0 * M_PI * M_PI * Nmesh * Nmesh,
                                                               knyq,
                                                               Nmesh,
                                                               NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT);
        check(Gnyq == 0.0,
              "discrete_4pt Green's function is 0 for the Nyquist modes where D4 = 0 (got " + num(Gnyq) + ")");
    }

    //=====================================================
    // 2. Fourier force on a plane wave with the 4-point kernels: F = sin(kx) / D4(k)
    //=====================================================
    {
        const auto options =
            force_options(NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT, NBODY::DISCRETE_FOUR_POINT_GRADIENT, false);
        const int n = 3;
        const double k = 2.0 * M_PI * n;
        FFTWGrid wave(Nmesh, 1, 1);
        wave.fill_real_grid([&](std::array<double, 3> & x) -> FloatType { return std::cos(k * x[1]); });
        wave.fftw_r2c();
        std::array<FFTWGrid, 3> force;
        NBODY::compute_force_from_density_fourier<3>(wave, force, "CIC", 1.0, options);
        const double D4 = (8.0 * std::sin(k * dx) - std::sin(2.0 * k * dx)) / (6.0 * dx);
        double maxerr = 0.0;
        for (auto && i : force[1].get_real_range()) {
            auto x = force[1].get_real_position(force[1].get_coord_from_index(i));
            maxerr = std::max(maxerr, std::fabs(force[1].get_real_from_index(i) - std::sin(k * x[1]) / D4) * D4);
            maxerr = std::max(maxerr, double(std::fabs(force[0].get_real_from_index(i))) * D4);
        }
        FML::MaxOverTasks(&maxerr);
        check(maxerr < 1e-10,
              "Fourier force on a plane wave with the 4-point kernels is sin(kx)/D4(k) (rel. error " + num(maxerr) +
                  ")");
    }

    //=====================================================
    // 3. The finite difference force uses the chosen Green's function and deconvolution
    // A stencil of order 2 (4) is the same as the Fourier 2-point (4-point) gradient for all modes
    //=====================================================
    {
        struct Case {
            int laplace;
            int stencil;
            int gradient;
            bool deconvolve;
            std::string name;
        };
        const std::vector<Case> cases{
            {NBODY::CONTINUOUS_GREENS_FUNCTION, 2, NBODY::DISCRETE_TWO_POINT_GRADIENT, false, "continuous, stencil 2"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT,
             2,
             NBODY::DISCRETE_TWO_POINT_GRADIENT,
             false,
             "discrete_2pt, stencil 2"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT,
             4,
             NBODY::DISCRETE_FOUR_POINT_GRADIENT,
             false,
             "discrete_4pt, stencil 4"},
            {NBODY::CONTINUOUS_GREENS_FUNCTION,
             4,
             NBODY::DISCRETE_FOUR_POINT_GRADIENT,
             true,
             "continuous, stencil 4, deconvolved"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT,
             4,
             NBODY::DISCRETE_FOUR_POINT_GRADIENT,
             true,
             "discrete_4pt, stencil 4, deconvolved"},
        };
        for (auto & c : cases) {
            const auto options = force_options(c.laplace, c.gradient, c.deconvolve);
            std::array<FFTWGrid, 3> force_fourier;
            NBODY::compute_force_from_density_fourier<3>(density, force_fourier, "CIC", 1.5, options);
            FFTWGrid potential;
            std::array<FFTWGrid, 3> force_fd;
            NBODY::compute_potential_real_from_density_fourier<3>(density, potential, 1.5, c.stencil / 2, options);
            NBODY::compute_force_from_potential_real<3>(potential, force_fd, "CIC", c.stencil, options);
            const double err = rel_diff(force_fd, force_fourier);
            check(err < 1e-10, "finite difference force = Fourier force (" + c.name + ", rel. diff " + num(err) + ")");
        }
    }

    //=====================================================
    // 4. Periodic wrap of the drift (pointer version, collective)
    //=====================================================
    {
        // pos, displacement
        const std::vector<std::array<double, 2>> cases{{0.0, -1e-18},
                                                       {0.3, 2.5},
                                                       {0.5, -1.7},
                                                       {0.9, 0.05},
                                                       {0.05, -0.1},
                                                       {0.25, 3.0},
                                                       {0.7, -2.7},
                                                       {0.4, 0.5}};
        std::vector<Particle> p(cases.size());
        for (size_t i = 0; i < cases.size(); i++)
            for (int idim = 0; idim < 3; idim++) {
                p[i].pos[idim] = cases[i][0];
                p[i].vel[idim] = cases[i][1];
            }
        NBODY::DriftParticles<3, Particle>(p.data(), p.size(), 1.0, true);
        bool in_box = true, correct = true, same_as_before = true;
        for (size_t i = 0; i < cases.size(); i++) {
            double x = cases[i][0] + cases[i][1];
            const double expected = x - std::floor(x) >= 1.0 ? 0.0 : x - std::floor(x);
            // The old code (one wrap) for positions in [-1, 2) that do not round to 1.0
            double xold = x;
            if (xold >= 1.0)
                xold -= 1.0;
            if (xold < 0.0)
                xold += 1.0;
            for (int idim = 0; idim < 3; idim++) {
                in_box = in_box and p[i].pos[idim] >= 0.0 and p[i].pos[idim] < 1.0;
                correct = correct and std::fabs(p[i].pos[idim] - expected) < 1e-14;
                if (x >= -1.0 and x < 2.0 and xold < 1.0)
                    same_as_before = same_as_before and p[i].pos[idim] == xold;
            }
        }
        check(in_box, "drift: positions are in [0,1) (never exactly 1.0)");
        check(correct, "drift: displacements larger than the box are wrapped");
        check(same_as_before, "drift: the usual positions are wrapped exactly as before (bit-identical)");
    }

    //=====================================================
    // 5. DriftParticles with tasks without particles: the particles are all on task 0
    //=====================================================
    {
        const int npart = 64;
        std::vector<Particle> part(npart);
        for (int i = 0; i < npart; i++) {
            for (int idim = 0; idim < 3; idim++) {
                part[i].pos[idim] = uniform(6 * i + idim);
                part[i].vel[idim] = (uniform(6 * i + 3 + idim) - 0.5) * 0.9; // less than half a box
            }
            part[i].pos[0] *= 0.1; // In the domain of task 0 for up to 10 tasks
            for (int idim = 0; idim < 3; idim++) {
                const double x = part[i].pos[idim] + part[i].vel[idim];
                part[i].expected[idim] = x - std::floor(x);
            }
        }
        MPIParticles<Particle> mp;
        mp.create(part.data(), npart, 2 * npart, FML::xmin_domain, FML::xmax_domain, true);
        NBODY::DriftParticles<3, Particle>(mp, 1.0, true);
        double maxerr = 0.0;
        int outside = 0;
        for (auto & p : mp) {
            if (p.pos[0] < FML::xmin_domain or p.pos[0] >= FML::xmax_domain)
                outside++;
            for (int idim = 0; idim < 3; idim++) {
                double d = std::fabs(p.pos[idim] - p.expected[idim]);
                maxerr = std::max(maxerr, std::min(d, 1.0 - d));
                if (p.pos[idim] < 0.0 or p.pos[idim] >= 1.0)
                    outside++;
            }
        }
        FML::MaxOverTasks(&maxerr);
        FML::SumOverTasks(&outside);
        check(mp.get_npart_total() == size_t(npart) and outside == 0 and maxerr < 1e-12,
              "DriftParticles with tasks without particles (npart_total = " + std::to_string(mp.get_npart_total()) +
                  ", outside the domain " + std::to_string(outside) + ", max error " + num(maxerr) + ")");
    }

    //=====================================================
    // 6. PGD with tasks without particles
    //=====================================================
    {
        const int npart = 200;
        std::vector<Particle> part(npart);
        for (int i = 0; i < npart; i++) {
            for (int idim = 0; idim < 3; idim++) {
                part[i].pos[idim] = uniform(3 * i + idim + 1000);
                part[i].vel[idim] = 0.0;
            }
            part[i].pos[0] *= 0.1; // All on task 0 (a big overdensity)
        }
        MPIParticles<Particle> mp;
        mp.create(part.data(), npart, 2 * npart, FML::xmin_domain, FML::xmax_domain, true);
        PGD::potential_gradient_decent_method<3, Particle>(mp.get_particles_ptr(),
                                                           mp.get_npart(),
                                                           2.0,
                                                           100.0,
                                                           0.01,
                                                           1.5,
                                                           Nmesh,
                                                           "CIC");
        mp.communicate_particles();
        double xsum = 0.0;
        int bad = 0;
        for (auto & p : mp)
            for (int idim = 0; idim < 3; idim++) {
                if (not std::isfinite(p.pos[idim]) or p.pos[idim] < 0.0 or p.pos[idim] >= 1.0)
                    bad++;
                xsum += p.pos[idim];
            }
        FML::SumOverTasks(&bad);
        FML::SumOverTasks(&xsum);
        check(mp.get_npart_total() == size_t(npart) and bad == 0 and std::isfinite(xsum),
              "PGD with tasks without particles (npart_total = " + std::to_string(mp.get_npart_total()) + ")");
    }

    //=====================================================
    // 7. PGD displacement S = -alpha * norm * grad O grad^-2 delta for a lattice displaced by a sine wave
    //=====================================================
    {
        const int Np = 16, n = 2;
        const double A = 0.01, k = 2.0 * M_PI * n, alpha = 0.01, norm = 1.5, kl = 1.0, ks = 1000.0;
        MPIParticles<Particle> mp;
        mp.create_particle_grid(Np, 2.0, FML::xmin_domain, FML::xmax_domain);
        for (auto & p : mp) {
            for (int idim = 0; idim < 3; idim++)
                p.vel[idim] = 0.0;
            const double q = p.pos[0];
            p.pos[0] = q - A / k * std::sin(k * q); // delta = A cos(kq) to linear order
            if (p.pos[0] < 0.0)
                p.pos[0] += 1.0;
            p.expected[0] = p.pos[0];
        }
        mp.communicate_particles();
        PGD::potential_gradient_decent_method<3, Particle>(mp.get_particles_ptr(),
                                                           mp.get_npart(),
                                                           kl,
                                                           ks,
                                                           alpha,
                                                           norm,
                                                           Np,
                                                           "CIC");
        const double filter = std::exp(-std::pow(k * k / (ks * ks), 2) - kl * kl / (k * k));
        const double W = std::pow(std::sin(k / (2.0 * Np)) / (k / (2.0 * Np)), 4); // CIC assignment + interpolation
        double num_ = 0.0, den = 0.0;
        for (auto & p : mp) {
            double s = p.pos[0] - p.expected[0];
            s -= std::round(s);
            const double e = -alpha * norm * A / k * std::sin(k * p.expected[0]) * filter * W;
            num_ += s * e;
            den += e * e;
        }
        FML::SumOverTasks(&num_);
        FML::SumOverTasks(&den);
        const double ratio = num_ / den;
        check(std::fabs(ratio - 1.0) < 0.02,
              "PGD displacement is -alpha norm grad grad^-2 delta (towards the overdensity), measured/expected = " +
                  num(ratio));
        mp.communicate_particles();
    }

    //=====================================================
    // 8. NBodyInitialConditions with a scale-dependent 2LPT velocity norm(k): v_2LPT = norm(k) Psi_2LPT(k) with k the
    // wavenumber of the 2LPT field. delta = A [cos(k1 x) + cos(k1 y) + cos(k2 z)]: the 2LPT source phi_ii phi_jj -
    // phi_ij^2 has the modes (k1, k1, 0) in the x-y plane and (k1, 0, k2), (0, k1, k2), so the z-velocity is
    // norm(sqrt(k1^2 + k2^2)) Psi_2LPT,z. Compared with the same IC with the constant norm velocity_norms[1]
    //=====================================================
    {
        const int Np = 16, n1 = 1, n2 = 3;
        const double A = 0.05, k1 = 2.0 * M_PI * n1, k2 = 2.0 * M_PI * n2;
        auto make_ic = [&](bool with_z, std::function<double(double)> norm2) {
            FFTWGrid delta(Np, 1, 1);
            delta.fill_real_grid([&](std::array<double, 3> & x) -> FloatType {
                return A * (std::cos(k1 * x[0]) + std::cos(k1 * x[1]) + (with_z ? std::cos(k2 * x[2]) : 0.0));
            });
            delta.fftw_r2c();
            MPIParticles<Particle> mp;
            std::vector<FFTWGrid> potentials;
            // No 1LPT velocity: the velocities are the 2LPT velocities
            const std::vector<double> norms{0.0, norm2 ? 0.0 : 1.0};
            NBODY::NBodyInitialConditions<3, Particle>(mp,
                                                       Np,
                                                       2.0,
                                                       delta,
                                                       potentials,
                                                       2,
                                                       1.0,
                                                       10.0,
                                                       norms,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       norm2);
            return mp;
        };
        auto g = [](double kBox) { return 1.0 + kBox / (2.0 * M_PI); };
        // Compare the velocities (component idim) of b with factor * those of a, particle by particle (the same order:
        // the same lattice and displacements on the same tasks)
        auto compare = [&](MPIParticles<Particle> & a, MPIParticles<Particle> & b, int idim, double factor) {
            double maxerr = 0.0, maxv = 0.0, maxdpos = 0.0;
            for (size_t i = 0; i < a.get_npart(); i++) {
                maxerr = std::max(maxerr, std::fabs(b[i].vel[idim] - factor * a[i].vel[idim]));
                maxv = std::max(maxv, std::fabs(a[i].vel[idim]));
                for (int j = 0; j < 3; j++)
                    maxdpos = std::max(maxdpos, std::fabs(b[i].pos[j] - a[i].pos[j]));
            }
            FML::MaxOverTasks(&maxerr);
            FML::MaxOverTasks(&maxv);
            FML::MaxOverTasks(&maxdpos);
            return std::array<double, 3>{maxerr / maxv, maxv, maxdpos};
        };
        // Two plane waves in x and y: one 2LPT shell |k| = sqrt(2) k1
        auto a = make_ic(false, nullptr);
        auto b = make_ic(false, g);
        const double expected_xy = g(std::sqrt(2.0) * k1);
        for (int idim = 0; idim < 2; idim++) {
            auto r = compare(a, b, idim, expected_xy);
            check(a.get_npart() == b.get_npart() and r[0] < 1e-5 and r[1] > 0.0 and r[2] == 0.0,
                  "NBodyInitialConditions: 2LPT velocity " + std::string(1, "xyz"[idim]) +
                      " = norm(k) Psi_2LPT at |k| = sqrt(2) k1 (rel. error " + num(r[0]) +
                      ", positions unchanged: max diff " + num(r[2]) + ")");
        }
        // With the third wave: the z-velocity comes from the shell |k| = sqrt(k1^2 + k2^2)
        auto c = make_ic(true, nullptr);
        auto d = make_ic(true, g);
        auto r = compare(c, d, 2, g(std::sqrt(k1 * k1 + k2 * k2)));
        check(r[0] < 1e-5 and r[1] > 0.0 and r[2] == 0.0,
              "NBodyInitialConditions: 2LPT z-velocity = norm(k) Psi_2LPT at |k| = sqrt(k1^2 + k2^2) (rel. error " +
                  num(r[0]) + ")");
        // A constant norm(k) is the same as velocity_norms[1]
        auto e = make_ic(true, [](double) { return 1.0; });
        double maxerr = 0.0;
        for (int idim = 0; idim < 3; idim++)
            maxerr = std::max(maxerr, compare(c, e, idim, 1.0)[0]);
        check(maxerr < 1e-6,
              "NBodyInitialConditions: a constant 2LPT velocity norm(k) = velocity_norms[1] (rel. diff " + num(maxerr) +
                  ")");
    }

    //=====================================================
    // 9. The force is covariant under reflections x_a -> -x_a also for the Nyquist modes of an even mesh (the
    // continuous gradient used +k_Ny for them, see force_reflection_error). The discrete gradients and the finite
    // difference force are (numerically) zero there already; an odd mesh has no Nyquist mode
    //=====================================================
    {
        struct Case {
            int laplace;
            int gradient;
            bool deconvolve;
            bool finite_difference;
            std::string name;
        };
        const std::vector<Case> cases{
            {NBODY::CONTINUOUS_GREENS_FUNCTION, NBODY::CONTINUOUS_GRADIENT, false, false, "default (continuous)"},
            {NBODY::CONTINUOUS_GREENS_FUNCTION, NBODY::CONTINUOUS_GRADIENT, true, false, "continuous, deconvolved"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_TWO_POINT,
             NBODY::CONTINUOUS_GRADIENT,
             false,
             false,
             "discrete_2pt Laplace, continuous gradient"},
            {NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT,
             NBODY::DISCRETE_FOUR_POINT_GRADIENT,
             false,
             false,
             "discrete_4pt"},
            {NBODY::CONTINUOUS_GREENS_FUNCTION,
             NBODY::CONTINUOUS_GRADIENT,
             false,
             true,
             "finite difference, stencil 4"},
        };
        for (auto & c : cases) {
            const auto options = force_options(c.laplace, c.gradient, c.deconvolve);
            const double err3 = force_reflection_error<3>(16, options, c.finite_difference);
            check(err3 < 1e-10,
                  "3D force (Nmesh = 16) is covariant under reflections (" + c.name + ", rel. err " + num(err3) + ")");
        }
        // The general (N != 3) loop of compute_force_from_density_fourier and an odd mesh
        const double err2 = force_reflection_error<2>(16, NBODY::ForceOptions{});
        check(err2 < 1e-10,
              "2D force (Nmesh = 16) is covariant under reflections (default, rel. err " + num(err2) + ")");
        // (the number of tasks must divide Nmesh)
        if (15 % FML::NTasks == 0) {
            const double errodd = force_reflection_error<3>(15, NBODY::ForceOptions{});
            check(errodd < 1e-10,
                  "3D force (Nmesh = 15) is covariant under reflections (default, rel. err " + num(errodd) + ")");
        }
    }

    check_streamed_pgd_suite();

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All NBody tests passed\n" : std::to_string(fails) + " NBody test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
