#ifndef FML_COUNTER_BASED_RANDOM_FIELD_H
#define FML_COUNTER_BASED_RANDOM_FIELD_H
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
namespace FML::RANDOM::COUNTER_BASED {
    inline uint32_t signed_counter(int wave) {
        return wave >= 0 ? uint32_t(wave) * 2u : uint32_t(-int64_t(wave) * 2 - 1);
    }
    template <int N>
    void generate_fourier(FML::GRID::FFTWGrid<N> & grid,
                          uint64_t seed,
                          uint32_t stream,
                          const std::function<double(double)> & power,
                          bool fixed) {
        static_assert(N >= 2 && N <= 3, "Philox wavevector mapping supports 2D and 3D");
        assert_mpi(grid.get_nmesh() > 0 && bool(power), "[Philox random field] Grid and spectrum must be initialized");
        const int mesh = grid.get_nmesh();
        int invalid = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(| : invalid)
#endif
        for (int slice = 0; slice < grid.get_local_nx(); slice++) {
            for (auto index : grid.get_fourier_range(slice, slice + 1)) {
                auto wave = grid.get_fourier_coord_from_index(index);
                wave[0] += grid.get_local_x_start();
                bool zero = true, nyquist = false;
                double norm2 = 0;
                for (int d = 0; d < N; d++) {
                    if (d < N - 1 && wave[d] > mesh / 2)
                        wave[d] -= mesh;
                    zero = zero && wave[d] == 0;
                    nyquist = nyquist || (mesh % 2 == 0 && std::abs(wave[d]) == mesh / 2);
                    norm2 += double(wave[d]) * wave[d];
                }
                if (zero || nyquist) {
                    grid.set_fourier_from_index(index, 0.0);
                    continue;
                }
                // First nonzero coordinate positive selects one representative of {k,-k}.
                bool conjugate = false;
                for (auto k : wave)
                    if (k) {
                        conjugate = k < 0;
                        break;
                    }
                std::array<uint32_t, 4> counter{0, 0, 0, stream};
                for (int d = 0; d < N; d++)
                    counter[d] = signed_counter(conjugate ? -wave[d] : wave[d]);
                const auto bits = philox4x32(counter, {uint32_t(seed), uint32_t(seed >> 32)});
                const double phase = 2 * M_PI * uniform_open(bits[0], bits[1]);
                const double p = power(2 * M_PI * std::sqrt(norm2));
                if (!std::isfinite(p) || p < 0) {
                    invalid = 1;
                    grid.set_fourier_from_index(index, 0.0);
                    continue;
                }
                const double amplitude = std::sqrt(p * (fixed ? 1.0 : -std::log(uniform_open(bits[2], bits[3]))));
                using Float = FML::GRID::FloatType;
                grid.set_fourier_from_index(
                    index,
                    std::complex<Float>(Float(amplitude * std::cos(phase)),
                                        Float((conjugate ? -1 : 1) * amplitude * std::sin(phase))));
            }
        }
        FML::MaxOverTasks(&invalid);
        assert_mpi(!invalid, "[Philox random field] Power spectrum must be nonnegative and finite");
        grid.set_grid_status_real(false);
    }
} // namespace FML::RANDOM::COUNTER_BASED
#endif
