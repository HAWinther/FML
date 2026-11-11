#ifndef FML_NUFFT_CORE_H
#define FML_NUFFT_CORE_H

#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace FML::NUFFT::Core {

    /// A coordinate or frequency vector. Coordinates use radians in the
    /// periodic transforms; type 3 frequencies are arbitrary real values.
    template <int N, class Real>
    using Point = std::array<Real, N>;

    /// Number of Fourier modes in each dimension, in x/y/z order.
    template <int N>
    using ModeShape = std::array<std::size_t, N>;

    enum class TransformType { Type1, Type2, Type3 };

    /// Shared controls for one-shot transforms and plans.
    struct Options {
        double tolerance{1e-6};
        int sign{-1};
        double upsampling_factor{2.0};
        int threads{0}; // 0 selects the backend default.

        void validate() const {
            if (!(tolerance > 0.0 && tolerance < 1.0))
                throw std::invalid_argument("[NUFFT::Core] tolerance must be in (0, 1)");
            if (sign != -1 && sign != 1)
                throw std::invalid_argument("[NUFFT::Core] sign must be -1 or +1");
            if (!(upsampling_factor > 1.0))
                throw std::invalid_argument("[NUFFT::Core] upsampling_factor must be greater than 1");
            if (threads < 0)
                throw std::invalid_argument("[NUFFT::Core] threads must be nonnegative");
        }
    };

    /// Full-complex Fourier coefficients in canonical signed-mode order.
    /// Axis 0 (x) is contiguous, matching the FINUFFT C++ storage convention.
    template <int N, class Real>
    class ModeArray {
        static_assert(N >= 1 && N <= 3, "NUFFT::Core supports dimensions 1 through 3");
        static_assert(std::is_floating_point_v<Real>, "NUFFT::Core real type must be floating point");

      public:
        using value_type = std::complex<Real>;
        using shape_type = ModeShape<N>;

        ModeArray() = default;
        explicit ModeArray(const shape_type & shape) { resize(shape); }

        void resize(const shape_type & shape) {
            std::size_t count = 1;
            for (const auto n : shape) {
                if (n == 0)
                    throw std::invalid_argument("[NUFFT::Core::ModeArray] mode counts must be positive");
                if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
                    throw std::length_error("[NUFFT::Core::ModeArray] mode count exceeds ptrdiff_t range");
                if (count > std::numeric_limits<std::size_t>::max() / n)
                    throw std::length_error("[NUFFT::Core::ModeArray] mode array size overflows size_t");
                count *= n;
            }
            modes = shape;
            coefficients.resize(count);
        }

        const shape_type & shape() const noexcept { return modes; }
        std::size_t size() const noexcept { return coefficients.size(); }
        bool empty() const noexcept { return coefficients.empty(); }
        value_type * data() noexcept { return coefficients.data(); }
        const value_type * data() const noexcept { return coefficients.data(); }
        value_type & operator[](std::size_t i) noexcept { return coefficients[i]; }
        const value_type & operator[](std::size_t i) const noexcept { return coefficients[i]; }
        auto begin() noexcept { return coefficients.begin(); }
        auto end() noexcept { return coefficients.end(); }
        auto begin() const noexcept { return coefficients.begin(); }
        auto end() const noexcept { return coefficients.end(); }

        /// Map signed mode numbers to a linear index. Signed modes range from
        /// -floor(N/2) to floor((N-1)/2), with x varying fastest.
        std::size_t index(const std::array<std::ptrdiff_t, N> & mode) const {
            std::size_t offset = 0;
            std::size_t stride = 1;
            for (int axis = 0; axis < N; axis++) {
                const auto n = static_cast<std::ptrdiff_t>(modes[axis]);
                const auto lo = -(n / 2);
                const auto hi = (n - 1) / 2;
                if (mode[axis] < lo || mode[axis] > hi)
                    throw std::out_of_range("[NUFFT::Core::ModeArray] signed mode is outside the array");
                offset += static_cast<std::size_t>(mode[axis] - lo) * stride;
                stride *= modes[axis];
            }
            return offset;
        }

        /// Recover signed mode numbers from a linear index.
        std::array<std::ptrdiff_t, N> mode(std::size_t offset) const {
            if (offset >= coefficients.size())
                throw std::out_of_range("[NUFFT::Core::ModeArray] coefficient index is outside the array");
            std::array<std::ptrdiff_t, N> result{};
            for (int axis = 0; axis < N; axis++) {
                const auto n = modes[axis];
                const auto coordinate = offset % n;
                offset /= n;
                result[axis] = static_cast<std::ptrdiff_t>(coordinate) - static_cast<std::ptrdiff_t>(n / 2);
            }
            return result;
        }

      private:
        shape_type modes{};
        std::vector<value_type> coefficients;
    };

} // namespace FML::NUFFT::Core

#endif // FML_NUFFT_CORE_H
