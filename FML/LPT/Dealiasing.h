#ifndef FML_LPT_DEALIASING_HEADER
#define FML_LPT_DEALIASING_HEADER

#include <FML/FFTWGrid/FourierResampling.h>
#include <climits>
#include <numeric>

namespace FML::COSMOLOGY::LPT {
    /// Padding evaluates nonlinear products on a larger mesh and crops the result.
    /// None preserves the original, aliased calculation. Padding excludes input and
    /// output Nyquist planes (for even meshes); all strictly sub-Nyquist modes remain.
    enum class Dealiasing { None, Padding };

    namespace detail {
        inline int padded_lpt_mesh(int nmesh, int order) {
            // M >= (order+1) N/2 prevents a degree-order product from wrapping
            // into |k_i| < N/2. Keep the FFT mesh even and divisible by MPI tasks.
            const long long multiple = std::lcm(2, FML::NTasks);
            const long long minimum = ((order + 1LL) * nmesh + 1) / 2;
            const long long padded = ((minimum + multiple - 1) / multiple) * multiple;
            assert_mpi(padded <= INT_MAX, "[LPT padding] Mesh size exceeds int range");
            return int(padded);
        }

    } // namespace detail
} // namespace FML::COSMOLOGY::LPT
#endif
