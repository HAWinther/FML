#ifndef FML_SHORT_RANGE_LONG_RANGE_PM_HEADER
#define FML_SHORT_RANGE_LONG_RANGE_PM_HEADER
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/ShortRangeGravity/ShortRangeGravity.h>

namespace FML::SHORT_RANGE_GRAVITY {
    /// Apply to a dedicated Fourier density/potential copy before the ordinary continuous-k PM force solve.
    /// FML wavevectors use box units; convert the kernel's length unit explicitly using boxsize.
    inline void apply_long_range_filter(FML::GRID::FFTWGrid<3> & fourier, const Kernel & kernel) {
        if (fourier.get_grid_status_real())
            throw std::invalid_argument("[ShortRangeGravity] Long-range filter requires a Fourier-space grid");
        const double rs_over_box = kernel.options().split_scale / kernel.options().boxsize;
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int slice = 0; slice < fourier.get_local_nx(); ++slice)
            for (auto i : fourier.get_fourier_range(slice, slice + 1)) {
                const auto k = fourier.get_fourier_wavevector_from_index(i);
                const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                fourier.set_fourier_from_index(i,
                                               fourier.get_fourier_from_index(i) *
                                                   FML::GRID::FloatType(std::exp(-k2 * rs_over_box * rs_over_box)));
            }
    }
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
