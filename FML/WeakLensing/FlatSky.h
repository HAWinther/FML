#ifndef FML_WEAK_LENSING_FLAT_SKY_HEADER
#define FML_WEAK_LENSING_FLAT_SKY_HEADER
#ifndef USE_FFTW
#error "WeakLensing/FlatSky.h requires USE_FFTW; the scalar WeakLensing.h API does not"
#endif
#include <FML/FFTWGrid/FFTWGrid.h>
#include "WeakLensing.h"
#include <array>

namespace FML::WEAKLENSING {
    using AngularGrid = FML::GRID::FFTWGrid<2>;
    struct ShearMaps {
        AngularGrid gamma1, gamma2;
    };
    namespace detail {
        inline void collective_require(bool valid, const char * message) {
            int good = valid;
            FML::MinOverTasks(&good);
            if (!good)
                throw std::invalid_argument(message);
        }
    } // namespace detail
    /// Periodic flat-sky spin-2 shear: gamma1=(lx^2-ly^2)/l^2 kappa,
    /// gamma2=2lxly/l^2 kappa. theta_x/y in radians; input is preserved.
    /// DC and both even-N Nyquist lines are removed from both shear components.
    /// All ranks participate; outputs have no halos and contain owned real pixels.
    inline ShearMaps shear_from_convergence(const AngularGrid & kappa, double theta_x, double theta_y) {
        auto spectrum = kappa;
        const int n = spectrum.get_nmesh();
        detail::collective_require(n >= 2 && std::isfinite(theta_x) && std::isfinite(theta_y) && theta_x > 0 &&
                                       theta_y > 0 && std::isfinite(n / theta_x) && std::isfinite(n / theta_y) &&
                                       spectrum.get_grid_status_real(),
                                   "WeakLensing: shear needs real allocated grid and positive finite angular widths");
        int min_n = n, max_n = n;
        FML::MinOverTasks(&min_n);
        FML::MaxOverTasks(&max_n);
        double low[2] = {theta_x, theta_y}, high[2] = {theta_x, theta_y};
        for (int j = 0; j < 2; ++j) {
            FML::MinOverTasks(&low[j]);
            FML::MaxOverTasks(&high[j]);
        }
        detail::collective_require(min_n == max_n && low[0] == high[0] && low[1] == high[1],
                                   "WeakLensing: ranks disagree on shear geometry");
        bool finite = true;
        for (int ix = 0; ix < spectrum.get_local_nx(); ++ix)
            for (int iy = 0; iy < n; ++iy)
                finite = finite && std::isfinite(spectrum.get_real({ix, iy}));
        detail::collective_require(finite, "WeakLensing: nonfinite convergence pixels");
        spectrum.fftw_r2c();
        ShearMaps result{AngularGrid(n), AngularGrid(n)};
        auto * one = result.gamma1.get_fourier_grid();
        auto * two = result.gamma2.get_fourier_grid();
        const auto * input = spectrum.get_fourier_grid();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType i = 0; i < spectrum.get_ntot_fourier(); ++i) {
            const auto mode = spectrum.get_fourier_integer_wavevector_from_index(i);
            if ((mode[0] == 0 && mode[1] == 0) || (n % 2 == 0 && (std::abs(mode[0]) == n / 2 || mode[1] == n / 2))) {
                one[i] = two[i] = FML::GRID::ComplexType(0);
                continue;
            }
            const double lx = mode[0] / theta_x, ly = mode[1] / theta_y;
            const double scale = std::max(std::abs(lx), std::abs(ly));
            const double x = lx / scale, y = ly / scale, norm = x * x + y * y;
            one[i] = input[i] * FML::GRID::FloatType((x * x - y * y) / norm);
            two[i] = input[i] * FML::GRID::FloatType(2 * x * y / norm);
        }
        result.gamma1.set_grid_status_real(false);
        result.gamma2.set_grid_status_real(false);
        result.gamma1.fftw_c2r();
        result.gamma2.fftw_c2r();
        return result;
    }
} // namespace FML::WEAKLENSING
#endif
