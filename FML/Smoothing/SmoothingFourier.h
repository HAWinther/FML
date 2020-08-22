#ifndef SMOOTHINGFOURIER_HEADER
#define SMOOTHINGFOURIER_HEADER

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>

namespace FML {
    namespace GRID {


        //===================================================================================
        /// Take a fourier grid and divide each mode by its norm: \f$ f(k) \to f(k) / |f(k)| \f$
        ///
        /// @tparam N The dimension of the grid
        ///
        /// @param[out] fourier_grid The fourier grid we do the whitening on
        ///
        //===================================================================================
        template <int N>
        void whitening_fourier_space(FFTWGrid<N> & fourier_grid) {
            std::vector<double> kvec(N);
            [[maybe_unused]] double kmag2;
            for (auto & index : fourier_grid.get_fourier_range()) {
                auto value = fourier_grid.get_fourier_from_index(index);
                double norm = std::sqrt(std::norm(value));
                if (norm == 0.0)
                    norm = 0.0;
                fourier_grid.set_fourier_from_index(index, value * norm);
            }
        }

        //===================================================================================
        /// Low-pass filters (tophat, gaussian, sharpk)
        ///
        /// @tparam N The dimension of the grid
        ///
        /// @param[out] fourier_grid The fourier grid we do the smoothing of
        /// @param[in] smoothing_scale The smoothing radius of the filter (in units of the boxsize)
        /// @param[in] smoothing_method The smoothing filter (tophat, gaussian, sharpk)
        ///
        //===================================================================================
        template <int N>
        void smoothing_filter_fourier_space(FFTWGrid<N> & fourier_grid,
                                            double smoothing_scale,
                                            std::string smoothing_method) {

            // Sharp cut off kR = 1
            std::function<double(double)> filter_sharpk = [=](double k2) -> double {
                double kR2 = k2 * smoothing_scale * smoothing_scale;
                if (kR2 < 1.0)
                    return 1.0;
                return 0.0;
            };
            // Gaussian exp(-kR^2/2)
            std::function<double(double)> filter_gaussian = [=](double k2) -> double {
                double kR2 = k2 * smoothing_scale * smoothing_scale;
                return std::exp(-0.5 * kR2);
            };
            // Top-hat F[ (|x| < R) ]. Implemented only for 2D and 3D
            std::function<double(double)> filter_tophat_2D = [=](double k2) -> double {
                double kR2 = k2 * smoothing_scale * smoothing_scale;
                double kR = std::sqrt(kR2);
                if (kR2 < 1e-8)
                    return 1.0;
                return 2.0 / (kR2) * (1.0 - std::cos(kR));
            };
            std::function<double(double)> filter_tophat_3D = [=](double k2) -> double {
                double kR2 = k2 * smoothing_scale * smoothing_scale;
                double kR = std::sqrt(kR2);
                if (kR2 < 1e-8)
                    return 1.0;
                return 3.0 * (std::sin(kR) - kR * std::cos(kR)) / (kR2 * kR);
            };

            // Select the filter
            std::function<double(double)> filter;
            if (smoothing_method == "sharpk") {
                filter = filter_sharpk;
            } else if (smoothing_method == "gaussian") {
                filter = filter_gaussian;
            } else if (smoothing_method == "tophat") {
                assert_mpi(N == 2 or N == 3,
                           "[smoothing_filter_fourier_space] Tophat filter only implemented in 2D and 3D");
                if (N == 2)
                    filter = filter_tophat_2D;
                if (N == 3)
                    filter = filter_tophat_3D;
            } else {
                throw std::runtime_error("Unknown filter " + smoothing_method + " Options: sharpk, gaussian, tophat");
            }

            // Do the smoothing
            std::array<double,N> kvec;
            double kmag2;
            for (auto & index : fourier_grid.get_fourier_range()) {
                fourier_grid.get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag2);
                auto value = fourier_grid.get_fourier_from_index(index);
                value *= filter(kmag2);
                fourier_grid.set_fourier_from_index(index, value);
            }
        }
    } // namespace GRID
} // namespace FML
#endif
