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
            auto Local_nx = fourier_grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : fourier_grid.get_fourier_range(islice, islice + 1)) {
                    auto value = fourier_grid.get_fourier_from_index(fourier_index);
                    double norm = std::sqrt(std::norm(value));
                    norm = norm == 0.0 ? 0.0 : 1.0 / norm;
                    fourier_grid.set_fourier_from_index(fourier_index, value * norm);
                }
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
            auto Local_nx = fourier_grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : fourier_grid.get_fourier_range(islice, islice + 1)) {
                    fourier_grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    auto value = fourier_grid.get_fourier_from_index(fourier_index);
                    value *= filter(kmag2);
                    fourier_grid.set_fourier_from_index(fourier_index, value);
                }
            }
        }

        //===================================================================================
        /// @brief From two fourier grids, f and g, compute the convolution
        /// \f$ f(k) * g(k) = \int d^{\rm N}q f(q) g(k-q) \f$ This is done via multuplication in reals-space. We
        /// allocate one temporary grid and perform 3 fourier tranforms.
        ///
        /// @tparam N dimension of the grid
        ///
        /// @param[in] fourier_grid_f The fourier grid f
        /// @param[in] fourier_grid_g The fourier grid g
        /// @param[out] fourier_grid_result Fourier grid containing the convolution of the two gridsf
        ///
        //===================================================================================
        template <int N>
        void convolution_fourier_space(const FFTWGrid<N> & fourier_grid_f,
                                       const FFTWGrid<N> & fourier_grid_g,
                                       FFTWGrid<N> & fourier_grid_result) {

            bool f_and_g_are_the_same_grid = (&fourier_grid_f == &fourier_grid_g);

            // Make copies: tmp contains f and result contains g
            // unless f = g for which we can don't need the copy
            // and save doing 1 FFT
            FFTWGrid<N> tmp;
            fourier_grid_result = fourier_grid_g;
            fourier_grid_result.add_memory_label("FFTWGrid::convolution_fourier_space::fourier_grid_result");

            // Fourier transform to real space
            fourier_grid_result.fftw_c2r();
            if (!f_and_g_are_the_same_grid) {
                tmp = fourier_grid_f;
                tmp.add_memory_label("FFTWGrid::convolution_fourier_space::tmp");
                tmp.fftw_c2r();
            }

            // Multiply the two grids in real space
            auto Local_nx = fourier_grid_result.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : fourier_grid_result.get_real_range(islice, islice + 1)) {
                    auto g_real = fourier_grid_result.get_real_from_index(real_index);
                    auto f_real = g_real;
                    if (!f_and_g_are_the_same_grid)
                        f_real = tmp.get_real_from_index(real_index);
                    fourier_grid_result.set_real_from_index(real_index, f_real * g_real);
                }
            }

            // Transform back to obtain the desired convolution
            fourier_grid_result.fftw_r2c();
        }

        //===================================================================================
        /// @brief From two real grids, f and g, compute the convolution
        /// \f$ f(x) * g(x) = \int d^{\rm N}y f(y) g(x-y) \f$ This is done via multiplication in fourier-space.
        /// We allocate one temporary grid and perform 3 fourier tranforms. We can merge this with
        /// convolution_fourier_space and just have one method and get do the real or fourier space convolution
        /// depening on the status of the grids, but its easy to forget to set the status so we have two methods
        /// for this.
        ///
        /// @tparam N dimension of the grid
        ///
        /// @param[in] real_grid_f The real grid f
        /// @param[in] real_grid_g The real grid g
        /// @param[out] real_grid_result Real grid containing the convolution of the two grids.
        ///
        //===================================================================================
        template <int N>
        void convolution_real_space(const FFTWGrid<N> & real_grid_f,
                                    const FFTWGrid<N> & real_grid_g,
                                    FFTWGrid<N> & real_grid_result) {

            bool f_and_g_are_the_same_grid = (&real_grid_f == &real_grid_g);

            // Make copies: tmp contains f and result contains g
            // unless f = g for which we can don't need the copy
            // and save doing 1 FFT
            FFTWGrid<N> tmp;
            real_grid_result = real_grid_g;
            real_grid_result.add_memory_label("FFTWGrid::convolution_real_space::real_grid_result");

            // Fourier transform to fourier space
            real_grid_result.fftw_r2c();
            if (!f_and_g_are_the_same_grid) {
                tmp = real_grid_f;
                tmp.add_memory_label("FFTWGrid::convolution_real_space::tmp");
                tmp.fftw_r2c();
            }

            // Multiply the two grids in fourier space
            auto Local_nx = real_grid_result.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : real_grid_result.get_fourier_range(islice, islice + 1)) {
                    auto g_fourier = real_grid_result.get_fourier_from_index(fourier_index);
                    auto f_fourier = g_fourier;
                    if (!f_and_g_are_the_same_grid)
                        f_fourier = tmp.get_fourier_from_index(fourier_index);
                    real_grid_result.set_fourier_from_index(fourier_index, f_fourier * g_fourier);
                }
            }

            // Transform back to obtain the desired convolution
            real_grid_result.fftw_c2r();
        }
    } // namespace GRID
} // namespace FML
#endif
