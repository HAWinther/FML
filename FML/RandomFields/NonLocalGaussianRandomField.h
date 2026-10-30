#ifndef NONLOCALGAUSSIANRANDOMFIELD_HEADER
#define NONLOCALGAUSSIANRANDOMFIELD_HEADER
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>

namespace FML {
    namespace RANDOM {

        //=================================================================================
        ///
        /// Generate a non-local gaussian random field in real or fourier space from a given
        /// power-spectrum and a source of random numbers. Templated on dimension.
        ///
        /// The type of fNL is local, orthogonal, equilateral and generic (supply kernel values
        /// yourself. See 1108.5512 for more info)
        ///
        /// fix_amplitude means we fix the amplitude of the modes in the gaussian random field
        /// we generate to start with
        ///
        //=================================================================================

        namespace NONGAUSSIAN {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;

            //======================================================================
            ///
            /// \f$ P(k) \to \phi_{\rm gaussian} \to \phi = \phi_{\rm gaussian} + f_{\rm NL} K(\phi_{\rm
            /// gaussian},\phi_{\rm gaussian}) \f$ where the kernel is some quadratic form which in its simples form
            /// (local) is just \f$ \phi_{\rm gaussian}^2 \f$.
            ///
            /// Uses up to 6 simultaneous grids and 8 Fourier transforms. Inactive
            /// nonlocal outputs are skipped, retaining their shared input dependencies.
            ///
            /// NB: for generating IC for cosmological simulations see the related method
            /// and see 1108.5512 for more info about the algorithm.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[out] phi_fourier The fourier grid we generate
            /// @param[in] rng The random number generator
            /// @param[in] Pofk_of_kBox_over_volume This is \f$ P(kB) / V \f$ where \f$ kB \f$ is the dimesnionless
            /// wavenumber where \f$ B \f$ is the boxsize and \f$ V = B^{\rm N} \f$ is the volume of the box.
            /// @param[in] fix_amplitude Fix the amplitude of the norm of \f$ \delta(k) \f$.
            /// @param[in] fNL The value of \f$ f_{\rm NL} \f$ you want
            /// @param[in] type_of_fnl The type of \f$ f_{\rm NL} \f$ (local, equilaterial, orthogonal, generic)
            /// @param[in] u Optional advanced option. See 1108.5512 for more info.
            /// @param[in] kernel_values Optional. If you specify generic fNL then this gives the kernel values.
            ///
            //======================================================================
            template <int N>
            void
            generate_nonlocal_gaussian_random_field_fourier(FFTWGrid<N> & phi_fourier,
                                                            RandomGenerator * rng,
                                                            std::function<double(double)> Pofk_of_kBox_over_volume,
                                                            bool fix_amplitude,
                                                            double fNL,
                                                            std::string type_of_fnl,
                                                            double u = 0.0,
                                                            std::vector<double> kernel_values = {},
                                                            const GAUSSIAN::RandomFieldOptions & random_options = {}) {

                // Ensure that <phi> = 0 in the end (this costs 2 extra FT)
                // This is not really needed as all terms apart from phi^2 mean to zero
                // and we take care of phi^2 separately. Also it is not observable anyway
                const double subtract_mean = false;

                // Set up the kernel values:
                // 0 is coefficient of (phi^2 - <phi^2>), 1 is P13[ phi Pm13 ]
                // 2 is P23[ phi Pm23 - Pm13^2 ], 3 is P1 [ phi Pm1 - Pm23 Pm13 ]
                std::vector<double> kernel_values_gaussian = {0.0, 0.0, 0.0, 0.0};
                std::vector<double> kernel_values_local = {+1.0, 0.0, 0.0, 0.0};
                std::vector<double> kernel_values_orthogonal = {-9.0 * (1.0 - u), 10.0 - 9.0 * u, +8.0, -9.0 * u};
                std::vector<double> kernel_values_equilateral = {-3.0 * (1.0 - u), 4.0 - 3.0 * u, +2.0, -3.0 * u};

                if (type_of_fnl == "gaussian") {
                    kernel_values = kernel_values_gaussian;
                    fNL = 0.0;
                } else if (type_of_fnl == "local") {
                    kernel_values = kernel_values_local;
                } else if (type_of_fnl == "equilateral") {
                    kernel_values = kernel_values_equilateral;
                } else if (type_of_fnl == "orthogonal") {
                    kernel_values = kernel_values_orthogonal;
                } else if (type_of_fnl == "generic") {
                    assert_mpi(kernel_values.size() == 4,
                               "[generate_nonlocal_gaussian_random_field_fourier] We need 4 kernel values supplied");
                } else {
                    std::string error = "[generate_nonlocal_gaussian_random_field_fourier] Unknown fNL type " +
                                        type_of_fnl +
                                        ". Available options are: local, equilateral, orthogonal and generic";
                    assert_mpi(false, error.c_str());
                }

                // Scale kernel values by fNL
                for (auto & val : kernel_values)
                    val *= fNL;

                const auto Nmesh = phi_fourier.get_nmesh();
                const auto Local_nx = phi_fourier.get_local_nx();
                const auto nleft = phi_fourier.get_n_extra_slices_left();
                const auto nright = phi_fourier.get_n_extra_slices_right();
                assert_mpi(Nmesh > 0,
                           "[generate_nonlocal_gaussian_random_field_fourier] Grid must already be allocated");

                // Generate a gaussian random field in fourier space. We generate white noise (P = 1) and
                // multiply by sqrt(P(k)) evaluated exactly in every cell: the gaussian generator works with a
                // cubic spline of P(k), and for the steep primordial power-spectrum (~k^-3, with a
                // constant value at k = 0) this spline is inaccurate at low k and can even go negative
                // (giving NaN). The random numbers are the same so this is the same field as before
                auto white_noise = []([[maybe_unused]] double kBox) { return 1.0; };
                FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(phi_fourier,
                                                                              rng,
                                                                              white_noise,
                                                                              fix_amplitude,
                                                                              random_options);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_fourier.get_fourier_range(islice, islice + 1)) {
                        phi_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        if (kmag == 0.0)
                            continue;
                        auto value = phi_fourier.get_fourier_from_index(fourier_index);
                        value *= FML::GRID::FloatType(std::sqrt(Pofk_of_kBox_over_volume(kmag)));
                        phi_fourier.set_fourier_from_index(fourier_index, value);
                    }
                }

                // Set DC mode to zero (this should be the case, but just to be sure)
                if (FML::ThisTask == 0)
                    phi_fourier.set_fourier_from_index(0, 0.0);

                // If fNL = 0 no point to continue
                if (fNL == 0.0)
                    return;

                // Get phi in real space
                FFTWGrid<N> phi_real = phi_fourier;
                phi_real.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_real");
                phi_real.set_grid_status_real(false);
                phi_real.fftw_c2r();

                // Compute phi^2 - <phi>^2 in real space and store in source
                FFTWGrid<N> source(Nmesh, nleft, nright);
                source.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::source");
                double phi_squared_mean = 0.0;
                double phi_mean = 0.0;
                long long int ncells = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : phi_squared_mean, phi_mean, ncells)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : phi_real.get_real_range(islice, islice + 1)) {
                        auto phi = phi_real.get_real_from_index(real_index);
                        auto value = phi * phi;
                        source.set_real_from_index(real_index, value);
                        phi_squared_mean += value;
                        phi_mean += phi;
                        ncells += 1;
                    }
                }

                FML::SumOverTasks(&phi_squared_mean);
                FML::SumOverTasks(&phi_mean);
                FML::SumOverTasks(&ncells);
                assert_mpi(ncells == (long long int)(FML::power(Nmesh, N)),
                           "[generate_nonlocal_gaussian_random_field_fourier] Number of cells we have summed over does "
                           "not agree with how many cells are in the grid");
                phi_squared_mean /= std::pow(Nmesh, N);
                phi_mean /= std::pow(Nmesh, N);

                if (FML::ThisTask == 0)
                    std::cout << "[generate_nonlocal_gaussian_random_field_fourier] <Phi^2>: " << phi_squared_mean
                              << " <Phi>: " << phi_mean << "\n";

#ifdef USE_OMP
#pragma omp parallel for
#endif
                // Subtract <phi^2>
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : source.get_real_range(islice, islice + 1)) {
                        auto phi2 = source.get_real_from_index(real_index);
                        auto phi = phi_real.get_real_from_index(real_index);
                        auto value =
                            (phi - phi_mean) + FML::GRID::FloatType(kernel_values[0]) * (phi2 - phi_squared_mean);
                        source.set_real_from_index(real_index, value);
                    }
                }

                // Back to fourier space. We now have F[phi] + const * F[phi^2 - <phi^2>] in source
                source.fftw_r2c();

                // Set DC mode to zero
                if (FML::ThisTask == 0)
                    source.set_fourier_from_index(0, 0.0);

                // If we only want standard local phi + fNL[phi^2 - <phi^2>] then we can return now
                if (kernel_values[1] == 0.0 and kernel_values[2] == 0.0 and kernel_values[3] == 0.0) {
                    phi_fourier = std::move(source);
                    return;
                }

                // NB: we evaluate P(k) exactly and not with a spline (FFTWGrid::make_fourier_spline) as
                // such a spline of the steep primordial power-spectrum is inaccurate (see above)
                const auto & Pofk_of_kBox_over_volume_spline = Pofk_of_kBox_over_volume;

                // Output terms and their filtered-field dependencies are different:
                // term 2 needs P^-1/3, and term 3 needs both P^-1/3 and P^-2/3.
                const bool term1_active = kernel_values[1] != 0.0;
                const bool term2_active = kernel_values[2] != 0.0;
                const bool term3_active = kernel_values[3] != 0.0;
                const bool need_m23 = term2_active || term3_active;
                FFTWGrid<N> phi_m13(Nmesh);
                FFTWGrid<N> phi_m23, phi_m33;
                if (need_m23)
                    phi_m23 = FFTWGrid<N>(Nmesh);
                if (term3_active)
                    phi_m33 = FFTWGrid<N>(Nmesh);
                phi_m13.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m13");
                if (need_m23)
                    phi_m23.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m23");
                if (term3_active)
                    phi_m33.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m33");
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_fourier.get_fourier_range(islice, islice + 1)) {
                        phi_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                        FML::GRID::FloatType pofk_m13 = std::pow(Pofk_of_kBox_over_volume_spline(kmag), -1.0 / 3.0);
                        FML::GRID::FloatType pofk_m23 = need_m23 ? pofk_m13 * pofk_m13 : 0;
                        FML::GRID::FloatType pofk_m33 = term3_active ? pofk_m23 * pofk_m13 : 0;

                        auto phi = phi_fourier.get_fourier_from_index(fourier_index);
                        phi_m13.set_fourier_from_index(fourier_index, phi * pofk_m13);
                        if (need_m23)
                            phi_m23.set_fourier_from_index(fourier_index, phi * pofk_m23);
                        if (term3_active)
                            phi_m33.set_fourier_from_index(fourier_index, phi * pofk_m33);
                    }
                }

                // Set DC mode to 0
                if (FML::ThisTask == 0) {
                    phi_m13.set_fourier_from_index(0, 0.0);
                    if (need_m23)
                        phi_m23.set_fourier_from_index(0, 0.0);
                    if (term3_active)
                        phi_m33.set_fourier_from_index(0, 0.0);
                }
                phi_m13.fftw_c2r();
                if (need_m23)
                    phi_m23.fftw_c2r();
                if (term3_active)
                    phi_m33.fftw_c2r();

                // We use the grids already allocated as temporary grids
                // We must make sure we do things in the right order below
                FFTWGrid<N> & temp1 = phi_m13;
                FFTWGrid<N> & temp2 = phi_m23;
                FFTWGrid<N> & temp3 = phi_m33;

                // Compute the square of some quantities in real space
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : temp1.get_real_range(islice, islice + 1)) {
                        auto pm13 = phi_m13.get_real_from_index(real_index);
                        auto pm23 = need_m23 ? phi_m23.get_real_from_index(real_index) : FML::GRID::FloatType(0);
                        auto pm33 = term3_active ? phi_m33.get_real_from_index(real_index) : FML::GRID::FloatType(0);
                        auto phi = phi_real.get_real_from_index(real_index);

                        if (term1_active)
                            temp1.set_real_from_index(real_index, phi * pm13);
                        if (term2_active)
                            temp2.set_real_from_index(real_index, phi * pm23 - pm13 * pm13);
                        if (term3_active)
                            temp3.set_real_from_index(real_index, phi * pm33 - pm13 * pm23);
                    }
                }

                // Transform these back to fourier space
                if (term1_active)
                    temp1.fftw_r2c();
                if (term2_active)
                    temp2.fftw_r2c();
                if (term3_active)
                    temp3.fftw_r2c();
                if (FML::ThisTask == 0) {
                    if (term1_active)
                        temp1.set_fourier_from_index(0, 0.0);
                    if (term2_active)
                        temp2.set_fourier_from_index(0, 0.0);
                    if (term3_active)
                        temp3.set_fourier_from_index(0, 0.0);
                }

                // Add up to get phi + fNL K(phi,phi) in fourier space
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : source.get_fourier_range(islice, islice + 1)) {
                        source.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                        FML::GRID::FloatType pofk_p13 = std::pow(Pofk_of_kBox_over_volume_spline(kmag), 1.0 / 3.0);
                        FML::GRID::FloatType pofk_p23 = need_m23 ? pofk_p13 * pofk_p13 : 0;
                        FML::GRID::FloatType pofk_p33 = term3_active ? pofk_p23 * pofk_p13 : 0;

                        auto term1 = term1_active ? temp1.get_fourier_from_index(fourier_index) * pofk_p13 :
                                                    FML::GRID::ComplexType(0);
                        auto term2 = term2_active ? temp2.get_fourier_from_index(fourier_index) * pofk_p23 :
                                                    FML::GRID::ComplexType(0);
                        auto term3 = term3_active ? temp3.get_fourier_from_index(fourier_index) * pofk_p33 :
                                                    FML::GRID::ComplexType(0);

                        auto s = source.get_fourier_from_index(fourier_index);
                        s += term1 * FML::GRID::FloatType(kernel_values[1]) +
                             term2 * FML::GRID::FloatType(kernel_values[2]) +
                             term3 * FML::GRID::FloatType(kernel_values[3]);
                        source.set_fourier_from_index(fourier_index, s);
                    }
                }

                // Set DC mode to 0
                if (FML::ThisTask == 0)
                    source.set_fourier_from_index(0, 0.0);

                // Transfer ownership instead of copying the full Fourier payload.
                phi_fourier = std::move(source);

                // Ensure that <phi> = 0
                if (subtract_mean) {
                    phi_fourier.fftw_c2r();

                    double phi_mean = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : phi_mean)
#endif
                    for (int islice = 0; islice < Local_nx; islice++) {
                        for (auto && real_index : phi_fourier.get_real_range(islice, islice + 1)) {
                            auto value = phi_fourier.get_real_from_index(real_index);
                            phi_mean += value;
                        }
                    }
                    FML::SumOverTasks(&phi_mean);
                    phi_mean /= std::pow(Nmesh, N);

                    if (FML::ThisTask == 0)
                        std::cout << "Subtracting mean: " << phi_mean << "\n";

#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (int islice = 0; islice < Local_nx; islice++) {
                        for (auto && real_index : phi_fourier.get_real_range(islice, islice + 1)) {
                            auto phi = phi_fourier.get_real_from_index(real_index);
                            phi_fourier.set_real_from_index(real_index, phi - phi_mean);
                        }
                    }
                    phi_fourier.fftw_r2c();

                    // Set DC mode to zero
                    if (FML::ThisTask == 0)
                        phi_fourier.set_fourier_from_index(0, 0.0);
                }
            }

            //======================================================================
            ///
            /// This computes a non-gaussian density field at any redshift.
            /// There is a sign between the Bardeen potential and the gravitational potential
            /// which leads to a sign difference in \f$ f_{\rm NL} \f$ so check that its consistent with your
            /// convention.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[out] delta_fourier The fourier grid we generate
            /// @param[in] rng The random number generator
            /// @param[in] Pofk_of_kBox_over_Pofk_primordal The ratio \f$ P_\delta(kB) / P_{\rm primordial}(kB) \f$
            /// where \f$ kB \f$ is the dimesnionless wavenumber and \f$ B \f$ the boxsize (i.e. the ratio of the
            /// power-spectrum at the time you want to generate delta to the primordial one
            /// @param[in] Pofk_of_kBox_over_volume_primordial The dimensionless primordial power-spectrum \f$ P_{\rm
            /// primordial}(kB) / V\f$ where \f$ V = B^{\rm N} \f$ is the volume of the box. For the fiducial primordial
            /// power-spectrum in 3D we have \f$ P_{\rm primordial}(kB) / V = \frac{2\pi^2}{(kB)^3} A_s (k/k_{\rm
            /// pivot})^{n_s-1}\f$
            /// @param[in] fix_amplitude Fix the amplitude of the norm of \f$ \delta(k) \f$.
            /// @param[in] fNL The value of \f$ f_{\rm NL} \f$ you want
            /// @param[in] type_of_fnl The type of \f$ f_{\rm NL} \f$ (local, equilaterial, orthogonal, generic)
            /// @param[in] u Optional advanced option. See 1108.5512 for more info.
            /// @param[in] kernel_values Optional. If you specify generic fNL then this gives the kernel values.
            ///
            //======================================================================
            template <int N>
            void generate_nonlocal_gaussian_random_field_fourier_cosmology(
                FFTWGrid<N> & delta_fourier,
                RandomGenerator * rng,
                std::function<double(double)> Pofk_of_kBox_over_Pofk_primordal,
                std::function<double(double)> Pofk_of_kBox_over_volume_primordial,
                bool fix_amplitude,
                double fNL,
                std::string type_of_fnl,
                double u = 0.0,
                std::vector<double> kernel_values = {},
                const GAUSSIAN::RandomFieldOptions & random_options = {}) {

                // Generate a gaussian random field delta using a primordial power-spectrum
                FFTWGrid<N> & phi_fourier = delta_fourier;
                generate_nonlocal_gaussian_random_field_fourier(phi_fourier,
                                                                rng,
                                                                Pofk_of_kBox_over_volume_primordial,
                                                                fix_amplitude,
                                                                fNL,
                                                                type_of_fnl,
                                                                u,
                                                                kernel_values,
                                                                random_options);

                const auto Local_nx = phi_fourier.get_local_nx();
                const auto Local_x_start = phi_fourier.get_local_x_start();

                // Transform to delta by multiplying by sqrt(P(k) / Pprimodial(k))
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_fourier.get_fourier_range(islice, islice + 1)) {
                        // The DC mode (k=0) is zero. We skip it as the ratio is often 0/0 = NaN there
                        if (Local_x_start == 0 and fourier_index == 0) {
                            phi_fourier.set_fourier_from_index(fourier_index, 0.0);
                            continue;
                        }
                        phi_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        auto rescaling_factor = std::sqrt(Pofk_of_kBox_over_Pofk_primordal(kmag));
                        auto value = phi_fourier.get_fourier_from_index(fourier_index);
                        phi_fourier.set_fourier_from_index(fourier_index,
                                                           value * FML::GRID::FloatType(rescaling_factor));
                    }
                }

                // We now have delta(k,zini) in delta_fourier
            }

        } // namespace NONGAUSSIAN
    } // namespace RANDOM
} // namespace FML
#endif
