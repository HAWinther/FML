#ifndef NONLOCALGAUSSIANRANDOMFIELD_HEADER
#define NONLOCALGAUSSIANRANDOMFIELD_HEADER
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
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
        /// youself. See 1108.5512 for more info)
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
            /// P(k) -> phi_gaussian -> phi = phi_gaussian + fNL phi_gaussian^2
            /// or fNL K(phi_gaussian,phi_gaussian) in general
            ///
            /// This requires in general 6 grids to be allocated at the same time
            /// and up to 8 fourier transforms
            ///
            /// NB: for generating IC for cosmological simulations Pofk_of_kBox_over_volume is the
            /// power-spectrum of the gravitational potential and the fNL value is the
            /// fNL value at the redshift the power-spectrum is defined at
            /// See 1108.5512 for more info or see the _cosmology method below.
            ///
            /// There is a sign between the Bardeen potential and the gravitational potential
            /// which leads to a sign difference in fNL!
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[out] phi_fourier The fourier grid we generate
            /// @param[in] rng The random number generator
            /// @param[in] Pofk_of_kBox_over_volume This is \f$ P(kB) / V \f$ where $kB$ is the dimesnionless wavenumber
            /// (B the boxsize) and \f$ V = B^{\rm N} \f$ is the volume of the box.
            /// @param[in] fix_amplitude Fix the amplitude of the norm of \f$ \delta(k) \f$.
            /// @param[in] fNL The value of fNL you want
            /// @param[in] type_of_fnl The type of fNL (local, equilaterial, orthogonal, generic)
            /// @param[in] u Optional advanced option. See 1108.5512 for more info.
            /// @param[in] kernel_values Optional. If you specify generic fNL then this gives the kernel values.
            ///
            //======================================================================
            template <int N>
            void generate_nonlocal_gaussian_random_field_fourier(FFTWGrid<N> & phi_fourier,
                                                                 RandomGenerator * rng,
                                                                 std::function<double(double)> Pofk_of_kBox_over_volume,
                                                                 bool fix_amplitude,
                                                                 double fNL,
                                                                 std::string type_of_fnl,
                                                                 double u = 0.0,
                                                                 std::vector<double> kernel_values = {}) {

                // Ensure that <phi> = 0 in the end (this costs 2 extra FT)
                // This is not really needed as all terms apart from phi^2 mean to zero
                // and we take care of phi^2 seperately. Also it is not observable anyway
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
                                        ". Availiable options are: local, equilateral, orthogonal and generic";
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

                // Generate a gaussian random field in fourier space
                FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(
                    phi_fourier, rng, Pofk_of_kBox_over_volume, fix_amplitude);

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
                assert_mpi(ncells == FML::power((long long int)(Nmesh), N),
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
                        auto value = (phi - phi_mean) + kernel_values[0] * (phi2 - phi_squared_mean);
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
                    phi_fourier = source;
                    return;
                }

                // Compute the kernel terms
                FFTWGrid<N> phi_m13(Nmesh);
                FFTWGrid<N> phi_m23(Nmesh);
                FFTWGrid<N> phi_m33(Nmesh); // XXX Not needed when u=0
                phi_m13.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m13");
                phi_m23.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m23");
                phi_m33.add_memory_label("FFTWGrid::generate_nonlocal_gaussian_random_field_fourier::phi_m33");
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_fourier.get_fourier_range(islice, islice + 1)) {
                        phi_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                        double pofk_m13 = std::pow(Pofk_of_kBox_over_volume(kmag), -1.0 / 3.0);
                        double pofk_m23 = pofk_m13 * pofk_m13;
                        double pofk_m33 = pofk_m23 * pofk_m13; // XXX Not needed when u=0

                        auto phi = phi_fourier.get_fourier_from_index(fourier_index);
                        auto value1 = phi * pofk_m13;
                        auto value2 = phi * pofk_m23;
                        auto value3 = phi * pofk_m33; // XXX Not needed when u=0

                        phi_m13.set_fourier_from_index(fourier_index, value1);
                        phi_m23.set_fourier_from_index(fourier_index, value2);
                        phi_m33.set_fourier_from_index(fourier_index, value3); // XXX Not needed when u=0
                    }
                }

                // Set DC mode to 0
                if (FML::ThisTask == 0) {
                    phi_m13.set_fourier_from_index(0, 0.0);
                    phi_m23.set_fourier_from_index(0, 0.0);
                    phi_m33.set_fourier_from_index(0, 0.0); // XXX Not needed when u=0
                }
                phi_m13.fftw_c2r();
                phi_m23.fftw_c2r();
                phi_m33.fftw_c2r(); // XXX Not needed when u=0

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
                        auto pm23 = phi_m23.get_real_from_index(real_index);
                        auto pm33 = phi_m33.get_real_from_index(real_index); // XXX Not needed when u=0
                        auto phi = phi_real.get_real_from_index(real_index);

                        auto value1 = phi * pm13;
                        auto value2 = phi * pm23 - pm13 * pm13;
                        auto value3 = phi * pm33 - pm13 * pm23; // XXX Not needed when u=0
                        temp1.set_real_from_index(real_index, value1);
                        temp2.set_real_from_index(real_index, value2);
                        temp3.set_real_from_index(real_index, value3); // XXX Not needed when u=0
                    }
                }

                // Transform these back to fourier space
                temp1.fftw_r2c();
                temp2.fftw_r2c();
                temp3.fftw_r2c(); // XXX Not needed when u=0
                if (FML::ThisTask == 0) {
                    temp1.set_fourier_from_index(0, 0.0);
                    temp2.set_fourier_from_index(0, 0.0);
                    temp3.set_fourier_from_index(0, 0.0); // XXX Not needed when u=0
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

                        double pofk_p13 = std::pow(Pofk_of_kBox_over_volume(kmag), 1.0 / 3.0);
                        double pofk_p23 = pofk_p13 * pofk_p13;
                        double pofk_p33 = pofk_p23 * pofk_p13;

                        auto term1 = temp1.get_fourier_from_index(fourier_index) * pofk_p13;
                        auto term2 = temp2.get_fourier_from_index(fourier_index) * pofk_p23;
                        auto term3 = temp3.get_fourier_from_index(fourier_index) * pofk_p33; // XXX Not needed when u=0

                        auto old_source = source.get_fourier_from_index(fourier_index);
                        auto value =
                            old_source + kernel_values[1] * term1 + kernel_values[2] * term2 + kernel_values[3] * term3;
                        if (kmag == 0.0) {
                            value = 0.0;
                        }
                        source.set_fourier_from_index(fourier_index, value);
                    }
                }

                // Set DC mode to 0
                if (FML::ThisTask == 0)
                    source.set_fourier_from_index(0, 0.0);

                // Copy over result
                phi_fourier = source;

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
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[out] delta_fourier The fourier grid we generate
            /// @param[in] rng The random number generator
            /// @param[in] Pofk_of_kBox_over_Pofk_primordal The ratio \f$ P_\delta(kB) / P_{\rm primordial}(kB) \f$
            /// where $kB$ is the dimesnionless wavenumber and B the boxsize (i.e. the ratio of the power-spectrum at
            /// the time you want to generate delta to the primordial one
            /// @param[in] Pofk_of_kBox_over_volume_primordial The dimensionless primordial power-spectrum \f$ P_{\rm
            /// primordial}(kB) / V\f$ where \f$ V = B^{\rm N} \f$ is the volume of the box. For the fiducial primordial
            /// power-spectrum in 3D we have \f$ P_{\rm primordial}(kB) / V = \frac{2\pi^2}{(kB)^3} A_s (k/k_{\rm
            /// pivot})^{n_s-1}\f$
            /// @param[in] fix_amplitude Fix the amplitude of the norm of \f$ \delta(k) \f$.
            /// @param[in] fNL The value of fNL you want
            /// @param[in] type_of_fnl The type of fNL (local, equilaterial, orthogonal, generic)
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
                std::vector<double> kernel_values = {}) {

                // Generate a gaussian random field delta using a primordial power-spectrum
                FFTWGrid<N> & phi_fourier = delta_fourier;
                generate_nonlocal_gaussian_random_field_fourier(phi_fourier,
                                                                rng,
                                                                Pofk_of_kBox_over_volume_primordial,
                                                                fix_amplitude,
                                                                fNL,
                                                                type_of_fnl,
                                                                u,
                                                                kernel_values);

                const auto Local_nx = phi_fourier.get_local_nx();

                // Transform to delta by multiplying by sqrt(P(k) / Pprimodial(k))
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] double kmag;
                    [[maybe_unused]] std::array<double, N> kvec;
                    for (auto && fourier_index : phi_fourier.get_fourier_range(islice, islice + 1)) {
                        phi_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        auto rescaling_factor = std::sqrt(Pofk_of_kBox_over_Pofk_primordal(kmag));
                        auto value = phi_fourier.get_fourier_from_index(fourier_index);
                        phi_fourier.set_fourier_from_index(fourier_index, value * rescaling_factor);
                    }
                }

                // We now have delta(k,zini) in delta_fourier
            }

        } // namespace NONGAUSSIAN
    }     // namespace RANDOM
} // namespace FML
#endif
