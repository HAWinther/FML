#ifndef DISPLACEMENTFIELDS_HEADER
#define DISPLACEMENTFIELDS_HEADER
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>

namespace FML {
    namespace COSMOLOGY {
        /// This namespace contains things related to Lagrangian Perturbation Theory (displacement fields,
        /// reconstruction, initial conditions etc.)
        namespace LPT {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;

            template <int N>
            void from_LPT_potential_to_displacement_vector(const FFTWGrid<N> & phi_fourier,
                                                           std::vector<FFTWGrid<N>> & psi_real);

            template <int N>
            void compute_1LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_1LPT_fourier);

            template <int N>
            void compute_2LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_2LPT_fourier);

            // This is a method not tested
            template <int N>
            void compute_1LPT_2LPT_3LPT_displacment_field(const FFTWGrid<N> & delta_fourier,
                                                          std::vector<FFTWGrid<N>> & Psi,
                                                          std::vector<FFTWGrid<N>> & dPsidt,
                                                          double dlogDdt,
                                                          double DoverDini = 1.0);
            template <int N>
            void from_LPT_potential_to_displacement_vector_scaledependent(
                const FFTWGrid<N> & phi_fourier,
                std::vector<FFTWGrid<N>> & psi_real,
                std::function<double(double)> & growth_function_ratio);

            //=================================================================================
            /// Function is the ratio of the scale-dependent growth-factor at
            /// the time we want to generate particles to the time where phi was generated at
            /// as function of k
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] phi The LPT potential in fourier space
            /// @param[out] psi The displacement vector in real space
            /// @param[in] function The function D(k,z)/D(k,zini) as function of k.
            ///
            //=================================================================================
            template <int N>
            void from_LPT_potential_to_displacement_vector_scaledependent(const FFTWGrid<N> & phi,
                                                                          std::vector<FFTWGrid<N>> & psi,
                                                                          std::function<double(double)> & function) {

                // We require phi to exist and if psi exists it must have the same size as phi
                assert_mpi(phi.get_nmesh() > 0,
                           "[from_LPT_potential_to_displacement_vector_scaledependent] phi grid has to be already "
                           "allocated");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "From LPT potential to displacement field scaledependent\n";
#endif

                auto nleft = phi.get_n_extra_slices_left();
                auto nright = phi.get_n_extra_slices_right();
                auto Local_x_start = phi.get_local_x_start();
                auto Nmesh = phi.get_nmesh();
                size_t NmeshTotFourier = phi.get_ntot_fourier();

                // Create the output grids if they don't exist already
                for (int idim = 0; idim < N; idim++) {
                    if (psi[idim].get_nmesh() == 0) {
                        psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        psi.add_memory_label(
                            "FFTWGrid::from_LPT_potential_to_displacement_vector_scaledependent::Psi_" +
                            std::to_string(idim));
                    }
                }

                std::array<double, N> kvec;
                double kmag;
                std::complex<double> I(0, 1);
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    if (Local_x_start == 0 && ind == 0)
                        continue;

                    // Get wavevector and magnitude
                    phi.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);

                    // Psi_vec = D Phi => F[Psi_vec] = ik_vec F[Phi]
                    auto value = phi.get_fourier_from_index(ind) * I * function(kmag);

                    for (int idim = 0; idim < N; idim++) {
                        psi[idim].set_fourier_from_index(ind, value * kvec[idim]);
                    }
                }

                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transforming Dphi to real space: " << idim + 1 << " / " << N << "\n";
#endif
                    psi[idim].fftw_c2r();
                }
            }

            //=================================================================================
            /// Generate the displaceement field \f$ \Psi = \nabla \phi \f$ from the LPT potential \f$ \phi \f$.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] phi The LPT potential in fourier space
            /// @param[out] psi The displacement vector in real space
            ///
            //=================================================================================
            template <int N>
            void from_LPT_potential_to_displacement_vector(const FFTWGrid<N> & phi, std::vector<FFTWGrid<N>> & psi) {

                // We require phi to exist and if psi exists it must have the same size as phi
                assert_mpi(phi.get_nmesh() > 0,
                           "[from_LPT_potential_to_displacement_vector] Grid has to be already allocated!");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "From LPT potential to displaceent vector\n";
#endif

                auto nleft = phi.get_n_extra_slices_left();
                auto nright = phi.get_n_extra_slices_right();
                auto Local_x_start = phi.get_local_x_start();
                auto Nmesh = phi.get_nmesh();
                size_t NmeshTotFourier = phi.get_ntot_fourier();

                // Create the output grids if they don't exist already
                psi.resize(N);
                for (int idim = 0; idim < N; idim++) {
                    if (psi[idim].get_nmesh() == 0) {
                        psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        psi[idim].add_memory_label("FFTWGrid::from_LPT_potential_to_displacement_vector::Psi_" +
                                                   std::to_string(idim));
                    }
                }

                std::array<double, N> kvec;
                double kmag;
                std::complex<double> I(0, 1);
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {

                    // Get wavevector and magnitude
                    phi.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);

                    // Psi_vec = D Phi => F[Psi_vec] = ik_vec F[Phi]
                    auto value = phi.get_fourier_from_index(ind);
                    if (Local_x_start && ind == 0)
                        value = 0.0;
                    for (int idim = 0; idim < N; idim++) {
                        psi[idim].set_fourier_from_index(ind, I * value * kvec[idim]);
                    }
                }

                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transforming Dphi to real space: " << idim + 1 << " / " << N << "\n";
#endif
                    psi[idim].fftw_c2r();
                }
            }

            //=================================================================================
            /// Generate the 1LPT potential defined as \f$ \Psi^{\rm 1LPT} = \nabla \phi^{\rm 1LPT} \f$ and \f$
            /// \nabla^2 \phi^{\rm 1LPT} = -\delta \f$. Returns it in Fourier space.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta_fourier The density contrast in fourier space
            /// @param[out] phi_1LPT_fourier The LPT potential in fourier space
            ///
            //=================================================================================
            template <int N>
            void compute_1LPT_potential_fourier(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & phi_1LPT_fourier) {

                // We require delta to exist and if phi_1LPT is allocated it must have the same size as delta
                assert_mpi(delta_fourier.get_nmesh() > 0,
                           "[compute_1LPT_potential_fourier] delta grid has to be already allocated!");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute 1LPT potential\n";
#endif

                auto Nmesh_phi = phi_1LPT_fourier.get_nmesh();

                auto nleft = delta_fourier.get_n_extra_slices_left();
                auto nright = delta_fourier.get_n_extra_slices_right();
                auto Local_x_start = delta_fourier.get_local_x_start();
                auto Nmesh = delta_fourier.get_nmesh();
                size_t NmeshTotFourier = delta_fourier.get_ntot_fourier();

                // Create 1LPT grid
                if (Nmesh_phi == 0) {
                    phi_1LPT_fourier = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_fourier.add_memory_label("FFTWGrid::compute_1LPT_potential_fourier::phi_1LPT_fourier");
                }

                // Divide grid by k^2. Assuming delta was created in fourier-space so no FFTW normalization needed
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    std::array<double, N> kvec;
                    double kmag2;

                    // Get wavevector and magnitude
                    phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);

                    // D^2 Phi_1LPT = -delta => F[Phi_1LPT] = F[delta] / k^2
                    auto value = delta_fourier.get_fourier_from_index(ind) / kmag2;
                    if (Local_x_start == 0 and ind == 0)
                        value = 0.0;
                    phi_1LPT_fourier.set_fourier_from_index(ind, value);
                }
            }

            //=================================================================================
            /// Generate the 2LPT potential defined as \f$ \Psi^{\rm 2LPT} = \nabla \phi^{\rm 2LPT} \f$ and \f$
            /// \nabla^2 \phi^{\rm 2LPT} = \ldots \f$. Returns the grid in Fourier space.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta The density contrast in fourier space
            /// @param[out] phi_2LPT The LPT potential in fourier space
            ///
            //=================================================================================
            template <int N>
            void compute_2LPT_potential_fourier(const FFTWGrid<N> & delta, FFTWGrid<N> & phi_2LPT) {

                // We require delta to exist and if phi_2LPT is allocated it must have the same size as delta
                assert_mpi(delta.get_nmesh() > 0,
                           "[compute_2LPT_potential_fourier] delta grid has to be already allocated!");

                // This is the -3/7 factor coming from D2 = -3/7 D1^2 for the growing mode in Einstein-deSitter
                const double prefactor_2LPT = -3.0 / 7.0;

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute 2LPT potential (require N(N-1)/2 = 3 for N = 3 temporary grids)\n";
#endif

                auto nleft = delta.get_n_extra_slices_left();
                auto nright = delta.get_n_extra_slices_right();
                auto Local_x_start = delta.get_local_x_start();
                auto Nmesh = delta.get_nmesh();
                auto local_nx = delta.get_local_nx();
                size_t NmeshTotFourier = delta.get_ntot_fourier();

                // Create grids
                FFTWGrid<N> phi_1LPT_ii[N];
                for (int i = 0; i < N; i++) {
                    phi_1LPT_ii[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ii[i].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_1LPT_ii_" +
                                                    std::to_string(i));
                }
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute [DiDi phi_1LPT] in fourier space\n";
#endif
                    // Compute phi_xx, phi_yy, ...
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    std::array<double, N> kvec;
                    double kmag2;

                    // Get wavevector and magnitude
                    delta.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);

                    // D^2Phi = -delta => F[DiDj Phi] = F[delta] kikj/k^2
                    auto value = delta.get_fourier_from_index(ind) / kmag2;
                    if (Local_x_start == 0 and ind == 0)
                        value = 0.0;

                    for (int idim = 0; idim < N; idim++) {
                        phi_1LPT_ii[idim].set_fourier_from_index(ind, value * kvec[idim] * kvec[idim]);
                    }
                }

                // Fourier transform
                for (int idim = 0; idim < N; idim++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transform [DiDi phi_1LPT] to real space: " << idim + 1 << " / " << N
                                  << "\n";
#endif
                    phi_1LPT_ii[idim].fftw_c2r();
                }

                // Crete output grid
                phi_2LPT = FFTWGrid<N>(Nmesh, nleft, nright);
                phi_2LPT.add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_2LPT");

                // Copy over source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Add 0.5[(D^2 phi_1LPT)^2 - DiDi phi_1LPT^2] to real space grid containing "
                                 "(D^2phi_2LPT) \n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < local_nx; islice++) {
                    for (auto & real_index : phi_2LPT.get_real_range(islice, islice + 1)) {
                        auto laplacian = 0.0, sum_squared = 0.0;
                        for (int idim = 0; idim < N; idim++) {
                            auto curpsi = phi_1LPT_ii[idim].get_real_from_index(real_index);
                            laplacian += curpsi;
                            sum_squared += curpsi * curpsi;
                        }
                        auto value = 0.5 * (laplacian * laplacian - sum_squared);
                        phi_2LPT.set_real_from_index(real_index, value);
                    }
                }

                // Free memory
                for (int idim = 0; idim < N; idim++)
                    phi_1LPT_ii[idim].free();

                // Create grids
                const int num_pairs = (N * (N - 1)) / 2;
                FFTWGrid<N> phi_1LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ij[i].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_1LPT_ij_" +
                                                    std::to_string(i));
                }

                // Compute phi_xixj for all pairs of i,j
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute [DiDj phi_1LPT] in fourier space\n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    std::array<double, N> kvec;
                    double kmag2;

                    // Get wavevector and magnitude
                    delta.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);

                    auto value = delta.get_fourier_from_index(ind) / kmag2;
                    if (Local_x_start == 0 && ind == 0)
                        value = 0.0;

                    int pair = 0;
                    for (int idim1 = 0; idim1 < N; idim1++) {
                        for (int idim2 = idim1 + 1; idim2 < N; idim2++) {
                            phi_1LPT_ij[pair++].set_fourier_from_index(ind, kvec[idim1] * kvec[idim2] * value);
                        }
                    }
                }

                // Fourier transform
                for (int pair = 0; pair < num_pairs; pair++) {
#ifdef DEBUG_LPT
                    if (FML::ThisTask == 0)
                        std::cout << "Fourier transform [DiDj phi_1LPT] to real space: " << pair + 1 << " / "
                                  << num_pairs << "\n";
#endif
                    phi_1LPT_ij[pair].fftw_c2r();
                }

                // Copy over source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Add [-DiDjphi_1LPT^2] to real space grid containing (D^2phi_2LPT) \n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < local_nx; islice++) {
                    for (auto & real_index : phi_2LPT.get_real_range(islice, islice + 1)) {
                        auto sum_squared = 0.0;
                        for (int pair = 0; pair < num_pairs; pair++) {
                            auto curpsi = phi_1LPT_ij[pair].get_real_from_index(real_index);
                            sum_squared += curpsi * curpsi;
                        }
                        auto value = (phi_2LPT.get_real_from_index(real_index) - sum_squared);

                        phi_2LPT.set_real_from_index(real_index, value);
                    }
                }

                // Free memory
                for (int pair = 0; pair < num_pairs; pair++) {
                    phi_1LPT_ij[pair].free();
                }

                // Fourier transform source
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Fourier transform [D^2phi_2LPT] to fourier space\n";
#endif
                phi_2LPT.fftw_r2c();

                // Divide by -k^2 and normalize
#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Computing phi_2LPT in fourier space\n";
#endif

#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    std::array<double, N> kvec;
                    double kmag2;

                    // Get wavevector and magnitude
                    phi_2LPT.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);

                    // Add in the -3/7 factor
                    auto value = phi_2LPT.get_fourier_from_index(ind);
                    value *= -prefactor_2LPT / kmag2;
                    if (Local_x_start == 0 and ind == 0)
                        value = 0.0;

                    phi_2LPT.set_fourier_from_index(ind, value);
                }
            }

            //===========================================================================================
            /// In this method we have so far given completely up what we do in the other methods
            /// and try to be as memory efficient as possible. We can reduce the memory footprint of
            /// this method with some work. Right now we allocate ~15 grid at the same time. It is possible to
            /// get that down to ~10. This method is not well tested!
            /// The units of the dlogDdt term is what sets the units of dPsidt
            /// In this methods the displacement field is assumed to be on the EdS/LCDM form
            /// Psi = D * Psi1LPT - 3/7 D^2 Psi2LPT + D^3 (...3LPT...) i.e. each term multiplied with powers of D
            ///
            /// @tparam N Dimensions we are working in (only 2 or 3)
            ///
            /// @param[in] delta_fourier A realisation of the density field.
            /// @param[out] Psi The displacement field
            /// @param[out] dPsidt The derivative of the displacement field (units is that of the next factor)
            /// @param[in] dlogDdt The logarithmic derivative of D at the time we want Psi in whatever units you want it
            /// to be.
            /// @param[in] DoverDini The ratio of D at the time we want Psi to itself at the time where delta is
            /// generated.
            ///
            //===========================================================================================
            template <int N>
            void compute_1LPT_2LPT_3LPT_displacment_field(const FFTWGrid<N> & delta_fourier,
                                                          std::vector<FFTWGrid<N>> & Psi,
                                                          std::vector<FFTWGrid<N>> & dPsidt,
                                                          double dlogDdt,
                                                          double DoverDini) {

                assert(N == 2 or N == 3);

                // Include the D x A term which appears at 3rd order
                // If not included then the resulting field is curl free and we can
                // if we want return the LPT potentials (though not implemented this)
                const bool include_curl_term = true;

                auto nleft = delta_fourier.get_n_extra_slices_left();
                auto nright = delta_fourier.get_n_extra_slices_right();
                auto local_x_start = delta_fourier.get_local_x_start();
                auto Nmesh = delta_fourier.get_nmesh();
                auto local_nx = delta_fourier.get_local_nx();
                size_t NmeshTotFourier = delta_fourier.get_ntot_fourier();

                // Store -k^2phi_1LPT
                FFTWGrid<N> phi_1LPT_fourier = delta_fourier;
                phi_1LPT_fourier.add_memory_label(
                    "FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::phi_1LPT_fourier");

                // Compute all terms phi_1LPT_ij. These are absolutely needed
                const int num_pairs = (N * (N + 1)) / 2;
                FFTWGrid<N> phi_1LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT_ij[i].add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::phi_1LPT_ij_" +
                                                    std::to_string(i));
                }

                if (FML::ThisTask == 0)
                    std::cout << "phi_1LPT_ij...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    double kmag2;
                    std::array<double, N> kvec;

                    // Get wavevector and magnitude
                    phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);
                    auto value = -phi_1LPT_fourier.get_fourier_from_index(ind) / kmag2;

                    // Deal with DC mode
                    if (ind == 0 and local_x_start == 0)
                        value = 0.0;

                    int pair = 0;
                    for (int idim1 = 0; idim1 < N; idim1++) {
                        for (int idim2 = idim1; idim2 < N; idim2++) {
                            phi_1LPT_ij[pair++].set_fourier_from_index(ind, -kvec[idim1] * kvec[idim2] * value);
                        }
                    }
                }

                // Fourier transform it all to real-space
                for (int i = 0; i < num_pairs; i++)
                    phi_1LPT_ij[i].fftw_c2r();

                // Compute 2LPT
                FFTWGrid<N> phi_2LPT_fourier(Nmesh, nleft, nright);
                phi_2LPT_fourier.add_memory_label(
                    "FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::phi_2LPT_fourier");

                if (FML::ThisTask == 0)
                    std::cout << "phi_2LPT...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < local_nx; islice++) {
                    for (auto & real_index : phi_2LPT_fourier.get_real_range(islice, islice + 1)) {
                        // Compute laplacian and sum of squares to get Sum_i,j Phi_iiPhi_jj Phi_ij^2
                        double laplacian = 0.0;
                        double sum_squared = 0.0;
                        int pair = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            auto phi_ij = phi_1LPT_ij[pair++].get_real_from_index(real_index);
                            laplacian += phi_ij;
                            sum_squared += phi_ij * phi_ij;
                            for (int idim2 = idim1 + 1; idim2 < N; idim2++) {
                                phi_ij = phi_1LPT_ij[pair++].get_real_from_index(real_index);
                                sum_squared += 2.0 * phi_ij * phi_ij;
                            }
                        }
                        phi_2LPT_fourier.set_real_from_index(real_index, 0.5 * (laplacian * laplacian - sum_squared));
                    }
                }

                // Back to fourier space: We now have -k^2 phi_2LPT in this grid
                phi_2LPT_fourier.fftw_r2c();

                // Time to compute all the phi_2LPT_ij terms
                FFTWGrid<N> phi_2LPT_ij[num_pairs];
                for (int i = 0; i < num_pairs; i++) {
                    phi_2LPT_ij[i] = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_2LPT_ij[i].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_2LPT_ij" +
                                                    std::to_string(i));
                }

                if (FML::ThisTask == 0)
                    std::cout << "phi_2LPT_ij...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    double kmag2;
                    std::array<double, N> kvec;

                    // Get wavevector and magnitude
                    phi_2LPT_fourier.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);
                    auto value = -phi_2LPT_fourier.get_fourier_from_index(ind) / kmag2;

                    // Deal with DC mode
                    if (ind == 0 and local_x_start == 0)
                        value = 0.0;

                    int pair = 0;
                    for (int idim1 = 0; idim1 < N; idim1++) {
                        for (int idim2 = idim1; idim2 < N; idim2++) {
                            phi_2LPT_ij[pair++].set_fourier_from_index(ind, -kvec[idim1] * kvec[idim2] * value);
                        }
                    }
                }

                // Compute phi_3LPT_a
                FFTWGrid<N> phi_3LPT_fourier(Nmesh, nleft, nright);
                phi_3LPT_fourier.add_memory_label(
                    "FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::phi_3LPT_fourier");
                // Compute phi_3LPT_b
                FFTWGrid<N> phi_3LPT_b(Nmesh, nleft, nright);
                phi_3LPT_b.add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::phi_3LPT_b");
                // And then finally the A-terms
                FFTWGrid<N> phi_3LPT_Avec[N];
                if constexpr (include_curl_term)
                    for (int idim = 0; idim < N; idim++) {
                        phi_3LPT_Avec[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                        phi_3LPT_Avec[idim].add_memory_label("FFTWGrid::compute_2LPT_potential_fourier::phi_3LPT_Avec" +
                                                             std::to_string(idim));
                    }

                if (FML::ThisTask == 0)
                    std::cout << "phi_3LPT...\n";
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < local_nx; islice++) {
                    for (auto & real_index : phi_2LPT_fourier.get_real_range(islice, islice + 1)) {

                        if constexpr (N == 2) {
                            auto psi1_xx = phi_1LPT_ij[0].get_real_from_index(real_index);
                            auto psi1_xy = phi_1LPT_ij[1].get_real_from_index(real_index);
                            auto psi1_yy = phi_1LPT_ij[2].get_real_from_index(real_index);
                            auto psi2_xx = phi_2LPT_ij[0].get_real_from_index(real_index);
                            auto psi2_xy = phi_2LPT_ij[1].get_real_from_index(real_index);
                            auto psi2_yy = phi_2LPT_ij[2].get_real_from_index(real_index);

                            auto value_a = psi1_xx * psi1_yy - psi1_xy * psi1_xy;

                            auto value_b = psi1_xx * psi2_yy - psi1_xy * psi2_xy;

                            phi_3LPT_fourier.set_real_from_index(real_index, value_a);
                            phi_3LPT_b.set_real_from_index(real_index, value_b);
                            if constexpr (include_curl_term) {
                                phi_3LPT_Avec[0].set_real_from_index(real_index, 0.0);
                                phi_3LPT_Avec[1].set_real_from_index(real_index, 0.0);
                            }
                        }
                        if constexpr (N == 3) {
                            auto psi1_xx = phi_1LPT_ij[0].get_real_from_index(real_index);
                            auto psi1_xy = phi_1LPT_ij[1].get_real_from_index(real_index);
                            auto psi1_zx = phi_1LPT_ij[2].get_real_from_index(real_index);
                            auto psi1_yy = phi_1LPT_ij[3].get_real_from_index(real_index);
                            auto psi1_yz = phi_1LPT_ij[4].get_real_from_index(real_index);
                            auto psi1_zz = phi_1LPT_ij[5].get_real_from_index(real_index);

                            auto psi2_xx = phi_2LPT_ij[0].get_real_from_index(real_index);
                            auto psi2_xy = phi_2LPT_ij[1].get_real_from_index(real_index);
                            auto psi2_zx = phi_2LPT_ij[2].get_real_from_index(real_index);
                            auto psi2_yy = phi_2LPT_ij[3].get_real_from_index(real_index);
                            auto psi2_yz = phi_2LPT_ij[4].get_real_from_index(real_index);
                            auto psi2_zz = phi_2LPT_ij[5].get_real_from_index(real_index);

                            auto value_a = psi1_xx * psi1_yy * psi1_zz;
                            value_a += 2.0 * psi1_xy * psi1_yz * psi1_zx;
                            value_a += -psi1_xx * psi1_yz * psi1_yz;
                            value_a += -psi1_yy * psi1_zx * psi1_zx;
                            value_a += -psi1_zz * psi1_xy * psi1_xy;

                            auto value_b = 0.5 * psi1_xx * (psi2_yy + psi2_zz);
                            value_b += 0.5 * psi1_yy * (psi2_zz + psi2_xx);
                            value_b += 0.5 * psi1_zz * (psi2_xx + psi2_yy);
                            value_b += -psi1_xy * psi2_xy - psi1_yz * psi2_yz - psi1_zx * psi2_zx;

                            phi_3LPT_fourier.set_real_from_index(real_index, value_a);
                            phi_3LPT_b.set_real_from_index(real_index, value_b);
                            if constexpr (include_curl_term) {
                                auto value_Avec_x = psi1_zx * psi2_xy - psi2_zx * psi1_xy;
                                value_Avec_x += psi1_yz * (psi2_yy - psi2_zz) - psi2_yz * (psi1_yy - psi1_zz);
                                auto value_Avec_y = psi1_xy * psi2_yz - psi2_xy * psi1_yz;
                                value_Avec_y += psi1_zx * (psi2_zz - psi2_xx) - psi2_zx * (psi1_zz - psi1_xx);
                                auto value_Avec_z = psi1_yz * psi2_zx - psi2_yz * psi1_zx;
                                value_Avec_z += psi1_xy * (psi2_xx - psi2_yy) - psi2_xy * (psi1_xx - psi1_yy);
                                phi_3LPT_Avec[0].set_real_from_index(real_index, value_Avec_x);
                                phi_3LPT_Avec[1].set_real_from_index(real_index, value_Avec_y);
                                phi_3LPT_Avec[2].set_real_from_index(real_index, value_Avec_z);
                            }
                        }

                        /* General method for the b term... but determinant is messy so we do it explicit
                        // Compute laplacian and sum of squares
                        double laplacian1 = 0.0;
                        double laplacian2 = 0.0;
                        double sum_squared = 0.0;
                        int pair1 = 0, pair2 = 0;
                        for (int idim1 = 0; idim1 < N; idim1++) {
                            auto phi1_ij =
                                phi_1LPT_ij[pair1++].get_real_from_index(real_index);
                            auto phi2_ij =
                                phi_2LPT_ij[pair2++].get_real_from_index(real_index);
                            laplacian1 += phi1_ij;
                            laplacian2 += phi2_ij;
                            sum_squared += phi1_ij * phi2_ij;
                            for (int idim2 = idim1+1; idim2 < N; idim2++) {
                                phi1_ij = phi_1LPT_ij[pair1++].get_real_from_index(real_index);
                                phi2_ij = phi_2LPT_ij[pair2++].get_real_from_index(real_index);
                                sum_squared += 2.0 * phi1_ij * phi2_ij;
                            }
                        }
                        phi_3LPT_b.set_real_from_index(real_index, 0.5 * (laplacian1 * laplacian2 - sum_squared));
                        */
                    }
                }

                // Free up memory
                for (int i = 0; i < num_pairs; i++) {
                    phi_1LPT_ij[i].free();
                    phi_2LPT_ij[i].free();
                }

                // Fourier transform and voila we have -k^2phi_3LPT_a stored in phi_3LPT_a
                phi_3LPT_fourier.fftw_r2c();

                // Fourier transform and voila we have -k^2phi_3LPT_b stored in phi_3LPT_b
                phi_3LPT_b.fftw_r2c();

                // Fourier transform and voila we have -k^2phi_3LPT_Avec stored in phi_3LPT_Avec
                if constexpr (include_curl_term) {
                    for (int idim = 0; idim < N; idim++)
                        phi_3LPT_Avec[idim].fftw_r2c();
                }

                // Make the displacment field
                for (int idim = 0; idim < N; idim++) {
                    Psi[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                    Psi[idim].add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::Psi" +
                                               std::to_string(idim));
                    dPsidt[idim] = FFTWGrid<N>(Nmesh, nleft, nright);
                    dPsidt[idim].add_memory_label("FFTWGrid::compute_1LPT_2LPT_3LPT_potential_fourier::dPsidt" +
                                                  std::to_string(idim));
                }

                if (FML::ThisTask == 0)
                    std::cout << "Psi...\n";

                // Now add up everything apart from the A-term
                std::complex<double> I{0, 1};
                const double DoverDini2 = DoverDini * DoverDini;
                const double DoverDini3 = DoverDini * DoverDini * DoverDini;
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {
                    double kmag2;
                    std::array<double, N> kvec;

                    // Get wavevector and magnitude
                    phi_3LPT_fourier.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);
                    double fac = -1.0 / kmag2;

                    // Deal with DC mode
                    if (local_x_start == 0 and ind == 0)
                        fac = 0.0;

                    auto value_1 = phi_1LPT_fourier.get_fourier_from_index(ind);
                    auto value_2 = phi_2LPT_fourier.get_fourier_from_index(ind);
                    auto value_3a = phi_3LPT_fourier.get_fourier_from_index(ind);
                    auto value_3b = phi_3LPT_b.get_fourier_from_index(ind);
                    auto value = -value_1 * DoverDini - 3.0 / 7.0 * value_2 * DoverDini2 +
                                 (value_3a / 3.0 - 10.0 / 21.0 * value_3b) * DoverDini3;
                    auto dvaluedt = -value_1 * DoverDini - 2.0 * 3.0 / 7.0 * value_2 * DoverDini2 +
                                    3.0 * (value_3a / 3.0 - 10.0 / 21.0 * value_3b) * DoverDini3;

                    if constexpr (N == 2) {
                        Psi[0].set_fourier_from_index(ind, -I * kvec[0] * value * fac);
                        Psi[1].set_fourier_from_index(ind, -I * kvec[1] * value * fac);
                        dPsidt[0].set_fourier_from_index(ind, -I * kvec[0] * dvaluedt * fac);
                        dPsidt[1].set_fourier_from_index(ind, -I * kvec[1] * dvaluedt * fac);
                    } else if (N == 3) {
                        std::array<std::complex<double>, N> A;
                        if constexpr (include_curl_term) {
                            A[0] = phi_3LPT_Avec[0].get_fourier_from_index(ind);
                            A[1] = phi_3LPT_Avec[1].get_fourier_from_index(ind);
                            A[2] = phi_3LPT_Avec[2].get_fourier_from_index(ind);
                        } else {
                            A.fill(0.0);
                        }

                        Psi[0].set_fourier_from_index(
                            ind,
                            (-I * kvec[0] * value + I * DoverDini3 / 7.0 * (kvec[1] * A[2] - kvec[2] * A[1])) * fac);
                        Psi[1].set_fourier_from_index(
                            ind,
                            (-I * kvec[1] * value + I * DoverDini3 / 7.0 * (kvec[2] * A[0] - kvec[0] * A[2])) * fac);
                        Psi[2].set_fourier_from_index(
                            ind,
                            (-I * kvec[2] * value + I * DoverDini3 / 7.0 * (kvec[0] * A[1] - kvec[1] * A[0])) * fac);

                        fac *= dlogDdt;
                        dPsidt[0].set_fourier_from_index(
                            ind,
                            (-I * kvec[0] * dvaluedt + 3.0 * I * DoverDini3 / 7.0 * (kvec[1] * A[2] - kvec[2] * A[1])) *
                                fac);
                        dPsidt[1].set_fourier_from_index(
                            ind,
                            (-I * kvec[1] * dvaluedt + 3.0 * I * DoverDini3 / 7.0 * (kvec[2] * A[0] - kvec[0] * A[2])) *
                                fac);
                        dPsidt[2].set_fourier_from_index(
                            ind,
                            (-I * kvec[2] * dvaluedt + 3.0 * I * DoverDini3 / 7.0 * (kvec[0] * A[1] - kvec[1] * A[0])) *
                                fac);
                    }
                }

                // Free up memory.. though not needed as we exit now anyway
                phi_1LPT_fourier.free();
                phi_2LPT_fourier.free();
                phi_3LPT_b.free();
                phi_3LPT_fourier.free();

                // Fourier transform to real space and we are done
                for (int idim = 0; idim < N; idim++) {
                    Psi[idim].fftw_c2r();
                    dPsidt[idim].fftw_c2r();
                }
            }
        } // namespace LPT
    }     // namespace COSMOLOGY
} // namespace FML
#endif
