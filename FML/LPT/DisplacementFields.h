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
                assert_mpi(
                    phi.get_nmesh() > 0,
                    "[from_LPT_potential_to_displacement_vector_scaledependent] phi grid has to be already allocated");

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
            /// Generate the 1LPT potential defined as \f$ \Psi^{\rm 1LPT} = \nabla \phi^{\rm 1LPT} \f$ and \f$ \nabla^2
            /// \phi^{\rm 1LPT} = -\delta \f$. Returns it in Fourier space.
            ///
            /// @tparam N The dimension of the grid
            ///
            /// @param[in] delta The density contrast in fourier space
            /// @param[out] phi_1LPT The LPT potential in fourier space
            ///
            //=================================================================================
            template <int N>
            void compute_1LPT_potential_fourier(const FFTWGrid<N> & delta, FFTWGrid<N> & phi_1LPT) {

                // We require delta to exist and if phi_1LPT is allocated it must have the same size as delta
                assert_mpi(delta.get_nmesh() > 0,
                           "[compute_1LPT_potential_fourier] delta grid has to be already allocated!");

#ifdef DEBUG_LPT
                if (FML::ThisTask == 0)
                    std::cout << "Compute 1LPT potential\n";
#endif

                auto Nmesh_phi = phi_1LPT.get_nmesh();

                auto nleft = delta.get_n_extra_slices_left();
                auto nright = delta.get_n_extra_slices_right();
                auto Local_x_start = delta.get_local_x_start();
                auto Nmesh = delta.get_nmesh();
                size_t NmeshTotFourier = delta.get_ntot_fourier();

                // Create 1LPT grid
                if (Nmesh_phi == 0) {
                    phi_1LPT = FFTWGrid<N>(Nmesh, nleft, nright);
                    phi_1LPT.add_memory_label("FFTWGrid::compute_1LPT_potential_fourier::phi_1LPT");
                }

                // Divide grid by k^2. Assuming delta was created in fourier-space so no FFTW normalization needed
                std::array<double, N> kvec;
                double kmag2;
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {

                    // Get wavevector and magnitude
                    phi_1LPT.get_fourier_wavevector_and_norm2_by_index(ind, kvec, kmag2);

                    // D^2 Phi_1LPT = -delta => F[Phi_1LPT] = F[delta] / k^2
                    auto value = delta.get_fourier_from_index(ind) / kmag2;
                    if (Local_x_start == 0 && ind == 0)
                        value = 0.0;
                    phi_1LPT.set_fourier_from_index(ind, value);
                }
            }

            //=================================================================================
            /// Generate the 2LPT potential defined as \f$ \Psi^{\rm 2LPT} = \nabla \phi^{\rm 2LPT} \f$ and \f$ \nabla^2
            /// \phi^{\rm 2LPT} = \ldots \f$. Returns the grid in Fourier space.
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
                std::array<double, N> kvec;
                double kmag;
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {

                    // Get wavevector and magnitude xxx Get norm here
                    delta.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);

                    // D^2Phi = -delta => F[DiDj Phi] = F[delta] kikj/k^2
                    auto value = delta.get_fourier_from_index(ind) / (kmag * kmag);
                    if (Local_x_start == 0 && ind == 0)
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
                    std::cout
                        << "Add 0.5[(D^2 phi_1LPT)^2 - DiDi phi_1LPT^2] to real space grid containing (D^2phi_2LPT) \n";
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
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {

                    // Get wavevector and magnitude
                    delta.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);

                    auto value = delta.get_fourier_from_index(ind) / (kmag * kmag);
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
                for (size_t ind = 0; ind < NmeshTotFourier; ind++) {

                    // Get wavevector and magnitude
                    phi_2LPT.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);

                    auto value = phi_2LPT.get_fourier_from_index(ind);
                    value *= -1.0 / (kmag * kmag);
                    if (Local_x_start == 0 && ind == 0)
                        value = 0.0;

                    phi_2LPT.set_fourier_from_index(ind, value);
                }
            }
        } // namespace LPT
    }     // namespace COSMOLOGY
} // namespace FML

#endif
