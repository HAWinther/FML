#ifndef COMPUTEPOWERSPECTRUM_HEADER
#define COMPUTEPOWERSPECTRUM_HEADER

#include <array>
#include <cassert>
#include <iostream>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h> // Only for compute_multipoles from particles

// The classes that define how to bin and return the data
#include <FML/ComputePowerSpectra/BispectrumBinning.h>
#include <FML/ComputePowerSpectra/PolyspectrumBinning.h>
#include <FML/ComputePowerSpectra/PowerSpectrumBinning.h>

namespace FML {

    //================================================================================
    /// This namespace deals with computing correlations functions (N-point functions)
    /// in real and fourier space.
    //================================================================================

    namespace CORRELATIONFUNCTIONS {

        using namespace FML::INTERPOLATION;

        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;

        //================================================================================
        // Keep track of everything we need for a binned power-spectrum
        // The k-spacing (linear or log or ...), the count in each bin,
        // the power in each bin etc. Its through this interface the results
        // of the methods below are given. See PowerSpectrumBinning.h for how it works
        //================================================================================

        // The powerspectrum result, see PowerspectrumBinning.h
        template <int N>
        class PowerSpectrumBinning;

        // The bispectrum result, see BispectrumBinning.h
        template <int N>
        class BispectrumBinning;

        // General polyspectrum result, see PolyspectrumBinning.h
        template <int N, int ORDER>
        class PolyspectrumBinning;

        //================================================================================
        /// @brief Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
        /// Fourier transform, decolvolve the window function for the assignement above,
        /// bin up power-spectrum and subtract shot-noise 1/NumPartTotal
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] pofk The binned power-spectrum. We required it to be initialized with the number of bins, kmin
        /// and kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        ///
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum(int Ngrid,
                                    T * part,
                                    size_t NumPart,
                                    size_t NumPartTotal,
                                    PowerSpectrumBinning<N> & pofk,
                                    std::string density_assignment_method);

        //================================================================================
        /// @brief Assign particles to a grid and an interlaced grid (displaced by dx/2 in all directions)
        /// Fourier transform both and add the together to cancel the leading aliasing contributions
        /// Decolvolve the window function for the assignements above,
        /// bin up power-spectrum and subtract shot-noise 1/NumPartTotal
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] pofk The binned power-spectrum. We required it to be initialized with the number of bins, kmin
        /// and kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        ///
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum_interlacing(int Ngrid,
                                                T * part,
                                                size_t NumPart,
                                                size_t NumPartTotal,
                                                PowerSpectrumBinning<N> & pofk,
                                                std::string density_assignment_method);

        //================================================================================
        /// @brief Brute force (but aliasing free) computation of the power spectrum.
        /// Loop over all grid-cells and all particles and add up contribution and subtracts shot-noise term.
        /// Since we need to combine all particles with all cells this is not easiy parallelizable with MPI
        /// so we assume all CPUs have exactly the same particles when this is run on more than 1 MPI tasks (so best run
        /// just using OpenMP).
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task. With MPI assumes all tasks have the same
        /// particles.
        /// @param[out] pofk The binned power-spectrum. We required it to be initialized with the number of bins, kmin
        /// and kmax.
        ///
        //================================================================================
        template <int N, class T>
        void
        compute_power_spectrum_direct_summation(int Ngrid, T * part, size_t NumPart, PowerSpectrumBinning<N> & pofk);

        //==========================================================================================
        /// @brief Compute the power-spectrum of a fourier grid. The result has no scales. Get
        /// scales by calling pofk.scale(boxsize) which does k *= 1/Boxsize and
        /// pofk *= Boxsize^N once spectrum has been computed.
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] pofk Binned power-spectrum
        ///
        //==========================================================================================
        template <int N>
        void bin_up_power_spectrum(FFTWGrid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk);

        //================================================================================
        /// @brief Compute power-spectrum multipoles (P0,P1,...,Pn-1) from a Fourier grid
        /// where we have a fixed line_of_sight_direction (typical coordinate axes like (0,0,1)).
        /// Pell contains P0,P1,P2,...Pell_max where ell_max = n-1 is the size of Pell
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid The fourier grid to compute the multipoles from.
        /// @param[out] Pell Vector of power-spectrum binnings. The size of Pell is the maximum ell to compute.
        /// All binnings has to have nbins, kmin and kmax set. At the end Pell[ ell ] is a binning of P_ell(k).
        /// @param[in] line_of_sight_direction The line of sight direction, e.g. (0,0,1) for the z-axis.
        ///
        //================================================================================
        template <int N>
        void compute_power_spectrum_multipoles(FFTWGrid<N> & fourier_grid,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::vector<double> line_of_sight_direction);

        //================================================================================
        /// @brief A simple power-spectrum estimator for multipoles in simulations - nothing fancy.
        /// Displacing particles from realspace to redshift-space using their velocities
        /// along each of the coordinate axes. Result is the mean of this.
        /// Deconvolving the window-function and subtracting shot-noise (1/NumPartTotal) for monopole.
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Particles in the form of a MPIParticle container
        /// @param[in] velocity_to_displacement Factor to convert a velocity to a displacement.
        /// This is c / ( aH(a) Boxsize ) for peculiar and c / (H(a)Boxsize) for comoving velocities
        /// At z = 0 velocity_to_displacement = 1.0/(100 * Boxsize) when Boxsize is in Mpc/h
        /// @param[out] Pell Vector of power-spectrum binnings. The size of Pell is the maximum ell to compute.
        /// All binnings has to have nbins, kmin and kmax set. At the end Pell[ ell ] is a binning of P_ell(k).
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS) to use.
        ///
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum_multipoles(int Ngrid,
                                               FML::PARTICLE::MPIParticles<T> & part,
                                               double velocity_to_displacement,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::string density_assignment_method);

        //================================================================================
        /// @brief Computes the bispectrum B(k1,k2,k3) from particles
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] bofk The binned bispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        ///
        //================================================================================
        template <int N, class T>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N> & bofk,
                                std::string density_assignment_method);

        //================================================================================
        /// @brief Computes the bispectrum B(k1,k2,k3) from a fourier grid
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] bofk The binned bispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N>
        void compute_bispectrum(FFTWGrid<N> & fourier_grid, BispectrumBinning<N> & bofk);

        //================================================================================
        /// @brief Computes the polyspectrum P(k1,k2,k3,...) from particles.
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam ORDER The order. 2 is the power-spectrum, 3 is the bispectrum, 4 is the trispectrum.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] polyofk The binned polyspectrum. We required it to be initialized with the number of bins, kmin
        /// and kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        ///
        //================================================================================
        template <int N, class T, int ORDER>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER> & polyofk,
                                  std::string density_assignment_method);

        //================================================================================
        /// @brief Computes the polyspectrum P(k1,k2,k3,...) from a fourier grid
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam ORDER The order. 2 is the power-spectrum, 3 is the bispectrum, 4 is the trispectrum.
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] polyofk The binned polyspectrum.
        ///
        //================================================================================
        template <int N, int ORDER>
        void compute_polyspectrum(FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, ORDER> & polyofk);

        //=====================================================================
        //=====================================================================

        //==========================================================================================
        // Compute the power-spectrum multipoles of a fourier grid assuming a fixed line of sight
        // direction (typically coordinate axes). Provide Pell with [ell+1] initialized binnings to compute
        // the first 0,1,...,ell multipoles The result has no scales. Get scales by scaling
        // PowerSpectrumBinning using scale(kscale, pofkscale) with kscale = 1/Boxsize
        // and pofkscale = Boxsize^N once spectrum has been computed
        //==========================================================================================
        template <int N>
        void compute_power_spectrum_multipoles(FFTWGrid<N> & fourier_grid,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::vector<double> line_of_sight_direction) {

            assert_mpi(line_of_sight_direction.size() == N,
                       "[compute_power_spectrum_multipoles] Line of sight direction has wrong number of dimensions\n");
            assert_mpi(Pell.size() > 0, "[compute_power_spectrum_multipoles] Pell must have size > 0\n");
            assert_mpi(fourier_grid.get_nmesh() > 0, "[compute_power_spectrum_multipoles] grid must have Nmesh > 0\n");

            int Nmesh = fourier_grid.get_nmesh();
            auto Local_nx = fourier_grid.get_local_nx();
            auto Local_x_start = fourier_grid.get_local_x_start();

            // Norm of LOS vector
            double rmag = 0.0;
            for (int idim = 0; idim < N; idim++)
                rmag += line_of_sight_direction[idim] * line_of_sight_direction[idim];
            rmag = std::sqrt(rmag);
            assert_mpi(rmag > 0.0, "[compute_power_spectrum_multipoles] Line of sight vector has zero length\n");

            // Initialize binning just in case
            for (size_t ell = 0; ell < Pell.size(); ell++)
                Pell[ell].reset();

                // Bin up mu^k |delta|^2
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : fourier_grid.get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode( k=0)

                    // Special treatment of k = 0 plane
                    int last_coord = fourier_index % (Nmesh / 2 + 1);
                    double weight = last_coord > 0 and last_coord < Nmesh / 2 ? 2.0 : 1.0;

                    // Compute kvec, |kvec| and |delta|^2
                    fourier_grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                    auto delta = fourier_grid.get_fourier_from_index(fourier_index);
                    double power = std::norm(delta);

                    // Compute mu = k_vec*r_vec
                    double mu2 = 0.0;
                    for (int idim = 0; idim < N; idim++)
                        mu2 += kvec[idim] * line_of_sight_direction[idim];
                    mu2 /= (kmag * rmag);
                    mu2 = mu2 * mu2;

                    // Add to bin |delta|^2, |delta|^2mu^2, |delta^2|mu^4, ...
                    double mutotwoell = 1.0;
                    for (size_t ell = 0; ell < Pell.size(); ell += 2) {
                        Pell[ell].add_to_bin(kmag, power * mutotwoell, weight);
                        mutotwoell *= mu2;
                    }
                }
            }

#ifdef USE_MPI
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                MPI_Allreduce(MPI_IN_PLACE, Pell[ell].pofk.data(), Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, Pell[ell].count.data(), Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, Pell[ell].kbin.data(), Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
#endif

            // Normalize
            for (size_t ell = 0; ell < Pell.size(); ell++)
                Pell[ell].normalize();

            // Binomial coefficient
            auto binomial = [](const double n, const double k) -> double {
                double res = 1.0;
                for (int i = 0; i < k; i++) {
                    res *= double(n - i) / double(k - i);
                }
                return res;
            };

            // P_ell(x) = Sum_{k=0}^{ell/2} summand_legendre_polynomial * x^(ell - 2k)
            auto summand_legendre_polynomial = [&](const int k, const int ell) -> double {
                double sign = (k % 2) == 0 ? 1.0 : -1.0;
                return sign * binomial(ell, k) * binomial(2 * ell - 2 * k, ell) / std::pow(2.0, ell);
            };

            // Go from <mu^k |delta|^2> to (2ell+1) <L_ell(mu) |delta|^2>
            std::vector<std::vector<double>> temp;
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                std::vector<double> sum(Pell[0].pofk.size(), 0.0);
                for (size_t k = 0; k <= ell / 2; k++) {
                    std::vector<double> & mu_power = Pell[ell - 2 * k].pofk;
                    for (size_t i = 0; i < sum.size(); i++)
                        sum[i] += mu_power[i] * summand_legendre_polynomial(k, ell) * (2 * ell + 1);
                }
                temp.push_back(sum);
            }

            // Copy over data. We now have P0,P1,... in Pell
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                Pell[ell].pofk = temp[ell];
            }
        }

        // Bin up the power-spectrum of a given fourier grid
        template <int N>
        void bin_up_power_spectrum(FFTWGrid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk) {

            assert_mpi(fourier_grid.get_nmesh() > 0, "[bin_up_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_power_spectrum] Binning has inconsistent parameters\n");

            int Nmesh = fourier_grid.get_nmesh();
            auto Local_nx = fourier_grid.get_local_nx();
            auto Local_x_start = fourier_grid.get_local_x_start();

            // Initialize binning just in case
            pofk.reset();

            // Bin up P(k)
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : fourier_grid.get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode( k=0)

                    // Special treatment of k = 0 plane (Safer way: fetch coord)
                    // auto coord = fourier_grid.get_fourier_coord_from_index(fourier_index);
                    // int last_coord = coord[N-1];
                    int last_coord = fourier_index % (Nmesh / 2 + 1);
                    double weight = last_coord > 0 && last_coord < Nmesh / 2 ? 2.0 : 1.0;

                    auto delta = fourier_grid.get_fourier_from_index(fourier_index);
                    auto delta_norm = std::norm(delta);

                    // Add norm to bin
                    fourier_grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                    pofk.add_to_bin(kmag, delta_norm, weight);
                }
            }

            // Normalize to get P(k) (this communicates over tasks)
            pofk.normalize();
        }

        // Brute force. Add particles to the grid using direct summation
        // This gives alias free P(k), but scales as O(Npart)*O(Nmesh^N)
        template <int N, class T>
        void
        compute_power_spectrum_direct_summation(int Ngrid, T * part, size_t NumPart, PowerSpectrumBinning<N> & pofk) {

            static_assert(
                FML::PARTICLE::has_get_pos<T>(),
                "[compute_power_spectrum_direct_summation] Particle class needs to have positions to use this method");

#ifdef USE_MPI
            // Simple check to see if all tasks do have the same particles
            if (FML::NTasks > 1) {
                long long int tmp1 = NumPart;
                MPI_Allreduce(MPI_IN_PLACE, &tmp1, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
                long long int tmp2 = NumPart;
                MPI_Allreduce(MPI_IN_PLACE, &tmp2, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
                double x = FML::PARTICLE::GetPos(part[0])[0];
                double y = x;
                MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &y, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                assert_mpi(tmp1 == tmp2 and std::abs(x - y) < 1e-10,
                           "[direct_summation_power_spectrum] All tasks must have the same particles for this method "
                           "to work\n");
            }
#endif

            assert_mpi(Ngrid > 0, "[direct_summation_power_spectrum] Ngrid > 0 required\n");
            if (NTasks > 1 and ThisTask == 0)
                std::cout << "[direct_summation_power_spectrum] Warning: this method assumes all tasks have the same "
                             "particles\n";

            const std::complex<double> I(0, 1);
            const double norm = 1.0 / double(NumPart);

            FFTWGrid<N> density_k(Ngrid, 1, 1);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum_direct_summation::density_k");
            density_k.set_grid_status_real(false);

            for (auto && fourier_index : density_k.get_fourier_range()) {
                auto kvec = density_k.get_fourier_wavevector_from_index(fourier_index);
                double real = 0.0;
                double imag = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : real, imag)
#endif
                for (size_t i = 0; i < NumPart; i++) {
                    auto * x = FML::PARTICLE::GetPos(part[i]);
                    double kx = 0.0;
                    for (int idim = 0; idim < N; idim++) {
                        kx += kvec[idim] * x[idim];
                    }
                    auto val = std::exp(-kx * I);
                    real += val.real();
                    imag += val.imag();
                }

                std::complex<double> sum = {real, imag};
                if (ThisTask == 0 and fourier_index == 0)
                    sum -= 1.0;
                density_k.set_fourier_from_index(fourier_index, sum * norm);
            }

            // Bin up the power-spectrum
            bin_up_power_spectrum<N>(density_k, pofk);

            // Subtract shot-noise
            for (int i = 0; i < pofk.n; i++)
                pofk.pofk[i] -= 1.0 / double(NumPart);
        }

        // Simple method to estimate multipoles from simulation data
        // Take particles in realspace and use their velocity to put them into
        // redshift space. Fourier transform and compute multipoles from this like in the method above.
        // We do this for all coordinate axes and return the mean P0,P1,... we get from this
        // velocity_to_displacement is factor to convert your velocity to a coordinate shift in [0,1)
        // e.g. c/(a H Box) with H ~ 100 h km/s/Mpc and Box boxsize in Mpc/h if velocities are peculiar
        template <int N, class T>
        void compute_power_spectrum_multipoles(int Ngrid,
                                               FML::PARTICLE::MPIParticles<T> & part,
                                               double velocity_to_displacement,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::string density_assignment_method) {

            // Sanity check
            static_assert(FML::PARTICLE::has_get_pos<T>(),
                          "[compute_power_spectrum_multipoles] Particle class needs to have positions to use "
                          "this method");
            static_assert(
                FML::PARTICLE::has_get_vel<T>(),
                "[compute_power_spectrum_multipoles] Particle class needs to have velocity to use this method");

            // Set how many extra slices we need for the density assignment to go smoothly
            auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second;

            // Initialize binning just in case
            for (size_t ell = 0; ell < Pell.size(); ell++)
                Pell[ell].reset();

            // Set a binning for each axes
            std::vector<std::vector<PowerSpectrumBinning<N>>> Pell_all(N);
            for (int dir = 0; dir < N; dir++) {
                Pell_all[dir] = Pell;
            }

            // Allocate density grid
            FFTWGrid<N> density_k(Ngrid, nleft, nright);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum_multipoles::density_k");

            // Loop over all the axes we are going to put the particles
            // into redshift space
            for (int dir = 0; dir < N; dir++) {

                // Set up binning for current axis
                std::vector<PowerSpectrumBinning<N>> Pell_current = Pell_all[dir];
                for (size_t ell = 0; ell < Pell_current.size(); ell++)
                    Pell_current[ell].reset();

                // Make line of sight direction unit vector
                std::vector<double> line_of_sight_direction(N, 0.0);
                line_of_sight_direction[dir] = 1.0;

                // Displace particles
                double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
                for (size_t i = 0; i < part.get_npart(); i++) {
                    auto * pos = FML::PARTICLE::GetPos(part[i]);
                    auto * vel = FML::PARTICLE::GetVel(part[i]);
                    double vdotr = 0.0;
                    for (int idim = 0; idim < N; idim++) {
                        vdotr += vel[idim] * line_of_sight_direction[idim];
                    }
                    for (int idim = 0; idim < N; idim++) {
                        pos[idim] += vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
                        // Periodic boundary conditions
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                    }

                    max_disp = std::max(max_disp, std::fabs(vdotr));
                }
                max_disp *= velocity_to_displacement;
                FML::MaxOverTasks(&max_disp);
                if (FML::ThisTask == 0)
                    std::cout << "Maximum displacement: " << max_disp << "\n";

                // Only displacements along the x-axis can trigger communication needs so we can avoid some
                // calls
                if (dir == 0)
                    part.communicate_particles();

                // Bin particles to grid
                density_k.set_grid_status_real(true);
                particles_to_grid<N, T>(part.get_particles_ptr(),
                                        part.get_npart(),
                                        part.get_npart_total(),
                                        density_k,
                                        density_assignment_method);

                // Displace particles back
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < part.get_npart(); i++) {
                    auto * pos = FML::PARTICLE::GetPos(part[i]);
                    auto * vel = FML::PARTICLE::GetVel(part[i]);
                    double vdotr = 0.0;
                    for (int idim = 0; idim < N; idim++) {
                        vdotr += vel[idim] * line_of_sight_direction[idim];
                    }
                    for (int idim = 0; idim < N; idim++) {
                        pos[idim] -= vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
                        // Periodic boundary conditions
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                    }
                }
                // Only displacements along the x-axis can trigger communication needs so we can avoid one call
                if (dir == 0)
                    part.communicate_particles();

                // Fourier transform
                density_k.fftw_r2c();

                // Deconvolve window function
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

                // Compute power-spectrum multipoles
                compute_power_spectrum_multipoles(density_k, Pell_current, line_of_sight_direction);

                // Assign back
                Pell_all[dir] = Pell_current;
            }

            // Normalize
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                for (int dir = 0; dir < N; dir++) {
                    Pell[ell] += Pell_all[dir][ell];
                }
            }
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                for (int i = 0; i < Pell[ell].n; i++) {
                    Pell[ell].pofk[i] /= double(N);
                    Pell[ell].count[i] /= double(N);
                    Pell[ell].kbin[i] /= double(N);
                }
            }

            // XXX Compute variance of pofk

            // Subtract shotnoise for monopole
            for (int i = 0; i < Pell[0].n; i++) {
                Pell[0].pofk[i] -= 1.0 / double(part.get_npart_total());
            }
        }

        //================================================================================
        // A simple power-spectrum estimator - nothing fancy
        // Deconvolving the window-function and subtracting shot-noise (1/NumPartTotal)
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum(int Ngrid,
                                    T * part,
                                    size_t NumPart,
                                    size_t NumPartTotal,
                                    PowerSpectrumBinning<N> & pofk,
                                    std::string density_assignment_method) {

            // Set how many extra slices we need for the density assignment to go smoothly
            auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            int nleft = nleftright.first;
            int nright = nleftright.second;

            // Bin particles to grid
            FFTWGrid<N> density_k(Ngrid, nleft, nright);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum::density_k");
            particles_to_grid<N, T>(part, NumPart, NumPartTotal, density_k, density_assignment_method);

            // Fourier transform
            density_k.fftw_r2c();

            // Deconvolve window function
            deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

            // Bin up power-spectrum
            bin_up_power_spectrum<N>(density_k, pofk);

            // Subtract shotnoise
            for (int i = 0; i < pofk.n; i++) {
                pofk.pofk[i] -= 1.0 / double(NumPartTotal);
            }
        }

        //======================================================================
        // Computes the power-spectum by using two interlaced grids
        // to reduce the effect of aliasing (allowing us to use a smaller Ngrid)
        // Deconvolves the window function and subtracts shot-noise
        //======================================================================
        template <int N, class T>
        void compute_power_spectrum_interlacing(int Ngrid,
                                                T * part,
                                                size_t NumPart,
                                                size_t NumPartTotal,
                                                PowerSpectrumBinning<N> & pofk,
                                                std::string density_assignment_method) {

            // Compute delta(k) using interlacing to reduce alias
            FFTWGrid<N> density_k;
            FML::INTERPOLATION::particles_to_fourier_grid_interlacing(
                density_k, Ngrid, part, NumPart, NumPartTotal, density_assignment_method);

            // Bin up power-spectrum
            bin_up_power_spectrum<N>(density_k, pofk);

            // Subtract shotnoise
            for (int i = 0; i < pofk.n; i++) {
                pofk.pofk[i] -= 1.0 / double(NumPartTotal);
            }
        }

        // https://arxiv.org/pdf/1506.02729.pdf
        // The general quadrupole estimator Eq. 20
        // P(k) = <delta0(k)delta2^*(k>>
        // We compute delta0 and delta2
        // template<int N>
        //  void quadrupole_estimator_3D(FFTWGrid<N> & density_real)
        //  {

        //    FFTWGrid<N> Q_xx, Q_yy, Q_zz, Q_xy, Q_yz, Q_zx;

        //    compute_multipole_Q_term(density_real, Q_xx, {0,0}, origin);
        //    compute_multipole_Q_term(density_real, Q_xy, {0,1}, origin);
        //    compute_multipole_Q_term(density_real, Q_yy, {1,1}, origin);
        //    Q_xx.fftw_r2c();
        //    Q_xy.fftw_r2c();
        //    Q_yy.fftw_r2c();
        //    if constexpr (N > 2){
        //      compute_multipole_Q_term(density_real, Q_yz, {1,2}, origin);
        //      compute_multipole_Q_term(density_real, Q_zx, {2,0}, origin);
        //      compute_multipole_Q_term(density_real, Q_zz, {2,2}, origin);
        //      Q_yz.fftw_r2c();
        //      Q_zx.fftw_r2c();
        //      Q_zz.fftw_r2c();
        //    }

        //    // Density to fourier space
        //    density_real.fftw_r2c();

        //    FFTWGrid<N> delta2 =  density_real;

        //    double kmag;
        //    std::array<double,N> kvec;
        //    for(auto & fourier_index : density_real.get_fourier_range()){
        //      density_real.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag2);

        //      // Unit kvector
        //      if(kmag > 0.0)
        //        for(auto & k : kvec)
        //          k /= kmag;

        //      auto Q_xx_of_k = Q_xx.get_fourier_from_index(fourier_index);
        //      auto Q_xy_of_k = Q_xy.get_fourier_from_index(fourier_index);
        //      auto Q_yy_of_k = Q_yy.get_fourier_from_index(fourier_index);

        //      auto Q_yz_of_k = Q_yz.get_fourier_from_index(fourier_index);
        //      auto Q_zx_of_k = Q_zx.get_fourier_from_index(fourier_index);
        //      auto Q_zz_of_k = Q_zz.get_fourier_from_index(fourier_index);
        //      auto delta0    = density_real.get_fourier_from_index(fourier_index);

        //      auto res = Q_xx_of_k * kvec[0] * kvec[0] + Q_yy_of_k * kvec[1] * kvec[1] +  2.0 * (Q_xy_of_k *
        //      kvec[0] * kvec[1])
        //        if constexpr (N > 2) res += Q_zz_of_k * kvec[2] * kvec[2] + 2.0 * (Q_yz_of_k * kvec[1] *
        //        kvec[2] + Q_zx_of_k * kvec[2] * kvec[0]);
        //      res = 1.5 * res - 0.5 * delta0;

        //      delta2.set_real_from_fourier(fourier_index, res);
        //    }
        //  }

        // Compute delta(x) -> delta(x) * xi * xj * xk * .... (i,j,k,... in Qindex)
        // where xi's being the i'th component of the unit norm line of sight direction
        // For the quadrupole we put Qindex = {ii,jj} and we need the 6 combinations xx,yy,zz,xy,yz,zx
        // For the hexadecapole we need {ii,jj,kk,ll} for 15 different combinations
        // xxxx + cyc (3), xxxy + cyc, xxyy + cyc and xxyz + cyc
        // template<int N>
        //  void compute_multipole_Q_term(
        //      FFTWGrid<N> & density_real,
        //      FFTWGrid<N> & Q_real,
        //      std::vector<int> Qindex,
        //      std::vector<double> origin)
        //  {
        //    assert(Qindex.size() > 0);
        //    assert(origin.size() == N);

        //    for(auto && real_index : density_real.get_real_range()){
        //      auto coord = density_real.get_coord_from_index(real_index);
        //      auto pos = density_real.get_real_position(coord);

        //      double norm = 0.0;
        //      for(int idim = 0; idim < N; idim++){
        //        pos[idim] -= origin[idim];
        //        norm += pos[idim]*pos[idim];
        //      }
        //      norm = std::sqrt(norm);
        //      for(int idim = 0; idim < N; idim++){
        //        pos[idim] /= norm;
        //      }
        //      auto value = density_real.get_real_from_index(real_index);
        //      for(auto ii: Qindex)
        //        value *= pos[ii];

        //      Q_real.set_real_from_index(real_index, value);
        //    }
        //  }

        // Computes the bispectrum B(k1,k2,k3) for *all* k1,k2,k3
        // One can make this much faster if one *just* wants say k1=k2=k3
        template <int N, class T>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N> & bofk,
                                std::string density_assignment_method) {

            // Bin particles to grid (use interlaced to reduce alias) and deconvolve window
            FFTWGrid<N> density_k;
            FML::INTERPOLATION::particles_to_fourier_grid_interlacing(
                density_k, Ngrid, part, NumPart, NumPartTotal, density_assignment_method);

            // Compute bispectrum
            compute_bispectrum<N>(density_k, bofk);
        }

        // Computes the polyspectrum P(k1,k2,k3,...)
        template <int N, class T, int ORDER>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER> & polyofk,
                                  std::string density_assignment_method) {

            // Bin particles to grid (use interlaced to reduce alias) and deconvolve window
            FFTWGrid<N> density_k;
            FML::INTERPOLATION::particles_to_fourier_grid_interlacing(
                density_k, Ngrid, part, NumPart, NumPartTotal, density_assignment_method);

            // Compute polyspectrum
            compute_polyspectrum<N, ORDER>(density_k, polyofk);
        }

        template <int N>
        void compute_bispectrum(FFTWGrid<N> & density_k, BispectrumBinning<N> & bofk) {

            // Reset binning
            bofk.reset();

            const auto Nmesh = density_k.get_nmesh();
            const auto Local_nx = density_k.get_local_nx();
            const int nbins = bofk.n;

            assert_mpi(nbins > 0, "[compute_bispectrum] nbins has to be >= 0\n");
            assert_mpi(Nmesh > 0, "[compute_bispectrum] grid is not allocated\n");

            // Now loop over bins and do FFTs
            std::vector<FFTWGrid<N>> F_k(nbins);
            std::vector<FFTWGrid<N>> N_k(nbins);

            // Set ranges for which we will compute F_k
            std::vector<double> khigh(nbins);
            std::vector<double> k_bin(nbins);
            std::vector<double> klow(nbins);
            double deltak = (bofk.k[1] - bofk.k[0]);
            for (int i = 0; i < nbins; i++) {
                F_k[i] = density_k;
                N_k[i] = density_k;
                F_k[i].add_memory_label("FFTWGrid::compute_bispectrum::F_" + std::to_string(i));
                N_k[i].add_memory_label("FFTWGrid::compute_bispectrum::N_" + std::to_string(i));
                N_k[i].fill_fourier_grid(0.0);

                if (i == 0) {
                    klow[i] = bofk.k[0];
                    khigh[i] = bofk.k[0] + (bofk.k[1] - bofk.k[0]) / 2.0;
                } else if (i < nbins - 1) {
                    klow[i] = khigh[i - 1];
                    khigh[i] = bofk.k[i] + (bofk.k[i + 1] - bofk.k[i]) / 2.0;
                } else {
                    klow[i] = khigh[i - 1];
                    khigh[i] = bofk.k[nbins - 1];
                }
                k_bin[i] = (khigh[i] + klow[i]) / 2.0;
            }

            // Compute how many configurations we have to store
            // This is (n+ORDER choose ORDER)
            // nbins_tot = 1;
            // int faculty = 1;
            // for(int i = 0; i < order-1; i++){
            //  nbins_tot *= (nbins+i);
            //  faculty *= (1+i);
            //}
            // nbins_tot /= faculty;
            // We just store all the symmmetry configurations as written

            // Set up results vector
            size_t nbins_tot = FML::power(size_t(nbins), 3);
            std::vector<double> & B123 = bofk.B123;
            std::vector<double> & N123 = bofk.N123;
            std::vector<double> & kmean_bin = bofk.kbin;
            std::vector<double> & pofk_bin = bofk.pofk;

            for (int i = 0; i < nbins; i++) {
#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "Computing bispectrum " << i + 1 << " / " << nbins
                              << " kbin: " << klow[i] / (2.0 * M_PI) << " -> " << khigh[i] / (2.0 * M_PI) << "\n";
                ;
#endif

                FFTWGrid<N> & grid = F_k[i];
                FFTWGrid<N> & count_grid = N_k[i];

                // For each bin get klow, khigh and deltak
                const double kmag_max = khigh[i];
                const double kmag_min = klow[i];
                const double kmag2_max = kmag_max * kmag_max;
                const double kmag2_min = kmag_min * kmag_min;

                //=====================================================
                // Bin weights, currently not in use. Basically some averaging about
                // k for each bin. Not tested so well, but gives very good results for local
                // nn-gaussianity (much less noisy estimates), but not so much in general
                //=====================================================
                // double sigma2 = 8.0 * std::pow( deltak ,2);
                // auto weight = [&](double kmag, double kbin){
                //  //return std::fabs(kmag - kbin) < deltak/2 ? 1.0 : 0.0;
                //  return 1.0/std::sqrt(2.0 * M_PI * sigma2) * std::exp( -0.5*(kmag - kbin)*(kmag -
                //  kbin)/sigma2 );
                //};

                // Loop over all cells XXX Add OpenMP
                double kmean = 0.0;
                double nk = 0;
                double kmag2;
                std::array<double, N> kvec;
                for (auto && fourier_index : grid.get_fourier_range()) {
                    grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                    //=====================================================
                    // Set to zero outside the bin (N_k already init to 0)
                    //=====================================================
                    if (kmag2 >= kmag2_max or kmag2 < kmag2_min) {
                        grid.set_fourier_from_index(fourier_index, 0.0);
                        count_grid.set_fourier_from_index(fourier_index, 0.0);
                    } else {
                        // Compute mean k in the bin
                        kmean += std::sqrt(kmag2);
                        pofk_bin[i] += std::norm(grid.get_fourier_from_index(fourier_index));
                        nk += 1.0;
                        count_grid.set_fourier_from_index(fourier_index, 1.0);
                    }

                    //=====================================================
                    // Alternative to the above: add with weights
                    //=====================================================
                    // double kmag = std::sqrt(kmag2);
                    // double fac = weight(kmag, k_bin[i]);
                    // kmean += kmag * fac;
                    // pofk_bin[i] += std::norm(grid.get_fourier_from_index(fourier_index)) * fac;
                    // nk += fac;
                    // grid.set_fourier_from_index(fourier_index, grid.get_fourier_from_index(fourier_index) *
                    // fac); count_grid.set_fourier_from_index(fourier_index, fac);
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &kmean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &pofk_bin[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &nk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
                // The mean k in the bin
                kmean_bin[i] = (nk == 0) ? k_bin[i] : kmean / double(nk);
                // Power spectrum in the bin
                pofk_bin[i] = (nk == 0) ? 0.0 : pofk_bin[i] / double(nk);

#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "kmean: " << kmean_bin[i] / (2.0 * M_PI) << "\n";
#endif

                // Transform to real space
                grid.fftw_c2r();
                count_grid.fftw_c2r();
            }

            // We now have F_k and N_k for all bins
            for (size_t i = 0; i < nbins_tot; i++) {
#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    if ((i * 10) / nbins_tot != ((i + 1) * 10) / nbins_tot)
                        std::cout << "Integrating up " << 100 * (i + 1) / nbins_tot << " %\n";
                ;
#endif

                // Current values of k1,k2,k3
                std::array<int, 3> ik;
                for (int ii = 0, n = 1; ii < 3; ii++, n *= nbins) {
                    ik[ii] = i / n % nbins;
                }
                // Only compute stuff for k1 <= k2 <= k3
                if (ik[0] > ik[1] or ik[1] > ik[2]) {
                    continue;
                }

                // No valid triangles if k1+k2 < k3 so just set too zero right away
                if (k_bin[ik[0]] + k_bin[ik[1]] < k_bin[ik[2]] - 3 * deltak / 2.) {
                    N123[i] = 0.0;
                    B123[i] = 0.0;
                    continue;
                }

                // Compute number of triangles in current bin (norm insignificant below)
                double N123_current = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : N123_current)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : N_k[0].get_real_range(islice, islice + 1)) {
                        double N1 = N_k[ik[0]].get_real_from_index(real_index);
                        double N2 = N_k[ik[1]].get_real_from_index(real_index);
                        double N3 = N_k[ik[2]].get_real_from_index(real_index);
                        N123_current += N1 * N2 * N3;
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &N123_current, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

                // Compute sum over triangles
                double F123_current = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : F123_current)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : F_k[0].get_real_range(islice, islice + 1)) {
                        auto F1 = F_k[ik[0]].get_real_from_index(real_index);
                        auto F2 = F_k[ik[1]].get_real_from_index(real_index);
                        auto F3 = F_k[ik[2]].get_real_from_index(real_index);
                        F123_current += F1 * F2 * F3;
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &F123_current, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
                // Normalize dx^N / (2pi)^N
                F123_current *= std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);
                N123_current *= std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);

                // Set the result
                B123[i] = (N123_current > 0.0) ? F123_current / N123_current : 0.0;
                N123[i] = (N123_current > 0.0) ? N123_current : 0.0;
            }

            // Set the values we didn't compute by using symmetry
            for (int i = 0; i < nbins; i++) {
                for (int j = 0; j < nbins; j++) {
                    for (int k = 0; k < nbins; k++) {
                        size_t index = (i * nbins + j) * nbins + k;

                        std::vector<int> inds{i, j, k};
                        std::sort(inds.begin(), inds.end(), std::less<int>());

                        size_t index0 = (inds[0] * nbins + inds[1]) * nbins + inds[2];
                        size_t index1 = (inds[0] * nbins + inds[2]) * nbins + inds[1];
                        size_t index2 = (inds[1] * nbins + inds[2]) * nbins + inds[0];
                        size_t index3 = (inds[1] * nbins + inds[0]) * nbins + inds[2];
                        size_t index4 = (inds[2] * nbins + inds[0]) * nbins + inds[1];
                        size_t index5 = (inds[2] * nbins + inds[1]) * nbins + inds[0];

                        if (B123[index0] == 0.0)
                            B123[index0] = B123[index];
                        if (B123[index1] == 0.0)
                            B123[index1] = B123[index];
                        if (B123[index2] == 0.0)
                            B123[index2] = B123[index];
                        if (B123[index3] == 0.0)
                            B123[index3] = B123[index];
                        if (B123[index4] == 0.0)
                            B123[index4] = B123[index];
                        if (B123[index5] == 0.0)
                            B123[index5] = B123[index];
                    }
                }
            }
        }

        template <int N, int ORDER>
        void compute_polyspectrum(FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, ORDER> & polyofk) {

            // Reset the binning
            polyofk.reset();

            const auto Nmesh = fourier_grid.get_nmesh();
            const auto Local_nx = fourier_grid.get_local_nx();
            const int nbins = polyofk.n;

            assert_mpi(nbins > 0, "[compute_polyspectrum] nbins has to be >=0\n");
            assert_mpi(Nmesh > 0, "[compute_polyspectrum] grid is not allocated\n");
            static_assert(ORDER > 1);

            // Now loop over bins and do FFTs
            std::vector<FFTWGrid<N>> F_k(nbins);
            std::vector<FFTWGrid<N>> N_k(nbins);

            // Set ranges for which we will compute F_k
            std::vector<double> khigh(nbins);
            std::vector<double> k_bin(nbins);
            std::vector<double> klow(nbins);
            double deltak = (polyofk.k[1] - polyofk.k[0]);
            for (int i = 0; i < nbins; i++) {
                F_k[i] = fourier_grid;
                N_k[i] = fourier_grid;
                F_k[i].add_memory_label("FFTWGrid::compute_polyspectrum::F_" + std::to_string(i));
                N_k[i].add_memory_label("FFTWGrid::compute_polyspectrum::N_" + std::to_string(i));
                N_k[i].fill_fourier_grid(0.0);

                if (i == 0) {
                    klow[i] = polyofk.k[0];
                    khigh[i] = polyofk.k[0] + (polyofk.k[1] - polyofk.k[0]) / 2.0;
                } else if (i < nbins - 1) {
                    klow[i] = khigh[i - 1];
                    khigh[i] = polyofk.k[i] + (polyofk.k[i + 1] - polyofk.k[i]) / 2.0;
                } else {
                    klow[i] = khigh[i - 1];
                    khigh[i] = polyofk.k[nbins - 1];
                }
                k_bin[i] = (khigh[i] + klow[i]) / 2.0;
            }

            // Set up results vector
            size_t nbins_tot = FML::power(size_t(nbins), ORDER);
            std::vector<double> & P123 = polyofk.P123;
            std::vector<double> & N123 = polyofk.N123;
            std::vector<double> & kmean_bin = polyofk.kbin;
            std::vector<double> & pofk_bin = polyofk.pofk;

            for (int i = 0; i < nbins; i++) {
#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "Computing polyspectrum " << i + 1 << " / " << nbins
                              << " kbin: " << klow[i] / (2.0 * M_PI) << " -> " << khigh[i] / (2.0 * M_PI) << "\n";
                ;
#endif

                FFTWGrid<N> & grid = F_k[i];
                FFTWGrid<N> & count_grid = N_k[i];

                // For each bin get klow, khigh and deltak
                const double kmag_max = khigh[i];
                const double kmag_min = klow[i];
                const double kmag2_max = kmag_max * kmag_max;
                const double kmag2_min = kmag_min * kmag_min;

                //=====================================================
                // Bin weights, currently not in use. Basically some averaging about
                // k for each bin. Not tested so well, but gives very good results for local
                // non-gaussianity (much less noisy estimates), but not so much in general
                //=====================================================
                // double sigma2 = 8.0 * std::pow( deltak ,2);
                // auto weight = [&](double kmag, double kbin){
                //  //return std::fabs(kmag - kbin) < deltak/2 ? 1.0 : 0.0;
                //  return 1.0/std::sqrt(2.0 * M_PI * sigma2) * std::exp( -0.5*(kmag - kbin)*(kmag -
                //  kbin)/sigma2 );
                //};

                // Loop over all cells
                double kmean = 0.0;
                double nk = 0;
                double kmag2;
                std::array<double, N> kvec;
                for (auto && fourier_index : grid.get_fourier_range()) {
                    grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                    //=====================================================
                    // Set to zero outside the bin (N_k already init to 0)
                    //=====================================================
                    if (kmag2 >= kmag2_max or kmag2 < kmag2_min) {
                        grid.set_fourier_from_index(fourier_index, 0.0);
                        count_grid.set_fourier_from_index(fourier_index, 0.0);
                    } else {
                        // Compute mean k in the bin
                        kmean += std::sqrt(kmag2);
                        pofk_bin[i] += std::norm(grid.get_fourier_from_index(fourier_index));
                        nk += 1.0;
                        count_grid.set_fourier_from_index(fourier_index, 1.0);
                    }

                    //=====================================================
                    // Alternative to the above: add with weights
                    //=====================================================
                    // double kmag = std::sqrt(kmag2);
                    // double fac = weight(kmag, k_bin[i]);
                    // kmean += kmag * fac;
                    // pofk_bin[i] += std::norm(grid.get_fourier_from_index(fourier_index)) * fac;
                    // nk += fac;
                    // grid.set_fourier_from_index(fourier_index, grid.get_fourier_from_index(fourier_index) *
                    // fac); count_grid.set_fourier_from_index(fourier_index, fac);
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &kmean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &pofk_bin[i], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, &nk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
                // The mean k in the bin
                kmean_bin[i] = (nk == 0) ? k_bin[i] : kmean / double(nk);
                // Power spectrum in the bin
                pofk_bin[i] = (nk == 0) ? 0.0 : pofk_bin[i] / double(nk);

#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "kmean: " << kmean_bin[i] / (2.0 * M_PI) << "\n";
#endif

                // Transform to real space
                grid.fftw_c2r();
                count_grid.fftw_c2r();
            }

            // We now have F_k and N_k for all bins
            for (size_t i = 0; i < nbins_tot; i++) {
#ifdef DEBUG_BISPECTRUM
                if (FML::ThisTask == 0)
                    if ((i * 10) / nbins_tot != ((i + 1) * 10) / nbins_tot)
                        std::cout << "Integrating up " << 100 * (i + 1) / nbins_tot << " %\n";
                ;
#endif

                // Current values of ik1,ik2,ik3,...
                std::array<int, ORDER> ik;
                for (int ii = ORDER - 1, n = 1; ii >= 0; ii--, n *= nbins) {
                    ik[ii] = i / n % nbins;
                }

                // Symmetry: only do ik1 <= ik2 <= ...
                bool valid = true;
                double ksum = 0.0;
                for (int ii = 1; ii < ORDER; ii++) {
                    if (ik[ii] > ik[ii - 1])
                        valid = false;
                    ksum += k_bin[ik[ii - 1]];
                }
                if (!valid)
                    continue;

                // No valid 'triangles' if k1+k2+... < kN so just set too zero right away
                if (ksum < k_bin[ik[ORDER - 1]] - ORDER * deltak / 2.) {
                    N123[i] = 0.0;
                    P123[i] = 0.0;
                    continue;
                }

                // Compute number of triangles in current bin (norm insignificant below)
                double N123_current = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : N123_current)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : N_k[0].get_real_range(islice, islice + 1)) {
                        double Nproduct = 1.0;
                        for (int ii = 0; ii < ORDER; ii++)
                            Nproduct *= N_k[ik[ii]].get_real_from_index(real_index);
                        N123_current += Nproduct;
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &N123_current, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

                // Compute sum over triangles
                double F123_current = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : F123_current)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : F_k[0].get_real_range(islice, islice + 1)) {
                        double Fproduct = 1.0;
                        for (int ii = 0; ii < ORDER; ii++)
                            Fproduct *= F_k[ik[ii]].get_real_from_index(real_index);
                        F123_current += Fproduct;
                    }
                }
#ifdef USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &F123_current, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
                // Normalize dx^N / (2pi)^N
                F123_current *= std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);
                N123_current *= std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);

                // Set the result
                P123[i] = (N123_current > 0.0) ? F123_current / N123_current : 0.0;
                N123[i] = (N123_current > 0.0) ? N123_current : 0.0;
            }

            // Set stuff not computed
            for (size_t i = 0; i < nbins_tot; i++) {

                // Current values of ik1,ik2,ik3
                std::vector<int> ik(ORDER);
                for (int ii = 0, n = 1; ii < ORDER; ii++, n *= nbins) {
                    ik[ii] = i / n % nbins;
                }

                // Symmetry: only do ik1 <= ik2 <= ...
                bool valid = true;
                for (int ii = 1; ii < ORDER; ii++) {
                    if (ik[ii] > ik[ii - 1])
                        valid = false;
                }
                if (!valid)
                    continue;

                // Find a cell given by symmetry that we have computed
                // by sorting ik in increasing order
                std::sort(ik.begin(), ik.end(), std::less<int>());

                // Compute cell index
                size_t index = 0;
                for (int ii = 0; ii < ORDER; ii++)
                    index = index * nbins + ik[ii];

                // Set value (if it has not been set before just in case we fucked up)
                if (P123[index] == 0.0)
                    P123[index] = P123[i];
                if (N123[index] == 0.0)
                    N123[index] = N123[i];
            }
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
