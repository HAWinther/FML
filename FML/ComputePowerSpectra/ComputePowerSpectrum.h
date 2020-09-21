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
#include <FML/LPT/Reconstruction.h>        // For particles->redshiftspace
#include <FML/MPIParticles/MPIParticles.h> // Only for compute_multipoles from particles

// The classes that define how to bin and return the data
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
        // of the methods below are given. See PowerSpectrumBinning.h for more info
        //================================================================================

        // The powerspectrum result, see PowerspectrumBinning.h
        template <int N>
        class PowerSpectrumBinning;

        // General polyspectrum result, see PolyspectrumBinning.h
        template <int N, int ORDER>
        class PolyspectrumBinning;

        // Type aliases for mono-,bi- and tri-spectrum. See PolyspectrumBinning.h for more info
        template <int N>
        using MonospectrumBinning = PolyspectrumBinning<N, 2>;

        template <int N>
        using BispectrumBinning = PolyspectrumBinning<N, 3>;

        template <int N>
        using TrispectrumBinning = PolyspectrumBinning<N, 4>;

        //================================================================================
        /// @brief Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
        /// Fourier transform, decolvolve the window function for the assignement above.
        /// With interlacing bin to a grid and an interlaced grid (displaced by dx/2 in all directions)
        /// Fourier transform both and add the together to cancel the leading aliasing contributions
        /// bin up power-spectrum and subtract shot-noise 1/NumPartTotal. Note that with interlacing we change the
        /// particle positions, but when returning they should be in the same state as when we started.
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
        /// @param[in] interlacing Use interlaced grids for alias reduction. Twice as expensive, but allows us to use a
        /// smaller grid (so its actually faster if used correctly).
        ///
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum(int Ngrid,
                                    T * part,
                                    size_t NumPart,
                                    size_t NumPartTotal,
                                    PowerSpectrumBinning<N> & pofk,
                                    std::string density_assignment_method,
                                    bool interlacing);

        //================================================================================
        /// @brief Brute force (but aliasing free) computation of the power spectrum.
        /// Loop over all grid-cells and all particles and add up contribution and subtracts shot-noise term.
        /// Since we need to combine all particles with all cells this is not easiy parallelizable with MPI
        /// so we assume all CPUs have exactly the same particles when this is run on more than 1 MPI tasks (so best run
        /// just using OpenMP).
        /// This method scales as O(Npart)*O(Nmesh^N) so will be slow!
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
        void compute_power_spectrum_direct_summation(int Ngrid,
                                                     const T * part,
                                                     size_t NumPart,
                                                     PowerSpectrumBinning<N> & pofk);

        //==========================================================================================
        /// @brief Compute the power-spectrum of a fourier grid. The result has no scales. Get
        /// scales by calling pofk.scale(boxsize) which does k *= 1/Boxsize and
        /// pofk *= Boxsize^N once spectrum has been computed. The method assumes the two grids are
        /// fourier transforms of real grids (i.e. f(-k) = f^*(k)).
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] pofk Binned power-spectrum
        ///
        //==========================================================================================
        template <int N>
        void bin_up_power_spectrum(const FFTWGrid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk);

        //==========================================================================================
        /// @brief Compute the cross power-spectrum of two fourier grids. The result has no scales. Get
        /// scales by calling pofk.scale(boxsize) which does k *= 1/Boxsize and
        /// pofk *= Boxsize^N once spectrum has been computed. The method assumes the two grids are
        /// fourier transforms of real grids (i.e. f(-k) = f^*(k)) and we only bin up the real part of f1(k)f2^*(k).
        /// The imaginary part is also binned up, but not returned. Instead we check that it's indeed small and give a
        /// warning if not.
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid_1 Grid in fourier space
        /// @param[in] fourier_grid_2 Grid in fourier space
        /// @param[out] pofk Binned cross power-spectrum
        ///
        //==========================================================================================
        template <int N>
        void bin_up_cross_power_spectrum(const FFTWGrid<N> & fourier_grid_1,
                                         const FFTWGrid<N> & fourier_grid_2,
                                         PowerSpectrumBinning<N> & pofk);

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
        void compute_power_spectrum_multipoles_fourier(const FFTWGrid<N> & fourier_grid,
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
        /// @param[in] interlacing Use interlaced grids for alias reduction when computing the density field
        ///
        //================================================================================
        template <int N, class T>
        void compute_power_spectrum_multipoles(int Ngrid,
                                               FML::PARTICLE::MPIParticles<T> & part,
                                               double velocity_to_displacement,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::string density_assignment_method,
                                               bool interlacing);

        //================================================================================
        /// @brief Computes the polyspectrum P(k1,k2,k3,...,kORDER) from particles. Note that with interlacing we change
        /// the particle positions, but when returning they should be in the same state as when we started. This method
        /// allocates nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^ORDER
        /// integrals.
        /// If one is to compute many spectra with the same Ngrid and binning then one can precompute N123 in polyofk
        /// and set it using polyofk.set_bincount(N123). This speeds up the polyspectrum estimation by a factor of 2.
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
        /// @param[in] interlacing Use interlaced grids when computing density field for alias reduction
        ///
        //================================================================================
        template <int N, class T, int ORDER>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER> & polyofk,
                                  std::string density_assignment_method,
                                  bool interlacing);

        //================================================================================
        /// @brief Computes the polyspectrum P(k1,k2,k3,...,kORDER) from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^ORDER integrals.
        /// If one is to compute many spectra with the same Ngrid and binning then one can precompute N123 in polyofk
        /// and set it using polyofk.set_bincount(N123). This speeds up the polyspectrum estimation by a factor of 2.
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam ORDER The order. 2 is the power-spectrum, 3 is the bispectrum, 4 is the trispectrum.
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] polyofk The binned polyspectrum.
        ///
        //================================================================================
        template <int N, int ORDER>
        void compute_polyspectrum(const FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, ORDER> & polyofk);

        //================================================================================
        /// @brief Computes the monospectrum P(k1,k2) from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^2 integrals.
        /// This is just an alias for compute_polyspectrum<N, 2>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] pofk The binned monospectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N>
        void compute_monospectrum(const FFTWGrid<N> & fourier_grid, MonospectrumBinning<N> & pofk);

        //================================================================================
        /// @brief Computes the monospectrum P(k1,k2) from particles. Note that with interlacing we change the
        /// particle positions, but when returning they should be in the same state as when we started. This method
        /// allocates nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^2
        /// integrals.
        /// This is just an alias for compute_polyspectrum<N, 2>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] pofk The binned monospectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        /// @param[in] interlacing Use interlaced grids when computing density field for alias reduction
        ///
        //================================================================================
        template <int N, class T>
        void compute_monospectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  MonospectrumBinning<N> & pofk,
                                  std::string density_assignment_method,
                                  bool interlacing);

        //================================================================================
        /// @brief Computes the bispectrum B(k1,k2,k3) from particles. Note that with interlacing we change the
        /// particle positions, but when returning they should be in the same state as when we started. This method
        /// allocates nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^3
        /// integrals.
        /// This is just an alias for compute_polyspectrum<N, 3>
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
        /// @param[in] interlacing Use interlaced grids when computing density field for alias reduction
        ///
        //================================================================================
        template <int N, class T>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N> & bofk,
                                std::string density_assignment_method,
                                bool interlacing);

        //================================================================================
        /// @brief Computes the bispectrum B(k1,k2,k3) from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^3 integrals.
        /// This is just an alias for compute_polyspectrum<N, 3>
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
        void compute_bispectrum(const FFTWGrid<N> & fourier_grid, BispectrumBinning<N> & bofk);

        //================================================================================
        /// @brief Computes the trispectrum T(k1,k2,k3,k4) from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^4 integrals.
        /// This is just an alias for compute_polyspectrum<N, 4>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] tofk The binned trispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N>
        void compute_trispectrum(const FFTWGrid<N> & fourier_grid, TrispectrumBinning<N> & tofk);

        //================================================================================
        /// @brief Computes the truspectrum T(k1,k2,k3,k4) from particles. Note that with interlacing we change the
        /// particle positions, but when returning they should be in the same state as when we started. This method
        /// allocates nbins FFTWGrids at the same time and performs 2*nbins fourier transforms and does nbins^4
        /// integrals.
        /// This is just an alias for compute_polyspectrum<N, 4>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] Ngrid Size of the grid to use.
        /// @param[in] part Pointer to the first particle.
        /// @param[in] NumPart Number of particles on the local task.
        /// @param[in] NumPartTotal Total number of particles on all tasks.
        /// @param[out] tofk The binned trispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS)
        /// @param[in] interlacing Use interlaced grids when computing density field for alias reduction
        ///
        //================================================================================
        template <int N, class T>
        void compute_trispectrum(int Ngrid,
                                 T * part,
                                 size_t NumPart,
                                 size_t NumPartTotal,
                                 TrispectrumBinning<N> & tofk,
                                 std::string density_assignment_method,
                                 bool interlacing);

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
        void compute_power_spectrum_multipoles_fourier(const FFTWGrid<N> & fourier_grid,
                                                       std::vector<PowerSpectrumBinning<N>> & Pell,
                                                       std::vector<double> line_of_sight_direction) {

            assert_mpi(
                line_of_sight_direction.size() == N,
                "[compute_power_spectrum_multipoles_fourier] Line of sight direction has wrong number of dimensions\n");
            assert_mpi(Pell.size() > 0, "[compute_power_spectrum_multipoles_fourier] Pell must have size > 0\n");
            assert_mpi(fourier_grid.get_nmesh() > 0,
                       "[compute_power_spectrum_multipoles_fourier] grid must have Nmesh > 0\n");

            int Nmesh = fourier_grid.get_nmesh();
            auto Local_nx = fourier_grid.get_local_nx();
            auto Local_x_start = fourier_grid.get_local_x_start();

            // Norm of LOS vector
            double rmag = 0.0;
            for (int idim = 0; idim < N; idim++)
                rmag += line_of_sight_direction[idim] * line_of_sight_direction[idim];
            rmag = std::sqrt(rmag);
            assert_mpi(rmag > 0.0,
                       "[compute_power_spectrum_multipoles_fourier] Line of sight vector has zero length\n");

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
        void bin_up_power_spectrum(const FFTWGrid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk) {

            assert_mpi(fourier_grid.get_nmesh() > 0, "[bin_up_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_power_spectrum] Binning has inconsistent parameters\n");

            const auto Nmesh = fourier_grid.get_nmesh();
            const auto Local_nx = fourier_grid.get_local_nx();
            const auto Local_x_start = fourier_grid.get_local_x_start();

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

        // Bin up the cross power-spectrum of a given fourier grids
        template <int N>
        void bin_up_cross_power_spectrum(FFTWGrid<N> & fourier_grid_1,
                                         FFTWGrid<N> & fourier_grid_2,
                                         PowerSpectrumBinning<N> & pofk) {

            assert_mpi(fourier_grid_1.get_nmesh() > 0, "[bin_up_cross_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(fourier_grid_1.get_nmesh() == fourier_grid_2.get_nmesh(),
                       "[bin_up_cross_power_spectrum] Grids must have the same gridsize\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_cross_power_spectrum] Binning has inconsistent parameters\n");

            const auto Nmesh = fourier_grid_1.get_nmesh();
            const auto Local_nx = fourier_grid_1.get_local_nx();
            const auto Local_x_start = fourier_grid_1.get_local_x_start();

            // Initialize binning just in case
            pofk.reset();

            // Take a copy and bin up the imaginary part also
            PowerSpectrumBinning<N> pofk_imag = pofk;
            pofk_imag.reset();

            // Bin up P(k)
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : fourier_grid_1.get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode( k=0)

                    // Special treatment of k = 0 plane (Safer way: fetch coord)
                    // auto coord = fourier_grid.get_fourier_coord_from_index(fourier_index);
                    // int last_coord = coord[N-1];
                    int last_coord = fourier_index % (Nmesh / 2 + 1);
                    double weight = last_coord > 0 && last_coord < Nmesh / 2 ? 2.0 : 1.0;

                    auto delta_1 = fourier_grid_1.get_fourier_from_index(fourier_index);
                    auto delta_2 = fourier_grid_2.get_fourier_from_index(fourier_index);
                    auto delta12_real = delta_1.real() * delta_2.real() + delta_1.imag() * delta_2.imag();
                    auto delta12_imag = -delta_1.real() * delta_2.imag() + delta_1.imag() * delta_2.real();

                    // Add norm to bin
                    fourier_grid_1.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                    pofk.add_to_bin(kmag, delta12_real, weight);
                    pofk_imag.add_to_bin(kmag, delta12_imag, weight);
                }
            }

            // Normalize to get P(k) (this communicates over tasks)
            pofk.normalize();
            pofk_imag.normalize();

            // NB: we currently don't return the imaginary part. For real fields this should be zero
            // so we just check this and give a warning if its large
            for (int i = 0; i < pofk.n; i++) {
                if (std::abs(pofk.pofk[i]) < 1e3 * std::abs(pofk_imag.pofk[i])) {
                    std::cout
                        << "Warning: the imaginary part of the cross spectrum is > 0.1\% times the real part [ k: "
                        << pofk.kbin[i] << " Real(d1d2): " << pofk.pofk[i] << " Imag(d1d2): " << pofk_imag.pofk[i]
                        << "]\n";
                }
            }
        }

        // Brute force. Add particles to the grid using direct summation
        // This gives alias free P(k), but scales as O(Npart)*O(Nmesh^N)
        template <int N, class T>
        void
        compute_power_spectrum_direct_summation(int Ngrid, T * part, size_t NumPart, PowerSpectrumBinning<N> & pofk) {

            static_assert(
                FML::PARTICLE::has_get_pos<T>(),
                "[compute_power_spectrum_direct_summation] Particle class needs to have positions to use this method");

            // Very simple check to see if all tasks do have the same particles
            if (FML::NTasks > 1) {
                long long int tmp1 = NumPart;
                FML::MinOverTasks(&tmp1);
                long long int tmp2 = NumPart;
                FML::MaxOverTasks(&tmp2);
                double x = FML::PARTICLE::GetPos(part[0])[0];
                double y = x;
                FML::MaxOverTasks(&x);
                FML::MinOverTasks(&y);
                assert_mpi(tmp1 == tmp2 and std::abs(x - y) < 1e-10,
                           "[direct_summation_power_spectrum] All tasks must have the same particles for this method "
                           "to work\n");
            }

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
                                               std::string density_assignment_method,
                                               bool interlacing) {

            // Sanity check
            static_assert(FML::PARTICLE::has_get_pos<T>(),
                          "[compute_power_spectrum_multipoles] Particle class needs to have positions to use "
                          "this method");
            static_assert(
                FML::PARTICLE::has_get_vel<T>(),
                "[compute_power_spectrum_multipoles] Particle class needs to have velocity to use this method");

            // Set how many extra slices we need for the density assignment to go smoothly
            // One extra slice if we use interlacing due to displacing particles by one half cell to the right
            const auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + (interlacing ? 1 : 0);

            // Initialize binning just in case
            for (size_t ell = 0; ell < Pell.size(); ell++)
                Pell[ell].reset();

            // Set a binning for each axes
            std::vector<std::vector<PowerSpectrumBinning<N>>> Pell_all(N);
            for (int idim = 0; idim < N; idim++) {
                Pell_all[idim] = Pell;
            }

            // Allocate density grid
            FFTWGrid<N> density_k(Ngrid, nleft, nright);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum_multipoles::density_k");

            // Loop over all the N axes we are going to put the particles
            // into redshift space
            for (int idim = 0; idim < N; idim++) {

                // Set up binning for current axis
                std::vector<PowerSpectrumBinning<N>> Pell_current = Pell_all[idim];
                for (size_t ell = 0; ell < Pell_current.size(); ell++)
                    Pell_current[ell].reset();

                // Make line of sight direction unit vector
                std::vector<double> line_of_sight_direction(N, 0.0);
                line_of_sight_direction[idim] = 1.0;

                // Transform to redshift-space
                FML::COSMOLOGY::particles_to_redshiftspace(part, line_of_sight_direction, velocity_to_displacement);

                // Bin particles to grid
                density_k.set_grid_status_real(true);
                if (interlacing) {

                    // Bin to grid and interlaced grid and deconvolve window function
                    FML::INTERPOLATION::particles_to_fourier_grid_interlacing(part.get_particles_ptr(),
                                                                              part.get_npart(),
                                                                              part.get_npart_total(),
                                                                              density_k,
                                                                              density_assignment_method);
                    deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

                } else {

                    // Bin to grid, fourier transform and deconvolve window function
                    particles_to_grid<N, T>(part.get_particles_ptr(),
                                            part.get_npart(),
                                            part.get_npart_total(),
                                            density_k,
                                            density_assignment_method);
                    density_k.fftw_r2c();
                    deconvolve_window_function_fourier<N>(density_k, density_assignment_method);
                }

                // Compute power-spectrum multipoles
                compute_power_spectrum_multipoles_fourier(density_k, Pell_current, line_of_sight_direction);

                // Copy over binning we computed
                Pell_all[idim] = Pell_current;

                // Transform particles back to real-space (we don't want to ruin the particles)
                // Ideally we should have taken a copy, but this is fine
                FML::COSMOLOGY::particles_to_redshiftspace(part, line_of_sight_direction, -velocity_to_displacement);
            }

            // Normalize
            for (size_t ell = 0; ell < Pell.size(); ell++) {
                for (int idim = 0; idim < N; idim++) {
                    Pell[ell] += Pell_all[idim][ell];
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
                                    std::string density_assignment_method,
                                    bool interlacing) {

            // Set how many extra slices we need for the density assignment to go smoothly
            const auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + (interlacing ? 1 : 0);

            // Bin particles to grid
            FFTWGrid<N> density_k(Ngrid, nleft, nright);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum::density_k");

            if (interlacing) {

                // Bin to grid using interlaced grids to produce a fourier space density field
                FML::INTERPOLATION::particles_to_fourier_grid_interlacing(
                    part, NumPart, NumPartTotal, density_k, density_assignment_method);
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

            } else {

                // Bin to grid, fourier transform and deconvolves the window function
                particles_to_grid<N, T>(part, NumPart, NumPartTotal, density_k, density_assignment_method);
                density_k.fftw_r2c();
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);
            }

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

        template <int N, class T>
        void compute_monospectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  MonospectrumBinning<N> & pofk,
                                  std::string density_assignment_method,
                                  bool interlacing) {
            compute_polyspectrum<N, T, 2>(
                Ngrid, part, NumPart, NumPartTotal, pofk, density_assignment_method, interlacing);
        }

        template <int N, class T>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N> & bofk,
                                std::string density_assignment_method,
                                bool interlacing) {
            compute_polyspectrum<N, T, 3>(
                Ngrid, part, NumPart, NumPartTotal, bofk, density_assignment_method, interlacing);
        }

        template <int N, class T>
        void compute_trispectrum(int Ngrid,
                                 T * part,
                                 size_t NumPart,
                                 size_t NumPartTotal,
                                 TrispectrumBinning<N> & tofk,
                                 std::string density_assignment_method,
                                 bool interlacing) {
            compute_polyspectrum<N, T, 3>(
                Ngrid, part, NumPart, NumPartTotal, tofk, density_assignment_method, interlacing);
        }

        // Computes the polyspectrum P(k1,k2,k3,...)
        template <int N, class T, int ORDER>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER> & polyofk,
                                  std::string density_assignment_method,
                                  bool interlacing) {

            // Set how many extra slices we need for the density assignment to go smoothly
            const auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + 1;

            FFTWGrid<N> density_k(Ngrid, nleft, nright);
            density_k.add_memory_label("FFTWGrid::compute_polyspectrum::density_k");

            if (interlacing) {

                // Bin particles to grid (use interlaced to reduce alias) and deconvolve window
                FML::INTERPOLATION::particles_to_fourier_grid_interlacing(
                    part, NumPart, NumPartTotal, density_k, density_assignment_method);
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

            } else {

                // Bin to grid, fourier transform and deconvolve window function
                particles_to_grid<N, T>(part, NumPart, NumPartTotal, density_k, density_assignment_method);
                density_k.fftw_r2c();
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);
            }

            // Compute polyspectrum
            compute_polyspectrum<N, ORDER>(density_k, polyofk);
        }

        //================================================================================
        /// @brief This method is used by compute_polyspectrum. It computes the number of
        /// generalized triangles of the bins needed to normalize the polyspectra up to symmetry (i.e. we only compute
        /// it for k1<=k2<=k3 and only for valid triangle configurations (k1+k2 >= k3) and then set rest using
        /// symmetry). If one is to compute many spectra with the same Nmesh and binning then one can precompute N123
        /// and set it using polyofk.set_bincount(N123) (which sets polyofk.bincount_is_set = true and avoid a call to this function) 
        /// This speeds up the polyspectrum estimation by a factor of 2.
        ///
        /// @tparam N The dimension we work in
        /// @tparam ORDER The order (mono = 2, bi = 3, tri = 4)
        ///
        /// @param[in] Nmesh The size of the grid we us
        /// @param[out] polyofk The binning (we compute and store the volumes of each bin, N123, in this binning).
        ///
        //================================================================================
        template <int N, int ORDER>
        void compute_polyspectrum_bincount(int Nmesh, PolyspectrumBinning<N, ORDER> & polyofk) {

            const auto nbins = polyofk.n;
            const auto klow = polyofk.klow;
            const auto khigh = polyofk.khigh;
            const auto kbin = polyofk.kbin;
            auto & N123 = polyofk.N123;
            const size_t nbins_tot = N123.size();

            // Allocate grids
            std::vector<FFTWGrid<N>> N_k(nbins);
            for (int i = 0; i < nbins; i++) {
                N_k[i] = FFTWGrid<N>(Nmesh);
                N_k[i].add_memory_label("FFTWGrid::compute_polyspectrum_bincount::N_" + std::to_string(i));
                N_k[i].set_grid_status_real(false);
                N_k[i].fill_fourier_grid(0.0);
            }

            // Set the grids
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int i = 0; i < nbins; i++) {
                FFTWGrid<N> & grid = N_k[i];
                const double kmag2_max = khigh[i] * khigh[i];
                const double kmag2_min = klow[i] * klow[i];
                double kmag2;
                std::array<double, N> kvec;
                for (auto && fourier_index : grid.get_fourier_range()) {
                    grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    if (not(kmag2 > kmag2_max or kmag2 < kmag2_min)) {
                        grid.set_fourier_from_index(fourier_index, 1.0);
                    }
                }
            }

            // Transform to real space
            for (int i = 0; i < nbins; i++) {
                N_k[i].fftw_c2r();
            }

            // We now have N_k for all bins, integrate up
            for (size_t i = 0; i < nbins_tot; i++) {

                // Current values of ik1,ik2,ik3,...
                std::array<int, ORDER> ik = polyofk.get_coord_from_index(i);

                // Symmetry: only do ik1 <= ik2 <= ...
                if (not polyofk.compute_this_configuration(ik)) {
                    N123[i] = 0.0;
                    continue;
                }

                // Compute number of triangles in current bin
                // Norm represents integration measure dx^N / (2pi)^N
                double N123_current = 0.0;
                const double norm = std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);
                const auto Local_nx = N_k[0].get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : N123_current)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : N_k[0].get_real_range(islice, islice + 1)) {
                        if constexpr (ORDER == 2) {
                            double N1 = N_k[ik[0]].get_real_from_index(real_index);
                            double N2 = N_k[ik[1]].get_real_from_index(real_index);
                            N123_current += N1 * N2;
                        } else if constexpr (ORDER == 3) {
                            double N1 = N_k[ik[0]].get_real_from_index(real_index);
                            double N2 = N_k[ik[1]].get_real_from_index(real_index);
                            double N3 = N_k[ik[2]].get_real_from_index(real_index);
                            N123_current += N1 * N2 * N3;
                        } else if constexpr (ORDER == 4) {
                            double N1 = N_k[ik[0]].get_real_from_index(real_index);
                            double N2 = N_k[ik[1]].get_real_from_index(real_index);
                            double N3 = N_k[ik[2]].get_real_from_index(real_index);
                            double N4 = N_k[ik[3]].get_real_from_index(real_index);
                            N123_current += N1 * N2 * N3 * N4;
                        } else {
                            double Nproduct = 1.0;
                            for (int ii = 0; ii < ORDER; ii++)
                                Nproduct *= N_k[ik[ii]].get_real_from_index(real_index);
                            N123_current += Nproduct;
                        }
                    }
                }
                FML::SumOverTasks(&N123_current);
                N123[i] = N123_current * norm;

                // We cannot have less than 1 generalized triangle so put to zero if small
                // due to numerical noise
                if (N123[i] < 1.0)
                    N123[i] = 0.0;
            }

            // Set stuff not computed
            for (size_t i = 0; i < nbins_tot; i++) {

                // Current values of ik1,ik2,ik3,...
                std::array<int, ORDER> ik = polyofk.get_coord_from_index(i);

                // If its already computed we don't need to set it
                if (polyofk.compute_this_configuration(ik))
                    continue;

                // Find a cell given by symmetry that we have computed
                // by sorting ik in increasing order
                std::sort(ik.begin(), ik.end(), std::less<int>());

                // Compute cell index of cell we have computed
                size_t index = polyofk.get_index_from_coord(ik);

                // Set value
                N123[i] = N123[index];
            }
        }

        template <int N>
        void compute_monospectrum(const FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, 2> & pofk) {
            compute_polyspectrum<N, 2>(fourier_grid, pofk);
        }

        template <int N>
        void compute_bispectrum(const FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, 3> & bofk) {
            compute_polyspectrum<N, 3>(fourier_grid, bofk);
        }

        template <int N>
        void compute_trispectrum(const FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, 4> & tofk) {
            compute_polyspectrum<N, 4>(fourier_grid, tofk);
        }

        template <int N, int ORDER>
        void compute_polyspectrum(const FFTWGrid<N> & fourier_grid, PolyspectrumBinning<N, ORDER> & polyofk) {

            const auto Nmesh = fourier_grid.get_nmesh();
            const auto Local_nx = fourier_grid.get_local_nx();
            const auto nbins = polyofk.n;

            assert_mpi(nbins > 0, "[compute_polyspectrum] nbins has to be >=0\n");
            assert_mpi(Nmesh > 0, "[compute_polyspectrum] grid is not allocated\n");
            assert_mpi(polyofk.P123.size() == polyofk.N123.size() and
                           polyofk.N123.size() == size_t(FML::power(nbins, ORDER)),
                       "[compute_polyspectrum] Binning is not good\n");
            static_assert(ORDER > 1);

            // Get where to store the resuts in
            // We don't add to thes below so we don't need to clear the arrays
            std::vector<double> & P123 = polyofk.P123;
            std::vector<double> & N123 = polyofk.N123;
            std::vector<double> & pofk_bin = polyofk.pofk;
            std::vector<double> & kmean = polyofk.kmean;
            const size_t nbins_tot = P123.size();

            // Get ranges for which we will compute F_k on
            const std::vector<double> & kbin = polyofk.kbin;
            const std::vector<double> & klow = polyofk.klow;
            const std::vector<double> & khigh = polyofk.khigh;

            // Compute the bincount N123 if it does not already exist
            if (not polyofk.bincount_is_set)
                compute_polyspectrum_bincount<N, ORDER>(Nmesh, polyofk);

            // Allocate grids
            std::vector<FFTWGrid<N>> F_k(nbins);
            for (int i = 0; i < nbins; i++) {
                F_k[i] = fourier_grid;
                F_k[i].add_memory_label("FFTWGrid::compute_polyspectrum::F_" + std::to_string(i));
            }

            for (int i = 0; i < nbins; i++) {
#ifdef DEBUG_POLYSPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "Computing polyspectrum<" << ORDER << "> " << i + 1 << " / " << nbins
                              << " kbin: " << klow[i] / (2.0 * M_PI) << " -> " << khigh[i] / (2.0 * M_PI) << "\n";
#endif

                FFTWGrid<N> & grid = F_k[i];

                // For each bin get klow, khigh
                const double kmag2_max = khigh[i] * khigh[i];
                const double kmag2_min = klow[i] * klow[i];

                // Loop over all cells
                double kmean_bin = 0.0;
                double nk = 0;
                double kmag2;
                pofk_bin[i] = 0.0;
                std::array<double, N> kvec;
                for (auto && fourier_index : grid.get_fourier_range()) {
                    grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                    // Set to zero outside the bin
                    if (kmag2 >= kmag2_max or kmag2 < kmag2_min) {
                        grid.set_fourier_from_index(fourier_index, 0.0);
                    } else {
                        // Compute mean k in the bin
                        kmean_bin += std::sqrt(kmag2);
                        pofk_bin[i] += std::norm(grid.get_fourier_from_index(fourier_index));
                        nk += 1.0;
                    }
                }
                FML::SumOverTasks(&kmean_bin);
                FML::SumOverTasks(&pofk_bin[i]);
                FML::SumOverTasks(&nk);

                // The mean k in the bin
                kmean[i] = (nk == 0) ? kbin[i] : kmean_bin / double(nk);

                // Power spectrum in the bin
                pofk_bin[i] = (nk == 0) ? 0.0 : pofk_bin[i] / double(nk);

#ifdef DEBUG_POLYSPECTRUM
                if (FML::ThisTask == 0)
                    std::cout << "kmean: " << kmean[i] / (2.0 * M_PI) << "\n";
#endif

                // Transform to real space
                grid.fftw_c2r();
            }

            // We now have F_k and N_k for all bins
            for (size_t i = 0; i < nbins_tot; i++) {
#ifdef DEBUG_POLYSPECTRUM
                if (FML::ThisTask == 0)
                    if ((i * 10) / nbins_tot != ((i + 1) * 10) / nbins_tot)
                        std::cout << "Integrating up " << 100 * (i + 1) / nbins_tot << " %\n";
                ;
#endif

                // Current values of ik1,ik2,ik3,...
                const auto ik = polyofk.get_coord_from_index(i);

                // Symmetry: only do ik1 <= ik2 <= ... and don't need to do configurations that don't satisfy the
                // triangle inequality
                if (not polyofk.compute_this_configuration(ik)) {
                    P123[i] = 0.0;
                    continue;
                }

                // Compute the sum over triangles by evaluating the integral Int dx^N/(2pi)^N
                // delta_k1(x)delta_k2(x)...delta_kORDER(x)
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
                FML::SumOverTasks(&F123_current);

                // Normalize by the integration measure dx^N / (2pi)^N
                F123_current *= std::pow(1.0 / double(Nmesh) / (2.0 * M_PI), N);

                // Set the result
                const double N123_current = N123[i];
                P123[i] = (N123_current > 0.0) ? F123_current / N123_current : 0.0;
            }

            // Set stuff not computed above which follows from symmetry
            for (size_t i = 0; i < nbins_tot; i++) {

                // Current values of ik1,ik2,ik3
                auto ik = polyofk.get_coord_from_index(i);

                // If its valid its already computed
                if (polyofk.compute_this_configuration(ik))
                    continue;

                // Find a cell given by symmetry that we have computed
                // by sorting ik in increasing order
                std::sort(ik.begin(), ik.end(), std::less<int>());

                // Compute cell index of configuration we have computed
                size_t index = polyofk.get_index_from_coord(ik);

                // Set value
                P123[i] = P123[index];
            }
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
