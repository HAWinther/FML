#ifndef COMPUTEPOWERSPECTRUM_HEADER
#define COMPUTEPOWERSPECTRUM_HEADER

#include <array>
#include <cassert>
#include <string>
#include <iostream>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/FFTWGrid/CopyFourierModes.h>
#include <FML/FFTWGrid/FourierResampling.h>
#include <numeric>
#include <climits>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/Reconstruction.h>        // For particles->redshiftspace
#include <FML/MPIParticles/MPIParticles.h> // Only for compute_multipoles from particles

// The classes that define how to bin and return the data
#include <FML/ComputePowerSpectra/PolyspectrumBinning.h>
#include <FML/ComputePowerSpectra/PolyspectrumWorkspace.h>
#include <FML/ComputePowerSpectra/ShellOverlap.h>
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
        template <int N, int ORDER, bool COMPACT>
        class PolyspectrumBinning;

        // Type aliases for mono-,bi- and tri-spectrum. See PolyspectrumBinning.h for more info
        template <int N, bool COMPACT = false>
        using MonospectrumBinning = PolyspectrumBinning<N, 2, COMPACT>;

        template <int N, bool COMPACT = false>
        using BispectrumBinning = PolyspectrumBinning<N, 3, COMPACT>;

        template <int N, bool COMPACT = false>
        using TrispectrumBinning = PolyspectrumBinning<N, 4, COMPACT>;

        template <int N>
        using SparseMonospectrumBinning = SparsePolyspectrumBinning<N, 2>;
        template <int N>
        using SparseBispectrumBinning = SparsePolyspectrumBinning<N, 3>;
        template <int N>
        using SparseTrispectrumBinning = SparsePolyspectrumBinning<N, 4>;

        //================================================================================
        /// @brief Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
        /// Fourier transform, decolvolve the window function for the assignment above.
        /// With interlacing bin to a grid and an interlaced grid (displaced by dx/2 in all directions)
        /// Fourier transform both and add the together to cancel the leading aliasing contributions
        /// bin up power-spectrum and subtract shot-noise 1/NumPartTotal. Interlacing uses virtual assignment
        /// offsets and never modifies particle positions.
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
        template <int N, class T, template <int> class Grid = FFTWGrid>
        void compute_power_spectrum(int Ngrid,
                                    T * part,
                                    size_t NumPart,
                                    size_t NumPartTotal,
                                    PowerSpectrumBinning<N> & pofk,
                                    std::string density_assignment_method,
                                    bool interlacing,
                                    const FML::Domain & domain = FML::domain);

        //================================================================================
        /// @brief Brute force (but aliasing free) computation of the power spectrum.
        /// Loop over all grid-cells and all particles and add up contribution and subtracts shot-noise term.
        /// Since we need to combine all particles with all cells this is not easily parallelizable with MPI
        /// so we assume all CPUs have exactly the same particles when this is run on more than 1 MPI tasks (so best run
        /// just using OpenMP).
        /// This method scales as \f$ \mathcal{O}({\rm Npart}\cdot {\rm Nmesh}^N) \f$ so will be slow!
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
        /// scales by calling pofk.scale(boxsize) which does \f$ k \to k/B \f$ and
        /// \f$ P(k) \to B^N P(k) \f$ where \f$ B\f$ is the boxsize once spectrum has been computed. The method assumes
        /// the two grids are fourier transforms of real grids (i.e. \f$ f(-k) = f^*(k) \f$).
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] pofk Binned power-spectrum
        ///
        //==========================================================================================
        template <int N, template <int> class Grid = FFTWGrid>
        void bin_up_power_spectrum(const Grid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk);

        //==========================================================================================
        /// @brief Compute the power-spectrum of a fourier grid. The result has no scales. Get
        /// scales by calling pofk.scale(boxsize) which does \f$ k \to k/B \f$ and
        /// \f$ P(k) \to B^N P(k) \f$ where \f$ B\f$ is the boxsize once spectrum has been computed. The method assumes
        /// the two grids are fourier transforms of real grids (i.e. \f$ f(-k) = f^*(k) \f$).
        /// This method divides the power-spectrum by the square of the window function corresponding to the density
        /// assignment method for every mode we bin up.
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] pofk Binned power-spectrum
        /// @param[in] density_assignment_method The density assignment method we use to create the grid
        ///
        //==========================================================================================
        template <int N, template <int> class Grid = FFTWGrid>
        void bin_up_deconvolved_power_spectrum(const Grid<N> & fourier_grid,
                                               PowerSpectrumBinning<N> & pofk,
                                               std::string density_assignment_method);

        //==========================================================================================
        /// @brief Compute the cross power-spectrum of two fourier grids. The result has no scales. Get
        /// scales by calling pofk.scale(boxsize) which does k\f$ k \to k/B \f$ and
        /// \f$ P(k) \to B^N P(k) \f$ where \f$ B\f$ is the boxsize once spectrum has been computed. The method assumes
        /// the two grids are fourier transforms of real grids (i.e. \f$ f(-k) = f^*(k) \f$) and we only bin up the real
        /// part of \f$ f_1(k)f_2^*(k) \f$. The imaginary part is also binned up, but not returned.
        ///
        /// @tparam N Dimension of the grid
        ///
        /// @param[in] fourier_grid_1 Grid in fourier space
        /// @param[in] fourier_grid_2 Grid in fourier space
        /// @param[out] pofk Binned cross power-spectrum
        ///
        //==========================================================================================
        template <int N, template <int> class Grid = FFTWGrid>
        void bin_up_cross_power_spectrum(const Grid<N> & fourier_grid_1,
                                         const Grid<N> & fourier_grid_2,
                                         PowerSpectrumBinning<N> & pofk);

        //================================================================================
        /// @brief Compute power-spectrum multipoles from a Fourier grid
        /// where we have a fixed line_of_sight_direction (typical coordinate axes like \f$ (0,0,1) \f$).
        /// Pell contains \f$ P_0,P_1,P_2,\ldots,P_{\ell_{\rm max}} \f$ where \f$ \ell_{\rm max} = n-1 \f$ is the size
        /// of Pell
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid The fourier grid to compute the multipoles from.
        /// @param[out] Pell Vector of power-spectrum binnings. The size of Pell is the maximum ell to compute.
        /// All binnings has to have nbins, kmin and kmax set. At the end Pell[ ell ] is a binning of P_ell(k).
        /// @param[in] line_of_sight_direction The line of sight direction, e.g. \f$ (0,0,1) \f$ for the z-axis.
        ///
        //================================================================================
        template <int N, template <int> class Grid = FFTWGrid>
        void compute_power_spectrum_multipoles_fourier(const Grid<N> & fourier_grid,
                                                       std::vector<PowerSpectrumBinning<N>> & Pell,
                                                       std::vector<double> line_of_sight_direction);

        //================================================================================
        /// @brief Cross power-spectrum multipoles of two Fourier grids (same size and layout) with a fixed line of
        /// sight: \f$ P_\ell(k) = (2\ell+1) \left< {\rm Re}[\delta_1(\vec k) \delta_2^*(\vec k)] L_\ell(\mu) \right> \f$
        /// with \f$ \mu = \hat k \cdot \hat n \f$, binned as compute_power_spectrum_multipoles_fourier (which is this with
        /// \f$ \delta_1 = \delta_2 \f$; the two give the same result for the same grid). Only the even multipoles are
        /// computed (the odd ones are zero; for two real fields that are statistically symmetric under
        /// \f$ \vec k \to -\vec k \f$ the odd multipoles and the imaginary part vanish on average). The result has no
        /// scales (pofk.scale(boxsize) as for the other estimators). No window deconvolution or shot noise is applied
        /// here: do it on the grids before (deconvolve_window_function_fourier).
        ///
        /// @param[in] fourier_grid_1, fourier_grid_2 The grids in Fourier space
        /// @param[out] Pell Pell[ell] is P_ell(k) for ell < Pell.size() (initialized with nbins, kmin, kmax)
        /// @param[in] line_of_sight_direction The line of sight, e.g. (0,0,1)
        ///
        //================================================================================
        template <int N, template <int> class Grid = FFTWGrid>
        void compute_cross_power_spectrum_multipoles_fourier(const Grid<N> & fourier_grid_1,
                                                             const Grid<N> & fourier_grid_2,
                                                             std::vector<PowerSpectrumBinning<N>> & Pell,
                                                             std::vector<double> line_of_sight_direction);

        //================================================================================
        /// @brief A simple power-spectrum estimator for multipoles in simulations.
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
        /// This is \f$ \frac{c}{aH(a)B} \f$ for peculiar and \f$ \frac{c}{H(a)B} \f$ (where \f$ B\f$ is the boxsize)
        /// for comoving velocities At \f$ z = 0 \f$ velocity_to_displacement = \f$ \frac{1}{100 {\rm Boxsize}} \f$ when
        /// Boxsize is in Mpc/h
        /// @param[out] Pell Vector of power-spectrum binnings. The size of Pell is the maximum ell to compute.
        /// All binnings has to have nbins, kmin and kmax set. At the end Pell[ ell ] is a binning of P_ell(k).
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS) to use.
        /// @param[in] interlacing Use interlaced grids for alias reduction when computing the density field
        ///
        //================================================================================
        template <int N, class T, template <int> class Grid = FFTWGrid>
        void compute_power_spectrum_multipoles(int Ngrid,
                                               FML::PARTICLE::MPIParticles<T> & part,
                                               double velocity_to_displacement,
                                               std::vector<PowerSpectrumBinning<N>> & Pell,
                                               std::string density_assignment_method,
                                               bool interlacing);

        //================================================================================
        /// @brief Computes the polyspectrum \f$ P(k_1,k_2,\ldots,k_{\rm ORDER}) = \left<\delta(k_1)\cdots\delta(k_{\rm
        /// ORDER})\right> \f$ from particles. Note that with interlacing we change the particle positions, but when
        /// returning they should be in the same state as when we started. This method allocates nbins FFTWGrids at the
        /// same time and performs \f$ 2{\rm nbins} \f$ fourier transforms and does \f$ {\rm nbins}^{\rm ORDER} \f$
        /// integrals. If one is to compute many spectra with the same Ngrid and binning then one can precompute N123 in
        /// polyofk and set it using polyofk.set_bincount(N123). This speeds up the polyspectrum estimation by a factor
        /// of 2. Shot-noise term is subtracted if ORDER=2 (power) or ORDER=3 (bi). Doing this for higher order requires
        /// all lower order correlators and we currently only bin up the polyspectrum and the power-spectrum.
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
        template <int N, class T, int ORDER, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER, COMPACT> & polyofk,
                                  std::string density_assignment_method,
                                  bool interlacing,
                                    const FML::Domain & domain = FML::domain);

        //================================================================================
        /// @brief Computes the polyspectrum \f$ P(k_1,k_2,\ldots,k_{\rm ORDER}) = \left<\delta(k_1)\cdots\delta(k_{\rm
        /// ORDER})\right> \f$ from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs \f$ 2{\rm nbins} \f$ fourier transforms and does \f$ {\rm
        /// nbins}^{\rm ORDER} \f$ integrals. If one is to compute many spectra with the same Ngrid and binning then one
        /// can precompute N123 in polyofk and set it using polyofk.set_bincount(N123). This speeds up the polyspectrum
        /// estimation by a factor of 2 by avoiding half of the fourier transforms.
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam ORDER The order. 2 is the power-spectrum, 3 is the bispectrum, 4 is the trispectrum.
        ///
        /// @param[in] fourier_grid Grid in fourier space
        /// @param[out] polyofk The binned polyspectrum.
        ///
        //================================================================================
        template <int N, int ORDER, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_polyspectrum(const Grid<N> & fourier_grid, PolyspectrumBinning<N, ORDER, COMPACT> & polyofk);

        //================================================================================
        /// @brief Computes the monospectrum \f$ P(k_1,k_2) = \left<\delta(k_1)\delta(k_2)\right> \f$ from a fourier
        /// grid. This method allocates nbins FFTWGrids at the same time and performs \f$ 2{\rm nbins} \f$ fourier
        /// transforms and does \f$ {\rm nbins}^{2} \f$ integrals. This is just an alias for compute_polyspectrum<N, 2>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] pofk The binned monospectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_monospectrum(const Grid<N> & fourier_grid, MonospectrumBinning<N, COMPACT> & pofk);

        //================================================================================
        /// @brief Computes the monospectrum \f$ P(k_1,k_2) = \left<\delta(k_1)\delta(k_2)\right> \f$ from particles.
        /// Note that with interlacing we change the particle positions, but when returning they should be in the same
        /// state as when we started. This method allocates nbins FFTWGrids at the same time and performs \f$ 2{\rm
        /// nbins} \f$ fourier transforms and does \f$ {\rm nbins}^{2} \f$ integrals. This is just an alias for
        /// compute_polyspectrum<N, 2>. Shot-noise is subtracted
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
        template <int N, class T, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_monospectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  MonospectrumBinning<N, COMPACT> & pofk,
                                  std::string density_assignment_method,
                                  bool interlacing);

        //================================================================================
        /// @brief Computes the bispectrum \f$ B(k_1,k_2,k_3) = \left<\delta(k_1)\delta(k_2)\delta(k_3)\right> \f$ from
        /// particles. Note that with interlacing we change the particle positions, but when returning they should be in
        /// the same state as when we started. This method allocates nbins FFTWGrids at the same time and performs \f$
        /// 2{\rm nbins} \f$ fourier transforms and does \f$ {\rm nbins}^{3} \f$ integrals. This is just an alias for
        /// compute_polyspectrum<N, 3>. Shot-noise is subtracted.
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
        template <int N, class T, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N, COMPACT> & bofk,
                                std::string density_assignment_method,
                                bool interlacing);

        //================================================================================
        /// @brief Computes the bispectrum \f$ B(k_1,k_2,k_3) = \left<\delta(k_1)\delta(k_2)\delta(k_3)\right> \f$ from
        /// a fourier grid. This method allocates nbins FFTWGrids at the same time and performs \f$ 2{\rm nbins} \f$
        /// fourier transforms and does \f$ {\rm nbins}^{3} \f$ integrals. This is just an alias for
        /// compute_polyspectrum<N, 3>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] bofk The binned bispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_bispectrum(const Grid<N> & fourier_grid, BispectrumBinning<N, COMPACT> & bofk);

        //================================================================================
        /// @brief Computes the trispectrum \f$ T(k_1,k_2,k_3,k_4) =
        /// \left<\delta(k_1)\delta(k_2)\delta(k_3)\delta(k_4)\right> \f$ from a fourier grid. This method allocates
        /// nbins FFTWGrids at the same time and performs \f$ 2{\rm nbins} \f$ fourier transforms and does \f$ {\rm
        /// nbins}^{4} \f$ integrals. This is just an alias for compute_polyspectrum<N, 4>
        ///
        /// @tparam N The dimension of the particles.
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] fourier_grid A fourier grid.
        /// @param[out] tofk The binned trispectrum. We required it to be initialized with the number of bins, kmin and
        /// kmax.
        ///
        //================================================================================
        template <int N, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_trispectrum(const Grid<N> & fourier_grid, TrispectrumBinning<N, COMPACT> & tofk);

        //================================================================================
        /// @brief Computes the truspectrum \f$ T(k_1,k_2,k_3,k_4) =
        /// \left<\delta(k_1)\delta(k_2)\delta(k_3)\delta(k_4)\right> \f$ from particles. Note that with interlacing we
        /// change the particle positions, but when returning they should be in the same state as when we started. This
        /// method allocates nbins FFTWGrids at the same time and performs \f$ 2{\rm nbins} \f$ fourier transforms and
        /// does \f$ {\rm nbins}^{4} \f$ integrals. This is just an alias for compute_polyspectrum<N, 4> The shot-noise
        /// term (something like (B(123) + cyc)/n + (P1P2 + cyc)/n^2 + 1/n^3) is not subtracted as it requires the
        /// bispectrum which we currently don't bin up together with the trispectrum
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
        template <int N, class T, bool COMPACT, template <int> class Grid = FFTWGrid>
        void compute_trispectrum(int Ngrid,
                                 T * part,
                                 size_t NumPart,
                                 size_t NumPartTotal,
                                 TrispectrumBinning<N, COMPACT> & tofk,
                                 std::string density_assignment_method,
                                 bool interlacing);

        //=====================================================================
        //=====================================================================

        //==========================================================================================
        // Compute the power-spectrum multipoles of a fourier grid assuming a fixed line of sight
        // direction (typically coordinate axes). Provide Pell with [ell+1] initialized binnings to compute
        // the first \f$ 0,1,\ldots,\ell \f$ multipoles The result has no scales. Get scales by scaling
        // PowerSpectrumBinning using scale(kscale, pofkscale) with kscale = 1/Boxsize
        // and pofkscale = Boxsize^N once spectrum has been computed
        //==========================================================================================
        namespace detail {
            /// The multipoles of power(fourier_index) (|delta|^2 or Re[delta_1 delta_2^*]) on the modes of grid
            template <int N, class Grid, class Power>
            void power_spectrum_multipoles_fourier_impl(const Grid & fourier_grid,
                                                        Power && power_of_mode,
                                                        std::vector<PowerSpectrumBinning<N>> & Pell,
                                                        std::vector<double> line_of_sight_direction,
                                                        const char * name) {
                const std::string where = std::string("[") + name + "] ";
                assert_mpi(line_of_sight_direction.size() == N,
                           (where + "Line of sight direction has wrong number of dimensions\n").c_str());
                assert_mpi(Pell.size() > 0, (where + "Pell must have size > 0\n").c_str());
                assert_mpi(fourier_grid.get_nmesh() > 0, (where + "grid must have Nmesh > 0\n").c_str());

                // Norm of LOS vector
                double rmag = 0.0;
                for (int idim = 0; idim < N; idim++)
                    rmag += line_of_sight_direction[idim] * line_of_sight_direction[idim];
                rmag = std::sqrt(rmag);
                assert_mpi(rmag > 0.0, (where + "Line of sight vector has zero length\n").c_str());

                // Initialize binning just in case
                for (size_t ell = 0; ell < Pell.size(); ell++)
                    Pell[ell].reset();

                // Bin up mu^k |delta|^2
                FML::GRID::for_each_nonzero_fourier_mode(
                    fourier_grid,
                    [&](IndexIntType fourier_index, const std::array<double, N> & kvec, double kmag, double weight) {
                        const double power = power_of_mode(fourier_index);

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
                    });

                // Normalize (sums over the threads and the MPI tasks)
                for (size_t ell = 0; ell < Pell.size(); ell++)
                    Pell[ell].normalize();

                // Binomial coefficient
                auto binomial = [](int n, int k) -> double {
                    double res = 1.0;
                    for (int i = 0; i < k; i++) {
                        res *= double(n - i) / double(k - i);
                    }
                    return res;
                };

                // P_ell(x) = Sum_{k=0}^{ell/2} summand_legendre_polynomial * x^(ell - 2k)
                auto summand_legendre_polynomial = [&](int k, int ell) -> double {
                    double sign = (k % 2) == 0 ? 1.0 : -1.0;
                    return sign * binomial(ell, k) * binomial(2 * ell - 2 * k, ell) / std::pow(2.0, ell);
                };

                // Go from <mu^k |delta|^2> to (2ell+1) <L_ell(mu) |delta|^2>
                std::vector<std::vector<double>> temp;
                for (int ell = 0; ell < int(Pell.size()); ell++) {
                    std::vector<double> sum(Pell[0].pofk.size(), 0.0);
                    for (int k = 0; k <= ell / 2; k++) {
                        std::vector<double> & mu_power = Pell[ell - 2 * k].pofk;
                        for (size_t i = 0; i < sum.size(); i++)
                            sum[i] += mu_power[i] * summand_legendre_polynomial(k, ell) * double(2 * ell + 1);
                    }
                    temp.push_back(sum);
                }

                // Copy over data. We now have P0,P1,... in Pell
                for (size_t ell = 0; ell < Pell.size(); ell++) {
                    Pell[ell].pofk = temp[ell];
                }
            }
        } // namespace detail

        template <int N, template <int> class Grid>
        void compute_power_spectrum_multipoles_fourier(const Grid<N> & fourier_grid,
                                                       std::vector<PowerSpectrumBinning<N>> & Pell,
                                                       std::vector<double> line_of_sight_direction) {
            detail::power_spectrum_multipoles_fourier_impl<N>(
                fourier_grid,
                [&](IndexIntType fourier_index) { return double(std::norm(fourier_grid.get_fourier_from_index(fourier_index))); },
                Pell,
                line_of_sight_direction,
                "compute_power_spectrum_multipoles_fourier");
        }

        template <int N, template <int> class Grid>
        void compute_cross_power_spectrum_multipoles_fourier(const Grid<N> & fourier_grid_1,
                                                             const Grid<N> & fourier_grid_2,
                                                             std::vector<PowerSpectrumBinning<N>> & Pell,
                                                             std::vector<double> line_of_sight_direction) {
            assert_mpi(fourier_grid_1.get_nmesh() == fourier_grid_2.get_nmesh(),
                       "[compute_cross_power_spectrum_multipoles_fourier] Grids must have the same gridsize\n");
            // (the two grids have the same layout: the same index is the same mode)
            detail::power_spectrum_multipoles_fourier_impl<N>(
                fourier_grid_1,
                [&](IndexIntType fourier_index) {
                    const auto d1 = fourier_grid_1.get_fourier_from_index(fourier_index);
                    const auto d2 = fourier_grid_2.get_fourier_from_index(fourier_index);
                    return double(d1.real()) * double(d2.real()) + double(d1.imag()) * double(d2.imag());
                },
                Pell,
                line_of_sight_direction,
                "compute_cross_power_spectrum_multipoles_fourier");
        }

        /// @brief Loop over the local Fourier modes of a 3D grid (except k = 0): FML::GRID::for_each_nonzero_fourier_mode
        template <class F>
        void bin_up_fourier_modes_3d(const FFTWGrid<3> & grid, F && f) {
            FML::GRID::for_each_nonzero_fourier_mode(grid, std::forward<F>(f));
        }

        template <int N, template <int> class Grid>
        void bin_up_deconvolved_power_spectrum(const Grid<N> & fourier_grid,
                                               PowerSpectrumBinning<N> & pofk,
                                               std::string density_assignment_method) {

            assert_mpi(fourier_grid.get_nmesh() > 0, "[bin_up_deconvolved_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_deconvolved_power_spectrum] Binning has inconsistent parameters\n");

            const auto Nmesh = fourier_grid.get_nmesh();
            const FML::INTERPOLATION::WindowFunctionTable<N> window_function(density_assignment_method, Nmesh);

            // Initialize binning just in case
            pofk.reset();

            // Bin up P(k)
            FML::GRID::for_each_nonzero_fourier_mode(
                fourier_grid,
                [&](IndexIntType fourier_index, const std::array<double, N> & kvec, double kmag, double weight) {
                    auto delta = fourier_grid.get_fourier_from_index(fourier_index);
                    auto delta_norm = std::norm(delta);
                    auto window = window_function(kvec);
                    delta_norm /= (window * window);
                    pofk.add_to_bin(kmag, delta_norm, weight);
                });

            // Normalize to get P(k) (this communicates over tasks)
            pofk.normalize();
        }

        // Bin up the power-spectrum of a given fourier grid
        template <int N, template <int> class Grid>
        void bin_up_power_spectrum(const Grid<N> & fourier_grid, PowerSpectrumBinning<N> & pofk) {

            assert_mpi(fourier_grid.get_nmesh() > 0, "[bin_up_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_power_spectrum] Binning has inconsistent parameters\n");

            // Initialize binning just in case
            pofk.reset();

            // Bin up P(k)
            FML::GRID::for_each_nonzero_fourier_mode(
                fourier_grid,
                [&](IndexIntType fourier_index, const std::array<double, N> &, double kmag, double weight) {
                    auto delta = fourier_grid.get_fourier_from_index(fourier_index);
                    pofk.add_to_bin(kmag, std::norm(delta), weight);
                });

            // Normalize to get P(k) (this communicates over tasks)
            pofk.normalize();
        }

        // Bin up the cross power-spectrum of a given fourier grids
        template <int N, template <int> class Grid>
        void bin_up_cross_power_spectrum(const Grid<N> & fourier_grid_1,
                                         const Grid<N> & fourier_grid_2,
                                         PowerSpectrumBinning<N> & pofk) {

            assert_mpi(fourier_grid_1.get_nmesh() > 0, "[bin_up_cross_power_spectrum] grid must have Nmesh > 0\n");
            assert_mpi(fourier_grid_1.get_nmesh() == fourier_grid_2.get_nmesh(),
                       "[bin_up_cross_power_spectrum] Grids must have the same gridsize\n");
            assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0,
                       "[bin_up_cross_power_spectrum] Binning has inconsistent parameters\n");

            // Initialize binning just in case
            pofk.reset();

            // Take a copy and bin up the imaginary part also
            PowerSpectrumBinning<N> pofk_imag = pofk;
            pofk_imag.reset();

            // Bin up P(k) (the two grids have the same layout: the same index is the same mode)
            FML::GRID::for_each_nonzero_fourier_mode(
                fourier_grid_1,
                [&](IndexIntType fourier_index, const std::array<double, N> &, double kmag, double weight) {
                    auto delta_1 = fourier_grid_1.get_fourier_from_index(fourier_index);
                    auto delta_2 = fourier_grid_2.get_fourier_from_index(fourier_index);
                    auto delta12_real = delta_1.real() * delta_2.real() + delta_1.imag() * delta_2.imag();
                    auto delta12_imag = -delta_1.real() * delta_2.imag() + delta_1.imag() * delta_2.real();
                    pofk.add_to_bin(kmag, delta12_real, weight);
                    pofk_imag.add_to_bin(kmag, delta12_imag, weight);
                });

            // Normalize to get P(k) (this communicates over tasks)
            pofk.normalize();
            pofk_imag.normalize();

            // NB: we currently don't return the imaginary part. For real fields this should be zero
            // so we just check this and give a warning if its large
            const bool show_warning = false;
            if (show_warning) {
                for (int i = 0; i < pofk.n; i++) {
                    if (std::abs(pofk.pofk[i]) < 1e3 * std::abs(pofk_imag.pofk[i]) and
                        std::abs(pofk_imag.pofk[i]) > 1e-15) {
                        std::cout
                            << "Warning: the imaginary part of the cross spectrum is > 0.1%% times the real part [ k: "
                            << pofk.kbin[i] << " Real(d1d2): " << pofk.pofk[i] << " Imag(d1d2): " << pofk_imag.pofk[i]
                            << "]\n";
                    }
                }
            }
        }

        // Brute force. Add particles to the grid using direct summation
        //================================================================================
        /// @brief The shot noise of the density field of a set of particles as assigned by particles_to_grid:
        /// returns \f$ (S_2, S_3) \f$ with \f$ S_n = \sum m_i^n / (\sum m_i)^n \f$ (the shot noise of the
        /// power spectrum is \f$ S_2 \f$ and the constant term of the bispectrum \f$ S_3 \f$). For equal masses
        /// this is \f$ (1/N, 1/N^2) \f$. For particles with get_mass the masses are used and for delta-f particles
        /// (get_deltaf_weight) \f$ S_n = \sum w_i^n / N^n \f$. Sums over the MPI tasks.
        //================================================================================
        template <class T>
        std::pair<double, double> particle_shotnoise(const T * part, size_t NumPart, size_t NumPartTotal) {
            const double n = double(NumPartTotal);
            constexpr bool has_deltaf_weight = FML::PARTICLE::has_get_deltaf_weight<T>();
            constexpr bool has_mass = FML::PARTICLE::has_get_mass<T>() and not has_deltaf_weight;
            if constexpr (has_mass or has_deltaf_weight) {
                double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                for (size_t i = 0; i < NumPart; i++) {
                    double m;
                    if constexpr (has_deltaf_weight)
                        m = FML::PARTICLE::GetDeltafWeight(const_cast<T &>(part[i]));
                    else
                        m = FML::PARTICLE::GetMass(const_cast<T &>(part[i]));
                    sum1 += m;
                    sum2 += m * m;
                    sum3 += m * m * m;
                }
                FML::SumOverTasks(&sum1);
                FML::SumOverTasks(&sum2);
                FML::SumOverTasks(&sum3);
                // delta = sum_i w_i W / (N/Nmesh^N) for delta-f particles, sum_i m_i W / (sum m/Nmesh^N) otherwise
                const double norm = has_deltaf_weight ? n : sum1;
                return {sum2 / (norm * norm), sum3 / (norm * norm * norm)};
            }
            return {1.0 / n, 1.0 / (n * n)};
        }

        // This gives alias free P(k), but scales as O(Npart)*O(Nmesh^N)
        template <int N, class T>
        void compute_power_spectrum_direct_summation(int Ngrid,
                                                     const T * part,
                                                     size_t NumPart,
                                                     PowerSpectrumBinning<N> & pofk) {

            static_assert(
                FML::PARTICLE::has_get_pos<T>(),
                "[compute_power_spectrum_direct_summation] Particle class needs to have positions to use this method");

            // Very simple check to see if all tasks do have the same particles
            if (FML::NTasks > 1) {
                long long int tmp1 = NumPart;
                FML::MinOverTasks(&tmp1);
                long long int tmp2 = NumPart;
                FML::MaxOverTasks(&tmp2);
                double x = FML::PARTICLE::GetPos(const_cast<T &>(part[0]))[0];
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

            const std::complex<FML::GRID::FloatType> I(0, 1);
            const FML::GRID::FloatType norm = 1.0 / FML::GRID::FloatType(NumPart);

            FFTWGrid<N> density_k(Ngrid, 1, 1);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum_direct_summation::density_k");
            density_k.set_grid_status_real(false);

            for (auto && fourier_index : density_k.get_fourier_range()) {
                auto kvec = density_k.get_fourier_wavevector_from_index(fourier_index);
                FML::GRID::FloatType real = 0.0;
                FML::GRID::FloatType imag = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : real, imag)
#endif
                for (size_t i = 0; i < NumPart; i++) {
                    const auto * x = FML::PARTICLE::GetPos(const_cast<T &>(part[i]));
                    FML::GRID::FloatType kx = 0.0;
                    for (int idim = 0; idim < N; idim++) {
                        kx += kvec[idim] * x[idim];
                    }
                    auto val = std::exp(-kx * I);
                    real += val.real();
                    imag += val.imag();
                }

                std::complex<FML::GRID::FloatType> sum = {real, imag};
                if (ThisTask == 0 and fourier_index == 0)
                    sum -= 1.0;
                density_k.set_fourier_from_index(fourier_index, sum * norm);
            }

            // Bin up the power-spectrum
            bin_up_power_spectrum<N>(density_k, pofk);

            // Subtract shot-noise
            if (pofk.subtract_shotnoise)
                for (int i = 0; i < pofk.n; i++)
                    pofk.pofk[i] -= 1.0 / double(NumPart);
        }

        // Simple method to estimate multipoles from simulation data
        // Take particles in realspace and use their velocity to put them into
        // redshift space. Fourier transform and compute multipoles from this like in the method above.
        // We do this for all coordinate axes and return the mean P0,P1,... we get from this
        // velocity_to_displacement is factor to convert your velocity to a coordinate shift in [0,1)
        // e.g. c/(a H Box) with H ~ 100 h km/s/Mpc and Box boxsize in Mpc/h if velocities are peculiar
        template <int N, class T, template <int> class Grid>
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
            const auto nleftright = get_extra_slices(density_assignment_method);
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

            // Allocate density grid (decomposed as the particles)
            auto density_k = FML::GRID::make_grid<Grid<N>>(Ngrid, nleft, nright, part.get_domain());
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
            if (Pell[0].subtract_shotnoise) {
                const double shotnoise =
                    particle_shotnoise(part.get_particles_ptr(), part.get_npart(), part.get_npart_total()).first;
                for (int i = 0; i < Pell[0].n; i++) {
                    Pell[0].pofk[i] -= shotnoise;
                }
            }
        }

        //================================================================================
        // A simple power-spectrum estimator - nothing fancy
        // Deconvolving the window-function and subtracting shot-noise (1/NumPartTotal)
        //================================================================================
        template <int N, class T, template <int> class Grid>
        void compute_power_spectrum(int Ngrid,
                                    T * part,
                                    size_t NumPart,
                                    size_t NumPartTotal,
                                    PowerSpectrumBinning<N> & pofk,
                                    std::string density_assignment_method,
                                    bool interlacing,
                                    const FML::Domain & domain) {

            // Set how many extra slices we need for the density assignment to go smoothly
            const auto nleftright = get_extra_slices(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + (interlacing ? 1 : 0);

            // Bin particles to grid (decomposed as the domain of the particles: the fiducial FML::domain unless given)
            auto density_k = FML::GRID::make_grid<Grid<N>>(Ngrid, nleft, nright, domain);
            density_k.add_memory_label("FFTWGrid::compute_power_spectrum::density_k");

            if (interlacing) {

                // Bin to grid using interlaced grids to produce a fourier space density field
                FML::INTERPOLATION::particles_to_fourier_grid_interlacing(part,
                                                                          NumPart,
                                                                          NumPartTotal,
                                                                          density_k,
                                                                          density_assignment_method);
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
            if (pofk.subtract_shotnoise) {
                const double shotnoise = particle_shotnoise(part, NumPart, NumPartTotal).first;
                for (int i = 0; i < pofk.n; i++) {
                    pofk.pofk[i] -= shotnoise;
                }
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

        template <int N, class T, bool COMPACT, template <int> class Grid>
        void compute_monospectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  MonospectrumBinning<N, COMPACT> & pofk,
                                  std::string density_assignment_method,
                                  bool interlacing) {

            // This will subtract shot-noise if pofk.subtract_shotnoise = true
            compute_polyspectrum<N, T, 2, COMPACT, Grid>(Ngrid,
                                          part,
                                          NumPart,
                                          NumPartTotal,
                                          pofk,
                                          density_assignment_method,
                                          interlacing);
        }

        template <int N, class T, bool COMPACT, template <int> class Grid>
        void compute_bispectrum(int Ngrid,
                                T * part,
                                size_t NumPart,
                                size_t NumPartTotal,
                                BispectrumBinning<N, COMPACT> & bofk,
                                std::string density_assignment_method,
                                bool interlacing) {

            // This will subtract shot-noise if bofk.subtract_shotnoise = true
            compute_polyspectrum<N, T, 3, COMPACT, Grid>(Ngrid,
                                          part,
                                          NumPart,
                                          NumPartTotal,
                                          bofk,
                                          density_assignment_method,
                                          interlacing);
        }

        template <int N, class T, bool COMPACT, template <int> class Grid>
        void compute_trispectrum(int Ngrid,
                                 T * part,
                                 size_t NumPart,
                                 size_t NumPartTotal,
                                 TrispectrumBinning<N, COMPACT> & tofk,
                                 std::string density_assignment_method,
                                 bool interlacing) {
            compute_polyspectrum<N, T, 4, COMPACT, Grid>(Ngrid,
                                          part,
                                          NumPart,
                                          NumPartTotal,
                                          tofk,
                                          density_assignment_method,
                                          interlacing);
        }

        // Computes the polyspectrum P(k1,k2,k3,...)
        template <int N, class T, int ORDER, bool COMPACT, template <int> class Grid>
        void compute_polyspectrum(int Ngrid,
                                  T * part,
                                  size_t NumPart,
                                  size_t NumPartTotal,
                                  PolyspectrumBinning<N, ORDER, COMPACT> & polyofk,
                                  std::string density_assignment_method,
                                  bool interlacing,
                                    const FML::Domain & domain) {

            // Set how many extra slices we need for the density assignment to go smoothly
            const auto nleftright = get_extra_slices(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + 1;

            auto density_k = FML::GRID::make_grid<Grid<N>>(Ngrid, nleft, nright, domain);
            density_k.add_memory_label("FFTWGrid::compute_polyspectrum::density_k");

            if (interlacing) {

                // Bin particles to grid (use interlaced to reduce alias) and deconvolve window
                FML::INTERPOLATION::particles_to_fourier_grid_interlacing(part,
                                                                          NumPart,
                                                                          NumPartTotal,
                                                                          density_k,
                                                                          density_assignment_method);
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);

            } else {

                // Bin to grid, fourier transform and deconvolve window function
                particles_to_grid<N, T>(part, NumPart, NumPartTotal, density_k, density_assignment_method);
                density_k.fftw_r2c();
                deconvolve_window_function_fourier<N>(density_k, density_assignment_method);
            }

            // Compute polyspectrum
            compute_polyspectrum<N, ORDER>(density_k, polyofk);

            // Subtract shotnoise if ORDER = 2 or 3
            if constexpr (ORDER == 2) {
                // Subtract shot-noise (along the diagonal)
                if (polyofk.subtract_shotnoise) {
                    const double shotnoise = particle_shotnoise(part, NumPart, NumPartTotal).first;
                    for (int i = 0; i < polyofk.n; i++) {
                        polyofk.pofk[i] -= shotnoise;
                        std::array<int, 2> ik{i, i};
                        if (polyofk.get_bincount(ik) > 0.0)
                            polyofk.P123[polyofk.get_index_from_coord(ik)] -= shotnoise;
                    }
                }
            } else if constexpr (ORDER == 3) {
                // Subtract shot-noise for both P(k) and B(k1,k2,k3)
                if (polyofk.subtract_shotnoise) {
                    // S2 = 1/n and S3 = 1/n^2 for equal masses (sum m^2/(sum m)^2 and sum m^3/(sum m)^3 otherwise)
                    const auto [S2, S3] = particle_shotnoise(part, NumPart, NumPartTotal);
                    for (int i = 0; i < polyofk.n; i++) {
                        polyofk.pofk[i] -= S2;
                    }
                    // For Poisson sampling <d1 d2 d3> = B + (P1 + P2 + P3) S2 + S3 with P the shot noise
                    // subtracted power spectrum (which pofk is now)
                    for (size_t i = 0; i < polyofk.P123.size(); i++) {
                        if (polyofk.N123[i] <= 0.0)
                            continue;
                        auto ik = polyofk.get_coord_from_index(i);
                        polyofk.P123[i] -= S2 * (polyofk.pofk[ik[0]] + polyofk.pofk[ik[1]] + polyofk.pofk[ik[2]]) + S3;
                    }
                }
            }
        }

        namespace detail {
            template <int ORDER>
            int polyspectrum_work_mesh(int nmesh, const std::vector<double> & khigh, bool dealias) {
                if (!dealias || ORDER == 2 || ORDER * *std::max_element(khigh.begin(), khigh.end()) < 2 * M_PI * nmesh)
                    return nmesh;
                // Only the zero mode of the ORDER-fold product is needed. Excluding
                // input Nyquist planes gives |sum n_i| < ORDER*Nmesh/2 in each axis.
                const long long multiple = std::lcm(2LL, static_cast<long long>(FML::NTasks));
                const long long minimum = (ORDER * static_cast<long long>(nmesh) + 1) / 2;
                const long long padded = ((minimum + multiple - 1) / multiple) * multiple;
                assert_mpi(padded <= INT_MAX, "[polyspectrum] Padded mesh exceeds int range");
                return int(padded);
            }
            /// The work mesh (polyspectrum_work_mesh) made to fit the decomposition of a grid (FFTWGrid: as it is)
            template <int ORDER, class Grid>
            int polyspectrum_work_mesh_like(const Grid & like, int nmesh, const std::vector<double> & khigh, bool dealias) {
                int mesh = polyspectrum_work_mesh<ORDER>(nmesh, khigh, dealias);
                if constexpr (not FML::GRID::is_fftwgrid_v<Grid>) {
                    while (mesh != nmesh and not fits_grid_like(like, mesh))
                        mesh += 2;
                }
                return mesh;
            }
            /// The bytes of a shell grid of a mesh decomposed as like (0: FFTWGrid computes them itself)
            template <class Grid>
            size_t shell_bytes_like(const Grid & like, int mesh) {
                if constexpr (FML::GRID::is_fftwgrid_v<Grid>)
                    return 0;
                else
                    return make_grid_like(like, mesh, 0, 0).allocated_storage_bytes();
            }
            template <int N, int ORDER, class Grid>
            bool polyspectrum_uses_mode(const Grid & grid,
                                        const std::array<double, N> & wave,
                                        double k2,
                                        double low2,
                                        double high2,
                                        bool dealias) {
                if (!(k2 > 0 && k2 >= low2 && k2 < high2))
                    return false;
                if (dealias && ORDER > 2 && grid.get_nmesh() % 2 == 0)
                    for (auto k : wave)
                        if (std::abs(k) >= M_PI * grid.get_nmesh())
                            return false;
                return true;
            }

            /// The weight of a Fourier mode in the shell of a bin: 0 or 1 from where the cell centre is
            /// (polyspectrum_uses_mode), or with shell_overlap_weights the fraction of the cell inside the shell
            /// (ShellOverlap.h). The ball overlaps are remembered per cell (up to the symmetries of the grid) and edge,
            /// as every shell is built more than once. Not thread safe
            template <int N, int ORDER>
            class PolyspectrumShellWeights {
              public:
                template <class Binning>
                explicit PolyspectrumShellWeights(const Binning & bins)
                    : overlap(bins.shell_overlap_weights), dealias(bins.dealias), klow(bins.klow), khigh(bins.khigh) {
                    static_assert(N >= 1, "[PolyspectrumShellWeights] N >= 1");
                    assert_mpi(not overlap or N <= 3,
                               "[polyspectrum] shell_overlap_weights is only implemented for N = 1, 2 and 3");
                    // The distinct edges in units of the fundamental mode (k = 2 pi n for a box of size 1)
                    for (const auto & edges : {klow, khigh})
                        for (double k : edges)
                            if (edge_id.find(k) == edge_id.end()) {
                                edge_id[k] = int(radius.size());
                                radius.push_back(k / (2.0 * M_PI));
                            }
                }

                template <class Grid>
                double operator()(const Grid & grid, const std::array<double, N> & wave, double k2, int bin) {
                    if (not overlap)
                        return polyspectrum_uses_mode<N, ORDER>(
                                   grid, wave, k2, klow[bin] * klow[bin], khigh[bin] * khigh[bin], dealias) ?
                                   1.0 :
                                   0.0;
                    if (!(k2 > 0))
                        return 0.0;
                    if (dealias && ORDER > 2 && grid.get_nmesh() % 2 == 0)
                        for (auto k : wave)
                            if (std::abs(k) >= M_PI * grid.get_nmesh())
                                return 0.0;
                    if constexpr (N <= 3) {
                        // The cell up to reflections and permutations: sorted |n|
                        std::array<double, N> n;
                        for (int d = 0; d < N; d++)
                            n[d] = std::round(std::fabs(wave[d]) / (2.0 * M_PI));
                        std::sort(n.begin(), n.end());
                        const double d = std::sqrt(k2) / (2.0 * M_PI), half_diagonal = 0.5 * std::sqrt(double(N));
                        const double r_low = radius[edge_id[klow[bin]]], r_high = radius[edge_id[khigh[bin]]];
                        if (d + half_diagonal <= r_low or d - half_diagonal >= r_high)
                            return 0.0;
                        if (d - half_diagonal >= r_low and d + half_diagonal <= r_high)
                            return 1.0;
                        return std::clamp(ball(n, edge_id[khigh[bin]], d) - ball(n, edge_id[klow[bin]], d), 0.0, 1.0);
                    }
                    return 0.0;
                }

              private:
                bool overlap, dealias;
                std::vector<double> klow, khigh, radius;
                std::map<double, int> edge_id;
                std::unordered_map<uint64_t, double> memory;

                /// The volume of the unit cell centred at n (>= 0, sorted) inside the ball of edge e
                double ball(const std::array<double, N> & n, int e, double d) {
                    const double r = radius[e], half_diagonal = 0.5 * std::sqrt(double(N));
                    if (d + half_diagonal <= r)
                        return 1.0;
                    if (d - half_diagonal >= r)
                        return 0.0;
                    uint64_t key = uint64_t(e);
                    for (int i = 0; i < N; i++)
                        key = key * (uint64_t(1) << 20) + uint64_t(n[i]);
                    auto found = memory.find(key);
                    if (found != memory.end())
                        return found->second;
                    std::array<double, N> lo, hi;
                    for (int i = 0; i < N; i++) {
                        lo[i] = n[i] - 0.5;
                        hi[i] = n[i] + 0.5;
                    }
                    const double v = FML::CORRELATIONFUNCTIONS::SHELLOVERLAP::box_ball_overlap<N>(lo, hi, r);
                    memory[key] = v;
                    return v;
                }
            };
        } // namespace detail

        //================================================================================
        /// @brief This method is used by compute_polyspectrum. It computes the number of
        /// generalized triangles of the bins needed to normalize the polyspectra up to symmetry (i.e. we only compute
        /// it for \f$ k_1 \leq k_2 \leq k_3 \f$ and only for valid triangle configurations \f$ k_1 + k_2 \geq k_3 \f$
        /// and then set rest using symmetry). If one is to compute many spectra with the same Nmesh and binning then
        /// one can precompute N123 and set it using polyofk.set_bincount(N123) (which sets polyofk.bincount_is_set =
        /// true and avoid a call to this function). Counts computed here are cached automatically; reuse checks
        /// the source mesh, shell edges and closure policy. Prefer set_bincount(counts, Nmesh) for imported data.
        ///
        /// @tparam N The dimension we work in
        /// @tparam ORDER The order (mono = 2, bi = 3, tri = 4)
        ///
        /// @param[in] Nmesh The size of the grid we us
        /// @param[out] polyofk The binning (we compute and store the volumes of each bin, N123, in this binning).
        ///
        //================================================================================
        namespace detail {
            /// The bin counts with shell grids decomposed as like (FFTWGrid: the slabs)
            template <int N, int ORDER, bool COMPACT, template <int> class Grid>
            void compute_polyspectrum_bincount_like(const Grid<N> & like,
                                                    int Nmesh,
                                                    PolyspectrumBinning<N, ORDER, COMPACT> & bins) {
            using FML::GRID::make_grid_like;
            constexpr bool is_fftwgrid = std::is_same_v<Grid<N>, FFTWGrid<N>>;
            assert_mpi(Nmesh > 0 && bins.n > 0 && bins.klow.size() == size_t(bins.n) &&
                           bins.khigh.size() == size_t(bins.n),
                       "[compute_polyspectrum_bincount] Mesh and shell binning must be initialized");
            bins.prepare_storage();
            assert_mpi(bins.valid_storage(), "[compute_polyspectrum_bincount] Invalid result storage");
            bins.statistics = {};
            const auto configs = bins.configurations();
            bins.statistics.memory.configuration_bytes = configs.capacity() * sizeof(configs[0]);
            double requested_kmax = 0;
            for (const auto & cfg : configs)
                for (auto bin : cfg)
                    requested_kmax = std::max(requested_kmax, bins.khigh[bin]);
            const int mesh = detail::polyspectrum_work_mesh_like<ORDER>(like, Nmesh, {requested_kmax}, bins.dealias);
            detail::PolyspectrumShellWeights<N, ORDER> shell_weight(bins);
            auto build = [&](int bin) {
                auto field = make_grid_like(like, Nmesh, 0, 0);
                field.set_grid_status_real(false);
                for (auto index : field.get_fourier_range()) {
                    std::array<double, N> wave;
                    double k2;
                    field.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                    field.set_fourier_from_index(index, shell_weight(field, wave, k2, bin));
                }
                if (mesh != Nmesh) {
                    auto & memory = bins.statistics.memory;
                    memory.padding_source_bytes =
                        std::max(memory.padding_source_bytes, field.allocated_storage_bytes());
                    if constexpr (is_fftwgrid) {
                        FFTWGrid<N> padded(mesh);
                        FML::GRID::FourierResamplingPlan<N> plan(field, padded);
                        plan.apply(field, padded);
                        memory.remapping_bytes = std::max(memory.remapping_bytes, plan.workspace_storage_bytes());
                        field = std::move(padded);
                    } else {
                        field = FML::GRID::resize_fourier_grid(field, mesh);
                    }
                }
                field.fftw_c2r();
                return field;
            };
            detail::ShellCache<N, ORDER, Grid<N>> cache(bins.n,
                                                        mesh,
                                                        configs,
                                                        bins.options,
                                                        bins.statistics,
                                                        bins.statistics.geometry_shell_builds,
                                                        build,
                                                        detail::shell_bytes_like(like, mesh));
            std::fill(bins.N123.begin(), bins.N123.end(), 0.0);
            const double measure = std::pow(2 * M_PI, N);
            const double norm = std::pow(1.0 / mesh / (2 * M_PI), N);
            detail::integrate_configurations<N, ORDER>(
                cache,
                configs,
                bins.options.reduction_batch_size,
                bins.statistics.geometry_reduction_batches,
                [&](auto cfg, double sum) {
                    double value = sum * norm;
                    // Counts are integers (round-off below 0.5); with overlap weights any positive count
                    if (value * measure < (bins.shell_overlap_weights ? 1e-6 : 0.5))
                        value = 0;
                    detail::store_permutations<N, ORDER>(bins, bins.N123, cfg, value);
                });
            bins.mark_bincount_computed(Nmesh);
            detail::finish_memory_report(bins);
        }
        } // namespace detail

        template <int N, int ORDER, bool COMPACT>
        void compute_polyspectrum_bincount(int Nmesh, PolyspectrumBinning<N, ORDER, COMPACT> & bins) {
            detail::compute_polyspectrum_bincount_like<N, ORDER>(FFTWGrid<N>(), Nmesh, bins);
        }

        template <int N, bool COMPACT, template <int> class Grid>
        void compute_monospectrum(const Grid<N> & fourier_grid, PolyspectrumBinning<N, 2, COMPACT> & pofk) {
            compute_polyspectrum<N, 2>(fourier_grid, pofk);
        }

        template <int N, bool COMPACT, template <int> class Grid>
        void compute_bispectrum(const Grid<N> & fourier_grid, PolyspectrumBinning<N, 3, COMPACT> & bofk) {
            compute_polyspectrum<N, 3>(fourier_grid, bofk);
        }

        template <int N, bool COMPACT, template <int> class Grid>
        void compute_trispectrum(const Grid<N> & fourier_grid, PolyspectrumBinning<N, 4, COMPACT> & tofk) {
            compute_polyspectrum<N, 4>(fourier_grid, tofk);
        }

        template <int N, int ORDER, bool COMPACT, template <int> class Grid>
        void compute_polyspectrum(const Grid<N> & input, PolyspectrumBinning<N, ORDER, COMPACT> & bins) {
            static_assert(ORDER > 1);
            using FML::GRID::make_grid_like;
            constexpr bool is_fftwgrid = std::is_same_v<Grid<N>, FFTWGrid<N>>;
            const int Nmesh = input.get_nmesh();
            assert_mpi(Nmesh > 0 && bins.n > 0 && bins.klow.size() == size_t(bins.n) &&
                           bins.khigh.size() == size_t(bins.n) && bins.pofk.size() == size_t(bins.n) &&
                           bins.kmean.size() == size_t(bins.n),
                       "[compute_polyspectrum] Invalid grid or binning");
            bins.prepare_storage();
            assert_mpi(bins.valid_storage(), "[compute_polyspectrum] Invalid result storage");
            bins.statistics = {};
            if (!bins.has_bincount(Nmesh))
                detail::compute_polyspectrum_bincount_like<N, ORDER>(input, Nmesh, bins);
            const auto configs = bins.configurations();
            bins.statistics.memory.configuration_bytes = configs.capacity() * sizeof(configs[0]);
            double requested_kmax = 0;
            for (const auto & cfg : configs)
                for (auto bin : cfg)
                    requested_kmax = std::max(requested_kmax, bins.khigh[bin]);
            const int mesh = detail::polyspectrum_work_mesh_like<ORDER>(input, Nmesh, {requested_kmax}, bins.dealias);
            // Auxiliary P(k) and mean k cover every shell, including unrequested configurations.
            // Their statistics require no shell meshes and one array reduction.
            const size_t metadata_count = 3 * size_t(bins.n);
            assert_mpi(metadata_count <= size_t(INT_MAX), "[polyspectrum] Too many shell statistics for MPI");
            std::vector<double> metadata(metadata_count, 0.0);
            detail::PolyspectrumShellWeights<N, ORDER> shell_weight(bins);
            for (auto index : input.get_fourier_range()) {
                std::array<double, N> wave;
                double k2;
                input.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                for (int bin = 0; bin < bins.n; bin++) {
                    const double fraction = shell_weight(input, wave, k2, bin);
                    if (fraction == 0.0)
                        continue;
                    const double weight = fraction * FML::GRID::fourier_mode_multiplicity(
                        is_fftwgrid ? int(index % (Nmesh / 2 + 1)) : input.get_fourier_integer_wavevector_from_index(index)[N - 1],
                        Nmesh);
                    metadata[3 * bin] += weight * std::sqrt(k2);
                    metadata[3 * bin + 1] += weight * std::norm(input.get_fourier_from_index(index));
                    metadata[3 * bin + 2] += weight;
                }
            }
            FML::SumArrayOverTasks(metadata.data(), metadata.size());
            bins.statistics.metadata_reduction_batches = 1;
            bins.statistics.memory.metadata_work_bytes = metadata.capacity() * sizeof(double);
            bins.statistics.memory.input_bytes = input.allocated_storage_bytes();
            for (int bin = 0; bin < bins.n; bin++) {
                const double count = metadata[3 * bin + 2];
                bins.kmean[bin] = count ? metadata[3 * bin] / count : bins.kbin[bin];
                bins.pofk[bin] = count ? metadata[3 * bin + 1] / count : 0.0;
            }
            auto build = [&](int bin) {
                auto field = make_grid_like(input, Nmesh, 0, 0);
                field.set_grid_status_real(false);
                for (auto index : input.get_fourier_range()) {
                    std::array<double, N> wave;
                    double k2;
                    input.get_fourier_wavevector_and_norm2_by_index(index, wave, k2);
                    const double fraction = shell_weight(input, wave, k2, bin);
                    field.set_fourier_from_index(index,
                                                 fraction != 0.0 ? fraction * input.get_fourier_from_index(index) :
                                                                   0.0);
                }
                if (mesh != Nmesh) {
                    auto & memory = bins.statistics.memory;
                    memory.padding_source_bytes =
                        std::max(memory.padding_source_bytes, field.allocated_storage_bytes());
                    if constexpr (is_fftwgrid) {
                        FFTWGrid<N> padded(mesh);
                        FML::GRID::FourierResamplingPlan<N> plan(field, padded);
                        plan.apply(field, padded);
                        memory.remapping_bytes = std::max(memory.remapping_bytes, plan.workspace_storage_bytes());
                        field = std::move(padded);
                    } else {
                        field = FML::GRID::resize_fourier_grid(field, mesh);
                    }
                }
                field.fftw_c2r();
                return field;
            };
            detail::ShellCache<N, ORDER, Grid<N>> cache(bins.n,
                                                     mesh,
                                                     configs,
                                                     bins.options,
                                                     bins.statistics,
                                                     bins.statistics.field_shell_builds,
                                                     build,
                                                     detail::shell_bytes_like(input, mesh));
            std::fill(bins.P123.begin(), bins.P123.end(), 0.0);
            const double norm = std::pow(1.0 / mesh / (2 * M_PI), N);
            detail::integrate_configurations<N, ORDER>(
                cache,
                configs,
                bins.options.reduction_batch_size,
                bins.statistics.field_reduction_batches,
                [&](auto cfg, double sum) {
                    const double count = bins.N123[bins.get_index_from_coord(cfg)];
                    detail::store_permutations<N, ORDER>(bins, bins.P123, cfg, count > 0 ? sum * norm / count : 0.0);
                });
            detail::finish_memory_report(bins);
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
