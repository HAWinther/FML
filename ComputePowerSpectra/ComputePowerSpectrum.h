#ifndef COMPUTEPOWERSPECTRUM_HEADER
#define COMPUTEPOWERSPECTRUM_HEADER

#include <vector>
#include <iostream>
#include <cassert>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/Global/Global.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h> // Only for compute_multipoles from particles

//=====================================================================
//=====================================================================

namespace FML {
  namespace CORRELATIONFUNCTIONS {

    using namespace FML::INTERPOLATION;
    
    template<int N>
      using FFTWGrid = FML::GRID::FFTWGrid<N>;

    // Keep track of everything we need for a binned power-spectrum
    // The k-spacing (linear or log or ...), the count in each bin,
    // the power in each bin etc. Its through this interface the results
    // of the methods below are given
    struct PowerSpectrumBinning;

    // Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
    // Fourier transform, decolvolve the window function for the assignement above,
    // bin up power-spectrum and subtract shot-noise 1/NumPartTotal
    template<int N, class T>
      void compute_power_spectrum(
          int Ngrid,
          T *part,
          size_t NumPart,
          size_t NumPartTotal,
          PowerSpectrumBinning &pofk,
          std::string density_assignment_method);
    
    // Assign particles to grid using density_assignment_method = NGP,CIC,TSC,PCS,...
    // Assign particles to an interlaced grid (displaced by dx/2 in all directions)
    // Fourier transform both and add the together to cancel the leading aliasing contributions
    // Decolvolve the window function for the assignements above,
    // bin up power-spectrum and subtract shot-noise 1/NumPartTotal
    template<int N, class T>
      void compute_power_spectrum_interlacing(
          int Ngrid,
          T *part,
          size_t NumPart,
          size_t NumPartTotal,
          PowerSpectrumBinning &pofk,
          std::string density_assignment_method);

    // Brute force. Add particles to the grid using direct summation
    // This gives alias free P(k), but scales as O(Npart)*O(Nmesh^N)
    template<int N, class T>
      void compute_power_spectrum_direct_summation(
          int Ngrid, 
          T *part, 
          size_t NumPart, 
          PowerSpectrumBinning &pofk);

    // Bin up the power-spectrum of a given fourier grid
    template <int N>
      void bin_up_power_spectrum(
          FFTWGrid<N> &fourier_grid,
          PowerSpectrumBinning &pofk);

    // Compute power-spectrum multipoles P0,P1,...,Pn-1 for the case
    // where we have a fixed line_of_sight_direction (typical coordinate axes like (0,0,1))
    // Pell contains P0,P1,P2,...Pell_max where ell_max = n-1 is the size of Pell
    template <int N>
      void compute_power_spectrum_multipoles(
          FFTWGrid<N> &fourier_grid, 
          std::vector<PowerSpectrumBinning> &Pell,
          std::vector<double> line_of_sight_direction);
   
    // Simple method to estimate multipoles from simulation data
    // Take particles in realspace and use their velocity to put them into
    // redshift space. Fourier transform and compute multipoles from this like in the method above.
    // We do this for all coordinate axes and return the mean P0,P1,... we get from this
    // velocity_to_displacement is factor to convert your velocity to a coordinate shift in [0,1)
    // e.g. c/(a H Box) with H ~ 100 h km/s/Mpc and Box boxsize in Mpc/h if velocities are peculiar
    template<int N, class T>
      void compute_power_spectrum_multipoles(
          int Ngrid,
          FML::PARTICLE::MPIParticles<T> & part,
          double velocity_to_displacement,
          std::vector<PowerSpectrumBinning> &Pell,
          std::string density_assignment_method) ;

    //=====================================================================
    //=====================================================================

    // This class is thread-safe, but cannot be created or normalized inside a OMP parallel region
    // Summation over MPI tasks is done in normalize()
    typedef struct PowerSpectrumBinning {
      enum BinningType {
        LINEAR_SPACING,
        LOG_SPACING
      };

      std::vector<double> k;
      std::vector<double> count; 
      std::vector<double> pofk; 
      std::vector<double> kbin; 

#ifdef USE_OMP
      std::vector< std::vector<double> > count_thread; 
      std::vector< std::vector<double> > pofk_thread; 
      std::vector< std::vector<double> > kbin_thread; 
#endif

      int n;
      int bin_type;
      double kmin;
      double kmax;

      PowerSpectrumBinning();
      PowerSpectrumBinning(const int n);
      PowerSpectrumBinning(const double kmin, const double kmax, const int n, const int bin_type);

      // Add two binnings together
      PowerSpectrumBinning & operator +=(const PowerSpectrumBinning & rhs){
        assert(k.size() == rhs.k.size());
        assert(count.size() == rhs.count.size());
        assert(pofk.size() == rhs.pofk.size());
        assert(kbin.size() == rhs.kbin.size());
        assert(bin_type == rhs.bin_type);
        assert(kmin == rhs.kmin);
        assert(kmax == rhs.kmax);
        for(int i = 0; i < n; i++){
          count[i] += rhs.count[i];
          pofk[i] += rhs.pofk[i];
          kbin[i] += rhs.kbin[i];
        }
        return *this;
      }
      
      // Reset everything. Call before starting to bim
      void reset();

      // Add a new point to a bin
      void add_to_bin(double kvalue, double power, double weight = 1.0);
      
      // Normalize (i.e. find mean in each bin) Do summation over MPI tasks
      void normalize();

      // Scale to physical units kscale = 1/Box and pofkscale = Box^N
      void scale(const double kscale, const double pofkscale);

      // For controlling the bins
      int get_bin_index(const double kvalue, const double kmin, const double kmax, const int n, const int bin_type);
      double get_k_from_bin_index(const int index, const double kmin, const double kmax, const int n, const int bin_type);
      
      // Print some info
      void print_info();

    } PowerSpectrumBinning;

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
      void compute_power_spectrum_multipoles(
          FFTWGrid<N> &fourier_grid, 
          std::vector<PowerSpectrumBinning> &Pell,
          std::vector<double> line_of_sight_direction) { 

        assert_mpi(line_of_sight_direction.size() == N,
            "[compute_power_spectrum_multipoles] Line of sight direction has wrong number of dimensions\n");
        assert_mpi(Pell.size() > 0, 
            "[compute_power_spectrum_multipoles] Pell must have size > 0\n");
        assert_mpi(fourier_grid.get_nmesh() > 0, 
            "[compute_power_spectrum_multipoles] grid must have Nmesh > 0\n");

        int Nmesh           = fourier_grid.get_nmesh();
        auto Local_nx       = fourier_grid.get_local_nx();
        auto NmeshTotTotal  = Local_nx * power(Nmesh,N-2) * (Nmesh/2+1);//fourier_grid.get_ntot_fourier();
        auto *cdelta = fourier_grid.get_fourier_grid();

        // Norm of LOS vector
        double rnorm = 0.0;
        for(int idim = 0; idim < N; idim++)
          rnorm += line_of_sight_direction[idim]*line_of_sight_direction[idim];
        rnorm = std::sqrt(rnorm);
        assert_mpi(rnorm > 0.0, 
            "[compute_power_spectrum_multipoles] Line of sight vector has zero length\n");

        // Initialize binning just in case
        for (int ell = 0; ell < Pell.size(); ell++)
          Pell[ell].reset();

        // Bin up mu^k |delta|^2
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t ind = 0; ind < NmeshTotTotal; ind++) {
          // Special treatment of k = 0 plane
          int last_coord = ind % (Nmesh / 2 + 1);
          double weight = last_coord > 0 && last_coord < Nmesh / 2 ? 2.0 : 1.0;

          // Compute kvec, |kvec| and |delta|^2
          double kmag;
          std::vector<double> kvec(N);
          fourier_grid.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);
          double power = std::norm(cdelta[ind]);
          
          // Compute mu = k_vec*r_vec
          double mu = 0.0;
          for(int idim = 0; idim < N; idim++)
            mu += kvec[idim] * line_of_sight_direction[idim];
          mu /= (kmag * rnorm);

          // Add to bin |delta|^2, |delta|^2mu^2, |delta^2|mu^4, ...
          double mutoell = 1.0;
          for (int ell = 0; ell < Pell.size(); ell++) {
            Pell[ell].add_to_bin(kmag, power * mutoell, weight);
            mutoell *= mu;
            //Pell[ell].add_to_bin(kmag, power * std::pow(mu, ell), weight);
          }
        }

#ifdef USE_MPI
        for (int ell = 0; ell < Pell.size(); ell++) {
          MPI_Allreduce(MPI_IN_PLACE, Pell[ell].pofk.data(),  Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(MPI_IN_PLACE, Pell[ell].count.data(), Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(MPI_IN_PLACE, Pell[ell].kbin.data(),  Pell[ell].n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif

        // Normalize
        for (int ell = 0; ell < Pell.size(); ell++)
          Pell[ell].normalize();

        // Binomial coefficient
        auto binomial = [](const double n, const double k){
          double res = 1.0;
          for (int i = 0; i < k; i++) {
            res *= double(n - i) / double(k - i);
          }
          return res;
        };

        // P_ell(x) = Sum_{k=0}^{ell/2} summand_legendre_polynomial * x^(ell - 2k)
        auto summand_legendre_polynomial = [&](const int k, const int ell) {
          double sign = (k % 2) == 0 ? 1.0 : -1.0;
          return sign * binomial(ell, k) * binomial(2 * ell - 2 * k, ell) /
            std::pow(2.0, ell);
        };

        // Go from <mu^k |delta|^2> to <L_ell(mu) |delta|^2>
        std::vector<std::vector<double>> temp;
        for (int ell = 0; ell < Pell.size(); ell++) {
          std::vector<double> sum(Pell[0].pofk.size(), 0.0);
          for (int k = 0; k <= ell / 2; k++) {
            std::vector<double> mu_power = Pell[ell - 2 * k].pofk;
            for(size_t i = 0; i < sum.size(); i++)
              sum[i] += mu_power[i] * summand_legendre_polynomial(k, ell);
          }
          temp.push_back(sum);
        }

        // Copy over data. We now have P0,P1,... in Pell
        for (int ell = 0; ell < Pell.size(); ell++) {
          Pell[ell].pofk = temp[ell];
        }
      }

    //==========================================================================================
    // Compute the power-spectrum of a fourier grid. The result has no scales. Get
    // scales by calling pofk.scale(kscale, pofkscale) with kscale = 1/Boxsize and
    // pofkscale = Boxsize^N once spectrum has been computed
    //==========================================================================================
    template <int N>
      void bin_up_power_spectrum(
          FFTWGrid<N> &fourier_grid,
          PowerSpectrumBinning &pofk) {

        assert_mpi(fourier_grid.get_nmesh() > 0, 
            "[bin_up_power_spectrum] grid must have Nmesh > 0\n");
        assert_mpi(pofk.n > 0 && pofk.kmax > pofk.kmin && pofk.kmin >= 0.0, 
            "[bin_up_power_spectrum] Binning has inconsistent parameters\n");

        auto *cdelta = fourier_grid.get_fourier_grid();
        int Nmesh = fourier_grid.get_nmesh();
        auto Local_nx = fourier_grid.get_local_nx();
        auto NmeshTotLocal = Local_nx * power(Nmesh,N-2) * (Nmesh/2+1);//fourier_grid.get_ntot_fourier();

        // Initialize binning just in case
        pofk.reset();

        // Bin up P(k)
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t ind = 0; ind < NmeshTotLocal; ind++) {
          // Special treatment of k = 0 plane
          int last_coord = ind % (Nmesh / 2 + 1);
          double weight = last_coord > 0 && last_coord < Nmesh / 2 ? 2.0 : 1.0;

          auto delta_norm = std::norm(cdelta[ind]);

          // Add norm to bin
          double kmag;
          std::vector<double> kvec(N);
          fourier_grid.get_fourier_wavevector_and_norm_by_index(ind, kvec, kmag);
          pofk.add_to_bin(kmag, delta_norm, weight);
        }

        // Normalize to get P(k) (this communicates over tasks)
        pofk.normalize();
      }

    //==============================================================================================
    // Brute force (but aliasing free) computation of the power spectrum
    // Loop over all grid-cells and all particles and add up contribution and subtracts shot-noise term
    // Since we need to combine all particles with all cells this is not easiy parallelizable with MPI
    // so we assume all CPUs have exactly the same particles when this is run on more than 1 MPI tasks
    //==============================================================================================

    template<int N, class T>
      void compute_power_spectrum_direct_summation(
          int Ngrid, 
          T *part, 
          size_t NumPart, 
          PowerSpectrumBinning &pofk)
      {

        assert_mpi(Ngrid > 0, 
            "[direct_summation_power_spectrum] Ngrid > 0 required\n");
        if(NTasks > 1 and ThisTask == 0) 
          std::cout << "[direct_summation_power_spectrum] Warning: this method assumes all tasks have the same particles\n";

        const std::complex<double> I(0,1);
        const double norm = 1.0/double(NumPart);

        FFTWGrid<N> density_k(Ngrid, 1, 1);
        density_k.add_memory_label("FFTWGrid::compute_power_spectrum_direct_summation::density_k");

        auto *f = density_k.get_fourier_grid();
        for(auto && complex_index: density_k.get_fourier_range()) {
          auto kvec = density_k.get_fourier_wavevector_from_index(complex_index);
          double real = 0.0;
          double imag = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+:real) reduction(+:imag) 
#endif
          for(size_t i = 0; i < NumPart; i++){
            auto *x = part[i].get_pos();
            double kx = 0.0;
            for(int idim = 0; idim < N; idim++){
              kx += kvec[idim] * x[idim];
            }
            auto val = std::exp(-kx * I);
            real += val.real();
            imag += val.imag();
          }
          std::complex<double> sum = {real, imag};
          if(ThisTask == 0 and complex_index == 0)
            sum -= 1.0;
          f[complex_index] = sum * norm;
        }

        // Bin up the power-spectrum
        bin_up_power_spectrum<N>(density_k, pofk);

        // Subtract shot-noise
        for(int i = 0; i < pofk.n; i++)
          pofk.pofk[i] -= 1.0 / double(NumPart);
      }
    
    //================================================================================
    // A simple power-spectrum estimator for multipoles in simulations - nothing fancy
    // Displacing particles from realspace to redshift-space using their velocities
    // along each of the coordinate axes. Result is the mean of this
    // Deconvolving the window-function and subtracting shot-noise (1/NumPartTotal) (for monopole)
    //
    // velocity_to_distance is the factor to convert a velocity to a displacement in units
    // of the boxsize. This is c / ( aH(a) Boxsize ) for peculiar and c / (H(a)Boxsize) for comoving velocities 
    // At z = 0 velocity_to_displacement = 1.0/(100 * Boxsize) when Boxsize is in Mpc/h
    //================================================================================
    template<int N, class T>
      void compute_power_spectrum_multipoles(
          int Ngrid,
          FML::PARTICLE::MPIParticles<T> & part,
          double velocity_to_displacement,
          std::vector<PowerSpectrumBinning> &Pell,
          std::string density_assignment_method) {

        // Set how many extra slices we need for the density assignment to go smoothly
        auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
        const int nleft  = nleftright.first;
        const int nright = nleftright.second;
        
        // Initialize binning just in case
        for (int ell = 0; ell < Pell.size(); ell++)
          Pell[ell].reset();

        // Set a binning for each axes
        std::vector<std::vector<PowerSpectrumBinning>> Pell_all(N);
        for(int dir = 0; dir < N; dir++) {
          Pell_all[dir] = Pell;
        }

        // Loop over all the axes we are going to put the particles
        // into redshift space
        for(int dir = 0; dir < N; dir++) {
        
          // Set up binning for current axis
          std::vector<PowerSpectrumBinning> Pell_current = Pell_all[dir];
          for (int ell = 0; ell < Pell_current.size(); ell++)
            Pell_current[ell].reset();

          // Make line of sight direction unit vector
          std::vector<double> line_of_sight_direction(N,0.0);
          line_of_sight_direction[dir] = 1.0;

          // Displace particles XXX add OMP
          for(auto &p : part){
            auto *pos = p.get_pos();
            auto *vel = p.get_vel();
            double vdotr = 0.0;
            for(int idim = 0; idim < N; idim++) {
              vdotr += vel[idim] * line_of_sight_direction[idim];
            }
            for(int idim = 0; idim < N; idim++) {
              pos[idim] +=  vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
              // Periodic boundary conditions
              if(pos[idim] <  0.0) pos[idim] += 1.0;
              if(pos[idim] >= 1.0) pos[idim] -= 1.0;
            }
          }
          // Only displacements along the x-axis can trigger communication needs so we can avoid one call
          // if(dir == 0) part.communicate_particles();
          part.communicate_particles();

          // Bin particles to grid
          FFTWGrid<N> density_k(Ngrid, nleft, nright);
          density_k.add_memory_label("FFTWGrid::compute_power_spectrum_multipoles::density_k");
          density_k.set_grid_status_real(true);
          particles_to_grid<N,T>(
              part.get_particles_ptr(),
              part.get_npart(),
              part.get_npart_total(),
              density_k,
              density_assignment_method);
          
          // Displace particles XXX add OMP
          for(auto &p : part){
            auto *pos = p.get_pos();
            auto *vel = p.get_vel();
            double vdotr = 0.0;
            for(int idim = 0; idim < N; idim++) {
              vdotr += vel[idim] * line_of_sight_direction[idim];
            }
            for(int idim = 0; idim < N; idim++) {
              pos[idim] -= vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
              // Periodic boundary conditions
              if(pos[idim] <  0.0) pos[idim] += 1.0;
              if(pos[idim] >= 1.0) pos[idim] -= 1.0;
            }
          }
          // Only displacements along the x-axis can trigger communication needs so we can avoid one call
          // if(dir == 0) part.communicate_particles();
          part.communicate_particles();

          // Fourier transform
          density_k.fftw_r2c();
          
          // Deconvolve window function
          deconvolve_window_function_fourier<N>(density_k,  density_assignment_method);

          // Compute power-spectrum multipoles
          compute_power_spectrum_multipoles(
              density_k,
              Pell_current,
              line_of_sight_direction);
          
          // Assign back
          Pell_all[dir] = Pell_current;
        }
       
        // Normalize
        for (int ell = 0; ell < Pell.size(); ell++){
          for(int dir = 0; dir < N; dir++) {
            Pell[ell] += Pell_all[dir][ell];
          }
        }
        for (int ell = 0; ell < Pell.size(); ell++){
          for(int i = 0; i < Pell[ell].n; i++){
            Pell[ell].pofk[i] /= double(N);
            Pell[ell].count[i] /= double(N);
            Pell[ell].kbin[i] /= double(N);
          }
        }
        
        // XXX Compute variance of pofk

        // Subtract shotnoise for monopole
        for(int i = 0; i < Pell[0].n; i++){
          Pell[0].pofk[i] -= 1.0/double(part.get_npart_total());
        }
      }

    //================================================================================
    // A simple power-spectrum estimator - nothing fancy
    // Deconvolving the window-function and subtracting shot-noise (1/NumPartTotal)
    //================================================================================
    template<int N, class T>
      void compute_power_spectrum(
          int Ngrid,
          T *part,
          size_t NumPart,
          size_t NumPartTotal,
          PowerSpectrumBinning &pofk,
          std::string density_assignment_method) {

        // Set how many extra slices we need for the density assignment to go smoothly
        auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
        int nleft  = nleftright.first;
        int nright = nleftright.second;

        // Bin particles to grid
        FFTWGrid<N> density_k(Ngrid, nleft, nright);
        density_k.add_memory_label("FFTWGrid::compute_power_spectrum::density_k");
        particles_to_grid<N,T>(
            part,
            NumPart,
            NumPartTotal,
            density_k,
            density_assignment_method);

        // Fourier transform
        density_k.fftw_r2c();

        // Deconvolve window function
        deconvolve_window_function_fourier<N>(density_k,  density_assignment_method);

        // Bin up power-spectrum
        bin_up_power_spectrum<N>(density_k, pofk);

        // Subtract shotnoise
        for(int i = 0; i < pofk.n; i++){
          pofk.pofk[i] -= 1.0/double(NumPartTotal);
        }
      }

    //======================================================================
    // Computes the power-spectum by using two interlaced grids
    // to reduce the effect of aliasing (allowing us to use a smaller Ngrid)
    // Deconvolves the window function and subtracts shot-noise
    //======================================================================
    template<int N, class T>
      void compute_power_spectrum_interlacing(
          int Ngrid,
          T *part,
          size_t NumPart,
          size_t NumPartTotal,
          PowerSpectrumBinning &pofk,
          std::string density_assignment_method) {

        // Set how many extra slices we need for the density assignment to go smoothly
        auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
        int nleft  = nleftright.first;
        int nright = nleftright.second;

        // One extra slice in general as we need to shift the particle half a grid-cell
        nright += 1; 

        // Bin particles to grid
        FFTWGrid<N> density_k(Ngrid, nleft, nright);
        density_k.add_memory_label("FFTWGrid::compute_power_spectrum_interlacing::density_k");
        particles_to_grid<N,T>(
            part,
            NumPart,
            NumPartTotal,
            density_k,
            density_assignment_method);

        // Shift particles
        const double shift = 1.0/double(2*Ngrid);
#ifdef USE_OMP
#pragma omp parallel for 
#endif
        for(size_t i = 0; i < NumPart; i++){
          auto *pos = part[i].get_pos();
          pos[0] += shift;
          for(int idim = 1; idim < N; idim++){
            pos[idim] += shift;
            if(pos[idim] >= 1.0) pos[idim] -= 1.0;
          }
        }

        // Bin shifted particles to grid
        FFTWGrid<N> density_k2(Ngrid, nleft, nright);
        density_k2.add_memory_label("FFTWGrid::compute_power_spectrum_interlacing::density_k2");
        particles_to_grid<N,T>(
            part,
            NumPart,
            NumPartTotal,
            density_k2,
            density_assignment_method);

        // Shift particles back as not to ruin anything
#ifdef USE_OMP
#pragma omp parallel for 
#endif
        for(size_t i = 0; i < NumPart; i++){
          auto *pos = part[i].get_pos();
          pos[0] -= shift;
          for(int idim = 1; idim < N; idim++){
            pos[idim] -= shift;
            if(pos[idim] < 0.0) pos[idim] += 1.0;
          }
        }

        // Fourier transform
        density_k.fftw_r2c();
        density_k2.fftw_r2c();

        // The mean of the two grids
        auto ff = density_k.get_fourier_grid();
        auto gg = density_k2.get_fourier_grid();
        const std::complex<double> I(0,1);
        for(auto && complex_index: density_k.get_fourier_range()) {
          auto kvec = density_k.get_fourier_wavevector_from_index(complex_index);
          double ksum = 0.0;
          for(int idim = 0; idim < N; idim++)
            ksum += kvec[idim];
          auto norm = std::exp( I * ksum * shift);
          ff[complex_index] = (ff[complex_index] + norm * gg[complex_index])/2.0;
        }

        // Deconvolve window function
        deconvolve_window_function_fourier<N>(density_k,  density_assignment_method);

        // Bin up power-spectrum
        bin_up_power_spectrum<N>(density_k, pofk);

        // Subtract shotnoise
        for(int i = 0; i < pofk.n; i++){
          pofk.pofk[i] -= 1.0/double(NumPartTotal);
        }
      }
  }
}

#endif
