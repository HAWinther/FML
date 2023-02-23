#ifndef PGD_HEADER
#define PGD_HEADER
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>

namespace FML {
  namespace PGD {

    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    template <class T1>
    using MPIParticles = FML::PARTICLE::MPIParticles<T1>;

    //=================================================================================
    /// Applies the potential gradient decent method (1804.00671) to a set of particles
    ///
    /// @tparam N The dimension of the grid
    /// @tparam T The type of the particles
    ///
    /// @param[in] p Pointer to particles
    /// @param[in] NumPart Number of particles
    /// @param[in] k_lowpass Wavenumber in boxunits (k*Boxsize) for the lowpass filter we apply
    /// @param[in] k_highpass Wavenumber in boxunits (k*Boxsize) for the highpass filter we apply
    /// @param[in] displacement_factor The strength of the displacement (alpha in the linked paper)
    /// @param[in] norm_poisson_equation The prefactor to the Poisson equation (1.5*OmegaM0*a for cosmological sims)
    /// @param[in] Ngrid Size of the grid we do the calculations on
    /// @param[in] density_assignment_method The density assignement method we use for assigning particles to grid
    //             and doing interpolation when computing things.
    ///
    //=================================================================================
    template <int N, class T>
      void potential_gradient_decent_method(T * p,
          size_t NumPart,
          double k_lowpass,
          double k_highpass,
          double displacement_factor,
          double norm_poisson_equation, 
          int Ngrid,
          std::string density_assignment_method) {

        const std::complex<FML::GRID::FloatType> I(0, 1);

        // Sanity checks
        assert_mpi(NumPart > 0, "[potential_gradient_decent_method] NumPart is zero");
        assert_mpi(k_lowpass > 0.0, "[potential_gradient_decent_method] k_lowpass has to be nonzero");
        assert_mpi(k_highpass > 0.0, "[potential_gradient_decent_method] k_highpass has to be nonzero");

        // For interpolating the grid to the particle positions
        std::string interpolation_method = density_assignment_method;
        auto NumPartTotal = NumPart;
        FML::SumOverTasks(&NumPartTotal);

        // Bin particles to grid and fourier transform
        const auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
        FFTWGrid<N> density_k(Ngrid, nleftright.first, nleftright.second);
        const bool interlacing = false;
        FML::INTERPOLATION::particles_to_fourier_grid(p,
            NumPart,
            NumPartTotal,
            density_k,
            density_assignment_method,
            interlacing);

        // Apply high- and low-pass filter
        double kl2 = k_lowpass * k_lowpass;
        double ks2 = k_highpass * k_highpass;
        std::function<double(double)> filter = [=](double k2) -> double {
          double k_over_ks_squared = k2 / ks2 < 10.0 ? k2 / ks2 : 10.0;
          double kl_over_k_squared = k2 / kl2 < 0.01 ? 100.0 : kl2 / k2;
          double factor = std::exp(-k_over_ks_squared*k_over_ks_squared - kl_over_k_squared);
          return factor;
        };
        FML::GRID::custom_smoothing_filter_fourier_space(density_k, filter);

        // Density field -> force
        std::array<FFTWGrid<N>, N> force_real;
        FML::NBODY::compute_force_from_density_fourier<N>(density_k, 
            force_real, 
            density_assignment_method, 
            norm_poisson_equation);

        // Free memory
        density_k.free();

        // Interpolate force to particle positions
        for (int idim = 0; idim < N; idim++)
          force_real[idim].communicate_boundaries();

        std::array<std::vector<FML::GRID::FloatType>, N> force;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N, T>(force_real, 
            p, 
            NumPart, 
            force, 
            interpolation_method);

        // Free memory
        for (int idim = 0; idim < N; idim++)
          force_real[idim].free();

        // Add displacement to particles
        double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
        for (size_t i = 0; i < NumPart; i++) {
          auto * pos = FML::PARTICLE::GetPos(p[i]);
          for (int idim = 0; idim < N; idim++) {
            double disp = -force[idim][i] * displacement_factor;
            max_disp = std::max(max_disp, std::abs(disp));
            pos[idim] += disp;

            // Periodic wrap
            if(pos[idim] >= 1.0) pos[idim] -= 1.0;
            if(pos[idim] <  0.0) pos[idim] += 1.0;
          }
        }
        FML::MaxOverTasks(&max_disp);
        if (FML::ThisTask == 0)
          std::cout << "[potential_gradient_decent_method] Maximum displacement : " 
            << max_disp << " = " << max_disp * Ngrid << " grid-cells\n";
      }

    /// Fitting formula for the displacment factor needed in the PGD method
    /// Delta is the mean particle separation
    double pgd_fitting_formula_alpha(double Delta, double scale_factor){
      const double alpha0 = 0.0061*std::pow(Delta,25) + 0.0051*std::pow(Delta,3) + 0.00314;
      const double mu = -5.18*Delta*Delta*Delta + 11.57*Delta*Delta - 8.58*Delta + 0.77;
      const double alpha = alpha0 * std::pow(scale_factor, mu);
      return alpha;
    }
    /// Fitting formula for the lowpass filter scale needed in the PGD method
    /// Delta is the mean particle separation
    double pgd_fitting_formula_klowpass_hmpc(double Delta){
      const double kl = (1.52 - 0.3*Delta);
      return kl;
    }
    /// Fitting formula for the highpass filter scale needed in the PGD method
    /// Delta is the mean particle separation
    double pgd_fitting_formula_khighpass_hmpc(double Delta){
      const double ks = (33.4 - 30*Delta);
      return ks;
    }
  }
}
#endif
