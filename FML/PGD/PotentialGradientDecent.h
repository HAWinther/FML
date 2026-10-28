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
        /// Applies the potential gradient decent method (Dai, Feng & Seljak 2018, arXiv:1804.00671) to a set of
        /// particles. The particles (positions in [0,1)) are displaced by
        /// \f$ S = -\alpha\ {\rm norm}\ \nabla \hat{O}_l \hat{O}_s \nabla^{-2}\delta \f$ (in units of the boxsize),
        /// i.e. towards the minima of the filtered potential, with the filters
        /// \f$ \hat{O}_l = \exp(-k_l^2/k^2) \f$ (removes large scales) and \f$ \hat{O}_s = \exp(-k^4/k_s^4) \f$
        /// (removes small scales). In the paper (eq. 2.8) the displacement is
        /// \f$ S = (4\pi G\bar{\rho}\alpha/H_0^2) \nabla \hat{O}_l \hat{O}_s \nabla^{-2}\delta \f$ so the prefactor
        /// that goes with their (fitted, dimensionless) alpha is \f$ {\rm norm} = 4\pi G\bar{\rho}/H_0^2 = 1.5\Omega_M
        /// \f$ if
        /// \f$\bar{\rho}\f$ is the comoving mean matter density. The paper does not say whether \f$\bar{\rho}\f$ is
        /// comoving and has no factor of the scale factor a in the definition: the time dependence is in the fitted
        /// \f$ \alpha(a) \f$ (pgd_fitting_formula_alpha). The example (PGD/example) passes the N-body code-unit Poisson
        /// prefactor \f$ 1.5\Omega_M a \f$ which agrees with this only at a = 1. The sign convention of the paper is
        /// ambiguous (eq. 2.7 vs 2.14); here a positive alpha moves particles towards overdensities.
        ///
        /// NB: this moves the particles but does not communicate them: afterwards particles can be outside the domain
        /// of the current task and the caller has to call communicate_particles (or use the MPIParticles overload below
        /// which does this). All tasks have to call this method (it is collective) also if they have no particles.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The type of the particles
        ///
        /// @param[in] p Pointer to particles
        /// @param[in] NumPart Number of (local) particles (can be zero on some tasks)
        /// @param[in] k_lowpass The wavenumber \f$ k_l \f$ in boxunits (k*Boxsize) of the filter
        /// \f$ \exp(-k_l^2/k^2) \f$ that removes the large scales (pgd_fitting_formula_klowpass_hmpc times the boxsize)
        /// @param[in] k_highpass The wavenumber \f$ k_s \f$ in boxunits (k*Boxsize) of the filter
        /// \f$ \exp(-k^4/k_s^4) \f$ that removes the small scales (pgd_fitting_formula_khighpass_hmpc times the
        /// boxsize)
        /// @param[in] displacement_factor The strength of the displacement (alpha in the linked paper)
        /// @param[in] norm_poisson_equation The prefactor norm in \f$ \nabla^2\Phi = {\rm norm}\cdot\delta\f$ (see
        /// above:
        /// \f$ 4\pi G\bar{\rho}/H_0^2 = 1.5\Omega_M \f$ in the paper's definition of alpha)
        /// @param[in] Ngrid Size of the grid we do the calculations on
        /// @param[in] density_assignment_method The density assignment method we use for assigning particles to grid
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
            assert_mpi(k_lowpass > 0.0, "[potential_gradient_decent_method] k_lowpass has to be nonzero");
            assert_mpi(k_highpass > 0.0, "[potential_gradient_decent_method] k_highpass has to be nonzero");

            // For interpolating the grid to the particle positions
            std::string interpolation_method = density_assignment_method;
            auto NumPartTotal = NumPart;
            FML::SumOverTasks(&NumPartTotal);
            assert_mpi(NumPartTotal > 0, "[potential_gradient_decent_method] There are no particles (on any task)");

            // Bin particles to grid and fourier transform
            const auto nleftright =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
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
                double factor = std::exp(-k_over_ks_squared * k_over_ks_squared - kl_over_k_squared);
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

                    // Periodic wrap (to [0,1) also for displacements larger than the box)
                    FML::NBODY::periodic_wrap_unit_box(pos[idim]);
                }
            }
            FML::MaxOverTasks(&max_disp);
            if (FML::ThisTask == 0)
                std::cout << "[potential_gradient_decent_method] Maximum displacement : " << max_disp << " = "
                          << max_disp * Ngrid << " grid-cells\n";
        }

        //=================================================================================
        /// As above, but for MPIParticles and the particles are communicated afterwards (so they are on the right task)
        //=================================================================================
        template <int N, class T>
        void potential_gradient_decent_method(MPIParticles<T> & part,
                                              double k_lowpass,
                                              double k_highpass,
                                              double displacement_factor,
                                              double norm_poisson_equation,
                                              int Ngrid,
                                              std::string density_assignment_method) {
            potential_gradient_decent_method<N, T>(part.get_particles_ptr(),
                                                   part.get_npart(),
                                                   k_lowpass,
                                                   k_highpass,
                                                   displacement_factor,
                                                   norm_poisson_equation,
                                                   Ngrid,
                                                   density_assignment_method);
            part.communicate_particles();
        }

        /// Fitting formula for the displacement factor alpha needed in the PGD method (eqs. 4.3-4.5 in 1804.00671,
        /// valid for 0.3 < Delta < 1 and 0 < z < 2). Delta is the mean particle separation in Mpc/h
        inline double pgd_fitting_formula_alpha(double Delta, double scale_factor) {
            const double alpha0 = 0.0061 * std::pow(Delta, 25) + 0.0051 * std::pow(Delta, 3) + 0.00314;
            const double mu = -5.18 * Delta * Delta * Delta + 11.57 * Delta * Delta - 8.58 * Delta + 0.77;
            const double alpha = alpha0 * std::pow(scale_factor, mu);
            return alpha;
        }
        /// Fitting formula for k_l in h/Mpc (eq. 4.1 in 1804.00671): the scale of the filter exp(-k_l^2/k^2) that
        /// removes large scales (the k_lowpass argument above). Delta is the mean particle separation in Mpc/h
        inline double pgd_fitting_formula_klowpass_hmpc(double Delta) {
            const double kl = (1.52 - 0.3 * Delta);
            return kl;
        }
        /// Fitting formula for k_s in h/Mpc (eq. 4.2 in 1804.00671): the scale of the filter exp(-k^4/k_s^4) that
        /// removes small scales (the k_highpass argument above). Delta is the mean particle separation in Mpc/h
        inline double pgd_fitting_formula_khighpass_hmpc(double Delta) {
            const double ks = (33.4 - 30 * Delta);
            return ks;
        }
    } // namespace PGD
} // namespace FML
#endif
