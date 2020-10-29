#ifndef NBODY_HEADER
#define NBODY_HEADER

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/Timing/Timings.h>

namespace FML {

    /// This namespace deals with N-body simulations. Computing forces, moving particles
    /// and generating initial conditions.
    namespace NBODY {

        // Type alias
        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;
        template <class T>
        using MPIParticles = FML::PARTICLE::MPIParticles<T>;

        template <int N, class T>
        void DriftParticles(FML::PARTICLE::MPIParticles<T> & part, double delta_time, bool periodic_box = true);

        template <int N, class T>
        void DriftParticles(T * p, size_t NumPart, double delta_time, bool periodic_box = true);

        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           MPIParticles<T> & part,
                           double delta_time,
                           std::string interpolation_method);

        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           T * p,
                           size_t NumPart,
                           double delta_time,
                           std::string interpolation_method);

        template <int N>
        void compute_force_from_density_real(const FFTWGrid<N> & density_grid_real,
                                             std::array<FFTWGrid<N>, N> & force_real,
                                             std::string density_assignment_method_used,
                                             double norm_poisson_equation);

        template <int N>
        void compute_force_from_density_fourier(const FFTWGrid<N> & density_grid_fourier,
                                                std::array<FFTWGrid<N>, N> & force_real,
                                                std::string density_assignment_method_used,
                                                double norm_poisson_equation);

        //===================================================================================
        /// @brief Take a N-body step with a simple Kick-Drift-Kick method (this
        /// method serves mainly as an example for how one can do this).
        /// 1. Particles to grid to get \f$ \delta \f$
        /// 2. Compute the Newtonian potential via \f$ \nabla^2 \Phi = {\rm norm} \cdot \delta \f$
        /// 3. Compute the force  \f$ F = \nabla \Phi \f$
        /// 4. Move the particles using \f$ x \to x + v \Delta t \f$ and \f$ v \to v + F \Delta t \f$
        /// This method assumes that the velocities are in units of (boxsize / time-step-unit), in other words that \f$
        /// v\Delta t\f$ gives rise to a shift in [0,1). For cosmological N-body norm_poisson_equation depends on a and
        /// it should be set at the correct time. If one does simple sims with fixed time-step then the last kick of the
        /// previous step can be combined with the first kick of the current step to save one force evaluation per step
        /// (so basically two times as fast).
        ///
        /// @tparam N The dimension of the grid.
        /// @tparam T The particle class.
        ///
        /// @param[in] Nmesh The gridsize to use for computing the density and force.
        /// @param[out] part The particles
        /// @param[in] delta_time The time \f$ \Delta t \f$ we move forward.
        /// @param[in] density_assignment_method The density assignement method (NGP, CIC, TSC, PCS or PQS).
        /// @param[in] norm_poisson_equation A possible prefactor to the Poisson equation
        ///
        //===================================================================================
        template <int N, class T>
        void KickDriftKickNBodyStep(int Nmesh,
                                    MPIParticles<T> & part,
                                    double delta_time,
                                    std::string density_assignment_method,
                                    double norm_poisson_equation) {

            const bool periodic_box = true;

            // Particles -> density field
            auto nleftright =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
            FFTWGrid<N> density_grid_real(Nmesh, nleftright.first, nleftright.second);
            density_grid_real.add_memory_label("FFTWGrid::KickDriftKickNBodyStep::density_grid_real");
            density_grid_real.set_grid_status_real(true);
            FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles_ptr(),
                                                        part.get_npart(),
                                                        part.get_npart_total(),
                                                        density_grid_real,
                                                        density_assignment_method);

            // Density field -> force
            std::array<FFTWGrid<N>, N> force_real;
            compute_force_from_density_real(
                density_grid_real, force_real, density_assignment_method, norm_poisson_equation);

            // Update velocity of particles
            KickParticles(force_real, part, delta_time * 0.5, density_assignment_method);

            // Move particles (this does communication)
            DriftParticles<N, T>(part, delta_time, periodic_box);

            // Particles -> density field
            FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles_ptr(),
                                                        part.get_npart(),
                                                        part.get_npart_total(),
                                                        density_grid_real,
                                                        density_assignment_method);

            // Density field -> force
            compute_force_from_density_real(
                density_grid_real, force_real, density_assignment_method, norm_poisson_equation);

            // Update velocity of particles
            KickParticles(force_real, part, delta_time * 0.5, density_assignment_method);
        }

        //===================================================================================
        /// @brief Take a N-body step with a 4th order symplectic Yoshida method. This method is mainly an illustration,
        /// for using this with cosmology we should take into account that norm_poisson_equation is a function of time
        ///
        /// @tparam N The dimension of the grid.
        /// @tparam T The particle class.
        ///
        /// @param[in] Nmesh The gridsize to use for computing the density and force.
        /// @param[out] part The particles
        /// @param[in] delta_time The time \f$ \Delta t \f$ we move forward.
        /// @param[in] density_assignment_method The density assignement method (NGP, CIC, TSC, PCS or PQS).
        /// @param[in] norm_poisson_equation A possible prefactor to the Poisson equation
        ///
        //===================================================================================
        template <int N, class T>
        void YoshidaNBodyStep(int Nmesh,
                              MPIParticles<T> & part,
                              double delta_time,
                              std::string density_assignment_method,
                              double norm_poisson_equation) {

            const bool periodic_box = true;

            // The Yoshida coefficients
            const double w1 = 1.0 / (2 - std::pow(2.0, 1.0 / 3.0));
            const double w0 = 1.0 - 2.0 * w1;
            const double c1 = w1 / 2.0, c4 = c1;
            const double c2 = (w0 + w1) / 2.0, c3 = c2;
            const double d1 = w1, d3 = d1;
            const double d2 = w0;

            // They must sum to unity
            assert(std::fabs(c1 + c2 + c3 + c4 - 1.0) < 1e-10);
            assert(std::fabs(d1 + d2 + d3 - 1.0) < 1e-10);

            // Set up a density grid to use
            auto nleftright =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
            FFTWGrid<N> density_grid_real(Nmesh, nleftright.first, nleftright.second);
            density_grid_real.add_memory_label("FFTWGrid::YoshidaNBodyStep::density_grid_real");
            density_grid_real.set_grid_status_real(true);

            // Perform one step: delta_time_pos is the advance for pos positions and delta_time_vel is for velocity
            auto one_step = [&](double delta_time_pos, double delta_time_vel, double norm_poisson) {
                // Move particles (this does communication)
                DriftParticles<N, T>(part, delta_time_pos, periodic_box);

                // Particles -> density field
                FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles_ptr(),
                                                            part.get_npart(),
                                                            part.get_npart_total(),
                                                            density_grid_real,
                                                            density_assignment_method);
                // Density field -> force
                std::array<FFTWGrid<N>, N> force_real;
                compute_force_from_density_real(density_grid_real, force_real, density_assignment_method, norm_poisson);

                // Update velocity of particles
                KickParticles(force_real, part, delta_time_vel, density_assignment_method);
            };

            // The norm_poisson_equation in a cosmo sim depends on [aexp] so this should be changed
            one_step(delta_time * c1, delta_time * d1, norm_poisson_equation);
            one_step(delta_time * c2, delta_time * d2, norm_poisson_equation);
            one_step(delta_time * c3, delta_time * d3, norm_poisson_equation);

            // Move particles (this does communication)
            DriftParticles<N, T>(part, delta_time * c4, periodic_box);
        }

        //===================================================================================
        /// Take a density grid in real space and returns the force \f$ \nabla \phi \f$  where
        /// \f$ \nabla^2 \phi = {\rm norm} \cdot \delta \f$
        /// Different choices for what kernel to use for \f$ \nabla / \nabla^2\f$ are availiable, see the function body
        /// (is set too be a compile time option). Fiducial choice is the continuous greens function \f$ 1/k^2\f$, but
        /// we can also choose to also devonvolve the window and discrete kernels (Hamming 1989; same as used in GADGET)
        /// and Hockney & Eastwood 1988. See e.g. 1603.00476 for a list.
        ///
        /// @tparam N The dimension of the grid
        ///
        /// @param[in] density_grid_real The density contrast in real space.
        /// @param[out] force_real The force in real space.
        /// @param[in] density_assignment_method_used The density assignement we used to compute the density field.
        /// Needed only in case kernel_choice (defined in the body of this function) is not CONTINUOUS_GREENS_FUNCTION.
        /// @param[in] norm_poisson_equation The prefactor (norm) to the Poisson equation.
        ///
        //===================================================================================
        template <int N>
        void compute_force_from_density_real(const FFTWGrid<N> & density_grid_real,
                                             std::array<FFTWGrid<N>, N> & force_real,
                                             std::string density_assignment_method_used,
                                             double norm_poisson_equation) {

            FFTWGrid<N> density_grid_fourier = density_grid_real;
            density_grid_fourier.add_memory_label("FFTWGrid::compute_force_from_density_real::density_grid_fourier");
            density_grid_fourier.set_grid_status_real(true);
            density_grid_fourier.fftw_r2c();
            compute_force_from_density_fourier(
                density_grid_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
        }

        //===================================================================================
        /// Take a density grid in fourier space and returns the force \f$ \nabla \phi \f$  where
        /// \f$ \nabla^2 \phi = {\rm norm} \cdot \delta \f$
        /// Different choices for what kernel to use for \f$ \nabla / \nabla^2\f$ are availiable, see the function body
        /// (is set too be a compile time option). Fiducial choice is the continuous greens function \f$ 1/k^2\f$, but
        /// we can also choose to also devonvolve the window and discrete kernels (Hamming 1989; same as used in GADGET)
        /// and Hockney & Eastwood 1988. See e.g. 1603.00476 for a list and references.
        ///
        /// @tparam N The dimension of the grid
        ///
        /// @param[in] density_grid_fourier The density contrast in fourier space.
        /// @param[out] force_real The force in real space.
        /// @param[in] density_assignment_method_used The density assignement we used to compute the density field.
        /// Needed only in case kernel_choice (defined in the body of this function) is not CONTINUOUS_GREENS_FUNCTION.
        /// @param[in] norm_poisson_equation The prefactor (norm) to the Poisson equation.
        ///
        //===================================================================================
        template <int N>
        void compute_force_from_density_fourier(const FFTWGrid<N> & density_grid_fourier,
                                                std::array<FFTWGrid<N>, N> & force_real,
                                                std::string density_assignment_method_used,
                                                double norm_poisson_equation) {

            // What fourier space kernel to use for D/D^2
            enum KernelChoices {
                // 1/k^2
                CONTINUOUS_GREENS_FUNCTION,
                // Divide by square of density assignment window function 1/k^2W^2
                CONTINUOUS_GREENS_FUNCTION_DECONVOLVE,
                // Hockney & Eastwood 1988: 1 / [ 4/dx^2 * Sum sin(ki * dx / 2)^2 ] with dx = 1/Ngrid
                DISCRETE_GREENS_FUNCTION_HOCKNEYEASTWOOD,
                // Hockney & Eastwood 1988: 1 / [ 4/dx^2 * Sum sin(ki * dx / 2)^2 ] / W^2 with dx = 1/Ngrid
                DISCRETE_GREENS_FUNCTION_HOCKNEYEASTWOOD_DECONVOLVE,
                // Hamming: D = D/k^2 where D = (8 sin(k) - sin(2k))/6
                DISCRETE_GREENS_FUNCTION_HAMMING,
                // Hamming: D/k^2W^2 where D = (8 sin(k) - sin(2k))/6 (GADGET2 kernel)
                DISCRETE_GREENS_FUNCTION_HAMMING_DECONVOLVE
            };
            constexpr int kernel_choice = CONTINUOUS_GREENS_FUNCTION;

            auto Nmesh = density_grid_fourier.get_nmesh();
            auto Local_nx = density_grid_fourier.get_local_nx();
            auto Local_x_start = density_grid_fourier.get_local_x_start();

            // This is needed in case kernel_choice != CONTINUOUS_GREENS_FUNCTION
            // The order of the density assignment method
            const int order = FML::INTERPOLATION::interpolation_order_from_name(density_assignment_method_used);
            // Window function for density assignment
            const double knyquist = M_PI * Nmesh;
            [[maybe_unused]] auto window_function = [&](std::array<double, N> & kvec) -> double {
                double w = 1.0;
                for (int idim = 0; idim < N; idim++) {
                    const double koverkny = M_PI / 2. * (kvec[idim] / knyquist);
                    w *= koverkny == 0.0 ? 1.0 : std::sin(koverkny) / (koverkny);
                }
                // res = pow(w,p);
                double res = 1;
                for (int i = 0; i < order; i++)
                    res *= w;
                return res;
            };

            // Copy over
            for (int idim = 0; idim < N; idim++) {
                force_real[idim] = density_grid_fourier;
                force_real[idim].add_memory_label("FFTWGrid::compute_force_from_density_fourier::force_real_" +
                                                  std::to_string(idim));
                force_real[idim].set_grid_status_real(idim == 0);
            }

            // Loop over all local fourier grid cells
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                std::complex<FML::GRID::FloatType> I(0, 1);
                for (auto && fourier_index : force_real[0].get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode (k=0)

                    force_real[0].get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    auto value = force_real[0].get_fourier_from_index(fourier_index);

                    // Divide by k^2 (different kernel choices here, fiducial is just 1/k^2)
                    if constexpr (kernel_choice == CONTINUOUS_GREENS_FUNCTION) {
                        value /= kmag2;
                    } else if constexpr (kernel_choice == CONTINUOUS_GREENS_FUNCTION_DECONVOLVE) {
                        double W = window_function(kvec);
                        value /= (kmag2 * W * W);
                    } else if constexpr (kernel_choice == DISCRETE_GREENS_FUNCTION_HOCKNEYEASTWOOD) {
                        double sum = 0.0;
                        for (int idim = 0; idim < N; idim++) {
                            double s = std::sin(kvec[idim] / (2.0 * double(Nmesh)));
                            sum += s * s;
                        }
                        sum *= 4.0 * double(Nmesh * Nmesh);
                        value /= sum;
                    } else if constexpr (kernel_choice == DISCRETE_GREENS_FUNCTION_HOCKNEYEASTWOOD_DECONVOLVE) {
                        double W = window_function(kvec);
                        double sum = 0.0;
                        for (int idim = 0; idim < N; idim++) {
                            double s = std::sin(kvec[idim] / (2.0 * double(Nmesh)));
                            sum += s * s;
                        }
                        sum *= 4.0 * double(Nmesh * Nmesh) * W * W;
                        value /= sum;
                    } else if constexpr (kernel_choice == DISCRETE_GREENS_FUNCTION_HAMMING) {
                        value *= 1.0 / kmag2;
                    } else if constexpr (kernel_choice == DISCRETE_GREENS_FUNCTION_HAMMING_DECONVOLVE) {
                        double W = window_function(kvec);
                        value *= 1.0 / (kmag2 * W * W);
                    } else {
                        FML::assert_mpi(
                            false,
                            "Unknown kernel_choice in compute_force_from_density_fourier. Method set at the "
                            "head of this function");
                    }

                    // Modify F[D] = kvec -> (8*sin(ki dx) - sin(2 ki dx))/6dx
                    if constexpr (kernel_choice == DISCRETE_GREENS_FUNCTION_HAMMING or
                                  kernel_choice == DISCRETE_GREENS_FUNCTION_HAMMING_DECONVOLVE) {
                        for (int idim = 0; idim < N; idim++) {
                            kvec[idim] = (8.0 * std::sin(kvec[idim] / double(Nmesh)) - std::sin(2 * double(Nmesh))) /
                                         6.0 * double(Nmesh);
                        }
                    }

                    // Compute force -ik/k^2 delta(k)
                    for (int idim = 0; idim < N; idim++) {
                        force_real[idim].set_fourier_from_index(fourier_index,
                                                                -I * value * FML::GRID::FloatType(kvec[idim] * norm_poisson_equation));
                    }
                }
            }

            // Deal with DC mode
            if (Local_x_start == 0)
                for (int idim = 0; idim < N; idim++)
                    force_real[idim].set_fourier_from_index(0, 0.0);

            // Fourier transform back to real space
            for (int idim = 0; idim < N; idim++)
                force_real[idim].fftw_c2r();
        }

        //===================================================================================
        /// This moves the particles according to \f$ x_{\rm new} = x + v \Delta t \f$. Note that we assume the
        /// velocities are in such units that \f$ v \Delta t\f$ is a dimensionless shift in [0,1).
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        ///
        /// @param[out] part MPIParticles containing the particles.
        /// @param[in] delta_time The size of the timestep.
        /// @param[in] periodic_box Is the box periodic?
        ///
        //===================================================================================
        template <int N, class T>
        void DriftParticles(MPIParticles<T> & part, double delta_time, bool periodic_box) {
            if (part.get_npart() == 0)
                return;
            if (delta_time == 0.0)
                return;

            // Sanity check on particle
            T tmp{};
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N,
                       "[DriftParticles] NDIM of particles and of grid does not match");

            DriftParticles<N, T>(part.get_particles_ptr(), part.get_npart(), delta_time, periodic_box);

            // Particles might have left the current task
            part.communicate_particles();
        }

        //===================================================================================
        /// This moves the particles according to \f$ x_{\rm new} = x + v \Delta t \f$. Note that we assume the
        /// velocities are in such units that \f$ v \Delta t\f$ is a dimensionless shift in [0,1). NB: after this
        /// methods is done the particles might have left the current task and must be communicated (this is done
        /// automatically if you use the MPIParticles version of this method).
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        ///
        /// @param[out] p Pointer to the first particle.
        /// @param[in] NumPart The number of local particles.
        /// @param[in] delta_time The size of the timestep.
        /// @param[in] periodic_box Is the box periodic?
        ///
        //===================================================================================
        template <int N, class T>
        void DriftParticles(T * p, size_t NumPart, double delta_time, bool periodic_box) {
            if (NumPart == 0)
                return;
            if (delta_time == 0.0)
                return;

            // Sanity check on particle
            T tmp{};
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N,
                       "[DriftParticles] NDIM of particles and of grid does not match");
            static_assert(FML::PARTICLE::has_get_pos<T>(),
                          "[DriftParticles] Particle class must have a get_pos method to use this method");
            static_assert(FML::PARTICLE::has_get_vel<T>(),
                          "[DriftParticles] Particle class must have a get_vel method to use this method");

            double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * pos = FML::PARTICLE::GetPos(p[i]);
                auto * vel = FML::PARTICLE::GetVel(p[i]);
                for (int idim = 0; idim < N; idim++) {
                    double disp = vel[idim] * delta_time;
                    pos[idim] += disp;
                    max_disp = std::max(max_disp, std::abs(disp));

                    // Periodic wrap
                    if (periodic_box) {
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;
                    }
                }
            }
            FML::MaxOverTasks(&max_disp);

            if (FML::ThisTask == 0)
                std::cout << "[Drift] Max displacement: " << max_disp << "\n";
        }

        //===================================================================================
        /// This moves the particle velocities according to \f$ v_{\rm new} = v + F \Delta t \f$. This method
        /// assumes the force is normalized such that \f$ F \Delta t \f$ has the same units as your v.
        /// If the flag free_force_grids is set in the source then we free up memory of the force grids after we have
        /// used them. The defalt is false.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        ///
        /// @param[in] force_grid Grid containing the force.
        /// @param[out] part MPIParticles containing the particles.
        /// @param[in] delta_time The size of the timestep.
        /// @param[in] interpolation_method The interpolation method for interpolating the force to the particle
        /// positions.
        ///
        //===================================================================================
        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           MPIParticles<T> & part,
                           double delta_time,
                           std::string interpolation_method) {
            if (delta_time == 0.0)
                return;
            KickParticles<N, T>(
                force_grid, part.get_particles_ptr(), part.get_npart(), delta_time, interpolation_method);
        }

        //===================================================================================
        /// This moves the particle velocities according to \f$ v_{\rm new} = v + F \Delta t \f$. This method
        /// assumes the force is normalized such that \f$ F \Delta t \f$ has the same units as your v.
        /// If the flag free_force_grids is set in the source then we free up memory of the force grids after we have
        /// used them. The defalt is false.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        ///
        /// @param[in] force_grid The force \f$ \nabla \Phi \f$.
        /// @param[out] p Pointer to the first particle.
        /// @param[in] NumPart The number of local particles.
        /// @param[in] delta_time The size of the timestep.
        /// @param[in] interpolation_method The interpolation method for interpolating the force to the particle
        /// positions.
        ///
        //===================================================================================
        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           T * p,
                           size_t NumPart,
                           double delta_time,
                           std::string interpolation_method) {

            // Nothing to do if delta_time = 0.0
            if (delta_time == 0.0)
                return;

            // Sanity check on particle
            T tmp{};
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N, "[KickParticles] Dimension of particle and grid do not match");
            static_assert(FML::PARTICLE::has_get_vel<T>(),
                          "[KickParticles] Particle must have velocity to use this method");

            // Deallocate the force grids (after interpolating to the particles we don't need it here and probably
            // not elsewhere so lets save some memory)
            constexpr bool free_force_grids = false;

            // Interpolate force to particle positions
            for (int idim = 0; idim < N; idim++) {
                force_grid[idim].communicate_boundaries();
            }
            std::array<std::vector<FML::GRID::FloatType>, N> force;
            FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N, T>(
                force_grid, p, NumPart, force, interpolation_method);
            if (free_force_grids) {
                for (int idim = 0; idim < N; idim++) {
                    force_grid[idim].free();
                }
            }

            double max_dvel = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_dvel)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * vel = FML::PARTICLE::GetVel(p[i]);
                for (int idim = 0; idim < N; idim++) {
                    double dvel = -force[idim][i] * delta_time;
                    max_dvel = std::max(max_dvel, std::abs(dvel));
                    vel[idim] += dvel;
                }
            }

            FML::MaxOverTasks(&max_dvel);

            if (FML::ThisTask == 0)
                std::cout << "[Kick] Max delta_vel * delta_time : " << max_dvel * delta_time << "\n";
        }

        template <int N, class T>
        void NBodyInitialConditions(MPIParticles<T> & part,
                                    int Npart_1D,
                                    double buffer_factor,

                                    const FFTWGrid<N> & delta_fourier,
                                    std::vector<FFTWGrid<N>> & phi_nLPT_potentials,
                                    int LPT_order,

                                    double box,
                                    double zini,
                                    std::vector<double> velocity_norms);

        //=====================================================================
        /// Generate particles from a given power-spectrum using Lagrangian perturbation theory.
        /// We generate particles in [0,1) and velocities are given by \f$ v_{\rm code} = \frac{a^2 \frac{dx}{dt}}{H_0
        /// B} \f$
        ///
        /// @tparam N The dimension we are working in.
        /// @tparam T The particle class. Must have methods get_pos, get_vel, get_D_1LPT and get_D_2LPT. But only
        /// get_pos data is required to exist. Return a nullptr if the data does not exist in the particle.
        ///
        /// @param[out] part Particle container for particles we are to create.
        /// @param[in] Npart_1D Number of particles per dimension (i.e. total is \f$ {\rm Npart}_{\rm 1D}^N \f$)
        /// @param[in] buffer_factor How many more particles to allocate?
        /// @param[in] Nmesh The grid to generate the IC on
        /// @param[in] fix_amplitude Amplitude fixed? Only random phases if true.
        /// @param[in] rng Random number generator
        /// @param[in] Pofk_of_kBox_over_Pofk_primordal The ratio of the power-spectrum (for delta) at the time you
        /// want the density field to be created at to the primordial one (the function above).
        /// @param[in] Pofk_of_kBox_over_volume_primordial The dimensionless function \f$ P/V\f$ where \f$ V = B^N\f$
        /// is the box volume as function of the dimensionless wavenumber \f$ kB \f$ where \f$ B \f$ is the boxsize and
        /// \f$ P(k) \f$ is the primordial power-spectrum for \f$\Phi\f$.
        /// @param[in] LPT_order The LPT order (1 or 2)
        /// @param[in] type_of_random_field What random field: gaussian, local, equilateral, orthogonal
        /// @param[in] fNL If non-gaussianity the value of fNL
        /// @param[in] box The boxsize (only for prining maximum displacement)
        /// @param[in] zini The initial redshift
        /// @param[in] velocity_norms A vector of the factors we need to multiply the nLPT displacement fields by to get
        /// velocities. E.g. \f$ 100 {\rm Box_in_Mpch} f_i(z_{\rm ini}) H(z_{\rm ini})/H_0 \cdot a_{\rm ini} \f$ to
        /// get peculiar velocities in km/s and \f$ f_i(z_{\rm ini}) H(z_{\rm ini})/H_0 \cdot a_{\rm ini}^2 \f$ to get
        /// the velocities we use as the fiducial choice in N-body. The order is: 1LPT, 2LPT, 3LPTa, 3LPTb
        ///
        //=====================================================================
        template <int N, class T>
        void NBodyInitialConditions(MPIParticles<T> & part,
                                    int Npart_1D,
                                    double buffer_factor,

                                    int Nmesh,
                                    bool fix_amplitude,
                                    FML::RANDOM::RandomGenerator * rng,
                                    std::function<double(double)> Pofk_of_kBox_over_Pofk_primordal,
                                    std::function<double(double)> Pofk_of_kBox_over_volume_primordial,
                                    int LPT_order,
                                    std::string type_of_random_field,
                                    double fNL,

                                    double box,
                                    double zini,
                                    std::vector<double> velocity_norms) {

            // Some sanity checks
            assert_mpi(Npart_1D > 0 and Nmesh > 0 and zini >= 0.0 and rng != nullptr and box > 0.0,
                       "[NBodyInitialConditions] Invalid parameters");

            // Generate the random field first
            auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment("CIC");
            FFTWGrid<N> delta_fourier(Nmesh, nextra.first, nextra.second);
            delta_fourier.add_memory_label("FFTWGrid::NBodyInitialConditions::delta_fourier");
            delta_fourier.set_grid_status_real(false);

            // Make a gaussian or non-local non-gaussian random field in fourier space
            if (type_of_random_field == "gaussian") {
                auto Pofk_of_kBox_over_volume = [&](double kBox) {
                    return Pofk_of_kBox_over_Pofk_primordal(kBox) * Pofk_of_kBox_over_volume_primordial(kBox);
                };
                FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(
                    delta_fourier, rng, Pofk_of_kBox_over_volume, fix_amplitude);
            } else {
                FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier_cosmology(
                    delta_fourier,
                    rng,
                    Pofk_of_kBox_over_Pofk_primordal,
                    Pofk_of_kBox_over_volume_primordial,
                    fix_amplitude,
                    fNL,
                    type_of_random_field);
            }

            // Generate IC from a given fourier grid
            std::vector<FFTWGrid<N>> phi_nLPT_potentials;
            NBodyInitialConditions<N, T>(part,
                                         Npart_1D,
                                         buffer_factor,
                                         delta_fourier,
                                         phi_nLPT_potentials,
                                         LPT_order,
                                         box,
                                         zini,
                                         velocity_norms);
        }

        //=====================================================================
        /// Generate particles from a given initial density field using Lagrangian perturbation theory.
        /// We generate particles in [0,1) and velocities are given by \f$ v_{\rm code} = \frac{a^2 \frac{dx}{dt}}{H_0
        /// B} \f$
        ///
        /// @tparam N The dimension we are working in.
        /// @tparam T The particle class. Must have methods get_pos, get_vel, get_D_1LPT and get_D_2LPT. But only
        /// get_pos data is required to exist. Return a nullptr if the data does not exist in the particle.
        ///
        /// @param[out] part Particle container for particles we are to create.
        /// @param[in] Npart_1D Number of particles per dimension (i.e. total is \f$ {\rm Npart}_{\rm 1D}^N \f$)
        /// @param[in] buffer_factor How many more particles to allocate?
        /// @param[in] delta_fourier The initial density field \f$ \delta(k,z_{\rm ini})\f$ in fourier space
        /// @param[in] phi_nLPT_potentials Return the LPT potentials: 2LPT, 3LPTa, 3LPTb, ... If the vector has zero
        /// size then nothing will be returned.
        /// @param[in] LPT_order The LPT order (1 or 2)
        /// @param[in] box The boxsize (only for prining maximum displacement)
        /// @param[in] zini The initial redshift
        /// @param[in] velocity_norms A vector of the factors we need to multiply the nLPT displacement fields by to get
        /// velocities. E.g. \f$ 100 {\rm Box_in_Mpch} f_i(z_{\rm ini}) H(z_{\rm ini})/H_0 \cdot a_{\rm ini} \f$ to
        /// get peculiar velocities in km/s and \f$ f_i(z_{\rm ini}) H(z_{\rm ini})/H_0 \cdot a_{\rm ini}^2 \f$ to get
        /// the velocities we use as the fiducial choice in N-body.
        ///
        //=====================================================================
        template <int N, class T>
        void NBodyInitialConditions(MPIParticles<T> & part,
                                    int Npart_1D,
                                    double buffer_factor,

                                    const FFTWGrid<N> & delta_fourier,
                                    std::vector<FFTWGrid<N>> & phi_nLPT_potentials,
                                    int LPT_order,

                                    double box,
                                    double zini,
                                    std::vector<double> velocity_norms) {

            T tmp{};
            if (FML::ThisTask == 0) {
                std::cout << "\n";
                std::cout << "#=====================================================\n";
                std::cout << "#\n";
                std::cout << "#                .___  _________   \n";
                std::cout << "#                |   | \\_   ___ \\  \n";
                std::cout << "#                |   | /    \\  \\/  \n";
                std::cout << "#                |   | \\     \\____ \n";
                std::cout << "#                |___|  \\______  / \n";
                std::cout << "#                              \\/  \n";
                std::cout << "#\n";
                std::cout << "# Generating initial conditions for N-body\n";
                std::cout << "# Order in LPT = " << LPT_order << "\n";
                std::cout << "# The boxsize is " << box << " comoving Mpc/h\n";
                std::cout << "# The initial redshift zini = " << zini << "\n";
                if (FML::PARTICLE::has_get_pos<T>())
                    std::cout << "# Particle has [Position] x_code = x / Box ("
                              << sizeof(FML::PARTICLE::GetVel(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_vel<T>())
                    std::cout << "# Particle has [Velocity] v_code = a^2 dxdt / (H0 Box) ("
                              << sizeof(FML::PARTICLE::GetPos(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_set_mass<T>())
                    std::cout << "# Particle has [Mass] (" << sizeof(FML::PARTICLE::GetMass(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_set_id<T>())
                    std::cout << "# Particle has [ID] (" << sizeof(FML::PARTICLE::GetID(tmp)) << " bytes)\n";
                if (FML::PARTICLE::has_get_D_1LPT<T>())
                    std::cout << "# Particle has [1LPT Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_1LPT(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_2LPT<T>())
                    std::cout << "# Particle has [2LPT Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_2LPT(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_3LPTa<T>())
                    std::cout << "# Particle has [3LPTa Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_3LPTa(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_D_3LPTb<T>())
                    std::cout << "# Particle has [3LPTb Displacement field] ("
                              << sizeof(FML::PARTICLE::GetD_3LPTb(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_q<T>())
                    std::cout << "# Particle has [Lagrangian position] ("
                              << sizeof(FML::PARTICLE::GetLagrangianPos(tmp)[0]) * N << " bytes)\n";
                std::cout << "# Total size of particle is " << FML::PARTICLE::GetSize(tmp) << " bytes\n";
                std::cout << "# We will make " << Npart_1D << "^" << N << " particles\n";
                std::cout << "# Plus a buffer with room for " << (buffer_factor - 1.0) * 100.0 << "%% more particles\n";
                std::cout << "# We will allocate ~"
                          << buffer_factor * double(FML::PARTICLE::GetSize(tmp)) * double(FML::power(Npart_1D, N)) /
                                 1e6 / double(FML::NTasks)
                          << " MB per task for the particles\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }

            // Sanity checks
            const auto Nmesh = delta_fourier.get_nmesh();
            assert_mpi(Nmesh > 0, "[NBodyInitialConditions] delta_fourier has to be already allocated");
            assert_mpi(LPT_order == 1 or LPT_order == 2 or LPT_order == 3,
                       "[NBodyInitialConditions] Only 1LPT, 2LPT and 3LPT implemented so valid choices here are "
                       "LPT_order = 1, 2 or 3");
            const std::string interpolation_method = "CIC"; // We use n-linear interpolation below
            const auto nextra_cic =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
            assert_mpi(delta_fourier.get_n_extra_slices_left() >= nextra_cic.first and
                           delta_fourier.get_n_extra_slices_right() >= nextra_cic.second,
                       "[NBodyInitialConditions] We use CIC interpolation in this routine so the grid needs to have "
                       "atleast one extra slice on the right");

            // Sanity check on particle
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N,
                       "[NBodyInitialConditions] NDIM of particles and of grid does not match");
            assert_mpi(FML::PARTICLE::has_get_pos<T>(),
                       "[NBodyInitialConditions] Particle class must have a get_pos method");

            // The scalefactor and log(a) at the initial time
            const double aini = 1.0 / (1.0 + zini);

            FFTWGrid<N> phi_1LPT;
            FFTWGrid<N> phi_2LPT;
            FFTWGrid<N> phi_3LPTa;
            FFTWGrid<N> phi_3LPTb;
            if (LPT_order == 1) {
                // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta_fourier, phi_1LPT);
            } else if (LPT_order == 2) {
                // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta_fourier, phi_1LPT);
                // Generate the 2LPT potential phi_2LPT = -1/2k^2 F[phi_ii phi_jj - phi_ij^2]
                FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_fourier, phi_2LPT);
            } else if (LPT_order == 3) {
                // Generate the 3LPT potentials phi_3LPTa, phi_3LPTb plus 3LPT curl term
                // We ignore the curl term in this implementation for simplicity
                const bool ignore_3LPT_curl_term = true;
                std::array<FFTWGrid<N>, N> phi_3LPT_Avec_fourier;
                FML::COSMOLOGY::LPT::compute_3LPT_potential_fourier<N>(delta_fourier,
                                                                       phi_1LPT,
                                                                       phi_2LPT,
                                                                       phi_3LPTa,
                                                                       phi_3LPTb,
                                                                       phi_3LPT_Avec_fourier,
                                                                       ignore_3LPT_curl_term);
            }

            //================================================================
            // Function to compute the displacement from a LPT potential
            // Frees the memory of phi_nLPT after its used
            //================================================================
            auto comp_displacement = [&]([[maybe_unused]] int nLPT,
                                         FFTWGrid<N> & phi_nLPT,
                                         std::array<std::vector<FML::GRID::FloatType>, N> & displacements_nLPT) {
                // Generate Psi from phi
                std::array<FFTWGrid<N>, N> Psi_nLPT_vector;
                FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<N>(phi_nLPT, Psi_nLPT_vector);
                for (int idim = 0; idim < N; idim++) {
                    Psi_nLPT_vector[idim].communicate_boundaries();
                }
                phi_nLPT.free();

                // Interpolate it to particle Lagrangian positions
                FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N, T>(Psi_nLPT_vector,
                                                                                        part.get_particles_ptr(),
                                                                                        part.get_npart(),
                                                                                        displacements_nLPT,
                                                                                        interpolation_method);
            };

            auto add_displacement = [&]([[maybe_unused]] int nLPT,
                                        char type,
                                        std::array<std::vector<FML::GRID::FloatType>, N> & displacements_nLPT,
                                        double vfac_nLPT) -> void {
                // Generate Psi from phi
                // Add displacement to particle position
                double max_disp_nLPT = 0.0;
                double max_vel_nLPT = 0.0;
                auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp_nLPT, max_vel_nLPT)
#endif
                for (size_t ind = 0; ind < part.get_npart(); ind++) {

                    // Fetch displacement
                    std::array<double, N> disp;
                    for (int idim = 0; idim < N; idim++) {
                        disp[idim] = displacements_nLPT[idim][ind];
                    }

                    // Add to position (particle must have position)
                    auto * pos = FML::PARTICLE::GetPos(part_ptr[ind]);
                    if (ind == 0 and FML::ThisTask == 0)
                        std::cout << "Adding " << std::to_string(nLPT) << "LPT position to particle\n";
                    for (int idim = 0; idim < N; idim++) {
                        const double dpos_nLPT = disp[idim];
                        pos[idim] += dpos_nLPT;

                        // Periodic BC
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;

                        // Compute maximum displacement
                        if (std::fabs(dpos_nLPT) > max_disp_nLPT)
                            max_disp_nLPT = std::fabs(dpos_nLPT);
                    }

                    // Add to velocity (if it exists)
                    if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                        if (ind == 0 and FML::ThisTask == 0)
                            std::cout << "Adding " << std::to_string(nLPT) << "LPT velocity to particle\n";
                        auto * vel = FML::PARTICLE::GetVel(part_ptr[ind]);
                        for (int idim = 0; idim < N; idim++) {
                            vel[idim] += vfac_nLPT * disp[idim];

                            if (std::fabs(vfac_nLPT * disp[idim]) > max_vel_nLPT)
                                max_vel_nLPT = std::fabs(vfac_nLPT * disp[idim]);
                        }
                    }

                    // Store displacement fields at particle (if it exists)
                    // This is needed if we want to do COLA
                    if (nLPT == 1) {
                        if constexpr (FML::PARTICLE::has_get_D_1LPT<T>()) {
                            if (ind == 0 and FML::ThisTask == 0)
                                std::cout << "Storing 1LPT displacment field in particle\n";
                            auto * D = FML::PARTICLE::GetD_1LPT(part_ptr[ind]);
                            for (int idim = 0; idim < N; idim++) {
                                D[idim] = disp[idim];
                            }
                        }
                    }

                    if (nLPT == 2) {
                        if constexpr (FML::PARTICLE::has_get_D_2LPT<T>()) {
                            if (ind == 0 and FML::ThisTask == 0)
                                std::cout << "Storing 2LPT displacment field in particle\n";
                            auto * D2 = FML::PARTICLE::GetD_2LPT(part_ptr[ind]);
                            for (int idim = 0; idim < N; idim++) {
                                D2[idim] = disp[idim];
                            }
                        }
                    }

                    if (nLPT == 3 and type == 'a') {
                        if constexpr (FML::PARTICLE::has_get_D_3LPTa<T>()) {
                            if (ind == 0 and FML::ThisTask == 0)
                                std::cout << "Storing 3LPTa displacment field in particle\n";
                            auto * D3a = FML::PARTICLE::GetD_3LPTa(part_ptr[ind]);
                            for (int idim = 0; idim < N; idim++) {
                                D3a[idim] = disp[idim];
                            }
                        }
                    }

                    if (nLPT == 3 and type == 'b') {
                        if constexpr (FML::PARTICLE::has_get_D_3LPTb<T>()) {
                            if (ind == 0 and FML::ThisTask == 0)
                                std::cout << "Storing 3LPTb displacment field in particle\n";
                            auto * D3b = FML::PARTICLE::GetD_3LPTb(part_ptr[ind]);
                            for (int idim = 0; idim < N; idim++) {
                                D3b[idim] = disp[idim];
                            }
                        }
                    }
                }

                // Output the maximum displacment and velocity
                FML::MaxOverTasks(&max_disp_nLPT);
                FML::MaxOverTasks(&max_vel_nLPT);
                if (FML::ThisTask == 0)
                    std::cout << "Maximum " << std::to_string(nLPT) << "LPT displacements: " << max_disp_nLPT * box
                              << " Mpc/h\n";
                if (FML::ThisTask == 0)
                    std::cout << "Maximum " << std::to_string(nLPT)
                              << "LPT velocity: " << max_vel_nLPT * 100.0 * box / aini << " km/s peculiar\n";
            };

            // Create particles
            part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);
            part.info();

            // Set unique IDs if we have that availiable in the particles
            if constexpr (FML::PARTICLE::has_set_id<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing unique ID in particle\n";
                long long int npart_local = part.get_npart();
                auto part_per_task = FML::GatherFromTasks(&npart_local);
                long long int id_start = 0;
                for (int i = 0; i < FML::ThisTask; i++)
                    id_start += part_per_task[i];
                long long int count = 0;
                for (auto & p : part) {
                    FML::PARTICLE::SetID(p, id_start + count++);
                }
            }

            // Set mass if we have that availiable in the particles
            if constexpr (FML::PARTICLE::has_set_mass<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing Mass (1.0) in particle\n";
                for (auto & p : part) {
                    FML::PARTICLE::SetMass(p, 1.0);
                }
            }

            // Set Lagrangian position of the particle if we have that availiable
            if constexpr (FML::PARTICLE::has_get_q<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing Lagrangian position q in particle\n";
                for (auto & p : part) {
                    auto pos = FML::PARTICLE::GetPos(p);
                    auto q = FML::PARTICLE::GetLagrangianPos(p);
                    for (int idim = 0; idim < N; idim++)
                        q[idim] = pos[idim];
                }
            }

            // Compute and add displacements
            // NB: we must do this in one go as add_displacement changes the position of the particles
            std::array<std::vector<FML::GRID::FloatType>, N> displacements_1LPT;
            if (LPT_order >= 1) {
                const int nLPT = 1;
                comp_displacement(nLPT, phi_1LPT, displacements_1LPT);
            }

            std::array<std::vector<FML::GRID::FloatType>, N> displacements_2LPT;
            if (LPT_order >= 2) {
                const int nLPT = 2;
                // Store potential if asked for
                if (phi_nLPT_potentials.size() > 0)
                    phi_nLPT_potentials[0] = phi_2LPT;
                comp_displacement(nLPT, phi_2LPT, displacements_2LPT);
            }

            std::array<std::vector<FML::GRID::FloatType>, N> displacements_3LPTa;
            if (LPT_order >= 3) {
                const int nLPT = 3;
                // Store potential if asked for
                if (phi_nLPT_potentials.size() > 1)
                    phi_nLPT_potentials[1] = phi_3LPTa;
                comp_displacement(nLPT, phi_3LPTa, displacements_3LPTa);
            }

            std::array<std::vector<FML::GRID::FloatType>, N> displacements_3LPTb;
            if (LPT_order >= 3) {
                const int nLPT = 3;
                // Store potential if asked for
                if (phi_nLPT_potentials.size() > 2)
                    phi_nLPT_potentials[2] = phi_3LPTb;
                comp_displacement(nLPT, phi_3LPTb, displacements_3LPTb);
            }

            if (LPT_order >= 1) {
                const int nLPT = 1;
                add_displacement(nLPT, 0, displacements_1LPT, velocity_norms[0]);
                for (auto & d : displacements_1LPT) {
                    d.clear();
                    d.shrink_to_fit();
                }
            }

            if (LPT_order >= 2) {
                const int nLPT = 2;
                add_displacement(nLPT, 0, displacements_2LPT, velocity_norms[1]);
                for (auto & d : displacements_2LPT) {
                    d.clear();
                    d.shrink_to_fit();
                }
            }

            if (LPT_order >= 3) {
                const int nLPT = 3;
                add_displacement(nLPT, 'a', displacements_3LPTa, velocity_norms[2]);
                for (auto & d : displacements_3LPTa) {
                    d.clear();
                    d.shrink_to_fit();
                }
            }

            if (LPT_order >= 3) {
                const int nLPT = 3;
                add_displacement(nLPT, 'b', displacements_3LPTb, velocity_norms[3]);
                for (auto & d : displacements_3LPTb) {
                    d.clear();
                    d.shrink_to_fit();
                }
            }

            // Communicate particles (they might have left the current task)
            part.communicate_particles();
        }

        //===================================================================================
        /// @brief This method computes the fifth-force potential for modified gravity models using the linear
        /// approximation This computes \f$ \delta_{\rm MG}(k) \f$ where the total force in fourier space is
        /// \f$ F(k) \propto \frac{\vec{k}}{k^2}[\delta(k) + \delta_{\rm MG}(k)] \f$ by solving
        /// \f$ \nabla^2 \phi = m^2 \phi + F^{-1}[g(k) \delta(k)] \f$ where \f$ \delta_{\rm MG}(k) = -k^2\phi(k) \f$.
        /// For example in \f$ f(R) \f$ gravity we have \f$ g(k) = \frac{1}{3}\frac{k^2}{k^2 + m^2}\f$ and in DGP
        /// we have \f$ g(k) = \frac{1}{3\beta} \f$ (independent of scale).
        ///
        /// @tparam N The dimension we work in.
        ///
        /// @param[in] density_fourier The density contrast in fourier space.
        /// @param[out] density_mg_fourier The force potential.
        /// @param[in] coupling_factor_of_kBox The coupling factor \f$ g(k) \f$
        ///
        //===================================================================================
        template <int N>
        void compute_delta_fifth_force(const FFTWGrid<N> & density_fourier,
                                       FFTWGrid<N> & density_mg_fourier,
                                       std::function<double(double)> coupling_factor_of_kBox) {

            auto coupling_factor_of_kBox_spline = density_fourier.make_fourier_spline(coupling_factor_of_kBox, "MG coupling(k)");
            const auto Local_nx = density_fourier.get_local_nx();
            density_mg_fourier = density_fourier;
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : density_mg_fourier.get_fourier_range(islice, islice + 1)) {
                    // Get wavevector and magnitude
                    density_mg_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                    // Compute coupling
                    auto coupling = coupling_factor_of_kBox_spline(kmag);

                    // Multiply by coupling
                    auto value = density_mg_fourier.get_fourier_from_index(fourier_index);
                    density_mg_fourier.set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(coupling));
                }
            }
        }

        //===================================================================================
        /// @brief This method computes the fifth-force potential for modified gravity models which has a screening
        /// mechanism using the approximate method of Winther & Ferreira 2015. This computes \f$ \delta_{\rm MG}(k) \f$
        /// where the total force in fourier is given by \f$ F(k) \propto \frac{\vec{k}}{k^2}[\delta(k) + \delta_{\rm
        /// MG}(k)] \f$ by solving \f$ \nabla^2 \phi = m^2 \phi + f(\Phi)F^{-1}[g(k) \delta(k)] \f$ where \f$
        /// \delta_{\rm MG}(k) = -k^2\phi(k) \f$ For example in \f$ f(R) \f$ gravity we have \f$ g(k) =
        /// \frac{1}{3}\frac{k^2}{k^2 + m^2} \f$ and the screening function is \f$ f(\Phi) = \min(1,
        /// \left|\frac{3f_R}{2\Phi}\right|) \f$ If you don't want screening then simpy pass the function \f$ f \equiv 1
        /// \f$ and the equation reduces to the one in the linear regime
        ///
        /// @tparam N The dimension we work in.
        ///
        /// @param[in] density_fourier The density contrast in fourier space.
        /// @param[out] density_mg_fourier The force potential.
        /// @param[in] coupling_factor_of_kBox The coupling factor \f$g(k)\f$
        /// @param[in] screening_factor_of_newtonian_potential The screening factor \f$ f(\Phi_N) \f$ Should be in
        /// \f$ [0,1] \f$ and go to 1 for \f$ \Phi_N \to 0 \f$ and 0 for very large \f$ \Phi_N \f$
        /// @param[in] poisson_norm The factor \f$ C \f$ in \f$ \nabla^2\Phi = C\delta \f$ to get the potential in the
        /// metric (not the code-potential) so \f$ C = \frac{3}{2}\Omega_M a \frac{(H_0B)^2}{a^2} \f$
        ///
        //===================================================================================
        template <int N>
        void compute_delta_fifth_force_potential_screening(
            const FFTWGrid<N> & density_fourier,
            FFTWGrid<N> & density_mg_fourier,
            std::function<double(double)> coupling_factor_of_kBox,
            std::function<double(double)> screening_factor_of_newtonian_potential,
            double poisson_norm) {

            const auto Local_nx = density_fourier.get_local_nx();
            const auto Local_x_start = density_fourier.get_local_x_start();

            // Make copy of density grid
            density_mg_fourier = density_fourier;

            // Transform to Newtonian potential
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : density_mg_fourier.get_fourier_range(islice, islice + 1)) {

                    auto value = density_mg_fourier.get_fourier_from_index(fourier_index);
                    density_mg_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    value *= -poisson_norm / kmag2;

                    density_mg_fourier.set_fourier_from_index(fourier_index, value);
                }
            }

            // Set DC mode
            if (Local_x_start == 0)
                density_mg_fourier.set_fourier_from_index(0, 0.0);

            // Take another copy of the density field as we need it in real space
            auto delta_real = density_fourier;

            // Transform to real space: Phi(x) and delta(x)
            density_mg_fourier.fftw_c2r();
            delta_real.fftw_c2r();

            // Apply screening function
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : density_mg_fourier.get_real_range(islice, islice + 1)) {
                    auto phi_newton = density_mg_fourier.get_real_from_index(real_index);
                    auto delta = delta_real.get_real_from_index(real_index);
                    auto screening_factor = screening_factor_of_newtonian_potential(phi_newton);
                    density_mg_fourier.set_real_from_index(real_index, delta * screening_factor);
                }
            }

            // Back to fourier space: delta(k)
            density_mg_fourier.fftw_r2c();

            // Apply coupling
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : density_mg_fourier.get_fourier_range(islice, islice + 1)) {
                    auto value = density_mg_fourier.get_fourier_from_index(fourier_index);

                    // Get wavevector and magnitude
                    density_mg_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                    // Compute coupling
                    auto coupling = coupling_factor_of_kBox(kmag);

                    // Multiply by coupling
                    density_mg_fourier.set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(coupling));
                }
            }
        }

        //===================================================================================
        /// @brief This method computes the fifth-force potential for modified gravity models which has a screening
        /// mechanism using the approximate method of Winther & Ferreira 2015. This computes \f$ \delta_{\rm MG}(k) \f$
        /// where the total force in fourier space is \f$ F(k) \propto \frac{\vec{k}}{k^2}[\delta(k) + \delta_{\rm
        /// MG}(k)] \f$ by solving \f$ \nabla^2 \phi = m^2 \phi + f(\rho)F^{-1}[g(k) \delta(k)] \f$ where \f$
        /// \delta_{\rm MG}(k) = -k^2\phi(k) \f$ and \f$ \rho \f$ is the density in units of the mean density. For
        /// example in DGP gravity we have \f$ g(k) = \frac{1}{3\beta} \f$ (independent of scale) and the screening
        /// function is \f$ f(\rho) = 2\frac{\sqrt{1 + C} - 1}{C} \f$ where \f$ C =
        /// \frac{8\Omega_M(r_cH_0)^2}{9\beta^2}\rho \f$ If you don't want screening then simply pass the function \f$ f
        /// \equiv 1 \f$ and the equation reduces to the one in the linear regime.
        ///
        /// @tparam N The dimension we work in.
        ///
        /// @param[in] density_fourier The density contrast in fourier space.
        /// @param[out] density_mg_fourier The force potential.
        /// @param[in] coupling_factor_of_kBox The coupling factor \f$ g(k) \f$
        /// @param[in] screening_factor_of_density The screening factor \f$ f(\rho) \f$ Should be in \f$ [0,1] \f$ and
        /// go to 1 for \f$ \rho \to 0 \f$ and to 0 for \f$ \rho \to \infty \f$.
        /// @param[in] smoothing_scale The smoothing radius in units of the boxsize.
        /// @param[in] smoothing_method The k-space smoothing filter (gaussian, tophat, sharpk).
        ///
        //===================================================================================
        template <int N>
        void compute_delta_fifth_force_density_screening(const FFTWGrid<N> & density_fourier,
                                                         FFTWGrid<N> & density_mg_fourier,
                                                         std::function<double(double)> coupling_factor_of_kBox,
                                                         std::function<double(double)> screening_factor_of_density,
                                                         double smoothing_scale,
                                                         std::string smoothing_method) {

            const auto Local_nx = density_fourier.get_local_nx();

            // Copy over
            density_mg_fourier = density_fourier;

            // Smooth density field
            if (smoothing_scale > 0.0)
                FML::GRID::smoothing_filter_fourier_space(density_mg_fourier, smoothing_scale, smoothing_method);

            // To real space density field
            density_mg_fourier.fftw_c2r();

            // Apply screening function
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : density_mg_fourier.get_real_range(islice, islice + 1)) {
                    auto delta = density_mg_fourier.get_real_from_index(real_index);
                    auto screening_factor = screening_factor_of_density(1.0 + delta);
                    density_mg_fourier.set_real_from_index(real_index, delta * screening_factor);
                }
            }

            // To fourier spacce
            density_mg_fourier.fftw_r2c();

            // Apply coupling
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : density_mg_fourier.get_fourier_range(islice, islice + 1)) {
                    auto value = density_mg_fourier.get_fourier_from_index(fourier_index);

                    // Get wavevector and magnitude
                    density_mg_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);

                    // Compute coupling
                    auto coupling = coupling_factor_of_kBox(kmag);

                    // Multiply by coupling
                    density_mg_fourier.set_fourier_from_index(fourier_index, value * FML::GRID::FloatType(coupling));
                }
            }
        }

#ifdef USE_GSL

        //===================================================================================
        /// @brief This computes the standard deviation of linear fluctuations smoothed over a sphere of radius, \f$
        /// \sigma(R) \f$, by integrating the power-spectrum convolved with a top-hat windowfunction of radius \f$ R
        /// \f$. We do this by solving \f$ \sigma^2(R) = \int \frac{k^3P(k)}{2\pi^2} |W(kR)|^2 d\log k \f$
        ///
        /// @param[in] Pofk_of_kBox_over_volume Dimensionless power-spectrum \f$ P / V \f$ as function of the
        /// dimensionless scale \f$ kB \f$ where \f$B\f$ is the boxsize and \f$V = B^3\f$ is the box volume.
        /// @param[in] R_mpch R in units of Mpc/h
        /// @param[in] boxsize_mpch Boxsize in units of Mpc/h
        ///
        //===================================================================================
        double
        compute_sigma_of_R(std::function<double(double)> Pofk_of_kBox_over_volume, double R_mpch, double boxsize_mpch) {
            using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
            using DVector = FML::SOLVERS::ODESOLVER::DVector;
            using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;

            // We integrate from k = 1e-5 h/Mpc to k = 100 h/Mpc
            const double kBoxmin = 1e-5 * boxsize_mpch;
            const double kBoxmax = 1e2 * boxsize_mpch;

            ODEFunction deriv = [&](double logkBox, [[maybe_unused]] const double * sigma2, double * dsigma2dlogk) {
                const double kBox = std::exp(logkBox);
                const double dimless_pofk = kBox * kBox * kBox / (2.0 * M_PI * M_PI) * Pofk_of_kBox_over_volume(kBox);
                const double kR = kBox * R_mpch / boxsize_mpch;
                const double window = kR > 0.0 ? 3.0 * (std::sin(kR) - kR * std::cos(kR)) / (kR * kR * kR) : 1.0;
                dsigma2dlogk[0] = dimless_pofk * window * window;
                return GSL_SUCCESS;
            };

            // The initial conditions
            DVector sigmaini{0.0};
            DVector logkarr{std::log(kBoxmin), std::log(kBoxmax)};

            // Solve the ODE
            ODESolver ode(1e-3, 1e-10, 1e-10);
            ode.solve(deriv, logkarr, sigmaini);
            auto sigma2 = ode.get_final_data_by_component(0);

            return std::sqrt(sigma2);
        }
#endif

    } // namespace NBODY
} // namespace FML
#endif
