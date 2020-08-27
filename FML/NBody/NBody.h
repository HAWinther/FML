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

namespace FML {
    namespace NBODY {

        // Type alias
        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;

        template <int N, class T>
        void DriftParticles(FML::PARTICLE::MPIParticles<T> & part, double delta_time, bool periodic_box = true);

        template <int N, class T>
        void DriftParticles(T * p, size_t NumPart, double delta_time, bool periodic_box = true);

        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           FML::PARTICLE::MPIParticles<T> & part,
                           double delta_time,
                           std::string interpolation_method);

        template <int N, class T>
        void KickParticles(std::array<FFTWGrid<N>, N> & force_grid,
                           T * p,
                           size_t NumPart,
                           double delta_time,
                           std::string interpolation_method);

        template <int N>
        void compute_force_from_density(const FFTWGrid<N> & density_grid_real,
                                        std::array<FFTWGrid<N>, N> & force_real,
                                        double norm_poisson_equation = 1.0);

        //===================================================================================
        /// @brief Take a N-body step with a naive Kick-Drift method (thus this
        /// methods should mainly serve as an example for how to do this and not be used for anything serious).
        /// 1. Particles to grid to get \f$ \delta \f$
        /// 2. Compute the Newtonian potential via \f$ \nabla^2 \Phi = {\rm norm} \cdot \delta \f$
        /// 3. Compute the force  \f$ F = \nabla \Phi \f$
        /// 4. Move the particles using \f$ x \to x + v \Delta t \f$ and \f$ v \to v + F \Delta t \f$
        /// This method assumes that the velocities are in units of (boxsize / time-step-unit), in other words that \f$
        /// v\Delta t\f$ gives rise to a shift in [0,1).
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
        void KickDriftNBodyStep(int Nmesh,
                                FML::PARTICLE::MPIParticles<T> & part,
                                double delta_time,
                                std::string density_assignment_method,
                                double norm_poisson_equation) {

            const bool periodic_box = true;

            // Particles -> density field
            auto nleftright =
                FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);
            FFTWGrid<N> density_grid_real(Nmesh, nleftright.first, nleftright.second);
            FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles().data(),
                                                        part.get_npart(),
                                                        part.get_npart_total(),
                                                        density_grid_real,
                                                        density_assignment_method);

            // Density field -> force
            std::array<FFTWGrid<N>, N> force_real;
            compute_force_from_density(density_grid_real, force_real, norm_poisson_equation);

            // Update velocity of particles
            KickParticles(force_real, part, delta_time, density_assignment_method);

            // Move particles (this does communication)
            DriftParticles<N, T>(part, delta_time, periodic_box);
        }

        //===================================================================================
        /// Take a density grid in real space and returns the force \f$ \nabla \phi \f$  where
        /// \f$ \nabla^2 \phi = {\rm norm} \cdot \delta \f$
        ///
        /// @tparam N The dimension of the grid
        ///
        /// @param[in] density_grid_real The density contrast in real space.
        /// @param[out] force_real The force in real space.
        /// @param[in] norm_poisson_equation The prefactor (norm) to the Poisson equation.
        ///
        //===================================================================================
        template <int N>
        void compute_force_from_density(const FFTWGrid<N> & density_grid_real,
                                        std::array<FFTWGrid<N>, N> & force_real,
                                        double norm_poisson_equation) {

            // Copy over
            for (int idim = 0; idim < N; idim++)
                force_real[idim] = density_grid_real;

            // Density grid to fourier space
            force_real[0].fftw_r2c();

            auto Local_x_start = density_grid_real.get_local_x_start();

            // Loop over all local fourier grid cells
            std::array<double, N> kvec;
            double kmag2;
            std::complex<double> I(0, 1);
            for (auto & fourier_index : force_real[0].get_fourier_range()) {
                force_real[0].get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                auto value = force_real[0].get_fourier_from_index(fourier_index);

                // Multiply by -i/k^2
                value *= -norm_poisson_equation * I / kmag2;
                if (Local_x_start == 0 and fourier_index == 0)
                    value = 0.0;

                // Compute force -ik/k^2 delta(k)
                for (int idim = 0; idim < N; idim++)
                    force_real[idim].set_fourier_from_index(fourier_index, value * kvec[idim]);
            }

            // DC mode
            if (FML::ThisTask == 0) {
                for (int idim = 0; idim < N; idim++)
                    force_real[idim].set_fourier_from_index(0, 0.0);
            }

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
        void DriftParticles(FML::PARTICLE::MPIParticles<T> & part, double delta_time, bool periodic_box) {
            if (part.get_npart() == 0)
                return;
            const int Ndim = part[0].get_ndim();
            assert(Ndim == N);

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
            const int Ndim = p[0].get_ndim();
            assert(Ndim == N);

            double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * pos = p[i].get_pos();
                auto * vel = p[i].get_vel();
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
                std::cout << "[Drift] Max disp: " << max_disp << "\n";
        }

        //===================================================================================
        /// This moves the particle velocities according to \f$ v_{\rm new} = v + F \Delta t \f$. This method assumes
        /// the force is normalized such that \f$ F \Delta t \f$ has the same units as your v. This method frees up
        /// memory in the force grids after we have used them. Can be changed with a flag in the source.
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
                           FML::PARTICLE::MPIParticles<T> & part,
                           double delta_time,
                           std::string interpolation_method) {

            KickParticles<N, T>(
                force_grid, part.get_particles_ptr(), part.get_npart(), delta_time, interpolation_method);
        }

        //===================================================================================
        /// This moves the particle velocities according to \f$ v_{\rm new} = v + F \Delta t \f$. This method assumes
        /// the force is normalized such that \f$ F \Delta t \f$ has the same units as your v. This method frees up
        /// memory in the force grids after we have used them. Can be changed with a flag in the source.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        ///
        /// @param[in] force_grid The force \f$ \nabla \Phi \f$. This grid is deallocated after use. Set
        /// free_force_grids = false in the source to chansource to change this.
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

            // Deallocate the force grids (after interpolating to the particles we don't need it here and probably not
            // elsewhere so lets save some memory)
            const bool free_force_grids = true;

            // Interpolate force to particle positions
            std::array<std::vector<double>, N> force;
            for (int idim = 0; idim < N; idim++) {
                force_grid[idim].communicate_boundaries();
                FML::INTERPOLATION::interpolate_grid_to_particle_positions<N, T>(
                    force_grid[idim], p, NumPart, force[idim], interpolation_method);
                if (free_force_grids)
                    force_grid[idim].free();
            }

            double max_dvel = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_dvel)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * vel = p[i].get_vel();
                for (int idim = 0; idim < N; idim++) {
                    double dvel = -force[idim][i] * delta_time;
                    max_dvel = std::max(max_dvel, std::abs(dvel));
                    vel[idim] += dvel;
                }
            }

            FML::MaxOverTasks(&max_dvel);

            if (FML::ThisTask == 0)
                std::cout << "[Kick] Max dvel disp: " << max_dvel * delta_time << "\n";
        }

    } // namespace NBODY
} // namespace FML
#endif
