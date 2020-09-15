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
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/RandomFields/GaussianRandomField.h>

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
                                    FML::PARTICLE::MPIParticles<T> & part,
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
            FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles().data(),
                                                        part.get_npart(),
                                                        part.get_npart_total(),
                                                        density_grid_real,
                                                        density_assignment_method);

            // Density field -> force
            std::array<FFTWGrid<N>, N> force_real;
            compute_force_from_density(density_grid_real, force_real, norm_poisson_equation);

            // Update velocity of particles
            KickParticles(force_real, part, delta_time * 0.5, density_assignment_method);

            // Move particles (this does communication)
            DriftParticles<N, T>(part, delta_time, periodic_box);

            // Particles -> density field
            FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles().data(),
                                                        part.get_npart(),
                                                        part.get_npart_total(),
                                                        density_grid_real,
                                                        density_assignment_method);

            // Density field -> force
            compute_force_from_density(density_grid_real, force_real, norm_poisson_equation);

            // Update velocity of particles
            KickParticles(force_real, part, delta_time * 0.5, density_assignment_method);
        }

        //===================================================================================
        /// @brief Take a N-body step with a 4th order symplectic Yoshida method.
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
                              FML::PARTICLE::MPIParticles<T> & part,
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
                FML::INTERPOLATION::particles_to_grid<N, T>(part.get_particles().data(),
                                                            part.get_npart(),
                                                            part.get_npart_total(),
                                                            density_grid_real,
                                                            density_assignment_method);
                // Density field -> force
                std::array<FFTWGrid<N>, N> force_real;
                compute_force_from_density(density_grid_real, force_real, norm_poisson);

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
            for (int idim = 0; idim < N; idim++) {
                force_real[idim] = density_grid_real;
                force_real[idim].add_memory_label("FFTWGrid::compute_force_from_density::force_real_" +
                                                  std::to_string(idim));
                force_real[idim].set_grid_status_real(idim == 0);
            }

            // Density grid to fourier space
            force_real[0].fftw_r2c();

            auto Local_nx = density_grid_real.get_local_nx();
            auto Local_x_start = density_grid_real.get_local_x_start();

            // Loop over all local fourier grid cells
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                std::complex<double> I(0, 1);
                for (auto && fourier_index : force_real[0].get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode (k=0)

                    force_real[0].get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    auto value = force_real[0].get_fourier_from_index(fourier_index);

                    // Multiply by -i/k^2
                    value *= -norm_poisson_equation * I / kmag2;

                    // Compute force -ik/k^2 delta(k)
                    for (int idim = 0; idim < N; idim++)
                        force_real[idim].set_fourier_from_index(fourier_index, value * kvec[idim]);
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
        void DriftParticles(FML::PARTICLE::MPIParticles<T> & part, double delta_time, bool periodic_box) {
            if (part.get_npart() == 0)
                return;

            // Sanity check on particle
            T tmp;
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

            // Sanity check on particle
            T tmp;
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
        /// assumes the force is normalized such that \f$ F \Delta t \f$ has the same units as your v. This method
        /// frees up memory in the force grids after we have used them. Can be changed with a flag in the source.
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
        /// This moves the particle velocities according to \f$ v_{\rm new} = v + F \Delta t \f$. This method
        /// assumes the force is normalized such that \f$ F \Delta t \f$ has the same units as your v. This method
        /// frees up memory in the force grids after we have used them. Can be changed with a flag in the source.
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

            // Sanity check on particle
            T tmp;
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N, "[KickParticles] Dimension of particle and grid do not match");
            static_assert(FML::PARTICLE::has_get_vel<T>(),
                          "[KickParticles] Particle must have velocity to use this method");

            // Deallocate the force grids (after interpolating to the particles we don't need it here and probably
            // not elsewhere so lets save some memory)
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

        ///=====================================================================
        /// Generate particles from a given power-spectrum using Lagrangian perturbation theory.
        /// We generate particles in [0,1) and velocities are given by v_code = a^2 dxdt / (H0 Box)
        ///
        /// @tparam N The dimension we are working in.
        /// @tparam T The particle class. Must have methods get_pos, get_vel, get_D_1LPT and get_D_2LPT. But only
        /// get_pos data is required to exist. Return a nullptr if the data does not exist in the particle.
        ///
        /// @param[out] part Particle container for particles we are to create.
        /// @param[in] Npart_1D Number of particles per dimension (i.e. total is Npart_1D^N)
        /// @param[in] buffer_factor How many more particles to allocate?
        /// @param[in] Nmesh The grid to generate the IC on
        /// @param[in] fix_amplitude Amplitude fixed? Only random phases if true.
        /// @param[in] rng Random number generator
        /// @param[in] Pofk_of_kBox_over_volume The dimensionless function P(k)/VolumeOfBox as function of the
        /// dimensionless wavenumber k*Box
        /// @param[in] LPT_order The LPT order (1 or 2)
        /// @param[in] box The boxsize (only for prining maximum displacement)
        /// @param[in] zini The initial redshift
        /// @param[in] H_over_H0_of_loga The function H/H0 as function of x = log(a)
        /// @param[in] growth_rate_f_of_loga The growth-rate f_1LPT, f_2LPT, ... as function of x = log(a)
        ///
        ///=====================================================================
        template <int N, class T>
        void NBodyInitialConditions(FML::PARTICLE::MPIParticles<T> & part,
                                    int Npart_1D,
                                    double buffer_factor,

                                    int Nmesh,
                                    bool fix_amplitude,
                                    FML::RANDOM::RandomGenerator * rng,
                                    std::function<double(double)> & Pofk_of_kBox_over_volume,
                                    int LPT_order,

                                    double box,
                                    double zini,
                                    std::function<double(double)> & H_over_H0_of_loga,
                                    std::vector<std::function<double(double)>> & growth_rate_f_of_loga) {

            T tmp;
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
                std::cout << "# Gaussian random field " << (fix_amplitude ? "with fixed amplitude" : "") << "\n";
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
                              << sizeof(FML::PARTICLE::GetD_1LPT(tmp)[0]) * N << " bytes)\n";
                if (FML::PARTICLE::has_get_q<T>())
                    std::cout << "# Particle has [Lagrangian position] ("
                              << sizeof(FML::PARTICLE::GetLagrangianPos(tmp)[0]) * N << " bytes)\n";
                std::cout << "# Total size of particle is " << FML::PARTICLE::GetSize(tmp) << " bytes\n";
                std::cout << "# We will make " << Npart_1D << "^" << N << " particles\n";
                std::cout << "# Plus a buffer with room for " << (buffer_factor - 1.0) * 100.0 << "\% more particles\n";
                std::cout << "# We will allocate ~"
                          << buffer_factor * FML::PARTICLE::GetSize(tmp) * FML::power(double(Npart_1D), N) / 1e6 /
                                 double(FML::NTasks)
                          << " MB per task for the particles\n";
                std::cout << "#\n";
                std::cout << "#=====================================================\n";
                std::cout << "\n";
            }

            // Sanity checks
            assert_mpi(Npart_1D > 0 and Nmesh > 0 and zini >= 0.0 and rng != nullptr,
                       "[NBodyInitialConditions] Invalid parameters");
            assert_mpi(LPT_order == 1 or LPT_order == 2 or LPT_order == 3,
                       "[NBodyInitialConditions] Only 1LPT, 2LPT and 3LPT implemented so valid choices here are "
                       "LPT_order = 1, 2 or 3");

            // Sanity check on particle
            assert_mpi(FML::PARTICLE::GetNDIM(tmp) == N,
                       "[NBodyInitialConditions] NDIM of particles and of grid does not match");
            assert_mpi(FML::PARTICLE::has_get_pos<T>(),
                       "[NBodyInitialConditions] Particle class must have a get_pos method");

            // The scalefactor and log(a) at the initial time
            const double aini = 1.0 / (1.0 + zini);
            const double xini = std::log(aini);

            // We provide the power-spectrum at the initial time
            auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment("CIC");
            FFTWGrid<N> delta(Nmesh, nextra.first, nextra.second);
            delta.set_grid_status_real(false);

            // Make a random field in fourier space
            FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(
                delta, rng, Pofk_of_kBox_over_volume, fix_amplitude);

            FFTWGrid<N> phi_1LPT;
            FFTWGrid<N> phi_2LPT;
            FFTWGrid<N> phi_3LPT;
            if (LPT_order == 1) {
                // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi_1LPT);
            } else if (LPT_order == 2) {
                // Generate the 1LPT potential phi_1LPT = delta(k)/k^2
                FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi_1LPT);
                // Generate the 2LPT potential phi_2LPT = -1/2k^2 F[phi_ii phi_jj - phi_ij^2]
                FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta, phi_2LPT);
            } else if (LPT_order == 3) {
                // Generate the 3LPT potentials phi_3LPT_a, phi_3LPT_b plus 3LPT curl term
                // We ignore the curl term in this implementation for simplicity
                const bool ignore_3LPT_curl_term = true;
                FFTWGrid<N> & phi_3LPT_a = phi_3LPT;
                FFTWGrid<N> phi_3LPT_b;
                std::vector<FFTWGrid<N>> phi_3LPT_Avec_fourier;
                FML::COSMOLOGY::LPT::compute_3LPT_potential_fourier(
                    delta, phi_1LPT, phi_2LPT, phi_3LPT_a, phi_3LPT_b, phi_3LPT_Avec_fourier, ignore_3LPT_curl_term);

                // Add up the two 3LPT potentials
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < phi_3LPT.get_local_nx(); islice++) {
                    for (auto && fourier_index : phi_3LPT.get_fourier_range(islice, islice + 1)) {
                        auto value_a = phi_3LPT_a.get_fourier_from_index(fourier_index);
                        auto value_b = phi_3LPT_b.get_fourier_from_index(fourier_index);
                        phi_3LPT.set_fourier_from_index(fourier_index, value_a + value_b);
                    }
                }
            }

            // Free memory no longer needed
            delta.free();

            //================================================================
            // Function to compute the displacement from a LPT potential
            // Frees the memory of phi_nLPT after its used
            //================================================================
            auto comp_displacement = [&]([[maybe_unused]] int nLPT,
                                         FFTWGrid<N> & phi_nLPT,
                                         std::vector<std::vector<FML::GRID::FloatType>> & displacements_nLPT) {
                // Generate Psi from phi
                std::vector<FFTWGrid<N>> Psi_nLPT_vector;
                FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector(phi_nLPT, Psi_nLPT_vector);
                phi_nLPT.free();

                // Interpolate it to particle Lagrangian positions
                for (int idim = 0; idim < N; idim++) {
                    FML::INTERPOLATION::interpolate_grid_to_particle_positions(Psi_nLPT_vector[idim],
                                                                               part.get_particles_ptr(),
                                                                               part.get_npart(),
                                                                               displacements_nLPT[idim],
                                                                               "CIC");
                    Psi_nLPT_vector[idim].free();
                }
            };

            auto add_displacement = [&]([[maybe_unused]] int nLPT,
                                        std::vector<std::vector<FML::GRID::FloatType>> & displacements_nLPT,
                                        double vfac_nLPT) {
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
                    if (FML::PARTICLE::has_get_vel<T>()) {
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
                    // If the particles has get_D_1LPT and get_D_2LPT not being a nullptr we store it
                    // If you don't want particles to have these methods just comment this out
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

                    if (nLPT == 3) {
                        if constexpr (FML::PARTICLE::has_get_D_3LPT<T>()) {
                            if (ind == 0 and FML::ThisTask == 0)
                                std::cout << "Storing 3LPT displacment field in particle\n";
                            auto * D2 = FML::PARTICLE::GetD_3LPT(part_ptr[ind]);
                            for (int idim = 0; idim < N; idim++) {
                                D2[idim] = disp[idim];
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

            // Set unique IDs if we have that availiable in the particles
            if constexpr (FML::PARTICLE::has_set_id<T>()) {
                if (FML::ThisTask == 0)
                    std::cout << "Storing unique ID in particle\n";
                long long int npart_local = part.get_npart();
                auto part_per_task = FML::GatherFromAllTasks(&npart_local);
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
            std::vector<std::vector<FML::GRID::FloatType>> displacements_1LPT(N);
            if (LPT_order >= 1) {
                const int nLPT = 1;
                comp_displacement(nLPT, phi_1LPT, displacements_1LPT);
                phi_1LPT.free();
            }

            std::vector<std::vector<FML::GRID::FloatType>> displacements_2LPT(N);
            if (LPT_order >= 2) {
                const int nLPT = 2;
                comp_displacement(nLPT, phi_2LPT, displacements_2LPT);
                phi_2LPT.free();
            }

            std::vector<std::vector<FML::GRID::FloatType>> displacements_3LPT(N);
            if (LPT_order >= 3) {
                const int nLPT = 3;
                comp_displacement(nLPT, phi_3LPT, displacements_3LPT);
                phi_3LPT.free();
            }

            if (LPT_order >= 1) {
                const int nLPT = 1;
                const double growth_rate1 = growth_rate_f_of_loga[nLPT - 1](xini);
                const double vfac_1LPT = std::exp(2 * xini) * H_over_H0_of_loga(xini) * growth_rate1;
                add_displacement(nLPT, displacements_1LPT, vfac_1LPT);
            }

            if (LPT_order >= 2) {
                const int nLPT = 2;
                const double growth_rate2 = growth_rate_f_of_loga[nLPT - 1](xini);
                const double vfac_2LPT = std::exp(2 * xini) * H_over_H0_of_loga(xini) * growth_rate2;
                add_displacement(nLPT, displacements_2LPT, vfac_2LPT);
            }

            if (LPT_order >= 3) {
                const int nLPT = 3;
                const double growth_rate3 = growth_rate_f_of_loga[nLPT - 1](xini);
                const double vfac_3LPT = std::exp(2 * xini) * H_over_H0_of_loga(xini) * growth_rate3;
                add_displacement(nLPT, displacements_3LPT, vfac_3LPT);
            }

            // Communicate particles (they might have left the current task)
            part.communicate_particles();
        }

    } // namespace NBODY
} // namespace FML
#endif
