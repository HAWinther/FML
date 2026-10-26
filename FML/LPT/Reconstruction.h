#ifndef RECONSTRUCTION_HEADER
#define RECONSTRUCTION_HEADER
#include <array>
#include <cassert>
#include <climits>
#include <complex>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Smoothing/SmoothingFourier.h>

namespace FML {
    namespace COSMOLOGY {
        namespace LPT {

            template <int N>
            using FFTWGrid = FML::GRID::FFTWGrid<N>;
            template <class T>
            using MPIParticles = FML::PARTICLE::MPIParticles<T>;
            using FloatType = FML::GRID::FloatType;

            namespace detail {
                /// Helper particle for RSDReconstructionFourierMethod (only a position)
                template <int N>
                struct ReconstructionParticle {
                    double pos[N];
                    double * get_pos() { return pos; }
                    constexpr int get_ndim() const { return N; }
                };
            } // namespace detail

            //============================================================================
            ///
            /// RSD removal for particles in a periodic box
            /// We use a fixed line of sight direction los_direction, typically
            /// coordinate axes (e.g. los_direction = {0,0,1} )
            /// We are solving the equation LPT equation
            ///  \f$ \nabla(b\Psi) + \beta \nabla((b\Psi\cdot r)r) = -\delta_{\rm tracer} \f$
            /// and assuming zero-curl (which is not perfectly true) of the second term
            /// so we can use Fourier methods.
            /// Then we subtract the RSD which is given by \f$ \Psi_{\rm rsd} = \beta (b\Psi\cdot r)r) \f$
            /// Here \f$ \beta = f / b \f$ growth-rate over galaxy bias.
            /// The smoothing filter is an (optional) filter we apply to the density field in
            /// k-space. If R < gridspacing then no smoothing will be done
            ///
            /// The iterative scheme of Burden et al. 2015 (arXiv:1504.02591): with \f$ s \f$ the observed
            /// (redshift-space) positions and \f$ x_0 = s \f$, iteration n computes \f$ b\Psi_n \f$ from the density
            /// of the positions \f$ x_{n-1} \f$ and sets \f$ x_n = s - \beta (b\Psi_n(s)\cdot r)r \f$. The first
            /// iteration overestimates the shift (the density is in redshift space); the iterations converge for
            /// \f$ \beta < 1 \f$ (3-5 iterations are enough).
            ///
            /// Only a periodic box with a fixed line of sight is supported (survey_data = true, a line of sight
            /// towards an observer, is not implemented and aborts).
            ///
            /// This method assumes scale-independent growth factor, but that is easily changed
            ///
            /// @tparam N The dimension of the grid
            /// @tparam T The particle class
            ///
            /// @param[out] part MPIParticles. Particles gets updated.
            /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS, PQS)
            /// @param[in] los_direction The fixed line of sight direction, e.g. (0,0,1) for the z-axis
            /// @param[in] Nmesh The size of the grid we use
            /// @param[in] niterations How many iterations to do (not too many and not too few, 3 is good)
            /// @param[in] beta This is beta = f/b the growth-rate over the bias
            /// @param[in] smoothing_options The smoothing filter (gaussian, tophat, sharph) and the smothing scale (in
            /// units of the boxsize)
            /// @param[in] survey_data Must be false (survey data is not supported)
            ///
            //============================================================================

            template <int N, class T>
            void RSDReconstructionFourierMethod(MPIParticles<T> & part,
                                                std::string density_assignment_method,
                                                std::vector<double> los_direction,
                                                int Nmesh,
                                                int niterations,
                                                double beta,
                                                std::pair<std::string, double> smoothing_options,
                                                bool survey_data) {

                static_assert(FML::PARTICLE::has_get_pos<T>(),
                              "[RSDReconstructionFourierMethod] Particle must have a get_pos method");
                assert_mpi(not survey_data,
                           "[RSDReconstructionFourierMethod] Survey data (line of sight towards an observer, no "
                           "periodic box) is not supported\n");
                assert_mpi(niterations >= 1, "[RSDReconstructionFourierMethod] Need at least one iteration\n");

                // Use N-linear interpolation
                const std::string interpolation_method = "CIC";

                // The particles stay at the observed positions s until the end
                const size_t NumPart = part.get_npart();
                auto * p = part.get_particles_ptr();

                // Normalize the los_direction to a unit vector
                assert_mpi(los_direction.size() == N,
                           "[RSDReconstructionFourierMethod] Line of sight direction has wrong dimension\n");
                double norm = 0.0;
                for (int idim = 0; idim < N; idim++) {
                    norm += los_direction[idim] * los_direction[idim];
                }
                assert_mpi(norm > 0.0,
                           "[RSDReconstructionFourierMethod] Line of sight vector cannot be the zero vector\n");
                norm = 1.0 / std::sqrt(norm);
                for (int idim = 0; idim < N; idim++) {
                    los_direction[idim] *= norm;
                }

                // The RSD shift beta (bPsi*r)r of each particle (from the last iteration)
                std::vector<double> shift(NumPart, 0.0);

                auto wrap = [](double x) {
                    x -= std::floor(x);
                    return x >= 1.0 ? 0.0 : x;
                };

                auto nleftright =
                    FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);

                for (int i = 0; i < niterations; i++) {
                    if (FML::ThisTask == 0)
                        std::cout << "[RSDReconstructionFourierMethod] Iteration: " << i + 1 << " / " << niterations
                                  << "\n";

                    // The density field of the current estimate of the real-space positions x = s - shift
                    // (the observed positions in the first iteration)
                    FFTWGrid<N> density(Nmesh, nleftright.first, nleftright.second);
                    density.add_memory_label("FFTWGrid::RSDReconstructionFourierMethod::density");
                    density.set_grid_status_real(true);
                    if (i == 0) {
                        FML::INTERPOLATION::particles_to_grid(
                            p, NumPart, part.get_npart_total(), density, density_assignment_method);
                    } else {
                        // Copies at the shifted positions (they can be on another task)
                        using Helper = detail::ReconstructionParticle<N>;
                        std::vector<Helper> shifted(NumPart);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                        for (size_t j = 0; j < NumPart; j++) {
                            const auto * pos = FML::PARTICLE::GetPos(p[j]);
                            for (int idim = 0; idim < N; idim++)
                                shifted[j].pos[idim] = wrap(pos[idim] - shift[j] * los_direction[idim]);
                        }
                        long long int nmax = NumPart;
                        FML::MaxOverTasks(&nmax);
                        FML::PARTICLE::MPIParticles<Helper> shifted_part;
                        shifted_part.create(shifted.data(),
                                            NumPart,
                                            size_t(2 * nmax + 1000),
                                            FML::xmin_domain,
                                            FML::xmax_domain,
                                            false);
                        std::vector<Helper>().swap(shifted);
                        FML::INTERPOLATION::particles_to_grid(shifted_part.get_particles_ptr(),
                                                              shifted_part.get_npart(),
                                                              shifted_part.get_npart_total(),
                                                              density,
                                                              density_assignment_method);
                    }

                    density.fftw_r2c();

                    // Perform a smoothing
                    FML::GRID::smoothing_filter_fourier_space(
                        density, smoothing_options.second, smoothing_options.first);

                    // The 1LPT potential
                    FFTWGrid<N> phi_1LPT;
                    compute_1LPT_potential_fourier(density, phi_1LPT);

                    // Free some memory
                    density.free();

                    // The 1LPT displacement field
                    std::array<FFTWGrid<N>, N> Psi;
                    from_LPT_potential_to_displacement_vector<N>(phi_1LPT, Psi);
                    for (int idim = 0; idim < N; idim++) {
                        Psi[idim].communicate_boundaries();
                    }

                    // Free some memory
                    phi_1LPT.free();

                    // Interpolate Psi to the observed particle positions
                    std::array<std::vector<FloatType>, N> Psi_particle_positions;
                    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<N, T>(
                        Psi, p, NumPart, Psi_particle_positions, interpolation_method);
                    for (int idim = 0; idim < N; idim++) {
                        Psi[idim].free();
                    }

                    // The RSD shift beta (bPsi*r)r
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t j = 0; j < NumPart; j++) {
                        double Psidotr = 0.0;
                        for (int idim = 0; idim < N; idim++)
                            Psidotr += los_direction[idim] * Psi_particle_positions[idim][j];
                        shift[j] = beta * Psidotr;
                    }
                }

                // Subtract the RSD shift, do a periodic wrap and communicate the particles in case they have left the
                // current domain
                double max_shift = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_shift)
#endif
                for (size_t j = 0; j < NumPart; j++) {
                    auto * pos = FML::PARTICLE::GetPos(p[j]);
                    for (int idim = 0; idim < N; idim++)
                        pos[idim] = wrap(pos[idim] - shift[j] * los_direction[idim]);
                    max_shift = std::max(max_shift, std::abs(shift[j]));
                }
                part.communicate_particles();

                // Show maximum shift
                FML::MaxOverTasks(&max_shift);
                if (FML::ThisTask == 0)
                    std::cout << "[RSDReconstructionFourierMethod] Maximum shift (boxsize units): " << max_shift
                              << "\n";
            }
        } // namespace LPT

        // NAMESPACE FML::COSMOLOGY

        //================================================================================
        /// This takes a set of particles and displace them from realspace to redshiftspace
        /// Using a fixed line of sight direction
        /// DeltaX = (v * r)r * velocity_to_displacement
        /// If velocities are peculiar then velocity_to_displacement = 1/(100 * aH/H0 * box) with box in Mpc/h
        /// You can revert back to real-space by calling the method again with velocity_to_displacement being negative
        ///
        /// @tparam T The particle class
        ///
        /// @param[out] part MPIParticle container
        /// @param[in] line_of_sight_direction The fixed line of sight vector, e.g. (0,0,1) for the z-axis
        /// @param[in] velocity_to_displacement Convert from user velocity to RSD shift.
        ///
        //================================================================================
        template <class T>
        void particles_to_redshiftspace(FML::PARTICLE::MPIParticles<T> & part,
                                        std::vector<double> line_of_sight_direction,
                                        double velocity_to_displacement) {

            // Fetch how many dimensjons we are working in
            const int N = FML::PARTICLE::GetNDIM(T());

            // Check that velocities really exists (i.e. get_vel is not just set to return a nullptr)
            static_assert(FML::PARTICLE::has_get_pos<T>(),
                          "[particles_to_redshiftspace] Partices must have positions to use this method");
            static_assert(FML::PARTICLE::has_get_vel<T>(),
                          "[particles_to_redshiftspace] Partices must have velocity to use this method");

            // Periodic box? Yes, this is only meant to be used with simulation boxes
            const bool periodic_box = true;

            // Make sure line_of_sight_direction is a unit vector
            double norm = 0.0;
            for (int idim = 0; idim < N; idim++) {
                norm += line_of_sight_direction[idim] * line_of_sight_direction[idim];
            }
            norm = std::sqrt(norm);
            assert_mpi(norm > 0.0, "[particles_to_redshiftspace] Line of sight vector cannot be the zero vector\n");
            for (int idim = 0; idim < N; idim++) {
                line_of_sight_direction[idim] /= norm;
            }

            double max_disp = 0.0;
            auto NumPart = part.get_npart();
            auto * p = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
            for (size_t i = 0; i < NumPart; i++) {
                auto * pos = FML::PARTICLE::GetPos(p[i]);
                auto * vel = FML::PARTICLE::GetVel(p[i]);
                double vdotr = 0.0;
                for (int idim = 0; idim < N; idim++) {
                    vdotr += vel[idim] * line_of_sight_direction[idim];
                }
                max_disp = std::max(max_disp, std::fabs(vdotr));
                for (int idim = 0; idim < N; idim++) {
                    pos[idim] += vdotr * line_of_sight_direction[idim] * velocity_to_displacement;
                    // Periodic boundary conditions
                    if (periodic_box) {
                        if (pos[idim] < 0.0)
                            pos[idim] += 1.0;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1.0;
                    }
                }
            }

            // Only need to communicate if we have shifted along the x-axis
            if (line_of_sight_direction[0] != 0.0)
                part.communicate_particles();

            // If velocity_to_displacement < 0 we are going the other way around
            FML::MaxOverTasks(&max_disp);
            max_disp *= velocity_to_displacement;
            if (FML::ThisTask == 0)
                std::cout << "[particles_to_redshiftspace] Maximum displacement "
                          << (velocity_to_displacement < 0 ? "(reversing) : " : "            :  ") << max_disp << "\n";
        }

    } // namespace COSMOLOGY
} // namespace FML
#endif
