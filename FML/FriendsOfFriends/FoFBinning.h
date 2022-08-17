#ifndef FOFBINNING_HEADER
#define FOFBINNING_HEADER
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <array>
#include <cassert>
#include <vector>

namespace FML {
    namespace FOF {

        //=========================================================================
        /// This class defines how to compute the halos from the particles
        ///
        /// If you want to compute more things then make your own class like this one
        /// and supply as template parameter to the FriendsOfFriends method.
        ///
        /// NB: The computation of the COM has to be done incrementally: we add up the mean
        /// values as we go along instead of doing it in the end. For other quantities you
        /// can do it as you wish.
        ///
        /// If you want access to all particles then one can add a vector of
        /// pointers to the particles, push them in this vector during add and
        /// then process them in finalize.
        ///
        //=========================================================================
        template <class T, int NDIM>
        class FoFHalo {
          public:
            /// Id of the halo
            size_t id{0};
            /// Number of particles in the halo
            size_t np{0};
            /// Mass of the halo (units is the same as those in the particles; get_mass)
            double mass{0.0};
            /// Center of the halo
            std::array<double, NDIM> pos;
            /// Velocity of the halo
            std::array<double, NDIM> vel;
            /// RMS velocity of the particles in the halo
            /// Contains <v^2> before finalizing
            std::array<double, NDIM> vel_rms;

            // To be able to use this with MPIParticles we need these methods
            constexpr int get_ndim() const { return NDIM; }
            double * get_pos() { return pos.data(); }
            double * get_vel() { return vel.data(); }

            // NB: we don't add a get_mass as this will then be used if
            // we compute density field of halos which is usually not what
            // we want
            double get_halo_mass() const { return mass; };

            size_t get_npart() const { return np; };
            size_t get_id() const { return id; };
            double get_velocity_dispersion() const { return vel_rms; };

#ifdef STORE_MEMBERS_PTRS
            // If you want access to all particles in the end
            std::vector<T *> member_particles;
#endif

            FoFHalo() = default;
            FoFHalo(size_t _id) { id = _id; }

            /// @brief Add a new particle to the FoF group. We update the center of mass
            /// position to be that of the particles added so far.
            /// @param[in] particle The particle we are to add to the halo
            /// @param[in] periodic (Optional) Is the box periodic?
            void add(T & particle, bool periodic = true) {
                const double _mass = FML::PARTICLE::GetMass(particle);
                static_assert(FML::PARTICLE::has_get_pos<T>());
                const auto * _pos = FML::PARTICLE::GetPos(particle);

                // Sanity check
                assert_mpi(_mass > 0.0, "FoFBinning::add::Particle mass is negative or zero, this is not normal!");

#ifdef STORE_MEMBERS_PTRS
                // If you want access to all particles in finalize
                member_particles.push_back(&particle);
#endif

                // Initialize
                if (np == 0) {
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] = 0.0;
                        vel[idim] = 0.0;
                        vel_rms[idim] = 0.0;
                    }
                    mass = 0.0;
                }

                // Update center of mass
                std::array<double, NDIM> dx;
                double weigth = _mass / (mass + _mass);
                for (int idim = 0; idim < NDIM; idim++) {
                    dx[idim] = _pos[idim] - pos[idim];
                    if (periodic) {
                        if (dx[idim] < -0.5)
                            dx[idim] += 1;
                        if (dx[idim] >= 0.5)
                            dx[idim] -= 1;
                    }
                    pos[idim] += dx[idim] * weigth;
                    if (periodic) {
                        if (pos[idim] < 0.0)
                            pos[idim] += 1;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1;
                    }
                }

                // Add velocity if particle has velocity
                if constexpr (FML::PARTICLE::has_get_vel<T>()) {
                    auto _vel = FML::PARTICLE::GetVel(particle);
                    for (int idim = 0; idim < NDIM; idim++) {
                        vel[idim] = vel[idim] * (1 - weigth) + _vel[idim] * weigth;
                        vel_rms[idim] = vel_rms[idim] * (1 - weigth) + _vel[idim] * _vel[idim] * weigth;
                    }
                }

                // Update this last
                np++;
                mass += _mass;
            }

            /// @brief This method is called after all particles has been added to the halo
            /// and we can do any normalization etc we want.
            /// @param[in] periodic (Optional) Is the box periodic?
            void finalize([[maybe_unused]] bool periodic = true) {

                // Veloity dispersion. v_rms contains <v^2> at this point
                for (int idim = 0; idim < NDIM; idim++)
                    vel_rms[idim] = std::sqrt(vel_rms[idim] - vel[idim] * vel[idim]);

#ifdef STORE_MEMBERS_PTRS
                // Free up memory
                member_particles = std::vector<T *>();
#endif
            }
        };

    } // namespace FOF
} // namespace FML
#endif
