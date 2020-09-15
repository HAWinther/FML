#ifndef FOFBINNING_HEADER
#define FOFBINNING_HEADER
#include <array>
#include <cassert>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {
    namespace FOF {

        //=========================================================================
        /// This class defines how to compute the halos from the particles
        ///
        /// NB: no things that needs to be dynamically allocated in this class as currently
        /// we simply use sizeof(FoFHalo) to get the memory layout when communicating
        /// If you want to compute more things then make your own class like this one
        /// and supply as template parameter to the FriendsOfFriends method.
        ///
        /// The computation below is incremental: we add up the mean values as we go
        /// along instead of doing it in the end.
        ///
        /// NB: must have public bool merged for book-keeping when merging
        ///
        //=========================================================================
        template <class T, int NDIM>
        class FoFHalo {
          public:
            size_t id{0};
            size_t np{0};
            double mass{0.0};
            std::array<double, NDIM> pos;
            std::array<double, NDIM> vel;
            double vel2{0.0}; // <v^2>
            bool shared{false};
            bool merged{false};

            FoFHalo() = default;
            FoFHalo(size_t _id, bool _shared) {
                id = _id;
                shared = _shared;
            }

            // Add a new particle to the group
            // Update data
            void add(T & particle, bool periodic) {
                double _mass = FML::PARTICLE::GetMass(particle);
                static_assert(FML::PARTICLE::has_get_pos<T>());
                auto * _pos = FML::PARTICLE::GetPos(particle);

                // Initialize
                if (np == 0) {
                    for (int idim = 0; idim < NDIM; idim++) {
                        pos[idim] = 0.0;
                        vel[idim] = 0.0;
                    }
                    mass = 0.0;
                    vel2 = 0.0;
                }

                // Update center of mass
                std::array<double, NDIM> dx;
                double v2 = 0;
                for (int idim = 0; idim < NDIM; idim++) {
                    dx[idim] = _pos[idim] - pos[idim];
                    if (periodic) {
                        if (dx[idim] < -0.5)
                            dx[idim] += 1;
                        if (dx[idim] >= 0.5)
                            dx[idim] -= 1;
                    }
                    pos[idim] += dx[idim] * _mass / (mass + _mass);
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
                        vel[idim] = (vel[idim] * mass + _vel[idim] * _mass) / (mass + _mass);
                        v2 += vel[idim] * vel[idim];
                    }
                }

                // Update <v^2>
                vel2 = (vel2 * mass + _mass * v2) / (mass + _mass);
                np++;
                mass += _mass;
            }

            // Merge the groups
            void merge(FoFHalo & g, bool periodic) {
                if (g.np == 0)
                    return;
                if (np == 0) {
                    assert(false); // Should not happen
                }

                // Merge the two centers of mass
                std::array<double, NDIM> dx;
                for (int idim = 0; idim < NDIM; idim++) {
                    dx[idim] = g.pos[idim] - pos[idim];
                    if (periodic) {
                        if (dx[idim] < -0.5)
                            dx[idim] += 1;
                        if (dx[idim] >= 0.5)
                            dx[idim] -= 1;
                    }
                    // Update center of mass
                    pos[idim] += dx[idim] * g.mass / double(mass + g.mass);
                    if (periodic) {
                        if (pos[idim] < 0.0)
                            pos[idim] += 1;
                        if (pos[idim] >= 1.0)
                            pos[idim] -= 1;
                    }
                    // Update COM velocity
                    vel[idim] = (vel[idim] * mass + g.vel[idim] * g.mass) / (mass + g.mass);
                }
                // Update <v^2>
                vel2 = (vel2 * mass + g.vel2 * g.mass) / (mass + g.mass);
                np += g.np;
                g.np = 0;
            }
        };

        //=========================================================================
        /// A gridcell in a grid used for speeding up the linking of particles
        /// The ID of the particles is their position in the particle list
        ///
        //=========================================================================
        class FoFCells {
          public:
            int np{0};
            std::vector<size_t> ParticleIndex{};
            FoFCells() = default;
        };

    } // namespace FOF
} // namespace FML
#endif
