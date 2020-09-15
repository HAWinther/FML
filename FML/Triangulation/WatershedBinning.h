#ifndef WATERSHEDBINNING_HEADER
#define WATERSHEDBINNING_HEADER

#include <vector>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {
    namespace TRIANGULATION {

        /// This is a general binning class for doing watershed based on [quantity] (typically volume)
        /// When we bin particles to the watershed basins we compute stuff based on what is in this class
        template <class T, int NDIM>
        struct WatershedBasin {
            double pos_min[NDIM];
            double pos_barycenter[NDIM];
            double volume{0.0};
            double volume_min{0.0};
            double density_min{0.0};
            double mass{0.0};
            double mean_quantity{0.0};
            int ningroup{0};

            // Initialize with the position of the minimum particle
            void init(double * pos) {
                for (int idim = 0; idim < NDIM; idim++) {
                    pos_min[idim] = pos[idim];
                    pos_barycenter[idim] = 0.0;
                }
                density_min = 1e100;
                volume_min = 1e100;
                volume = 0;
                mass = 0;
                mean_quantity = 0;
                ningroup = 0;
            }

            // Add data from a particle belonging to the group
            // We measure distances relative to the minimum point
            void add_particle(T * p, double quantity) {
                double curmass = 1.0;
                if constexpr(FML::PARTICLE::has_get_mass<T>()){
                  curmass = FML::PARTICLE::GetMass(*p);
                }
                double curvolume = 1.0;
                if constexpr(FML::PARTICLE::has_get_volume<T>()){
                  curvolume = FML::PARTICLE::GetVolume(*p);
                }

                auto *pos = FML::PARTICLE::GetPos(*p);
                for (int idim = 0; idim < NDIM; idim++) {
                    double dx = pos[idim] - pos_min[idim];
                    if (dx > 0.5)
                        dx -= 1.0;
                    if (dx < -0.5)
                        dx += 1.0;
                    // pos_barycenter[idim] += dx;
                    pos_barycenter[idim] += dx * curvolume;
                }
                if (curvolume < volume_min)
                    volume_min = curvolume;
                double d = curmass / curvolume;
                if (d < density_min)
                    density_min = d;
                volume += curvolume;
                mass += curmass;
                mean_quantity += quantity;
                ningroup++;
            }

            // Merge groups from a different task
            void merge(struct WatershedBasin & other) {
                if (other.ningroup == 0)
                    return;
                for (int idim = 0; idim < NDIM; idim++) {
                    assert(pos_min[idim] == other.pos_min[idim]);
                    pos_barycenter[idim] += other.pos_barycenter[idim];
                }
                volume_min = std::min(volume_min, other.volume_min);
                density_min = std::min(density_min, other.density_min);
                volume += other.volume;
                mass += other.mass;
                mean_quantity += other.mean_quantity;
                ningroup += other.ningroup;
            }

            // The barycenter is so far the sum of distances from the minimum point
            // so correct this and make sure the position is inside the box
            // Normalize the mean of quantity
            void finalize() {
                for (int idim = 0; idim < NDIM; idim++) {
                    double x = pos_min[idim] + pos_barycenter[idim] / volume;
                    if (x < 0)
                        x += 1.0;
                    if (x >= 1.0)
                        x -= 1.0;
                    pos_barycenter[idim] = x;
                }
                mean_quantity /= double(ningroup);
            }
        };

    } // namespace TRIANGULATION
} // namespace FML
#endif
