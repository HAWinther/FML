#ifndef FML_VOIDFINDERS_VOID_HEADER
#define FML_VOIDFINDERS_VOID_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <FML/Global/Global.h>

namespace FML {

    //========================================================================================
    /// Void finders for periodic boxes, following Revolver (Nadathur et al. 2019, github.com/seshnadathur/Revolver):
    ///
    /// ZobovVoidFinder: watershed on the Voronoi densities of the particles (ZOBOV, Neyrinck 2008), see
    /// ZobovVoidFinder.h. VoxelVoidFinder: watershed on a smoothed density field on a grid, see VoxelVoidFinder.h.
    ///
    /// As in Revolver the voids are the watershed basins (zones) without any merging. Both can also find
    /// overdense structures ("clusters") by running the same algorithm on 1/density.
    ///
    /// SphericalVoidFinder: spheres with a given integrated density contrast (e.g. Padilla et al. 2005), see
    /// SphericalVoidFinder.h.
    ///
    /// Positions, volumes and radii are in units of the box (positions in [0,1)); multiply by the box size.
    /// Densities are in units of the mean density.
    //========================================================================================

    namespace VOIDFINDER {

        /// A void (or cluster) from one of the void finders
        struct Void {
            /// Number in the catalogue (the catalogue is sorted by the minimum density, maximum for clusters)
            long long int id{0};
            /// The center: the circumcenter of the minimum and its 3 lowest density mutually adjacent neighbors (ZOBOV)
            /// or the position of the minimum (voxel)
            std::array<double, 3> pos{};
            /// The position of the minimum density particle/voxel (maximum for clusters)
            std::array<double, 3> pos_min{};
            /// The barycenter: weighted by the Voronoi volumes (ZOBOV) or by 1/density (voxel; density for clusters)
            std::array<double, 3> pos_barycenter{};
            /// The volume
            double volume{0.0};
            /// The effective radius (3 volume / 4 pi)^(1/3)
            double radius{0.0};
            /// The density of the minimum (maximum for clusters) in units of the mean
            double density_min{0.0};
            /// The average density in units of the mean (ZOBOV: mass/volume, voxel: mean over the voxels)
            double density_avg{0.0};
            /// The lowest density at which the void links to a neighbor, divided by density_min (ZOBOV's
            /// DensRatio, the significance of the void). For clusters: the same for 1/density
            double density_ratio{0.0};
            /// Number of particles (ZOBOV) or voxels (voxel) in the void
            long long int nmembers{0};
            /// Is this a cluster (overdensity) and not a void
            bool cluster{false};

            /// delta_min (delta_max for clusters)
            double delta_min() const { return density_min - 1.0; }
            /// delta_avg
            double delta_avg() const { return density_avg - 1.0; }
            /// The lambda parameter delta_avg R^1.2 (R^1.6 for clusters) with R in the units of the box size given
            double lambda(double boxsize) const { return delta_avg() * std::pow(radius * boxsize, cluster ? 1.6 : 1.2); }
        };

        /// Sort a catalogue in increasing order of minimum density (decreasing for clusters) and set the ids.
        /// Ties are broken by the position of the minimum so the order is reproducible
        inline void sort_catalogue(std::vector<Void> & voids) {
            std::sort(voids.begin(), voids.end(), [](const Void & a, const Void & b) {
                if (a.density_min != b.density_min)
                    return a.cluster ? a.density_min > b.density_min : a.density_min < b.density_min;
                return a.pos_min < b.pos_min;
            });
            for (size_t i = 0; i < voids.size(); i++)
                voids[i].id = (long long int)(i);
        }

        /// Write the catalogue to file (only task 0 writes). Distances are multiplied by boxsize.
        /// Columns as in Revolver followed by the extra info:
        /// id x y z R delta_min delta_avg lambda density_ratio x_min y_min z_min x_bary y_bary z_bary nmembers
        inline void write_catalogue(const std::string & filename, const std::vector<Void> & voids, double boxsize) {
            if (FML::ThisTask != 0)
                return;
            std::ofstream fp(filename);
            assert_mpi(fp.good(), ("[write_catalogue] Could not open " + filename).c_str());
            const bool cluster = not voids.empty() and voids[0].cluster;
            fp << "# " << voids.size() << (cluster ? " clusters" : " voids") << ", boxsize " << boxsize << "\n";
            fp << "# id x y z R " << (cluster ? "delta_max" : "delta_min") << " delta_avg "
               << (cluster ? "lambda_c" : "lambda_v")
               << " density_ratio x_min y_min z_min x_barycenter y_barycenter z_barycenter nmembers\n";
            fp << std::setprecision(8);
            for (auto & v : voids) {
                fp << v.id << " " << v.pos[0] * boxsize << " " << v.pos[1] * boxsize << " " << v.pos[2] * boxsize << " "
                   << v.radius * boxsize << " " << v.delta_min() << " " << v.delta_avg() << " " << v.lambda(boxsize)
                   << " " << v.density_ratio << " " << v.pos_min[0] * boxsize << " " << v.pos_min[1] * boxsize << " "
                   << v.pos_min[2] * boxsize << " " << v.pos_barycenter[0] * boxsize << " "
                   << v.pos_barycenter[1] * boxsize << " " << v.pos_barycenter[2] * boxsize << " " << v.nmembers
                   << "\n";
            }
        }

        /// Send the catalogue on task 0 to all tasks
        inline void broadcast_catalogue([[maybe_unused]] std::vector<Void> & voids) {
#ifdef USE_MPI
            long long int n = (long long int)(voids.size());
            MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, FML::communicator);
            voids.resize(size_t(n));
            const size_t chunk = size_t(1) << 20;
            for (size_t offset = 0; offset < voids.size(); offset += chunk) {
                const size_t count = std::min(chunk, voids.size() - offset);
                MPI_Bcast(voids.data() + offset, int(count * sizeof(Void)), MPI_BYTE, 0, FML::communicator);
            }
#endif
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
