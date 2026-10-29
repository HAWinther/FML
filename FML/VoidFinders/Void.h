#ifndef FML_VOIDFINDERS_VOID_HEADER
#define FML_VOIDFINDERS_VOID_HEADER

#include <FML/Global/MPICounts.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/VoidFinders/ZoneMerging.h>
#include <functional>

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
            /// With zone merging: the id of the void this void is part of (-1 for a top-level void) and the number of
            /// zones in it. Without merging every void is one zone with parent -1
            long long int parent{-1};
            long long int nzones{1};

            /// delta_min (delta_max for clusters)
            double delta_min() const { return density_min - 1.0; }
            /// delta_avg
            double delta_avg() const { return density_avg - 1.0; }
            /// The lambda parameter delta_avg R^1.2 (R^1.6 for clusters) with R in the units of the box size given
            double lambda(double boxsize) const {
                return delta_avg() * std::pow(radius * boxsize, cluster ? 1.6 : 1.2);
            }
        };

        /// Sort a catalogue in increasing order of minimum density (decreasing for clusters) and set the ids.
        /// Ties are broken by the position of the minimum and the number of zones (larger first) so the order is
        /// reproducible. On input parent must be the index of the parent in the vector (or -1); it is converted to
        /// the new ids
        inline void sort_catalogue(std::vector<Void> & voids) {
            for (size_t i = 0; i < voids.size(); i++)
                voids[i].id = (long long int)(i);
            std::vector<Void> sorted(voids);
            std::sort(sorted.begin(), sorted.end(), [](const Void & a, const Void & b) {
                if (a.density_min != b.density_min)
                    return a.cluster ? a.density_min > b.density_min : a.density_min < b.density_min;
                if (a.pos_min != b.pos_min)
                    return a.pos_min < b.pos_min;
                return a.nzones > b.nzones;
            });
            std::vector<long long int> new_id(voids.size());
            for (size_t i = 0; i < sorted.size(); i++)
                new_id[sorted[i].id] = (long long int)(i);
            for (size_t i = 0; i < sorted.size(); i++) {
                sorted[i].id = (long long int)(i);
                if (sorted[i].parent >= 0)
                    sorted[i].parent = new_id[sorted[i].parent];
            }
            voids = std::move(sorted);
        }

        /// Make a catalogue from merged zones: voids that pass the cuts, parent = index in the result
        /// (the parent of a void that passes the cuts always passes them too). density_avg is mass/volume
        /// (in units of mean_density) if use_mass, else the mean of sum_density over the members
        inline std::vector<Void> catalogue_from_merged_zones(const std::vector<MergedVoid> & merged,
                                                             bool find_clusters,
                                                             double mean_density,
                                                             bool use_mass,
                                                             std::function<bool(const Void &)> pass_cuts) {
            std::vector<Void> voids;
            std::vector<long long int> index(merged.size(), -1);
            for (size_t a = 0; a < merged.size(); a++) {
                const auto & m = merged[a];
                const auto & z = m.sum;
                if (z.n == 0)
                    continue;
                Void v;
                v.cluster = find_clusters;
                v.density_min = (find_clusters ? 1.0 / z.quantity_core : z.quantity_core) / mean_density;
                v.density_ratio = m.quantity_leak / z.quantity_core;
                v.volume = z.volume;
                v.radius = std::cbrt(3.0 * z.volume / (4.0 * M_PI));
                v.density_avg = use_mass ? z.mass / z.volume / mean_density : z.sum_density / double(z.n);
                v.nmembers = z.n;
                v.nzones = m.nzones;
                v.parent = m.parent_zone; // zone index for now
                for (int j = 0; j < 3; j++) {
                    v.pos[j] = z.pos_center[j];
                    v.pos_min[j] = z.pos_min[j];
                    double x = z.weight > 0.0 ? z.pos_min[j] + z.weighted_dx[j] / z.weight : z.pos_min[j];
                    x -= std::floor(x);
                    v.pos_barycenter[j] = x >= 1.0 ? 0.0 : x;
                }
                if (not pass_cuts(v))
                    continue;
                index[a] = (long long int)(voids.size());
                voids.push_back(v);
            }
            for (auto & v : voids)
                v.parent = v.parent >= 0 ? index[v.parent] : -1;
            sort_catalogue(voids);
            return voids;
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
               << " density_ratio x_min y_min z_min x_barycenter y_barycenter z_barycenter nmembers nzones parent\n";
            fp << std::setprecision(8);
            for (auto & v : voids) {
                fp << v.id << " " << v.pos[0] * boxsize << " " << v.pos[1] * boxsize << " " << v.pos[2] * boxsize << " "
                   << v.radius * boxsize << " " << v.delta_min() << " " << v.delta_avg() << " " << v.lambda(boxsize)
                   << " " << v.density_ratio << " " << v.pos_min[0] * boxsize << " " << v.pos_min[1] * boxsize << " "
                   << v.pos_min[2] * boxsize << " " << v.pos_barycenter[0] * boxsize << " "
                   << v.pos_barycenter[1] * boxsize << " " << v.pos_barycenter[2] * boxsize << " " << v.nmembers << " "
                   << v.nzones << " " << v.parent << "\n";
            }
        }

        /// Send the catalogue on task 0 to all tasks
        inline void broadcast_catalogue([[maybe_unused]] std::vector<Void> & voids) {
#ifdef USE_MPI
            FML::COMMUNICATION::Count n = voids.size();
            MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, FML::communicator);
            FML::COMMUNICATION::require_count(n <= voids.max_size(),
                                              "[void catalogue] Broadcast exceeds vector capacity");
            const size_t bytes =
                FML::COMMUNICATION::checked_bytes(n, sizeof(Void), std::numeric_limits<std::ptrdiff_t>::max());
            voids.resize(size_t(n));
            auto *data = reinterpret_cast<char *>(voids.data());
            for (size_t offset = 0; offset < bytes;) {
                const int count = int(std::min(size_t(1) << 28, bytes - offset));
                MPI_Bcast(data + offset, count, MPI_BYTE, 0, FML::communicator);
                offset += size_t(count);
            }
#endif
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
