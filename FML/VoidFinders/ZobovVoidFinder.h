#ifndef FML_VOIDFINDERS_ZOBOV_HEADER
#define FML_VOIDFINDERS_ZOBOV_HEADER

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/PeriodicDelaunay.h>
#include <FML/VoidFinders/Void.h>

namespace FML {
    namespace VOIDFINDER {

        /// Options for ZobovVoidFinder (defaults as in Revolver)
        struct ZobovOptions {
            /// How large part of the neighbor domains we include in the tessellation (MPI)
            double buffer_fraction{0.5};
            /// How many guard points we add (fraction of the particles) to speed up the tessellation (MPI)
            double random_fraction{0.3};
            /// Find overdensities (clusters) instead of voids
            bool find_clusters{false};
            /// Voids: the minimum density must be below this (in units of the mean)
            double min_dens_cut{1.0};
            /// Clusters: the maximum density must be above this (in units of the mean)
            double max_dens_cut{1.0};
            /// Minimum number of particles in a void/cluster
            long long int min_num_particles{5};
            /// How to merge the zones into voids. None: every zone is a void (Revolver). Threshold: merge zones that
            /// are linked below link_density_threshold (VIDE). Full: the full ZOBOV hierarchy. With merging the
            /// catalogue has all the voids in the hierarchy; parent is the id of the void a void is part of
            /// (-1 for the top-level voids)
            ZoneMerging merging{ZoneMerging::None};
            /// Link density (in units of the mean) below which zones are merged (above for clusters, which need a
            /// value > 1)
            double link_density_threshold{0.2};
        };

        namespace detail {
            // Internal particle with what the watershed needs
            struct ZobovParticle {
                double pos[3];
                double mass{1.0};
                double volume{0.0};
                constexpr int get_ndim() const { return 3; }
                double * get_pos() { return pos; }
                double get_mass() const { return mass; }
                double get_volume() const { return volume; }
                void set_volume(double v) { volume = v; }
            };

            // What we compute for each watershed basin (zone)
            struct ZobovZone {
                double pos_min[3];
                double pos_circ[3];
                double pos_barycenter[3];
                double volume{0.0};
                double mass{0.0};
                double quantity_min{std::numeric_limits<double>::infinity()};
                double quantity_leak{std::numeric_limits<double>::infinity()};
                long long int n{0};

                void init(double * pos) {
                    for (int i = 0; i < 3; i++) {
                        pos_min[i] = pos_circ[i] = pos[i];
                        pos_barycenter[i] = 0.0;
                    }
                }
                void set_pos_circ(const double * pos) {
                    for (int i = 0; i < 3; i++)
                        pos_circ[i] = pos[i];
                }
                void set_leak_quantity(double q) { quantity_leak = q; }
                void add_particle(ZobovParticle * p, double quantity) {
                    for (int i = 0; i < 3; i++) {
                        double dx = p->pos[i] - pos_min[i];
                        dx -= std::round(dx);
                        pos_barycenter[i] += dx * p->volume;
                    }
                    volume += p->volume;
                    mass += p->mass;
                    quantity_min = std::min(quantity_min, quantity);
                    n++;
                }
                void merge(const ZobovZone & other) {
                    for (int i = 0; i < 3; i++)
                        pos_barycenter[i] += other.pos_barycenter[i];
                    volume += other.volume;
                    mass += other.mass;
                    quantity_min = std::min(quantity_min, other.quantity_min);
                    n += other.n;
                }
                void finalize() {
                    for (int i = 0; i < 3; i++) {
                        double x = volume > 0.0 ? pos_min[i] + pos_barycenter[i] / volume : pos_min[i];
                        x -= std::floor(x);
                        pos_barycenter[i] = x >= 1.0 ? 0.0 : x;
                    }
                }
            };
        } // namespace detail

        //========================================================================================
        /// The ZOBOV void finder as used in Revolver. We make a periodic Delaunay tessellation of the particles,
        /// compute the Voronoi densities (mass / Voronoi volume), and every particle is linked to its lowest density
        /// Delaunay neighbor. A zone (watershed basin) is all the particles that end up at the same density minimum.
        /// As in Revolver the zones are not merged: every zone that passes the cuts (minimum density below
        /// min_dens_cut, at least min_num_particles particles) is a void.
        ///
        /// For each void: the center is the circumcenter of the minimum and its 3 lowest density mutually adjacent
        /// neighbors, the volume is the sum of the Voronoi volumes, the average density is mass/volume, and the
        /// density ratio is the lowest link density to a neighboring zone (a link between two neighboring
        /// particles has the larger of the two densities) divided by the minimum density.
        ///
        /// With find_clusters we run the same on 1/density to find the overdensities.
        ///
        /// The particles must be in [0,1)^3 and inside the local domain (e.g. MPIParticles). The particles only need a
        /// get_pos method (get_mass is used if it exists). The result is the same on all tasks.
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] options The options (cuts, buffer size, voids or clusters)
        ///
        //========================================================================================
        template <class T>
        std::vector<Void> ZobovVoidFinder(T * p, size_t NumPart, const ZobovOptions & options = ZobovOptions()) {
            static_assert(CGAL_NDIM == 3, "ZobovVoidFinder is only implemented in 3D");
            assert_mpi(FML::PARTICLE::GetNDIM(T()) == 3, "[ZobovVoidFinder] Only implemented in 3D");
            // Clusters merge across links above the threshold: the void default (0.2) would merge everything
            assert_mpi(not(options.find_clusters and options.merging == ZoneMerging::Threshold and
                           options.link_density_threshold <= 1.0),
                       "[ZobovVoidFinder] Clusters with ZoneMerging::Threshold need link_density_threshold > 1 (in "
                       "units of the mean; the default 0.2 is for voids)");

            // Copy what we need
            std::vector<detail::ZobovParticle> parts(NumPart);
            double total_mass = 0.0;
            for (size_t i = 0; i < NumPart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(p[i]);
                for (int j = 0; j < 3; j++)
                    parts[i].pos[j] = double(pos[j]);
                if constexpr (FML::PARTICLE::has_get_mass<T>())
                    parts[i].mass = FML::PARTICLE::GetMass(p[i]);
                total_mass += parts[i].mass;
            }
            FML::SumOverTasks(&total_mass);
            assert_mpi(total_mass > 0.0, "[ZobovVoidFinder] No particles");
            const double mean_density = total_mass; // The box has volume 1

            if (FML::ThisTask == 0)
                std::cout << "[ZobovVoidFinder] Finding " << (options.find_clusters ? "clusters" : "voids") << "\n";

            // The watershed. For voids the quantity is mass/volume, for clusters volume/mass
            std::vector<detail::ZobovZone> zones;
            std::vector<FML::TRIANGULATION::WatershedLink> links;
            FML::TRIANGULATION::WatershedDensity(parts.data(),
                                                 NumPart,
                                                 zones,
                                                 options.buffer_fraction,
                                                 options.random_fraction,
                                                 options.find_clusters,
                                                 &links);
            parts = std::vector<detail::ZobovParticle>();

            // Merge the zones and make the catalogue (the zones are only on task 0)
            std::vector<Void> voids;
            if (FML::ThisTask == 0) {
                std::vector<Zone> zz(zones.size());
                for (size_t a = 0; a < zones.size(); a++) {
                    const auto & z = zones[a];
                    auto & y = zz[a];
                    y.quantity_core = z.quantity_min;
                    y.volume = z.volume;
                    y.mass = z.mass;
                    y.weight = z.volume;
                    y.n = z.n;
                    for (int j = 0; j < 3; j++) {
                        y.pos_min[j] = z.pos_min[j];
                        y.pos_center[j] = z.pos_circ[j];
                        double dx = z.pos_barycenter[j] - z.pos_min[j];
                        dx -= std::round(dx);
                        y.weighted_dx[j] = dx * z.volume;
                    }
                }
                std::vector<ZoneLink> zlinks(links.size());
                for (size_t k = 0; k < links.size(); k++)
                    zlinks[k] = {links[k].a, links[k].b, links[k].quantity};
                links = std::vector<FML::TRIANGULATION::WatershedLink>();
                const double threshold_quantity = options.find_clusters ?
                                                      1.0 / (options.link_density_threshold * mean_density) :
                                                      options.link_density_threshold * mean_density;
                const auto merged = merge_zones(zz, zlinks, options.merging, threshold_quantity);
                voids =
                    catalogue_from_merged_zones(merged, options.find_clusters, mean_density, true, [&](const Void & v) {
                        const bool pass_density = options.find_clusters ? v.density_min > options.max_dens_cut :
                                                                          v.density_min < options.min_dens_cut;
                        return pass_density and v.nmembers >= options.min_num_particles;
                    });
                long long int ntop = 0;
                for (auto & v : voids)
                    ntop += v.parent < 0;
                std::cout << "[ZobovVoidFinder] " << zones.size() << " zones, " << voids.size()
                          << (options.find_clusters ? " clusters" : " voids") << " pass the cuts (" << ntop
                          << " top-level)\n";
            }
            broadcast_catalogue(voids);
            return voids;
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
