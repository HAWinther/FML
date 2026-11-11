#ifndef FML_ANALYSIS_VOIDS_H
#define FML_ANALYSIS_VOIDS_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/Output.h>
#include <FML/Global/Global.h>
#include <FML/VoidFinders/SphericalVoidFinder.h>
#include <FML/VoidFinders/Void.h>
#include <FML/VoidFinders/VoxelVoidFinder.h>
#ifdef USE_CGAL
#include <FML/VoidFinders/ZobovVoidFinder.h>
#endif

#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
// Void catalogues of a particle set with FML/VoidFinders: the voxel watershed (Revolver, Nadathur et al. 2019, with
// VIDE zone merging, Sutter et al. 2015), spherical underdensities (Padilla, Ceccarelli & Lambas 2005) and ZOBOV
// (Neyrinck 2008, needs CGAL). The catalogues are the same on every task; the files are written by task 0:
// voids_z<z>.txt, voids_spherical_z<z>.txt and voids_zobov_z<z>.txt.
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        /// The density grid and watershed of the voxel finder (also the grid of the spherical finder and the zone
        /// merging of ZOBOV). Keys void_<name>
        struct VoidSettings {
            static constexpr const char * prefix = "void_";
            int nmesh;
            double smoothing_scale;
            std::string smoothing_method;
            std::string density_assignment_method;
            double min_density;
            std::string merging;
            double link_density_threshold;

            VoidSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("nmesh", nmesh, 0, "Density grid size (0: the Revolver choice); divisible by the number of tasks");
                f("smoothing_scale", smoothing_scale, -1.0, "Smoothing scale (box units; negative: the mean separation)");
                f("smoothing_method", smoothing_method, std::string("revolver"), "revolver or gaussian");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method");
                f("min_density", min_density, 1.0, "Keep voids with minimum density below this (mean units)");
                f("merging", merging, std::string("none"), "Zone merging: none, threshold or full");
                f("link_density_threshold", link_density_threshold, 0.2, "Density threshold of the zone merging");
            }

            void validate() const {
                if (nmesh < 0 or (nmesh > 0 and (nmesh < FML::NTasks or nmesh % FML::NTasks != 0)))
                    throw std::runtime_error("void_nmesh must be 0 or a multiple of the number of MPI tasks");
                if (not std::isfinite(smoothing_scale) or not(min_density > 0.0) or
                    not(link_density_threshold > 0.0))
                    throw std::runtime_error("Invalid void_smoothing_scale, void_min_density or "
                                             "void_link_density_threshold");
                if (smoothing_method != "revolver" and smoothing_method != "gaussian")
                    throw std::runtime_error("void_smoothing_method must be revolver or gaussian");
                if (merging != "none" and merging != "threshold" and merging != "full")
                    throw std::runtime_error("void_merging must be none, threshold or full");
            }

            FML::VOIDFINDER::ZoneMerging zone_merging() const {
                if (merging == "threshold")
                    return FML::VOIDFINDER::ZoneMerging::Threshold;
                if (merging == "full")
                    return FML::VOIDFINDER::ZoneMerging::Full;
                return FML::VOIDFINDER::ZoneMerging::None;
            }

            FML::VOIDFINDER::VoxelOptions voxel_options() const {
                FML::VOIDFINDER::VoxelOptions o;
                o.Nmesh = nmesh;
                o.smoothing_scale = smoothing_scale;
                o.smoothing_method = smoothing_method;
                o.density_assignment_method = density_assignment_method;
                o.min_dens_cut = min_density;
                o.link_density_threshold = link_density_threshold;
                o.merging = zone_merging();
                return o;
            }
        };

        /// Spherical underdensities (on the void_ grid). Keys spherical_void_<name>
        struct SphericalVoidSettings {
            static constexpr const char * prefix = "spherical_void_";
            double delta;
            double candidate_density_cut;
            double max_radius_mean_separation;
            double min_radius_mean_separation;
            int recentering_steps;
            double max_overlap;

            SphericalVoidSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("delta", delta, -0.8, "The integrated density contrast of a void (-1 < delta < 0)");
                f("candidate_density_cut", candidate_density_cut, 0.5, "Centers: grid cells below this density");
                f("max_radius_mean_separation", max_radius_mean_separation, 8.0, "Largest radius (mean separations)");
                f("min_radius_mean_separation", min_radius_mean_separation, 2.0, "Smallest radius (mean separations)");
                f("recentering_steps", recentering_steps, 0, "Recentering iterations");
                f("max_overlap", max_overlap, 0.0, "Two voids overlap if their distance is below (1 - max_overlap)(R1 + R2); largest kept first");
            }

            void validate() const {
                if (not(delta > -1.0 and delta < 0.0) or not std::isfinite(candidate_density_cut) or
                    not(max_radius_mean_separation > 0.0) or not(min_radius_mean_separation > 0.0) or
                    recentering_steps < 0 or not(max_overlap >= 0.0 and max_overlap < 1.0))
                    throw std::runtime_error("Invalid spherical_void_* settings");
            }
        };

        /// ZOBOV (the zone merging from the void_ settings). Keys zobov_<name>
        struct ZobovSettings {
            static constexpr const char * prefix = "zobov_";
            double buffer_fraction;
            double random_fraction;
            double min_density_cut;
            int min_particles;

            ZobovSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("buffer_fraction", buffer_fraction, 0.5, "Buffer copied from the neighbor tasks (fraction of a slab)");
                f("random_fraction", random_fraction, 0.3, "Guard points (fraction)");
                f("min_density_cut", min_density_cut, 1.0, "Keep voids with minimum density below this (mean units)");
                f("min_particles", min_particles, 5, "Minimum number of particles in a void");
            }

            void validate() const {
                if (not(buffer_fraction >= 0.0) or not(random_fraction >= 0.0) or not(min_density_cut > 0.0) or
                    min_particles < 1)
                    throw std::runtime_error("Invalid zobov_* settings");
            }
        };

        template <class P>
        std::vector<FML::VOIDFINDER::Void> find_voxel_voids(P * part, size_t npart, const VoidSettings & settings) {
            settings.validate();
            return FML::VOIDFINDER::VoxelVoidFinder(part, npart, settings.voxel_options());
        }

        template <class P>
        std::vector<FML::VOIDFINDER::Void> find_spherical_voids(P * part,
                                                                size_t npart,
                                                                const SphericalVoidSettings & settings,
                                                                const VoidSettings & grid) {
            settings.validate();
            grid.validate();
            FML::VOIDFINDER::SphericalOptions o;
            o.delta_threshold = settings.delta;
            o.candidate_density_cut = settings.candidate_density_cut;
            o.grid = grid.voxel_options();
            o.max_radius_in_mean_separation = settings.max_radius_mean_separation;
            o.min_radius_in_mean_separation = settings.min_radius_mean_separation;
            o.recentering_steps = settings.recentering_steps;
            o.max_overlap = settings.max_overlap;
            return FML::VOIDFINDER::SphericalVoidFinder(part, npart, o);
        }

#ifdef USE_CGAL
        template <class P>
        std::vector<FML::VOIDFINDER::Void> find_zobov_voids(P * part,
                                                            size_t npart,
                                                            const ZobovSettings & settings,
                                                            const VoidSettings & merging) {
            settings.validate();
            merging.validate();
            FML::VOIDFINDER::ZobovOptions o;
            o.buffer_fraction = settings.buffer_fraction;
            o.random_fraction = settings.random_fraction;
            o.min_dens_cut = settings.min_density_cut;
            o.min_num_particles = settings.min_particles;
            o.link_density_threshold = merging.link_density_threshold;
            o.merging = merging.zone_merging();
            return FML::VOIDFINDER::ZobovVoidFinder(part, npart, o);
        }
#endif

        /// A void catalogue (the same columns as FML::VOIDFINDER::write_catalogue; positions and radii in Mpc/h),
        /// written by task 0. Collective
        inline void write_void_catalogue(const std::string & filename,
                                         const std::vector<FML::VOIDFINDER::Void> & voids,
                                         const AnalysisContext & context,
                                         const std::string & header = "") {
            const double boxsize = context.boxsize;
            write_on_root(filename, [&](std::ostream & fp) {
                const bool cluster = not voids.empty() and voids[0].cluster;
                fp << header;
                fp << "# " << voids.size() << (cluster ? " clusters" : " voids") << ", boxsize " << boxsize
                   << ", redshift " << context.redshift() << "\n";
                fp << "# id x y z R " << (cluster ? "delta_max" : "delta_min") << " delta_avg "
                   << (cluster ? "lambda_c" : "lambda_v")
                   << " density_ratio x_min y_min z_min x_barycenter y_barycenter z_barycenter nmembers nzones "
                      "parent\n";
                fp << std::setprecision(8);
                for (auto & v : voids) {
                    fp << v.id << " " << v.pos[0] * boxsize << " " << v.pos[1] * boxsize << " " << v.pos[2] * boxsize
                       << " " << v.radius * boxsize << " " << v.delta_min() << " " << v.delta_avg() << " "
                       << v.lambda(boxsize) << " " << v.density_ratio << " " << v.pos_min[0] * boxsize << " "
                       << v.pos_min[1] * boxsize << " " << v.pos_min[2] * boxsize << " "
                       << v.pos_barycenter[0] * boxsize << " " << v.pos_barycenter[1] * boxsize << " "
                       << v.pos_barycenter[2] * boxsize << " " << v.nmembers << " " << v.nzones << " " << v.parent
                       << "\n";
                }
            });
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
