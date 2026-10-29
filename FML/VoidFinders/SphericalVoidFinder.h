#ifndef FML_VOIDFINDERS_SPHERICAL_HEADER
#define FML_VOIDFINDERS_SPHERICAL_HEADER

#include <FML/Global/MPICounts.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Global/PeriodicSlabExchange.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/VoidFinders/Void.h>
#include <FML/VoidFinders/VoxelVoidFinder.h>

namespace FML {
    namespace VOIDFINDER {

        /// Options for SphericalVoidFinder
        struct SphericalOptions {
            /// The integrated density contrast inside a void: Delta(<R) = threshold
            double delta_threshold{-0.8};
            /// Candidate centers are the grid cells with (smoothed) density below this, in units of the mean
            double candidate_density_cut{0.5};
            /// The grid used for the candidate centers (Nmesh, smoothing scale and method, density assignment).
            /// The defaults are as for the voxel void finder (Revolver)
            VoxelOptions grid{};
            /// Largest void radius we look for, in units of the mean particle separation N^(-1/3)
            double max_radius_in_mean_separation{8.0};
            /// Smallest void radius we keep, in units of the mean particle separation
            double min_radius_in_mean_separation{2.0};
            /// Number of random walk steps for moving the center to get a larger sphere (0: keep the grid cell).
            /// A step moves the center by up to R/4 and is accepted if the sphere gets larger. The center stays within
            /// half the largest radius from the candidate
            int recentering_steps{0};
            /// Overlap allowed between voids: two voids overlap if their distance is less than
            /// (1 - max_overlap) (R1 + R2). We keep the largest voids first
            double max_overlap{0.0};
        };

        namespace detail {
            struct PosMass {
                double x[3];
                double m;
            };
            struct SphereCandidate {
                double center[3];
                double candidate[3];
                double radius;
                double mass_inside;
                double candidate_density;
                long long int n_inside;
                long long int index;
            };
        } // namespace detail

        //========================================================================================
        /// Spherical underdensity void finder (e.g. Padilla et al. 2005, Ruiz et al. 2015). Candidate centers are the
        /// cells of a smoothed density grid with density below candidate_density_cut. Around each candidate we grow a
        /// sphere until the density inside reaches (1 + delta_threshold) times the mean: the radius is the distance
        /// to the first particle that would make the density inside larger than this (computed with the exact
        /// particle counts, using the masses if the particles have get_mass). Optionally we move the center with a
        /// random walk to get a larger sphere. Then we go through the spheres from the largest to the smallest and
        /// keep the ones that do not overlap with the ones we already kept, and the ones larger than the minimum
        /// radius.
        ///
        /// For each void: pos = the center, pos_min = the candidate cell, pos_barycenter = the center, density_min =
        /// the density of the candidate cell, density_avg = the density inside the sphere (<= 1 + delta_threshold),
        /// nmembers = the number of particles inside, density_ratio = 0 (not defined). The catalogue is sorted by
        /// radius (largest first).
        ///
        /// The particles must be in [0,1)^3 and inside the local domain (e.g. MPIParticles). The result is the
        /// same on all tasks (for a given grid). Smoothing the candidate grid requires FFTW (USE_FFTW).
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] options The options
        ///
        //========================================================================================
        template <class T>
        std::vector<Void>
        SphericalVoidFinder(T * p, size_t NumPart, const SphericalOptions & options = SphericalOptions()) {

            using detail::PosMass;
            using detail::SphereCandidate;
            assert_mpi(FML::PARTICLE::GetNDIM(T()) == 3, "[SphericalVoidFinder] Only implemented in 3D");
            assert_mpi(options.delta_threshold > -1.0 and options.delta_threshold < 0.0,
                       "[SphericalVoidFinder] delta_threshold must be in (-1,0)");

            // Particles (positions and masses)
            std::vector<PosMass> parts(NumPart);
            double total_mass = 0.0;
            for (size_t i = 0; i < NumPart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(p[i]);
                for (int j = 0; j < 3; j++)
                    parts[i].x[j] = double(pos[j]);
                parts[i].m = 1.0;
                if constexpr (FML::PARTICLE::has_get_mass<T>())
                    parts[i].m = FML::PARTICLE::GetMass(p[i]);
                total_mass += parts[i].m;
            }
            long long int NumPartTot = (long long int)(NumPart);
            FML::SumOverTasks(&NumPartTot);
            FML::SumOverTasks(&total_mass);
            assert_mpi(NumPartTot > 0 and total_mass > 0.0, "[SphericalVoidFinder] No particles");
            const double mean_density = total_mass; // The box has volume 1
            const double mean_separation = std::cbrt(1.0 / double(NumPartTot));
            // We need rmax well below half the box for the periodic distances (and the buffers)
            const double rmax = std::min(0.2, options.max_radius_in_mean_separation * mean_separation);
            const double rmin = options.min_radius_in_mean_separation * mean_separation;
            if (FML::ThisTask == 0 and rmax < options.max_radius_in_mean_separation * mean_separation)
                std::cout << "[SphericalVoidFinder] The max radius is limited to 0.2 times the box size\n";
            // max_overlap < 0 would need overlap checks beyond the neighbouring cells of the overlap hash
            assert_mpi(options.max_overlap >= 0.0 and options.max_overlap < 1.0,
                       "[SphericalVoidFinder] max_overlap must be in [0, 1)");
            assert_mpi(rmin <= rmax, "[SphericalVoidFinder] The min radius is larger than the max radius");
            const double max_drift = options.recentering_steps > 0 ? 0.5 * rmax : 0.0;
            const double threshold = (1.0 + options.delta_threshold) * mean_density * 4.0 * M_PI / 3.0;

            if (FML::ThisTask == 0)
                std::cout << "[SphericalVoidFinder] Delta = " << options.delta_threshold
                          << " candidate density cut = " << options.candidate_density_cut << " radius range [" << rmin
                          << " , " << rmax << "] (box units)\n";

            // Candidate centers from the density grid
            FML::GRID::FFTWGrid<3> grid;
            void_density_field(p, NumPart, options.grid, grid);
            const int Nmesh = grid.get_nmesh();
            const auto local_nx = grid.get_local_nx();
            const auto local_x_start = grid.get_local_x_start();
            std::vector<SphereCandidate> candidates;
            for (int ix = 0; ix < local_nx; ix++)
                for (int iy = 0; iy < Nmesh; iy++)
                    for (int iz = 0; iz < Nmesh; iz++) {
                        const double d = grid.get_real({ix, iy, iz});
                        if (d >= options.candidate_density_cut)
                            continue;
                        SphereCandidate c{};
                        c.candidate[0] = double(local_x_start + ix) / Nmesh;
                        c.candidate[1] = double(iy) / Nmesh;
                        c.candidate[2] = double(iz) / Nmesh;
                        for (int j = 0; j < 3; j++)
                            c.center[j] = c.candidate[j];
                        c.candidate_density = d;
                        c.index = ((local_x_start + ix) * Nmesh + iy) * (long long int)(Nmesh) + iz;
                        candidates.push_back(c);
                    }
            grid = FML::GRID::FFTWGrid<3>();
            if (local_nx > 0)
                assert_mpi(double(local_x_start) / Nmesh >= FML::xmin_domain - 1e-12 and
                               double(local_x_start + local_nx - 1) / Nmesh < FML::xmax_domain,
                           "[SphericalVoidFinder] The grid slab is not inside the particle domain");
            long long int ncandidates = (long long int)(candidates.size());
            FML::SumOverTasks(&ncandidates);
            if (FML::ThisTask == 0)
                std::cout << "[SphericalVoidFinder] " << ncandidates << " candidate centers on a " << Nmesh
                          << "^3 grid\n";

            // Get the particles within rmax (+ drift) of our domain from the other tasks
            const double width = FML::xmax_domain - FML::xmin_domain;
            const double buffer = rmax + max_drift + 1.0 / Nmesh;
            // Crossing a thin slab does not require replicating the whole box.
            // Route to every slab whose periodic buffer overlaps each owned particle.
            const bool everything = FML::NTasks == 1 or width + 2.0 * buffer >= 1.0;
            FML::COMMUNICATION::append_periodic_slab_ghosts(parts, buffer, [](const PosMass & q) { return q.x[0]; });

            // Cell list for the particles we have. Cells of size >= 2 mean separations. In x we only store
            // the cells covering our domain plus the buffer
            const int Nc = std::max(1, std::min(1024, int(1.0 / (2.0 * mean_separation))));
            const int cx0 = everything ? 0 : int(std::floor((FML::xmin_domain - buffer) * Nc));
            const int ncx = everything ? Nc : std::min(Nc, int(std::ceil((width + 2.0 * buffer) * Nc)) + 2);
            auto cell_x = [&](int cx) { return ((cx - cx0) % Nc + Nc) % Nc; };
            auto cell_of = [&](const double * x, int & lx, int & cy, int & cz) {
                int cx = int(x[0] * Nc);
                cy = std::min(Nc - 1, int(x[1] * Nc));
                cz = std::min(Nc - 1, int(x[2] * Nc));
                lx = cell_x(std::min(Nc - 1, cx));
            };
            const size_t ncells = size_t(ncx) * Nc * Nc;
            std::vector<size_t> offset(ncells + 1, 0);
            std::vector<size_t> cell_index(parts.size());
            for (size_t i = 0; i < parts.size(); i++) {
                int lx, cy, cz;
                cell_of(parts[i].x, lx, cy, cz);
                cell_index[i] = lx < ncx ? (size_t(lx) * Nc + cy) * Nc + cz : ncells;
                if (cell_index[i] < ncells)
                    offset[cell_index[i] + 1]++;
            }
            for (size_t c = 0; c < ncells; c++)
                offset[c + 1] += offset[c];
            std::vector<PosMass> sorted(offset[ncells]);
            {
                std::vector<size_t> fill(offset.begin(), offset.end() - 1);
                for (size_t i = 0; i < parts.size(); i++)
                    if (cell_index[i] < ncells)
                        sorted[fill[cell_index[i]]++] = parts[i];
            }
            parts = std::vector<PosMass>();
            cell_index = std::vector<size_t>();

            // The cells overlapping [x - r, x + r] in one dimension (each cell once)
            auto cell_range = [&](double x, double r, std::vector<int> & cells) {
                cells.clear();
                const int lo = int(std::floor((x - r) * Nc)), hi = int(std::floor((x + r) * Nc));
                if (hi - lo + 1 >= Nc) {
                    for (int c = 0; c < Nc; c++)
                        cells.push_back(c);
                } else {
                    for (int c = lo; c <= hi; c++)
                        cells.push_back((c % Nc + Nc) % Nc);
                }
            };

            // The radius of the sphere around a center (0 if not found within rmax)
            long long int nfailed_buffer = 0;
            std::vector<std::pair<double, double>> dist_mass;
            std::vector<int> range_x, range_y, range_z;
            auto sphere_radius = [&](const double * center, double & mass_inside, long long int & n_inside) -> double {
                double r = std::min(rmax, 3.0 * mean_separation);
                while (true) {
                    dist_mass.clear();
                    cell_range(center[0], r, range_x);
                    cell_range(center[1], r, range_y);
                    cell_range(center[2], r, range_z);
                    for (int cx : range_x) {
                        const int lx = cell_x(cx);
                        if (lx >= ncx) {
                            nfailed_buffer++;
                            continue;
                        }
                        for (int cy : range_y)
                            for (int cz : range_z) {
                                const size_t c = (size_t(lx) * Nc + cy) * Nc + cz;
                                for (size_t k = offset[c]; k < offset[c + 1]; k++) {
                                    double d2 = 0.0;
                                    for (int j = 0; j < 3; j++) {
                                        double dx = sorted[k].x[j] - center[j];
                                        dx -= std::round(dx);
                                        d2 += dx * dx;
                                    }
                                    if (d2 <= r * r)
                                        dist_mass.push_back({std::sqrt(d2), sorted[k].m});
                                }
                            }
                    }
                    std::sort(dist_mass.begin(), dist_mass.end());
                    double mass = 0.0;
                    for (size_t k = 0; k < dist_mass.size(); k++) {
                        const double d = dist_mass[k].first;
                        if (mass + dist_mass[k].second > threshold * d * d * d) {
                            mass_inside = mass;
                            n_inside = (long long int)(k);
                            return d;
                        }
                        mass += dist_mass[k].second;
                    }
                    if (r >= rmax)
                        return 0.0;
                    r = std::min(rmax, 2.0 * r);
                }
            };

            // Grow the spheres
            std::vector<SphereCandidate> spheres;
            long long int nfailed = 0;
            for (auto & c : candidates) {
                c.radius = sphere_radius(c.center, c.mass_inside, c.n_inside);
                if (c.radius > 0.0 and options.recentering_steps > 0) {
                    std::mt19937_64 rng(1234567ULL + 7919ULL * (unsigned long long)(c.index));
                    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
                    for (int step = 0; step < options.recentering_steps; step++) {
                        double u[3], u2;
                        do {
                            u2 = 0.0;
                            for (auto & ui : u) {
                                ui = uniform(rng);
                                u2 += ui * ui;
                            }
                        } while (u2 > 1.0);
                        double trial[3], drift2 = 0.0;
                        for (int j = 0; j < 3; j++) {
                            trial[j] = c.center[j] + 0.25 * c.radius * u[j];
                            double dx = trial[j] - c.candidate[j];
                            dx -= std::round(dx);
                            drift2 += dx * dx;
                            trial[j] -= std::floor(trial[j]);
                            if (trial[j] >= 1.0)
                                trial[j] = 0.0;
                        }
                        if (drift2 > max_drift * max_drift)
                            continue;
                        double mass_trial;
                        long long int n_trial;
                        const double r_trial = sphere_radius(trial, mass_trial, n_trial);
                        if (r_trial > c.radius) {
                            for (int j = 0; j < 3; j++)
                                c.center[j] = trial[j];
                            c.radius = r_trial;
                            c.mass_inside = mass_trial;
                            c.n_inside = n_trial;
                        }
                    }
                }
                if (c.radius == 0.0)
                    nfailed++;
                else if (c.radius >= rmin)
                    spheres.push_back(c);
            }
            candidates = std::vector<SphereCandidate>();
            FML::SumOverTasks(&nfailed);
            FML::SumOverTasks(&nfailed_buffer);
            if (FML::ThisTask == 0 and nfailed > 0)
                std::cout << "[SphericalVoidFinder] Warning: " << nfailed
                          << " candidates did not reach the density threshold within the max radius (increase "
                             "max_radius_in_mean_separation)\n";
            if (FML::ThisTask == 0 and nfailed_buffer > 0)
                std::cout << "[SphericalVoidFinder] Warning: " << nfailed_buffer
                          << " cell lookups outside of the particle buffer (results might be wrong)\n";

            // Gather the spheres on task 0
#ifdef USE_MPI
            if (FML::NTasks > 1) {
                spheres = FML::COMMUNICATION::gather_records_checked(spheres);
            }
#endif

            // Remove overlaps: largest first (ties by the candidate density and the cell index)
            std::vector<Void> voids;
            if (FML::ThisTask == 0) {
                std::sort(spheres.begin(), spheres.end(), [](const SphereCandidate & a, const SphereCandidate & b) {
                    if (a.radius != b.radius)
                        return a.radius > b.radius;
                    if (a.candidate_density != b.candidate_density)
                        return a.candidate_density < b.candidate_density;
                    return a.index < b.index;
                });
                const double rlargest = spheres.empty() ? rmax : spheres.front().radius;
                const int nh = std::max(1, int(1.0 / (2.0 * rlargest)));
                std::vector<std::vector<int>> hash(size_t(nh) * nh * nh);
                std::vector<SphereCandidate> kept;
                std::vector<int> hx, hy, hz;
                auto hash_range = [&](double x, std::vector<int> & cells) {
                    cells.clear();
                    const int c = std::min(nh - 1, int(x * nh));
                    for (int d = -1; d <= 1; d++) {
                        const int cc = ((c + d) % nh + nh) % nh;
                        if (std::find(cells.begin(), cells.end(), cc) == cells.end())
                            cells.push_back(cc);
                    }
                };
                for (auto & s : spheres) {
                    bool overlap = false;
                    hash_range(s.center[0], hx);
                    hash_range(s.center[1], hy);
                    hash_range(s.center[2], hz);
                    for (int a : hx) {
                        for (int b : hy) {
                            for (int c : hz) {
                                for (int k : hash[(size_t(a) * nh + b) * nh + c]) {
                                    double d2 = 0.0;
                                    for (int j = 0; j < 3; j++) {
                                        double dx = s.center[j] - kept[k].center[j];
                                        dx -= std::round(dx);
                                        d2 += dx * dx;
                                    }
                                    const double dmin = (1.0 - options.max_overlap) * (s.radius + kept[k].radius);
                                    if (d2 < dmin * dmin) {
                                        overlap = true;
                                        break;
                                    }
                                }
                                if (overlap)
                                    break;
                            }
                            if (overlap)
                                break;
                        }
                        if (overlap)
                            break;
                    }
                    if (overlap)
                        continue;
                    const int a = std::min(nh - 1, int(s.center[0] * nh));
                    const int b = std::min(nh - 1, int(s.center[1] * nh));
                    const int c = std::min(nh - 1, int(s.center[2] * nh));
                    hash[(size_t(a) * nh + b) * nh + c].push_back(int(kept.size()));
                    kept.push_back(s);
                }

                for (auto & s : kept) {
                    Void v;
                    for (int j = 0; j < 3; j++) {
                        v.pos[j] = v.pos_barycenter[j] = s.center[j];
                        v.pos_min[j] = s.candidate[j];
                    }
                    v.radius = s.radius;
                    v.volume = 4.0 * M_PI / 3.0 * s.radius * s.radius * s.radius;
                    v.density_min = s.candidate_density;
                    v.density_avg = s.mass_inside / v.volume / mean_density;
                    v.density_ratio = 0.0;
                    v.nmembers = s.n_inside;
                    v.id = (long long int)(voids.size());
                    voids.push_back(v);
                }
                std::cout << "[SphericalVoidFinder] " << spheres.size() << " spheres larger than the minimum radius, "
                          << voids.size() << " voids after removing overlaps\n";
            }
            broadcast_catalogue(voids);
            return voids;
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
