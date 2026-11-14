#ifndef FML_VOIDFINDERS_VOXEL_HEADER
#define FML_VOIDFINDERS_VOXEL_HEADER

#include <FML/Global/MPICounts.h>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GhostedCells.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/VoidFinders/Void.h>

namespace FML {
    namespace VOIDFINDER {

        /// Options for VoxelVoidFinder (defaults as in Revolver)
        struct VoxelOptions {
            /// Grid size. 0: as in Revolver, the cell size is half the radius of the sphere with one particle
            /// on average, i.e. Nmesh = floor(2 (4 pi N / 3)^(1/3)), rounded to the nearest multiple of 8 (32 from 256;
            /// the same for any number of tasks; set Nmesh if the task count does not divide it)
            int Nmesh{0};
            /// Gaussian smoothing scale in units of the box. Negative: as in Revolver, the mean particle
            /// separation N^(-1/3). 0: no smoothing
            double smoothing_scale{-1.0};
            /// "revolver": convolve with a Gaussian sampled on the grid and truncated at 4 sigma (what scipy's
            /// gaussian_filter, used in Revolver, does), "gaussian": multiply by exp(-k^2 R^2/2) in Fourier space
            std::string smoothing_method{"revolver"};
            /// Density assignment (NGP, CIC, TSC, PCS, PQS). Revolver uses CIC
            std::string density_assignment_method{"CIC"};
            /// Find overdensities (clusters) instead of voids
            bool find_clusters{false};
            /// Voids: the minimum density must be below this (in units of the mean)
            double min_dens_cut{1.0};
            /// Clusters: the maximum density must be above this (in units of the mean)
            double max_dens_cut{1.0};
            /// How to merge the zones into voids (None: every zone is a void as in Revolver), see ZoneMerging
            ZoneMerging merging{ZoneMerging::None};
            /// Link density (in units of the mean) below which zones are merged (above for clusters, which need a
            /// value > 1)
            double link_density_threshold{0.2};
        };

        //========================================================================================
        /// Watershed void finder on a density grid (the second step of the voxel void finder in Revolver, which can
        /// also be used directly on any density field, e.g. from the DTFE). Every voxel is linked to the lowest of
        /// its 6 neighbors if that is lower than itself, and a zone is all the voxels that end up at the same
        /// minimum. Ties are broken by the voxel index. As in Revolver the zones are not merged: every zone whose
        /// minimum density is below min_dens_cut is a void.
        ///
        /// For each void: pos = pos_min = the position of the minimum voxel, the barycenter is weighted by
        /// 1/density, the volume is the number of voxels times the voxel volume, density_avg is the mean density of
        /// the voxels, and the density ratio is the lowest link density to a neighboring zone (a link between two
        /// neighboring voxels has the larger of the two densities) divided by the minimum density.
        ///
        /// With find_clusters we run the same on 1/density (then the barycenter is weighted by the density).
        ///
        /// Voxel (ix,iy,iz) is at (ix,iy,iz)/Nmesh. The result is the same on all tasks.
        ///
        /// @param[in] density The density field in units of the mean density (1 + delta)
        /// @param[in] find_clusters Find maxima instead of minima
        /// @param[in] density_cut Voids: keep the zones with minimum density below this. Clusters: maximum density
        /// above
        /// @param[in] merging How to merge the zones into voids (None: every zone is a void as in Revolver)
        /// @param[in] link_density_threshold For ZoneMerging::Threshold: merge zones linked below this density (above
        /// for clusters)
        ///
        //========================================================================================
        template <template <int> class Grid = FML::GRID::FFTWGrid>
        std::vector<Void> VoxelWatershed(const Grid<3> & density,
                                         bool find_clusters = false,
                                         double density_cut = 1.0,
                                         ZoneMerging merging = ZoneMerging::None,
                                         double link_density_threshold = 0.2) {

            // Clusters merge across links above the threshold: the void default (0.2) would merge everything
            assert_mpi(not(find_clusters and merging == ZoneMerging::Threshold and link_density_threshold <= 1.0),
                       "[VoxelWatershed] Clusters with ZoneMerging::Threshold need link_density_threshold > 1 (in "
                       "units of the mean; the default 0.2 is for voids)");
            const int N = density.get_nmesh();
            // The local cells: [x0, x0 + nx) x [y0, y0 + ny) x [0, N) (slabs: y whole)
            const int nx = int(density.get_local_range(0)[1]), ny = int(density.get_local_range(1)[1]);
            const long long int x0 = density.get_local_range(0)[0], y0 = density.get_local_range(1)[0];
            const long long int nloc = (long long int)(nx) * ny * N;
            assert_mpi(nloc > 0, "[VoxelWatershed] Every task must have some of the grid");

            // The quantity we find minima of, with one extra cell on each side of the split axes
            auto q = make_grid_like(density, N, 1, 1);
            for (int ix = 0; ix < nx; ix++)
                for (int iy = 0; iy < ny; iy++)
                    for (int iz = 0; iz < N; iz++) {
                        const double d = density.get_real({ix, iy, iz});
                        q.set_real({ix, iy, iz}, FML::GRID::FloatType(find_clusters ? 1.0 / d : d));
                    }
            q.communicate_boundaries();
            auto q_at = [&](int ix, int iy, int iz) { return q.get_real(FML::GRID::wrap_whole_axes<3>(q, {ix, iy, iz})); };

            // The labels of our cells and of the ghost cells of the neighbours (-1: none yet)
            const long long int no_label = -1;
            FML::GRID::GhostedCells<long long int, 3> label(density, 1, 1, no_label);
            const bool serial = not label.is_split(0) and not label.is_split(1);
            auto local_index = [&](int ix, int iy, int iz) -> long long int { return label.index({ix, iy, iz}); };
            auto global_index = [&](int ix, int iy, int iz) -> long long int {
                const long long int gx = (x0 + ix + N) % N, gy = (y0 + iy + N) % N;
                return (gx * N + gy) * N + iz;
            };
            // Order: by quantity then by the global index
            auto lower = [](double qa, long long int ga, double qb, long long int gb) {
                return qa != qb ? qa < qb : ga < gb;
            };
            const int dirs[6][3] = {{0, 0, 1}, {0, 0, -1}, {0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}};
            auto for_each_neighbor = [&](int ix, int iy, int iz, auto && f) {
                for (auto & d : dirs)
                    f(ix + d[0], iy + d[1], (iz + d[2] + N) % N);
            };
            // Is a stored label one of our cells?
            std::vector<char> is_ghost(size_t(label.size()));
            for (long long int i = 0; i < label.size(); i++)
                is_ghost[i] = char(label.is_ghost(i));

            // Link every voxel to its lowest neighbor
            std::vector<long long int> parent(size_t(label.size()), -1);
            std::vector<long long int> minima;
            for (int ix = 0; ix < nx; ix++)
                for (int iy = 0; iy < ny; iy++)
                    for (int iz = 0; iz < N; iz++) {
                        double qbest = q.get_real({ix, iy, iz});
                        long long int gbest = global_index(ix, iy, iz);
                        long long int best = local_index(ix, iy, iz);
                        const long long int self = best;
                        for_each_neighbor(ix, iy, iz, [&](int jx, int jy, int jz) {
                            const double qj = q_at(jx, jy, jz);
                            const long long int gj = global_index(jx, jy, jz);
                            if (lower(qj, gj, qbest, gbest)) {
                                qbest = qj;
                                gbest = gj;
                                best = local_index(jx, jy, jz);
                            }
                        });
                        parent[self] = best;
                        if (best == self)
                            minima.push_back(self);
                    }

            // Give the minima an ID that is unique across tasks
            std::vector<long long int> nminima_per_task(FML::NTasks, 0);
            nminima_per_task[FML::ThisTask] = (long long int)(minima.size());
            FML::SumArrayOverTasks(nminima_per_task.data(), nminima_per_task.size());
            FML::COMMUNICATION::checked_size_layout(
                std::vector<FML::COMMUNICATION::Count>(nminima_per_task.begin(), nminima_per_task.end()),
                size_t(LLONG_MAX));
            long long int id_start = 0, nzones = 0;
            for (int i = 0; i < FML::NTasks; i++) {
                if (i < FML::ThisTask)
                    id_start += nminima_per_task[i];
                nzones += nminima_per_task[i];
            }
            if (FML::ThisTask == 0)
                std::cout << "[VoxelWatershed] Found " << nzones << " zones on a " << N << "^3 grid\n";

            // Labels: follow the links. For links into the ghost cells we need the labels from the neighbor tasks
            for (size_t k = 0; k < minima.size(); k++)
                label[minima[k]] = id_start + (long long int)(k);
            std::vector<char> stuck;
            std::vector<long long int> path;
            auto assign_labels = [&]() -> long long int {
                stuck.assign(size_t(label.size()), 0);
                long long int nassigned = 0;
                for (long long int i = 0; i < label.size(); i++) {
                    if (is_ghost[i] or label[i] != no_label or stuck[i])
                        continue;
                    path.clear();
                    long long int j = i;
                    while (not is_ghost[j] and label[j] == no_label and not stuck[j]) {
                        path.push_back(j);
                        j = parent[j];
                    }
                    const bool found = (is_ghost[j] or not stuck[j]) and label[j] != no_label;
                    for (auto k : path) {
                        if (found)
                            label[k] = label[j];
                        else
                            stuck[k] = 1;
                    }
                    if (found)
                        nassigned += (long long int)(path.size());
                }
                return nassigned;
            };
            auto exchange_ghost_labels = [&]() { label.communicate(); };
            assign_labels();
            if (not serial) {
                for (int iteration = 0; iteration < 100000; iteration++) {
                    exchange_ghost_labels();
                    long long int nassigned = assign_labels();
                    FML::SumOverTasks(&nassigned);
                    if (nassigned == 0)
                        break;
                }
            }

            // The position and density of the minima (all tasks need this)
            std::vector<double> core(FML::COMMUNICATION::checked_bytes(nzones, 4, std::vector<double>().max_size()),
                                     0.0);
            for (size_t k = 0; k < minima.size(); k++) {
                const auto c3 = label.coord(minima[k]);
                double * c = &core[4 * (id_start + (long long int)(k))];
                c[0] = double(x0 + c3[0]) / N;
                c[1] = double(y0 + c3[1]) / N;
                c[2] = double(c3[2]) / N;
                c[3] = q.get_real(c3);
            }

            FML::SumArrayOverTasks(core.data(), core.size());

            // Bin up: number of voxels, sum of density, barycenter, leak quantity
            std::vector<double> sums(FML::COMMUNICATION::checked_bytes(nzones, 6, std::vector<double>().max_size()),
                                     0.0);
            std::vector<ZoneLink> links;
            long long int nunassigned = 0;
            for (int ix = 0; ix < nx; ix++)
                for (int iy = 0; iy < ny; iy++)
                    for (int iz = 0; iz < N; iz++) {
                        const long long int a = label[local_index(ix, iy, iz)];
                        if (a == no_label) {
                            nunassigned++;
                            continue;
                        }
                        const double qi = q.get_real({ix, iy, iz});
                        const double weight = 1.0 / qi;
                        const double pos[3] = {double(x0 + ix) / N, double(y0 + iy) / N, double(iz) / N};
                        double * s = &sums[6 * a];
                        s[0] += 1.0;
                        s[1] += find_clusters ? 1.0 / qi : qi;
                        s[2] += weight;
                        for (int j = 0; j < 3; j++) {
                            double dx = pos[j] - core[4 * a + j];
                            dx -= std::round(dx);
                            s[3 + j] += weight * dx;
                        }
                        for_each_neighbor(ix, iy, iz, [&](int jx, int jy, int jz) {
                            const long long int b = label[local_index(jx, jy, jz)];
                            if (b == no_label or b == a)
                                return;
                            const double link = std::max(qi, double(q_at(jx, jy, jz)));
                            links.push_back({std::min(a, b), std::max(a, b), link});
                        });
                        if (links.size() > size_t(4 * nzones + 1000000))
                            reduce_zone_links(links);
                    }
            reduce_zone_links(links);
            FML::SumOverTasks(&nunassigned);
            if (FML::ThisTask == 0 and nunassigned > 0)
                std::cout << "[VoxelWatershed] Warning: " << nunassigned << " voxels were not assigned to a zone\n";
#ifdef USE_MPI
            FML::SumArrayOverTasks(sums.data(), sums.size());
            // Gather the links on task 0
            if (not serial) {
                links = FML::COMMUNICATION::gather_records_checked(links);
            }
#endif

            // Merge the zones and make the catalogue on task 0
            std::vector<Void> voids;
            if (FML::ThisTask == 0) {
                const double voxel_volume = 1.0 / (double(N) * N * N);
                std::vector<Zone> zones(nzones);
                for (long long int a = 0; a < nzones; a++) {
                    const double * c = &core[4 * a];
                    const double * sm = &sums[6 * a];
                    auto & z = zones[a];
                    z.quantity_core = c[3];
                    z.n = (long long int)(sm[0]);
                    z.volume = sm[0] * voxel_volume;
                    z.mass = z.volume;
                    z.sum_density = sm[1];
                    z.weight = sm[2];
                    for (int j = 0; j < 3; j++) {
                        z.pos_min[j] = z.pos_center[j] = c[j];
                        z.weighted_dx[j] = sm[3 + j];
                    }
                }
                const double threshold_quantity = find_clusters ? 1.0 / link_density_threshold : link_density_threshold;
                const auto merged = merge_zones(zones, links, merging, threshold_quantity);
                voids = catalogue_from_merged_zones(merged, find_clusters, 1.0, false, [&](const Void & v) {
                    return find_clusters ? v.density_min > density_cut : v.density_min < density_cut;
                });
                long long int ntop = 0;
                for (auto & v : voids)
                    ntop += v.parent < 0;
                std::cout << "[VoxelWatershed] " << voids.size() << (find_clusters ? " clusters" : " voids")
                          << " pass the cut (" << ntop << " top-level)\n";
            }
            broadcast_catalogue(voids);
            return voids;
        }

        //========================================================================================
        /// The density field used by the voxel void finder (and for the candidate centers of the spherical void
        /// finder): assign the particles to a grid (CIC) and smooth with a Gaussian, with the grid size and the
        /// smoothing scale as in Revolver unless given in the options. The result is in units of the mean density.
        /// The particles must be in [0,1)^3 and inside the local domain. Smoothing requires FFTW (USE_FFTW).
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] options The options (Nmesh, smoothing_scale, smoothing_method, density_assignment_method)
        /// @param[out] grid The density field
        ///
        //========================================================================================
        template <class T, template <int> class Grid = FML::GRID::FFTWGrid>
        void void_density_field(T * p,
                                size_t NumPart,
                                const VoxelOptions & options,
                                Grid<3> & grid,
                                const FML::Domain & domain = FML::domain) {
            constexpr bool fftw = FML::GRID::is_fftwgrid_v<Grid<3>>;

            assert_mpi(FML::PARTICLE::GetNDIM(T()) == 3, "[void_density_field] Only implemented in 3D");
            long long int NumPartTot = (long long int)(NumPart);
            FML::SumOverTasks(&NumPartTot);
            assert_mpi(NumPartTot > 0, "[void_density_field] No particles");

            int Nmesh = options.Nmesh;
            if (Nmesh <= 0) {
                // Revolver's size rounded to the nearest multiple of 8 (32 from 256, a cell size within 6% of
                // Revolver's). It only depends on the number of particles, so the catalogue does not depend on the
                // number of tasks (it used to be rounded down to a multiple of the task count)
                const double n_revolver = std::floor(2.0 * std::cbrt(4.0 * M_PI * double(NumPartTot) / 3.0));
                const int multiple = n_revolver >= 256 ? 32 : 8;
                Nmesh = std::max(multiple, multiple * int(std::lround(n_revolver / multiple)));
                assert_mpi(not fftw or Nmesh % FML::NTasks == 0,
                           "[VoxelVoidFinder] The default Nmesh (Revolver's size rounded to a multiple of 8 or 32) is "
                           "not a multiple of the number of tasks: set Nmesh");
            }
            assert_mpi(not fftw or (Nmesh >= FML::NTasks and Nmesh % FML::NTasks == 0),
                       "[VoxelVoidFinder] Nmesh must be a multiple of the number of tasks");
            const double smoothing_scale =
                options.smoothing_scale < 0.0 ? std::cbrt(1.0 / double(NumPartTot)) : options.smoothing_scale;
            if (FML::ThisTask == 0)
                std::cout << "[void_density_field] Nmesh = " << Nmesh << " smoothing scale = " << smoothing_scale
                          << " (box units)\n";

            // Density assignment gives delta
            const auto nextra = FML::INTERPOLATION::get_extra_slices(options.density_assignment_method);
            grid = FML::GRID::make_grid<Grid<3>>(Nmesh, nextra.first, nextra.second, domain);
            FML::INTERPOLATION::particles_to_grid(p,
                                                  NumPart,
                                                  size_t(NumPartTot),
                                                  grid,
                                                  options.density_assignment_method);

            // Gaussian smoothing
            if (smoothing_scale > 0.0) {
#ifdef USE_FFTW
                grid.fftw_r2c();
                if (options.smoothing_method == "gaussian") {
                    FML::GRID::smoothing_filter_fourier_space(grid, smoothing_scale, "gaussian");
                } else {
                    assert_mpi(options.smoothing_method == "revolver",
                               "[VoxelVoidFinder] Unknown smoothing_method (revolver or gaussian)");
                    // The 1D kernel w(x) ~ exp(-x^2/2sigma^2) for |x| <= int(4 sigma + 0.5) (in units of the cell
                    // size, normalized to 1) and its discrete Fourier transform (real as w is symmetric)
                    const double sigma = smoothing_scale * Nmesh;
                    const int radius = int(4.0 * sigma + 0.5);
                    std::vector<double> w(radius + 1);
                    double norm = 0.0;
                    for (int x = 0; x <= radius; x++) {
                        w[x] = std::exp(-0.5 * x * x / (sigma * sigma));
                        norm += (x == 0 ? 1.0 : 2.0) * w[x];
                    }
                    std::vector<double> what(Nmesh, 0.0);
                    for (int m = 0; m < Nmesh; m++)
                        for (int x = 0; x <= radius; x++)
                            what[m] += (x == 0 ? 1.0 : 2.0) * w[x] / norm * std::cos(2.0 * M_PI * m * x / Nmesh);
                    std::array<double, 3> kvec;
                    double kmag2;
                    for (auto && index : grid.get_fourier_range()) {
                        grid.get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag2);
                        double factor = 1.0;
                        for (int j = 0; j < 3; j++) {
                            const int m = int(std::lround(kvec[j] / (2.0 * M_PI)));
                            factor *= what[((m % Nmesh) + Nmesh) % Nmesh];
                        }
                        grid.set_fourier_from_index(index,
                                                    grid.get_fourier_from_index(index) * FML::GRID::FloatType(factor));
                    }
                }
                grid.fftw_c2r();
#else
                assert_mpi(false, "[VoxelVoidFinder] Smoothing requires FFTW (USE_FFTW). Use smoothing_scale = 0");
#endif
            }

            // Density in units of the mean
            for (auto && index : grid.get_real_range())
                grid.set_real_from_index(index, FML::GRID::FloatType(1.0 + grid.get_real_from_index(index)));
        }

        //========================================================================================
        /// The voxel void finder in Revolver: assign the particles to a grid (CIC), smooth with a Gaussian, and find
        /// the watershed zones of the density field (see VoxelWatershed). The defaults for the grid size and the
        /// smoothing scale are as in Revolver. The particles must be in [0,1)^3 and inside the local domain
        /// (e.g. MPIParticles). get_mass is used if it exists. The result is the same on all tasks.
        /// Smoothing requires FFTW (USE_FFTW).
        ///
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] options The options
        /// @param[out] density_grid Optional: the smoothed density field (in units of the mean) we used
        /// @param[in] domain The domain of the particles (the grid is decomposed the same way; Grid = PencilGrid for
        /// a pencil domain)
        ///
        //========================================================================================
        template <class T, template <int> class Grid = FML::GRID::FFTWGrid>
        std::vector<Void> VoxelVoidFinder(T * p,
                                          size_t NumPart,
                                          const VoxelOptions & options = VoxelOptions(),
                                          FML::GRID::non_deduced_t<Grid<3>> * density_grid = nullptr,
                                          const FML::Domain & domain = FML::domain) {
            if (FML::ThisTask == 0)
                std::cout << "[VoxelVoidFinder] Finding " << (options.find_clusters ? "clusters" : "voids") << "\n";
            Grid<3> grid;
            void_density_field<T, Grid>(p, NumPart, options, grid, domain);
            auto voids = VoxelWatershed(grid,
                                        options.find_clusters,
                                        options.find_clusters ? options.max_dens_cut : options.min_dens_cut,
                                        options.merging,
                                        options.link_density_threshold);
            if (density_grid)
                *density_grid = grid;
            return voids;
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
