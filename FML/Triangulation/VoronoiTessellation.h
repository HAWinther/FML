#ifndef FML_TRIANGULATION_VORONOITESSELLATION_HEADER
#define FML_TRIANGULATION_VORONOITESSELLATION_HEADER

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <FML/Triangulation/PeriodicVoronoi.h>
#include <FML/Global/ChunkedExchange.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/Watershed.h>
#include <FML/Triangulation/WatershedBinning.h>

namespace FML {
    namespace TRIANGULATION {

        //==========================================================================================
        /// The periodic Voronoi tessellation of the particles in the unit box with FML's own Delaunay triangulation
        /// (Triangulation/PeriodicVoronoi.h, no CGAL), in the form the library's tessellation users need: the Voronoi volume
        /// and the neighbours (Delaunay edges) of every local particle, with the neighbours on other tasks as
        /// boundary particles whose values we can fetch from their owners (like MPIPeriodicDelaunay).
        ///
        /// Unlike MPIPeriodicDelaunay there is no buffer or guard points to choose: the particles can be on any task
        /// (any domain) and the tessellation of every local particle is exact (the ghosts are found and certified).
        /// The result does not depend on the number of tasks.
        ///
        /// Local particle i has index i; boundary particle j (a neighbour on another task) has index NumPart + j.
        /// The positions must be in [0,1).
        //==========================================================================================
        template <int NDIM>
        class PeriodicVoronoiTessellation {
          public:
            using Voronoi = FML::DELAUNAY::PeriodicVoronoi<NDIM>;

            struct Options {
                /// Coordinates are rounded to multiples of 2^-quantization_bits (of the box)
                int quantization_bits = 48;
                /// Keep the Delaunay edges whose Voronoi face has (numerically) zero area, as CGAL does
                bool drop_zero_area_faces = false;
                bool verbose = false;
                /// false: only the triangulation (no volumes or neighbours), e.g. for the DTFE
                bool compute_geometry = true;
                /// Compute and keep the Voronoi faces (voronoi().faces()); otherwise only the volumes and the
                /// neighbours are kept (much less memory)
                bool keep_faces = false;
            };

            PeriodicVoronoiTessellation() = default;

            /// Collective. Tessellate the local particles p[0], ..., p[NumPart-1]
            template <class T>
            void create(T * p, size_t NumPart, const Options & options = Options()) {
                assert_mpi(FML::PARTICLE::GetNDIM(T{}) == NDIM,
                           "[PeriodicVoronoiTessellation] The particles have the wrong dimension");
                npart = NumPart;
                std::vector<typename Voronoi::Point> points(NumPart);
                pos.assign(NumPart * NDIM, 0.0);
                for (size_t i = 0; i < NumPart; i++) {
                    const auto * x = FML::PARTICLE::GetPos(p[i]);
                    for (int d = 0; d < NDIM; d++)
                        pos[i * NDIM + d] = points[i][d] = double(x[d]);
                }
                typename Voronoi::Options opt;
                opt.quantization_bits = options.quantization_bits;
                opt.drop_zero_area_faces = options.drop_zero_area_faces;
                opt.verbose = options.verbose;
                opt.compute_geometry = options.compute_geometry;
                opt.compute_faces = options.keep_faces;
                if (FML::ThisTask == 0 and options.verbose)
                    std::cout << "[PeriodicVoronoiTessellation::create] Tessellating\n";
                V.build(points, 1.0, opt);
                make_graph();
                // The neighbours are in the graph now
                if (not options.keep_faces)
                    V.release_faces();
            }

            /// The underlying tessellation (cells, faces, triangulation)
            const Voronoi & voronoi() const { return V; }

            /// Collective. Make the triangulation exact also over the box [lo, hi] (in the coordinates of the
            /// triangulation: the unit box and the images of the points around it), e.g. for point location in the
            /// region of a grid this task fills. A task with nothing to cover passes lo > hi. Fetches ghosts; the
            /// cells, neighbours and boundary particles do not change
            void cover_region(const std::array<double, NDIM> & lo, const std::array<double, NDIM> & hi) {
                V.cover(lo, hi);
            }

            size_t get_npart() const { return npart; }
            size_t nboundary() const { return boundary_task.size(); }
            /// Owner and index on the owner of boundary particle j
            int get_boundary_task(size_t j) const { return boundary_task[j]; }
            size_t get_boundary_index(size_t j) const { return boundary_index[j]; }

            /// The Voronoi volume of each local particle
            void VoronoiVolume(std::vector<double> & volumes) const {
                assert_mpi(V.cells().size() == npart,
                           "[PeriodicVoronoiTessellation] The volumes were not computed (compute_geometry = false)");
                volumes.resize(npart);
                for (size_t i = 0; i < npart; i++)
                    volumes[i] = V.cells()[i].volume;
            }

            /// The position of local particle k < NumPart or boundary particle k - NumPart (as given, in [0,1))
            const double * position(size_t k) const { return &pos[k * NDIM]; }

            /// Call f(k) for the neighbours of local particle i (each once; local or boundary index)
            template <class F>
            void for_each_neighbor(size_t i, F && f) const {
                for (size_t n = neighbor_offset[i]; n < neighbor_offset[i + 1]; n++)
                    f(neighbor_list[n]);
            }

            /// Collective. The values of the boundary particles from their owners (local[i] is the value of local
            /// particle i)
            template <class Q>
            std::vector<Q> communicate_boundary_values(const std::vector<Q> & local) const {
                static_assert(std::is_trivially_copyable_v<Q>, "Values must be trivially copyable");
                assert_mpi(local.size() == npart, "[PeriodicVoronoiTessellation] Need one value per local particle");
                struct Record {
                    size_t slot;
                    Q value;
                };
                std::vector<std::vector<Record>> outgoing(FML::NTasks);
                for (int t = 0; t < FML::NTasks; t++)
                    for (auto & r : serve[t])
                        outgoing[t].push_back({r.second, local[r.first]});
                std::vector<Q> result(nboundary());
                FML::COMMUNICATION::exchange_records_chunked(outgoing, [&](const Record * rec, size_t n) {
                    for (size_t k = 0; k < n; k++)
                        result[rec[k].slot] = rec[k].value;
                });
                return result;
            }

            /// Collective. The neighbours of boundary particles k[n] >= NumPart (from their owners) as the indices of
            /// the particles this task knows (local or boundary); the others are left out
            void boundary_neighbors(const std::vector<size_t> & k, std::vector<std::vector<size_t>> & out) const {
                struct Request {
                    int from;
                    size_t index;
                    size_t n;
                };
                struct Reply {
                    size_t n;
                    int task;
                    size_t index;
                };
                std::vector<std::vector<Request>> requests(FML::NTasks);
                for (size_t n = 0; n < k.size(); n++) {
                    const size_t j = k[n] - npart;
                    requests[boundary_task[j]].push_back({FML::ThisTask, boundary_index[j], n});
                }
                std::vector<std::vector<Reply>> replies(FML::NTasks);
                FML::COMMUNICATION::exchange_records_chunked(requests, [&](const Request * r, size_t nr) {
                    for (size_t m = 0; m < nr; m++)
                        for_each_neighbor(r[m].index, [&](size_t w) {
                            const int task = w < npart ? FML::ThisTask : boundary_task[w - npart];
                            const size_t index = w < npart ? w : boundary_index[w - npart];
                            replies[r[m].from].push_back({r[m].n, task, index});
                        });
                });
                out.assign(k.size(), {});
                FML::COMMUNICATION::exchange_records_chunked(replies, [&](const Reply * r, size_t nr) {
                    for (size_t m = 0; m < nr; m++) {
                        const size_t w = lookup(r[m].task, r[m].index);
                        if (w != npos)
                            out[r[m].n].push_back(w);
                    }
                });
            }

          private:
            static constexpr size_t npos = size_t(-1);
            Voronoi V;
            size_t npart{0};
            std::vector<double> pos;
            std::vector<int> boundary_task;
            std::vector<size_t> boundary_index;
            std::unordered_map<uint64_t, size_t> boundary_slot;
            std::vector<size_t> neighbor_offset;
            std::vector<size_t> neighbor_list;
            // serve[t]: (local index, slot on task t) of our particles that are boundary particles on task t
            std::vector<std::vector<std::pair<size_t, size_t>>> serve;

            static uint64_t task_index_key(int task, size_t index) { return (uint64_t(task) << 40) | uint64_t(index); }

            size_t lookup(int task, size_t index) const {
                if (task == FML::ThisTask)
                    return index;
                auto it = boundary_slot.find(task_index_key(task, index));
                return it == boundary_slot.end() ? npos : npart + it->second;
            }

            void make_graph() {
                assert_mpi(npart < (size_t(1) << 40), "[PeriodicVoronoiTessellation] Too many particles on a task");
                boundary_task.clear();
                boundary_index.clear();
                boundary_slot.clear();
                auto index_of = [&](int task, size_t index) -> size_t {
                    if (task == FML::ThisTask)
                        return index;
                    auto ins = boundary_slot.insert({task_index_key(task, index), boundary_task.size()});
                    if (ins.second) {
                        boundary_task.push_back(task);
                        boundary_index.push_back(index);
                    }
                    return npart + ins.first->second;
                };
                neighbor_offset.assign(npart + 1, 0);
                neighbor_list.clear();
                if (V.faces().empty()) {
                    // Cell by cell from the neighbour lists
                    std::vector<size_t> nb;
                    for (size_t c = 0; c < V.cells().size(); c++) {
                        nb.clear();
                        const auto range = V.cell_neighbors(c);
                        for (auto it = range.first; it != range.second; ++it) {
                            const size_t w = index_of(it->task(), it->index());
                            if (w != c) // not a periodic image of itself
                                nb.push_back(w);
                        }
                        std::sort(nb.begin(), nb.end());
                        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
                        neighbor_list.insert(neighbor_list.end(), nb.begin(), nb.end());
                        neighbor_offset[c + 1] = neighbor_list.size();
                    }
                    neighbor_list.shrink_to_fit();
                } else {
                    // (local particle, neighbour) pairs from the faces
                    std::vector<std::pair<size_t, size_t>> pairs;
                    auto add = [&](size_t c, int task, size_t index) {
                        const size_t w = index_of(task, index);
                        if (w != c)
                            pairs.push_back({c, w});
                    };
                    for (auto & f : V.faces()) {
                        if (f.local_i != Voronoi::npos)
                            add(f.local_i, f.task_j, f.index_j);
                        if (f.local_j != Voronoi::npos)
                            add(f.local_j, f.task_i, f.index_i);
                    }
                    std::sort(pairs.begin(), pairs.end());
                    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
                    neighbor_list.resize(pairs.size());
                    for (size_t n = 0; n < pairs.size(); n++) {
                        neighbor_offset[pairs[n].first + 1]++;
                        neighbor_list[n] = pairs[n].second;
                    }
                    for (size_t i = 0; i < npart; i++)
                        neighbor_offset[i + 1] += neighbor_offset[i];
                }

                // Tell the owners which of their particles we need
                struct Request {
                    int from;
                    size_t index;
                    size_t slot;
                };
                std::vector<std::vector<Request>> requests(FML::NTasks);
                for (size_t j = 0; j < nboundary(); j++)
                    requests[boundary_task[j]].push_back({FML::ThisTask, boundary_index[j], j});
                serve.assign(FML::NTasks, {});
                FML::COMMUNICATION::exchange_records_chunked(requests, [&](const Request * r, size_t n) {
                    for (size_t k = 0; k < n; k++)
                        serve[r[k].from].push_back({r[k].index, r[k].slot});
                });

                // The positions of the boundary particles
                std::vector<std::array<double, NDIM>> x(npart);
                for (size_t i = 0; i < npart; i++)
                    for (int d = 0; d < NDIM; d++)
                        x[i][d] = pos[i * NDIM + d];
                const auto xb = communicate_boundary_values(x);
                pos.resize((npart + nboundary()) * NDIM);
                for (size_t j = 0; j < nboundary(); j++)
                    for (int d = 0; d < NDIM; d++)
                        pos[(npart + j) * NDIM + d] = xb[j][d];
            }
        };

        //===============================================================================
        /// The watershed on the native tessellation (see WatershedOnGraph in Watershed.h)
        ///
        /// @param[in] D The tessellation of the particles (already created)
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] quantity quantity[i] is the quantity of particle i (e.g. its density)
        /// @param[out] watershed_groups The watershed groups (only on task 0)
        /// @param[out] links Optional: the links between the basins (index into watershed_groups), only on task 0
        //===============================================================================
        template <class T, class U, int NDIM>
        void WatershedGeneral(PeriodicVoronoiTessellation<NDIM> & D,
                              T * p,
                              size_t NumPart,
                              std::vector<double> & quantity,
                              std::vector<U> & watershed_groups,
                              std::vector<WatershedLink> * links = nullptr) {
            assert_mpi(D.get_npart() == NumPart,
                       "[WatershedGeneral] The tessellation was not made with these particles");
            WatershedOnGraph<NDIM>(D, p, NumPart, quantity, watershed_groups, links);
        }

        //==========================================================================================
        /// WatershedDensity (PeriodicDelaunay.h) with the native tessellation: tessellate, set the Voronoi volume
        /// of the particles (set_volume), and watershed on the density mass/volume (voids) or, with
        /// do_density_maximum, on volume/mass (clusters). The particle's get_ndim must be constexpr
        //==========================================================================================
        template <class T, class U>
        void WatershedDensityNative(
            T * p,
            size_t NumPart,
            std::vector<U> & watershed_groups,
            bool do_density_maximum = false,
            std::vector<WatershedLink> * links = nullptr,
            const typename PeriodicVoronoiTessellation<FML::PARTICLE::GetNDIM(T{})>::Options & options = {}) {
            static_assert(FML::PARTICLE::has_set_volume<T>(),
                          "[WatershedDensity] We require the particle to have a set_volume / get_volume method");
            constexpr int NDIM = FML::PARTICLE::GetNDIM(T{});
            if (FML::ThisTask == 0)
                std::cout << "[WatershedDensity] We will locate minima of "
                          << (do_density_maximum ? "1/(voronoi density)" : "the voronoi density")
                          << " of particles (native tessellation)\n";
            PeriodicVoronoiTessellation<NDIM> D;
            D.create(p, NumPart, options);
            std::vector<double> volumes;
            D.VoronoiVolume(volumes);
            for (size_t i = 0; i < NumPart; i++)
                FML::PARTICLE::SetVolume(p[i], volumes[i]);
            std::vector<double> density = watershed_density_quantity(p, volumes, do_density_maximum);
            WatershedGeneral(D, p, NumPart, density, watershed_groups, links);
        }

    } // namespace TRIANGULATION
} // namespace FML
#endif
