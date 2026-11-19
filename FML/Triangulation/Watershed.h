#ifndef FML_TRIANGULATION_WATERSHED_HEADER
#define FML_TRIANGULATION_WATERSHED_HEADER

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <FML/Experimental/Delaunay/Predicates.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Triangulation/WatershedBinning.h>

//==========================================================================================
// The watershed on the neighbour graph of a tessellation, independent of how the tessellation is made (CGAL's
// MPIPeriodicDelaunay in PeriodicDelaunay.h or the native PeriodicVoronoiTessellation in VoronoiTessellation.h).
//
// WatershedOnGraph works on any Graph with (positions in the unit box [0,1)^NDIM):
//   size_t nboundary() const                  number of boundary particles (copies of particles on other tasks
//                                              that are neighbours of the local ones)
//   const double * position(size_t k) const   position of local particle k < NumPart or boundary particle
//                                              k - NumPart
//   void for_each_neighbor(size_t i, F f)     f(k) for each neighbour k (local or boundary index as above) of
//                                              local particle i, each at least once, never i itself
//   std::vector<Q> communicate_boundary_values(const std::vector<Q> & local)
//                                              (collective) the values of the boundary particles from their owners
//   void boundary_neighbors(const std::vector<size_t> & k, std::vector<std::vector<size_t>> & out)
//                                              (collective) the neighbours of the boundary particles k[n] >= NumPart
//                                              (out[n]), as indices of the particles this task knows (local or
//                                              boundary); the others are left out
//==========================================================================================

namespace FML {
    namespace TRIANGULATION {

        /// Keep track of which type of point we are adding (CGAL tessellation)
        enum PointType : char { REGULAR_POINT, BOUNDARY_POINT, GUARD_POINT };

        // Type definitions
        using IDType = int;
        using QuantityType = double;
        const IDType NoWatershedID = std::numeric_limits<IDType>::max();
        const QuantityType Infinity = std::numeric_limits<QuantityType>::infinity();

        /// A link between two watershed basins: the lowest quantity at which they connect. For a pair of neighboring
        /// particles in the two basins the link has the larger of the two quantities
        struct WatershedLink {
            long long int a{0};
            long long int b{0};
            double quantity{0.0};
        };

        namespace detail {
            /// Keep only the lowest link for each pair (a < b)
            inline void reduce_links(std::vector<WatershedLink> & links) {
                std::sort(links.begin(), links.end(), [](const WatershedLink & x, const WatershedLink & y) {
                    if (x.a != y.a)
                        return x.a < y.a;
                    if (x.b != y.b)
                        return x.b < y.b;
                    return x.quantity < y.quantity;
                });
                links.erase(std::unique(links.begin(),
                                        links.end(),
                                        [](const WatershedLink & x, const WatershedLink & y) {
                                            return x.a == y.a and x.b == y.b;
                                        }),
                            links.end());
            }

#ifdef USE_MPI
            /// MPI reduction operation that merges watershed groups
            template <class U>
            void merge_groups_op(void * in, void * inout, int * len, MPI_Datatype *) {
                U * a = static_cast<U *>(in);
                U * b = static_cast<U *>(inout);
                for (int k = 0; k < *len; k++)
                    b[k].merge(a[k]);
            }
#endif

            /// Does the watershed group class have a set_pos_circ(const double *) method
            template <class U, class = void>
            struct has_set_pos_circ : std::false_type {};
            template <class U>
            struct has_set_pos_circ<U, std::void_t<decltype(std::declval<U &>().set_pos_circ((const double *)nullptr))>>
                : std::true_type {};

            /// Does the watershed group class have a set_leak_quantity(double) method
            template <class U, class = void>
            struct has_set_leak_quantity : std::false_type {};
            template <class U>
            struct has_set_leak_quantity<U, std::void_t<decltype(std::declval<U &>().set_leak_quantity(0.0))>>
                : std::true_type {};

            /// Map x into [0,1)
            inline double wrap_periodic(double x) {
                x -= std::floor(x);
                return x >= 1.0 ? 0.0 : x;
            }

            /// The circumcenter of the simplex 0, q[1], ..., q[NDIM] (q[0] is the origin). False if it is degenerate
            /// (exact test) or the result is not finite
            template <int NDIM>
            bool circumcenter_from_origin(const std::array<std::array<double, NDIM>, NDIM + 1> & q, double * cc) {
                std::array<const double *, NDIM + 1> ptr;
                for (int j = 0; j <= NDIM; j++)
                    ptr[j] = q[j].data();
                if (FML::DELAUNAY::orient<NDIM>(ptr) == 0)
                    return false;
                // 2 q_j . c = |q_j|^2 for j = 1..NDIM (Cramer's rule)
                double A[NDIM][NDIM], b[NDIM];
                for (int j = 0; j < NDIM; j++) {
                    b[j] = 0.0;
                    for (int d = 0; d < NDIM; d++) {
                        A[j][d] = q[j + 1][d];
                        b[j] += 0.5 * q[j + 1][d] * q[j + 1][d];
                    }
                }
                auto det = [](const double(&M)[NDIM][NDIM]) {
                    if constexpr (NDIM == 2)
                        return M[0][0] * M[1][1] - M[0][1] * M[1][0];
                    else
                        return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                               M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                               M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
                };
                const double D = det(A);
                for (int d = 0; d < NDIM; d++) {
                    double M[NDIM][NDIM];
                    for (int j = 0; j < NDIM; j++)
                        for (int e = 0; e < NDIM; e++)
                            M[j][e] = e == d ? b[j] : A[j][e];
                    cc[d] = det(M) / D;
                    if (!std::isfinite(cc[d]))
                        return false;
                }
                return true;
            }
        } // namespace detail

        //===============================================================================
        /// The watershed on the neighbour graph of a tessellation (see the top of this file for what the graph
        /// must provide): every particle is linked to the neighbor with the smallest quantity (steepest descent)
        /// and a basin is all the particles that end up at the same local minimum. Ties are broken using the
        /// positions, so the result does not depend on the number of tasks. When quantity is the density or
        /// inverse density of the particle then we get a void or cluster finder.
        ///
        /// @param[in] G The graph
        /// @param[in] p Pointer to the particles
        /// @param[in] NumPart Number of local particles
        /// @param[in] quantity quantity[i] is the quantity of particle i
        /// @param[out] watershed_groups The watershed groups (only on task 0)
        /// @param[out] links Optional: the links between the basins (index into watershed_groups), only on task 0
        /// @param[out] labels Optional: the basin of each local particle (NumPart) and boundary particle
        ///
        //===============================================================================
        template <int NDIM, class Graph, class T, class U>
        void WatershedOnGraph(Graph & G,
                              T * p,
                              size_t NumPart,
                              const std::vector<double> & quantity,
                              std::vector<U> & watershed_groups,
                              std::vector<WatershedLink> * links = nullptr,
                              std::vector<IDType> * labels = nullptr) {

            assert_mpi(quantity.size() == NumPart, "[WatershedGeneral] quantity must have size NumPart");
            const size_t nboundary = G.nboundary();

            // The quantity of the regular particles followed by the boundary particles
            std::vector<double> q(quantity);
            {
                const std::vector<double> qb = G.communicate_boundary_values(quantity);
                q.insert(q.end(), qb.begin(), qb.end());
            }

            // Ordering: by quantity and then by position to break ties
            auto lower = [&](size_t a, size_t b) {
                if (q[a] != q[b])
                    return q[a] < q[b];
                const double * pa = G.position(a);
                const double * pb = G.position(b);
                for (int idim = 0; idim < NDIM; idim++)
                    if (pa[idim] != pb[idim])
                        return pa[idim] < pb[idim];
                return false;
            };

            // For each regular particle find the neighbor with the lowest quantity (the particle itself if it is a
            // local minimum). Index i < NumPart is a regular particle, i >= NumPart a boundary particle
            std::vector<long long int> parent(NumPart);
            for (size_t i = 0; i < NumPart; i++) {
                size_t best = i;
                G.for_each_neighbor(i, [&](size_t k) {
                    if (lower(k, best))
                        best = k;
                });
                parent[i] = (long long int)(best);
            }

            // Find the minima and give them an ID that is unique across tasks
            std::vector<size_t> minima;
            for (size_t i = 0; i < NumPart; i++)
                if (parent[i] == (long long int)(i))
                    minima.push_back(i);
            std::vector<long long int> nminima_per_task(FML::NTasks, 0);
            nminima_per_task[FML::ThisTask] = (long long int)(minima.size());
            FML::SumArrayOverTasks(nminima_per_task.data(), nminima_per_task.size());
            FML::COMMUNICATION::checked_size_layout(
                std::vector<FML::COMMUNICATION::Count>(nminima_per_task.begin(), nminima_per_task.end()),
                size_t(LLONG_MAX));
            long long int id_start = 0;
            for (int i = 0; i < FML::ThisTask; i++)
                id_start += nminima_per_task[i];
            long long int ntotal_minima_ll = 0;
            for (int i = 0; i < FML::NTasks; i++)
                ntotal_minima_ll += nminima_per_task[i];
            assert_mpi(ntotal_minima_ll < (long long int)(NoWatershedID), "[WatershedGeneral] Too many minima");
            const size_t ntotal_minima = size_t(ntotal_minima_ll);

            if (FML::ThisTask == 0)
                std::cout << "[WatershedGeneral] We found " << ntotal_minima << " minimum points\n";

            // Labels for the regular and boundary particles
            std::vector<IDType> label(NumPart + nboundary, NoWatershedID);
            for (size_t k = 0; k < minima.size(); k++)
                label[minima[k]] = IDType(id_start + (long long int)(k));

            // Follow the parent links until we reach a particle with a label. If we end up at a boundary
            // particle we don't have the label for yet we have to wait until we get it from its owner
            std::vector<char> stuck;
            std::vector<long long int> path;
            auto assign_labels = [&]() -> size_t {
                stuck.assign(NumPart, 0);
                size_t nassigned = 0;
                for (size_t i = 0; i < NumPart; i++) {
                    if (label[i] != NoWatershedID or stuck[i])
                        continue;
                    path.clear();
                    long long int j = (long long int)(i);
                    while (j < (long long int)(NumPart) and label[j] == NoWatershedID and not stuck[j]) {
                        path.push_back(j);
                        j = parent[j];
                    }
                    const bool found = (j >= (long long int)(NumPart) or not stuck[j]) and label[j] != NoWatershedID;
                    for (auto k : path) {
                        if (found)
                            label[k] = label[j];
                        else
                            stuck[k] = 1;
                    }
                    if (found)
                        nassigned += path.size();
                }
                return nassigned;
            };
            assign_labels();

            // Exchange labels of the boundary particles until no more particles get assigned
            if (FML::NTasks > 1) {
                const int max_iterations = 1000;
                int iteration = 0;
                for (; iteration < max_iterations; iteration++) {
                    const std::vector<IDType> label_regular(label.begin(), label.begin() + NumPart);
                    const std::vector<IDType> label_boundary = G.communicate_boundary_values(label_regular);
                    std::copy(label_boundary.begin(), label_boundary.end(), label.begin() + NumPart);
                    long long int nassigned = (long long int)(assign_labels());
                    FML::SumOverTasks(&nassigned);
                    if (nassigned == 0)
                        break;
                }
#ifdef DEBUG_TESSELATION
                if (FML::ThisTask == 0)
                    std::cout << "[WatershedGeneral] Labels converged after " << iteration << " exchanges\n";
#endif
            }

            // The leak quantity of each basin (as in ZOBOV): the lowest quantity at which the basin links to a
            // neighboring basin. A link between two neighboring particles in different basins has the larger
            // of the two quantities. Pairs that cross tasks are seen by the task that owns one of the particles
            std::vector<double> leak_quantity(ntotal_minima, std::numeric_limits<double>::infinity());
            std::vector<WatershedLink> local_links;
            for (size_t i = 0; i < NumPart; i++) {
                const IDType a = label[i];
                if (a == NoWatershedID)
                    continue;
                G.for_each_neighbor(i, [&](size_t k) {
                    const IDType b = label[k];
                    if (b == NoWatershedID or b == a)
                        return;
                    const double link = std::max(q[i], q[k]);
                    leak_quantity[a] = std::min(leak_quantity[a], link);
                    leak_quantity[b] = std::min(leak_quantity[b], link);
                    if (links)
                        local_links.push_back({std::min(a, b), std::max(a, b), link});
                });
                // Keep only the lowest link for each pair of basins now and then to save memory
                if (links and local_links.size() > 4 * ntotal_minima + 1000000)
                    detail::reduce_links(local_links);
            }
            if (links) {
                detail::reduce_links(local_links);
                // Gather on task 0 (where the groups are)
                links->clear();
#ifdef USE_MPI
                if (FML::NTasks > 1) {
                    *links = FML::COMMUNICATION::gather_records_checked(local_links);
                    if (FML::ThisTask == 0)
                        detail::reduce_links(*links);
                } else {
                    *links = std::move(local_links);
                }
#else
                *links = std::move(local_links);
#endif
            }
            FML::MinArrayOverTasks(leak_quantity.data(), leak_quantity.size());

            // Count how many regular particles we have assigned
            long long int assigned = 0;
            int nwarnings = 0;
            for (size_t i = 0; i < NumPart; i++) {
                if (label[i] != NoWatershedID) {
                    assigned++;
                } else if (nwarnings++ < 10) {
                    std::cout << "[WatershedGeneral] Warning: on task " << FML::ThisTask << " particle " << i
                              << " pos:";
                    for (int idim = 0; idim < NDIM; idim++)
                        std::cout << " " << G.position(i)[idim];
                    std::cout << " has not been assigned!\n";
                }
            }
            long long int npartotal = (long long int)(NumPart);
            FML::SumOverTasks(&assigned);
            FML::SumOverTasks(&npartotal);
            if (FML::ThisTask == 0) {
                std::cout << "[WatershedGeneral] Total particles assigned " << assigned << " / " << npartotal << "\n";
                if (assigned < npartotal) {
                    std::cout << "[WatershedGeneral] Warning we have not managed to assign all particles to their "
                                 "respective minima, cannot guarantee the results are perfect\n";
                }
            }

            // The position of the minima and the circumcenter of the minimum and its NDIM lowest-quantity mutually
            // adjacent neighbors (as in Revolver, Nadathur et al.): the first neighbor is the neighbor of the minimum
            // with the lowest quantity, the second the one with the lowest quantity among the common neighbors of
            // the minimum and the first neighbor, etc. In 3D this is the center of an empty sphere in the basin.
            // If we can't find such neighbors (or they are degenerate) we use the minimum. Make sure all tasks have
            // this
            std::vector<double> pos_minima(ntotal_minima * NDIM, 0.0);
            std::vector<double> pos_circumcenter(ntotal_minima * NDIM, 0.0);
            long long int ncircumcenter_failed = 0;
            {
                // The minima in rounds: in round n we pick the n'th neighbour of every minimum, which needs the
                // neighbours of the neighbour picked in the previous round (from its owner if it is a boundary
                // particle: G.boundary_neighbors is collective)
                struct State {
                    std::vector<size_t> chosen;
                    std::vector<size_t> candidates;
                    bool ok{true};
                };
                std::vector<State> state(minima.size());
                std::vector<size_t> work;
                for (size_t k = 0; k < minima.size(); k++) {
                    state[k].chosen = {minima[k]};
                    G.for_each_neighbor(minima[k], [&](size_t w) { state[k].candidates.push_back(w); });
                }
                for (int n = 0; n < NDIM; n++) {
                    if (n > 0) {
                        std::vector<size_t> request;
                        for (auto & s : state)
                            if (s.ok and s.chosen.back() >= NumPart)
                                request.push_back(s.chosen.back());
                        std::sort(request.begin(), request.end());
                        request.erase(std::unique(request.begin(), request.end()), request.end());
                        std::vector<std::vector<size_t>> request_neighbors;
                        G.boundary_neighbors(request, request_neighbors);
                        for (auto & s : state) {
                            if (not s.ok)
                                continue;
                            const size_t c = s.chosen.back();
                            if (c < NumPart) {
                                work.clear();
                                G.for_each_neighbor(c, [&](size_t w) { work.push_back(w); });
                            } else {
                                work = request_neighbors[size_t(std::lower_bound(request.begin(), request.end(), c) -
                                                                request.begin())];
                            }
                            std::vector<size_t> common;
                            for (auto & w : s.candidates)
                                if (std::find(work.begin(), work.end(), w) != work.end())
                                    common.push_back(w);
                            s.candidates = common;
                        }
                    }
                    for (auto & s : state) {
                        if (not s.ok)
                            continue;
                        if (s.candidates.empty()) {
                            s.ok = false;
                            continue;
                        }
                        size_t best = s.candidates[0];
                        for (auto & w : s.candidates)
                            if (lower(w, best))
                                best = w;
                        s.chosen.push_back(best);
                    }
                }

                for (size_t k = 0; k < minima.size(); k++) {
                    const size_t vmin = minima[k];
                    const double * pmin = G.position(vmin);
                    double * pos = &pos_minima[(size_t(id_start) + k) * NDIM];
                    double * circ = &pos_circumcenter[(size_t(id_start) + k) * NDIM];
                    for (int idim = 0; idim < NDIM; idim++)
                        circ[idim] = pos[idim] = pmin[idim];
                    const auto & chosen = state[k].chosen;
                    bool ok = state[k].ok;

                    // The circumcenter (positions relative to the minimum so that it works across the boundary)
                    if (ok) {
                        std::array<std::array<double, NDIM>, NDIM + 1> rel;
                        for (int j = 0; j <= NDIM; j++)
                            for (int idim = 0; idim < NDIM; idim++) {
                                double dx = G.position(chosen[j])[idim] - pmin[idim];
                                dx -= std::round(dx);
                                rel[j][idim] = dx;
                            }
                        double cc[NDIM];
                        ok = detail::circumcenter_from_origin<NDIM>(rel, cc);
                        if (ok)
                            for (int idim = 0; idim < NDIM; idim++)
                                circ[idim] = detail::wrap_periodic(pmin[idim] + cc[idim]);
                    }
                    if (not ok)
                        ncircumcenter_failed++;
                }
            }
            FML::SumOverTasks(&ncircumcenter_failed);
            if (FML::ThisTask == 0 and ncircumcenter_failed > 0)
                std::cout << "[WatershedGeneral] Warning: could not compute the circumcenter for "
                          << ncircumcenter_failed << " basins, using the position of the minimum for these\n";
            for (auto * vec : {&pos_minima, &pos_circumcenter})
                FML::SumArrayOverTasks(vec->data(), vec->size());

            // Make the groups and bin up the particles
            watershed_groups.resize(ntotal_minima);
            for (size_t k = 0; k < ntotal_minima; k++) {
                watershed_groups[k].init(&pos_minima[k * NDIM]);
                if constexpr (detail::has_set_pos_circ<U>::value)
                    watershed_groups[k].set_pos_circ(&pos_circumcenter[k * NDIM]);
                if constexpr (detail::has_set_leak_quantity<U>::value)
                    watershed_groups[k].set_leak_quantity(leak_quantity[k]);
            }
            pos_circumcenter = std::vector<double>();
            pos_minima = std::vector<double>();
            for (size_t i = 0; i < NumPart; i++)
                if (label[i] != NoWatershedID)
                    watershed_groups[label[i]].add_particle(&p[i], quantity[i]);

                    // Merge the groups from all tasks on task 0
                    // NB: assumes U is a simple type so that we can send it as bytes
#ifdef USE_MPI
            if (FML::NTasks > 1) {
                static_assert(std::is_trivially_copyable_v<U>, "The watershed group class must be trivially copyable");
                static_assert(sizeof(U) <= size_t(INT_MAX), "Watershed MPI record extent exceeds int");
                MPI_Datatype group_type;
                MPI_Type_contiguous(int(sizeof(U)), MPI_BYTE, &group_type);
                MPI_Type_commit(&group_type);
                MPI_Op merge_op;
                MPI_Op_create(&detail::merge_groups_op<U>, 1, &merge_op);
                const size_t chunk = size_t(1) << 20;
                for (size_t offset = 0; offset < ntotal_minima; offset += chunk) {
                    const int count = int(std::min(chunk, ntotal_minima - offset));
                    if (FML::ThisTask == 0)
                        MPI_Reduce(MPI_IN_PLACE,
                                   watershed_groups.data() + offset,
                                   count,
                                   group_type,
                                   merge_op,
                                   0,
                                   FML::communicator);
                    else
                        MPI_Reduce(watershed_groups.data() + offset,
                                   nullptr,
                                   count,
                                   group_type,
                                   merge_op,
                                   0,
                                   FML::communicator);
                }
                MPI_Op_free(&merge_op);
                MPI_Type_free(&group_type);
            }
#endif

            // Finalize the binning
            if (FML::ThisTask == 0) {
                for (size_t i = 0; i < ntotal_minima; i++) {
                    watershed_groups[i].finalize();
                }
            } else {
                // We only keep the data on task 0 so clear the rest to avoid any confusion
                watershed_groups.clear();
                watershed_groups.shrink_to_fit();
            }
            if (labels)
                *labels = std::move(label);
        }

        /// Density (or 1/density) of the particles from their Voronoi volumes: the quantity for the watershed
        template <class T>
        std::vector<double>
        watershed_density_quantity(T * p, const std::vector<double> & volumes, bool density_maximum) {
            std::vector<double> density(volumes.size());
            for (size_t i = 0; i < volumes.size(); i++) {
                double mass = 1.0;
                if constexpr (FML::PARTICLE::has_get_mass<T>())
                    mass = FML::PARTICLE::GetMass(p[i]);
                // To find clusters use 1/density as the quantity to watershed, for voids the density
                density[i] = density_maximum ? volumes[i] / mass : mass / volumes[i];
            }
            return density;
        }

    } // namespace TRIANGULATION
} // namespace FML
#endif
