#ifndef KDTREE_HEADER
#define KDTREE_HEADER

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {

    /// A kd-tree for nearest-neighbour and range searches (optionally in a periodic box)
    namespace KDTREE {

        //============================================================================
        /// A kd-tree of points in ndim dimensions, optionally in a periodic box [0, boxsize)^ndim (distances are then
        /// minimum-image distances). Queries:
        ///   nearest_index / nearest_point / nearest_pointIndex  : the nearest point
        ///   knearest_indices (+ squared distances)             : the k nearest points, nearest first
        ///   neighborhood_indices / _points / neighborhood       : all the points within a radius (any radius; in a
        ///                                                         periodic box each point is returned once)
        ///   neighborhood_indices_and_distances                 : the same with the squared distances
        /// Indices refer to the order of the points given to the constructor. The queries are const and can be run
        /// from several threads at once.
        ///
        /// Implementation: the coordinates are stored in one flat array (reordered so each node is a contiguous
        /// range), the nodes in another (bounding box, children); leaves hold up to leaf_size points. The split is
        /// at the median of the widest dimension of the node. In a periodic box the search prunes with the
        /// minimum-image distance between the query point and the bounding box of a node, so no periodic images of
        /// the query are needed. Memory: (ndim + 1) * 8 bytes per point plus ~(2 ndim + 3) * 8 bytes per node.
        ///
        /// History: this replaces an adaptation of https://github.com/crvs/KDTree (J. Frederico Carvalho, BSD
        /// license) with periodic images, which had one heap-allocated node per point (~180 bytes per point) and
        /// was several times slower. The public interface is the same, with the k-nearest and distance queries
        /// added (and the internal node classes removed).
        //============================================================================

        using point_t = std::vector<double>;
        using indexArr = std::vector<size_t>;
        using pointIndex = std::pair<std::vector<double>, size_t>;
        using pointIndexArr = std::vector<pointIndex>;
        using pointVec = std::vector<point_t>;
        using DVector = std::vector<double>;

        template <typename T, typename U>
        using BinningFunction = std::function<void(int iprofile, T & p1, U & p2)>;

        /// Squared euclidean distance
        inline double dist2(const point_t & a, const point_t & b) {
            double d = 0.0;
            for (size_t i = 0; i < a.size(); i++) {
                const double dx = a[i] - b[i];
                d += dx * dx;
            }
            return d;
        }
        /// Euclidean distance
        inline double dist(const point_t & a, const point_t & b) { return std::sqrt(dist2(a, b)); }

        /// x wrapped into [0, boxsize)
        inline double periodic_wrap(double x, double boxsize) {
            double y = std::fmod(x, boxsize);
            if (y < 0.0)
                y += boxsize;
            if (y >= boxsize) // y = boxsize from round-off for tiny negative x
                y -= boxsize;
            return y;
        }

        /// Squared minimum-image distance of two points in [0, boxsize)^ndim
        inline double periodic_dist(const point_t & a, const point_t & b, double boxsize) {
            double d = 0.0;
            for (size_t i = 0; i < a.size(); i++) {
                double dx = std::fabs(a[i] - b[i]);
                if (dx > 0.5 * boxsize)
                    dx = boxsize - dx;
                d += dx * dx;
            }
            return d;
        }

        class KDTree {
          public:
            KDTree() = default;
            /// A tree of the points (boxsize > 0: periodic box [0, boxsize)^ndim; the points are wrapped into it)
            explicit KDTree(const pointVec & points, double boxsize = 0.0, int leaf_size = 8);
            /// A tree of the positions of particles (anything with get_pos(); wrapped into [0, boxsize) if periodic)
            template <typename T>
            KDTree(T * particles, size_t numpart, double boxsize = 0.0, int leaf_size = 8);

            size_t size() const { return npts; }
            int get_ndim() const { return ndim; }
            double get_boxsize() const { return boxsize; }

            /// The nearest point (an empty point for an empty tree)
            point_t nearest_point(const point_t & pt) const;
            /// The index of the nearest point (size_t(-1) for an empty tree)
            size_t nearest_index(const point_t & pt) const;
            pointIndex nearest_pointIndex(const point_t & pt) const;

            /// The indices of the k nearest points, nearest first (fewer if the tree has fewer points); dist2 gets
            /// their squared distances if not null
            indexArr knearest_indices(const point_t & pt, size_t k, std::vector<double> * dist2 = nullptr) const;

            /// All the points within a distance rad (inclusive) of pt, in no particular order
            pointIndexArr neighborhood(const point_t & pt, const double & rad) const;
            pointVec neighborhood_points(const point_t & pt, const double & rad) const;
            indexArr neighborhood_indices(const point_t & pt, const double & rad) const;
            /// The same with the squared distances
            void neighborhood_indices_and_distances(const point_t & pt,
                                                    double rad,
                                                    indexArr & indices,
                                                    std::vector<double> & dist2) const;

          private:
            struct Node {
                size_t begin{0}, end{0}; // the points of the node: [begin, end) of the reordered arrays
                int left{-1}, right{-1}; // the children (-1 for a leaf)
            };

            int ndim{0};
            size_t npts{0};
            double boxsize{0.0};
            bool periodic{false};
            int leaf_size{8};
            std::vector<double> coords; // the points, reordered (point i at coords[ndim * i])
            std::vector<size_t> index;  // the original index of point i
            std::vector<Node> nodes;    // nodes[0] is the root
            std::vector<double> bbox;   // node n: lower corner at bbox[2 ndim n], upper corner after it

            void build(std::vector<double> && points);
            int build_node(size_t begin, size_t end, std::vector<size_t> & order, const std::vector<double> & points);
            point_t wrapped(const point_t & pt) const;
            double point_dist2(const double * x, const double * q) const;
            double box_dist2(int node, const double * q) const;
            template <class Visitor>
            void range_search(const double * q, double r2, Visitor && visit) const;
            void knearest(const double * q, size_t k, std::vector<std::pair<double, size_t>> & heap) const;
        };

        inline KDTree::KDTree(const pointVec & points, double boxsize_, int leaf_size_)
            : boxsize(boxsize_), periodic(boxsize_ > 0.0), leaf_size(std::max(1, leaf_size_)) {
            npts = points.size();
            ndim = npts > 0 ? int(points[0].size()) : 0;
            std::vector<double> flat(npts * size_t(ndim));
            for (size_t i = 0; i < npts; i++) {
                if (int(points[i].size()) != ndim)
                    throw std::runtime_error("[KDTree] All the points must have the same dimension");
                for (int d = 0; d < ndim; d++)
                    flat[i * ndim + d] = points[i][d];
            }
            build(std::move(flat));
        }

        template <typename T>
        KDTree::KDTree(T * part, size_t numpart, double boxsize_, int leaf_size_)
            : boxsize(boxsize_), periodic(boxsize_ > 0.0), leaf_size(std::max(1, leaf_size_)) {
            npts = numpart;
            T tmp;
            ndim = FML::PARTICLE::GetNDIM(tmp);
            std::vector<double> flat(npts * size_t(ndim));
            for (size_t i = 0; i < npts; i++) {
                const auto * pos = FML::PARTICLE::GetPos(part[i]);
                for (int d = 0; d < ndim; d++)
                    flat[i * ndim + d] = double(pos[d]);
            }
            build(std::move(flat));
        }

        inline void KDTree::build(std::vector<double> && points) {
            if (periodic)
                for (auto & x : points)
                    x = periodic_wrap(x, boxsize);
            nodes.clear();
            bbox.clear();
            coords.assign(npts * size_t(ndim), 0.0);
            index.resize(npts);
            if (npts == 0)
                return;
            std::vector<size_t> order(npts);
            std::iota(order.begin(), order.end(), size_t(0));
            nodes.reserve(4 * (npts / size_t(leaf_size) + 1));
            bbox.reserve(nodes.capacity() * 2 * size_t(ndim));
            build_node(0, npts, order, points);
            // The points in tree order
            for (size_t i = 0; i < npts; i++) {
                index[i] = order[i];
                for (int d = 0; d < ndim; d++)
                    coords[i * ndim + d] = points[order[i] * ndim + d];
            }
        }

        inline int KDTree::build_node(size_t begin,
                                      size_t end,
                                      std::vector<size_t> & order,
                                      const std::vector<double> & points) {
            const int inode = int(nodes.size());
            nodes.push_back(Node{begin, end, -1, -1});
            // Bounding box of the points of the node
            const size_t b0 = bbox.size();
            bbox.resize(b0 + 2 * size_t(ndim));
            for (int d = 0; d < ndim; d++) {
                bbox[b0 + d] = std::numeric_limits<double>::max();
                bbox[b0 + ndim + d] = std::numeric_limits<double>::lowest();
            }
            for (size_t i = begin; i < end; i++)
                for (int d = 0; d < ndim; d++) {
                    const double x = points[order[i] * ndim + d];
                    bbox[b0 + d] = std::min(bbox[b0 + d], x);
                    bbox[b0 + ndim + d] = std::max(bbox[b0 + ndim + d], x);
                }
            if (end - begin <= size_t(leaf_size))
                return inode;
            // Split at the median of the widest dimension
            int split = 0;
            double widest = -1.0;
            for (int d = 0; d < ndim; d++) {
                const double w = bbox[b0 + ndim + d] - bbox[b0 + d];
                if (w > widest) {
                    widest = w;
                    split = d;
                }
            }
            if (widest <= 0.0) // all the points are identical: keep them in one leaf
                return inode;
            const size_t mid = begin + (end - begin) / 2;
            std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end, [&](size_t a, size_t b) {
                return points[a * ndim + split] < points[b * ndim + split];
            });
            const int left = build_node(begin, mid, order, points);
            const int right = build_node(mid, end, order, points);
            nodes[inode].left = left;
            nodes[inode].right = right;
            return inode;
        }

        inline point_t KDTree::wrapped(const point_t & pt) const {
            if (int(pt.size()) != ndim and npts > 0)
                throw std::runtime_error("[KDTree] The query point has the wrong dimension");
            point_t q = pt;
            if (periodic)
                for (auto & x : q)
                    x = periodic_wrap(x, boxsize);
            return q;
        }

        inline double KDTree::point_dist2(const double * x, const double * q) const {
            double d2 = 0.0;
            for (int d = 0; d < ndim; d++) {
                double dx = std::fabs(x[d] - q[d]);
                if (periodic and dx > 0.5 * boxsize)
                    dx = boxsize - dx;
                d2 += dx * dx;
            }
            return d2;
        }

        // The smallest (minimum-image) squared distance from q to the bounding box of a node
        inline double KDTree::box_dist2(int node, const double * q) const {
            const double * lo = &bbox[2 * size_t(ndim) * node];
            const double * hi = lo + ndim;
            double d2 = 0.0;
            for (int d = 0; d < ndim; d++) {
                double dx = 0.0;
                if (q[d] < lo[d])
                    dx = lo[d] - q[d];
                else if (q[d] > hi[d])
                    dx = q[d] - hi[d];
                if (periodic and dx > 0.0) {
                    // The box and q are in [0, boxsize): the distance to the image of the box on the other side
                    const double other = q[d] < lo[d] ? q[d] + boxsize - hi[d] : lo[d] + boxsize - q[d];
                    dx = std::min(dx, std::max(other, 0.0));
                }
                d2 += dx * dx;
            }
            return d2;
        }

        template <class Visitor>
        void KDTree::range_search(const double * q, double r2, Visitor && visit) const {
            if (npts == 0)
                return;
            std::vector<int> stack;
            stack.reserve(64);
            stack.push_back(0);
            while (not stack.empty()) {
                const int n = stack.back();
                stack.pop_back();
                if (box_dist2(n, q) > r2)
                    continue;
                const Node & node = nodes[n];
                if (node.left < 0) {
                    for (size_t i = node.begin; i < node.end; i++) {
                        const double d2 = point_dist2(&coords[i * ndim], q);
                        if (d2 <= r2)
                            visit(i, d2);
                    }
                } else {
                    stack.push_back(node.left);
                    stack.push_back(node.right);
                }
            }
        }

        // The k nearest: a max-heap of (squared distance, tree position) of size <= k
        inline void KDTree::knearest(const double * q, size_t k, std::vector<std::pair<double, size_t>> & heap) const {
            heap.clear();
            if (npts == 0 or k == 0)
                return;
            auto worst = [&]() { return heap.size() < k ? std::numeric_limits<double>::max() : heap.front().first; };
            std::vector<std::pair<double, int>> stack;
            stack.reserve(64);
            stack.push_back({box_dist2(0, q), 0});
            while (not stack.empty()) {
                const auto [bd, n] = stack.back();
                stack.pop_back();
                if (heap.size() == k and bd >= worst())
                    continue;
                const Node & node = nodes[n];
                if (node.left < 0) {
                    for (size_t i = node.begin; i < node.end; i++) {
                        const double d2 = point_dist2(&coords[i * ndim], q);
                        if (heap.size() < k) {
                            heap.push_back({d2, i});
                            std::push_heap(heap.begin(), heap.end());
                        } else if (d2 < heap.front().first) {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.back() = {d2, i};
                            std::push_heap(heap.begin(), heap.end());
                        }
                    }
                } else {
                    // Visit the nearer child first (pushed last)
                    const double dl = box_dist2(node.left, q), dr = box_dist2(node.right, q);
                    if (dl <= dr) {
                        stack.push_back({dr, node.right});
                        stack.push_back({dl, node.left});
                    } else {
                        stack.push_back({dl, node.left});
                        stack.push_back({dr, node.right});
                    }
                }
            }
            std::sort_heap(heap.begin(), heap.end());
        }

        inline indexArr KDTree::knearest_indices(const point_t & pt, size_t k, std::vector<double> * dist2) const {
            const point_t q = wrapped(pt);
            std::vector<std::pair<double, size_t>> heap;
            knearest(q.data(), k, heap);
            indexArr result(heap.size());
            if (dist2)
                dist2->resize(heap.size());
            for (size_t i = 0; i < heap.size(); i++) {
                result[i] = index[heap[i].second];
                if (dist2)
                    (*dist2)[i] = heap[i].first;
            }
            return result;
        }

        inline size_t KDTree::nearest_index(const point_t & pt) const {
            auto r = knearest_indices(pt, 1);
            return r.empty() ? size_t(-1) : r[0];
        }

        inline pointIndex KDTree::nearest_pointIndex(const point_t & pt) const {
            const point_t q = wrapped(pt);
            std::vector<std::pair<double, size_t>> heap;
            knearest(q.data(), 1, heap);
            if (heap.empty())
                return pointIndex(point_t(), size_t(-1));
            const size_t i = heap[0].second;
            return pointIndex(point_t(&coords[i * ndim], &coords[i * ndim] + ndim), index[i]);
        }

        inline point_t KDTree::nearest_point(const point_t & pt) const { return nearest_pointIndex(pt).first; }

        inline void KDTree::neighborhood_indices_and_distances(const point_t & pt,
                                                               double rad,
                                                               indexArr & indices,
                                                               std::vector<double> & dist2) const {
            indices.clear();
            dist2.clear();
            const point_t q = wrapped(pt);
            range_search(q.data(), rad * rad, [&](size_t i, double d2) {
                indices.push_back(index[i]);
                dist2.push_back(d2);
            });
        }

        inline indexArr KDTree::neighborhood_indices(const point_t & pt, const double & rad) const {
            indexArr result;
            const point_t q = wrapped(pt);
            range_search(q.data(), rad * rad, [&](size_t i, double) { result.push_back(index[i]); });
            return result;
        }

        inline pointIndexArr KDTree::neighborhood(const point_t & pt, const double & rad) const {
            pointIndexArr result;
            const point_t q = wrapped(pt);
            range_search(q.data(), rad * rad, [&](size_t i, double) {
                result.emplace_back(point_t(&coords[i * ndim], &coords[i * ndim] + ndim), index[i]);
            });
            return result;
        }

        inline pointVec KDTree::neighborhood_points(const point_t & pt, const double & rad) const {
            pointVec result;
            const point_t q = wrapped(pt);
            range_search(q.data(), rad * rad, [&](size_t i, double) {
                result.emplace_back(&coords[i * ndim], &coords[i * ndim] + ndim);
            });
            return result;
        }

        //============================================================================
        /// Compute profiles (paircounts) of tracer2 around tracer1 using a kd-tree of tracer2. The binning function
        /// is called for every pair (i, tracer1[i], tracer2[j]) with |x1 - x2| <= rmax. Positions in [0,1), i.e.
        /// boxsize = 1 if periodic. Uses threads (make sure the binning function has no races, e.g. only writes to
        /// the profile i) and MPI (each task does every NTasks-th tracer1; reduce the results afterwards)
        //============================================================================
        template <typename T, typename U>
        void GenericPairBinner(T * tracer1,
                               size_t npart1,
                               U * tracer2,
                               size_t npart2,
                               double rmax,
                               bool periodic,
                               BinningFunction<T, U> & binning_function) {
            T tmp1; // (GetNDIM(T()) needs a particle that can be built at compile time)
            U tmp2;
            constexpr int ndim1 = FML::PARTICLE::GetNDIM(tmp1);
            constexpr int ndim2 = FML::PARTICLE::GetNDIM(tmp2);
            static_assert(ndim1 == ndim2, "[GenericPairBinner] The tracers must have the same dimension");

            const KDTree tree(tracer2, npart2, periodic ? 1.0 : 0.0);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
            for (size_t i = FML::ThisTask; i < npart1; i += FML::NTasks) {
                const auto * pos1 = FML::PARTICLE::GetPos(tracer1[i]);
                const point_t pt(pos1, pos1 + ndim1);
                for (auto index : tree.neighborhood_indices(pt, rmax))
                    binning_function(int(i), tracer1[i], tracer2[index]);
            }
        }

        //============================================================================
        /// The individual (linearly spaced bins) density profiles of tracer2 around each tracer1, in units of the
        /// mean density of tracer2 (npart2 / volume with volume = 1: positions in [0,1)). NB: without a periodic box
        /// there is no correction for the edges (profiles near the edges are biased low)
        //============================================================================
        template <typename T, typename U>
        void DensityProfileBinner(T * tracer1,
                                  size_t npart1,
                                  U * tracer2,
                                  size_t npart2,
                                  double rmin,
                                  double rmax,
                                  int nbins,
                                  bool periodic,
                                  DVector & rbin_center,
                                  std::vector<DVector> & individual_density_profiles) {

            const double rmin_squared = rmin * rmin;
            const double rmax_squared = rmax * rmax;
            T tmp1; // (GetNDIM(T()) needs a particle that can be built at compile time)
            U tmp2;
            constexpr int ndim = FML::PARTICLE::GetNDIM(tmp1);
            static_assert(ndim == FML::PARTICLE::GetNDIM(tmp2),
                          "[DensityProfileBinner] The tracers must have the same dimension");

            // Set up binning
            DVector rbin_edge(nbins + 1, 0.0);
            rbin_center = DVector(nbins, 0.0);
            for (int i = 0; i < nbins; i++) {
                rbin_edge[i] = rmin + (rmax - rmin) * i / nbins;
                rbin_center[i] = rmin + (rmax - rmin) * (i + 0.5) / nbins;
            }
            rbin_edge[nbins] = rmax;

            // The counts of all the profiles in one array (profile i at i * nbins)
            DVector counts(npart1 * size_t(nbins), 0.0);

            // NB: with OpenMP there are no races as each thread only writes to its own profiles
            BinningFunction<T, U> binning_function = [&](int iprofile, T & p1, U & p2) {
                const auto * pos1 = FML::PARTICLE::GetPos(p1);
                const auto * pos2 = FML::PARTICLE::GetPos(p2);
                double dist_squared = 0.0;
                for (int idim = 0; idim < ndim; idim++) {
                    double dx = pos1[idim] - pos2[idim];
                    if (periodic) {
                        if (dx < -0.5)
                            dx += 1.0;
                        if (dx > 0.5)
                            dx -= 1.0;
                    }
                    dist_squared += dx * dx;
                }
                if (dist_squared >= rmax_squared or dist_squared < rmin_squared)
                    return;
                const double r = std::sqrt(dist_squared);
                const int bin_index = std::min(int((r - rmin) / (rmax - rmin) * nbins), nbins - 1); // rounding at rmax
                counts[size_t(iprofile) * nbins + bin_index] += 1.0;
            };

            GenericPairBinner(tracer1, npart1, tracer2, npart2, rmax, periodic, binning_function);

#ifdef USE_MPI
            // One reduction of all the profiles (in chunks that fit in an int)
            const size_t chunk = size_t(std::numeric_limits<int>::max() / 2);
            for (size_t start = 0; start < counts.size(); start += chunk)
                MPI_Allreduce(MPI_IN_PLACE,
                              counts.data() + start,
                              int(std::min(chunk, counts.size() - start)),
                              MPI_DOUBLE,
                              MPI_SUM,
                              FML::communicator);
#endif

            individual_density_profiles = std::vector<DVector>(npart1, DVector(nbins, 0.0));
            const double rho2_average = double(npart2);
            for (int bin_index = 0; bin_index < nbins; bin_index++) {
                const double volume_bin =
                    std::pow(M_PI, ndim / 2.0) / std::tgamma(ndim / 2.0 + 1.0) *
                    (std::pow(rbin_edge[bin_index + 1], ndim) - std::pow(rbin_edge[bin_index], ndim));
                const double expected2_in_bin = rho2_average * volume_bin;
                for (size_t i = 0; i < npart1; i++)
                    individual_density_profiles[i][bin_index] = counts[i * nbins + bin_index] / expected2_in_bin;
            }
        }

    } // namespace KDTREE
} // namespace FML
#endif
