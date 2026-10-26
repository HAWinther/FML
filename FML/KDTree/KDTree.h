#ifndef KDTREE_HEADER
#define KDTREE_HEADER

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {
    namespace KDTREE {

        //============================================================================
        // The KDTREE is an adaptation of https://github.com/crvs/KDTree by J. Frederico Carvalho
        // which again is an adaptation of the KD-tree implementation in rosetta code
        // https://rosettacode.org/wiki/K-d_tree
        //
        // The main modification here is to include periodic boxes
        //
        // If a boxsize is provided we assume a periodic box
        // and the particles will be wrapped inside [0, boxsize]^ndim
        // Searches are done by checking all relevant periodic images
        //============================================================================

        using point_t = std::vector<double>;
        using indexArr = std::vector<size_t>;
        using pointIndex = typename std::pair<std::vector<double>, size_t>;
        using DVector = std::vector<double>;

        template <typename T, typename U>
        using BinningFunction = std::function<void(int iprofile, T & p1, U & p2)>;

        //============================================================================
        // Compute profiles (paircounts) of tracer1 around tracer2 using a kdtree to speed it up
        // The binning function takes care of the binning.
        // Assumes position in [0,1], i.e. boxsize = 1.0
        // Uses threads (make sure binning functions do not lead to races)
        // Uses MPI (each task does a part of the profiles)
        //============================================================================
        template <typename T, typename U>
        void GenericPairBinner(T * tracer1,
                               size_t npart1,
                               U * tracer2,
                               size_t npart2,
                               double rmax,
                               bool periodic,
                               BinningFunction<T, U> & binning_function);

        //============================================================================
        // This method computes the individual (linearly spaced bins)
        // density profile of tracer2 around tracer1.
        // The density profile is normalized in terms of the mean density
        // of tracer2. This is an example of using the generic profile binner.
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
                                  std::vector<DVector> & individual_density_profiles);

        class KDNode {
          public:
            using KDNodePtr = std::shared_ptr<KDNode>;
            size_t index;
            point_t x;
            KDNodePtr left;
            KDNodePtr right;

            // initializer
            KDNode();
            KDNode(const point_t &, const size_t &, const KDNodePtr &, const KDNodePtr &);
            KDNode(const pointIndex &, const KDNodePtr &, const KDNodePtr &);
            ~KDNode();

            // getter
            double coord(const size_t &);

            // conversions
            explicit operator bool();
            explicit operator point_t();
            explicit operator size_t();
            explicit operator pointIndex();
        };

        using KDNodePtr = std::shared_ptr<KDNode>;

        inline KDNodePtr NewKDNodePtr();

        // Square euclidean distance
        inline double dist2(const point_t &, const point_t &);
        inline double dist2(const KDNodePtr &, const KDNodePtr &);

        // Euclidean distance
        inline double dist(const point_t &, const point_t &);
        inline double dist(const KDNodePtr &, const KDNodePtr &);

        // Need for sorting
        class comparer {
          public:
            size_t idx;
            explicit comparer(size_t idx_);
            inline bool compare_idx(const std::pair<std::vector<double>, size_t> &,
                                    const std::pair<std::vector<double>, size_t> &);
        };

        using pointIndexArr = typename std::vector<pointIndex>;

        inline void sort_on_idx(const pointIndexArr::iterator &, const pointIndexArr::iterator &, size_t idx);

        using pointVec = std::vector<point_t>;

        class KDTree {
            KDNodePtr root;
            KDNodePtr leaf;

            // Modifications for a periodic box
            double boxsize{0.0};
            bool periodic{false};
            int ndim{};
            int twotondim{};

            KDNodePtr make_tree(const pointIndexArr::iterator & begin,
                                const pointIndexArr::iterator & end,
                                const size_t & length,
                                const size_t & level);

          public:
            KDTree() = default;
            explicit KDTree(pointVec point_array, double boxsize = 0.0);
            template <typename T>
            explicit KDTree(T * particles, size_t numpart, double boxsize = 0.0);

          private:
            KDNodePtr nearest_(const KDNodePtr & branch,
                               const point_t & pt,
                               const size_t & level,
                               const KDNodePtr & best,
                               const double & best_dist);

            // default caller
            KDNodePtr nearest_(const point_t & pt);

          public:
            point_t nearest_point(const point_t & pt);
            size_t nearest_index(const point_t & pt);
            pointIndex nearest_pointIndex(const point_t & pt);

          private:
            pointIndexArr
            neighborhood_(const KDNodePtr & branch, const point_t & pt, const double & rad, const size_t & level);

          public:
            pointIndexArr neighborhood(const point_t & pt, const double & rad);

            pointVec neighborhood_points(const point_t & pt, const double & rad);

            indexArr neighborhood_indices(const point_t & pt, const double & rad);
        };

        inline KDNode::KDNode() = default;

        inline KDNode::KDNode(const point_t & pt, const size_t & idx_, const KDNodePtr & left_, const KDNodePtr & right_) {
            x = pt;
            index = idx_;
            left = left_;
            right = right_;
        }

        inline KDNode::KDNode(const pointIndex & pi, const KDNodePtr & left_, const KDNodePtr & right_) {
            x = pi.first;
            index = pi.second;
            left = left_;
            right = right_;
        }

        inline KDNode::~KDNode() = default;

        inline double KDNode::coord(const size_t & idx) { return x.at(idx); }
        inline KDNode::operator bool() { return (!x.empty()); }
        inline KDNode::operator point_t() { return x; }
        inline KDNode::operator size_t() { return index; }
        inline KDNode::operator pointIndex() { return pointIndex(x, index); }

        inline KDNodePtr NewKDNodePtr() {
            KDNodePtr mynode = std::make_shared<KDNode>();
            return mynode;
        }

        inline double dist2(const point_t & a, const point_t & b) {
            double distc = 0;
            for (size_t i = 0; i < a.size(); i++) {
                double di = a.at(i) - b.at(i);
                distc += di * di;
            }
            return distc;
        }

        inline double dist2(const KDNodePtr & a, const KDNodePtr & b) { return dist2(a->x, b->x); }

        inline double dist(const point_t & a, const point_t & b) { return std::sqrt(dist2(a, b)); }

        // Distance within a periodic box
        inline double periodic_dist(const point_t & a, const point_t & b, double boxsize) {
            double dist = 0.0;
            for (size_t idim = 0; idim < a.size(); idim++) {
                double dx = a[idim] - b[idim];
                if (dx > boxsize / 2)
                    dx -= boxsize;
                if (dx < -boxsize / 2)
                    dx += boxsize;
                dist += dx * dx;
            }
            return dist;
        }

        inline double dist(const KDNodePtr & a, const KDNodePtr & b) { return std::sqrt(dist2(a, b)); }

        inline comparer::comparer(size_t idx_) : idx{idx_} {};

        inline bool comparer::compare_idx(const pointIndex & a, const pointIndex & b) {
            return (a.first.at(idx) < b.first.at(idx));
        }

        inline void
        sort_on_idx(const pointIndexArr::iterator & begin, const pointIndexArr::iterator & end, size_t idx) {
            comparer comp(idx);
            comp.idx = idx;

            using std::placeholders::_1;
            using std::placeholders::_2;

            std::nth_element(
                begin, begin + std::distance(begin, end) / 2, end, std::bind(&comparer::compare_idx, comp, _1, _2));
        }

        using pointVec = std::vector<point_t>;

        inline KDNodePtr KDTree::make_tree(const pointIndexArr::iterator & begin,
                                    const pointIndexArr::iterator & end,
                                    const size_t & length,
                                    const size_t & level) {
            if (begin == end) {
                return NewKDNodePtr(); // empty tree
            }

            size_t dim = begin->first.size();

            if (length > 1) {
                sort_on_idx(begin, end, level);
            }

            auto middle = begin + (length / 2);

            auto l_begin = begin;
            auto l_end = middle;
            auto r_begin = middle + 1;
            auto r_end = end;

            size_t l_len = length / 2;
            size_t r_len = length - l_len - 1;

            KDNodePtr left;
            if (l_len > 0 && dim > 0) {
                left = make_tree(l_begin, l_end, l_len, (level + 1) % dim);
            } else {
                left = leaf;
            }
            KDNodePtr right;
            if (r_len > 0 && dim > 0) {
                right = make_tree(r_begin, r_end, r_len, (level + 1) % dim);
            } else {
                right = leaf;
            }

            return std::make_shared<KDNode>(*middle, left, right);
        }

        inline double periodic_wrap(double x, double boxsize) {
            double y = std::fmod(x, boxsize);
            if (y < 0.0)
                y += boxsize;
            if (y >= boxsize) // y = boxsize from round-off for tiny negative x
                y -= boxsize;
            return y;
        }

        template <typename T>
        KDTree::KDTree(T * part, size_t numpart, double boxsize) : boxsize(boxsize), periodic(boxsize != 0.0) {
            ndim = FML::PARTICLE::GetNDIM(T());
            twotondim = FML::power(2, ndim);
            leaf = std::make_shared<KDNode>();

            // iterators
            pointIndexArr arr;
            for (size_t i = 0; i < numpart; i++) {
                auto pos = FML::PARTICLE::GetPos(part[i]);
                point_t pt(ndim);
                for (int idim = 0; idim < ndim; idim++) {
                    pt[idim] = pos[idim];
                    if (periodic)
                        pt[idim] = periodic_wrap(pt[idim], boxsize);
                }
                arr.push_back(pointIndex(pt, i));
            }

            auto begin = arr.begin();
            auto end = arr.end();

            size_t length = arr.size();
            size_t level = 0; // starting

            root = KDTree::make_tree(begin, end, length, level);
        }

        inline KDTree::KDTree(pointVec point_array, double boxsize) : boxsize(boxsize), periodic(boxsize != 0.0) {
            leaf = std::make_shared<KDNode>();

            // Store ndim and 2^ndim
            if (point_array.size() > 0) {
                ndim = point_array[0].size();
                twotondim = 1;
                for (int idim = 0; idim < ndim; idim++)
                    twotondim *= 2;
            }

            // iterators
            pointIndexArr arr;
            for (size_t i = 0; i < point_array.size(); i++) {
                auto pt = point_array.at(i);

                // Ensure all points are inside the box
                if (periodic) {
                    for (auto & xi : pt)
                        xi = periodic_wrap(xi, boxsize);
                }

                arr.push_back(pointIndex(pt, i));
            }

            auto begin = arr.begin();
            auto end = arr.end();

            size_t length = arr.size();
            size_t level = 0; // starting

            root = KDTree::make_tree(begin, end, length, level);
        }

        inline KDNodePtr KDTree::nearest_(const KDNodePtr & branch,
                                   const point_t & pt,
                                   const size_t & level,
                                   const KDNodePtr & best,
                                   const double & best_dist) {
            double d, dx, dx2;

            if (!bool(*branch)) {
                return NewKDNodePtr(); // basically, null
            }

            point_t branch_pt(*branch);
            size_t dim = branch_pt.size();

            d = dist2(branch_pt, pt);
            dx = branch_pt.at(level) - pt.at(level);
            dx2 = dx * dx;

            KDNodePtr best_l = best;
            double best_dist_l = best_dist;

            if (d < best_dist) {
                best_dist_l = d;
                best_l = branch;
            }

            size_t next_lv = (level + 1) % dim;
            KDNodePtr section;
            KDNodePtr other;

            // select which branch makes sense to check
            if (dx > 0) {
                section = branch->left;
                other = branch->right;
            } else {
                section = branch->right;
                other = branch->left;
            }

            // keep nearest neighbor from further down the tree
            KDNodePtr further = nearest_(section, pt, next_lv, best_l, best_dist_l);
            if (!further->x.empty()) {
                double dl = dist2(further->x, pt);
                if (dl < best_dist_l) {
                    best_dist_l = dl;
                    best_l = further;
                }
            }
            // only check the other branch if it makes sense to do so
            if (dx2 < best_dist_l) {
                further = nearest_(other, pt, next_lv, best_l, best_dist_l);
                if (!further->x.empty()) {
                    double dl = dist2(further->x, pt);
                    if (dl < best_dist_l) {
                        best_dist_l = dl;
                        best_l = further;
                    }
                }
            }

            return best_l;
        };

        // default caller
        inline KDNodePtr KDTree::nearest_(const point_t & pt_in) {
            size_t level = 0;
            if (!root or !bool(*root))
                return NewKDNodePtr(); // empty tree

            if (not periodic) {
                double branch_dist = dist2(point_t(*root), pt_in);
                return nearest_(root, pt_in, level, root, branch_dist);
            }

            // Periodic box: first the query point itself (wrapped into the box), then only the periodic images that
            // can contain a point nearer than the best found so far (the bound is passed on to the searches)
            point_t pt = pt_in;
            for (auto & xi : pt)
                xi = periodic_wrap(xi, boxsize);
            KDNodePtr best = nearest_(root, pt, level, root, dist2(point_t(*root), pt));
            double best_dist = periodic_dist(point_t(*best), pt, boxsize);

            // For each direction the jump to the nearest periodic image and the distance to that boundary
            point_t jump(ndim), dist_boundary(ndim);
            for (int idim = 0; idim < ndim; idim++) {
                jump[idim] = pt[idim] > boxsize / 2 ? -boxsize : +boxsize;
                dist_boundary[idim] = pt[idim] > boxsize / 2 ? boxsize - pt[idim] : pt[idim];
            }
            for (int i = 1; i < twotondim; i++) {
                point_t search_pt = pt;
                double min_dist2 = 0.0; // the smallest possible distance to a point in this image
                for (int idim = 0, n = 1; idim < ndim; idim++, n *= 2) {
                    if (i / n % 2) {
                        search_pt[idim] += jump[idim];
                        min_dist2 += dist_boundary[idim] * dist_boundary[idim];
                    }
                }
                if (min_dist2 >= best_dist)
                    continue;
                auto candidate = nearest_(root, search_pt, level, best, best_dist);
                double d = periodic_dist(point_t(*candidate), pt, boxsize);
                if (d < best_dist) {
                    best_dist = d;
                    best = candidate;
                }
            }
            return best;
        };

        inline point_t KDTree::nearest_point(const point_t & pt) { return point_t(*nearest_(pt)); };
        inline size_t KDTree::nearest_index(const point_t & pt) { return size_t(*nearest_(pt)); };

        inline pointIndex KDTree::nearest_pointIndex(const point_t & pt) {
            KDNodePtr Nearest = nearest_(pt);
            return pointIndex(point_t(*Nearest), size_t(*Nearest));
        }

        inline pointIndexArr
        KDTree::neighborhood_(const KDNodePtr & branch, const point_t & pt, const double & rad, const size_t & level) {
            double d, dx, dx2;

            if (!bool(*branch)) {
                // branch has no point, means it is a leaf,
                // no points to add
                return pointIndexArr();
            }

            size_t dim = pt.size();

            double r2 = rad * rad;

            d = dist2(point_t(*branch), pt);
            dx = point_t(*branch).at(level) - pt.at(level);
            dx2 = dx * dx;

            pointIndexArr nbh, nbh_s, nbh_o;
            if (d <= r2) {
                nbh.push_back(pointIndex(*branch));
            }

            KDNodePtr section;
            KDNodePtr other;
            if (dx > 0) {
                section = branch->left;
                other = branch->right;
            } else {
                section = branch->right;
                other = branch->left;
            }

            nbh_s = neighborhood_(section, pt, rad, (level + 1) % dim);
            nbh.insert(nbh.end(), nbh_s.begin(), nbh_s.end());
            if (dx2 < r2) {
                nbh_o = neighborhood_(other, pt, rad, (level + 1) % dim);
                nbh.insert(nbh.end(), nbh_o.begin(), nbh_o.end());
            }

            return nbh;
        };

        // All the points within a distance rad of pt (the minimum-image distance in a periodic box). Each point is
        // returned once, also for rad > boxsize/2 (then all the 3^ndim neighbouring images are searched)
        inline pointIndexArr KDTree::neighborhood(const point_t & pt_in, const double & rad) {
            size_t level = 0;
            if (!root or !bool(*root))
                return pointIndexArr();
            if (not periodic)
                return neighborhood_(root, pt_in, rad, level);

            point_t pt = pt_in;
            for (auto & xi : pt)
                xi = periodic_wrap(xi, boxsize);

            if (rad <= boxsize / 2) {
                // At most one image per direction: the one across the nearest boundary (if within rad). A point
                // can then not be found in two images (they are a box size apart and 2 rad <= boxsize), except
                // exactly at rad = boxsize / 2 which we check below
                pointIndexArr nbh;
                for (int i = 0; i < twotondim; i++) {
                    point_t search_pt = pt;
                    bool skip = false;
                    for (int idim = 0, n = 1; idim < ndim; idim++, n *= 2) {
                        if (i / n % 2 == 0)
                            continue;
                        const double dist_from_boundary = pt[idim] < boxsize / 2 ? pt[idim] : boxsize - pt[idim];
                        if (dist_from_boundary > rad) {
                            skip = true;
                            break;
                        }
                        search_pt[idim] += pt[idim] > boxsize / 2 ? -boxsize : +boxsize;
                    }
                    if (skip)
                        continue;
                    auto part = neighborhood_(root, search_pt, rad, level);
                    nbh.insert(nbh.end(), part.begin(), part.end());
                }
                if (rad < boxsize / 2)
                    return nbh;
                std::sort(nbh.begin(), nbh.end(), [](const pointIndex & a, const pointIndex & b) { return a.second < b.second; });
                nbh.erase(std::unique(nbh.begin(), nbh.end(), [](const pointIndex & a, const pointIndex & b) { return a.second == b.second; }), nbh.end());
                return nbh;
            }

            // rad > boxsize / 2: search all the 3^ndim images, keep the points within rad (minimum image) once
            int threetondim = 1;
            for (int idim = 0; idim < ndim; idim++)
                threetondim *= 3;
            pointIndexArr nbh;
            for (int i = 0; i < threetondim; i++) {
                point_t search_pt = pt;
                for (int idim = 0, n = 1; idim < ndim; idim++, n *= 3)
                    search_pt[idim] += (i / n % 3 - 1) * boxsize;
                auto part = neighborhood_(root, search_pt, rad, level);
                nbh.insert(nbh.end(), part.begin(), part.end());
            }
            std::sort(nbh.begin(), nbh.end(), [](const pointIndex & a, const pointIndex & b) { return a.second < b.second; });
            nbh.erase(std::unique(nbh.begin(), nbh.end(), [](const pointIndex & a, const pointIndex & b) { return a.second == b.second; }), nbh.end());
            const double r2 = rad * rad;
            nbh.erase(std::remove_if(nbh.begin(), nbh.end(), [&](const pointIndex & p) { return periodic_dist(p.first, pt, boxsize) > r2; }), nbh.end());
            return nbh;
        }

        inline pointVec KDTree::neighborhood_points(const point_t & pt, const double & rad) {
            pointIndexArr nbh = neighborhood(pt, rad);
            pointVec nbhp(nbh.size());
            std::transform(nbh.begin(), nbh.end(), nbhp.begin(), [](const pointIndex & x) { return x.first; });
            return nbhp;
        }

        inline indexArr KDTree::neighborhood_indices(const point_t & pt, const double & rad) {
            pointIndexArr nbh = neighborhood(pt, rad);
            indexArr nbhi(nbh.size());
            std::transform(nbh.begin(), nbh.end(), nbhi.begin(), [](const pointIndex & x) { return x.second; });
            return nbhi;
        }

        template <typename T, typename U>
        void GenericPairBinner(T * tracer1,
                               size_t npart1,
                               U * tracer2,
                               size_t npart2,
                               double rmax,
                               bool periodic,
                               BinningFunction<T, U> & binning_function) {

            constexpr int ndim1 = FML::PARTICLE::GetNDIM(T());
            constexpr int ndim2 = FML::PARTICLE::GetNDIM(U());
            assert(ndim1 == ndim2);

            // Make kd-tree of tracer2 (galaxies)
            // If boxsize > 0.0 then the tree will be periodic
            double boxsize = periodic ? 1.0 : 0.0;
            const double radius = rmax;
            KDTree tree(tracer2, npart2, boxsize);

            // Loop over all tracers1
            // With MPI each task does their part of the full profiles
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (size_t i = FML::ThisTask; i < npart1; i += FML::NTasks) {
                auto pos1 = FML::PARTICLE::GetPos(tracer1[i]);
                point_t pt = point_t(pos1, pos1 + ndim1);

                // Search in tree
                auto nbor_index = tree.neighborhood_indices(pt, radius);

                // Loop over all tracers2 we find
                for (size_t j = 0; j < nbor_index.size(); j++) {
                    auto index = nbor_index[j];
                    // Callback to the function doing the binning
                    binning_function(i, tracer1[i], tracer2[index]);
                }
            }
        }

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

            const int ndim1 = FML::PARTICLE::GetNDIM(T());
            const int ndim2 = FML::PARTICLE::GetNDIM(U());
            assert(ndim1 == ndim2);

            // Set up binning
            DVector rbin_edge(nbins + 1, 0.0);
            rbin_center = DVector(nbins, 0.0);
            for (int i = 0; i < nbins; i++) {
                rbin_edge[i] = rmin + (rmax - rmin) * i / nbins;
                rbin_center[i] = rmin + (rmax - rmin) * (i + 0.5) / nbins;
            }
            rbin_edge[nbins] = rmax;

            // Individual bincount
            std::vector<DVector> bincount_individual(npart1, DVector(nbins, 0.0));

            // Set up the binning function
            // NB: with OpenMP we ensure there are no races by stacking profile by profile
            BinningFunction<T, U> binning_function = [&](int iprofile, T & p1, U & p2) {
                auto pos1 = FML::PARTICLE::GetPos(p1);
                auto pos2 = FML::PARTICLE::GetPos(p2);

                // Distance
                double dist_squared = 0.0;
                for (int idim = 0; idim < ndim1; idim++) {
                    double dx = pos1[idim] - pos2[idim];
                    if (periodic) {
                        if (dx < -0.5)
                            dx += 1.0;
                        if (dx > 0.5)
                            dx -= 1.0;
                    }
                    dist_squared += dx * dx;
                }
                if (dist_squared >= rmax_squared)
                    return;
                if (dist_squared < rmin_squared)
                    return;

                double r = std::sqrt(dist_squared);
                int bin_index = std::min(int((r - rmin) / (rmax - rmin) * nbins), nbins - 1); // rounding at r ~ rmax

                // Bin up the stuff we want...
                // Here we just do the bincount
                bincount_individual[iprofile][bin_index] += 1.0;
            };

            // Do the binning
            GenericPairBinner(tracer1, npart1, tracer2, npart2, rmax, periodic, binning_function);

#ifdef USE_MPI
            for (size_t i = 0; i < bincount_individual.size(); i++)
                MPI_Allreduce(MPI_IN_PLACE,
                              bincount_individual[i].data(),
                              bincount_individual[i].size(),
                              MPI_DOUBLE,
                              MPI_SUM,
                              FML::communicator);
#endif

            // Create density profiles
            individual_density_profiles = std::vector<DVector>(npart1, DVector(nbins, 0.0));
            for (int bin_index = 0; bin_index < nbins; bin_index++) {
                [[maybe_unused]] double r = rbin_center[bin_index];

                double rho2_average = npart2;
                double volume_bin = std::pow(M_PI, ndim2 / 2.0) / std::tgamma(ndim2 / 2.0 + 1.0) *
                                    (std::pow(rbin_edge[bin_index + 1], ndim2) - std::pow(rbin_edge[bin_index], ndim2));

                // Do individual density profiles
                double total_count = 0.0;
                for (size_t i = 0; i < npart1; i++) {
                    double count = bincount_individual[i][bin_index];
                    double expected2_in_bin = rho2_average * volume_bin;
                    double rho_over_rhomean = count / expected2_in_bin;
                    individual_density_profiles[i][bin_index] = rho_over_rhomean;
                    total_count += count;
                }
            }
        }

    } // namespace KDTREE
} // namespace FML
#endif
