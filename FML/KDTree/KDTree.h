#ifndef KDTREE_HEADER
#define KDTREE_HEADER

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>
#include <iostream>

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

    using point_t = std::vector< double >;
    using indexArr = std::vector< size_t >;
    using pointIndex = typename std::pair< std::vector< double >, size_t >;

    class KDNode {
      public:
        using KDNodePtr = std::shared_ptr< KDNode >;
        size_t index;
        point_t x;
        KDNodePtr left;
        KDNodePtr right;

        // initializer
        KDNode();
        KDNode(const point_t &, const size_t &, const KDNodePtr &,
            const KDNodePtr &);
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

    using KDNodePtr = std::shared_ptr< KDNode >;

    KDNodePtr NewKDNodePtr();

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
        inline bool compare_idx(
            const std::pair< std::vector< double >, size_t > &, 
            const std::pair< std::vector< double >, size_t > &  
            );
    };

    using pointIndexArr = typename std::vector< pointIndex >;

    inline void sort_on_idx(const pointIndexArr::iterator &, 
        const pointIndexArr::iterator &, 
        size_t idx);

    using pointVec = std::vector< point_t >;

    class KDTree {
      KDNodePtr root;
      KDNodePtr leaf;

      // Modifications for a periodic box
      double boxsize{0.0};
      bool periodic{false};
      int ndim{};
      int twotondim{};

      KDNodePtr make_tree(const pointIndexArr::iterator &begin, 
          const pointIndexArr::iterator &end,    
          const size_t &length,                 
          const size_t &level                   
          );

      public:
      KDTree() = default;
      explicit KDTree(pointVec point_array, double boxsize = 0.0);
      template<typename T>
        explicit KDTree(T *particles, size_t numpart, double boxsize = 0.0);

      private:
      KDNodePtr nearest_(           
          const KDNodePtr &branch,  
          const point_t &pt,        
          const size_t &level,      
          const KDNodePtr &best,    
          const double &best_dist   
          );

      // default caller
      KDNodePtr nearest_(const point_t &pt);

      public:
      point_t nearest_point(const point_t &pt);
      size_t nearest_index(const point_t &pt);
      pointIndex nearest_pointIndex(const point_t &pt);

      private:
      pointIndexArr neighborhood_(  
          const KDNodePtr &branch,  
          const point_t &pt,        
          const double &rad,        
          const size_t &level       
          );

      public:
      pointIndexArr neighborhood(  
          const point_t &pt,       
          const double &rad);

      pointVec neighborhood_points(  
          const point_t &pt,         
          const double &rad);

      indexArr neighborhood_indices( 
          const point_t &pt,         
          const double &rad);
    };

    KDNode::KDNode() = default;

    KDNode::KDNode(const point_t &pt, const size_t &idx_, const KDNodePtr &left_,
        const KDNodePtr &right_) {
      x = pt;
      index = idx_;
      left = left_;
      right = right_;
    }

    KDNode::KDNode(const pointIndex &pi, const KDNodePtr &left_,
        const KDNodePtr &right_) {
      x = pi.first;
      index = pi.second;
      left = left_;
      right = right_;
    }

    KDNode::~KDNode() = default;

    double KDNode::coord(const size_t &idx) { return x.at(idx); }
    KDNode::operator bool() { return (!x.empty()); }
    KDNode::operator point_t() { return x; }
    KDNode::operator size_t() { return index; }
    KDNode::operator pointIndex() { return pointIndex(x, index); }

    KDNodePtr NewKDNodePtr() {
      KDNodePtr mynode = std::make_shared< KDNode >();
      return mynode;
    }

    inline double dist2(const point_t &a, const point_t &b) {
      double distc = 0;
      for (size_t i = 0; i < a.size(); i++) {
        double di = a.at(i) - b.at(i);
        distc += di * di;
      }
      return distc;
    }

    inline double dist2(const KDNodePtr &a, const KDNodePtr &b) {
      return dist2(a->x, b->x);
    }

    inline double dist(const point_t &a, const point_t &b) {
      return std::sqrt(dist2(a, b));
    }

    // Distance within a periodic box
    inline double periodic_dist(const point_t &a, const point_t &b, double boxsize) {
      double dist = 0.0;
      for(size_t idim = 0; idim < a.size(); idim++){
        double dx = a[idim]-b[idim];
        if(dx > boxsize/2)
          dx -= boxsize;
        if(dx < -boxsize/2)
          dx += boxsize;
        dist += dx*dx;
      }
      return dist;
    }

    inline double dist(const KDNodePtr &a, const KDNodePtr &b) {
      return std::sqrt(dist2(a, b));
    }

    comparer::comparer(size_t idx_) : idx{idx_} {};

    inline bool comparer::compare_idx(const pointIndex &a,
        const pointIndex &b 
        ) {
      return (a.first.at(idx) < b.first.at(idx));
    }

    inline void sort_on_idx(const pointIndexArr::iterator &begin,
        const pointIndexArr::iterator &end, 
        size_t idx) {
      comparer comp(idx);
      comp.idx = idx;

      using std::placeholders::_1;
      using std::placeholders::_2;

      std::nth_element(begin, begin + std::distance(begin, end) / 2,
          end, std::bind(&comparer::compare_idx, comp, _1, _2));
    }

    using pointVec = std::vector< point_t >;

    KDNodePtr KDTree::make_tree(const pointIndexArr::iterator &begin,
        const pointIndexArr::iterator &end,   
        const size_t &length,               
        const size_t &level        
        ) {
      if (begin == end) {
        return NewKDNodePtr();  // empty tree
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

      return std::make_shared< KDNode >(*middle, left, right);
    }

    inline double periodic_wrap(double x, double boxsize){
      if(x < 0.0) return boxsize + std::fmod(x, boxsize);
      return std::fmod(x, boxsize);
    }
    
    template<typename T>
    KDTree::KDTree(T* part, size_t numpart, double boxsize) : boxsize(boxsize), periodic(boxsize != 0.0) {
      ndim = FML::PARTICLE::GetNDIM(T());
      twotondim = FML::power(2, ndim);
      leaf = std::make_shared< KDNode >();
    
      // iterators
      pointIndexArr arr;
      for (size_t i = 0; i < numpart; i++) {
        auto pos = FML::PARTICLE::GetPos(part[i]); 
        point_t pt(ndim);
        for(int idim = 0; idim < ndim; idim++){
          pt[idim] = pos[idim];
          if(periodic)  
            pt[idim] = periodic_wrap(pt[idim], boxsize);
        }
        arr.push_back(pointIndex(pt, i));
      }

      auto begin = arr.begin();
      auto end = arr.end();

      size_t length = arr.size();
      size_t level = 0;  // starting

      root = KDTree::make_tree(begin, end, length, level);

    }

    KDTree::KDTree(pointVec point_array, double boxsize) : boxsize(boxsize), periodic(boxsize != 0.0) {
      leaf = std::make_shared< KDNode >();

      // Store ndim and 2^ndim
      if(point_array.size() > 0){
        ndim = point_array[0].size();
        twotondim = 1;
        for(int idim = 0; idim < ndim; idim++)
          twotondim *= 2;
      }

      // iterators
      pointIndexArr arr;
      for (size_t i = 0; i < point_array.size(); i++) {
        auto pt = point_array.at(i);

        // Ensure all points are inside the box
        if(periodic){
          for(auto & xi : pt)
            xi = periodic_wrap(xi, boxsize);
        }

        arr.push_back(pointIndex(pt, i));
      }

      auto begin = arr.begin();
      auto end = arr.end();

      size_t length = arr.size();
      size_t level = 0;  // starting

      root = KDTree::make_tree(begin, end, length, level);
    }

    KDNodePtr KDTree::nearest_(  
        const KDNodePtr &branch, 
        const point_t &pt,       
        const size_t &level,     
        const KDNodePtr &best,   
        const double &best_dist  
        ) {
      double d, dx, dx2;

      if (!bool(*branch)) {
        return NewKDNodePtr();  // basically, null
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
    KDNodePtr KDTree::nearest_(const point_t &pt) {
      size_t level = 0;

      // Modifications for a periodic box
      if(periodic){
        KDNodePtr nearest_ptr;
        double nearest_dist = std::numeric_limits<double>::max();

        // Compute the jump for to the periodic images we will visit
        point_t jump = pt;
        for(int idim = 0; idim < ndim; idim++){
          jump[idim] = pt[idim] > boxsize/2 ? -boxsize : +boxsize;
        }

        // Loop over all periodic images
        for(int i = 0; i < twotondim; i++){
          point_t search_pt = pt;
          for(int idim = 0, n = 1; idim < ndim; idim++, n *= 2){
            search_pt[idim] += (i/n % 2) * jump[idim];
          }

          double branch_dist = dist2(point_t(*root), pt);
          auto current_nearest_ptr = nearest_(root,  // beginning of tree
              search_pt,                             // point we are querying
              level,                                 // start from level 0
              root,                                  // best is the root
              branch_dist);                          // best_dist = branch_dist

          // Check if the minimum distance is the smallest we found so far
          point_t current_nearest_pt = point_t(*current_nearest_ptr);
          double current_nearest_dist = periodic_dist(current_nearest_pt, pt, boxsize);
          if(current_nearest_dist < nearest_dist){
            nearest_dist = current_nearest_dist;
            nearest_ptr = current_nearest_ptr;
          }
        }
        return nearest_ptr;
      }

      // KDNodePtr best = branch;
      double branch_dist = dist2(point_t(*root), pt);
      return nearest_(root,  // beginning of tree
          pt,                // point we are querying
          level,             // start from level 0
          root,              // best is the root
          branch_dist);      // best_dist = branch_dist
    };

    point_t KDTree::nearest_point(const point_t &pt) {
      return point_t(*nearest_(pt));
    };
    size_t KDTree::nearest_index(const point_t &pt) {
      return size_t(*nearest_(pt));
    };

    pointIndex KDTree::nearest_pointIndex(const point_t &pt) {
      KDNodePtr Nearest = nearest_(pt);
      return pointIndex(point_t(*Nearest), size_t(*Nearest));
    }

    pointIndexArr KDTree::neighborhood_( 
        const KDNodePtr &branch,         
        const point_t &pt,               
        const double &rad,               
        const size_t &level              
        ) {
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

    pointIndexArr KDTree::neighborhood( 
        const point_t &pt,              
        const double &rad) {
      size_t level = 0;
      return neighborhood_(root, pt, rad, level);
    }

    pointVec KDTree::neighborhood_points( 
        const point_t &pt,                
        const double &rad) {
      size_t level = 0;

      // Modification for a periodic box
      if(periodic){

        // To avoid duplicated points we can only search over half the box at max
        double radius = std::min(rad, boxsize/2);

        // Compute the jump for to the periodic images we will visit
        point_t jump = pt;
        for(int idim = 0; idim < ndim; idim++){
          jump[idim] = pt[idim] > boxsize/2 ? -boxsize : +boxsize;

          // If the point is too far away from the boundary we don't
          // have to search periodic images
          double dist_from_boundary = pt[idim] < boxsize/2 ? pt[idim] : boxsize - pt[idim];
          if(dist_from_boundary > radius)
            jump[idim] = 0.0;
        }

        // Loop over all 2^NDIM possible search points
        std::vector<pointIndexArr> nbh_arr(twotondim);
        size_t total = 0;
        for(int i = 0; i < twotondim; i++){
          point_t search_pt = pt;
          bool skip = false;
          for(int idim = 0, n = 1; idim < ndim; idim++, n *= 2){
            // If the point is too far away from the boundary we don't
            // have to search periodic images
            int add = (i/n % 2);
            if(add == 1 and jump[idim] == 0.0){
              skip = true;
              break;
            }

            search_pt[idim] += add * jump[idim];
          }
          if(skip)
            continue;

          nbh_arr[i] = neighborhood_(root, search_pt, radius, level);
          total += nbh_arr[i].size();
        }

        pointVec nbhp;
        nbhp.resize(total);
        total = 0;
        for(int i = 0; i < twotondim; i++){
          std::transform(nbh_arr[i].begin(), nbh_arr[i].end(), nbhp.begin() + total,
              [](pointIndex x) { return x.first; });
          total += nbh_arr[i].size();
        }
        return nbhp;
      }

      pointIndexArr nbh = neighborhood_(root, pt, rad, level);
      pointVec nbhp;
      nbhp.resize(nbh.size());
      std::transform(nbh.begin(), nbh.end(), nbhp.begin(),
          [](pointIndex x) { return x.first; });
      return nbhp;
    }

    indexArr KDTree::neighborhood_indices( 
        const point_t &pt,                 
        const double &rad) {
      size_t level = 0;

      if(periodic){
        // To avoid duplicated points we can only search over half the box at max
        double radius = std::min(rad, boxsize/2);

        // Compute the jump for to the periodic images we will visit
        point_t jump = pt;
        for(int idim = 0; idim < ndim; idim++){
          jump[idim] = pt[idim] > boxsize/2 ? -boxsize : +boxsize;

          // If the point is too far away from the boundary we don't
          // have to search periodic images
          double dist_from_boundary = pt[idim] < boxsize/2 ? pt[idim] : boxsize - pt[idim];
          if(dist_from_boundary > radius)
            jump[idim] = 0.0;
        }

        // Loop over all 2^NDIM possible search points
        std::vector<pointIndexArr> nbh_arr(twotondim);
        size_t total = 0;
        for(int i = 0; i < twotondim; i++){
          point_t search_pt = pt;
          bool skip = false;
          for(int idim = 0, n = 1; idim < ndim; idim++, n *= 2){
            // If the point is too far away from the boundary we don't
            // have to search periodic images
            int add = (i/n % 2);
            if(add == 1 and jump[idim] == 0.0){
              skip = true;
              break;
            }

            search_pt[idim] += add * jump[idim];
          }
          if(skip)
            continue;

          nbh_arr[i] = neighborhood_(root, search_pt, radius, level);
          total += nbh_arr[i].size();
        }

        indexArr nbhi(total);
        total = 0;
        for(int i = 0; i < twotondim; i++){
          std::transform(nbh_arr[i].begin(), nbh_arr[i].end(), nbhi.begin() + total,
              [](pointIndex x) { return x.second; });
          total += nbh_arr[i].size();
        }
        return nbhi;
      } 

      pointIndexArr nbh = neighborhood_(root, pt, rad, level);
      indexArr nbhi;
      nbhi.resize(nbh.size());
      std::transform(nbh.begin(), nbh.end(), nbhi.begin(),
          [](pointIndex x) { return x.second; });
      return nbhi;
    }
  }
}
#endif
