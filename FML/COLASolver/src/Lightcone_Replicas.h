#ifndef LIGHTCONE_REPLICA_HEADER
#define LIGHTCONE_REPLICA_HEADER

#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <iostream>
#include <array>
#include <vector>

template<int NDIM>
class BoxReplicas {
  private:
    using Point = std::array<double, NDIM>;
    
    enum FlagValues { COMPLETELY_OUTSIDE_LC = -1, INTERSECTING_LC = 0, COMPLETELY_INSIDE_LC = 1};
    
    Point pos_observer;
    bool use_replicas;
    int ndim_rep;
    double boundary_over_boxsize;

    // Flag to mark if a replica is in use (has particles which cross the lightcone)
    std::vector<char> in_use_flag;
    size_t n_replicas_total;
    size_t n_replicas_in_use;

    // Number of replicas in use currently
    // The left/right direction means replicas in the negative/positive direction
    // from the main box
    std::array<int, NDIM> n_per_dim_left;
    std::array<int, NDIM> n_per_dim_right;

    // The size of the replica set when we start
    // This will change during the evolution as more and more
    // drift outside the lightcone
    std::array<int, NDIM> n_per_dim_left_alloc;
    std::array<int, NDIM> n_per_dim_right_alloc;

    bool replicas_initialized{false};

  public:

    size_t get_n_replicas() { return n_replicas_total; }

    void read_parameters(ParameterMap & param) {
      use_replicas = param.get<bool>("plc_use_replicas");
      if (FML::ThisTask == 0) {
        std::cout << "plc_use_replicas                         : " << use_replicas << "\n";
      }
      if(use_replicas) {
        double boxsize = param.get<double>("simulation_boxsize");
        double boundary_in_mpch = param.get<double>("plc_boundary_mpch");
        boundary_over_boxsize = boundary_in_mpch / boxsize;
        auto origin = param.get<std::vector<double>>("plc_pos_observer");
        FML::assert_mpi(origin.size() >= NDIM, "Position of observer much have NDIM components");
        for(int idim = 0; idim < NDIM; idim++)
          pos_observer[idim] = origin[idim];

        ndim_rep = param.get<int>("plc_ndim_rep");
        ndim_rep = std::min(ndim_rep, NDIM);

        if (FML::ThisTask == 0) {
          std::cout << "plc_pos_observer                         : ";
          for(auto & p : pos_observer)
            std::cout << p << " , ";
          std::cout << "\n";
          std::cout << "plc_ndim_rep                             : " << ndim_rep << "\n";
          std::cout << "plc_boundary_mpch                        : " << boundary_in_mpch << " Mpc/h\n";
        }
      } else {
        ndim_rep = 0;
        pos_observer = {};
        boundary_over_boxsize = 0.0;
      }
    }

    void init(double a_init, double R_init) {
      if(replicas_initialized) return;
      
      // Initialize the number of replicas to cover the whole lightcone
      // (will be adjusted below)
      int n_rep = use_replicas ? std::ceil(R_init + 1.0) : 0;
      if(not use_replicas and R_init > 1.0) {
        if(FML::ThisTask == 0) {
          throw std::runtime_error("We do not use replicas, but R_init / boxsize > 1 so lightcone quadrant will not be complete. Reduce z_init, increase boxsize or add using replicas");
        }
      }

      // Initialize how many replicas we have along each direction
      for(int idim = 0; idim < NDIM; idim++)
        n_per_dim_right[idim] = n_rep;
      n_per_dim_left = {};
      for(int idim = 0; idim < ndim_rep; idim++)
        n_per_dim_left[idim] = n_per_dim_right[idim];

      // Reduce the number of replications if possible
      n_replicas_total = 1;
      for(int idim = 0; idim < NDIM; idim++) {
        double max_dist_left_from_box_origin = ceil( std::fabs(R_init - pos_observer[idim]) );
        if (max_dist_left_from_box_origin < n_per_dim_left[idim]) 
          n_per_dim_left[idim] = int(max_dist_left_from_box_origin);
        double max_dist_right_from_box_origin = ceil( std::fabs(R_init + pos_observer[idim]) );
        if (max_dist_right_from_box_origin - 1 < n_per_dim_right[idim])
          n_per_dim_right[idim] = int(max_dist_right_from_box_origin)-1;
   
        n_replicas_total *= (n_per_dim_left[idim] + n_per_dim_right[idim] + 1);
      }
      if(FML::ThisTask == 0) {
        std::cout << "# Init replicas at a = " << a_init << " Rmax/boxsize = " << R_init << "\n";
        for(int idim = 0; idim < NDIM; idim++) {
          std::cout << "# For idim = " << idim << " we have replicas from -" << n_per_dim_left[idim] << " -> +" << n_per_dim_right[idim] << " boxes\n";
          std::cout << "# We have a total of n = " << n_replicas_total << " replicas\n";
        }
      }

      // The number of replicas might change, so store the maximum number we have 
      // This is used to look up the index of the replica in the array below
      n_per_dim_left_alloc = n_per_dim_left;
      n_per_dim_right_alloc = n_per_dim_right;

      // Create lookuptable
      // 0 mean in use, 1 means inside for current step and 2 means permanently outside from now on
      in_use_flag = std::vector<char>(n_replicas_total, INTERSECTING_LC);
      replicas_initialized = true;
    }

    size_t get_coord_of_replica(const std::array<int, NDIM> & index) {
      size_t coord = 0;
      for(int idim = 0; idim < NDIM; idim++)
        coord = coord * (n_per_dim_left_alloc[idim] + n_per_dim_right_alloc[idim] + 1) + (index[idim] + n_per_dim_left_alloc[idim]);
      return coord;
    }

    std::array<int, NDIM> get_replica_position(size_t coord) {
      std::array<int, NDIM> index;
      unsigned int n = 1;
      for(int idim = NDIM-1; idim >= 0; idim--){
        int n_per_dim = (n_per_dim_left[idim] + n_per_dim_right[idim] + 1);
        index[idim] = (coord / n) % n_per_dim;
        index[idim] -= n_per_dim_left[idim];
        n *= n_per_dim;
      }
      return index;
    }

    void flag(double r_old, double r_new) {
      double r_old_squared = r_old * r_old;
      double r_new_squared = r_new * r_new;

      // Update the number of replicas we need to even consider
      for(int idim = 0; idim < NDIM; idim++) {
        double max_dist_left_from_box_origin = ceil( std::fabs(r_old - pos_observer[idim]) );
        if (max_dist_left_from_box_origin < n_per_dim_left[idim]) 
          n_per_dim_left[idim] = int(max_dist_left_from_box_origin);
        double max_dist_right_from_box_origin = ceil( std::fabs(r_old + pos_observer[idim]) );
        if (max_dist_right_from_box_origin - 1 < n_per_dim_right[idim])
          n_per_dim_right[idim] = int(max_dist_right_from_box_origin) - 1;
      }
      if(FML::ThisTask == 0) {
        for(int idim = 0; idim < NDIM; idim++) {
          std::cout << "# For idim = " << idim << " we have replicas from -" << n_per_dim_left[idim] << " -> +" << n_per_dim_right[idim] << " boxes\n";
        }
      }

      // Compute the total number of replicas we have
      n_replicas_total = 1;
      std::array<int, NDIM> n_per_dim;
      for(int idim = 0; idim < NDIM; idim++) {
        n_per_dim[idim] = (n_per_dim_left[idim] + n_per_dim_right[idim] + 1);
        n_replicas_total *= n_per_dim[idim];
      }

      // Loop over all the replicas
      n_replicas_in_use = 0;
      for(size_t i = 0; i < n_replicas_total; i++) {

        // Extract position of replica relative to main box
        auto index = get_replica_position(i);

        // Compute coordinate of current replica
        const size_t coord = get_coord_of_replica(index);

        // If it has been set to be outside before then do not process it
        if(in_use_flag[coord] == COMPLETELY_OUTSIDE_LC) continue;

        // Compute minimum and maximum distance from observer to the box
        Point corner_point;
        for(int idim = 0; idim < NDIM; idim++) corner_point[idim] = index[idim];
        auto distances_squared = minmax_distance_from_observer_to_box(corner_point, pos_observer);
        double r_min_squared = distances_squared.first;
        double r_max_squared = distances_squared.second;

        // Flag it
        if(r_min_squared - std::pow(boundary_over_boxsize,2) > r_old_squared) {
          in_use_flag[coord] = COMPLETELY_OUTSIDE_LC;
        } else if(r_max_squared < r_new_squared + std::pow(boundary_over_boxsize,2)) {
          in_use_flag[coord] = COMPLETELY_INSIDE_LC;
        } else {
          in_use_flag[coord] = INTERSECTING_LC;
          n_replicas_in_use++;
        }
      }
      if(FML::ThisTask == 0) 
        std::cout << "# Number of replicas flagged as being in use on main task: " << n_replicas_in_use << "\n";
    }

    // Given a box [0,box_x] x [0,box_y] x ... defined by the point X compute the minimum and maximum
    // distance from an observer to all points in the box.
    // The point X is the corner with the smallest coordinates (see below).
    // The boxsize per dimension is given by the array boxsize_per_dim
    // I.e.
    //
    // In 1D x______
    // Corners: X = x and x+box_x
    //
    // In 2D ________
    //       |      |
    //       |      |
    //       X______|
    // Corners: X = (x,y), (x+box_x,y) (x,y+box_y), (x+box_x,y+box_y)
    //
    // In 3D  ________
    //       /       /|
    //      /_______/ |
    //      |       | |
    //      |       | /
    //      X_______|/
    //
    std::pair<double,double> minmax_distance_from_observer_to_box(Point & corner_point, Point & observer){
      constexpr auto twotondim = FML::power(2, NDIM);

      // Generate positions for all corners of the box
      // We do this by generating {0,1}^(NDIM-1) and adding that to the
      // left and right faces
      std::array<Point, twotondim> corners;
      for(int i = 0; i < twotondim; i++) {
        Point & pos = corners[i];
        // Generate the positions of all the corners of the cube
        // NB: the x-positions are fixed by the two faces we send in
        // For the other directions we assume the box has length unity
        for(int idim = 0, n = 1; idim < NDIM; idim++, n *= 2) {
          int add = (i/n % 2);
          pos[idim] = corner_point[idim] + add;
        }
      }

      // Check if the observers i-coordinate is within the box bounded by left and right faces
      // [      ][       ][       ] Here the boxes in the right column are inside and the rest outside 
      // [      ][       ][   x   ]
      std::array<bool, NDIM> inside_per_dim;
      for(int idim = 0; idim < NDIM; idim++) {
        inside_per_dim[idim] = (corner_point[idim] <= observer[idim] and (corner_point[idim] + 1.0) >= observer[idim]);
      }

      // Compute minimum distance to the box
      double dist_min_squared = 0.0;
      double dist_max_squared = 0.0;
      for(int idim = 0; idim < NDIM; idim++) {
        double dx_min = std::numeric_limits<double>::max();
        double dx_max = 0.0;
        // Loop over all the 2^NDIM corners and compute the minimum i-distance to them
        for(auto & corner : corners) {
          double dx = std::abs(corner[idim] - observer[idim]);
          dx_min = std::min(dx_min, dx);
          dx_max = std::max(dx_max, dx);
        }
        if(inside_per_dim[idim])
          dx_min = 0.0;
        dist_min_squared += dx_min * dx_min;
        dist_max_squared += dx_max * dx_max;
      }

      return {dist_min_squared, dist_max_squared};
    }

    bool is_flagged_in_use(size_t coord) {
      return in_use_flag[coord] == INTERSECTING_LC;
    }

    int get_ndim_rep() { return ndim_rep; }

};

#endif
