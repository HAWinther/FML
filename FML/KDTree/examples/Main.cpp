#include <iostream>
#include <vector>
#include <FML/KDTree/KDTree.h>

using point_t = FML::KDTREE::point_t;
using pointVec = FML::KDTREE::pointVec;
using KDTree = FML::KDTREE::KDTree;

// Periodic ND box
void test_periodic(int ndim){
  pointVec points;
  point_t pt(ndim);

  std::cout << "\n#=====================================================\n";
  std::cout << "# Running test of periodic kdTree for ndim = " << ndim << "\n";
  std::cout << "#=====================================================\n";

  double boxsize = 1.0;
  const int n = 1000;
  for(int i = 0; i < n-1; i++){
    pt = point_t(ndim);
    for(int idim = 0; idim < ndim; idim++){
      pt[idim] = boxsize * (rand() % RAND_MAX)/double(RAND_MAX);
    }
    points.push_back(pt);
  }
  // Add (0,0,0,...)
  points.push_back(point_t(ndim,0.0));

  // Make tree
  KDTree tree(points, 1.0);
  
  // Find closest point to our points (which should be themselves)
  for(size_t i = 0; i < points.size(); i++){
    pt = points[i];
    auto pt_closest = tree.nearest_point(pt);
    for(int idim = 0; idim < ndim; idim++){
      assert(std::fabs(pt[idim] - pt_closest[idim]) < 1e-10);
    }
  }

  // Find closest point to (boxsize,boxsize,...)
  pt = point_t(ndim,boxsize);
  auto pt_closest = tree.nearest_point(pt);
  for(int idim = 0; idim < ndim; idim++){
    assert(pt_closest[idim] == 0.0);
  }

  // Get neihborhood points and check that they all are within the distance
  pt = point_t(ndim,0.0);
  double radius = 0.1;
  auto res2 = tree.neighborhood_points(pt, radius);
  for (point_t pt_close : res2) {
    double dist = 0.0;
    for(int idim = 0; idim < ndim; idim++){
      double dx = pt_close[idim] - pt[idim];
      if(dx >= boxsize/2) dx -= boxsize;
      if(dx < -boxsize/2) dx += boxsize;
      dist += dx*dx;
    }
    dist = std::sqrt(dist);
    assert(dist <= radius);
  }

  // Count how many points we should find
  size_t count = 0;
  for(size_t i = 0; i < points.size(); i++){
    double dist = 0.0;
    for(int idim = 0; idim < ndim; idim++){
      double dx = points[i][idim] - pt[idim];
      if(dx >=  boxsize/2) dx -= boxsize;
      if(dx < -boxsize/2) dx += boxsize;
      dist += dx*dx;
    }
    dist = std::sqrt(dist);
    if(dist <= radius)
      count++;
  }
  assert(count == res2.size());

  std::cout << "All test passed for ndim = " << ndim << "\n";
}

template<int N>
class Particle{
  public:
  double pos[N];
  double *get_pos(){ return pos; }
  constexpr int get_ndim() const { return N; }
};

template <int N>
void periodic_kdtree_from_particles(){
  const int nparts = 1000;
  const double boxsize = 1.0;
  const double radius = 0.01*N*N*N;
  
  std::cout << "\n#=====================================================\n";
  std::cout << "# Periodic kdTree (from particles) ndim = " << N << "\n";
  std::cout << "#=====================================================\n";

  std::vector<Particle<N>> parts(nparts);
  for(auto & p : parts){
    auto pos = FML::PARTICLE::GetPos(p);
    for(int i = 0; i < N; i++){
      pos[i] = (rand() % RAND_MAX)/double(RAND_MAX);
    }
  }
  KDTree tree(parts.data(), parts.size(), boxsize);
  
  // Find nearst point to origin
  point_t pt(N,0.0);
  auto res1 = tree.nearest_point(pt);
  std::cout << "Closest point to origin: (";
  for(auto x : res1)
    std::cout << std::setw(15) << x << " ";
  std::cout << ")\n";

  // Find points within radius of origin
  double expected = 2*radius * parts.size();
  if(N == 2)
    expected = M_PI*radius*radius*parts.size();
  if(N == 3)
    expected = 4.0*M_PI/3.0*radius*radius*radius*parts.size();
  auto res2 = tree.neighborhood_points(pt, radius);
  std::cout << "All points within radius = " << radius << " of origin:\n";
  std::cout << "Expecting ~" << expected << " found " << res2.size() << "\n";
  for(auto & point : res2){
    for(auto x : point)
      std::cout << std::setw(15) << x << " ";
    std::cout << "\n";
  }
  
  // Find index of points within radius of origin
  auto res3 = tree.neighborhood_indices(pt, radius);
}

void kdtree_from_points(int ndim){
  const int nparts = 1000;
  const double radius = 0.01*ndim*ndim*ndim;

  std::cout << "\n#=====================================================\n";
  std::cout << "# kdTree (from pointset) ndim = " << ndim << "\n";
  std::cout << "#=====================================================\n";

  pointVec points(nparts, point_t(ndim,0.0));
  for(auto & pt : points){
    for(int i = 0; i < ndim; i++){
      pt[i] = (rand() % RAND_MAX)/double(RAND_MAX);
    }
  }
  KDTree tree(points);

  // Find nearst point to origin
  point_t pt(ndim,0.0);
  auto res1 = tree.nearest_point(pt);
  std::cout << "Closest point to origin: (";
  for(auto x : res1)
    std::cout << std::setw(15) << x << " ";
  std::cout << ")\n";

  // Find points within radius of origin
  double expected = radius * points.size();
  if(ndim == 2)
    expected = M_PI*(radius/2)*(radius/2)*points.size();
  if(ndim == 3)
    expected = 4.0*M_PI/3.0*(radius/2)*(radius/2)*(radius/2)*points.size();
  auto res2 = tree.neighborhood_points(pt, radius);
  auto res3 = tree.neighborhood_indices(pt, radius);
  std::cout << "All points within radius = " << radius << " of origin:\n";
  std::cout << "Expecting ~" << expected << " found " << res2.size() << "\n";
  for(size_t i = 0; i < res2.size(); i++){
    std::cout << "Index: " << std::setw(5) << res3[i] << " Pos: ";
    for(auto x : res2[i]){
      std::cout << std::setw(15) << x << " ";
    }
    std::cout << "\n";
  }
  
  // Find index of points within radius of origin
}

int main() {

  // Test the periodic kdTree
  test_periodic(1);
  test_periodic(2);
  test_periodic(3);

  // Example periodic kdtree from particles
  periodic_kdtree_from_particles<1>();
  periodic_kdtree_from_particles<2>();
  periodic_kdtree_from_particles<3>();

  // Example normal kdtree from points
  kdtree_from_points(1);
  kdtree_from_points(2);
  kdtree_from_points(3);

  return 0;
}
