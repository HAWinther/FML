#include <FML/Global/Global.h>
#include <FML/PairCounting/PairCount.h>
#include <FML/ParticlesInBoxes/ParticlesInBoxes.h>
#include <FML/Survey/GalaxiesToBox.h>
#include <FML/FileUtils/FileUtils.h>

#include <vector>
#include <fstream>
#include <cstring>

// We need to add: 
// reading in Euclid data
// putting it in a box and recording observer position
// add mu binning also

//=====================================================
// Data structure to keep particles in a box
// For our use we can add RA, DEC, z here also
// if we need that to define the jacknife regions
//=====================================================
template<int N>
struct ParticleType {
  double Pos[N];
  double *get_pos() { 
    return Pos; 
  }
  int get_ndim() const { 
    return N;
  }
  double get_weight() const { 
    return 1.0; 
  }
};

int main() {

  //=====================================================
  // Main options
  //=====================================================
  const int NDIM       = 3;
  const double boxsize = 1024.0;
  const bool periodic  = false;
  const bool verbose   = true;
  const int nbins      = 20;
  const int njacknife  = 3;
  const double rmax    = 50.0 / boxsize;
  const double rmax2   = rmax*rmax;
  using Particle  = ParticleType<NDIM>;
  using DVector   = std::vector<double>;
  using DVector2D = std::vector<DVector>;
  using DVector3D = std::vector<DVector2D>;

  //=====================================================
  // Read the galaxy data
  //=====================================================
  std::string filename_galaxies = "../../../../TestData/particles_B1024.txt";
  const int nskip_header_lines  = 0;
  const int ncols_file          = 3;
  std::vector<int> cols_to_keep{0,1,2};
  auto galaxy_data = FML::FILEUTILS::read_regular_ascii(
      filename_galaxies,
      ncols_file,
      cols_to_keep,
      nskip_header_lines);

  //=====================================================
  // Process this to galaxies
  //=====================================================
  std::vector<Particle> galaxies;
  galaxies.reserve(galaxy_data.size());
  for(auto &g : galaxy_data){
    // For testing: remove the first 1/3
    //if(g[0] < 1/3. * boxsize) continue;
    Particle p;
    for(int idim = 0; idim < NDIM; idim++){
      p.Pos[idim] = g[idim] / boxsize;
      if(p.Pos[idim] >= 1.0) p.Pos[idim] -= 1.0;
      if(p.Pos[idim] <  0.0) p.Pos[idim] += 1.0;
    }
    galaxies.push_back(p);
  }

  //=====================================================
  // Read the void data
  //=====================================================
  std::string filename_voids    = "../../../../TestData/voids_B1024.txt";
  const int nskip_header_lines_voids = 0;
  const int ncols_file_voids         = 3;
  std::vector<int> cols_to_keep_voids{0,1,2};
  auto void_data = FML::FILEUTILS::read_regular_ascii(
      filename_voids,
      ncols_file_voids,
      cols_to_keep_voids,
      nskip_header_lines_voids);

  //=====================================================
  // Process this to voids
  //=====================================================
  std::vector<Particle> voids;
  voids.reserve(void_data.size());
  for(auto &g : void_data){
    // For testing: remove the first 1/3
    //if(g[0] < 1/3. * boxsize) continue;
    Particle p;
    for(int idim = 0; idim < NDIM; idim++){
      p.Pos[idim] = g[idim] / boxsize;
      if(p.Pos[idim] >= 1.0) p.Pos[idim] -= 1.0;
      if(p.Pos[idim] <  0.0) p.Pos[idim] += 1.0;
    }
    voids.push_back(p);
  }

  // Container to store how many pairs in each bin (one for each thread we use)
  // The pair is in the same jacknife region
  DVector3D count_threads_same  ( FML::NThreads, DVector2D(njacknife, DVector(nbins, 0.0) ) );
  // The pair spans two regions, this holds pairs with galaxy in the current region
  DVector3D count_threads_galaxy( FML::NThreads, DVector2D(njacknife, DVector(nbins, 0.0) ) );
  // The pair spans two regions, this holds pairs with void in the current region
  DVector3D count_threads_void  ( FML::NThreads, DVector2D(njacknife, DVector(nbins, 0.0) ) );
  
  //=====================================================
  // Define a mapping from each particle to its jacknife
  // region. Here we simply use a split along the x-axis
  // so ijacknife = int(x * njacknife)
  //=====================================================
  auto get_jacknife_index = [=](Particle &p){
    const int ijacknife = int(p.Pos[0] * njacknife);
    assert(ijacknife >= 0 and ijacknife < njacknife);
    return ijacknife;
  };

  //=====================================================
  // Define the binning function
  //=====================================================
  std::function<void(int,double*, Particle&, Particle&)> binning = 
    [&](int thread_id, double* dist, Particle &p1, Particle &p2){

      const double weight1 = p1.get_weight();
      const double weight2 = p2.get_weight();

      // Compute squared distance between pairs
      double dist2 = dist[0]*dist[0];
      if constexpr(NDIM > 1) dist2 += dist[1]*dist[1];
      if constexpr(NDIM > 2) dist2 += dist[2]*dist[2];
      if(dist2 >= rmax2) return;

      // Compute the jacknife index of each particle
      // In this example we split by x
      const int ijacknife1 = get_jacknife_index(p1);
      const int ijacknife2 = get_jacknife_index(p2);

      // Compute bin index and add to bin
      if(ijacknife1 == ijacknife2){
        // Pair is in the same region
        const int ibin = int(sqrt(dist2 / rmax2) * nbins);
        count_threads_same[thread_id][ijacknife1][ibin] += weight1*weight2;
      } else {
        // Pair spans two regions
        const int ibin = int(sqrt(dist2 / rmax2) * nbins);
        count_threads_galaxy[thread_id][ijacknife1][ibin] += weight1*weight2;
        count_threads_void[thread_id][ijacknife2][ibin]   += weight1*weight2;
      }
    };

  //=========================================================================
  // Optimize the computation: select a good ngrid size for binning the particles and voids to a grid
  // 8 cells to get to rmax + 2 particles per cells on average + minimum 10 cells per dim
  //=========================================================================
  int ngrid1 = std::min( int(8.0/rmax), int(std::pow(galaxies.size() / 2.0,1./double(NDIM))) );
  if(ngrid1 < 10) ngrid1 = 10;                       
  int ngrid2 = std::min( int(8.0/rmax), int(std::pow(voids.size() / 2.0,1./double(NDIM))) );
  if(ngrid2 < 10) ngrid2 = 10;                       

  //=========================================================================
  // Assign particles to a grid
  //=========================================================================
  FML::PARTICLE::ParticlesInBoxes<Particle> grid1;
  FML::PARTICLE::ParticlesInBoxes<Particle> grid2;
  grid1.create(galaxies.data(), galaxies.size(), ngrid1);
  grid2.create(voids.data(), voids.size(), ngrid2);

  //=========================================================================
  // Do the pair counts
  //=========================================================================
  FML::CORRELATIONFUNCTIONS::CrossPairCountGridMethod<Particle, Particle>(grid1, grid2, binning, rmax, periodic, verbose);

  //=========================================================================
  // First reduce over threads
  //=========================================================================
  DVector2D pairs_same(njacknife, DVector(nbins, 0.0) );
  DVector2D pairs_different_galaxy(njacknife, DVector(nbins, 0.0) );
  DVector2D pairs_different_void  (njacknife, DVector(nbins, 0.0) );  
  DVector sum_weights(njacknife,0.0);
  for(int i = 0; i < FML::NThreads; i++){
    for(int j = 0; j < njacknife; j++){
      for(int k = 0; k < nbins; k++){
        pairs_same[j][k] += count_threads_same[i][j][k];
        pairs_different_galaxy[j][k] += count_threads_galaxy[i][j][k];
        pairs_different_void[j][k]   += count_threads_void[i][j][k];
      }
    }
  }

#ifdef USE_MPI
  //=========================================================================
  // Next reduce over tasks
  //=========================================================================
  for(int j = 0; j < njacknife; j++){
    MPI_Allreduce(MPI_IN_PLACE, pairs_same[j].data(), nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pairs_different_galaxy[j].data(), nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pairs_different_void[j].data(), nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
        
  //=========================================================================
  // Compute the sum of the weights and sum of weights^2 for galaxies
  //=========================================================================
  std::vector<double> sum_weights_galaxy(njacknife, 0.0);
  std::vector<double> sum_weights2_galaxy(njacknife, 0.0);
  auto &cells1 = grid1.get_cells();
  for(auto &cell : cells1){
    for(auto &p: cell.get_part()){
      const int ijacknife = get_jacknife_index(p);
      const double w = p.get_weight();
      sum_weights_galaxy[ijacknife] += w;
      sum_weights2_galaxy[ijacknife] += w*w;
    }
  }
  // Compute the sum of the weights and sum of weights^2 for voids
  std::vector<double> sum_weights_void(njacknife, 0.0);
  std::vector<double> sum_weights2_void(njacknife, 0.0);
  auto &cells2 = grid2.get_cells();
  for(auto &cell : cells2){
    for(auto &p: cell.get_part()){
      const int ijacknife = get_jacknife_index(p);
      const double w = p.get_weight();
      sum_weights_void[ijacknife] += w;
      sum_weights2_void[ijacknife] += w*w;
    }
  }
  // Compute the sum of the weights over all jacknife regions
  double sum_weights_galaxy_all  = 0.0;
  double sum_weights2_galaxy_all = 0.0;
  double sum_weights_void_all    = 0.0;
  double sum_weights2_void_all   = 0.0;
  for(int j = 0; j < njacknife; j++){
    sum_weights_galaxy_all  += sum_weights_galaxy[j];
    sum_weights2_galaxy_all += sum_weights2_galaxy[j];
    sum_weights_void_all    += sum_weights_void[j];
    sum_weights2_void_all   += sum_weights2_void[j];
  }

  //=========================================================================
  // Function that computes the number of pairs when excluding region ijacknife
  // For total corr func Sum up all vg+gv pairs (divided by 2 as we overcount) to other regions + all pairs in the same box
  // For excluding we subtract the pairs that go to other regions
  //=========================================================================
  auto numpair_excluding_region = [&](int ijacknife){
    DVector numpairs(nbins,0.0);
    for(int j = 0; j < njacknife; j++){
      for(int k = 0; k < nbins; k++){
        // Add up pairs in the same region 
        numpairs[k] += pairs_same[j][k];
        // Add up pairs with void in different region
        numpairs[k] += 0.5 * pairs_different_galaxy[j][k];
        numpairs[k] += 0.5 * pairs_different_void[j][k];
      }
    }

    // Subtract pairs that have anything to do with current region
    if(ijacknife >= 0){
      for(int k = 0; k < nbins; k++){
        numpairs[k] -= pairs_same[ijacknife][k];
        numpairs[k] -= pairs_different_galaxy[ijacknife][k];
        numpairs[k] -= pairs_different_void[ijacknife][k];
      }
    }

    return numpairs;
  };

  // Compute the total number of pairs (exclude -1 means no exclusion)
  DVector numpairstotal = numpair_excluding_region(-1);

  // Compute the number of pairs when excluding the first jacknife region 
  const int iexclude = 0;
  DVector numpairs = numpair_excluding_region(iexclude);

  // Output the pair counts
  if(FML::ThisTask == 0){
    std::cout << "Sum of all weights: " 
      << sum_weights_galaxy_all << " " 
      << sum_weights2_galaxy_all << " "
      << sum_weights_void_all << " " 
      << sum_weights2_void_all << "\n";
    std::cout << "Sum of all weights with exclusion: " 
      << sum_weights_galaxy_all - sum_weights_galaxy[iexclude] << " " 
      << sum_weights2_galaxy_all - sum_weights2_galaxy[iexclude] << " "
      << sum_weights_void_all - sum_weights_void[iexclude] << " " 
      << sum_weights2_void_all - sum_weights2_void[iexclude] << "\n";
    for(int k = 0; k < nbins; k++){
      double r = rmax * k / double(nbins) * boxsize;
      std::cout << r << " " << numpairstotal[k] << " " << numpairs[k] << "\n";
    }
  }
}

