#include <FML/RamsesUtils/RamsesUtils.h>

//====================================================
// Simple particle type to store the data in
//====================================================
template<int NDIM>
class Particle{
  public:
    double x[NDIM];
    double v[NDIM];
    Particle(){}
    Particle(double *_x) {
      std::memcpy(x, _x, NDIM * sizeof(double));
    }
    Particle(double *_x, double *_v) {
      std::memcpy(x, _x, NDIM * sizeof(double));
      std::memcpy(v, _v, NDIM * sizeof(double));
    }
    double *get_pos(){ return x; }
    double *get_vel(){ return v; }
    int get_ndim() { return NDIM; }
    ~Particle(){}
};

//====================================================
// Examples on how to use the RamsesReader class
//====================================================

using RamsesReader = FML::FILEUTILS::RAMSES::RamsesReader;

int main(){
  const int NDIM = 3;
  using ParticleType = Particle<NDIM>;

  // File info
  const int output_number = 8;
  const bool verbose = true;
  std::string outfolder = "../../../TestData/";
  std::vector<ParticleType> p;
 
  // Set up reader
  RamsesReader read(outfolder, output_number, verbose);
  
  // Read all particles (NB: with MPI all tasks will read all the data) 
  // and store it in p
  read.read_ramses(p);

  // Read a single file and store it in p2
  std::vector<ParticleType> p2;
  read.read_ramses_single(1, p2);
}

