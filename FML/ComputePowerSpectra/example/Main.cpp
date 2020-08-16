#include <fstream>
#include <cstring>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/FileUtils/FileUtils.h>

// Examples of using the library
void ExamplesPower();

// Some type aliases
template<int N>
  using FFTWGrid = FML::GRID::FFTWGrid<N>;

//=======================================================
// A simple particle class compatible with MPIParticles
//=======================================================
const int NDIM = 3;
struct Particle{
  double x[NDIM];
  Particle(){}
  Particle(double *_x){
    std::memcpy(x, _x, NDIM*sizeof(double));
  }
  double* get_pos(){ return x; }
  // We don't need this in the tests, but the method must exist
  double* get_vel(){ return nullptr; } 
  int get_particle_byte_size(){ return NDIM*sizeof(double);}
  void append_to_buffer(char *data){ std::memcpy(data, x, NDIM*sizeof(double)); }
  void assign_from_buffer(char *data){ std::memcpy(x, data, NDIM*sizeof(double)); }
};

int main(){
#ifdef MEMORY_LOGGING
  auto * mem = FML::MemoryLog::get();
  ExamplesPower();
  mem->print();
#else
  ExamplesPower();
#endif
}
    
void ExamplesPower(){

  const bool TEST_POFK = true;
  const bool TEST_POFK_INTERLACING = true;
  const bool TEST_POFK_BRUTEFORCE = false;
  const bool TEST_MULTIPOLES = false;
  const bool TEST_MULTIPOLES_GRID = false;

  const int Nmesh = 64;
  const int ell_max = 4;
  const std::string density_assignment_method = "CIC";

  //======================================================================================
  // Read particles from file
  //======================================================================================
  if(FML::ThisTask == 0) std::cout << "Reading particles from file\n";

  // Read ascii file with [x,y,z]
  const double box = 1024.0;
  const std::string filename = "../../../TestData/particles_B1024.txt";
  const int ncols = 3;
  const int nskip_header = 0;
  const std::vector<int> cols_to_keep{0,1,2};
  auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header);

  // Create particles and scale to [0,1)
  std::vector<Particle> part;
  for(auto& pos: data){
    for(auto &x : pos) x /= box;
    part.push_back(Particle(pos.data()));
  }
  
  // Create MPI particles by letting each task keep only the particles that falls in its domain
  FML::PARTICLE::MPIParticles<Particle> p;
  const bool all_tasks_have_the_same_particles = true;
  const int nalloc_per_task = part.size() / FML::NTasks * 2;
  p.create(part.data(), part.size(), nalloc_per_task, FML::xmin_domain, FML::xmax_domain, all_tasks_have_the_same_particles);

  //======================================================================================
  // Compute power-spectrum
  //======================================================================================

  if(TEST_POFK){
  
    if(FML::ThisTask == 0) std::cout << "Running compute_power_spectrum\n";

    // Naive power-spectrum evaluation
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Nmesh/2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<NDIM>(
        Nmesh, 
        p.get_particles().data(), 
        p.get_npart(), 
        p.get_npart_total(), 
        pofk,
        density_assignment_method);

    // To physical units and output
    pofk.scale(box);
    if(FML::ThisTask == 0){
      for(int i = 0; i < pofk.n; i++){
        std::cout << pofk.k[i] << " " << pofk.pofk[i] << "\n";
      }
      std::cout << "\n";
    }
  }
  
  //======================================================================================
  // Compute power-spectrum with interlacing
  //======================================================================================
  
  if(TEST_POFK_INTERLACING){
    
    if(FML::ThisTask == 0) std::cout << "Running compute_power_spectrum_interlacing\n";
    
    // Power-spectrum evaluation using interlacing
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Nmesh/2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_interlacing<NDIM>(
        Nmesh, 
        p.get_particles().data(), 
        p.get_npart(), 
        p.get_npart_total(), 
        pofk,
        density_assignment_method);

    // To physical units and output
    pofk.scale(box);
    if(FML::ThisTask == 0){
      for(int i = 0; i < pofk.n; i++){
        std::cout << pofk.k[i] << " " << pofk.pofk[i] << "\n";
      }
      std::cout << "\n";
    }
  }
  
  //======================================================================================
  // Compute power-spectrum direct summation
  //======================================================================================
  
  if(TEST_POFK_BRUTEFORCE){
    
    if(FML::ThisTask == 0) std::cout << "Running compute_power_spectrum_direct_summation\n";
    
    // Brute force (but alias free) direct summation power-spectrum
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> pofk(Nmesh/2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_direct_summation<NDIM>(
        Nmesh, 
        part.data(), 
        part.size(), 
        pofk);

    // To physical units and output
    pofk.scale(box);
    if(FML::ThisTask == 0){
      for(int i = 0; i < pofk.n; i++){
        std::cout << pofk.k[i] << " " << pofk.pofk[i] << "\n";
      }
    }
  }
  
  //======================================================================================
  // Compute power-spectrum multipoles
  //======================================================================================

  if(TEST_MULTIPOLES){
    
    if(FML::ThisTask == 0) std::cout << "Running compute_power_spectrum_multipoles\n";

    // Density assignment method and the number of extra slices we need for this
    auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);

    // Make density grid (make sure we have enough slices)
    FFTWGrid<NDIM> density_k(Nmesh, nleftright.first, nleftright.second);

    // Multipole computation
    std::vector< FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> > Pells(ell_max+1, Nmesh/2);
    double velocity_to_displacement = 1.0 / (100.0 * box);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles<NDIM>(
        Nmesh,
        p,
        velocity_to_displacement,
        Pells,
        density_assignment_method);

    // To physical units
    for(size_t ell = 0 ; ell < Pells.size(); ell++){
      Pells[ell].scale(box);
    }

    // Output
    if(FML::ThisTask == 0){
      for(int i = 0; i < Pells[0].n; i++){
        std::cout << Pells[0].k[i] << " ";
        for(size_t ell = 0 ; ell < Pells.size(); ell++){
          std::cout << Pells[ell].pofk[i] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  //======================================================================================
  // Compute power-spectrum multipoles 
  //======================================================================================

  if(TEST_MULTIPOLES_GRID){
    
    if(FML::ThisTask == 0) std::cout << "Running compute_power_spectrum_multipoles (from particles)\n";
    
    // Density assignment method and the number of extra slices we need for this
    auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(density_assignment_method);

    // Make density grid (make sure we have enough slices)
    FFTWGrid<NDIM> density_k(Nmesh, nleftright.first, nleftright.second);

#ifdef MEMORY_LOGGING
    // Example of memory logging
    density_k.add_memory_label("FFTWGrid::density_k");
    FML::MemoryLog::get()->print();
#endif

    // Interpolate particles to grid
    FML::INTERPOLATION::particles_to_grid<NDIM,Particle>(
        p.get_particles().data(),
        p.get_npart(),
        p.get_npart_total(),
        density_k,
        density_assignment_method);

    // Fourier transform it
    density_k.fftw_r2c();

    // Deconvolve window function
    FML::INTERPOLATION::deconvolve_window_function_fourier(density_k,  density_assignment_method);

    // Compute P_ell(k) when the LOS direction is the 
    const std::vector<double> los_direction{1.0, 0.0, 0.0};
    std::vector< FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<NDIM> > Pell(ell_max+1, Nmesh/2);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum_multipoles(
        density_k,
        Pell,
        los_direction);

    // Subtract shotnoise for P0
    for(int i = 0; i < Pell[0].n; i++){
      Pell[0].pofk[i] -= 1.0 / double(p.get_npart_total());
    }

    // To physical units
    for(size_t ell = 0 ; ell < Pell.size(); ell++){
      Pell[ell].scale(box);
    }

    // Output
    if(FML::ThisTask == 0){
      for(int i = 0; i < Pell[0].n; i++){
        std::cout << Pell[0].k[i] << " ";
        for(size_t ell = 0 ; ell < Pell.size(); ell++){
          std::cout << Pell[ell].pofk[i] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  //======================================================================================
  //======================================================================================
}

