#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/MemoryLogging/MemoryLogging.h>

//=====================================================
// 
// In this example we generate a gaussian random field
// delta from a given power-spectrum
//
// From this we compute the LPT displacement field to 
// second order
//
// We use this (just the 1LPT field some simlicity) 
// to generate particle positions
//
// Then we compute the power-spectrum of this particle
// distribution and compare it to the input P(k)
//
// So this test is a simple IC generator for N-body
// simulations
//
//=====================================================

const int Ndim = 2;
template<class T>
  using MPIParticles = FML::PARTICLE::MPIParticles<T>;
template<int N>
  using FFTWGrid = FML::GRID::FFTWGrid<N>;
using Particle = SimpleParticle<Ndim>;
 
//=====================================================
// The boxsize in your physical units (say Mpc/h)
//=====================================================
const double box = 100.0;

//=====================================================
// Power-spectrum in physical units 
// (i.e. Units of L^NDIM where L is the units of box)
//=====================================================
double power_spectrum(double k){
  if(k == 0.0) return 0.0;
  return 0.00001;
}

//=====================================================
// To generate a gaussian random field we need a RNG generator
// and a power-spectrum
//=====================================================
void generate_delta(FFTWGrid<Ndim> &delta){
  FML::RANDOM::RandomGenerator *rng = new FML::RANDOM::RandomGenerator;
  std::function<double(double)> Powspec;
  const bool fix_amplitude = true;

  // Function P(kBox / Box) / Volume  
  Powspec = [&](double kBox){
    return power_spectrum( kBox / box ) / std::pow(box,Ndim);
  };

  // Make a random field in fourier space
  FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(delta, rng, Powspec, fix_amplitude);
}

int main(){
#ifdef MEMORY_LOGGING
  auto * mem = FML::MemoryLog::get();
#endif

  const int Nmesh = 256;
  const int Npart_1D = 256;
  const double buffer_factor = 1.25;
  std::string interpolation_method = "CIC";

  //=====================================================
  // Generate density field (the extra slices we use in this grid 
  // propagates to the grids we generate below)
  //=====================================================
  auto nextra = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);
  FFTWGrid<Ndim> delta(Nmesh, nextra.first, nextra.second);
  generate_delta(delta);

  //=====================================================
  // Generate phi_1LPT = delta(k)/k^2
  //=====================================================
  FFTWGrid<Ndim> phi_1LPT;
  FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(
      delta,
      phi_1LPT);

  //=====================================================
  // Generate phi_2LPT
  // 7 FFTS
  //=====================================================
  FFTWGrid<Ndim> phi_2LPT;
  FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(
      delta,
      phi_2LPT);

  //=====================================================
  // Generate displacement field Psi = Dphi
  // 3+3 FFTS
  //=====================================================
  std::vector<FFTWGrid<Ndim>> Psi_1LPT_vector;
  FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector(
      phi_1LPT,
      Psi_1LPT_vector);
  std::vector<FFTWGrid<Ndim>> Psi_2LPT_vector;
  FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector(
      phi_2LPT,
      Psi_2LPT_vector);

  //=====================================================
  // We no longer need delta and the displacemnt fields
  // so we can free up memory
  //=====================================================
  delta.free();
  phi_1LPT.free();
  //phi_2LPT.free();

  //===============================Y======================
  // Make a regular particle grid (with Npart_1D^NDIM 
  // particles in total)
  //=====================================================
  MPIParticles<Particle> part;
  part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);

  //=====================================================
  // Interpolate displacement field to particle positions
  // (Note: if Npart_1D = Nmesh then we could assign directly 
  // without interpolation using the disp fields)
  //=====================================================
  std::vector<std::vector<FML::GRID::FloatType>> displacements_1LPT(Ndim);
  std::vector<std::vector<FML::GRID::FloatType>> displacements_2LPT(Ndim);
  for(int idim = Ndim-1; idim >= 0; idim--){
   if(FML::ThisTask == 0) std::cout << "Assigning particles for idim = " << idim << "\n";
    std::vector<FML::GRID::FloatType> & interpolated_values_1LPT = displacements_1LPT[idim];
    FML::INTERPOLATION::interpolate_grid_to_particle_positions(
        Psi_1LPT_vector[idim],
        part.get_particles_ptr(),
        part.get_npart(),
        interpolated_values_1LPT,
        interpolation_method);
    std::vector<FML::GRID::FloatType> & interpolated_values_2LPT = displacements_2LPT[idim];
    FML::INTERPOLATION::interpolate_grid_to_particle_positions(
        Psi_2LPT_vector[idim],
        part.get_particles_ptr(),
        part.get_npart(),
        interpolated_values_2LPT,
        interpolation_method);
  }
  
  //=====================================================
  // We no longer need Psi
  //=====================================================
  for(int idim = Ndim-1; idim >= 0; idim--){
    Psi_1LPT_vector[idim].free();
    Psi_2LPT_vector[idim].free();
  }

  //=====================================================
  // Add displacement to particle position
  // Note the -3/7 factor. This is because growth factors
  // are defined to be == 1 at the initial time, however
  // the physically relevant solution has D2 = -3/7 D1^2
  // so we need to put this in by hand
  //=====================================================
  double max_disp_1LPT = 0.0;
  double max_disp_2LPT = 0.0;
  auto *part_ptr = part.get_particles_ptr();
  for(size_t ind = 0; ind < part.get_npart(); ind++) {
    auto *pos = part_ptr[ind].get_pos();
    for(int idim = Ndim-1; idim >= 0; idim--){
      double dpos_1LPT = displacements_1LPT[idim][ind];
      double dpos_2LPT = - 3.0/7.0 * displacements_2LPT[idim][ind];
      pos[idim] += dpos_1LPT + dpos_2LPT;
      if(std::fabs(dpos_1LPT) > max_disp_1LPT) max_disp_1LPT = std::fabs(dpos_1LPT);
      if(std::fabs(dpos_2LPT) > max_disp_2LPT) max_disp_2LPT = std::fabs(dpos_2LPT);
      if(pos[idim] >= 1.0) pos[idim] -= 1.0;
      if(pos[idim] < 0.0) pos[idim] += 1.0;
    }
  }
  FML::MaxOverTasks(&max_disp_1LPT);
  FML::MaxOverTasks(&max_disp_2LPT);
  if(FML::ThisTask == 0) std::cout << "Maximum displacements: " << max_disp_1LPT * Nmesh << " grid cells\n";
  if(FML::ThisTask == 0) std::cout << "Maximum displacements: " << max_disp_2LPT * Nmesh << " grid cells\n";
  
  //=====================================================
  // We no longer need displacements
  //=====================================================
  displacements_1LPT.clear();
  displacements_2LPT.clear();
  displacements_1LPT.shrink_to_fit();
  displacements_2LPT.shrink_to_fit();
  part.communicate_particles();
  
  /*
  // Output particles
  for(size_t ind = 0; ind < part.get_npart(); ind++) {
  auto *pos = part_ptr[ind].get_pos();
  std::cout << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
  }
   */

  //=====================================================
  // Lets test that it works as expected
  // Compute power-spectrum with a larger grid
  //=====================================================
  FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning pofk(Nmesh/2);
  FML::CORRELATIONFUNCTIONS::compute_power_spectrum_interlacing<Ndim>(
      2*Nmesh,
      part.get_particles_ptr(),
      part.get_npart(),
      part.get_npart_total(),
      pofk,
      interpolation_method);

  //=====================================================
  // Add back shot-noise subtracted in the routine above
  // (The signal drowns otherwise)
  //=====================================================
  for(int i = 0; i < pofk.n; i++){
    pofk.pofk[i] += 1.0/double(part.get_npart_total());
  }

  //=====================================================
  // Convert to physical units
  //=====================================================
  pofk.scale(1.0/box, pow(box,Ndim));
  if(FML::ThisTask == 0){
    for(int i = 0; i < pofk.n; i++){
      double k = pofk.k[i];
      std::cout << k << " " << pofk.pofk[i] / power_spectrum(k) << "\n";
    }
  }

  //=====================================================
  // Free the last of the memory before printing
  //=====================================================
  part.free();

#ifdef MEMORY_LOGGING
  // Print memory summary
  mem->print();
#endif
}
