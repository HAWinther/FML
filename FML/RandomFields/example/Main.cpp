#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>

template<int N>
  using FFTWGrid = FML::GRID::FFTWGrid<N>;

int main(){
  
  // Set up the grid 
  const double box = 1.0;
  const int Nmesh  = 1024;
  const int Ndim   = 2;
  FFTWGrid<Ndim> grid(Nmesh);
  grid.info();

  // To generate a gaussian random field we need a RNG generator
  // and a power-spectrum
  FML::RANDOM::RandomGenerator *rng = new FML::RANDOM::RandomGenerator;
  std::function<double(double)> Powspec;
  const bool fix_amplitude = true;

  // This is P(k) / Volume with P(k) = 1/k^3
  Powspec = [&](double kBox){
    double k = kBox / box; // k in physical units
    double pofk = std::pow(1.0/k, Ndim); // pofk in physical units
    return pofk / std::pow(box,Ndim); // Dimensionless P/V
  };

  // Make a random field in fourier space
  //FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(grid, rng, Powspec, fix_amplitude);
  
  // Test generating non-gaussian potential
  FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier(grid, rng, Powspec, fix_amplitude, +0.1, "local");

  // Compute power-spectrum
  FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning pofk(Nmesh/2);
  FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(grid, pofk);

  // To physical units and output k, P(k)
  pofk.scale(1.0/box, std::pow(box,Ndim));
  if(FML::ThisTask == 0){
    std::cout << "k    P(k) / Pinput(k): \n";
    for(int i = 0; i < pofk.n; i++){
      double integer_k = pofk.k[i] * box / (2.0*M_PI);
      double pofk_over_pofk_input = pofk.pofk[i] / ( Powspec(pofk.k[i] * box) * std::pow(box,Ndim) );
      std::cout << integer_k << " " << pofk_over_pofk_input << "\n";
    }
  }

  // Compute density PDF
  grid.fftw_c2r();
  double delta_min  = 1e100;
  double delta_max  = -1e100;
  double delta_mean = 0.0;
  for(auto & real_index : grid.get_real_range()){
    auto delta  = grid.get_real_from_index(real_index);
    if(delta < delta_min) delta_min = delta;
    if(delta > delta_max) delta_max = delta;
    delta_mean += delta;
  }
#ifdef USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, &delta_min,  1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &delta_max,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &delta_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  delta_mean /= std::pow(grid.get_nmesh(), Ndim);
  
  if(FML::ThisTask == 0)
    std::cout << "Min/Max/Mean density contrast: " << delta_min << " " << delta_max << " " << delta_mean << "\n";

  delta_min = -5.0;
  delta_max =  5.0;

  int nbins = 100;
  std::vector<double> count(nbins,0.0);
  for(auto & real_index : grid.get_real_range()){
    auto delta  = grid.get_real_from_index(real_index);
    int index = int((delta - delta_min)/(delta_max-delta_min)*nbins);
    if(index < nbins and index >= 0){
      count[index] += 1.0;
    }
  }
#ifdef USE_MPI
  MPI_Allreduce(MPI_IN_PLACE, count.data(), nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if(FML::ThisTask == 0){
    for(int i = 0; i < nbins; i++)
      std::cout << delta_min + (delta_max-delta_min)*i/double(nbins) << " " << count[i] << "\n";
  }
}




