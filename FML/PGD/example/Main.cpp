#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PGD/PotentialGradientDecent.h>

//======================================================================
// This method:
// 1) reads in particles from gadgetfiles
// 2) applies the PGD method to shift the particle positions
// 3) computes P(k) and compares P(k) / Poriginal(k)
// Also takes a copy of the original positions in tmp in case
// one need this. They can be restored by calling restore_original_positions
//======================================================================

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
template <class T1>
using MPIParticles = FML::PARTICLE::MPIParticles<T1>;
using GadgetReader = FML::FILEUTILS::GADGET::GadgetReader;
using GadgetWriter = FML::FILEUTILS::GADGET::GadgetWriter;

struct Particle {
  double pos[3];
  double vel[3];
  double tmp[3];
  int id;
  int get_ndim() { return 3; }
  double * get_pos() { return pos; }
  double * get_vel() { return vel; }
  int get_id() { return id; }
  void set_id(long long int _id) { id = _id; }
};

void compute_pofk(
    MPIParticles<Particle> & part, 
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> & pofk, 
    double boxsize, 
    int ngrid, 
    std::string density_assignment_method);
void restore_original_positions(MPIParticles<Particle> & mpipart);
void pgd(
    MPIParticles<Particle> & part,
    int ngrid,
    double boxsize_in_mpch,
    double redshift,
    double OmegaM,
    double alpha, 
    double k_lowpass_hmpc, 
    double k_highpass_hmpc);

int main() {

  // Path + prefix for gadget files
  const std::string pathandfileprefix = "gadget_z0.000";

  // Read the gadgetfiles
  GadgetReader g;
  FML::Vector<Particle> part;
  const bool only_keep_part_in_domain = true;
  const double buffer_factor = 1.25;
  const bool verbose = false;
  g.read_gadget(pathandfileprefix, part, buffer_factor, only_keep_part_in_domain, verbose);
  const auto header = g.get_header();
  const double boxsize_in_mpch = header.BoxSize;
  const double redshift = header.redshift;
  const double OmegaM = header.Omega0;
  size_t NumPartTotal = part.size();
  FML::SumOverTasks(&NumPartTotal);
  FML::FILEUTILS::GADGET::print_header_info(header);

  // Make MPIParticles
  MPIParticles<Particle> mpipart;
  mpipart.move_from(std::move(part));
  
  // Make a copy of the original positions in tmp
  auto NumPart = mpipart.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for(size_t ipart = 0; ipart < NumPart; ipart++){
    auto * pos = mpipart[ipart].pos;
    auto * tmp = mpipart[ipart].tmp;
    for(int idim = 0; idim < 3; idim++)
      tmp[idim] = pos[idim];
  }

  // Parameters for the PGD method
  const int ngrid = 128;
  const double Delta = boxsize_in_mpch / std::pow(NumPartTotal, 1.0/3.0);
  double alpha = FML::PGD::pgd_fitting_formula_alpha(Delta, 1.0/(1.0+redshift));
  double k_lowpass_hmpc = FML::PGD::pgd_fitting_formula_klowpass_hmpc(Delta);
  double k_highpass_hmpc = FML::PGD::pgd_fitting_formula_khighpass_hmpc(Delta);
  
  // Compute P(k) of original positions
  FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk_fid(ngrid / 2);
  compute_pofk(mpipart, pofk_fid, boxsize_in_mpch, ngrid, "CIC");
  
  // Do the PGD method
  pgd(mpipart,
      ngrid,
      boxsize_in_mpch,
      redshift,
      OmegaM,
      alpha, 
      k_lowpass_hmpc, 
      k_highpass_hmpc);

  // Compute whatever you want...
  // Compute P(k) of positions after PGD
  FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(ngrid / 2);
  compute_pofk(mpipart, pofk, boxsize_in_mpch, ngrid, "CIC");
  if (FML::ThisTask == 0) {
    for (int i = 0; i < pofk.n; i++) {
      std::cout << std::setw(15) << pofk.k[i] << " ";
      std::cout << std::setw(15) << pofk.pofk[i]/pofk_fid.pofk[i] << " ";
      std::cout << "\n";
    }
    std::cout << "\n";
  }
  
  // Restore the original positions so we can do more calculations...
  restore_original_positions(mpipart);

  // ...
  // ...
  // ...
}

void compute_pofk(MPIParticles<Particle> & part, 
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> & pofk, 
    double boxsize, 
    int ngrid, 
    std::string density_assignment_method){

  // Power-spectrum evaluation using interlacing
  const bool interlacing = true;
  FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3>(
      ngrid, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, density_assignment_method, interlacing);

  // To physical units and output
  pofk.scale(boxsize);
}

void restore_original_positions(MPIParticles<Particle> & mpipart){
  auto NumPart = mpipart.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for(size_t ipart = 0; ipart < NumPart; ipart++){
    auto * pos = mpipart[ipart].pos;
    auto * tmp = mpipart[ipart].tmp;
    for(int idim = 0; idim < 3; idim++)
        pos[idim] = tmp[idim];
  }
  mpipart.communicate_particles();
}

void pgd(MPIParticles<Particle> & part,
    int ngrid,
    double boxsize_in_mpch,
    double redshift,
    double OmegaM,
    double alpha, 
    double k_lowpass_hmpc, 
    double k_highpass_hmpc) {

  // Fetch simulation data
  auto scale_factor = 1.0 / (1.0 + redshift);
  auto NumPartTotal = size_t(part.get_npart());
  FML::SumOverTasks(&NumPartTotal);

  const double Delta = boxsize_in_mpch / std::pow(NumPartTotal, 1.0/3.0);
  double kl = k_lowpass_hmpc * boxsize_in_mpch;
  double ks = k_highpass_hmpc * boxsize_in_mpch;
  if(FML::ThisTask == 0){
    std::cout << "#=====================================================\n";
    std::cout << "# Potential gradient decent method\n";
    std::cout << "# ks    = " << ks/boxsize_in_mpch << " h/Mpc\n";
    std::cout << "# kl    = " << kl/boxsize_in_mpch << " h/Mpc\n";
    std::cout << "# Delta = " << Delta << "\n";
    std::cout << "# alpha = " << alpha << "\n";
    std::cout << "#=====================================================\n";
  }

  FML::PGD::potential_gradient_decent_method<3,Particle>(part.get_particles_ptr(),
      part.get_npart(),
      kl,
      ks,
      alpha,
      1.5*OmegaM*scale_factor,
      ngrid,
      "CIC");

  // Particles have moved to make sure they are on the right CPU
  part.communicate_particles();
}

