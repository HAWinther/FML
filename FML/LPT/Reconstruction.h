#ifndef RECONSTRUCTION_HEADER
#define RECONSTRUCTION_HEADER
#include <vector>
#include <complex>
#include <numeric>
#include <functional>
#include <cassert>
#include <cstdio>
#include <climits>

#include <FML/Global/Global.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/LPT/DisplacementFields.h>

namespace FML {
  namespace COSMOLOGY {
    namespace LPT {

      template<int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;
      template<class T>
        using MPIParticles = FML::PARTICLE::MPIParticles<T>;
      using FloatType = FML::GRID::FloatType;

      //============================================================================
      // 
      // RSD removal for particles in a periodic box
      // We use a fixed line of sight direction los_direction, typically
      // coordinate axes (e.g. los_direction = {0,0,1} )
      // We are solving the equation LPT equation
      //  D(bPsi) + beta D((bPsi*r)r) = -delta_observed
      // and assuming zero-curl (which is not perfectly true) of the second term
      // so we can use Fourier methods
      // Then we subtract the RSD which is given by Psi_rsd = beta (bPsi*r)r
      // We then iterate this procedure niterations times
      // Here beta = f / b growth-rate over galaxy bias
      // smoothing_options = {smoother_filter, smothing_scale} with smothing_scale is units of the box
      // The smoothing filter is an (optional) filter we apply to the density field in 
      // k-space. Options here are sharpk, tophat, gaussian. If R < gridspacing then no
      // smoothing will be done
      //
      // This also works for a survey. Then los_direction is the observer position
      // and we need to have a box big enough so that particles don't wrap around
      // when shifted!
      //
      // This method assumes scale-independent growth factor, but that is easily changed
      //
      //============================================================================

      template<int N, class T>
        void RSDReconstructionFourierMethod(
            MPIParticles<T> &part,
            std::string density_assignment_method,
            std::vector<double> los_direction,
            int Nmesh,
            int niterations,
            double beta,
            std::pair<std::string,double> smoothing_options,
            bool survey_data)
        {
          const std::string interpolation_method = density_assignment_method;
          size_t NumPart = part.get_npart();

          // Normalize the los_direction to a unit vector
          assert(los_direction.size() == N);
          double norm = 0.0;
          for(int idim = 0; idim < N; idim++){
            norm += los_direction[idim]*los_direction[idim];
          }
          norm = 1.0 / std::sqrt(norm);
          assert(norm > 0.0);
          for(int idim = 0; idim < N; idim++){
            los_direction[idim] *= norm;
          }

          // Do this iteratively
          for(int i = 0; i < niterations; i++){

            // This is the density field for the observed galaxies
            // i.e. with RSD in it
            FFTWGrid<N> density(Nmesh,1,1);
            density.add_memory_label("FFTWGrid::RSDReconstructionFourierMethod::density");
            FML::INTERPOLATION::particles_to_grid(
                part.get_particles_ptr(), 
                part.get_npart(), 
                part.get_npart_total(), 
                density,
                density_assignment_method);

            // The 1LPT potential
            FFTWGrid<N> phi_1LPT;
            compute_1LPT_potential_fourier(
                density, 
                phi_1LPT);

            // Free some memory
            density.free();

            // The 1LPT displacement field
            std::vector<FFTWGrid<N> > Psi;
            from_LPT_potential_to_displacement_vector(
                phi_1LPT,
                Psi);

            // Free some memory
            phi_1LPT.free();

            // Interpolate Psi to particle positions
            std::vector<std::vector<FloatType>> Psi_particle_positions;
            for(int idim = 0; idim < N; idim++){
              FML::INTERPOLATION::interpolate_grid_to_particle_positions(
                  Psi[idim], 
                  part.get_particles_ptr(),
                  part.get_npart(),
                  Psi_particle_positions[idim],
                  interpolation_method);

              // Free some memory
              for(int idim = 0; idim < N; idim++){
                Psi[idim].free();
              }

              // Subtract the RSD component (Psi*r)*r / (1+beta) for each particle
              // Do periodic wrap and communicate particles in case they have left
              // the current domain
              auto *p = part.get_particles_ptr();
              for(size_t i = 0; i < NumPart; i++){
                auto *pos = p->get_pos();

                FloatType r[N];
                if(survey_data){
                  double norm = 0.0;
                  for(int idim = 0; idim < N; idim++){
                    r[idim] = pos[idim] - los_direction[idim];
                    norm += r[idim]*r[idim];
                  }
                  norm = 1.0 / std::sqrt(norm);
                  for(int idim = 0; idim < N; idim++){
                    r[idim] *= norm;
                  }
                } else {
                  for(int idim = 0; idim < N; idim++){
                    r[idim] = los_direction[idim];
                  }
                }

                FloatType Psi_rsd[N];
                FloatType Psidotr = 0.0;
                for(int idim = 0; idim < N; idim++){
                  Psidotr += los_direction[idim] * Psi_particle_positions[idim][i];
                }
                for(int idim = 0; idim < N; idim++){
                  Psi_rsd[idim] = Psidotr * los_direction[idim] / (1.0 + beta);
                }

                // For survey we need to have a box big enough so that we don't wrap around
                for(int idim = 0; idim < N; idim++){
                  pos[idim] -= Psi_rsd[idim];
                  if(pos[idim] < 0.0) pos[idim]  += 1.0;
                  if(pos[idim] >= 1.0) pos[idim] -= 1.0;
                }
              }
              part.communicate_particles();
            }
          }
        }
    }
  }
}
#endif
