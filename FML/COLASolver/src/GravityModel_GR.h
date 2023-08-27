#ifndef GRAVITYMODEL_GR_HEADER
#define GRAVITYMODEL_GR_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

/// General Relativity
template <int NDIM>
class GravityModelGR final : public GravityModel<NDIM> {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelGR() : GravityModel<NDIM>("GR") {}
    GravityModelGR(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "GR") {}

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi = norm_poisson_equation * delta
    //========================================================================
    void compute_force(double a,
                       [[maybe_unused]] double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        
        if (this->force_use_finite_difference_force) {
          // Use a by default a 4 point formula (using phi(i+/-2), phi(i+/-1) to compute DPhi)
          // This requires 2 boundary cells (stencil_order=2,4,6 implemented so far)
          const int stencil_order = this->force_finite_difference_stencil_order;
          const int nboundary_cells = stencil_order/2;

          FFTWGrid<NDIM> potential_real;
          FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fourier,
              potential_real,
              norm_poisson_equation,
              nboundary_cells);

          FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
              force_real,
              density_assignment_method_used,
              stencil_order);

        } else {
          // Computes gravitational force using fourier-methods
          FML::NBODY::compute_force_from_density_fourier<NDIM>(
              density_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
        }
    }

    //========================================================================
    // In GR GeffOverG == 1
    //========================================================================
    double GeffOverG([[maybe_unused]] double a, [[maybe_unused]] double koverH0 = 0) const override { return 1.0; }

    //========================================================================
    // Read parameters
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }
    
    //========================================================================
    // Show some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
