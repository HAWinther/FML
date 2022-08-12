#ifndef GRAVITYMODEL_JBD_HEADER
#define GRAVITYMODEL_JBD_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology_JBD.h"
#include "GravityModel.h"

/// Jordan-Brans-Dicke
template <int NDIM>
class GravityModelJBD : public GravityModel<NDIM> {
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelJBD() : GravityModel<NDIM>("JBD") {}
    GravityModelJBD(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "JBD") {
        FML::assert_mpi(
            cosmo->get_name() == "JBD",
            "As currently written the JBD gravity model requires a JBD cosmology as we need phi(a) from there");
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi = norm_poisson_equation * delta
    //========================================================================
    void compute_force(double a,
                       [[maybe_unused]] double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Computes gravitational force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a * GeffOverG(a);
        FML::NBODY::compute_force_from_density_fourier<NDIM>(
            density_fourier, force_real, density_assignment_method_used, norm_poisson_equation);
    }

    //========================================================================
    // In JBD GeffOverG = 1/phi. The value at 1/phi(a=1) is the parameter
    // cosmology_JBD_GeffG_today
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0) const override { 
      CosmologyJBD * jbd = dynamic_cast<CosmologyJBD *>(this->cosmo.get());
      return 1.0 / jbd->get_phi(a); 
    }

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
