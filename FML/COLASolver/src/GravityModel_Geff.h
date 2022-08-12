#ifndef GRAVITYMODEL_GEFF_HEADER
#define GRAVITYMODEL_GEFF_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include <iostream>

using Spline = FML::INTERPOLATION::SPLINE::Spline;

/// Geff(a)-models
template <int NDIM>
class GravityModelGeff : public GravityModel<NDIM> {
  private:
    std::string geffofa_filename;
    Spline geffofa_spline;
  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    GravityModelGeff() : GravityModel<NDIM>("Geff(a)") {}
    GravityModelGeff(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "Geff(a)") {}

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
      return geffofa_spline(a); 
    }

    //========================================================================
    // Read parameters
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
    
        // The file containing [a,Geff/G(a)]
        geffofa_filename = param.get<std::string>("gravity_model_geff_geffofa_filename");

        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }
    
    virtual void init() {
      // Read file with [a,Geff/G(a)] and spline it
      std::ifstream fp(geffofa_filename.c_str());
      FML::assert_mpi(fp.good(), "GeffG file count not be opened\n");
      std::vector<double> aarr, geffarr;
      while(true){
        double newa, newgeff;
        fp >> newa;
        if(fp.eof()) break;
        fp >> newgeff;
        aarr.push_back(newa);
        geffarr.push_back(newgeff);
      }
      geffofa_spline.create(aarr, geffarr, "Geff/G(a)");

      GravityModel<NDIM>::init();
    }

    //========================================================================
    // Show some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "GeffG(a=1.00) = " << GeffOverG(1.00) << "\n";
            std::cout << "GeffG(a=0.66) = " << GeffOverG(0.66) << "\n";
            std::cout << "GeffG(a=0.50) = " << GeffOverG(0.50) << "\n";
            std::cout << "GeffG(a=0.33) = " << GeffOverG(0.33) << "\n";
            std::cout << "GeffG(a=0.25) = " << GeffOverG(0.25) << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
