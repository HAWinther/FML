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
#include <sstream>
#include <string>

using Spline = FML::INTERPOLATION::SPLINE::Spline;

/// Geff(a)-models
template <int NDIM>
class GravityModelGeff final : public GravityModel<NDIM> {
  private:
    std::string geffofa_filename;
    Spline geffofa_spline;

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    explicit GravityModelGeff(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "Geff(a)") {}

    //========================================================================
    // G_eff/G(a) from the table (the force and the growth are the defaults of GravityModel)
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0) const override { return geffofa_spline(a); }

    //========================================================================
    // Read parameters
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);

        // The file containing [a,Geff/G(a)]
        geffofa_filename = param.get<std::string>("gravity_model_geff_geffofa_filename");
    }

    virtual void init() override {
        // Read file with [a,Geff/G(a)] and spline it
        std::ifstream fp(geffofa_filename.c_str());
        FML::assert_mpi(fp.good(), "GeffG file count not be opened\n");
        std::vector<double> aarr, geffarr;
        // Rows "a GeffG"; empty lines and lines starting with # are skipped (a header line used to make the
        // reader loop forever: >> failed without reaching eof)
        std::string line;
        while (std::getline(fp, line)) {
            const auto first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos or line[first] == '#')
                continue;
            std::istringstream is(line);
            double newa, newgeff;
            FML::assert_mpi(static_cast<bool>(is >> newa >> newgeff),
                            ("GeffG file: could not read the row [" + line + "] (expected 'a GeffG')").c_str());
            aarr.push_back(newa);
            geffarr.push_back(newgeff);
        }
        FML::assert_mpi(aarr.size() >= 2, "GeffG file: needs at least 2 rows");
        geffofa_spline.create(aarr, geffarr, "Geff/G(a)");

        GravityModel<NDIM>::init();
    }

    //========================================================================
    // Show some info
    //========================================================================
    void info_model() const override {
        if (FML::ThisTask == 0) {
            std::cout << "GeffG(a=1.00) = " << GeffOverG(1.00) << "\n";
            std::cout << "GeffG(a=0.66) = " << GeffOverG(0.66) << "\n";
            std::cout << "GeffG(a=0.50) = " << GeffOverG(0.50) << "\n";
            std::cout << "GeffG(a=0.33) = " << GeffOverG(0.33) << "\n";
            std::cout << "GeffG(a=0.25) = " << GeffOverG(0.25) << "\n";
        }
    }
};

#endif
