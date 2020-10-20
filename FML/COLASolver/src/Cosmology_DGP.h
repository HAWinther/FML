#ifndef COSMOLOGY_DGP_HEADER
#define COSMOLOGY_DGP_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

class BackgroundCosmologyDGP final : public BackgroundCosmology {
  public:
    BackgroundCosmologyDGP() { name = "DGP"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        BackgroundCosmology::read_parameters(param);
        OmegaRC = param.get<double>("cosmology_dgp_OmegaRC");
        this->OmegaLambda = 0.0;
        this->OmegaK = 1.0 - std::pow(std::sqrt(OmegaRC) +
                                          std::sqrt(OmegaRC + OmegaCDM + Omegab + OmegaR + this->get_rhoNu_exact(1.0)),
                                      2.0);
    }

    //========================================================================
    // Initialize the cosmology
    //========================================================================
    void init() override { BackgroundCosmology::init(); }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        BackgroundCosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# OmegaRC     : " << OmegaRC << "\n";
            std::cout << "# rcH0        : " << std::sqrt(0.25 / OmegaRC) << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Hubble function
    //========================================================================
    double HoverH0_of_a(double a) const override {
        double tmp = std::sqrt(OmegaRC) + std::sqrt(OmegaRC + (OmegaCDM + Omegab) / (a * a * a) +
                                                    OmegaR / (a * a * a * a) + this->get_rhoNu_exact(a));
        return std::sqrt(OmegaK / (a * a) + tmp * tmp);
    }

    double dlogHdloga_of_a(double a) const override {
        double tmp1 = std::sqrt(OmegaRC + (OmegaCDM + Omegab) / (a * a * a) + OmegaR / (a * a * a * a) +
                                this->get_rhoNu_exact(a));
        double tmp2 = std::sqrt(OmegaRC) + tmp1;
        double H2 = OmegaK / (a * a) + tmp2 * tmp2;
        return -OmegaK / (a * a * H2) + tmp2 * 1.0 / (2.0 * tmp1) *
                                            (-3.0 * (OmegaCDM + Omegab) / (a * a * a * H2) -
                                             4.0 * OmegaR / (a * a * a * a * H2) + this->get_drhoNudloga_exact(a) / H2);
    }

  protected:
    //========================================================================
    // Parameters specific to the DGP model
    //========================================================================
    double OmegaRC;
};
#endif
