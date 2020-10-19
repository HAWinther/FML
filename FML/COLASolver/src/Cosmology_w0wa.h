#ifndef COSMOLOGY_W0WA_HEADER
#define COSMOLOGY_W0WA_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

class BackgroundCosmologyw0waCDM final : public BackgroundCosmology {
  public:
    BackgroundCosmologyw0waCDM() { name = "w0waCDM"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        BackgroundCosmology::read_parameters(param);
        w0 = param.get<double>("cosmology_w0");
        wa = param.get<double>("cosmology_wa");
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
            std::cout << "# w0          : " << w0 << "\n";
            std::cout << "# wa          : " << wa << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // Hubble function
    //========================================================================
    double HoverH0_of_a(double a) const override {
        return std::sqrt(OmegaK / (a * a) + OmegaM / (a * a * a) + OmegaRtot / (a * a * a * a) +
                         OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * std::log(a)));
    }

    double dlogHdloga_of_a(double a) const override {
        double E = HoverH0_of_a(a);
        return 1.0 / (2.0 * E * E) *
               (-2.0 * OmegaK / (a * a) - 3.0 * OmegaM / (a * a * a) - 4.0 * OmegaRtot / (a * a * a * a) +
                OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * std::log(a)) *
                    (3.0 * wa * a - 3.0 * (1 + w0 + wa)));
    }

    double get_OmegaLambda(double a = 1.0) const override {
        double E = HoverH0_of_a(a);
        return OmegaLambda * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * std::log(a)) / (E * E);
    }

  protected:
    //========================================================================
    // Parameters specific to the w0waCDM model
    //========================================================================
    double w0;
    double wa;
};
#endif
