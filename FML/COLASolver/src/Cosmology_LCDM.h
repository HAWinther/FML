#ifndef COSMOLOGY_LCDM_HEADER
#define COSMOLOGY_LCDM_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Emulator/EuclidEmulator2.h>

#include "Cosmology.h"

class CosmologyLCDM final : public Cosmology {
  public:
    CosmologyLCDM() { name = "LCDM"; }

    double HoverH0_of_a(double a) const override {
        return std::sqrt(OmegaLambda + OmegaK / (a * a) + (OmegaCDM + Omegab) / (a * a * a) + OmegaR / (a * a * a * a) +
                         this->get_rhoNu_exact(a));
    }

    double dlogHdloga_of_a(double a) const override {
        double E = HoverH0_of_a(a);
        return 1.0 / (2.0 * E * E) *
               (-2.0 * OmegaK / (a * a) - 3.0 * (OmegaCDM + Omegab) / (a * a * a) - 4.0 * OmegaR / (a * a * a * a) +
                this->get_drhoNudloga_exact(a));
    }

    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
    
    //========================================================================
    // This method returns an estimate for the non-linear Pnl/Plinea
    // The fiducial option is to use the EuclidEmulator2
    //========================================================================
    Spline get_nonlinear_matter_power_spectrum_boost(double redshift) const override {
        Spline ee2_boost_of_k{"P/Plinear from EuclidEmulator2 uninitialized"};
        try {
            FML::EMULATOR::EUCLIDEMULATOR2::Cosmology ee2cosmo(Omegab, OmegaM, Mnu_eV, ns, h, -1.0, 0.0, As);
            if (ee2cosmo.is_good_to_use()) {
                FML::EMULATOR::EUCLIDEMULATOR2::EuclidEmulator ee2(ee2cosmo);
                auto result = ee2.compute_boost(redshift);
                ee2_boost_of_k = Spline(result.first, result.second, "P/Plinear from EuclidEmulator2");
            }
        } catch(...) {
            ee2_boost_of_k = Spline();
        }
        return ee2_boost_of_k;
    }

  protected:
};

#endif
