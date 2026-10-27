#ifndef COSMOLOGY_LCDM_HEADER
#define COSMOLOGY_LCDM_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>
#include <FML/Emulator/EuclidEmulator2.h>

#include "EE2Boost.h"

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
        // EuclidEmulator2 takes A_s at the pivot 0.05/Mpc: convert from our pivot (the factor is 1 for kpivot = 0.05)
        const double As_ee2 = As * std::pow(0.05 / kpivot_mpc, ns - 1.0);
        return ee2_boost.boost(Omegab, OmegaM, Mnu_eV, ns, h, -1.0, 0.0, As_ee2, redshift);
    }

  protected:
    // The emulator is built once (see EE2Boost.h)
    EE2BoostCache ee2_boost;
};

#endif
