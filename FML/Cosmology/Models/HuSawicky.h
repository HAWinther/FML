#ifndef FML_COSMOLOGY_MODELS_HUSAWICKY_H
#define FML_COSMOLOGY_MODELS_HUSAWICKY_H

// =============================================================================
// Hu-Sawicky f(R) gravity (Hu & Sawicki 2007), S = int sqrt(-g) [R + f(R)] / (16 pi G) + S_m, in the
// high-curvature regime where it is fixed by |f_R0| and n:
//   f_R = df/dR = -|f_R0| (R0/R)^(n+1),   f_RR = -(n+1) f_R / R
// The expansion is that of LCDM up to O(|f_R0|) corrections, which are neglected (as in MGCAMB and most
// analyses), so R = 8 pi G (rho - 3p) = 3 H0^2 (rho - 3p)/(rho_c0 c^2) of the LCDM species.
//
// Perturbations in the quasi-static approximation (e.g. Pogosian & Silvestri 2008), with the Compton
// wavelength of the scalaron, lambda_C^2 = 1/m^2 = 3 f_RR / (1 + f_R), and Q = (k/a)^2 f_RR / (1 + f_R)
// = (k/a)^2 lambda_C^2 / 3:
//   mu(k, a)  = (1 + 4 Q) / ((1 + 3 Q)(1 + f_R))    (4/3 below the Compton wavelength, 1 above)
//             = [1 + (1/3) k^2 / (k^2 + a^2 m^2)] / (1 + f_R)   (the Yukawa form)
//   Sigma(a)  = 1 / (1 + f_R)
// evolved with the mu-Sigma equations of MuSigma.h. Since the scalaron is heavy (m >> H) the quasi-static
// approximation is good on all scales where mu differs from 1.
// Note: the modification grows with k; set pert_k_max (and los_k_max) high enough for P(k) and lensing.
// =============================================================================

#include <FML/Cosmology/Models/MuSigma.h>

namespace FML::COSMOLOGY {

    class HuSawickyGravity : public MuSigmaGravity {
      public:
        HuSawickyGravity(double fR0, double n);
        std::string name() const override { return "Hu-Sawicky f(R)"; }
        std::string info() const override;
        void init_background(const BackgroundCosmology & bg) override;

        MuSigma mu_sigma(double x, double k, const BackgroundCosmology & bg) const override;
        // Ricci scalar [1/m^2], f_R, f_RR [m^2] and the Compton wavelength [m] of the background
        double R(double x, const BackgroundCosmology & bg) const;
        double fR(double x, const BackgroundCosmology & bg) const;
        double fRR(double x, const BackgroundCosmology & bg) const;
        double compton_wavelength(double x, const BackgroundCosmology & bg) const;
        // Tensors: M*^2 = 1 + f_R, alpha_M = f_RR dR/dx / (1 + f_R)
        TensorCoefficients tensor(double x, const BackgroundCosmology & bg) const override;

      private:
        double fR0, n;
        double lambdaC0_Mpc = 0.0; // Compton wavelength today (for info)
        // mu and Sigma at (x, k)
        void values(double x, double k, const BackgroundCosmology & bg, double & mu, double & Sigma) const;
    };

} // namespace FML::COSMOLOGY

#endif
