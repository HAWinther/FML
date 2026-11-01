#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HUSAWICKY_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_HUSAWICKY_H

// =====================================================================================================================
// Hu-Sawicky f(R) gravity (Hu & Sawicki 2007), S = int sqrt(-g) [R + f(R)] / (16 pi G) + S_m, in the high-curvature
// regime fixed by |f_R0| and n:
//   f_R = -|f_R0| (R0/R)^(n+1),   f_RR = -(n+1) f_R / R,   R = 3 H0^2 (rho - 3p) / (rho_c0 c^2)
// of the species of the (LCDM) background; the O(|f_R0|) corrections to the expansion are neglected (as MGCAMB).
// Quasi-static perturbations (Pogosian & Silvestri 2008) with Q = (k/a)^2 f_RR / (1 + f_R) = (k/a)^2 lambda_C^2 / 3:
//   mu(k, a) = (1 + 4 Q) / ((1 + 3 Q)(1 + f_R)),   Sigma(a) = 1 / (1 + f_R)
// with the mu-Sigma equations (PerturbedFieldEquation_MuSigma.h); d(2 Sigma - mu)/dx at fixed k by central
// differences (step 1e-4). Tensors: M*^2 = 1 + f_R, alpha_M = -(n+1) f_R dlnR/dx / (1 + f_R) (dlnR/dx by a central
// difference, step 1e-4). The modification grows with k: set the k range of the perturbations high enough for P(k).
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_HuSawicky : public PerturbedFieldEquation_MuSigma {
      public:
        PerturbedFieldEquation_HuSawicky(double fR0, double n, const PerturbationContext & ctx)
            : PerturbedFieldEquation_MuSigma(ctx), fR0(fR0), n(n) {
            if (not(fR0 >= 0.0))
                throw std::invalid_argument("Hu-Sawicky: need fR0 = |f_R0| >= 0");
            if (not(n > 0.0))
                throw std::invalid_argument("Hu-Sawicky: need n > 0");
            R0 = R(0.0);
            lambdaC0_Mpc = compton_wavelength(0.0) / FML::UTILS::SI::Mpc;
        }
        std::string name() const override { return "Hu-Sawicky f(R)"; }
        std::string info() const override {
            std::ostringstream os;
            os << "Hu-Sawicky f(R) (|f_R0| = " << fR0 << ", n = " << n << ", Compton wavelength today " << lambdaC0_Mpc
               << " Mpc; LCDM background, quasi-static)";
            return os.str();
        }

        /// Ricci scalar [1/m^2], f_R, f_RR [m^2] and the Compton wavelength [m] of the background
        double R(double x) const {
            const auto & bg = ctx.background;
            const double c = FML::UTILS::SI::c;
            return 3.0 * ctx.H0 * ctx.H0 * (bg.rho_total(x) - 3.0 * bg.p_total(x)) / (c * c);
        }
        double fR(double x) const { return -fR0 * std::pow(R0 / R(x), n + 1.0); }
        double fRR(double x) const { return -(n + 1.0) * fR(x) / R(x); }
        double compton_wavelength(double x) const { return std::sqrt(3.0 * fRR(x) / (1.0 + fR(x))); }

        /// d(2 Sigma - mu)/dx at fixed k by central differences (all terms are smooth functions of x)
        MuSigma mu_sigma(double x, double k) const override {
            MuSigma m;
            values(x, k, m.mu, m.Sigma);
            const double h = 1e-4;
            double mp, sp, mm, sm;
            values(x + h, k, mp, sp);
            values(x - h, k, mm, sm);
            m.dg = ((2.0 * sp - mp) - (2.0 * sm - mm)) / (2.0 * h);
            return m;
        }
        TensorCoefficients tensor(double x) const override {
            const double h = 1e-4;
            const double dlnR = (std::log(R(x + h)) - std::log(R(x - h))) / (2.0 * h);
            const double f = fR(x);
            return {1.0 + f, -(n + 1.0) * f * dlnR / (1.0 + f), 0.0};
        }

      private:
        double fR0, n, R0 = 0.0, lambdaC0_Mpc = 0.0;

        void values(double x, double k, double & mu, double & Sigma) const {
            const double Rx = R(x), f_R = -fR0 * std::pow(R0 / Rx, n + 1.0), f_RR = -(n + 1.0) * f_R / Rx;
            const double ka = k * std::exp(-x);
            const double Q = f_RR * ka * ka / (1.0 + f_R); // = (k/a)^2 lambda_C^2 / 3
            Sigma = 1.0 / (1.0 + f_R);
            mu = (1.0 + 4.0 * Q) / (1.0 + 3.0 * Q) * Sigma;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
