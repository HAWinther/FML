#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_HUSAWICKY_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_HUSAWICKY_H

// =====================================================================================================================
// Hu-Sawicky f(R) gravity (Hu & Sawicki 2007) in the high-curvature regime fixed by |f_R0| and n (QuasiStaticGravity.h):
//   f_R = -|f_R0| (R0/R)^(n+1),   f_RR = -(n+1) f_R / R
// with R the Ricci scalar of the background (the O(|f_R0|) corrections to the expansion are neglected, as MGCAMB),
// in units of 3 H0^2/c^2:
//   R c^2 / (3 H0^2) = rho - 3p                                  (all species; on a ComponentBackground)
//                    = 2 E^2 (2 + dlnE/dx) - 2 Omega_k e^{-2x}    (any background; the same for GR)
// Quasi-static perturbations (Pogosian & Silvestri 2008), Q = (k/a)^2 f_RR / (1 + f_R) = (k/a)^2 / (3 m^2):
//   mu(k, a) = (1 + 4 Q) / ((1 + 3 Q)(1 + f_R)) = [1 + (1/3) k^2 / (k^2 + a^2 m^2)] / (1 + f_R),   Sigma = 1 / (1 + f_R)
//   m^2 c^2 / H0^2 = (1 + f_R) R c^2 / (3 H0^2 (n+1) |f_R|)    (the scalaron mass; m = 1/lambda_C)
// d(2 Sigma - mu)/dx at fixed k: central differences (step 1e-4).
// Screening (chameleon, by the Newtonian potential): the fifth force is suppressed where |Phi_N| > Phi_crit =
// (3/2)|f_R(a)| (thin shell): factor = min(1, efficiency Phi_crit / |Phi_N|).
// =====================================================================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_HuSawicky : public QuasiStaticGravity {
      public:
        QuasiStaticGravity_HuSawicky(const BACKGROUND::Background & bg, double fR0, double n)
            : QuasiStaticGravity(bg), fR0(fR0), n(n), cbg(dynamic_cast<const BACKGROUND::ComponentBackground *>(&bg)) {
            if (not(fR0 >= 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_HuSawicky] Need fR0 = |f_R0| >= 0");
            if (not(n > 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_HuSawicky] Need n > 0");
            R0 = R(0.0);
        }
        std::string name() const override { return "Hu-Sawicky f(R)"; }
        std::string info() const override {
            std::ostringstream os;
            os << "Hu-Sawicky f(R) (|f_R0| = " << fR0 << ", n = " << n << ", Compton wavelength today "
               << compton_wavelength(0.0) * FML::UTILS::SI::c_over_H0_mpch << " Mpc/h; quasi-static)";
            return os.str();
        }
        bool scale_dependent() const override { return true; }

        /// The Ricci scalar in units of 3 H0^2 / c^2
        double R(double x) const {
            if (cbg)
                return cbg->rho_total(x) - 3.0 * cbg->p_total(x);
            const double E = bg.E(x);
            return 2.0 * E * E * (2.0 + bg.dlogEdx(x)) - 2.0 * bg.parameters().Omega_k * std::exp(-2.0 * x);
        }
        double fR(double x) const { return -fR0 * std::pow(R0 / R(x), n + 1.0); }
        /// f_RR in units of c^2 / (3 H0^2)
        double fRR(double x) const { return -(n + 1.0) * fR(x) / R(x); }
        /// The scalaron mass squared in units of H0^2 / c^2
        double mass2(double x) const { return (1.0 + fR(x)) / fRR(x); }
        /// The Compton wavelength 1/m in units of c / H0
        double compton_wavelength(double x) const { return 1.0 / std::sqrt(mass2(x)); }

        double mu(double x, double koverH0) const override {
            double m, s;
            values(x, koverH0, m, s);
            return m;
        }
        double Sigma(double x, double /*koverH0*/) const override { return 1.0 / (1.0 + fR(x)); }

        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Potential;
            sc.threshold = [this](double x) { return -1.5 * fR(x); };
            sc.factor = [this](double x, double PhiN, double efficiency) {
                const double f = std::abs(-1.5 * fR(x) / PhiN) * efficiency;
                return f > 1.0 ? 1.0 : f;
            };
            sc.description = "chameleon: min(1, Phi_crit / |Phi_N|), Phi_crit = 3/2 |f_R(a)|";
            return sc;
        }

        double get_fR0() const { return fR0; }
        double get_n() const { return n; }

      private:
        double fR0, n, R0 = 0.0;
        const BACKGROUND::ComponentBackground * cbg;

        void values(double x, double koverH0, double & mu, double & Sigma) const {
            const double Rx = R(x), f_R = -fR0 * std::pow(R0 / Rx, n + 1.0);
            const double f_RR = -(n + 1.0) * f_R / Rx; // c^2 / (3 H0^2)
            const double ka = koverH0 * std::exp(-x);
            const double Q = f_RR * ka * ka / (3.0 * (1.0 + f_R));
            Sigma = 1.0 / (1.0 + f_R);
            mu = (1.0 + 4.0 * Q) / (1.0 + 3.0 * Q) * Sigma;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
