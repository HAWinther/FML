#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_DGP_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBEDFIELDEQUATION_DGP_H

// =====================================================================================================================
// DGP braneworld gravity (the background: Background/FieldEquation_DGP.h, whose Omega_rc, branch eps and LCDM-background
// option are read here) in the quasi-static approximation (Lue, Scoccimarro & Starkman 2004; Koyama & Maartens 2006),
// the brane-bending mode strengthens (nDGP) or weakens
// (sDGP) the force on matter and does not change the lensing potential,
//   mu = 1 + 1/(3 beta),  beta = 1 - 2 eps (H r_c/c)(1 + Hdot/(3 H^2)) = 1 - eps (E + E'/3) / sqrt(Omega_rc),  Sigma = 1
// with the mu-Sigma equations (PerturbedFieldEquation_MuSigma.h); d(2 Sigma - mu)/dx = beta'/(3 beta^2) with
// beta' = -eps (E' + E''/3) / sqrt(Omega_rc). E, E', E'' from the background's DGP Friedmann equation
// E = eps sqrt(Omega_rc) + s, s = sqrt(Omega_rc + rho): E' = rho'/(2 s), E'' = rho''/(2 s) - rho'^2/(4 s^3) (rho: all
// species, rho'' = ComponentBackground::d2rho_total), or with the LCDM background from E^2 = rho + Omega_k e^{-2x}.
// The quasi-static approximation is used on all scales (as in most Boltzmann-code implementations).
// =====================================================================================================================

#include <FML/Cosmology/Background/FieldEquation_DGP.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/PerturbedFieldEquation_MuSigma.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class PerturbedFieldEquation_DGP : public PerturbedFieldEquation_MuSigma {
      public:
        PerturbedFieldEquation_DGP(const BACKGROUND::DGPFieldEquations & dgp, const PerturbationContext & ctx)
            : PerturbedFieldEquation_MuSigma(ctx), Omega_rc(dgp.get_Omega_rc()), sq(std::sqrt(Omega_rc)),
              branch(dgp.get_branch()), lcdm_bg(dgp.get_lcdm_background()) {}
        std::string name() const override { return branch > 0 ? "sDGP" : "nDGP"; }
        std::string info() const override {
            std::ostringstream os;
            os << name() << " (Omega_rc = " << Omega_rc << ", H0 r_c / c = " << 0.5 / sq
               << (lcdm_bg ? ", LCDM background" : "") << ")";
            return os.str();
        }

        /// E and its first two x-derivatives
        void E_derivs(double x, double & E, double & dE, double & ddE) const {
            const auto & bg = ctx.background;
            if (lcdm_bg) {
                const double e2 = bg.rho_total(x) + bg.rho_k(x), de2 = bg.drho_total(x) + bg.drho_k(x);
                const double dde2 = bg.d2rho_total(x) + 4.0 * bg.rho_k(x);
                E = std::sqrt(e2);
                dE = de2 / (2.0 * E);
                ddE = dde2 / (2.0 * E) - de2 * de2 / (4.0 * E * E * E);
                return;
            }
            const double rho = bg.rho_total(x), drho = bg.drho_total(x), ddrho = bg.d2rho_total(x);
            const double s = std::sqrt(Omega_rc + rho);
            E = branch * sq + s;
            dE = drho / (2.0 * s);
            ddE = ddrho / (2.0 * s) - drho * drho / (4.0 * s * s * s);
        }
        double beta(double x) const {
            double E, dE, ddE;
            E_derivs(x, E, dE, ddE);
            return 1.0 - branch * (E + dE / 3.0) / sq;
        }
        /// mu = 1 + 1/(3 beta), Sigma = 1, so 2 Sigma - mu = 1 - 1/(3 beta) and its derivative is beta'/(3 beta^2)
        MuSigma mu_sigma(double x, double /*k*/) const override {
            double E, dE, ddE;
            E_derivs(x, E, dE, ddE);
            const double b = 1.0 - branch * (E + dE / 3.0) / sq;
            const double db = -branch * (dE + ddE / 3.0) / sq;
            if (b == 0.0)
                throw std::runtime_error(name() + ": beta = 0");
            MuSigma m;
            m.mu = 1.0 + 1.0 / (3.0 * b);
            m.Sigma = 1.0;
            m.dg = db / (3.0 * b * b);
            return m;
        }

      private:
        double Omega_rc, sq;
        int branch;
        bool lcdm_bg;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
