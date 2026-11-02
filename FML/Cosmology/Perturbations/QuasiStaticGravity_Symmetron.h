#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_SYMMETRON_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_SYMMETRON_H

// =====================================================================================================================
// The symmetron (Hinterbichler & Khoury 2010) in the quasi-static limit (QuasiStaticGravity.h), with the parameters of
// Winther et al. 2012: a_ssb (symmetry breaking), beta (coupling) and L (the range of the fifth force today, in
// Mpc/h; mu^2 = 1/(2 L^2) with mu the mass of the potential). With the matter density going as a^-3
// (rho/rho_ssb = (a_ssb/a)^3), the background field and mass:
//   phi/phi0 = sqrt(1 - (a_ssb/a)^3) for a > a_ssb, else 0
//   m^2 = mu^2 ((a_ssb/a)^3 - 1) for a < a_ssb,   2 mu^2 (1 - (a_ssb/a)^3) for a > a_ssb
//   mu(k, a) = 1 + 2 beta^2 (phi/phi0)^2 k^2 / (k^2 + a^2 m^2)   (1 before symmetry breaking),   Sigma = 1
// Screening (by the Newtonian potential): factor = min(1, efficiency Phi_crit / |Phi_N|) with
//   Phi_crit = 3 Omega_m (H0 L/c)^2 / a_ssb^3 after symmetry breaking (no fifth force before: factor 0)
// (Omega_m = Omega_m(a = 1) of the background).
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_Symmetron : public QuasiStaticGravity {
      public:
        QuasiStaticGravity_Symmetron(const BACKGROUND::Background & bg, double assb, double beta, double L_mpch)
            : QuasiStaticGravity(bg), assb(assb), beta(beta), L_mpch(L_mpch),
              LH0(L_mpch / FML::UTILS::SI::c_over_H0_mpch), OmegaM(bg.Omega_m(0.0)) {
            if (not(assb > 0.0 and L_mpch > 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_Symmetron] Need a_ssb > 0 and L > 0");
        }
        std::string name() const override { return "symmetron"; }
        std::string info() const override {
            std::ostringstream os;
            os << "symmetron (a_ssb = " << assb << ", beta = " << beta << ", L = " << L_mpch << " Mpc/h)";
            return os.str();
        }
        bool scale_dependent() const override { return true; }

        /// phi/phi0
        double phi(double x) const {
            const double a = std::exp(x);
            return a < assb ? 0.0 : std::sqrt(1.0 - (assb * assb * assb) / (a * a * a));
        }
        /// The mass squared in units of H0^2 / c^2
        double mass2(double x) const {
            const double a = std::exp(x);
            const double mu2H02 = 1.0 / (2.0 * LH0 * LH0);
            const double rhorhossb = (assb * assb * assb) / (a * a * a);
            return a < assb ? mu2H02 * (rhorhossb - 1.0) : 2.0 * mu2H02 * (1.0 - rhorhossb);
        }
        double mu(double x, double koverH0) const override {
            const double a = std::exp(x);
            const double mass2a2 = a * a * mass2(x);
            const double k2 = koverH0 * koverH0;
            const double p = phi(x);
            // No fifth force before symmetry breaking. At a = assb the mass is also zero so k = 0 would give 0 / 0
            if (p == 0.0)
                return 1.0;
            return 1.0 + 2.0 * beta * beta * p * p * k2 / (k2 + mass2a2);
        }

        /// Phi_crit (0 before symmetry breaking)
        double phi_crit(double x) const {
            return std::exp(x) < assb ? 0.0 : 3.0 * OmegaM * LH0 * LH0 / (assb * assb * assb);
        }
        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Potential;
            sc.threshold = [this](double x) { return phi_crit(x); };
            sc.factor = [this](double x, double PhiN, double efficiency) {
                const double pc = phi_crit(x);
                // Before symmetry breaking there is no fifth force (and Phi_crit / Phi_N could be 0 / 0)
                if (pc == 0.0)
                    return 0.0;
                const double f = std::abs(pc / PhiN) * efficiency;
                return f > 1.0 ? 1.0 : f;
            };
            sc.description = "symmetron: min(1, Phi_crit / |Phi_N|), Phi_crit = 3 Omega_m (H0 L)^2 / a_ssb^3";
            return sc;
        }

        double get_assb() const { return assb; }
        double get_beta() const { return beta; }
        double get_L_mpch() const { return L_mpch; }

      private:
        double assb, beta, L_mpch, LH0, OmegaM;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
