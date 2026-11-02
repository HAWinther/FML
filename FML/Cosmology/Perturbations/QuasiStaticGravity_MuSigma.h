#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_MUSIGMA_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_MUSIGMA_H

// =====================================================================================================================
// The "late-time" mu-Sigma parametrisation of Planck (2015 XIV, 2018 VI) and DES (QuasiStaticGravity.h):
//   mu = 1 + mu0 f(a),   Sigma = 1 + Sigma0 f(a),   f = Omega_DE(a) / Omega_DE(a = 1)
// with Omega_DE the dark-energy component of the background (a cosmological constant, fluid or field). Scale
// independent. d(2 Sigma - mu)/dx is exact on a ComponentBackground (f' = f (rho_DE'/rho_DE - 2 dlnE/dx)), a central
// difference otherwise. Needs Omega_DE(a = 1) > 0 unless mu0 = Sigma0 = 0.
// =====================================================================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_MuSigma : public QuasiStaticGravity {
      public:
        QuasiStaticGravity_MuSigma(const BACKGROUND::Background & bg, double mu0, double Sigma0 = 0.0)
            : QuasiStaticGravity(bg), mu0(mu0), Sigma0(Sigma0),
              OmegaDE0(bg.Omega(BACKGROUND::Component::DarkEnergy, 0.0)),
              cbg(dynamic_cast<const BACKGROUND::ComponentBackground *>(&bg)) {
            if (not(OmegaDE0 > 0.0) and (mu0 != 0.0 or Sigma0 != 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_MuSigma] Needs dark energy (Omega_DE today > 0)");
        }
        std::string name() const override { return "mu-Sigma"; }
        std::string info() const override {
            std::ostringstream os;
            os << "mu-Sigma (Planck/DES: mu0 = " << mu0 << ", Sigma0 = " << Sigma0 << ")";
            return os.str();
        }

        /// Omega_DE(a) / Omega_DE(a = 1)
        double f(double x) const {
            return OmegaDE0 > 0.0 ? bg.Omega(BACKGROUND::Component::DarkEnergy, x) / OmegaDE0 : 0.0;
        }
        double mu(double x, double /*koverH0*/) const override { return 1.0 + mu0 * f(x); }
        double Sigma(double x, double /*koverH0*/) const override { return 1.0 + Sigma0 * f(x); }

        MuSigma mu_sigma(double x, double koverH0) const override {
            if (not cbg or not(OmegaDE0 > 0.0))
                return QuasiStaticGravity::mu_sigma(x, koverH0);
            const double fx = f(x);
            const double df =
                fx * (cbg->drhodx(BACKGROUND::Component::DarkEnergy, x) / cbg->rho(BACKGROUND::Component::DarkEnergy, x) -
                      2.0 * bg.dlogEdx(x));
            MuSigma m;
            m.mu = 1.0 + mu0 * fx;
            m.Sigma = 1.0 + Sigma0 * fx;
            m.dg = (2.0 * Sigma0 - mu0) * df;
            return m;
        }

        double get_mu0() const { return mu0; }
        double get_Sigma0() const { return Sigma0; }

      private:
        double mu0, Sigma0, OmegaDE0;
        const BACKGROUND::ComponentBackground * cbg;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
