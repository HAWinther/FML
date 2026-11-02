#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_DGP_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_DGP_H

// =====================================================================================================================
// DGP braneworld gravity in the quasi-static approximation (Lue, Scoccimarro & Starkman 2004; Koyama & Maartens 2006;
// QuasiStaticGravity.h): the brane-bending mode strengthens (normal branch, nDGP: branch = -1) or weakens
// (self-accelerating branch, sDGP: branch = +1) the force on matter and does not change the lensing potential,
//   mu = 1 + 1/(3 beta),   beta = 1 - 2 branch (H r_c/c)(1 + dlnH/dlna / 3),   Sigma = 1
// (nDGP: beta = 1 + 2 r_c H0/c E (1 + dlnE/dx / 3)) with H(a) of the background, whichever it is (the DGP Friedmann
// equation, LCDM, ...). r_c H0/c = 1/(2 sqrt(Omega_rc)).
// The nonlinear term of the field equation, (r_c^2/(3 beta a^2)) [(nabla^2 phi)^2 - (nabla_i nabla_j phi)^2], gives the
// second-order source
//   second_order_source = 1 - 2 (r_c H0/c)^2 rho_m(a) / (9 beta^3 mu),   rho_m = Omega_m(a) E^2(a)  (in rho_c0)
// d(2 Sigma - mu)/dx = beta'/(3 beta^2) with beta' = -2 branch (r_c H0/c)(E' + E''/3); E'' = E (dlnE/dx' + dlnE/dx^2)
// with d(dlnE/dx)/dx by a central difference (step 1e-4).
// Screening (Vainshtein, by the density): for a spherical top-hat the field equation gives the fifth force reduced by
//   factor = 2 (sqrt(1 + e) - 1) / e,   e = 8 (r_c H0/c)^2 rho_m(a) / (9 beta^2) * delta
// (rho_m(a) = Omega_m(a) E^2 ~ Omega_m a^-3). The screening approximation here is not that: it uses
//   e = 8 (r_c H0/c)^2 Omega_m / (9 beta^2) * efficiency * value,   value = rho/rho_mean = 1 + delta
// with Omega_m today in place of rho_m(a) (no 1/a^3) and 1 + delta in place of delta (it does not tend to 1, no
// screening, for delta -> 0: 0.990 at a = 1 for r_c H0/c = 1), because that gives results closer to the exact
// (multigrid) solution. In a test (100 Mpc/h, 64^3, 20 steps; P(k) vs the exact solution, rms for k < 1.5 h/Mpc):
//   z = 0: this form 0.5%, delta with 1/a^3 (the spherical solution) 1.2%, 1 + delta with 1/a^3 2.0% (screens the
//          mean density at early times: 2% low on all scales), no screening 1.9%
//   z = 1: this form 1.3%, 0.6%, 1.4%, 1.6%
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_DGP : public QuasiStaticGravity {
      public:
        /// rcH0 = r_c H0 / c > 0, branch = -1 (nDGP) or +1 (sDGP)
        QuasiStaticGravity_DGP(const BACKGROUND::Background & bg, double rcH0, int branch = -1)
            : QuasiStaticGravity(bg), rcH0(rcH0), branch(branch), OmegaM0(bg.Omega_m(0.0)) {
            if (not(rcH0 > 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_DGP] Need r_c H0 / c > 0");
            if (branch != 1 and branch != -1)
                throw std::invalid_argument("[QuasiStaticGravity_DGP] branch must be -1 (nDGP) or +1 (sDGP)");
        }
        std::string name() const override { return branch > 0 ? "sDGP" : "nDGP"; }
        std::string info() const override {
            std::ostringstream os;
            os << name() << " (r_c H0/c = " << rcH0 << ", Omega_rc = " << 0.25 / (rcH0 * rcH0) << "; quasi-static)";
            return os.str();
        }

        double beta(double x) const {
            return 1.0 - 2.0 * branch * rcH0 * bg.E(x) * (1.0 + bg.dlogEdx(x) / 3.0);
        }
        double mu(double x, double /*koverH0*/) const override { return 1.0 + 1.0 / (3.0 * nonzero(beta(x))); }
        double second_order_source(double x, double /*koverH0*/) const override {
            const double b = nonzero(beta(x)), E = bg.E(x);
            const double rho_m = bg.Omega_m(x) * E * E;
            return 1.0 - 2.0 * rcH0 * rcH0 * rho_m / (9.0 * b * b * b * (1.0 + 1.0 / (3.0 * b)));
        }
        MuSigma mu_sigma(double x, double /*koverH0*/) const override {
            const double h = 1e-4;
            const double E = bg.E(x), dlE = bg.dlogEdx(x);
            const double ddlE = (bg.dlogEdx(x + h) - bg.dlogEdx(x - h)) / (2.0 * h);
            const double dE = E * dlE, ddE = E * (ddlE + dlE * dlE);
            const double b = nonzero(1.0 - 2.0 * branch * rcH0 * (E + dE / 3.0));
            const double db = -2.0 * branch * rcH0 * (dE + ddE / 3.0);
            MuSigma m;
            m.mu = 1.0 + 1.0 / (3.0 * b);
            m.Sigma = 1.0;
            m.dg = db / (3.0 * b * b);
            return m;
        }

        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Density;
            sc.factor = [this](double x, double rho, double efficiency) {
                const double b = nonzero(beta(x));
                const double e = 8.0 * rcH0 * rcH0 * OmegaM0 / (9.0 * b * b) * efficiency * rho;
                return e < 1e-5 ? 1.0 : 2.0 * (std::sqrt(1.0 + e) - 1.0) / e;
            };
            sc.description =
                "Vainshtein (approximation): 2 (sqrt(1 + e) - 1) / e, e = 8 (r_c H0)^2 Omega_m / (9 beta^2) rho/rho_mean";
            return sc;
        }

        double get_rcH0() const { return rcH0; }
        int get_branch() const { return branch; }

      private:
        double rcH0;
        int branch;
        double OmegaM0;
        double nonzero(double b) const {
            if (b == 0.0)
                throw std::runtime_error("[QuasiStaticGravity_DGP] beta = 0");
            return b;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
