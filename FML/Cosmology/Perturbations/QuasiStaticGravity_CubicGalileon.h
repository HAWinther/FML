#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_CUBICGALILEON_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_CUBICGALILEON_H

// =====================================================================================================================
// The cubic Galileon on its tracker (Background/BackgroundModel_CubicGalileon.h: S = int sqrt(-g) [R/2 + c2 X - g3 X
// box phi] + S_m, c2 = -1, H dphi/dt = xi = sqrt(6 Omega_phi0) H0^2, g3 = 1/(3 xi); 8 pi G = 1) in the quasi-static
// limit (QuasiStaticGravity.h; Barreira et al. 2013 for the N-body equations). With phi = phi(t) + pi, the scalar
// field equation, c2 box phi - g3 [(box phi)^2 - (nabla nabla phi)^2 - R_mu nu nabla^mu phi nabla^nu phi] = 0, to
// second order in the spatial derivatives of pi (comoving nabla) and the Einstein equations give
//   nabla^2 Psi = a^2 rho delta / 2 - (g3 dphi^2 / 2) nabla^2 pi,   Phi = Psi (no slip)
//   [c2 + 2 g3 (ddphi + 2 H dphi)] nabla^2 pi - (g3/a^2) [(nabla^2 pi)^2 - (nabla_i nabla_j pi)^2] + g3 dphi^2 nabla^2
//   Psi = 0
// On the tracker, with chi = -g3 dphi^2 pi (Psi = Psi_N + chi/2), this is the equation of the brane-bending mode of
// DGP (QuasiStaticGravity_DGP.h, DGPSolver.h) with a time-dependent cross-over scale:
//   nabla^2 chi + (r_c^2 / (3 beta a^2)) [(nabla^2 chi)^2 - (nabla_i nabla_j chi)^2] = a^2 rho delta / (3 beta)
//   beta = (1 - 2 eps - Omega_phi(a)) / (3 Omega_phi(a)),   (r_c H0/c)^2 = E^2 / (2 Omega_phi(a) Omega_phi0)
// (eps = dlnH/dlna; on the tracker Omega_phi(a) = Omega_phi0 / E^4, so r_c H0/c = E^3 / (sqrt(2) Omega_phi0)). Hence
//   mu = Sigma = 1 + 1/(3 beta) = 1 + Omega_phi(a) / (1 - 2 eps - Omega_phi(a))
// (Sigma = mu: the Galileon enters Phi and Psi alike; mu = 1.77 today for Omega_phi0 = 0.68), the 2LPT source of DGP
// with r_c(a), and for a spherical top-hat the fifth force reduced by 2 (sqrt(1 + e) - 1) / e with
//   e = 8 r_c^2 rho_m(a) delta / (9 beta^2) = 4 Omega_m(a) delta / (1 - 2 eps - Omega_phi(a))^2   (on the tracker)
// (the screening approximation here, delta = rho/rho_mean - 1: the force is stronger in underdense regions, up to
// twice the linear fifth force at e = -1; there is no real solution for e < -1, delta < -0.6 today, where e = -1 is
// used, as the multigrid solver sets the discriminant to zero). mu agrees with the
// quasi-static limit of the Einstein-Boltzmann solver's equations (PerturbedFieldEquation_CubicGalileon.h; test_galileon)
// to 1e-15 and its full solution to 0.1% for k > 0.1/Mpc.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_CubicGalileon.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_CubicGalileon : public QuasiStaticGravity {
      public:
        /// bg: a background with a CubicGalileonSpecies (the cubic_galileon model)
        explicit QuasiStaticGravity_CubicGalileon(const BACKGROUND::Background & bg) : QuasiStaticGravity(bg) {
            if (const auto * cb = dynamic_cast<const BACKGROUND::ComponentBackground *>(&bg))
                for (const auto & s : cb->species())
                    if (const auto * g = dynamic_cast<const BACKGROUND::CubicGalileonSpecies *>(s.get()))
                        galileon = g;
            if (not galileon)
                throw std::invalid_argument("[QuasiStaticGravity_CubicGalileon] The background has no cubic Galileon "
                                            "(the cubic_galileon model)");
        }
        std::string name() const override { return "cubic Galileon"; }
        std::string info() const override {
            std::ostringstream os;
            os << "cubic Galileon (tracker, Omega_phi = " << galileon->Omega_phi()
               << "; quasi-static, mu(a = 1) = " << mu(0.0, 0.0) << ")";
            return os.str();
        }

        /// The Galileon's density parameter at x
        double Omega_phi(double x) const {
            const double E = bg.E(x);
            return galileon->rho(x) / (E * E);
        }
        /// beta(x) = (1 - 2 eps - Omega_phi) / (3 Omega_phi) of the DGP form (infinite without the Galileon)
        double beta(double x) const {
            const double O = Omega_phi(x);
            return O > 0.0 ? (1.0 - 2.0 * bg.dlogEdx(x) - O) / (3.0 * O) : HUGE_VAL;
        }
        /// (r_c H0/c)^2 = E^2 / (2 Omega_phi(x) Omega_phi0) of the DGP form
        double rcH0_squared(double x) const {
            const double E = bg.E(x);
            return E * E / (2.0 * Omega_phi(x) * galileon->Omega_phi());
        }
        double rcH0(double x) const { return std::sqrt(rcH0_squared(x)); }

        double mu(double x, double /*koverH0*/) const override {
            const double O = Omega_phi(x);
            return 1.0 + O / nonzero(1.0 - 2.0 * bg.dlogEdx(x) - O);
        }
        double Sigma(double x, double koverH0) const override { return mu(x, koverH0); }
        /// 1 - 2 r_c^2 rho_m(a) / (9 beta^3 mu) (DGP's, with r_c(a)) = 1 - 3 Omega_m(a) t Omega_phi(a) / (D^3 mu),
        /// D = 1 - 2 eps - Omega_phi(a), t = E^4 Omega_phi(a) / Omega_phi0 (1 on the tracker; finite for Omega_phi ->
        /// 0)
        double second_order_source(double x, double koverH0) const override {
            const double O = Omega_phi(x), D = nonzero(1.0 - 2.0 * bg.dlogEdx(x) - O);
            return 1.0 - 3.0 * bg.Omega_m(x) * tracker(x) * O / (D * D * D * mu(x, koverH0));
        }

        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Density;
            sc.factor = [this](double x, double rho, double efficiency) {
                const double D = nonzero(1.0 - 2.0 * bg.dlogEdx(x) - Omega_phi(x));
                const double e = std::max(4.0 * bg.Omega_m(x) * tracker(x) / (D * D) * efficiency * (rho - 1.0), -1.0);
                return std::fabs(e) < 1e-5 ? 1.0 : 2.0 * (std::sqrt(1.0 + e) - 1.0) / e;
            };
            sc.description = "Vainshtein (spherical top-hat): 2 (sqrt(1 + e) - 1) / e, e = 4 Omega_m(a) delta / (1 - "
                             "2 dlnH/dlna - Omega_phi(a))^2 >= -1";
            return sc;
        }

      private:
        const BACKGROUND::CubicGalileonSpecies * galileon{nullptr};
        /// E^4 Omega_phi(a) / Omega_phi0 (= 8 r_c^2 rho_m / (9 beta^2) / (4 Omega_m(a) / D^2); 1 on the tracker)
        double tracker(double x) const {
            const double E = bg.E(x);
            return E * E * galileon->rho(x) / galileon->Omega_phi();
        }
        static double nonzero(double b) {
            if (b == 0.0)
                throw std::runtime_error("[QuasiStaticGravity_CubicGalileon] 1 - 2 dlnH/dlna - Omega_phi = 0");
            return b;
        }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
