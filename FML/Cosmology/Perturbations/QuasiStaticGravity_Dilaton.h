#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_DILATON_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_DILATON_H

// =====================================================================================================================
// The environment-dependent dilaton (Brax, van de Bruck, Davis & Shaw 2010) in the quasi-static limit
// (QuasiStaticGravity.h), with the generalised parametrisation of Brax, Davis, Li, Winther & Zhao 2012
// (arXiv:1206.3568, Sec. III B): A(phi) = 1 + A2 phi^2 / (2 M_pl^2) (phi_* = 0) and the field at the minimum of its
// effective potential, with
//   m(a)    = m0 a^-r,                     xi = H0 / m0 (the range today is 2998 xi Mpc/h)
//   beta(a) = A2 phi(a) / M_pl = beta0 exp[ s (a^p - 1) / p ],   p = 2r - 3,   s = 9 Omega_m A2 xi^2
// (p -> 0, r = 3/2: beta0 a^s, the original dilaton). The paper needs r < 3/2; r = 3/2 is the limit; r > 3/2 is not
// allowed (the field cannot reach phi_* in dense regions and the potential, eq. 66, is not defined there).
//   mu(k, a) = 1 + 2 beta(a)^2 k^2 / (k^2 + a^2 m(a)^2),   Sigma = 1
// Screening (by the Newtonian potential, as for the chameleon): factor = min(1, efficiency Phi_crit / |Phi_N|) with
//   Phi_crit = (phi(a) - phi_c) / (2 beta(a) M_pl) = 1 / (2 A2)
// (phi_c = phi_* = 0 inside the screened object, phi(a)/M_pl = beta(a)/A2: independent of the time and of beta0)
// The field equation (for the multigrid solver, MultigridSolver/DilatonSolver.h), with u = ln(phi / phi(a)):
//   D^2 e^u = 3 A2 Omega_m (a H0 / c)^2 e^u [ (1 + delta) a^-3 - (1 + p y)^(-3/p) ],   y = (a^p - 1)/p + u / s
// (D the comoving gradient; (1 + p y)^(-3/p) -> exp(-3 y) for p -> 0). Omega_m = Omega_m(a = 1) of the background.
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_Dilaton : public QuasiStaticGravity {
      public:
        /// A2 > 0, beta0 (the coupling today), r <= 3/2 (m = m0 a^-r) and xi = H0/m0 > 0
        QuasiStaticGravity_Dilaton(const BACKGROUND::Background & bg, double A2, double beta0, double r, double xi)
            : QuasiStaticGravity(bg), A2(A2), beta0(beta0), r(r), xi(xi), OmegaM(bg.Omega_m(0.0)) {
            if (not(A2 > 0.0 and xi > 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_Dilaton] Need A2 > 0 and xi > 0");
            if (not(r <= 1.5))
                throw std::invalid_argument("[QuasiStaticGravity_Dilaton] Need r <= 3/2 (the potential is not "
                                            "defined for r > 3/2)");
            s = 9.0 * OmegaM * A2 * xi * xi;
            p = 2.0 * r - 3.0;
        }
        std::string name() const override { return "dilaton"; }
        std::string info() const override {
            std::ostringstream os;
            os << "dilaton (A2 = " << A2 << ", beta0 = " << beta0 << ", r = " << r << ", xi = " << xi
               << ", s = " << s << ")";
            return os.str();
        }
        bool scale_dependent() const override { return true; }

        /// (a^p - 1)/p as a function of x = ln a (ln a for p = 0)
        static double expm1_over_p(double p, double x) {
            return std::abs(p) < 1e-12 ? x : std::expm1(p * x) / p;
        }
        /// ln(1 + p y) / p (y for p = 0)
        static double log1p_over_p(double p, double y) {
            return std::abs(p) < 1e-12 ? y : std::log1p(p * y) / p;
        }

        /// The coupling beta(a) = beta0 exp[s (a^p - 1)/p]
        double beta(double x) const { return beta0 * std::exp(s * expm1_over_p(p, x)); }
        /// The background field phi(a) / M_pl = beta(a) / A2
        double phi(double x) const { return beta(x) / A2; }
        /// The mass squared in units of H0^2 / c^2: a^(-2r) / xi^2
        double mass2(double x) const { return std::exp(-2.0 * r * x) / (xi * xi); }
        double mu(double x, double koverH0) const override {
            const double a = std::exp(x);
            const double mass2a2 = a * a * mass2(x);
            const double k2 = koverH0 * koverH0;
            const double b = beta(x);
            return 1.0 + 2.0 * b * b * k2 / (k2 + mass2a2);
        }

        /// Phi_crit = 1 / (2 A2)
        double phi_crit() const { return 0.5 / A2; }
        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Potential;
            sc.threshold = [this](double) { return phi_crit(); };
            sc.factor = [this](double, double PhiN, double efficiency) {
                const double f = std::abs(phi_crit() / PhiN) * efficiency;
                return f > 1.0 ? 1.0 : f;
            };
            sc.description = "dilaton: min(1, Phi_crit / |Phi_N|), Phi_crit = phi(a) / (2 beta(a) M_pl) = 1 / (2 A2)";
            return sc;
        }

        double get_A2() const { return A2; }
        double get_beta0() const { return beta0; }
        double get_r() const { return r; }
        double get_xi() const { return xi; }
        /// s = 9 Omega_m A2 xi^2
        double get_s() const { return s; }
        /// The range of the fifth force today in Mpc/h (c / m0 = xi c / H0)
        double get_range_today_mpch() const { return xi * FML::UTILS::SI::c_over_H0_mpch; }

      private:
        double A2, beta0, r, xi, OmegaM, s{}, p{};
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
