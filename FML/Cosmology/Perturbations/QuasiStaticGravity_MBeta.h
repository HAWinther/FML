#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_MBETA_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_MBETA_H

// =====================================================================================================================
// The (m(a), beta(a)) parametrisation of chameleon-like theories (Brax, Davis, Li & Winther 2012) in the quasi-static
// limit (QuasiStaticGravity.h), with power laws
//   beta(a) = beta0 a^n_beta,   m(a) = m0 H0/c a^n_m
//   mu(k, a) = 1 + 2 beta^2 k^2 / (k^2 + a^2 m^2),   Sigma = 1
// Screening (by the Newtonian potential): factor = min(1, efficiency Phi_crit / |Phi_N|), Phi_crit = phi(a)/(2 beta(a)
// M_pl) with the background field (matter going as Omega_m a^-3, Omega_m = Omega_m(a = 1) of the background)
//   phi(a)/M_pl = 9 Omega_m int_0^a beta(a') / (a'^3 (m/H0)^2) dln a'
// The integrand goes as a^p, p = n_beta - 3 - 2 n_m, which must be > 0 for the integral to converge (else
// screening() throws): from 0 to a = 1e-3 it is integrand(1e-3) / p (exact for the power laws), then the trapezoidal
// rule in ln a (step ln(1000)/9999) up to a = 10, splined in a.
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_MBeta : public QuasiStaticGravity {
      public:
        QuasiStaticGravity_MBeta(
            const BACKGROUND::Background & bg, double beta0, double n_beta, double m0, double n_m, double a_max = 10.0)
            : QuasiStaticGravity(bg), beta0(beta0), n_beta(n_beta), m0(m0), n_m(n_m), OmegaM(bg.Omega_m(0.0)) {
            p = n_beta - 3.0 - 2.0 * n_m;
            if (p > 0.0) {
                auto integrand = [&](double a) {
                    return 9.0 * OmegaM * beta_of_a(a) / (a * a * a * mass2_of_a(a));
                };
                const double amin = 0.001;
                const double dloga = std::log(1.0 / amin) / 9999.0;
                const size_t npts = size_t(std::lround(std::log(a_max / amin) / dloga)) + 1;
                std::vector<double> a_array(npts, amin), loga_array(npts, std::log(amin)), phi_array(npts, 0.0);
                phi_array[0] = integrand(amin) / p;
                for (size_t i = 1; i < npts; i++) {
                    loga_array[i] = std::log(amin) + std::log(1.0 / amin) * i / 9999.0;
                    a_array[i] = std::exp(loga_array[i]);
                    phi_array[i] = phi_array[i - 1] + integrand(a_array[i]) * (loga_array[i] - loga_array[i - 1]);
                }
                phi_spline = std::make_shared<const FML::INTERPOLATION::SPLINE::Spline>(
                    a_array, phi_array, "phi(a)/Mpl spline for m(a),beta(a) models");
            }
        }
        std::string name() const override { return "(m(a), beta(a))"; }
        std::string info() const override {
            std::ostringstream os;
            os << "(m(a), beta(a)): beta = " << beta0 << " a^" << n_beta << ", m = " << m0 << " H0 a^" << n_m;
            return os.str();
        }
        bool scale_dependent() const override { return true; }

        double beta_of_a(double a) const { return beta0 * std::pow(a, n_beta); }
        /// (m/H0)^2
        double mass2_of_a(double a) const {
            const double m = m0 * std::pow(a, n_m);
            return m * m;
        }
        /// phi(a)/M_pl (needs n_beta - 3 - 2 n_m > 0)
        double phi_over_Mpl(double a) const {
            if (not phi_spline)
                throw std::runtime_error("[QuasiStaticGravity_MBeta] phi(a) diverges: need n_beta - 3 - 2 n_m > 0");
            return (*phi_spline)(a);
        }

        double mu(double x, double koverH0) const override {
            const double a = std::exp(x);
            const double mass2a2 = a * a * mass2_of_a(a);
            const double k2 = koverH0 * koverH0;
            const double b = beta_of_a(a);
            return 1.0 + 2.0 * b * b * k2 / (k2 + mass2a2);
        }

        Screening screening() const override {
            if (not phi_spline)
                throw std::runtime_error("[QuasiStaticGravity_MBeta] Screening: the integral for phi(a) diverges "
                                         "(need n_beta - 3 - 2 n_m > 0)");
            Screening sc;
            sc.type = ScreeningType::Potential;
            sc.threshold = [this](double x) {
                const double a = std::exp(x);
                return phi_over_Mpl(a) / (2.0 * beta_of_a(a));
            };
            sc.factor = [this](double x, double PhiN, double efficiency) {
                const double a = std::exp(x);
                const double f = std::abs(phi_over_Mpl(a) / (2.0 * beta_of_a(a)) / PhiN) * efficiency;
                return f > 1.0 ? 1.0 : f;
            };
            sc.description = "chameleon: min(1, Phi_crit / |Phi_N|), Phi_crit = phi(a) / (2 beta(a) M_pl)";
            return sc;
        }

      private:
        double beta0, n_beta, m0, n_m, OmegaM, p;
        std::shared_ptr<const FML::INTERPOLATION::SPLINE::Spline> phi_spline;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
