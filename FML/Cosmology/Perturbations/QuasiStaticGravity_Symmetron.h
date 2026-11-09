#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_SYMMETRON_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_SYMMETRON_H

// =====================================================================================================================
// The symmetron (Hinterbichler & Khoury 2010) in the quasi-static limit (QuasiStaticGravity.h), with the parameters of
// Winther et al. 2012: a_ssb (symmetry breaking), beta (coupling) and L (the range of the fifth force in vacuum, in
// Mpc/h; xi = H0 L / c), generalised with the powers (N, M) of Brax, Davis, Li, Winther & Zhao 2012 (arXiv:1206.3568,
// Sec. III A; even integers, M > N >= 2; (N, M) = (2, 4) is the original symmetron):
//   V(phi) = H0^2 phi_*^2 / (xi^2 (M - N)) [ -(phi/phi_*)^N / N + (phi/phi_*)^M / M ],   beta(phi) = beta (phi/phi_*)^(N-1)
//   phi_* / M_pl = 3 (M - N) Omega_m beta xi^2 / a_ssb^3   (phi_* the vacuum value; the mass there is H0 / xi = c / L)
// With the matter density going as a^-3 (rho/rho_ssb = (a_ssb/a)^3), the background field, coupling and mass:
//   chi = phi/phi_* = (1 - (a_ssb/a)^3)^(1/(M-N)) for a > a_ssb, else 0;   beta(a) = beta chi^(N-1)
//   m^2 = (H0 L)^-2 chi^(N-2) (1 - (a_ssb/a)^3) for a > a_ssb;   ((a_ssb/a)^3 - 1) / ((M - N) (H0 L)^2) for a < a_ssb
//   (N = 2; 0 for N > 2)
//   mu(k, a) = 1 + 2 beta(a)^2 k^2 / (k^2 + a^2 m^2)   (1 before symmetry breaking),   Sigma = 1
// (For (2, 4): phi/phi0 = sqrt(1 - (a_ssb/a)^3), m^2 = 2 mu^2 (1 - (a_ssb/a)^3), mu^2 = 1/(2 L^2).)
// Screening (by the Newtonian potential): factor = min(1, efficiency Phi_crit / |Phi_N|) with
//   Phi_crit = phi(a) / (2 beta(a) M_pl) = (3/2) (M - N) Omega_m (H0 L/c)^2 chi^(2-N) / a_ssb^3 after symmetry breaking
//   (3 Omega_m (H0 L/c)^2 / a_ssb^3 for (2, 4); no fifth force before: factor 0)
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
        QuasiStaticGravity_Symmetron(const BACKGROUND::Background & bg,
                                     double assb,
                                     double beta,
                                     double L_mpch,
                                     double N = 2.0,
                                     double M = 4.0)
            : QuasiStaticGravity(bg), assb(assb), beta(beta), L_mpch(L_mpch), N(N), M(M),
              LH0(L_mpch / FML::UTILS::SI::c_over_H0_mpch), OmegaM(bg.Omega_m(0.0)) {
            if (not(assb > 0.0 and L_mpch > 0.0))
                throw std::invalid_argument("[QuasiStaticGravity_Symmetron] Need a_ssb > 0 and L > 0");
            if (not valid_powers(N, M))
                throw std::invalid_argument("[QuasiStaticGravity_Symmetron] Need even integers M > N >= 2");
        }
        /// N and M even integers with M > N >= 2
        static bool valid_powers(double N, double M) {
            auto even = [](double x) { return x == std::round(x) and std::fmod(x, 2.0) == 0.0; };
            return even(N) and even(M) and N >= 2.0 and M > N;
        }
        std::string name() const override { return "symmetron"; }
        std::string info() const override {
            std::ostringstream os;
            os << "symmetron (a_ssb = " << assb << ", beta = " << beta << ", L = " << L_mpch << " Mpc/h, N = " << N
               << ", M = " << M << ")";
            return os.str();
        }
        bool scale_dependent() const override { return true; }

        /// chi = phi/phi_* (phi/phi0 for (2, 4))
        double phi(double x) const {
            const double a = std::exp(x);
            return a < assb ? 0.0 : std::pow(1.0 - (assb * assb * assb) / (a * a * a), 1.0 / (M - N));
        }
        /// The coupling in the background beta chi^(N-1)
        double beta_of_a(double x) const { return beta * std::pow(phi(x), N - 1.0); }
        /// The mass squared in units of H0^2 / c^2
        double mass2(double x) const {
            const double a = std::exp(x);
            const double rhorhossb = (assb * assb * assb) / (a * a * a);
            if (a < assb)
                return N == 2.0 ? (rhorhossb - 1.0) / ((M - N) * LH0 * LH0) : 0.0;
            return std::pow(phi(x), N - 2.0) * (1.0 - rhorhossb) / (LH0 * LH0);
        }
        double mu(double x, double koverH0) const override {
            const double a = std::exp(x);
            const double mass2a2 = a * a * mass2(x);
            const double k2 = koverH0 * koverH0;
            const double b = beta_of_a(x);
            // No fifth force before symmetry breaking. At a = assb the mass is also zero so k = 0 would give 0 / 0
            if (b == 0.0)
                return 1.0;
            return 1.0 + 2.0 * b * b * k2 / (k2 + mass2a2);
        }

        /// Phi_crit (0 before symmetry breaking; at a = assb it is infinite for N > 2: no screening)
        double phi_crit(double x) const {
            if (std::exp(x) < assb)
                return 0.0;
            const double chi = phi(x);
            const double chifac = N == 2.0 ? 1.0 : (chi > 0.0 ? std::pow(chi, 2.0 - N) : HUGE_VAL);
            return 1.5 * (M - N) * OmegaM * LH0 * LH0 / (assb * assb * assb) * chifac;
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
            sc.description = "symmetron: min(1, Phi_crit / |Phi_N|), Phi_crit = (3/2) (M - N) Omega_m (H0 L)^2 "
                             "chi^(2-N) / a_ssb^3";
            return sc;
        }

        double get_assb() const { return assb; }
        double get_beta() const { return beta; }
        double get_L_mpch() const { return L_mpch; }
        double get_N() const { return N; }
        double get_M() const { return M; }

      private:
        double assb, beta, L_mpch, N, M, LH0, OmegaM;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
