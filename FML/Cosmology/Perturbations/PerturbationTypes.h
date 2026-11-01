#ifndef FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONTYPES_H
#define FML_COSMOLOGY_PERTURBATIONS_PERTURBATIONTYPES_H

// =====================================================================================================================
// The data the linear perturbations of a model exchange with the Einstein-Boltzmann solver, and the context the
// perturbation objects are built in.
//
// Conventions (those of the Einstein-Boltzmann solver, Cosmology/Perturbations/Perturbations.h, so that it can switch
// to these types): Newtonian gauge, ds^2 = -(1 + 2 Psi) dt^2 + a^2 (1 + 2 Phi) dx^2 (Dodelson / Callin 2006), time
// variable x = ln a, SI units: k in 1/m, Hp = a H in 1/s, conformal time eta in m. Densities appear as they enter the
// Einstein equations, rho_i / rho_c0 times H0^2 a^2 (units 1/s^2).
//
// Spatial curvature K = -Omega_k H0^2 / c^2 [1/m^2] (K < 0 open, K > 0 closed). k is the Laplacian eigenvalue,
// -nabla^2 -> k^2, and the free-streaming couplings between the multipoles l-1 and l carry the factor
//   s_l = sqrt(1 - (l^2 - 1) K / k^2)                                  (0 where negative; 1 when flat)
// in the symmetric convention of all hierarchies (as CLASS); the physical shear of a hierarchy is (l = 2 moment)/s_2.
//
// PerturbationContext is what the species and field equations take from the solver: the background (a
// BACKGROUND::ComponentBackground, which must outlive the context and everything built from it), the settings of the
// species (PerturbationSettings, the solver's defaults), K and s_ell, and mode_at_x(x, k), which fills a ModeAtX from
// the background (everything but the recombination quantities tau', tau'', R, which the solver's core adds).
// =====================================================================================================================

#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// Background quantities at one x for one mode k, computed once per x and shared by all species
    struct ModeAtX {
        double x = std::numeric_limits<double>::quiet_NaN();
        double k = 0.0;          ///< [1/m]
        double s2 = 1.0;         ///< curvature factor s_2 of this mode (PerturbationContext::s_ell)
        double curv = 1.0;       ///< (k^2 - 3K) / k^2
        double a, Hp, dHpH, ckH; ///< a, Hp [1/s], Hp'/Hp, ck/Hp
        double dtau, ddtau, R;   ///< tau', tau'', R = 4 rho_gamma / (3 rho_b) (from the recombination history)
        double eta;              ///< conformal time [m]
        double cotK_H;           ///< c cot_K(eta) / Hp: free-streaming truncation rate (= c / (Hp eta) when flat)
        double H0a2;             ///< H0^2 a^2 [1/s^2]
        double H0a2_rho_b, H0a2_rho_gamma;
    };

    /// Contributions to the Einstein equations, all times H0^2 a^2 / rho_c0 (summed over species):
    ///   drho  = delta rho
    ///   dp    = delta p (enters only theories that couple to the trace, e.g. Brans-Dicke)
    ///   flux  = (rho + p) theta / (ck) = -(rho + p) v         (v as for CDM: delta_cdm' = (ck/Hp) v - 3 Phi')
    ///   shear = (3/2) (rho + p) sigma                         (= 4 rho Theta_2 for photons)
    struct Moments {
        double drho = 0.0, dp = 0.0, flux = 0.0, shear = 0.0;
        Moments & operator+=(const Moments & m) {
            drho += m.drho;
            dp += m.dp;
            flux += m.flux;
            shear += m.shear;
            return *this;
        }
    };

    /// The metric at one x: Phi (evolved), Psi and Phi' (from the field equations, PerturbedFieldEquation.h)
    struct MetricAtX {
        double Phi = 0.0, Psi = 0.0, dPhi = 0.0;
    };

    /// Explicit initial values of the standard species (isocurvature modes), in the variables of the solver: photon
    /// Theta_0, Theta_1, baryon and CDM delta, v, free-streaming neutrino N_0, N_1, N_2
    struct IsoValues {
        double Theta0 = 0, Theta1 = 0, delta_b = 0, v_b = 0, delta_c = 0, v_c = 0, N0 = 0, N1 = 0, N2 = 0;
    };

    /// Initial conditions (superhorizon, radiation domination): adiabatic with curvature perturbation R = 1, or, if
    /// iso is set, explicit values (species not covered by IsoValues start from the adiabatic form with Psi, Phi)
    struct InitialData {
        const IsoValues * iso = nullptr;
        double Psi, Phi;
        double f_nu;        ///< fraction of the radiation density in free-streaming species
        double OmNu;        ///< a^4 rho of the free-streaming species [rho_c0]
        double H0;          ///< [1/s]
        double G_eff = 1.0; ///< effective gravitational constant / G at the initial time (G_eff_initial)
    };

    /// Tensor modes obey h'' + (2 + alpha_M) Hp h' + (1 + alpha_T) k^2 h = (2 a^2 / M*^2) Pi_T (conformal time), with
    /// M*^2 the effective Planck mass in units of 1/(8 pi G) and alpha_M = dln M*^2/dx (GR: 1, 0, 0)
    struct TensorCoefficients {
        double M2 = 1.0, alpha_M = 0.0, alpha_T = 0.0;
    };

    /// The options of the solver that concern the species (the solver's PerturbationOptions defaults)
    struct PerturbationSettings {
        bool neutrinos = true;            ///< Massless neutrino hierarchy (only active if Omega_nu > 0)
        int lmax_nu = 22;                 ///< Highest massless-neutrino multipole
        int lmax_ncdm = 17;               ///< Highest massive-neutrino multipole (CLASS default 17)
        int n_q_ncdm = 8;                 ///< Momentum nodes (Gauss-Laguerre) for the massive neutrinos
        double ncdm_fluid_trigger = 31.0; ///< k eta above which the massive neutrinos are a fluid (CLASS default)
        double x_start = -18.0;           ///< The range the perturbations are integrated on (where the switch points
        double x_end = 0.0;               ///< of the species are searched)
    };

    /// What the perturbation objects are built from (see the header)
    struct PerturbationContext {
        const BACKGROUND::ComponentBackground & background;
        PerturbationSettings settings;
        /// Spatial curvature [1/m^2]
        double K;
        /// H0 [1/s]
        double H0;

        explicit PerturbationContext(const BACKGROUND::ComponentBackground & bg, PerturbationSettings s = {})
            : background(bg), settings(s), K(-bg.parameters().Omega_k * bg.H0_SI() * bg.H0_SI() / (c() * c())),
              H0(bg.H0_SI()) {
            if (not(settings.x_start < settings.x_end) or not bg.in_range(settings.x_start) or
                not bg.in_range(settings.x_end))
                throw std::invalid_argument("[PerturbationContext] Need x_start < x_end inside the range of the "
                                            "background [" +
                                            std::to_string(bg.x_min()) + ", " + std::to_string(bg.x_max()) + "]");
        }

        /// The curvature factor s_l of mode k (see the header)
        double s_ell(double k, int ell) const {
            if (K == 0.0)
                return 1.0;
            const double s2 = 1.0 - (ell * ell - 1.0) * K / (k * k);
            return s2 > 0.0 ? std::sqrt(s2) : 0.0;
        }

        /// Conformal time [m]
        double eta(double x) const { return background.eta(x) * FML::UTILS::SI::Mpc / background.h(); }
        /// Hp = a H [1/s]
        double Hp(double x) const { return H0 * std::exp(x) * background.E(x); }
        /// cot_K(chi) = d ln sin_K / d chi [1/m] for chi [m] (Background::cotK)
        double cotK(double chi) const {
            const double mpch = FML::UTILS::SI::Mpc / background.h();
            return background.cotK(chi / mpch) / mpch;
        }

        /// The background part of a ModeAtX (tau', tau'', R are NaN: the solver's core adds them). Throws for s_2 = 0
        /// (a closed-universe mode with nu <= 2)
        ModeAtX mode_at_x(double x, double k) const {
            ModeAtX b;
            b.x = x;
            b.k = k;
            b.s2 = s_ell(k, 2);
            b.curv = 1.0 - 3.0 * K / (k * k);
            if (b.s2 <= 0.0)
                throw std::runtime_error("[PerturbationContext] s_2 = 0 (closed universe mode with nu <= 2)");
            b.a = std::exp(x);
            b.Hp = Hp(x);
            b.dHpH = 1.0 + background.dlogEdx(x);
            b.ckH = c() * k / b.Hp;
            b.dtau = b.ddtau = b.R = std::numeric_limits<double>::quiet_NaN();
            b.eta = eta(x);
            b.cotK_H = c() * cotK(b.eta) / b.Hp;
            b.H0a2 = H0 * H0 * b.a * b.a;
            b.H0a2_rho_b = b.H0a2 * background.rho(BACKGROUND::Component::Baryons, x);
            b.H0a2_rho_gamma = b.H0a2 * background.rho(BACKGROUND::Component::Photons, x);
            return b;
        }

      private:
        static constexpr double c() { return FML::UTILS::SI::c; }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
