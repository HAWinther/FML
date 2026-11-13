#ifndef FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_DARKENERGYFLUID_H
#define FML_COSMOLOGY_PERTURBATIONS_SPECIESPERTURBATION_DARKENERGYFLUID_H

// =====================================================================================================================
// A dark-energy fluid with constant w (= c_a^2) and rest-frame sound speed cs2, evolved together with the cold dark
// matter it interacts with (one block: dark scattering and interacting dark energy). Newtonian gauge, x = ln a, variables (delta_c, v_c, delta_DE, u_DE) with u = theta/(ck)
// (= -v for the velocity v as of CDM). Without interaction (xi = 0) both are wCDM with dark-energy perturbations:
//   delta_DE' = -(1 + w)((ck/Hp) u + 3 Phi') - 3 (cs2 - w)[delta_DE + 3 (1 + w) u/(ck/Hp)]
//   u'        = -(1 - 3 cs2) u + (ck/Hp)[cs2 delta_DE/(1 + w) + Psi]
//   delta p_DE = rho_DE [cs2 delta_DE + 3 (1 + w)(cs2 - w) u/(ck/Hp)],   flux = -rho_c v_c + (1 + w) rho_DE u
// Adiabatic initial conditions: CDM as SpeciesPerturbation_CDM (isocurvature: IsoValues delta_c, v_c),
// delta_DE / delta_c = rho_DE'/rho_DE / (rho_c'/rho_c) of the uncoupled background, u = -v_c. Outputs delta_cdm, v_cdm,
// delta_de, v_de (= -u).
//
// SpeciesPerturbation_DarkScattering: elastic dark matter - dark energy scattering (Simpson 2010; Baldi & Simpson
// 2015) with cross section per dark-matter mass xi = sigma_D / m_c [m^2/kg]; only momentum is exchanged (the
// background is wCDM, Background/BackgroundModel_DarkScattering.h):
//   v_c' = -v_c - (ck/Hp) Psi + (Gamma_c/Hp)(v_DE - v_c),   u' = ... - (Gamma_DE/Hp)(u - u_c)
//   Gamma_DE = a c xi rho_c [1/s] (rho_c physical),   Gamma_c = (1 + w) rho_DE/rho_c Gamma_DE
// The covariantised dark Thomson-like scattering (Beltran Jimenez et al. 2020;
// DarkScatteringRate::Covariant), Q^mu = alpha_bar (u_DE^mu - u_c^mu), has the same equations with the rate
//   Gamma_c = alpha H0 a^4 / Omega_c0 [1/s],   Gamma_DE = Gamma_c R,   R = rho_c / ((1 + w) rho_DE)
// with the dimensionless alpha = 8 pi G alpha_bar / (3 H0^3) (strength = alpha; Omega_c0 = rho_c today / rho_crit).
// which conserves the total momentum. While Gamma_DE/Hp > 200 (early times) the dark-energy velocity is locked to that
// of the dark matter (mode 0, three variables; its momentum is negligible then, (1 + w) rho_DE << rho_c), afterwards
// both are evolved (mode 1). xi = 0: wCDM with dark-energy perturbations (one mode).
//
// SpeciesPerturbation_InteractingDarkEnergy: energy exchange Q = xi H rho_DE along the dark-matter 4-velocity
// (Valiviita, Majerotto & Maartens 2008; the background: Background/BackgroundModel_IDE.h), with the perturbed rate from
// the local expansion of the dark matter, delta Q / Q = delta_DE - Psi + E, E = (theta_c - 3 Phi_MB')/(3 Hp), and the
// doom factor d = xi / (3 (1 + w)):
//   delta_c'  = (ck/Hp) v_c - 3 Phi' + xi (rho_DE/rho_c)(delta_DE - delta_c + E),   v_c' = -v_c - (ck/Hp) Psi
//   delta_DE' = -(1 + w)((ck/Hp) u + 3 Phi') - 3 (cs2 - w)[delta_DE + 3 (1 + w)(1 + d) u/(ck/Hp)] - xi E
//   u'        = [3 cs2 - 1 + 3 d (cs2 - w)] u + (ck/Hp)[cs2 delta_DE/(1 + w) + Psi] + xi u + 3 d v_c
//   delta p_DE = rho_DE [cs2 delta_DE + 3 (1 + w)(1 + d)(cs2 - w) u/(ck/Hp)]
// (E = -(ck/Hp) v_c / 3 + Phi' in our variables). Adiabatic delta_DE = -(3 (1 + w) + xi)/2 Psi.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_IDE.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/Cosmology/Perturbations/PerturbationTypes.h>
#include <FML/Cosmology/Perturbations/SpeciesPerturbation.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    /// The rate of dark scattering: CrossSection (Simpson 2010: xi = sigma_D / m_c) or Covariant (the covariantised dark
    /// Thomson-like scattering: alpha)
    enum class DarkScatteringRate { CrossSection, Covariant };

    class SpeciesPerturbation_DarkScattering : public SpeciesPerturbation {
      public:
        /// cdm, dark_energy: the background species (dark_energy has constant w = p/rho); xi: the strength, sigma_D / m_c
        /// [m^2/kg] (CrossSection) or alpha (Covariant)
        SpeciesPerturbation_DarkScattering(const BACKGROUND::Species & cdm,
                                           const BACKGROUND::Species & dark_energy,
                                           double cs2,
                                           double xi,
                                           const PerturbationContext & ctx,
                                           DarkScatteringRate kind = DarkScatteringRate::CrossSection)
            : cdm(cdm), de(dark_energy), w(dark_energy.w(0.0)), cs2(cs2), xi(xi),
              rho_crit0(3.0 * ctx.H0 * ctx.H0 / (8.0 * M_PI * FML::UTILS::SI::G)), kind(kind), H0(ctx.H0),
              rho_c0(cdm.rho(0.0)) {
            const auto & s = ctx.settings;
            auto r = [&](double x) { return rate_DE(x) / ctx.Hp(x) - lock_ratio; };
            if (xi == 0.0 or r(s.x_start) <= 0.0)
                locked_start = false;
            else {
                locked_start = true;
                x_unlock = r(s.x_end) > 0.0 ? 1e30 : FML::MATH::find_root(r, s.x_start, s.x_end);
            }
        }
        std::string name() const override {
            return kind == DarkScatteringRate::Covariant ? "covariantised dark Thomson-like scattering (CDM + dark energy)"
                                                         : "dark scattering (CDM + dark energy)";
        }
        std::string info() const override {
            std::ostringstream os;
            os << name();
            if (locked_start)
                os << " (velocities locked until z = " << std::exp(-x_unlock) - 1.0 << ")";
            return os.str();
        }
        std::vector<double> switch_points(double) const override {
            if (locked_start and x_unlock < 1e29)
                return {x_unlock};
            return {};
        }
        bool locked(int mode) const { return locked_start and mode == 0; }
        int n_vars(int mode) const override { return locked(mode) ? 3 : 4; }

        /// Gamma_DE [1/s]: a c xi rho_c, or (Covariant) Gamma_c R with Gamma_c = alpha H0 a^4 / Omega_c0
        double rate_DE(double x) const {
            if (kind == DarkScatteringRate::Covariant) {
                const double a = std::exp(x);
                const double rate_c = xi * H0 * a * a * a * a / rho_c0;
                return rate_c * cdm.rho(x) / ((1.0 + w) * de.rho(x));
            }
            return std::exp(x) * FML::UTILS::SI::c * xi * cdm.rho(x) * rho_crit0;
        }

        enum { RC, RDE, GDE, GC, N_CACHE }; // H0^2 a^2 rho_c, H0^2 a^2 rho_DE, Gamma_DE/Hp, Gamma_c/Hp
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cc) const override {
            const double rc = cdm.rho(b.x), rde = de.rho(b.x);
            cc[RC] = b.H0a2 * rc;
            cc[RDE] = b.H0a2 * rde;
            cc[GDE] = rate_DE(b.x) / b.Hp;
            cc[GC] = (1.0 + w) * rde / rc * cc[GDE];
        }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            // CDM: the isocurvature values if set (the dark energy has no isocurvature mode of its own and starts from
            // the adiabatic form with the Psi of the mode, as the other species not in IsoValues)
            y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
            y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
            y[2] = -1.5 * (1.0 + w) * ic.Psi; // adiabatic: delta_DE/(1 + w) = delta_c
            if (not locked(0))
                y[3] = 0.5 * b.ckH * ic.Psi; // u = theta/(ck) = -v of the dark matter
        }
        void switch_mode(const ModeAtX &, const double *, int, const double * yo, double * yn) const override {
            yn[0] = yo[0];
            yn[1] = yo[1];
            yn[2] = yo[2];
            yn[3] = -yo[1];
        }
        Moments moments(const ModeAtX & b, const double * cc, int mode, const double * y) const override {
            const double u = locked(mode) ? -y[1] : y[3];
            Moments m;
            m.drho = cc[RC] * y[0] + cc[RDE] * y[2];
            m.dp = cc[RDE] * (cs2 * y[2] + 3.0 * (1.0 + w) * (cs2 - w) * u / b.ckH);
            m.flux = -cc[RC] * y[1] + (1.0 + w) * cc[RDE] * u;
            return m;
        }
        void rhs(const ModeAtX & b, const double * cc, int mode, const MetricAtX & m, const double * y, double * dydx)
            const override {
            const double ckH = b.ckH;
            const bool lk = locked(mode);
            const double vc = y[1], u = lk ? -vc : y[3];
            dydx[0] = ckH * vc - 3.0 * m.dPhi;
            dydx[1] = -vc - ckH * m.Psi + (lk ? 0.0 : cc[GC] * (-u - vc));
            dydx[2] = -(1.0 + w) * (ckH * u + 3.0 * m.dPhi) - 3.0 * (cs2 - w) * (y[2] + 3.0 * (1.0 + w) * u / ckH);
            if (not lk)
                dydx[3] = -(1.0 - 3.0 * cs2) * u + ckH * (cs2 * y[2] / (1.0 + w) + m.Psi) - cc[GDE] * (u + vc);
        }
        std::vector<std::string> output_names() const override { return {"delta_cdm", "v_cdm", "delta_de", "v_de"}; }
        void outputs(const ModeAtX &, const double *, int mode, const double * y, double * out) const override {
            out[0] = y[0];
            out[1] = y[1];
            out[2] = y[2];
            out[3] = locked(mode) ? y[1] : -y[3];
        }

        double get_w() const { return w; }
        double get_xi() const { return xi; }
        double get_rho_crit0() const { return rho_crit0; }

      private:
        const BACKGROUND::Species & cdm;
        const BACKGROUND::Species & de;
        double w, cs2, xi, rho_crit0;
        DarkScatteringRate kind;
        double H0, rho_c0;
        static constexpr double lock_ratio = 200.0;
        bool locked_start = false;
        double x_unlock = 1e30;
    };

    class SpeciesPerturbation_InteractingDarkEnergy : public SpeciesPerturbation {
      public:
        SpeciesPerturbation_InteractingDarkEnergy(const BACKGROUND::InteractingCDMSpecies & cdm,
                                                  const BACKGROUND::InteractingDarkEnergySpecies & dark_energy,
                                                  double cs2,
                                                  const PerturbationContext & /*ctx*/)
            : cdm(cdm), de(dark_energy), w(dark_energy.w), xi(dark_energy.xi), cs2(cs2), d(xi / (3.0 * (1.0 + w))),
              L(3.0 * (1.0 + w) + xi) {}
        std::string name() const override { return "interacting dark energy (CDM + dark energy)"; }
        int n_vars(int) const override { return 4; }

        enum { RC, RDE, N_CACHE }; // H0^2 a^2 rho_c, H0^2 a^2 rho_DE
        int cache_size() const override { return N_CACHE; }
        void update_cache(const ModeAtX & b, double * cc) const override {
            cc[RC] = b.H0a2 * cdm.rho(b.x);
            cc[RDE] = b.H0a2 * de.rho(b.x);
        }
        void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
            // CDM: the isocurvature values if set (the dark energy starts from the adiabatic form, see the header)
            y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
            y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
            y[2] = -1.5 * L / 3.0 * ic.Psi; // adiabatic: delta_DE / delta_c = rho_DE'/rho_DE / (rho_c'/rho_c)
            y[3] = 0.5 * b.ckH * ic.Psi;    // u = -v_c
        }
        Moments moments(const ModeAtX & b, const double * cc, int, const double * y) const override {
            Moments m;
            m.drho = cc[RC] * y[0] + cc[RDE] * y[2];
            m.dp = cc[RDE] * (cs2 * y[2] + 3.0 * (1.0 + w) * (1.0 + d) * (cs2 - w) * y[3] / b.ckH);
            m.flux = -cc[RC] * y[1] + (1.0 + w) * cc[RDE] * y[3];
            return m;
        }
        void rhs(const ModeAtX & b, const double * cc, int, const MetricAtX & m, const double * y, double * dydx)
            const override {
            const double ckH = b.ckH;
            const double dc = y[0], vc = y[1], dde = y[2], u = y[3];
            const double E = -ckH * vc / 3.0 + m.dPhi; // (theta_c - 3 Phi_MB')/(3 Hp)
            dydx[0] = ckH * vc - 3.0 * m.dPhi + xi * cc[RDE] / cc[RC] * (dde - dc + E);
            dydx[1] = -vc - ckH * m.Psi;
            dydx[2] = -(1.0 + w) * (ckH * u + 3.0 * m.dPhi) - 3.0 * (cs2 - w) * (dde + 3.0 * (1.0 + w) * (1.0 + d) * u / ckH) -
                      xi * E;
            dydx[3] = (3.0 * cs2 - 1.0 + 3.0 * d * (cs2 - w)) * u + ckH * (cs2 * dde / (1.0 + w) + m.Psi) + xi * u +
                      3.0 * d * vc;
        }
        std::vector<std::string> output_names() const override { return {"delta_cdm", "v_cdm", "delta_de", "v_de"}; }
        void outputs(const ModeAtX &, const double *, int, const double * y, double * out) const override {
            out[0] = y[0];
            out[1] = y[1];
            out[2] = y[2];
            out[3] = -y[3];
        }
        /// The doom factor xi / (3 (1 + w))
        double doom() const { return d; }

      private:
        const BACKGROUND::InteractingCDMSpecies & cdm;
        const BACKGROUND::InteractingDarkEnergySpecies & de;
        double w, xi, cs2, d, L;
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
