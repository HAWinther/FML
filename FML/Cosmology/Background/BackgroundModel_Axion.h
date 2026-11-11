#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_AXION_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_AXION_H

// =====================================================================================================================
// Mixed ultralight-axion dark matter: the standard species, with the dark matter shared between CDM and an
// ultralight axion field (ScalarFieldSpecies, Species.h), + Lambda, GR. As axionCAMB (Hlozek, Grin, Marsh & Ferreira
// 2015, arXiv:1410.2896; github.com/dgrin1/axionCAMB):
//
//   V(phi) = m^2 phi^2 / 2,   v = V / (3 H0^2 M_pl^2) = (m / H0)^2 phi^2 / 6   (phi in M_pl)
//   Omega_ax = ax_fraction * Omega_d,   Omega_cdm = (1 - ax_fraction) * Omega_d
//
// with Omega_d the dark matter of the parameters (BackgroundParameters::Omega_cdm: axionCAMB's omdah2 with
// use_axfrac = T). The field starts at rest (frozen by Hubble friction, w = -1) at x_start and phi_i is found so that
// the density today is Omega_ax (secant iteration on ln phi_i; Lambda from closure, which does not depend on phi_i).
// Once m >~ 3H it oscillates and redshifts like matter: the background replaces it by a w = 0 fluid, either at a fast
// crossing of the minimum (m / H ~ 125, the default; ScalarFieldSpecies) or, with ax_switch_mH > 0, at m / H =
// ax_switch_mH (axionCAMB: 3; its density history ~ a^-3 from there, with the instantaneous density at the switch).
// Light axions (m <~ 1e-27 eV) are still frozen or slowly rolling today and are dark-energy like; the whole axionCAMB
// range 1e-33 - 1e-18 eV works (heavier axions oscillate before the start of the background, a = 1e-10).
//
// The axion is its own component (Component::Axions), not in Omega_m. The total matter power spectrum (PowerSpectrum)
// includes it when it is dark matter, m >= 1e-25 eV (axionCAMB's convention); P_cb is the cold spectrum.
//   make_axion(p, log10_m_eV, ax_fraction, switch_mH, r)
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    class AxionSpecies : public ScalarFieldSpecies {
      public:
        /// m_eV: the mass; Omega_axion: the density today; others: all other species (not Lambda, which is
        /// Omega_Lambda_bare() from closure); switch_mH: 0 (fast-oscillation switch) or the m/H of the switch;
        /// h: for the mass in units of H0; [x_start, x_end]: the range of the solution
        AxionSpecies(double m_eV,
                     double Omega_axion,
                     SpeciesList others,
                     double Omega_k,
                     double h,
                     double switch_mH,
                     double x_start,
                     double x_end)
            : ScalarFieldSpecies(std::move(others), Omega_k), m_eV(m_eV), Omega_ax(Omega_axion) {
            const std::string msg = "[axion] ";
            if (not(m_eV > 0.0) or not std::isfinite(m_eV))
                throw std::invalid_argument(msg + "Need m > 0");
            if (not(Omega_ax > 0.0))
                throw std::invalid_argument(msg + "Need Omega_axion > 0");
            if (not(switch_mH >= 0.0))
                throw std::invalid_argument(msg + "Need ax_switch_mH >= 0 (0: switch at fast oscillation)");
            switch_at_mass_over_H = switch_mH;
            m_H0 = m_eV / H0_eV(h);
            OmegaL = 1.0 - rho_others(0.0) - Omega_ax;

            // The field must be frozen at the start: m / H < 0.1 there
            if (not(m_H0 / std::sqrt(rho_others(x_start)) < 0.1))
                throw std::invalid_argument(msg + "m = " + std::to_string(m_eV) +
                                            " eV oscillates before the start of the background (a = " +
                                            std::to_string(std::exp(x_start)) + "): use m <= 1e-15 eV");

            // Initial guess: frozen until m = 3 H (the other species), then ~ a^-3
            double x_osc = 0.0;
            for (double x = x_start; x < 0.0; x += 0.01)
                if (m_H0 >= 3.0 * std::sqrt(rho_others(x))) {
                    x_osc = x;
                    break;
                }
            const double rho_frozen_today = Omega_ax / std::exp(3.0 * x_osc);
            double lnphi = 0.5 * std::log(6.0 * rho_frozen_today / (m_H0 * m_H0));

            // Secant on g(ln phi_i) = ln(rho(0) / Omega_ax) (~ 2 ln phi_i + const for a test field)
            auto residual = [&](double lnp) {
                solve(std::exp(lnp), 0.0, x_start, x_end);
                return std::log(rho(0.0) / Omega_ax);
            };
            double g0 = residual(lnphi);
            double lnphi1 = lnphi - 0.5 * g0;
            double g1 = residual(lnphi1);
            for (int it = 0; it < 50 and std::abs(g1) > 1e-13; it++) {
                const double slope = (g1 - g0) / (lnphi1 - lnphi);
                const double step = (std::isfinite(slope) and slope > 0.1) ? -g1 / slope : -0.5 * g1;
                lnphi = lnphi1;
                g0 = g1;
                lnphi1 = lnphi1 + std::clamp(step, -2.0, 2.0);
                g1 = residual(lnphi1);
            }
            if (not(std::abs(g1) < 1e-10))
                throw std::runtime_error(msg + "No initial field gives Omega_axion (the shooting failed)");
            phi_ini = std::exp(lnphi1);
        }

        std::string name() const override { return "axion"; }
        Component component() const override { return Component::Axions; }
        std::string info() const override {
            std::ostringstream s;
            s << "ultralight axion: m = " << m_eV << " eV (" << m_H0 << " H0), Omega_ax = " << rho(0.0)
              << ", phi_i = " << phi_ini << " M_pl, Omega_Lambda = " << OmegaL;
            if (std::isfinite(x_fluid()))
                s << "; a w = 0 fluid for a > " << std::exp(x_fluid())
                  << (switch_at_mass_over_H > 0.0 ? " (m/H = " + std::to_string(switch_at_mass_over_H) + ")" : "");
            else
                s << "; not oscillating by the end of the background";
            return s.str();
        }

        double v(double phi) const override { return m_H0 * m_H0 * phi * phi / 6.0; }
        double v_phi(double phi) const override { return m_H0 * m_H0 * phi / 3.0; }
        double v_phiphi(double) const override { return m_H0 * m_H0 / 3.0; }
        bool has_minimum() const override { return true; }
        double phi_min() const override { return 0.0; }
        double w_oscillating() const override { return 0.0; }
        double mass_over_H0() const override { return m_H0; }

        double mass_eV() const { return m_eV; }
        /// Dark matter (part of the total matter power spectrum) or dark-energy like: m >= 1e-25 eV, axionCAMB's
        /// convention (its a_osc is then well before equality)
        static constexpr double dark_matter_mass_eV = 1e-25;
        bool is_dark_matter() const { return m_eV >= dark_matter_mass_eV * (1.0 - 1e-12); }
        /// The initial field [M_pl]
        double phi_initial() const { return phi_ini; }
        /// Where m = 3 H (axionCAMB's a_osc; +infinity if not by x_end)
        double x_m_equals_3H() const {
            const int n = 4000;
            for (int i = 0; i <= n; i++) {
                const double x = x_start() + (x_end() - x_start()) * i / double(n);
                if (m_H0 >= 3.0 * std::sqrt(E2(x))) {
                    // bisect between the previous node and this one
                    double lo = x_start() + (x_end() - x_start()) * (i - 1) / double(n), hi = x;
                    for (int k = 0; k < 60; k++) {
                        const double mid = 0.5 * (lo + hi);
                        (m_H0 >= 3.0 * std::sqrt(E2(mid)) ? hi : lo) = mid;
                    }
                    return 0.5 * (lo + hi);
                }
            }
            return std::numeric_limits<double>::infinity();
        }

        /// H0 in eV (hbar H0)
        static double H0_eV(double h) {
            using namespace FML::UTILS::SI;
            return hbar * (100.0 * h * 1e3 / Mpc) / eV;
        }

      private:
        double m_eV, Omega_ax, m_H0{0.0}, phi_ini{0.0};
    };

    /// Standard species with the dark matter p.Omega_cdm shared between CDM (1 - ax_fraction) and an ultralight axion
    /// of mass 10^log10_m_eV eV (ax_fraction), + Lambda from E(0) = 1, GR
    inline ComponentBackgroundPtr make_axion(const BackgroundParameters & p,
                                             double log10_m_eV,
                                             double ax_fraction,
                                             double switch_mH,
                                             BackgroundRange r = {}) {
        if (not(ax_fraction > 0.0 and ax_fraction <= 1.0))
            throw std::invalid_argument("[axion] Need 0 < ax_fraction <= 1");
        if (not(log10_m_eV > -35.0 and log10_m_eV < -14.0))
            throw std::invalid_argument("[axion] Need -35 < ax_log10_m < -14");
        BackgroundParameters pc = p;
        pc.Omega_cdm = (1.0 - ax_fraction) * p.Omega_cdm;
        auto s = standard_species(pc);
        const auto [x0, x1] = ScalarFieldSpecies::solution_range(r);
        auto ax = std::make_shared<AxionSpecies>(
            std::pow(10.0, log10_m_eV), ax_fraction * p.Omega_cdm, s, p.Omega_k, p.h, switch_mH, x0, x1);
        s.push_back(ax);
        s.push_back(std::make_shared<PowerLawSpecies>("lambda", Component::DarkEnergy, ax->Omega_Lambda_bare(), 0.0));
        return std::make_shared<const ComponentBackground>("axion", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
