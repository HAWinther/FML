// =============================================================================
// Tests of mixed ultralight-axion dark matter (FML/Cosmology/Background/BackgroundModel_Axion.h,
// FML/Cosmology/Perturbations/PerturbationModel_Axion.h, SpeciesPerturbation_ScalarField.h). Run with `make test`.
//
//   * Shooting: Omega_ax = ax_fraction Omega_cdm today for m = 1e-33 ... 1e-18 eV; E(0) = 1; frozen (w = -1) early;
//     a w = 0 fluid after the switch; the axionCAMB switch is at m / H = 3; the switch point varies smoothly with m
//   * Perturbations: the Einstein constraints that are not evolved (Poisson, momentum) hold with the axion's density
//     and momentum before and after the switch to the fluid
//   * axionCAMB (with ax_switch_mH = 3; its massless-neutrino fix, see python/compare_axioncamb.py): P_cb / P_cb(LCDM)
//     and delta_ax / delta_cdm for m = 1e-25 eV, f = 0.1; the axion transfer function of an all-axion dark matter
//     (f = 1) through the Jeans cutoff; the axion isocurvature mode
//   * nonlinear = axionhmcode: the P_NL/P_L table is AxionHMcode's total / linear total for the solver's P_cb, P_axion
//   * Limits: a tiny fraction is LCDM; a heavy axion (1e-20 eV) is CDM on these scales; the axion is in P_m iff
//     m >= 1e-25 eV
// =============================================================================

#include <FML/AxionHMcode/AxionHMcode.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/BackgroundModel_Axion.h>
#include <FML/Cosmology/Perturbations/Isocurvature.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <memory>

using namespace FML::COSMOLOGY;
using AxionSpecies = FML::COSMOLOGY::BACKGROUND::AxionSpecies;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;

namespace {
    // The cosmology of the axionCAMB runs (massless neutrinos)
    CosmologyParameters base() {
        CosmologyParameters p;
        p.h = 0.675;
        p.OmegaB = 0.0224 / (0.675 * 0.675);
        p.OmegaCDM = 0.12 / (0.675 * 0.675);
        p.Neff = 3.046;
        p.TCMB = 2.7255;
        p.A_s = 2.1e-9;
        p.n_s = 0.965;
        p.Yp = 0.245;
        p.z_reion = 8.0;
        return p;
    }
    CosmologyParameters axion(double log10m, double f, double switch_mH = 0.0) {
        CosmologyParameters p = base();
        p.model = "axion";
        p.model_params = {{"ax_log10_m", log10m}, {"ax_fraction", f}, {"ax_switch_mH", switch_mH}};
        return p;
    }

    struct Run {
        BackgroundCosmology cosmo;
        RecombinationHistory rec;
        Perturbations pert;
        Run(const CosmologyParameters & p, const PerturbationOptions & o)
            : cosmo(p), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            rec.solve();
            pert.solve();
        }
        // synchronous gauge (CDM frame), as axionCAMB
        double delta_sync(const std::string & what, double x, double k) const {
            const double f = cosmo.Hp_of_x(x) / (c * k), vc = pert.get_v_cdm(x, k);
            if (what == "cdm")
                return pert.get_delta_cdm(x, k) - 3.0 * f * vc;
            if (what == "b")
                return pert.get_delta_b(x, k) - 3.0 * f * vc;
            const auto A = BACKGROUND::Component::Axions;
            return pert.get("delta_axion", x, k) + cosmo.drho_of(A, x) / cosmo.rho_of(A, x) * f * vc;
        }
        double delta_cb(double x, double k) const {
            const double rc = cosmo.rhoCDM(x), rb = cosmo.rhoB(x);
            return (rc * delta_sync("cdm", x, k) + rb * delta_sync("b", x, k)) / (rc + rb);
        }
    };

    const AxionSpecies & field(const BackgroundCosmology & b) {
        for (const auto & s : b.background().species())
            if (auto e = dynamic_cast<const AxionSpecies *>(s.get()))
                return *e;
        throw std::runtime_error("no axion");
    }
    double kh(double k_hMpc) { return k_hMpc * 0.675 / Mpc; }
} // namespace

int main() {
    std::cout << std::unitbuf;
    std::cout << "Background: shooting for Omega_ax, m = 1e-33 ... 1e-18 eV (f = 0.1)\n";
    {
        double eO = 0.0, eE = 0.0, ew = 0.0, efl = 0.0;
        for (double lm : {-33.0, -30.0, -27.0, -25.0, -22.0, -18.0}) {
            BackgroundCosmology b(axion(lm, 0.1));
            b.solve();
            const auto & a = field(b);
            const double Omega_ax = 0.1 * base().OmegaCDM;
            eO = std::max(eO, std::fabs(a.rho(0.0) / Omega_ax - 1.0));
            eE = std::max(eE, std::fabs(b.E2_of_x(0.0) - 1.0));
            if (a.mass_over_H0() / std::sqrt(b.E2_of_x(-20.0)) < 1e-2) // (m = 1e-18 eV is rolling there)
                ew = std::max(ew, std::fabs(a.p(-20.0) / a.rho(-20.0) + 1.0));
            const double xf = a.x_fluid();
            if (std::isfinite(xf) and xf < -0.5) {
                const double x = 0.5 * (xf - 0.5);
                efl = std::max({efl, std::fabs(a.p(x) / a.rho(x)), std::fabs(a.drhodx(x) / a.rho(x) + 3.0)});
            }
            if (lm == -27.0)
                std::cout << "  " << a.info() << "\n";
        }
        check_close(eO, 0.0, 1e-9, "Omega_ax = ax_fraction Omega_cdm today");
        check_close(eE, 0.0, 1e-12, "E(0) = 1 (Lambda from closure)");
        check_close(ew, 0.0, 1e-6, "frozen at a = 2e-9 (w = -1) where m / H < 0.01");
        check_close(efl, 0.0, 1e-12, "after the switch: w = 0, rho ~ a^-3");
    }
    std::cout << "Background: the switch\n";
    {
        BackgroundCosmology b(axion(-25.0, 0.1, 3.0));
        b.solve();
        const auto & a = field(b);
        const double xf = a.x_fluid();
        check_rel(a.mass_over_H0() / std::sqrt(b.E2_of_x(xf)), 3.0, 1e-6, "ax_switch_mH = 3: the fluid from m = 3H");
        check_rel(xf, a.x_m_equals_3H(), 1e-6, "... = x_m_equals_3H()");
        // the switch point (at a crossing of the minimum) varies smoothly with the mass
        std::vector<double> xs;
        for (double lm : {-26.01, -26.0, -25.99}) {
            BackgroundCosmology bb(axion(lm, 0.1));
            bb.solve();
            xs.push_back(field(bb).x_fluid());
        }
        const double d1 = xs[1] - xs[0], d2 = xs[2] - xs[1];
        std::cout << "  x_switch steps " << d1 << ", " << d2 << " for d log10 m = 0.01\n";
        check_close(d2 - d1, 0.0, 2e-4, "smooth in m (no steps of the grid, dx = 0.002)");
        bool thrown = false;
        try {
            BackgroundCosmology bad(axion(-25.0, 0.1, 4.0));
            bad.solve();
        } catch (const std::exception &) {
            thrown = true;
        }
        check(not thrown, "ax_switch_mH = 4 is before the first crossing");
    }

    // The Einstein constraints (Poisson, momentum) with the axion's density and momentum
    auto constraints = [](const Run & r, const std::vector<double> & xs, double & eP, double & eM, double & frac) {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        const auto & f = field(cs);
        eP = eM = frac = 0.0;
        for (double k : pt.get_k_array())
            for (double x : xs) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double dfield = H0a2 * f.rho(x) * pt.get("delta_axion", x, k);
                const double ffield = H0a2 * f.rho(x) * pt.get("q_axion", x, k);
                const double drho = H0a2 * (rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) +
                                            4.0 * rg * pt.get_Theta(0, x, k) + 4.0 * rn * pt.get_Nu(0, x, k)) +
                                    dfield;
                const double flux = H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                            4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k)) +
                                    ffield;
                frac =
                    std::max(frac,
                             (std::fabs(dfield) + std::fabs(3.0 * ffield / ckH)) / std::fabs(drho + 3.0 * flux / ckH));
                const double D = drho + 3.0 * flux / ckH;
                eP = std::max(eP, std::fabs(c * c * k * k * pt.get_Phi(x, k) - 1.5 * D) / std::fabs(1.5 * drho));
                eM = std::max(
                    eM,
                    std::fabs((pt.get_dPhidx(x, k) - pt.get_Psi(x, k)) / (-1.5 * flux / (ckH * Hp * Hp)) - 1.0));
            }
    };
    std::cout << "Perturbations: Einstein constraints with the axion (m = 1e-25 eV, f = 0.5)\n";
    {
        PerturbationOptions o;
        o.n_k = 4;
        o.k_min = 0.002 / Mpc;
        o.k_max = 0.2 / Mpc;
        Run r(axion(-25.0, 0.5), o);
        const double xf = field(r.cosmo).x_fluid();
        auto nodes = [&](std::vector<double> xs) {
            for (auto & x : xs) {
                const auto & g = r.pert.get_x_array();
                x = *std::min_element(
                    g.begin(), g.end(), [x](double a, double b) { return std::fabs(a - x) < std::fabs(b - x); });
            }
            return xs;
        };
        double eP0, eM0, frac0, eP, eM, frac;
        constraints(r, nodes({xf - 2.0, xf - 1.0}), eP0, eM0, frac0);
        constraints(r, nodes({xf + 0.05, xf + 0.5, xf + 2.0, -3.0}), eP, eM, frac);
        std::cout << "  the fluid from a = " << std::exp(xf) << "; axion share of the comoving density perturbation: "
                  << frac0 << " before, " << frac << " after the switch\n";
        check_close(eP0, 0.0, 5e-6, "field: Poisson constraint");
        check_close(eM0, 0.0, 1e-5, "field: momentum constraint");
        check_close(eP, 0.0, 5e-6, "fluid: Poisson constraint");
        check_close(eM, 0.0, 1e-5, "fluid: momentum constraint");
        check(frac > 0.1, "the fluid's perturbations are tested (> 10%)");
    }

    PerturbationOptions o;
    o.n_k = 80; // dense in k around the cutoff of the axion transfer function
    o.k_min = 0.003 / Mpc;
    o.k_max = 0.5 / Mpc;
    std::cout << "axionCAMB (ax_switch_mH = 3; axionCAMB 2019 with its massless neutrinos in the axion background)\n";
    const Run lcdm(base(), o);
    {
        const Run r(axion(-25.0, 0.1, 3.0), o);
        // P_cb / P_cb(LCDM) and delta_ax / delta_cdm at z = 0 and k = 0.1, 0.3, 0.6 h/Mpc
        const double ref_R[3] = {0.99687536, 0.91335977, 0.68711947}, ref_ax[3] = {0.99996307, 0.99782035, 0.98170431};
        double eR = 0.0, eA = 0.0;
        int i = 0;
        for (double k : {0.1, 0.3, 0.6}) {
            const double R = std::pow(r.delta_cb(0.0, kh(k)) / lcdm.delta_cb(0.0, kh(k)), 2);
            const double A = r.delta_sync("ax", 0.0, kh(k)) / r.delta_sync("cdm", 0.0, kh(k));
            std::cout << "  k = " << k << " h/Mpc: P_cb/P_cb(LCDM) " << R << " (axionCAMB " << ref_R[i]
                      << "), delta_ax/delta_cdm " << A << " (" << ref_ax[i] << ")\n";
            eR = std::max(eR, std::fabs(R / ref_R[i] - 1.0));
            eA = std::max(eA, std::fabs(A / ref_ax[i] - 1.0));
            i++;
        }
        check_close(eR, 0.0, 1e-3, "m = 1e-25 eV, f = 0.1: P_cb / P_cb(LCDM) (to 1e-3)");
        check_close(eA, 0.0, 1e-3, "delta_ax / delta_cdm (to 1e-3)");
    }
    {
        const Run r(axion(-25.0, 1.0, 3.0), o);
        const double ref[3] = {0.99978503, 0.89234777, 0.57352975};
        double e = 0.0;
        int i = 0;
        for (double k : {0.01, 0.2, 0.3}) {
            const double T = r.delta_sync("ax", 0.0, kh(k)) / lcdm.delta_sync("cdm", 0.0, kh(k));
            std::cout << "  f = 1, k = " << k << " h/Mpc: delta_ax / delta_cdm(LCDM) " << T << " (axionCAMB " << ref[i]
                      << ")\n";
            e = std::max(e, std::fabs(T / ref[i] - 1.0));
            i++;
        }
        check_close(e, 0.0, 2e-3, "f = 1 (no CDM): the axion transfer function through the cutoff (to 2e-3)");
    }
    std::cout << "Axion isocurvature (m = 1e-25 eV, f = 0.1, ax_switch_mH = 3)\n";
    {
        const Run ad(axion(-25.0, 0.1, 3.0), o);
        const double Hinf = 1e14, A_s = base().A_s;
        const auto q = Isocurvature::with_axion_isocurvature(ad.cosmo.get_params(), ad.cosmo, std::log10(Hinf));
        const double H = Hinf / 2.435e18, phi_i = field(ad.cosmo).phi_initial();
        check_rel(q.iso_f[4], H / (M_PI * phi_i * std::sqrt(A_s)), 1e-12, "ax_log10_Hinf: f_axi = H / (pi phi_i sqrt(A_s))");
        check_rel(q.iso_n[4], 1.0 - 2.0 * H * H / (M_PI * M_PI * A_s) / 8.0, 1e-12, "n_axi = 1 - r/8");
        check(q.iso_c[4] == 0.0, "uncorrelated");
        PerturbationOptions oi = o;
        oi.ic_adiabatic = 0.0;
        oi.iso_amplitude[4] = 1.0;
        const Run iso(axion(-25.0, 0.1, 3.0), oi);
        // per unit entropy S = delta rho_ax / rho_ax initially, relative to the adiabatic mode (R = 1), at z = 0
        const double ref_c[2] = {0.02197599, 0.01308660}, ref_a[2] = {0.02339488, 0.01318497};
        double e = 0.0;
        int i = 0;
        for (double k : {0.01, 0.1}) {
            const double Tc = std::fabs(iso.delta_sync("cdm", 0.0, kh(k)) / ad.delta_sync("cdm", 0.0, kh(k)));
            const double Ta = std::fabs(iso.delta_sync("ax", 0.0, kh(k)) / ad.delta_sync("cdm", 0.0, kh(k)));
            std::cout << "  k = " << k << " h/Mpc: delta_cdm " << Tc << " (axionCAMB " << ref_c[i] << "), delta_ax "
                      << Ta << " (" << ref_a[i] << ")\n";
            e = std::max({e, std::fabs(Tc / ref_c[i] - 1.0), std::fabs(Ta / ref_a[i] - 1.0)});
            i++;
        }
        check_close(e, 0.0, 5e-3, "the isocurvature mode against axionCAMB (to 5e-3)");
        check_close(iso.pert.get_delta_cdm(o.x_start, kh(0.01)), 0.0, 1e-6, "... CDM unperturbed initially");
    }

    std::cout << "Limits\n";
    {
        PerturbationOptions o2 = o;
        o2.n_k = 8;
        o2.k_max = 0.3 / Mpc;
        const Run lc(base(), o2), tiny(axion(-25.0, 1e-5), o2), heavy(axion(-20.0, 0.1), o2);
        double et = 0.0, eh = 0.0;
        for (double k : lc.pert.get_k_array()) {
            et = std::max(et, std::fabs(tiny.delta_cb(0.0, k) / lc.delta_cb(0.0, k) - 1.0));
            const double Dm_lc = lc.delta_cb(0.0, k);
            const double rc = heavy.cosmo.rhoCDM(0.0), rb = heavy.cosmo.rhoB(0.0),
                         ra = heavy.cosmo.rho_of(BACKGROUND::Component::Axions, 0.0);
            const double Dm = (rc * heavy.delta_sync("cdm", 0.0, k) + rb * heavy.delta_sync("b", 0.0, k) +
                               ra * heavy.delta_sync("ax", 0.0, k)) /
                              (rc + rb + ra);
            eh = std::max(eh, std::fabs(Dm / Dm_lc - 1.0));
        }
        check_close(et, 0.0, 1e-4, "f = 1e-5: delta_cb today = LCDM (to 1e-4)");
        check_close(eh, 0.0, 1e-3, "m = 1e-20 eV: the matter of a mixed dark matter = LCDM for k < 0.3/Mpc (1e-3)");
        PowerSpectrumOptions so;
        const PowerSpectrum p25(heavy.cosmo, heavy.rec, heavy.pert, so);
        check(p25.has_axions() and p25.axions_in_matter(), "m = 1e-20 eV: the axion is in P_m");
        const Run light(axion(-27.0, 0.1), o2);
        const PowerSpectrum p27(light.cosmo, light.rec, light.pert, so);
        check(p27.has_axions() and not p27.axions_in_matter(), "m = 1e-27 eV: not in P_m (as axionCAMB)");
        const double k = 0.1 / Mpc;
        check_rel(p27.get_matter_power_spectrum(0.0, k), p27.get_matter_power_spectrum_cb(0.0, k), 1e-12,
                  "... P_m = P_cb");
        const double ra = heavy.cosmo.rho_of(BACKGROUND::Component::Axions, 0.0),
                     rcb = heavy.cosmo.rhoCDM(0.0) + heavy.cosmo.rhoB(0.0);
        check_rel(p25.get_Delta_M(0.0, k),
                  (rcb * p25.get_Delta_cb(0.0, k) + ra * p25.get_Delta_axion(0.0, k)) / (rcb + ra),
                  1e-12,
                  "Delta_M = (rho_cb Delta_cb + rho_ax Delta_ax) / (rho_cb + rho_ax)");
    }
    std::cout << "nonlinear = axionhmcode (m = 10^-24.5 eV, f = 0.1)\n";
    {
        PerturbationOptions o3;
        o3.n_k = 60;
        o3.k_max = 1.0 / Mpc;
        const Run r(axion(-24.5, 0.1), o3);
        PowerSpectrumOptions so;
        so.nonlinear = PowerSpectrumOptions::Nonlinear::AxionHMcode;
        so.axionhmcode_z_max = 3.0;
        so.axionhmcode_n_z = 4;
        PowerSpectrum ps(r.cosmo, r.rec, r.pert, so);
        ps.solve_matter();
        // the same input at z = 0 and k = 2.3 h/Mpc (nodes of the table), directly
        using A = FML::COSMOLOGY::AxionHMcode;
        const double h = r.cosmo.get_params().h, V = std::pow(Mpc / h, 3), kSI = h / Mpc;
        A::Input in;
        for (double lnk = std::log(1e-5); lnk <= std::log(1e3) + 1e-9; lnk += std::log(10.0) / 125.0) {
            in.k.push_back(std::exp(lnk));
            in.cold.push_back(ps.get_matter_power_spectrum_cb_extrapolated(0.0, in.k.back() * kSI) / V);
            in.axion.push_back(ps.get_matter_power_spectrum_axion_extrapolated(0.0, in.k.back() * kSI) / V);
        }
        A::Cosmology c;
        c.h = h;
        c.Omega_b = r.cosmo.rhoB(0.0);
        c.Omega_cdm = r.cosmo.rhoCDM(0.0);
        c.Omega_axion = r.cosmo.rho_of(BACKGROUND::Component::Axions, 0.0);
        c.mass_eV = std::pow(10.0, -24.5);
        c.z = 0.0;
        A::Options ao;
        ao.allow_extrapolation = true;
        const double lk = std::log(1e-3) + 100.0 * (std::log(100.0) - std::log(1e-3)) / 149.0; // node 100
        const auto direct = A::spectrum(c, in, {std::exp(lk)}, ao);
        const double k = std::exp(lk) * kSI;
        const double ratio = ps.get_matter_power_spectrum_nonlinear(0.0, k) / ps.get_matter_power_spectrum_extrapolated(0.0, k);
        std::cout << "  k = " << std::exp(lk) << " h/Mpc: P_NL/P_L = " << ratio << " (AxionHMcode "
                  << direct.total[0] / direct.linear_total[0] << ")\n";
        check_rel(ratio, direct.total[0] / direct.linear_total[0], 1e-6, "the table at a node = AxionHMcode");
        check_rel(ps.get_matter_power_spectrum_nonlinear(0.0, 1e-3 * kSI),
                  ps.get_matter_power_spectrum_extrapolated(0.0, 1e-3 * kSI), 1e-3, "linear at k = 1e-3 h/Mpc");
        check_rel(ps.get_matter_power_spectrum_nonlinear(-std::log(5.0), k),
                  ps.get_matter_power_spectrum_extrapolated(-std::log(5.0), k), 1e-12, "linear above axionhmcode_z_max");
        bool thrown = false;
        try {
            const Run l(base(), o3);
            PowerSpectrum pl(l.cosmo, l.rec, l.pert, so);
            pl.solve_matter();
        } catch (const std::exception &) {
            thrown = true;
        }
        check(thrown, "axionhmcode without an axion is an error");
    }
    return FML::TESTS::summary("EBS axion");
}
