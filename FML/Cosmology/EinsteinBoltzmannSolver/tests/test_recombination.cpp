// =============================================================================
// Tests for RecombinationHistory (Saha, Peebles, helium, reionization; RECFAST). Run with `make test`.
// The Peebles tests use options.model = Peebles explicitly; RECFAST (the default) is tested against reference
// values of CLASS 3.3 with recombination = RECFAST (same parameters, no reionization).
// =============================================================================

#include <FML/Cosmology/Background/BBN.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

// Max relative deviation between an analytic derivative and a central difference of f,
// normalised by the maximum of |f'| over the range (the scale that matters for integrals)
template <class F, class DF>
static double derivative_error(F f, DF df, double x_min, double x_max, double h = 1e-4) {
    double max_err = 0.0, max_df = 0.0;
    for (double x = x_min; x <= x_max; x += 0.0123) {
        const double fd = (f(x + h) - f(x - h)) / (2.0 * h);
        max_err = std::max(max_err, std::fabs(df(x) - fd));
        max_df = std::max(max_df, std::fabs(fd));
    }
    return max_err / max_df;
}

static void test_fiducial() {
    std::cout << "Fiducial model, Peebles (hydrogen only, Yp = 0.245 neutral helium)\n";
    CosmologyParameters p;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationOptions ho; // hydrogen only, no reionization
    ho.model = RecombinationOptions::Model::Peebles;
    ho.helium = false;
    ho.reionization = false;
    RecombinationHistory rec(cosmo, ho);
    rec.solve();

    // --- Saha -----------------------------------------------------------------
    {
        const double x = -7.2;
        const double T = cosmo.TCMB_of_x(x), kT = k_b * T;
        const double nH = (1.0 - p.Yp) * p.OmegaB * 3.0 * p.H0() * p.H0() / (8.0 * M_PI * G) / m_H * std::exp(-3.0 * x);
        const double rhs = std::pow(m_e * kT / (2.0 * M_PI * hbar * hbar), 1.5) *
                           std::exp(-2.0 * M_PI * hbar * c * 1.096787737e7 / kT) /
                           nH; // reduced-mass H ionization energy
        const double Xe = rec.Xe_saha_of_x(x);
        check_rel("Saha solution satisfies Xe^2/(1-Xe) = S", Xe * Xe / (1.0 - Xe), rhs, 1e-12);
        check_rel("n_H = (1 - Yp) Omega_b rho_c0 / (m_H a^3)", rec.nH_of_x(x), nH, 1e-12);
    }
    check_close("fully ionized early: X_e(x=-12) = 1", rec.Xe_of_x(-12.0), 1.0, 1e-6);
    check("Saha recombines earlier than Peebles", rec.x_Xe_half_saha() < rec.x_Xe_half());
    {
        // Shortly after the switch, Peebles should still track Saha closely
        const double xs = Utils::find_root([&](double x) { return rec.Xe_saha_of_x(x) - 0.98; }, -9.0, -6.0);
        check_close("Peebles ~ Saha just after the switch (X_e ~ 0.98)", rec.Xe_of_x(xs), rec.Xe_saha_of_x(xs), 5e-3);
    }

    // --- Optical depth ----------------------------------------------------------
    check_close("tau(0) = 0", rec.tau_of_x(0.0), 0.0, 1e-14);
    {
        bool monotone = true;
        for (double x = -19.9; x < -0.01; x += 0.01)
            monotone &= rec.tau_of_x(x) > rec.tau_of_x(x + 0.01);
        check("tau decreasing in x", monotone);
    }
    {
        const double x = -12.0; // fully ionized: tau' = -c sigma_T n_H / H
        check_rel("tau' fully ionized = -c sigma_T n_H / H",
                  rec.dtaudx_of_x(x),
                  -c * sigma_T * rec.nH_of_x(x) / cosmo.H_of_x(x),
                  1e-6);
    }
    check_close("tau' vs finite difference of tau",
                derivative_error([&](double x) { return rec.tau_of_x(x); },
                                 [&](double x) { return rec.dtaudx_of_x(x); },
                                 -12.0,
                                 -0.01),
                0.0,
                1e-6);
    {
        // d ln X_e/dx jumps from the Saha to the Peebles slope at the switch (X_e = 0.99, z ~ 1575),
        // so tau'' has a small discontinuity there. Harmless (tau ~ 1e3, g ~ 0), so test on both sides.
        const double x_sw = Utils::find_root([&](double x) { return rec.Xe_saha_of_x(x) - 0.99; }, -9.0, -6.0);
        auto dtau = [&](double x) { return rec.dtaudx_of_x(x); };
        auto ddtau = [&](double x) { return rec.ddtauddx_of_x(x); };
        check_close("tau'' vs finite difference of tau' (Saha regime)",
                    derivative_error(dtau, ddtau, -12.0, x_sw - 0.02),
                    0.0,
                    1e-5);
        check_close("tau'' vs finite difference of tau' (Peebles regime)",
                    derivative_error(dtau, ddtau, x_sw + 0.02, -0.01),
                    0.0,
                    1e-5);
    }

    // --- Visibility function ----------------------------------------------------
    {
        auto xs = Utils::linspace(-20.0, 0.0, 200001);
        std::vector<double> gs(xs.size());
        for (size_t i = 0; i < xs.size(); i++)
            gs[i] = rec.g_tilde_of_x(xs[i]);
        check_close("int g dx = 1 (trapezoid)", Utils::trapz(xs, gs), 1.0, 1e-6);
    }
    check_close("g' vs finite difference of g",
                derivative_error([&](double x) { return rec.g_tilde_of_x(x); },
                                 [&](double x) { return rec.dgdx_tilde_of_x(x); },
                                 -8.0,
                                 -5.0),
                0.0,
                1e-5);
    check_close("g'' vs finite difference of g'",
                derivative_error([&](double x) { return rec.dgdx_tilde_of_x(x); },
                                 [&](double x) { return rec.ddgddx_tilde_of_x(x); },
                                 -8.0,
                                 -5.0),
                0.0,
                1e-3);
    check_close("g = -tau' exp(-tau)",
                rec.g_tilde_of_x(-7.0),
                -rec.dtaudx_of_x(-7.0) * std::exp(-rec.tau_of_x(-7.0)),
                1e-8);

    const double z_ls = std::expm1(-rec.x_last_scattering());
    const double z_peak = std::expm1(-rec.x_visibility_peak());
    check("last scattering 1050 < z_* < 1130", z_ls > 1050 && z_ls < 1130);
    check("visibility peak within 1% of tau = 1", std::fabs(z_peak / z_ls - 1.0) < 0.01);
    check("X_e freeze-out in [1e-4, 5e-4]", rec.Xe_of_x(0.0) > 1e-4 && rec.Xe_of_x(0.0) < 5e-4);

    // --- Baryon temperature -----------------------------------------------------
    // Residual lag T_gamma - T_b ~ T H / Gamma_C ~ 1e-5 T at z ~ 1100
    check_rel("T_b = T_gamma during recombination (z ~ 1100)", rec.Tb_of_x(-7.0), cosmo.TCMB_of_x(-7.0), 1e-4);
    check("T_b < T_gamma after thermal decoupling (z = 50)",
          rec.Tb_of_x(-std::log(51.0)) < 0.98 * cosmo.TCMB_of_x(-std::log(51.0)));

    // --- Sound horizon vs independent quadrature in a ------------------------------
    {
        const double OmG = p.OmegaGamma(), OmB = p.OmegaB;
        const double x_ls = rec.x_last_scattering();
        const double s_ref = Utils::integrate(
            [&](double a) {
                const double R = 4.0 * OmG / (3.0 * OmB * a);
                const double cs = c * std::sqrt(R / (3.0 * (1.0 + R)));
                const double E = std::sqrt(p.OmegaM() / (a * a * a) + p.OmegaR() / (a * a * a * a) + p.OmegaLambda());
                return cs / (a * a * p.H0() * E);
            },
            0.0,
            std::exp(x_ls),
            0.0,
            1e-12);
        check_rel("sound horizon at z_* vs quadrature", rec.sound_horizon_of_x(x_ls), s_ref, 1e-7);
    }

    rec.info();
}

static void test_options() {
    std::cout << "Options and variants\n";
    CosmologyParameters p;
    BackgroundCosmology cosmo(p);
    cosmo.solve();

    RecombinationOptions h; // no reionization, so the freeze-out X_e is visible today
    h.model = RecombinationOptions::Model::Peebles;
    h.reionization = false;
    RecombinationOptions o = h;
    o.evolve_Tb = false;
    RecombinationHistory rec_Tg(cosmo, o);
    rec_Tg.solve();
    check_rel("evolve_Tb = false gives T_b = T_gamma", rec_Tg.Tb_of_x(-2.0), cosmo.TCMB_of_x(-2.0), 1e-14);

    RecombinationHistory rec(cosmo, h);
    rec.solve();
    check("cooler baryons recombine more: X_e(0) lower with T_b evolved", rec.Xe_of_x(0.0) < rec_Tg.Xe_of_x(0.0));

    RecombinationOptions of = h;
    of.peebles_fudge = 1.14;
    RecombinationHistory rec_f(cosmo, of);
    rec_f.solve();
    check("fudge 1.14 speeds up recombination: lower X_e(0)", rec_f.Xe_of_x(0.0) < rec.Xe_of_x(0.0));

    CosmologyParameters p0;
    p0.Yp = 0.0;
    BackgroundCosmology cosmo0(p0);
    cosmo0.solve();
    RecombinationHistory rec0(cosmo0, h);
    rec0.solve();
    check("pure hydrogen (Yp = 0): more electrons, tau larger at z = 1100",
          rec0.tau_of_x(-std::log(1101.0)) > rec.tau_of_x(-std::log(1101.0)));

    auto throws = [&](RecombinationOptions opt) {
        try {
            RecombinationHistory r(cosmo, opt);
        } catch (const std::invalid_argument &) {
            return true;
        }
        return false;
    };
    RecombinationOptions bad3;
    bad3.x_end = -1.0;
    check("x_end != 0 rejected", throws(bad3));

    CosmologyParameters pn = p;
    pn.reionization = false;
    BackgroundCosmology cosmo_n(pn);
    cosmo_n.solve();
    RecombinationHistory rec_n(cosmo_n);
    check("params.reionization = false switches reionization off", !rec_n.has_reionization());
}

static void test_helium_reionization() {
    std::cout << "Helium and reionization (Peebles model)\n";
    CosmologyParameters p;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationOptions po;
    po.model = RecombinationOptions::Model::Peebles;
    RecombinationHistory rec(cosmo, po);
    rec.solve();
    const double fHe = rec.f_He();
    auto Xz = [&](double z) { return rec.Xe_of_x(-std::log1p(z)); };

    check_rel("f_He = Yp / (4 (1 - Yp))", fHe, p.Yp / (4.0 * (1.0 - p.Yp)), 1e-14);
    check_rel("z = 1.6e5: H+ and He++ fully ionized, X_e = 1 + 2 f_He", rec.Xe_of_x(-12.0), 1.0 + 2.0 * fHe, 1e-6);
    check_rel("z = 4000: He++ recombined, X_e = 1 + f_He", Xz(4000.0), 1.0 + fHe, 1e-3);
    check_rel("z = 1700: He neutral, X_e = 1", Xz(1700.0), 1.0, 2e-3);
    {
        // X_e continuous across the Saha -> Peebles switch: a jump J shows up as |second difference| ~ J,
        // while a smooth ln X_e gives (d^2 ln X_e/dx^2) h^2 ~ 1e-5
        double max_d2 = 0.0;
        const double h = 5e-4;
        for (double x = -7.6; x < -7.2; x += h) {
            const double d2 =
                std::log(rec.Xe_of_x(x + h)) - 2.0 * std::log(rec.Xe_of_x(x)) + std::log(rec.Xe_of_x(x - h));
            max_d2 = std::max(max_d2, std::fabs(d2));
        }
        check_close("X_e continuous across the Saha/Peebles switch (max 2nd difference of ln X_e)", max_d2, 0.0, 1e-4);
    }
    check_rel("reionized (z = 6): X_e = 1 + f_He", Xz(6.0), 1.0 + fHe, 1e-3);
    check_rel("today: He fully reionized, X_e = 1 + 2 f_He", Xz(0.0), 1.0 + 2.0 * fHe, 1e-3);
    check_close("half-way at z_reion: X_e ~ (1 + f_He)/2", Xz(p.z_reion), 0.5 * (1.0 + fHe) + 0.5 * Xz(30.0), 2e-3);
    check_close("tau_reion = 0.0578 (CLASS reio_camb, same parameters)", rec.tau_reionization(), 0.0578, 0.002);
    {
        auto xs = Utils::linspace(-20.0, 0.0, 200001);
        std::vector<double> gs(xs.size());
        for (size_t i = 0; i < xs.size(); i++)
            gs[i] = rec.g_tilde_of_x(xs[i]);
        check_close("int g dx = 1 with reionization", Utils::trapz(xs, gs), 1.0, 1e-6);
        // Fraction of photons last scattered during reionization = 1 - exp(-tau_reion)
        std::vector<double> xr, gr;
        for (size_t i = 0; i < xs.size(); i++)
            if (xs[i] > -std::log(51.0)) {
                xr.push_back(xs[i]);
                gr.push_back(gs[i]);
            }
        check_rel("int g dx over z < 50 = 1 - exp(-tau_reion)",
                  Utils::trapz(xr, gr),
                  1.0 - std::exp(-rec.tau_reionization()),
                  1e-4);
    }
    {
        auto dtau = [&](double x) { return rec.dtaudx_of_x(x); };
        auto ddtau = [&](double x) { return rec.ddtauddx_of_x(x); };
        check_close("tau'' vs finite difference of tau' (reionization era)",
                    derivative_error(dtau, ddtau, -3.5, -0.01),
                    0.0,
                    1e-4);
    }
    rec.info();
}

static void test_recfast() {
    std::cout << "RECFAST (default model)\n";
    CosmologyParameters p;
    p.reionization = false;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationHistory rec(cosmo);
    rec.solve();
    auto Xz = [&](double z) { return rec.Xe_of_x(-std::log1p(z)); };
    check("RECFAST is the default", rec.get_options().model == RecombinationOptions::Model::Recfast);
    check_rel("f_He = Yp / (3.9715 (1 - Yp)) (RECFAST helium mass)", rec.f_He(), p.Yp / (3.9715 * (1.0 - p.Yp)), 1e-14);
    // CLASS 3.3, recombination = RECFAST, same cosmology, no reionization
    const double ref[][2] = {{5000, 1.0818e+00},
                             {2500, 1.0729e+00},
                             {2000, 1.0395e+00},
                             {1800, 1.0037e+00},
                             {1500, 9.5480e-01},
                             {1300, 5.6105e-01},
                             {1100, 1.4463e-01},
                             {1000, 4.8645e-02},
                             {800, 3.5530e-03},
                             {400, 5.2023e-04},
                             {100, 2.7160e-04},
                             {0, 1.7307e-04}};
    double e = 0.0;
    for (const auto & r : ref)
        e = std::max(e, std::fabs(Xz(r[0]) / r[1] - 1.0));
    check_close("X_e(z) = CLASS RECFAST (z = 5000 ... 0, 4-digit reference)", e, 0.0, 2e-4);
    check_close("z_* (tau = 1) = 1089.79 (CLASS RECFAST)",
                std::expm1(-rec.x_last_scattering()),
                1089.79,
                2e-5 * 1089.79);
    check_close("r_s(z_*) = 144.421 Mpc (CLASS RECFAST)",
                rec.sound_horizon_of_x(rec.x_last_scattering()) / Mpc,
                144.421,
                2e-5 * 144.421);
    {
        // smooth across the phase switches (Saha -> helium ODE -> full ODE)
        double max_d2 = 0.0;
        const double h = 5e-4;
        for (double x = -8.6; x < -7.0; x += h) {
            const double d2 =
                std::log(rec.Xe_of_x(x + h)) - 2.0 * std::log(rec.Xe_of_x(x)) + std::log(rec.Xe_of_x(x - h));
            max_d2 = std::max(max_d2, std::fabs(d2));
        }
        check_close("X_e continuous across the RECFAST phases (max 2nd difference of ln X_e)", max_d2, 0.0, 1e-4);
        auto dtau = [&](double x) { return rec.dtaudx_of_x(x); };
        auto ddtau = [&](double x) { return rec.ddtauddx_of_x(x); };
        // (excluding z ~ 1600, where the hydrogen equation takes over from Saha and X_e' has a small kink, as in
        // RECFAST)
        check_close("tau'' vs finite difference of tau' (before the H switch)",
                    derivative_error(dtau, ddtau, -8.5, -7.40),
                    0.0,
                    1e-4);
        check_close("tau'' vs finite difference of tau' (recombination)",
                    derivative_error(dtau, ddtau, -7.37, -6.0),
                    0.0,
                    1e-4);
    }
    // the Peebles model for comparison
    RecombinationOptions po;
    po.model = RecombinationOptions::Model::Peebles;
    po.reionization = false;
    RecombinationHistory rp(cosmo, po);
    rp.solve();
    std::cout << "  z_*: RECFAST " << std::expm1(-rec.x_last_scattering()) << ", Peebles "
              << std::expm1(-rp.x_last_scattering()) << "\n";
    // hydrogen only: RECFAST without helium tracks the hydrogen part
    RecombinationOptions ho;
    ho.helium = false;
    ho.reionization = false;
    RecombinationHistory rh(cosmo, ho);
    rh.solve();
    check_close("hydrogen only: X_e = 1 before H recombination (z = 2500)", rh.Xe_of_x(-std::log(2501.0)), 1.0, 1e-6);
    check_rel("hydrogen only: X_e(z = 1100) within 1% of the full model (helium is neutral then)",
              rh.Xe_of_x(-std::log(1101.0)),
              Xz(1100.0),
              1e-2);
    rec.info();
}

// Helium from BBN (BBN.h) against CLASS v3.4 (YHe = BBN, sBBN_2025.dat); CLASS defines Delta N = N_eff - 3.046,
// the table (and we) N_eff - 3.044, which shifts Y_He by 2.3e-5
void test_bbn() {
    std::cout << "Helium from BBN\n";
    struct Ref {
        double wb, N, Y;
    };
    for (const Ref & r : {Ref{0.02237, 3.044, 0.245274},
                          Ref{0.020, 3.044, 0.244192},
                          Ref{0.02237, 3.5, 0.251188},
                          Ref{0.024, 2.5, 0.238366}})
        check_close("Y_He(omega_b = " + std::to_string(r.wb).substr(0, 7) +
                        ", N_eff = " + std::to_string(r.N).substr(0, 5) + ") = CLASS",
                    BBN::helium_fraction(r.wb, r.N),
                    r.Y,
                    5e-5);
    const RunSettings s = ParameterFile::read("", {"preset=planck2018", "Yp_bbn=true"});
    check_close("Yp_bbn: helium_fraction() from omega_b, Neff",
                s.cosmo.helium_fraction(),
                BBN::helium_fraction(s.cosmo.OmegaB * s.cosmo.h * s.cosmo.h, s.cosmo.Neff),
                0.0);
    BackgroundCosmology bg(s.cosmo);
    bg.solve();
    RecombinationHistory rec(bg);
    check_rel("... used by the recombination (f_He)",
              rec.f_He(),
              s.cosmo.helium_fraction() / (3.9715 * (1.0 - s.cosmo.helium_fraction())),
              1e-14);
    bool threw = false;
    try {
        BBN::helium_fraction(0.1, 3.044);
    } catch (const std::exception &) {
        threw = true;
    }
    check("omega_b outside the table is an error", threw);
}

// eta(x) and t(x) are integrals from bg_x_start: outside the solved range they are an error (the spline returned the
// end value: with bg_x_start = -15 r_s was off by 6e-4), and so are recombination and perturbation ranges outside it
static void test_ranges() {
    auto throws = [](const std::function<void()> & f) {
        try {
            f();
        } catch (const std::exception & e) {
            std::cout << "  " << e.what() << "\n";
            return true;
        }
        return false;
    };
    CosmologyParameters p;
    BackgroundCosmology bg(p);
    bg.solve(-15.0, 1.0, 2000);
    check("eta_of_x below bg_x_start throws", throws([&] { bg.eta_of_x(-16.0); }));
    check("t_of_x above bg_x_end throws", throws([&] { bg.t_of_x(2.0); }));
    check("eta_of_x at the ends is fine", !throws([&] { bg.eta_of_x(-15.0) + bg.eta_of_x(1.0); }));
    check("recombination from x = -20 with bg_x_start = -15 throws",
          throws([&] { RecombinationHistory rec(bg, RecombinationOptions()); }));
    RecombinationOptions o;
    o.x_start = -15.0;
    check("recombination inside the background range is fine", !throws([&] { RecombinationHistory rec(bg, o); }));
}

// The presets against CLASS v3.4 (RECFAST, reio_camb, same N_eff and neutrinos): 100 theta_s = 100 r_s / chi at
// CLASS's z_rec (its visibility peak in conformal time). planck2018 (massless neutrinos) and planck2018_baseline
// (m_nu = 0.06 eV); the two differ by 1.5e-3
static void test_presets() {
    struct Ref {
        const char * preset;
        double z_rec, theta;
    };
    for (const Ref & r : {Ref{"planck2018", 1088.7056059980398, 1.0403343588735672},
                          Ref{"planck2018_baseline", 1088.8064871308648, 1.0418350187137981}}) {
        const RunSettings s = ParameterFile::read("", {std::string("preset=") + r.preset});
        BackgroundCosmology bg(s.cosmo);
        bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
        RecombinationHistory rec(bg, s.rec);
        rec.solve();
        const double x = -std::log1p(r.z_rec);
        const double theta = 100.0 * rec.sound_horizon_of_x(x) / bg.comoving_distance_of_x(x);
        check_rel(std::string(r.preset) + ": 100 theta_s = CLASS", theta, r.theta, 1e-5);
    }
}

int main() {
    test_recfast();
    test_fiducial();
    test_options();
    test_helium_reionization();
    test_bbn();
    test_ranges();
    test_presets();
    return Test::summary();
}
