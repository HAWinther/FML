//==========================================================
// Tests for FML::COSMOLOGY::SPHERICALCOLLAPSE screened spherical collapse (ScreenedSphericalCollapse.h):
//   - no fifth force = the standard spherical collapse of SphericalCollapse.h (delta_ini, delta_c, x_vir, DeltaVir)
//   - a constant unscreened F with mu = 1 + F = SphericalCollapse.h with that mu
//   - limits: f(R) with fR0 = 0 and a very massive (screened) halo -> GR, nDGP with Omega_rc -> 0 -> GR, the linear
//     nDGP force 1/(3 beta); delta_c decreases with decreasing halo mass between GR and the unscreened collapse
//   - against ReACT's spherical collapse (ACTio-ReACTio a36dda02, SCOL::myscol and SCOL::maxP_zeta called directly;
//     generator: ReACT/tests/scol_oracle.cpp): the most probable environment, delta_c, a_vir and delta(a_vir) for
//     Hu-Sawicki f(R) (|fR0| = 1e-6, 1e-5, 1e-4; R_th = 0.3 - 8 Mpc/h; z = 0, 0.5, 1) and nDGP (Omega_rc = 0.25,
//     1; z = 0, 1), OmegaM = 0.3 and sigma^2(8 Mpc/h) = 0.64 D(z)^2 / D(0)^2
//   - the environment threshold for OmegaM = 1, the Lagrangian radius, invalid input
// Run with: make run (in this folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/SphericalCollapse/ScreenedSphericalCollapse.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

using FML::TESTS::check_local;
using namespace FML::COSMOLOGY::SPHERICALCOLLAPSE;

namespace {
    std::string fmt(double v) {
        std::ostringstream s;
        s.precision(3);
        s << v;
        return s.str();
    }
    SphericalCollapseModel lcdm(double Om) {
        SphericalCollapseModel m;
        m.OmegaM = Om;
        m.set_model_lcdm();
        return m;
    }
    SphericalCollapseModel lcdm_mu(double Om, double mu) {
        auto m = lcdm(Om);
        SphericalCollapseModel r(m.Eofx, m.OmegaMofx, m.logEprimeofx, [mu](double) { return mu; }, m.wofx);
        r.OmegaM = Om;
        return r;
    }
    // ReACT: model (1 GR, 2 f(R), 3 nDGP), a, R_th [Mpc/h], sigma^2(8), d sigma^2 / dR (8), parameter (|fR0| or
    // Omega_rc), zeta, delta_c, a_vir, delta(a_vir)
    struct Ref {
        int model;
        double a, R, s2, ds2, p, zeta, deltac, avir, delta_avir;
    };
    const Ref react[] = {
        {1, 1.0, 1.0, 0.64, -0.1, 0, 7.4142172263e-01, 1.6755532704e+00, 9.2123211859e-01, 2.6438524908e+02},
        {2, 1.0, 0.3, 0.64, -0.1, 1e-06, 7.4142172263e-01, 1.4111486077e+00, 9.2421062335e-01, 2.1597329303e+02},
        {2, 1.0, 1.0, 0.64, -0.1, 1e-06, 7.4142172263e-01, 1.4971987692e+00, 9.2445936437e-01, 2.1805055397e+02},
        {2, 1.0, 3.0, 0.64, -0.1, 1e-06, 7.4142172263e-01, 1.5970555079e+00, 9.2133430714e-01, 2.1931174898e+02},
        {2, 1.0, 8.0, 0.64, -0.1, 1e-06, 7.4142172263e-01, 1.6607489327e+00, 9.2083921891e-01, 2.5151103445e+02},
        {2, 0.6667, 0.3, 0.3826264831, -0.05978538799, 1e-06, 4.4296845513e-01, 1.4598085551e+00, 6.2279334931e-01, 1.5338751755e+02},
        {2, 0.6667, 1.0, 0.3826264831, -0.05978538799, 1e-06, 4.4296845513e-01, 1.5472341412e+00, 6.2304197776e-01, 1.5724892383e+02},
        {2, 0.6667, 3.0, 0.3826264831, -0.05978538799, 1e-06, 4.4296845513e-01, 1.6441482311e+00, 6.2108487315e-01, 1.6918303535e+02},
        {2, 0.6667, 8.0, 0.3826264831, -0.05978538799, 1e-06, 4.4296845513e-01, 1.6758413947e+00, 6.2066101337e-01, 1.8188769016e+02},
        {2, 0.5, 0.3, 0.2395560191, -0.03743062798, 1e-06, 2.7500579521e-01, 1.5086560816e+00, 4.6905015822e-01, 1.3700639524e+02},
        {2, 0.5, 1.0, 0.2395560191, -0.03743062798, 1e-06, 2.7500579521e-01, 1.5997645266e+00, 4.6923146306e-01, 1.4470883957e+02},
        {2, 0.5, 3.0, 0.2395560191, -0.03743062798, 1e-06, 2.7500579521e-01, 1.6701110088e+00, 4.6772554247e-01, 1.5872655211e+02},
        {2, 0.5, 8.0, 0.2395560191, -0.03743062798, 1e-06, 2.7500579521e-01, 1.6821518098e+00, 4.6744314490e-01, 1.6225445278e+02},
        {2, 1.0, 0.3, 0.64, -0.1, 1e-05, 7.4142172263e-01, 1.3378990216e+00, 9.2428007257e-01, 2.1548054112e+02},
        {2, 1.0, 1.0, 0.64, -0.1, 1e-05, 7.4142172263e-01, 1.4146628254e+00, 9.2421714883e-01, 2.1601427496e+02},
        {2, 1.0, 3.0, 0.64, -0.1, 1e-05, 7.4142172263e-01, 1.4932425633e+00, 9.2446404998e-01, 2.1787349067e+02},
        {2, 1.0, 8.0, 0.64, -0.1, 1e-05, 7.4142172263e-01, 1.5788318295e+00, 9.2237400144e-01, 2.1760661381e+02},
        {2, 0.6667, 0.3, 0.3826264831, -0.05978538799, 1e-05, 4.4296845513e-01, 1.3844459026e+00, 6.2257508284e-01, 1.5247176966e+02},
        {2, 0.6667, 1.0, 0.3826264831, -0.05978538799, 1e-05, 4.4296845513e-01, 1.4634283498e+00, 6.2269226576e-01, 1.5340437349e+02},
        {2, 0.6667, 3.0, 0.3826264831, -0.05978538799, 1e-05, 4.4296845513e-01, 1.5430950069e+00, 6.2301788485e-01, 1.5687538608e+02},
        {2, 0.6667, 8.0, 0.3826264831, -0.05978538799, 1e-05, 4.4296845513e-01, 1.6317012200e+00, 6.2121290889e-01, 1.6539615345e+02},
        {2, 0.5, 0.3, 0.2395560191, -0.03743062798, 1e-05, 2.7500579521e-01, 1.4317754964e+00, 4.6888927964e-01, 1.3539761029e+02},
        {2, 0.5, 1.0, 0.2395560191, -0.03743062798, 1e-05, 2.7500579521e-01, 1.5124211208e+00, 4.6902866361e-01, 1.3712251987e+02},
        {2, 0.5, 3.0, 0.2395560191, -0.03743062798, 1e-05, 2.7500579521e-01, 1.5949633010e+00, 4.6932647780e-01, 1.4440402732e+02},
        {2, 0.5, 8.0, 0.2395560191, -0.03743062798, 1e-05, 2.7500579521e-01, 1.6647725811e+00, 4.6790516289e-01, 1.5723031115e+02},
        {2, 1.0, 0.3, 0.64, -0.1, 0.0001, 7.4142172263e-01, 1.2702212290e+00, 9.2424825760e-01, 2.1528188520e+02},
        {2, 1.0, 1.0, 0.64, -0.1, 0.0001, 7.4142172263e-01, 1.3410966447e+00, 9.2410299171e-01, 2.1532763997e+02},
        {2, 1.0, 3.0, 0.64, -0.1, 0.0001, 7.4142172263e-01, 1.4111486077e+00, 9.2421062335e-01, 2.1597329303e+02},
        {2, 1.0, 8.0, 0.64, -0.1, 0.0001, 7.4142172263e-01, 1.4802118958e+00, 9.2439905592e-01, 2.1743223885e+02},
        {2, 0.6667, 0.3, 0.3826264831, -0.05978538799, 0.0001, 4.4296845513e-01, 1.3141626698e+00, 6.2254620219e-01, 1.5225775259e+02},
        {2, 0.6667, 1.0, 0.3826264831, -0.05978538799, 0.0001, 4.4296845513e-01, 1.3877939363e+00, 6.2258432340e-01, 1.5255538950e+02},
        {2, 0.6667, 3.0, 0.3826264831, -0.05978538799, 0.0001, 4.4296845513e-01, 1.4598085551e+00, 6.2279334931e-01, 1.5338751755e+02},
        {2, 0.6667, 8.0, 0.3826264831, -0.05978538799, 0.0001, 4.4296845513e-01, 1.5299957586e+00, 6.2302144876e-01, 1.5589147442e+02},
        {2, 0.5, 0.3, 0.2395560191, -0.03743062798, 0.0001, 2.7500579521e-01, 1.3593901744e+00, 4.6883675679e-01, 1.3489441463e+02},
        {2, 0.5, 1.0, 0.2395560191, -0.03743062798, 0.0001, 2.7500579521e-01, 1.4351045217e+00, 4.6888864332e-01, 1.3538058890e+02},
        {2, 0.5, 3.0, 0.2395560191, -0.03743062798, 0.0001, 2.7500579521e-01, 1.5086560816e+00, 4.6905015822e-01, 1.3700639524e+02},
        {2, 0.5, 8.0, 0.2395560191, -0.03743062798, 0.0001, 2.7500579521e-01, 1.5804438445e+00, 4.6948864184e-01, 1.4340651565e+02},
        {3, 1.0, 1.0, 0.64, -0.1, 0.25, 7.4142172263e-01, 1.5781635706e+00, 9.1866821123e-01, 2.3362252132e+02},
        {3, 0.5, 1.0, 0.2395560191, -0.03743062798, 0.25, 2.7500579521e-01, 1.6311961243e+00, 4.6664011387e-01, 1.5146128726e+02},
        {3, 1.0, 1.0, 0.64, -0.1, 1.0, 7.4142172263e-01, 1.5111072344e+00, 9.1767324569e-01, 2.2052425367e+02},
        {3, 0.5, 1.0, 0.2395560191, -0.03743062798, 1.0, 2.7500579521e-01, 1.5877540308e+00, 4.6621413070e-01, 1.4401660776e+02},
    };
} // namespace

int main() {
    const double Om = 0.3;
    const auto model = lcdm(Om);

    std::cout << "No fifth force = SphericalCollapse.h\n";
    for (double z : {0.0, 1.0}) {
        const double xc = -std::log1p(z);
        const auto r = screened_spherical_collapse(model, fifth_force_none(), xc);
        SphericalCollapse sc(model);
        Spline d, dp, dl;
        double deltac, DeltaVir, xnl, xta, xvir, delta_ini;
        sc.run_single_redshift(z, d, dp, dl, deltac, DeltaVir, xnl, xta, xvir, delta_ini, false);
        check_local(std::abs(r.delta_ini / delta_ini - 1) < 1e-8 and std::abs(r.deltac / deltac - 1) < 1e-8,
                    "z = " + fmt(z) + ": delta_ini, delta_c = SphericalCollapse.h (" + fmt(r.deltac) + ")");
        check_local(std::abs(r.xvir - xvir) < 1e-6 and std::abs(r.xta - xta) < 1e-6 and
                        std::abs(r.DeltaVir / DeltaVir - 1) < 1e-5,
                    "z = " + fmt(z) + ": turnaround, virialization, DeltaVir = SphericalCollapse.h");
    }

    std::cout << "Constant unscreened F with mu = 1 + F = SphericalCollapse.h with that mu\n";
    {
        const double mu = 1.2;
        const auto m = lcdm_mu(Om, mu);
        const auto r = screened_spherical_collapse(m, fifth_force_unscreened(m.muofx), 0.0);
        SphericalCollapse sc(m);
        Spline d, dp, dl;
        double deltac, DeltaVir, xnl, xta, xvir, delta_ini;
        sc.run_single_redshift(0.0, d, dp, dl, deltac, DeltaVir, xnl, xta, xvir, delta_ini, false);
        check_local(std::abs(r.deltac / deltac - 1) < 1e-8 and std::abs(r.xvir - xvir) < 1e-6 and
                        std::abs(r.DeltaVir / DeltaVir - 1) < 1e-5,
                    "mu = 1.2: delta_c, x_vir, DeltaVir = SphericalCollapse.h");
    }

    std::cout << "Limits\n";
    {
        const auto gr = screened_spherical_collapse(model, fifth_force_none(), 0.0, 0.5);
        const auto f0 = screened_spherical_collapse(model, fifth_force_hu_sawicki(0.0, 1.0, Om, 1.0), 0.0, 0.5);
        check_local(f0.deltac == gr.deltac and f0.xvir == gr.xvir, "f(R) with fR0 = 0 is GR");
        const auto big = screened_spherical_collapse(model, fifth_force_hu_sawicki(1e-6, 1.0, Om, 100.0), 0.0, 0.5);
        check_local(std::abs(big.deltac / gr.deltac - 1) < 1e-4,
                    "f(R): a very massive halo (R_th = 100 Mpc/h, |fR0| = 1e-6) is screened: delta_c = GR to " +
                        fmt(big.deltac / gr.deltac - 1));
        double prev = gr.deltac;
        bool monotonic = true, bounded = true;
        const auto unscreened = screened_spherical_collapse(model, [](double, double, double) { return 1.0 / 3.0; }, 0.0, 0.5);
        for (double R : {8.0, 3.0, 1.0, 0.3, 0.1}) {
            const auto r = screened_spherical_collapse(model, fifth_force_hu_sawicki(1e-5, 1.0, Om, R), 0.0, 0.5);
            monotonic = monotonic and r.deltac < prev;
            bounded = bounded and r.deltac >= unscreened.deltac;
            prev = r.deltac;
        }
        check_local(monotonic and bounded,
                    "f(R): delta_c decreases with the halo mass, between GR and the unscreened collapse (" +
                        fmt(unscreened.deltac) + " - " + fmt(gr.deltac) + ")");
        const auto dgp0 = screened_spherical_collapse(model, fifth_force_ndgp(1e-10, model), 0.0);
        check_local(std::abs(dgp0.deltac / gr.deltac - 1) < 1e-5, "nDGP with Omega_rc = 1e-10 is GR");
        // Linear force: 1 / (3 beta) for delta -> 0
        const double Orc = 0.25, x = -0.5;
        const double Hrc = model.Eofx(x) / (2 * std::sqrt(Orc));
        const double beta = 1 + 2 * Hrc * (1 + model.logEprimeofx(x) / 3);
        check_local(std::abs(fifth_force_ndgp(Orc, model)(x, 1e-9, 0.0) * 3 * beta - 1) < 1e-8,
                    "nDGP: F -> 1 / (3 beta) in linear theory");
        check_local(fifth_force_ndgp(Orc, model)(x, 1e4, 0.0) < 0.1 / (3 * beta), "nDGP: Vainshtein screening (F < 10% of linear) at delta = 1e4");
    }

    std::cout << "Against ReACT (SCOL.cpp)\n";
    {
        double ez = 0, edc = 0, eav = 0, edv = 0;
        for (const auto & r : react) {
            const double z = 1.0 / r.a - 1.0;
            const double zeta = environment_most_probable(r.s2, r.ds2, Om, z);
            const FifthForce F = r.model == 1 ? fifth_force_none() :
                                 r.model == 2 ? fifth_force_hu_sawicki(r.p, 1.0, Om, r.R) :
                                                fifth_force_ndgp(r.p, model);
            const auto c = screened_spherical_collapse(model, F, std::log(r.a), zeta);
            ez = std::max(ez, std::abs(zeta / r.zeta - 1));
            edc = std::max(edc, std::abs(c.deltac / r.deltac - 1));
            eav = std::max(eav, std::abs(std::exp(c.xvir) / r.avir - 1));
            edv = std::max(edv, std::abs(c.delta_vir / r.delta_avir - 1));
        }
        std::cout << "  max rel diff: zeta " << ez << ", delta_c " << edc << ", a_vir " << eav << ", delta(a_vir) " << edv
                  << " (" << sizeof(react) / sizeof(react[0]) << " cases)\n";
        check_local(ez < 1e-5, "most probable environment = ReACT");
        check_local(edc < 2e-4, "delta_c(M, z) = ReACT (f(R), nDGP, GR)");
        check_local(eav < 5e-4, "a_vir = ReACT");
        check_local(edv < 3e-3, "delta(a_vir) = ReACT");
    }

    std::cout << "Helpers and invalid input\n";
    {
        check_local(std::abs(environment_collapse_threshold(1.0, 0.0) - 3.0 / 20.0 * std::pow(12 * M_PI, 2.0 / 3.0)) < 1e-12,
                    "environment threshold = 1.686 for OmegaM = 1");
        // ReACT: R_th = 0.1 (G M / (5 OmegaM))^(1/3) with G = 4.302e-9 Mpc (km/s)^2 / Msun
        const double M = 1e14;
        check_local(std::abs(lagrangian_radius(M, Om) / (0.1 * std::cbrt(4.302e-9 * M / (5 * Om))) - 1) < 1e-3,
                    "Lagrangian radius = ReACT's");
        check_local(FML::TESTS::throws([&] { fifth_force_hu_sawicki(-1e-5, 1.0, Om, 1.0); }), "negative fR0 rejected");
        check_local(FML::TESTS::throws([&] { fifth_force_ndgp(-1.0, model); }), "negative Omega_rc rejected");
        check_local(FML::TESTS::throws([&] { environment_most_probable(-1.0, -0.1, Om, 0.0); }), "sigma^2 <= 0 rejected");
        check_local(FML::TESTS::throws([&] { screened_spherical_collapse(model, fifth_force_none(), -11.0); }),
                    "collapse too close to the initial time rejected");
    }
    return FML::TESTS::summary("Screened spherical collapse");
}
