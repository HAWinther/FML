//==========================================================
// Tests for FML::COSMOLOGY::SPHERICALCOLLAPSE against analytic results and an independent integration
// (scipy, R(t) in cosmic time with the same definitions and second-order initial conditions):
//   - Einstein-de Sitter: deltac = 1.686470, DeltaVir = 18 pi^2, a_ta / a_c = 2^(-2/3), 1 + delta_ta = 5.5517 for any
//     collapse redshift and also when starting late (a_ini = 1e-3)
//   - LCDM (OmegaM = 0.3, 0.1), an open model (OmegaM = 0.3, OmegaK = 0.7) and w0wa + constant mu
//   - deltac(z) and DeltaVir(z) from compute_sphericalcollapse_splines equal the single-redshift values
//   - the linear solution keeps evolving after the collapse
//   - a copy of the LCDM model does not refer to the original (and uses its own OmegaM)
//   - errors: mu <= 0 and no collapse before the given time throw (also from inside the OpenMP loop)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace FML::COSMOLOGY::SPHERICALCOLLAPSE;
using Spline = FML::INTERPOLATION::SPLINE::Spline;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(8);
    os << x;
    return os.str();
}

// Flat or curved w0wa model with constant mu
SphericalCollapseModel make_model(double Om, double Ok, double w0, double wa, double mu0) {
    const double Ode = 1 - Om - Ok;
    auto rhode = [=](double x) {
        const double a = std::exp(x);
        return Ode * std::exp(3.0 * wa * (a - 1) - 3 * (1 + w0 + wa) * x);
    };
    Func1 E2 = [=](double x) { return Om * std::exp(-3 * x) + Ok * std::exp(-2 * x) + rhode(x); };
    Func1 E = [=](double x) { return std::sqrt(E2(x)); };
    Func1 OmX = [=](double x) { return Om * std::exp(-3 * x) / E2(x); };
    Func1 dlogE = [=](double x) {
        const double a = std::exp(x);
        return 0.5 / E2(x) *
               (-3 * Om * std::exp(-3 * x) - 2 * Ok * std::exp(-2 * x) + rhode(x) * (3 * wa * a - 3 * (1 + w0 + wa)));
    };
    Func1 mu = [=](double) { return mu0; };
    Func1 w = [=](double x) { return w0 + wa * (1 - std::exp(x)); };
    return SphericalCollapseModel(E, OmX, dlogE, mu, w);
}

struct Result {
    double deltac, DeltaVir, xta, xvir, delta_ini;
    Spline delta, delta_lin;
};
Result run(const SphericalCollapseModel & model, double z, double aini = 1e-5) {
    SphericalCollapse sc(model, false, 1000, std::log(aini));
    Result r;
    Spline dprime;
    double xnl;
    sc.run_single_redshift(z,
                           r.delta,
                           dprime,
                           r.delta_lin,
                           r.deltac,
                           r.DeltaVir,
                           xnl,
                           r.xta,
                           r.xvir,
                           r.delta_ini,
                           false);
    return r;
}

int main() {
    const double deltac_EdS = 3.0 / 20.0 * std::pow(12.0 * M_PI, 2.0 / 3.0);
    const double DeltaVir_EdS = 18.0 * M_PI * M_PI;

    // EdS for several collapse redshifts and two starting times
    {
        auto eds = make_model(1.0, 0.0, -1.0, 0.0, 1.0);
        double err_dc = 0.0, err_dv = 0.0, err_ta = 0.0, err_dta = 0.0;
        for (double z : {0.0, 1.0, 10.0}) {
            auto r = run(eds, z);
            const double xc = -std::log(1 + z);
            err_dc = std::max(err_dc, std::fabs(r.deltac / deltac_EdS - 1));
            err_dv = std::max(err_dv, std::fabs(r.DeltaVir / DeltaVir_EdS - 1));
            err_ta = std::max(err_ta, std::fabs(std::exp(r.xta - xc) / std::pow(2.0, -2.0 / 3.0) - 1));
            err_dta = std::max(err_dta, std::fabs((1 + r.delta(r.xta)) / (9.0 * M_PI * M_PI / 16.0) - 1));
        }
        check(err_dc < 2e-6, "EdS deltac = 1.686470 for z = 0, 1, 10 (max rel. error " + num(err_dc) + ")");
        check(err_dv < 2e-5, "EdS DeltaVir = 18 pi^2 (max rel. error " + num(err_dv) + ")");
        check(err_ta < 1e-5 and err_dta < 1e-4, "EdS turnaround a_ta / a_c = 2^(-2/3) and 1 + delta_ta = 9 pi^2 / 16");
        auto late = run(eds, 10.0, 1e-3);
        check(std::fabs(late.deltac / deltac_EdS - 1) < 5e-4,
              "EdS deltac when starting at a = 1e-3 and collapse at z = 10: " + num(late.deltac) + " (was 1.7226)");
    }

    // LCDM, open, w0wa + mu against the independent integration (deltac, DeltaVir)
    {
        struct Case {
            std::string name;
            double Om, Ok, w0, wa, mu, z, deltac, DeltaVir;
        };
        const Case cases[] = {
            {"LCDM OmegaM = 0.3, z = 0", 0.3, 0.0, -1.0, 0.0, 1.0, 0.0, 1.675527, 339.802},
            {"LCDM OmegaM = 0.3, z = 1", 0.3, 0.0, -1.0, 0.0, 1.0, 1.0, 1.684256, 200.825},
            {"LCDM OmegaM = 0.1, z = 0", 0.1, 0.0, -1.0, 0.0, 1.0, 0.0, 1.665141, 715.153},
            {"open OmegaM = 0.3, OmegaK = 0.7, z = 0", 0.3, 0.7, -1.0, 0.0, 1.0, 0.0, 1.651333, 402.341},
            {"open OmegaM = 0.3, OmegaK = 0.7, z = 1", 0.3, 0.7, -1.0, 0.0, 1.0, 1.0, 1.664576, 294.049},
        };
        for (auto & c : cases) {
            auto r = run(make_model(c.Om, c.Ok, c.w0, c.wa, c.mu), c.z);
            check(std::fabs(r.deltac / c.deltac - 1) < 2e-6 and std::fabs(r.DeltaVir / c.DeltaVir - 1) < 2e-4,
                  c.name + ": deltac " + num(r.deltac) + " (" + num(c.deltac) + "), DeltaVir " + num(r.DeltaVir) +
                      " (" + num(c.DeltaVir) + ")");
        }
    }

    // deltac(z) and DeltaVir(z) splines: smooth and equal to single-redshift runs
    {
        SphericalCollapseModel lcdm;
        lcdm.OmegaM = 0.3;
        Spline deltac, DeltaVir, xnl, xta, xvir, dini;
        compute_sphericalcollapse_splines(lcdm, deltac, DeltaVir, xnl, xta, xvir, dini, std::log(1e-5), 1000, false);
        double maxerr = 0.0;
        for (double z : {0.0, 0.37, 1.0, 2.9, 7.3}) {
            auto r = run(lcdm, z);
            const double x = -std::log(1 + z);
            maxerr = std::max({maxerr, std::fabs(deltac(x) / r.deltac - 1), std::fabs(DeltaVir(x) / r.DeltaVir - 1)});
        }
        check(maxerr < 1e-5,
              "deltac(z), DeltaVir(z) splines = single redshift runs (max rel. difference " + num(maxerr) + ")");
    }

    // The linear solution after the collapse follows the growth factor
    {
        auto eds = make_model(1.0, 0.0, -1.0, 0.0, 1.0);
        SphericalCollapse sc(eds, false, 1000, std::log(1e-5));
        Spline d, dp, dl;
        double deltac, DV, xnl, xta, xvir, dini = 2.0 * 1.686470 * 1e-5; // collapses at a = 0.5
        sc.run_single_redshift(0.0, d, dp, dl, deltac, DV, xnl, xta, xvir, dini, true);
        check(std::fabs(dl(0.0) / (dini * 1e5) - 1) < 1e-4,
              "linear delta keeps growing after the collapse (delta_lin(a=1) / (delta_ini a / a_ini) = " +
                  num(dl(0.0) / (dini * 1e5)) + ")");
    }

    // A copy of the LCDM model uses its own OmegaM and does not refer to the original
    {
        auto original = std::make_unique<SphericalCollapseModel>();
        original->OmegaM = 0.3;
        SphericalCollapseModel copy = *original;
        original.reset();
        copy.OmegaM = 0.1;
        const double om = copy.OmegaMofx(0.0);
        auto r = run(copy, 0.0);
        check(std::fabs(om - 0.1) < 1e-14 and std::fabs(r.DeltaVir / 715.15 - 1) < 2e-4,
              "copy of the LCDM model uses its own OmegaM (OmegaM(z=0) = " + num(om) + ")");
    }

    // Errors are reported as exceptions (also from the OpenMP loop)
    {
        auto neg = make_model(0.3, 0.0, -1.0, 0.0, -0.5);
        bool thrown = false;
        try {
            Spline a, b, c, d, e, f;
            compute_sphericalcollapse_splines(neg, a, b, c, d, e, f);
        } catch (std::exception & e) {
            thrown = std::string(e.what()).find("Negative Geff/G") != std::string::npos;
        }
        check(thrown, "mu < 0 throws a clear exception");

        // Starting at a = 1e-2 nothing collapses at z = 1000
        thrown = false;
        try {
            run(make_model(1.0, 0.0, -1.0, 0.0, 1.0), 1000.0, 1e-2);
        } catch (std::exception & e) {
            thrown = std::string(e.what()).find("No collapse") != std::string::npos;
        }
        check(thrown, "no collapse before the given redshift throws (no garbage result)");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All spherical collapse tests passed\n" :
                                   std::to_string(fails) + " spherical collapse test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
