// =============================================================================
// Tests of interacting dark energy (FML/Background/BackgroundModel_IDE.h,
// FML/Perturbations/SpeciesPerturbation_DarkEnergyFluid.h). Run with `make test`.
//
//   * Background: rho_DE ~ a^{-3(1+w)-xi}; rho_c against an independent integration of
//     rho_c' = -3 rho_c + xi rho_DE; derivatives; closure; rejection of rho_c < 0 and of the doom factor d > 0
//   * xi = 0 is wCDM: same solution as dark scattering without scattering (a separate implementation)
//   * Perturbations with exchange: the Einstein constraints that are not evolved (Poisson, momentum) hold
//     only if the energy and momentum exchanged cancel
//   * Direction: energy flowing out of the dark matter (xi < 0) gives more dark matter in the past and, as the
//     outflow is uniform, faster growth of delta_c
// Poisson is checked relative to delta rho (the comoving density is small on super-horizon scales)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>
#include <FML/Perturbations/PerturbationModel_IDE.h>

#include <cmath>

using namespace FML::COSMOLOGY;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;

namespace {
    CosmologyParameters ide(double xi, double w) {
        CosmologyParameters p;
        p.model = "ide";
        p.model_params = {{"ide_xi", xi}, {"ide_w", w}};
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
    };

    bool rejected(const CosmologyParameters & p) {
        try {
            BackgroundCosmology b(p);
        } catch (const std::invalid_argument & e) {
            std::cout << "  " << e.what() << "\n";
            return true;
        }
        return false;
    }

    void constraints(const Run & r, double & eP, double & eM) {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        eP = eM = 0.0;
        for (double k : pt.get_k_array())
            for (double x : {-6.0, -2.0, -1.0, 0.0}) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x),
                             rd = cs.rhoDE(x);
                const double w = cs.background().species().back()->p(x) / rd;
                const double drho =
                    H0a2 * (rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) + 4.0 * rg * pt.get_Theta(0, x, k) +
                            4.0 * rn * pt.get_Nu(0, x, k) + rd * pt.get("delta_de", x, k));
                const double flux =
                    H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) + 4.0 * rg * pt.get_Theta(1, x, k) +
                            4.0 * rn * pt.get_Nu(1, x, k) - (1.0 + w) * rd * pt.get("v_de", x, k));
                const double D = drho + 3.0 * flux / ckH;
                eP = std::max(eP, std::fabs(c * c * k * k * pt.get_Phi(x, k) - 1.5 * D) / std::fabs(1.5 * drho));
                eM = std::max(
                    eM,
                    std::fabs((pt.get_dPhidx(x, k) - pt.get_Psi(x, k)) / (-1.5 * flux / (ckH * Hp * Hp)) - 1.0));
            }
    }
} // namespace

int main() {
    std::cout << std::unitbuf;
    const double xi = -0.1, w = -0.95;
    std::cout << "Background (xi = " << xi << ", w = " << w << ")\n";
    {
        BackgroundCosmology b(ide(xi, w));
        std::cout << "  " << b.background().species().back()->info() << "\n";
        check_close(b.E2_of_x(0.0), 1.0, 1e-13, "E(0) = 1");
        check_rel(b.rhoDE(-2.0) / b.rhoDE(0.0),
                  std::exp(2.0 * (3.0 * (1.0 + w) + xi)),
                  1e-13,
                  "rho_DE ~ a^{-3(1+w)-xi}");
        // rho_c' = -3 rho_c + xi rho_DE, RK4 from today back to x = -10
        const int n = 100000;
        const double h = -10.0 / n;
        double rc = b.rhoCDM(0.0);
        auto f = [&](double x, double r) { return -3.0 * r + xi * b.rhoDE(x); };
        for (int i = 0; i < n; i++) {
            const double x = i * h;
            const double k1 = f(x, rc), k2 = f(x + h / 2, rc + h / 2 * k1), k3 = f(x + h / 2, rc + h / 2 * k2),
                         k4 = f(x + h, rc + h * k3);
            rc += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }
        check_rel(rc, b.rhoCDM(-10.0), 1e-10, "rho_c = integral of rho_c' = -3 rho_c + xi rho_DE");
        const double a10 = std::exp(-10.0);
        std::cout << "  rho_c a^3 at a = e^-10 relative to today: " << b.rhoCDM(-10.0) * a10 * a10 * a10 / b.rhoCDM(0.0)
                  << "\n";
        check(b.rhoCDM(-10.0) * a10 * a10 * a10 > b.rhoCDM(0.0),
              "xi < 0 (energy to the dark energy): more dark matter in the past");
        double e1 = 0.0;
        for (const auto & s : b.background().species())
            for (double x : {-5.0, -1.0, 0.0}) {
                const double hh = 1e-4;
                e1 = std::max(e1, std::fabs((s->rho(x + hh) - s->rho(x - hh)) / (2 * hh) - s->drhodx(x)) / s->rho(x));
                e1 = std::max(e1,
                              std::fabs((s->drhodx(x + hh) - s->drhodx(x - hh)) / (2 * hh) - s->d2rhodx2(x)) /
                                  s->rho(x));
            }
        check_close(e1, 0.0, 1e-6, "rho', rho'' vs finite differences");
        check(rejected(ide(0.1, -0.95)), "doom factor d > 0 rejected (xi > 0, w > -1)");
        check(rejected(ide(2.0, -1.1)), "rho_c < 0 in the past rejected (xi = 2, w = -1.1)");
        check(rejected(ide(-0.1, -1.0)), "w = -1 rejected");
    }

    PerturbationOptions o;
    o.n_k = 5;
    o.k_min = 0.001 / Mpc;
    o.k_max = 0.2 / Mpc;
    std::cout << "xi = 0: wCDM, against dark scattering without scattering\n";
    {
        CosmologyParameters pd;
        pd.model = "dark_scattering";
        pd.model_params = {{"ds_w", w}, {"ds_xi", 0.0}};
        Run a(ide(0.0, w), o), b(pd, o);
        double e = 0.0;
        for (double k : a.pert.get_k_array())
            for (double x : {-6.0, -2.0, 0.0}) {
                e = std::max(e, std::fabs(a.pert.get_delta_cdm(x, k) / b.pert.get_delta_cdm(x, k) - 1.0));
                e = std::max(e, std::fabs(a.pert.get_Phi(x, k) / b.pert.get_Phi(x, k) - 1.0));
                e = std::max(e, std::fabs(a.pert.get("delta_de", x, k) / b.pert.get("delta_de", x, k) - 1.0));
            }
        check_close(e, 0.0, 1e-6, "delta_cdm, Phi, delta_DE = dark scattering (xi = 0)");
    }

    std::cout << "Perturbations with exchange\n";
    Run r(ide(xi, w), o), r0(ide(0.0, w), o);
    {
        double eP, eM;
        constraints(r, eP, eM);
        check_close(eP, 0.0, 2e-6, "Poisson constraint");
        check_close(eM, 0.0, 1e-6, "momentum constraint");
        const double k = r.pert.get_k_array().back();
        const double g = r.pert.get_delta_cdm(0.0, k) / r.pert.get_delta_cdm(-3.0, k);
        const double g0 = r0.pert.get_delta_cdm(0.0, k) / r0.pert.get_delta_cdm(-3.0, k);
        std::cout << "  growth of delta_cdm from z = 19 to 0 relative to xi = 0: " << g / g0 << "\n";
        // the outflow is proportional to the smooth rho_DE: rho_c drops but delta rho_c does not, so delta_c grows
        check(g / g0 > 1.05, "xi < 0: faster growth of delta_c (energy leaves the dark matter uniformly)");
    }
    // xi > 0 with w < -1 (d < 0, stable): runs, constraints hold
    {
        Run rp(ide(0.05, -1.05), o);
        double eP, eM;
        constraints(rp, eP, eM);
        check_close(std::max(eP, eM), 0.0, 1e-6, "xi = 0.05, w = -1.05: Poisson and momentum constraints");
    }

    return FML::TESTS::summary("EBS ide");
}
