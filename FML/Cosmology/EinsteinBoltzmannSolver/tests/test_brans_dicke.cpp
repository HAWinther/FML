// =============================================================================
// Tests of Jordan-Brans-Dicke gravity (models/BransDicke.h). Run with `make test`.
//
//   * Background: phi(0) gives the Newtonian G, E(0) = 1, dlnE/dx and d2E2/dx2 vs finite differences,
//     the field equation vs finite differences of the solution, and phi ~ a^{1/(1 + omega)} in the matter era
//   * omega -> infinity is LCDM
//   * Perturbations: the momentum constraint (0i Einstein equation), which is not used in the evolution,
//     holds; it involves all terms of the derived equations (python/derive_brans_dicke.py)
//   * Growth on small scales follows G_eff = G (4 + 2 omega)/((3 + 2 omega) phi)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/BransDicke.h>

#include <cmath>
#include <functional>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    CosmologyParameters jbd(double omega) {
        CosmologyParameters p;
        p.model = "jbd";
        p.model_params["bd_omega"] = omega;
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

    // Growth factor from x0 to x = 0 with G_eff(x), starting with dlnD/dx = p0 at x0
    double growth(const BackgroundCosmology & cs, const std::function<double(double)> & Geff, double x0, double p0) {
        const int n = 60000;
        const double h = -x0 / n;
        auto acc = [&](double x, double D, double dD) {
            const double E2 = cs.E2_of_x(x), dlnH = 0.5 * cs.dE2dx_of_x(x) / E2;
            return -(2.0 + dlnH) * dD + 1.5 * Geff(x) * (cs.rhoCDM(x) + cs.rhoB(x)) / E2 * D;
        };
        double D = 1.0, dD = p0;
        for (int i = 0; i < n; i++) {
            const double x = x0 + i * h;
            const double k1 = dD, l1 = acc(x, D, dD);
            const double k2 = dD + 0.5 * h * l1, l2 = acc(x + 0.5 * h, D + 0.5 * h * k1, dD + 0.5 * h * l1);
            const double k3 = dD + 0.5 * h * l2, l3 = acc(x + 0.5 * h, D + 0.5 * h * k2, dD + 0.5 * h * l2);
            const double k4 = dD + h * l3, l4 = acc(x + h, D + h * k3, dD + h * l3);
            D += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
            dD += h / 6.0 * (l1 + 2 * l2 + 2 * l3 + l4);
        }
        return D;
    }
} // namespace

int main() {
    const double omega = 100.0;
    std::cout << "Background (omega = " << omega << ")\n";
    BackgroundCosmology b(jbd(omega));
    b.solve();
    const auto & g = dynamic_cast<const BransDickeGravity &>(b.gravity());
    {
        check_rel("phi(0) = (4 + 2 omega)/(3 + 2 omega): G_eff(0) = G",
                  g.phi(0.0),
                  (4.0 + 2.0 * omega) / (3.0 + 2.0 * omega),
                  1e-12);
        check_close("E(0) = 1", b.E2_of_x(0.0), 1.0, 1e-12);
        double e1 = 0.0, e2 = 0.0, eKG = 0.0;
        for (double x : {-12.0, -6.0, -2.0, 0.0, 1.0}) {
            const double h = 1e-4;
            e1 = std::max(e1, std::fabs((b.E2_of_x(x + h) - b.E2_of_x(x - h)) / (2 * h) / b.dE2dx_of_x(x) - 1.0));
            e2 = std::max(e2,
                          std::fabs((b.dE2dx_of_x(x + h) - b.dE2dx_of_x(x - h)) / (2 * h) / b.ddE2ddx_of_x(x) - 1.0));
            // F_xx (finite differences of the solution) = -(3 + dlnE/dx) F_x + 3 (rho - 3p) / ((3 + 2 omega) E^2)
            double p = 0.0;
            for (const auto & s : b.species())
                p += s->p(x);
            const double Fxx = (g.dphidx(x + h) - g.dphidx(x - h)) / (2 * h);
            const double L = 0.5 * b.dE2dx_of_x(x) / b.E2_of_x(x);
            const double rhs =
                -(3.0 + L) * g.dphidx(x) + 3.0 * (b.rho_species(x) - 3.0 * p) / ((3.0 + 2.0 * omega) * b.E2_of_x(x));
            eKG = std::max(eKG, std::fabs(Fxx - rhs) / std::fabs(g.dphidx(x)));
        }
        check_close("dE2/dx vs finite differences", e1, 0.0, 1e-7);
        check_close("d2E2/dx2 vs finite differences", e2, 0.0, 1e-5);
        check_close("field equation vs finite differences", eKG, 0.0, 1e-5);
        // Matter era (Nariai): phi ~ a^{1/(1 + omega)}; small corrections from radiation and Lambda at x = -4
        check_close("matter era: dln phi/dx = 1/(1 + omega)", g.dphidx(-3.0) / g.phi(-3.0) * (1.0 + omega), 1.0, 2e-2);
        check("radiation era: phi frozen (dln phi/dx < 1e-5)", std::fabs(g.dphidx(-16.0) / g.phi(-16.0)) < 1e-5);
    }

    PerturbationOptions o;
    o.n_k = 5;
    o.k_min = 0.001 / Mpc;
    o.k_max = 0.2 / Mpc;
    Run lcdm(CosmologyParameters(), o);
    const auto & ks = lcdm.pert.get_k_array();

    std::cout << "omega -> infinity is LCDM\n";
    {
        Run r(jbd(1e8), o);
        double e = 0.0;
        for (double k : ks)
            for (double x : {-10.0, -7.0, -2.0, 0.0}) {
                e = std::max(e, std::fabs(r.pert.get_Phi(x, k) / lcdm.pert.get_Phi(x, k) - 1.0));
                e = std::max(e, std::fabs(r.pert.get_delta_cdm(x, k) / lcdm.pert.get_delta_cdm(x, k) - 1.0));
            }
        check_close("omega = 1e8: Phi, delta_cdm = LCDM", e, 0.0, 1e-6);
    }

    std::cout << "Perturbations (omega = " << omega << ")\n";
    Run r(jbd(omega), o);
    {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        double emom = 0.0, scale = 0.0;
        for (double k : ks)
            for (double x : {-6.0, -3.0, -1.0, 0.0}) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double flux = H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                            4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k));
                const double F = g.phi(x), Fx = g.dphidx(x);
                const double dF = F * pt.get("delta_phi_BD", x, k), dFx = F * pt.get("ddelta_phi_BD_dx", x, k);
                const double Psi = pt.get_Psi(x, k), dPhi = pt.get_dPhidx(x, k);
                const double lhs = -omega * dF * Fx / F + 2.0 * F * Psi + dF - 2.0 * F * dPhi + Fx * Psi - dFx;
                const double rhs = 3.0 * flux / (ckH * Hp * Hp);
                emom = std::max(emom, std::fabs(lhs - rhs) / std::fabs(rhs));
                scale = std::max(scale, std::fabs(dF - omega * dF * Fx / F - dFx + Fx * Psi) / std::fabs(rhs));
            }
        std::cout << "  scalar-field terms in the momentum constraint: up to " << scale << " of the matter flux term\n";
        check_close("momentum constraint (0i) holds", emom, 0.0, 1e-5);
        check("the scalar-field terms are tested (> 1e-3 of the flux term)", scale > 1e-3);

        // Growth from x0 (matter era) to today, started with each run's own dln delta/dx at x0 (removes the
        // different early evolution and the baryon transient)
        const double k = ks.back(), x0 = -4.0, h = 1e-3;
        auto dln = [&](const Perturbations & P) {
            return (std::log(P.get_delta_cdm(x0 + h, k)) - std::log(P.get_delta_cdm(x0 - h, k))) / (2.0 * h);
        };
        const double ratio = (pt.get_delta_cdm(0.0, k) / pt.get_delta_cdm(x0, k)) /
                             (lcdm.pert.get_delta_cdm(0.0, k) / lcdm.pert.get_delta_cdm(x0, k));
        const double ref = growth(
                               cs,
                               [&](double x) { return g.G_eff(x); },
                               x0,
                               dln(pt)) /
                           growth(lcdm.cosmo, [](double) { return 1.0; }, x0, dln(lcdm.pert));
        std::cout << "  growth from z = 54 to 0 relative to LCDM: " << ratio << ", growth equation with G_eff: " << ref
                  << "\n";
        check_close("growth = growth equation with G_eff = G phi_0/phi", ratio / ref, 1.0, 1e-3);
    }

    return Test::summary();
}
