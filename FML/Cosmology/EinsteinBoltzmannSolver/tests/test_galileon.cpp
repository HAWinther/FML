// =============================================================================
// Tests of the cubic Galileon (models/CubicGalileon.h). Run with `make test`.
//
//   * Tracker background: E^4 - rho E^2 = Omega_phi, closure, derivatives; the pressure from the scalar field,
//     p = c2 phidot^2/2 - g3 phidot^2 phiddot (with H phidot = xi), agrees with the continuity equation;
//     w = -2/(1 + Omega_phi(a)) on the tracker when the radiation is negligible
//   * Perturbations: the momentum constraint (0i Einstein equation), which is not used in the evolution, holds;
//     it contains all the braiding terms (python/derive_cubic_galileon.py)
//   * Sub-horizon limit: the growth follows the quasi-static G_eff of the same equations,
//       mu = 1 / (1 - F1^4 g3^2 / (2 a^2 (2 g3 (F1 H + F2) + a^2 c2)))   (conformal time, H0 = 1)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/CubicGalileon.h>

#include <cmath>
#include <functional>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
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

    // Quasi-static mu at x (see the header of this file)
    double mu_qs(const BackgroundCosmology & b, const CubicGalileonGravity & g, double x) {
        const double a = std::exp(x), E2 = b.E2_of_x(x), H = a * std::sqrt(E2); // conformal Hubble, H0 = 1
        const double dHpH = b.dHpdx_of_x(x) / b.Hp_of_x(x);
        const double px = g.phi_x(E2), pxx = -px * b.dE2dx_of_x(x) / E2;
        const double F1 = H * px, F2 = H * H * (dHpH * px + pxx), g3 = g.g3(), c2 = g.c2();
        return 1.0 / (1.0 - std::pow(F1, 4) * g3 * g3 / (2.0 * a * a * (2.0 * g3 * (F1 * H + F2) + a * a * c2)));
    }

    double growth(const BackgroundCosmology & cs, const std::function<double(double)> & mu, double x0, double p0) {
        const int n = 40000;
        const double h = -x0 / n;
        double D = 1.0, dD = p0;
        auto acc = [&](double x, double D, double dD) {
            const double E2 = cs.E2_of_x(x), dlnH = 0.5 * cs.dE2dx_of_x(x) / E2;
            return -(2.0 + dlnH) * dD + 1.5 * mu(x) * (cs.rhoCDM(x) + cs.rhoB(x)) / E2 * D;
        };
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
    CosmologyParameters p;
    p.model = "cubic_galileon";
    std::cout << "Tracker background\n";
    BackgroundCosmology b(p);
    b.solve();
    const auto & g = dynamic_cast<const CubicGalileonGravity &>(b.gravity());
    const CubicGalileonSpecies * gs = nullptr;
    for (const auto & s : b.species())
        if (auto q = dynamic_cast<const CubicGalileonSpecies *>(s.get()))
            gs = q;
    {
        check_close("E(0) = 1", b.E2_of_x(0.0), 1.0, 1e-13);
        double eF = 0.0, eP = 0.0, eW = 0.0, e1 = 0.0, e2 = 0.0;
        for (double x : {-8.0, -3.0, -1.0, 0.0, 1.0}) {
            const double E2 = b.E2_of_x(x), rho = b.rho_species(x) - gs->rho(x);
            eF = std::max(eF, std::fabs(E2 * E2 - rho * E2 - gs->Omega_phi()) / (E2 * E2));
            // field: H phidot = xi (H0 = 1), phiddot = -xi Hdot/H^2 = -xi (dlnE/dx) ; kappa p = 3 p_hat
            const double H = std::sqrt(E2), L = 0.5 * b.dE2dx_of_x(x) / E2;
            const double phid = g.xi() / H, phidd = -g.xi() * L; // d/dt = H d/dx: phiddot = H d(xi/H)/dx = -xi L
            const double pfield = (g.c2() * phid * phid / 2.0 - g.g3() * phid * phid * phidd) / 3.0;
            eP = std::max(eP, std::fabs(pfield - gs->p(x)) / gs->rho(x));
            if (x >= -1.0)
                eW = std::max(eW, std::fabs(gs->p(x) / gs->rho(x) + 2.0 / (1.0 + gs->rho(x) / E2)));
            const double h = 1e-4;
            e1 = std::max(e1, std::fabs((gs->rho(x + h) - gs->rho(x - h)) / (2 * h) - gs->drhodx(x)) / gs->rho(x));
            e2 = std::max(e2,
                          std::fabs((gs->drhodx(x + h) - gs->drhodx(x - h)) / (2 * h) - gs->ddrhoddx(x)) / gs->rho(x));
        }
        check_close("E^4 - rho E^2 = Omega_phi", eF, 0.0, 1e-14);
        check_close("pressure from the field = continuity", eP, 0.0, 1e-10);
        check_close("w = -2/(1 + Omega_phi(a)) (matter + galileon; radiation 1e-4)", eW, 0.0, 5e-4);
        check_close("rho', rho'' vs finite differences", std::max(e1, e2), 0.0, 1e-6);
        std::cout << "  w0 = " << gs->p(0.0) / gs->rho(0.0) << "\n";
    }

    PerturbationOptions o;
    o.n_k = 5;
    o.k_min = 0.001 / Mpc;
    o.k_max = 0.2 / Mpc;
    std::cout << "Perturbations\n";
    Run r(p, o), l(CosmologyParameters(), o);
    {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        double e0i = 0.0, scale = 0.0;
        for (double k : pt.get_k_array())
            for (double x : {-3.0, -1.0, 0.0}) {
                const double a = std::exp(x), H0 = cs.get_H0(), E2 = cs.E2_of_x(x), H = a * std::sqrt(E2);
                const double kt = c * k / H0, F1 = H * g.phi_x(E2);
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double flux = H0 * H0 * a * a *
                                    (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                     4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k));
                const double Q = 3.0 * flux / (H0 * H0 * kt);
                const double D = pt.get("delta_phi_gal", x, k), U = pt.get("u_gal", x, k), Psi = pt.get_Psi(x, k);
                const double P1 = -H * pt.get_dPhidx(x, k); // Phi_P' = -Phi' (conformal, H0 = 1)
                const double g3 = g.g3(), c2 = g.c2();
                // 0i: 3 D F1^2 H g3 - F1^2 U g3 - 2 H Psi a^2 + Q a^2 - a^2 (-D F1 c2 + 2 P1) = 0
                const double gr = std::fabs(2.0 * H * Psi * a * a) + std::fabs(Q * a * a) + std::fabs(2.0 * P1 * a * a);
                const double scal = std::fabs(3.0 * D * F1 * F1 * H * g3) + std::fabs(F1 * F1 * U * g3) +
                                    std::fabs(a * a * D * F1 * c2);
                const double res = 3.0 * D * F1 * F1 * H * g3 - F1 * F1 * U * g3 - 2.0 * H * Psi * a * a + Q * a * a -
                                   a * a * (-D * F1 * c2 + 2.0 * P1);
                e0i = std::max(e0i, std::fabs(res) / gr);
                scale = std::max(scale, scal / gr);
            }
        std::cout << "  scalar terms in the 0i equation: up to " << scale << " of the metric/matter terms\n";
        check_close("momentum constraint (0i) holds", e0i, 0.0, 1e-6);
        check("the scalar terms are tested (> 1e-3)", scale > 1e-3);

        const double k = pt.get_k_array().back(), x0 = -4.0, h = 1e-3;
        auto dln = [&](const Perturbations & P) {
            return (std::log(P.get_delta_cdm(x0 + h, k)) - std::log(P.get_delta_cdm(x0 - h, k))) / (2.0 * h);
        };
        const double ratio = (pt.get_delta_cdm(0.0, k) / pt.get_delta_cdm(x0, k)) /
                             (l.pert.get_delta_cdm(0.0, k) / l.pert.get_delta_cdm(x0, k));
        const double ref = growth(
                               cs,
                               [&](double x) { return mu_qs(cs, g, x); },
                               x0,
                               dln(pt)) /
                           growth(l.cosmo, [](double) { return 1.0; }, x0, dln(l.pert));
        std::cout << "  mu_QS today = " << mu_qs(cs, g, 0.0) << "; growth from z = 54 relative to LCDM: " << ratio
                  << ", quasi-static growth equation: " << ref << "\n";
        check_close("sub-horizon growth = quasi-static growth equation", ratio / ref, 1.0, 1e-3);
    }

    return Test::summary();
}
