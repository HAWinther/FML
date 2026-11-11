// =============================================================================
// Tests of Jordan-Brans-Dicke gravity (the background: FML/Cosmology/Background/FieldEquation_BransDicke.h, the perturbations:
// FML/Cosmology/Perturbations/PerturbedFieldEquation_BransDicke.h). Run with `make test`.
//
//   * Background: phi(0) gives the Newtonian G, E(0) = 1, dlnE/dx and d2E2/dx2 vs finite differences,
//     the field equation vs finite differences of the solution, and phi ~ a^{1/(1 + omega)} in the matter era
//   * omega -> infinity is LCDM
//   * Perturbations: the momentum constraint (0i Einstein equation), which is not used in the evolution,
//     holds; it involves all terms of the derived equations (python/derive_brans_dicke.py)
//   * Growth on small scales follows G_eff = G (4 + 2 omega)/((3 + 2 omega) phi)
//   * nonlinear = jbd_boost (Sletmoen & Winther 2024): the GR universe (h_GR = h sqrt(phi_ini), the sigma_8 of BD)
//     and the boost against their bd.py (hi_class + EuclidEmulator2), omega -> infinity is EuclidEmulator2 for LCDM
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Background/FieldEquation_BransDicke.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>

#include <cmath>
#include <functional>
#include <memory>

using namespace FML::COSMOLOGY;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;

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
    const auto & g = *b.background().field_equations_as<BACKGROUND::BransDickeFieldEquations>();
    {
        check_rel(g.phi(0.0),
                  (4.0 + 2.0 * omega) / (3.0 + 2.0 * omega),
                  1e-12,
                  "phi(0) = (4 + 2 omega)/(3 + 2 omega): G_eff(0) = G");
        check_close(b.E2_of_x(0.0), 1.0, 1e-12, "E(0) = 1");
        double e1 = 0.0, e2 = 0.0, eKG = 0.0;
        for (double x : {-12.0, -6.0, -2.0, 0.0, 1.0}) {
            const double h = 1e-4;
            e1 = std::max(e1, std::fabs((b.E2_of_x(x + h) - b.E2_of_x(x - h)) / (2 * h) / b.dE2dx_of_x(x) - 1.0));
            e2 = std::max(e2,
                          std::fabs((b.dE2dx_of_x(x + h) - b.dE2dx_of_x(x - h)) / (2 * h) / b.ddE2ddx_of_x(x) - 1.0));
            // F_xx (finite differences of the solution) = -(3 + dlnE/dx) F_x + 3 (rho - 3p) / ((3 + 2 omega) E^2)
            const double p = b.background().p_total(x);
            const double Fxx = (g.dphidx(x + h) - g.dphidx(x - h)) / (2 * h);
            const double L = 0.5 * b.dE2dx_of_x(x) / b.E2_of_x(x);
            const double rhs =
                -(3.0 + L) * g.dphidx(x) + 3.0 * (b.rho_species(x) - 3.0 * p) / ((3.0 + 2.0 * omega) * b.E2_of_x(x));
            eKG = std::max(eKG, std::fabs(Fxx - rhs) / std::fabs(g.dphidx(x)));
        }
        check_close(e1, 0.0, 1e-7, "dE2/dx vs finite differences");
        check_close(e2, 0.0, 1e-5, "d2E2/dx2 vs finite differences");
        check_close(eKG, 0.0, 1e-5, "field equation vs finite differences");
        // Matter era (Nariai): phi ~ a^{1/(1 + omega)}; small corrections from radiation and Lambda at x = -4
        check_close(g.dphidx(-3.0) / g.phi(-3.0) * (1.0 + omega), 1.0, 2e-2, "matter era: dln phi/dx = 1/(1 + omega)");
        check(std::fabs(g.dphidx(-16.0) / g.phi(-16.0)) < 1e-5, "radiation era: phi frozen (dln phi/dx < 1e-5)");
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
        check_close(e, 0.0, 1e-6, "omega = 1e8: Phi, delta_cdm = LCDM");
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
        check_close(emom, 0.0, 1e-5, "momentum constraint (0i) holds");
        check(scale > 1e-3, "the scalar-field terms are tested (> 1e-3 of the flux term)");

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
                               [&](double x) { return g.Geff_over_Geff_today(x); },
                               x0,
                               dln(pt)) /
                           growth(lcdm.cosmo, [](double) { return 1.0; }, x0, dln(lcdm.pert));
        std::cout << "  growth from z = 54 to 0 relative to LCDM: " << ratio << ", growth equation with G_eff: " << ref
                  << "\n";
        check_close(ratio / ref, 1.0, 1e-3, "growth = growth equation with G_eff = G phi_0/phi");
    }

    std::cout << "nonlinear = jbd_boost (omega = 100, omega_m = 0.15, omega_b = 0.02, h = 0.7, A_s = 2e-9, n_s = 1)\n";
    {
        // bd.py (github.com/hersle/jbd, with hi_class 16ae0f6 and EuclidEmulator2; its GR universe with A_s from
        // sigma_8 by A_s ~ sigma_8^2): h_GR = 0.67416, A_s,GR = 2.21973e-9 and the boost at z = 0 and 3 at three k of
        // the emulator [h/Mpc]
        const std::vector<std::string> base{"h=0.7",
                                            "OmegaB=" + std::to_string(0.02 / 0.49),
                                            "OmegaCDM=" + std::to_string(0.13 / 0.49),
                                            "A_s=2e-9",
                                            "n_s=1",
                                            "Neff=3.046",
                                            "Yp_bbn=true",
                                            "compute_cells=false",
                                            "precision=fast"};
        auto run = [&](std::vector<std::string> extra) {
            auto o = base;
            o.insert(o.end(), extra.begin(), extra.end());
            auto e = std::make_unique<EinsteinBoltzmannSolver>(EinsteinBoltzmannSolver::from_file("", o));
            e->set_verbose(false);
            e->solve();
            return e;
        };
        const auto bd = run({"model=jbd", "bd_omega=100", "nonlinear=jbd_boost"});
        const auto & ps = bd->power_spectrum();
        const double h = 0.7;
        const double kref[3] = {0.097191, 1.00213, 4.99514}, B0[3] = {0.991797, 5.665705, 25.172904},
                     B3[3] = {1.000984, 1.689274, 8.960708};
        double e = 0.0;
        for (int i = 0; i < 3; i++) {
            const double k = kref[i] * h / Mpc;
            for (int j = 0; j < 2; j++) {
                const double x = j == 0 ? 0.0 : -std::log(4.0);
                const double B = ps.get_matter_power_spectrum_nonlinear(x, k) / ps.get_matter_power_spectrum_extrapolated(x, k);
                e = std::max(e, std::fabs(B / (j == 0 ? B0[i] : B3[i]) - 1.0));
            }
        }
        check_close(e, 0.0, 1e-3, "the boost P_NL/P_L at z = 0, 3 and k = 0.1, 1, 5 h/Mpc = that of bd.py (1e-3)");
        const auto * fe = bd->background().background().field_equations_as<BACKGROUND::BransDickeFieldEquations>();
        check_rel(h * std::sqrt(fe->phi_initial()), 0.67416, 1e-4, "h_GR = h sqrt(phi_ini) = that of bd.py");
        // omega -> infinity: the GR universe is the universe itself
        const auto big = run({"model=jbd", "bd_omega=1e7", "nonlinear=jbd_boost"});
        const auto gr = run({"nonlinear=ee2"});
        double el = 0.0;
        for (double kh : {0.1, 1.0, 5.0}) {
            const double k = kh * h / Mpc;
            const auto & a = big->power_spectrum();
            const auto & b = gr->power_spectrum();
            el = std::max(el, std::fabs(a.get_matter_power_spectrum_nonlinear(0.0, k) / a.get_matter_power_spectrum_extrapolated(0.0, k) /
                                            (b.get_matter_power_spectrum_nonlinear(0.0, k) / b.get_matter_power_spectrum_extrapolated(0.0, k)) -
                                        1.0));
        }
        check_close(el, 0.0, 1e-4, "omega = 1e7: the boost = EuclidEmulator2 for LCDM");
        bool thrown = false;
        try {
            run({"nonlinear=jbd_boost"});
        } catch (const std::exception &) {
            thrown = true;
        }
        check(thrown, "jbd_boost needs model = jbd");
    }
    return FML::TESTS::summary("EBS brans dicke");
}
