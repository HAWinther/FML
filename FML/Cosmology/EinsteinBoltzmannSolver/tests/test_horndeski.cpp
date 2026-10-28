// =============================================================================
// Tests of Horndeski gravity in the alpha parameterization (models/HorndeskiAlpha.h). Run with `make test`.
//
//   * All alphas zero is LCDM (same solution)
//   * The momentum constraint (0i Einstein equation), which is not used in the evolution, holds with braiding,
//     running Planck mass and kineticity (python/derive_horndeski_alpha.py)
//   * The stability conditions reject a ghost (alpha_K < 0) and a gradient instability
//   * Known theories rebuilt from their alphas reproduce the dedicated implementations (different variables
//     and equations):
//       cubic Galileon on the tracker:  alpha_K = 6 Omega_DE(a), alpha_B = 2 Omega_DE(a)  (propto_omega)
//       quintessence:                  alpha_K = 3 (rho + p)_phi / (H^2 M_pl^2), the field perturbations in v
//       Brans-Dicke (its own expansion): M*^2 = phi, alpha_M = -alpha_B = dln phi/dx, alpha_K = omega alpha_M^2
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/BransDicke.h>
#include <FML/Cosmology/Models/HorndeskiAlpha.h>

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
            finish();
        }
        Run(const CosmologyParameters & p, Model m, const PerturbationOptions & o)
            : cosmo(p, std::move(m)), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
            finish();
        }
        void finish() {
            rec.solve();
            pert.solve();
        }
    };

    // A species seen only through its background (its perturbations are the gravity's scalar)
    class BackgroundOnly : public Species {
      public:
        explicit BackgroundOnly(std::shared_ptr<const Species> s) : s_(std::move(s)) {}
        std::string name() const override { return s_->name() + " (background)"; }
        Role role() const override { return s_->role(); }
        double rho(double x) const override { return s_->rho(x); }
        double p(double x) const override { return s_->p(x); }
        double drhodx(double x) const override { return s_->drhodx(x); }
        double ddrhoddx(double x) const override { return s_->ddrhoddx(x); }

      private:
        std::shared_ptr<const Species> s_;
    };

    CosmologyParameters horndeski(double cK, double cB, double cM, double w0 = -1.0, double wa = 0.0) {
        CosmologyParameters p;
        p.model = "horndeski";
        p.model_params["hd_cK"] = cK;
        p.model_params["hd_cB"] = cB;
        p.model_params["hd_cM"] = cM;
        p.model_params["hd_w0"] = w0;
        p.model_params["hd_wa"] = wa;
        return p;
    }

    // max over k and x of the relative difference of Phi, Psi and delta_cdm
    double compare(const Perturbations & a, const Perturbations & b, const std::vector<double> & xs) {
        double e = 0.0;
        for (double k : a.get_k_array())
            for (double x : xs) {
                e = std::max(e, std::fabs(a.get_Phi(x, k) / b.get_Phi(x, k) - 1.0));
                e = std::max(e, std::fabs(a.get_Psi(x, k) / b.get_Psi(x, k) - 1.0));
                e = std::max(e, std::fabs(a.get_delta_cdm(x, k) / b.get_delta_cdm(x, k) - 1.0));
            }
        return e;
    }
} // namespace

int main() {
    std::cout << std::unitbuf;
    PerturbationOptions o;
    o.n_k = 5;
    o.k_min = 0.001 / Mpc;
    o.k_max = 0.2 / Mpc;
    const std::vector<double> xs = {-6.0, -3.0, -1.0, 0.0};

    std::cout << "GR limit\n";
    Run lcdm(CosmologyParameters(), o);
    {
        Run r(horndeski(0.0, 0.0, 0.0), o);
        check_close("all alphas zero = LCDM (Phi, Psi, delta_cdm)", compare(r.pert, lcdm.pert, xs), 0.0, 1e-6);
    }

    std::cout << "Momentum constraint, w0-wa expansion, kineticity + braiding + running Planck mass\n";
    {
        Run r(horndeski(1.0, 0.6, 0.3, -0.95, 0.1), o);
        const auto & g = dynamic_cast<const HorndeskiAlphaGravity &>(r.cosmo.gravity());
        std::cout << "  " << g.info() << "\n";
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        double e0i = 0.0, scal = 0.0;
        for (double k : pt.get_k_array())
            for (double x : {-3.0, -1.0, 0.0}) {
                const auto s = g.state(x, cs);
                const double a = s.a, H = s.H, M2 = s.al.M2, aB = s.al.aB, H0 = cs.get_H0(), kt = c * k / H0;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double flux = H0 * H0 * a * a *
                                    (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) +
                                     4.0 * rg * pt.get_Theta(1, x, k) + 4.0 * rn * pt.get_Nu(1, x, k));
                const double Q = 3.0 * flux / (H0 * H0 * kt);
                const double v = pt.get("v_horndeski", x, k), U = pt.get("u_horndeski", x, k), Psi = pt.get_Psi(x, k);
                const double P1 = -H * pt.get_dPhidx(x, k);
                // -2 H M2 Psi + H M2 aB U - 2 M2 P1 + Q + v (H^2 M2 aB - Rx) = 0
                const double t[] = {-2.0 * H * M2 * Psi,
                                    H * M2 * aB * U,
                                    -2.0 * M2 * P1,
                                    Q,
                                    v * (H * H * M2 * aB - s.Rx)};
                const double res = t[0] + t[1] + t[2] + t[3] + t[4];
                const double gr = std::fabs(t[0]) + std::fabs(t[2]) + std::fabs(t[3]);
                e0i = std::max(e0i, std::fabs(res) / gr);
                scal = std::max(scal, (std::fabs(t[1]) + std::fabs(t[4])) / gr);
            }
        std::cout << "  scalar terms in the 0i equation: up to " << scal << " of the metric/matter terms\n";
        check_close("momentum constraint (0i) holds", e0i, 0.0, 1e-6);
        check("the scalar terms are tested (> 1e-2)", scal > 1e-2);
    }

    std::cout << "Stability conditions\n";
    {
        auto throws = [](const CosmologyParameters & p) {
            try {
                BackgroundCosmology b(p);
            } catch (const std::invalid_argument & e) {
                std::cout << "  " << e.what() << "\n";
                return true;
            }
            return false;
        };
        check("ghost (c_K = -1) rejected", throws(horndeski(-1.0, 0.0, 0.0)));
        check("gradient instability (c_B = -1, LCDM) rejected", throws(horndeski(1.0, -1.0, 0.0)));
        check("stable model accepted", !throws(horndeski(1.0, 0.6, 0.3)));
    }

    std::cout << "Cubic Galileon from alpha_K = 6 Omega_DE, alpha_B = 2 Omega_DE\n";
    {
        CosmologyParameters p;
        p.model = "cubic_galileon";
        Run ref(p, o);
        Model m = Models::build(p);
        m.gravity = std::make_shared<HorndeskiAlphaGravity>(
            std::make_shared<StandardAlphas>(StandardAlphas::Type::ProptoOmega, 6.0, 2.0, 0.0, 1.0));
        Run r(p, m, o);
        const double e = compare(r.pert, ref.pert, xs);
        const double eL = compare(lcdm.pert, ref.pert, {0.0});
        std::cout << "  (Galileon vs LCDM: " << eL << ")\n";
        check_close("Phi, Psi, delta_cdm = cubic Galileon implementation", e, 0.0, 1e-6);
    }

    std::cout << "Quintessence from alpha_K = 3 (rho + p)_phi / H^2\n";
    {
        CosmologyParameters p;
        p.model = "quintessence";
        p.model_params["quint_lambda"] = 1.0;
        Run ref(p, o);
        Model m = Models::build(p);
        std::shared_ptr<const Species> q;
        for (auto & s : m.species)
            if (s->role() == Species::Role::DarkEnergy)
                s = q = std::make_shared<BackgroundOnly>(s);
        // alpha_K = 3 (rho + p)_phi / E^2 = -rho_phi' / E^2 (x-derivatives)
        auto al = std::make_shared<FunctionAlphas>("quintessence", [q](double x, const BackgroundCosmology & bg) {
            Alphas a;
            const double E2 = bg.rho_species(x), dE2 = bg.drho_species(x);
            a.aK = -q->drhodx(x) / E2;
            a.daKdx = -(q->ddrhoddx(x) - q->drhodx(x) * dE2 / E2) / E2;
            return a;
        });
        m.gravity = std::make_shared<HorndeskiAlphaGravity>(al);
        Run r(p, m, o);
        const double e = compare(r.pert, ref.pert, xs);
        std::cout << "  (quintessence vs LCDM: " << compare(lcdm.pert, ref.pert, {0.0}) << ")\n";
        check_close("Phi, Psi, delta_cdm = quintessence implementation", e, 0.0, 1e-8);
    }

    std::cout << "Brans-Dicke from M*^2 = phi, alpha_M = -alpha_B, alpha_K = omega alpha_M^2\n";
    {
        const double omega = 100.0;
        CosmologyParameters p;
        p.model = "jbd";
        p.model_params["bd_omega"] = omega;
        Run ref(p, o);
        Model m = Models::build(p);
        auto bd = std::dynamic_pointer_cast<BransDickeGravity>(m.gravity);
        auto al =
            std::make_shared<FunctionAlphas>("Brans-Dicke", [bd, omega](double x, const BackgroundCosmology & bg) {
                double rho = 0.0, p = 0.0;
                for (const auto & s : bg.species()) {
                    rho += s->rho(x);
                    p += s->p(x);
                }
                const double F = bd->phi(x), Fx = bd->dphidx(x), E2 = bd->E2(x, bg), L = 0.5 * bd->dE2dx(x, bg) / E2;
                const double S = 3.0 * (rho - 3.0 * p) / (3.0 + 2.0 * omega), Fxx = -(3.0 + L) * Fx + S / E2;
                const double y = Fx / F, q = 1.0 - omega * y / 3.0, a2 = std::exp(2.0 * x);
                Alphas a;
                a.M2 = F;
                a.dM2 = F - 1.0;
                a.aM = y;
                a.aB = -y;
                a.aK = omega * y * y;
                const double dy = Fxx / F - y * y;
                a.daBdx = -dy;
                a.daKdx = 2.0 * omega * y * dy;
                // Rx = 2 M2 (H^2 - H') - R without cancellations (from the Friedmann and field equations)
                a.Rx = a2 * E2 *
                       (2.0 * F * Fx + 2.0 * F * q * (S / E2 - 3.0 * Fx) + omega * F * Fx * y * y / 3.0 -
                        3.0 * (rho + p) * Fx / E2) /
                       (2.0 * F + Fx);
                return a;
            });
        m.gravity = std::make_shared<HorndeskiAlphaGravity>(al, bd);
        Run r(p, m, o);
        const double e = compare(r.pert, ref.pert, xs);
        std::cout << "  (Brans-Dicke vs LCDM: " << compare(lcdm.pert, ref.pert, {0.0}) << ")\n";
        check_close("Phi, Psi, delta_cdm = Brans-Dicke implementation", e, 0.0, 1e-6);
        double ev = 0.0;
        for (double k : r.pert.get_k_array())
            for (double x : {-1.0, 0.0}) {
                const double H = std::exp(x) * std::sqrt(r.cosmo.E2_of_x(x)); // phi' = H F_x
                const double v_bd = -ref.pert.get("delta_phi_BD", x, k) * bd->phi(x) / (H * bd->dphidx(x));
                ev = std::max(ev, std::fabs(r.pert.get("v_horndeski", x, k) / v_bd - 1.0));
            }
        check_close("v = -delta phi / phi' = Brans-Dicke field perturbation", ev, 0.0, 1e-5);
    }

    return Test::summary();
}
