// =============================================================================
// Tests of the models of chameleon-like theories in the quasi-static limit
// (Perturbations/PerturbationModel_QuasiStatic.h: symmetron, dilaton, mbeta; and fr, PerturbationModel_HuSawicky.h):
// the scale-dependent growth of delta_cdm from z = 49 to 0, relative to LCDM, against the quasi-static growth equation
// solved here independently with the theory's mu(k, a) (QuasiStaticGravity_*.h, the force of SimulationCode):
//   D'' + (2 + dlnE/dx) D' = (3/2) Omega_m(a) mu(k, a) D,   x = ln a
// The ratio to LCDM removes what the growth equation leaves out (baryons, radiation, neutrinos); the scalar's
// force is negligible at z = 49 for these parameters. Run with `make test` (from the folder above).
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Dilaton.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MBeta.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Symmetron.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>

using namespace FML::COSMOLOGY;
using namespace Constants;
using FML::TESTS::check;
namespace PT = FML::COSMOLOGY::PERTURBATIONS;

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

    CosmologyParameters params(const std::string & model, const std::map<std::string, double> & mp = {}) {
        CosmologyParameters p;
        p.h = 0.671;
        p.OmegaB = 0.049;
        p.OmegaCDM = 0.2685;
        p.Neff = 3.046;
        p.TCMB = 2.7255;
        p.model = model;
        p.model_params = mp;
        return p;
    }

    /// D(x = 0) / D(x0) of the quasi-static growth equation (RK4 in x, matter-era start D = D' = 1)
    double qs_growth(const BACKGROUND::Background & bg, const std::function<double(double)> & mu, double x0) {
        const int n = 20000;
        const double h = -x0 / n;
        auto rhs = [&](double x, double D, double dD, double & ddD) {
            const double e = 1e-4;
            const double dlnE = (std::log(bg.E(x + e)) - std::log(bg.E(x - e))) / (2.0 * e);
            ddD = 1.5 * bg.Omega_m(x) * mu(x) * D - (2.0 + dlnE) * dD;
        };
        double x = x0, D = 1.0, dD = 1.0;
        for (int i = 0; i < n; i++) {
            double a1, a2, a3, a4;
            rhs(x, D, dD, a1);
            rhs(x + h / 2, D + h / 2 * dD, dD + h / 2 * a1, a2);
            rhs(x + h / 2, D + h / 2 * (dD + h / 2 * a1), dD + h / 2 * a2, a3);
            rhs(x + h, D + h * (dD + h / 2 * a2), dD + h * a3, a4);
            D += h * (dD + h / 6 * (a1 + a2 + a3));
            dD += h / 6 * (a1 + 2 * a2 + 2 * a3 + a4);
            x += h;
        }
        return D;
    }
} // namespace

int main() {
    PerturbationOptions o;
    o.k_min = 0.005 / Mpc;
    o.k_max = 3.0 / Mpc;
    const double x49 = std::log(1.0 / 50.0);
    const std::vector<double> ks{0.05, 0.3, 1.0, 2.5}; // 1/Mpc

    Run lcdm(params("lcdm"), o);
    const auto & bg = lcdm.cosmo.background();
    std::map<double, double> D_lcdm, D_lcdm_qs;
    for (double kk : ks) {
        const double k = kk / Mpc;
        D_lcdm[kk] = lcdm.pert.get_delta_cdm(0.0, k) / lcdm.pert.get_delta_cdm(x49, k);
        D_lcdm_qs[kk] = qs_growth(bg, [](double) { return 1.0; }, x49);
    }

    struct Case {
        std::string model;
        std::map<std::string, double> mp;
        std::function<PT::QuasiStaticGravityPtr(const BACKGROUND::Background &)> qs;
    };
    const double c_over_H0_mpc = FML::UTILS::SI::c_over_H0_mpch / 0.671; // koverH0 = k [1/Mpc] c/H0 [Mpc]
    const std::vector<Case> cases{
        {"symmetron",
         {{"sym_assb", 0.5}, {"sym_beta", 1.0}, {"sym_L_mpch", 1.0}},
         [](const BACKGROUND::Background & b) {
             return std::make_shared<const PT::QuasiStaticGravity_Symmetron>(b, 0.5, 1.0, 1.0);
         }},
        {"dilaton",
         {{"dil_A2", 5e5}, {"dil_beta0", 0.5}, {"dil_r", 1.0}, {"dil_xi", 1e-3}},
         [](const BACKGROUND::Background & b) {
             return std::make_shared<const PT::QuasiStaticGravity_Dilaton>(b, 5e5, 0.5, 1.0, 1e-3);
         }},
        {"mbeta",
         {},
         [](const BACKGROUND::Background & b) {
             return std::make_shared<const PT::QuasiStaticGravity_MBeta>(b, 0.408248290463863, 0.0, 400.0, -4.5);
         }},
        {"fr",
         {{"fR0", 1e-5}, {"fR_n", 1.0}},
         [](const BACKGROUND::Background & b) {
             return std::make_shared<const PT::QuasiStaticGravity_HuSawicky>(b, 1e-5, 1.0);
         }},
    };
    for (const auto & cs : cases) {
        Run r(params(cs.model, cs.mp), o);
        const auto qs = cs.qs(r.cosmo.background());
        double worst = 0.0, largest = 0.0;
        for (double kk : ks) {
            const double k = kk / Mpc;
            const double D = r.pert.get_delta_cdm(0.0, k) / r.pert.get_delta_cdm(x49, k);
            const double koverH0 = kk * c_over_H0_mpc;
            const double Dqs = qs_growth(r.cosmo.background(), [&](double x) { return qs->mu(x, koverH0); }, x49);
            const double ratio = D / D_lcdm[kk], ratio_qs = Dqs / D_lcdm_qs[kk];
            std::printf("%-10s k = %.2f/Mpc: growth z = 49 -> 0 / LCDM: solver %.6f, quasi-static %.6f\n",
                        cs.model.c_str(),
                        kk,
                        ratio,
                        ratio_qs);
            worst = std::max(worst, std::fabs(ratio / ratio_qs - 1.0));
            largest = std::max(largest, ratio_qs - 1.0);
        }
        check(largest > 0.01, cs.model + ": the fifth force enhances the growth (> 1% at the largest k)");
        check(worst < 5e-4,
              cs.model + ": the scale-dependent growth relative to LCDM = the quasi-static growth equation (5e-4)");
    }
    return FML::TESTS::summary("EBS quasi-static models");
}
