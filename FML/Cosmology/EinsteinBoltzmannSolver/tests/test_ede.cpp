// =============================================================================
// Tests of early dark energy (models/EarlyDarkEnergy.h, models/ScalarField.h). Run with `make test`.
//
//   * Shooting: the maximum of rho_EDE/rho_tot is f_EDE at z_c; E(0) = 1 (Omega_Lambda from closure)
//   * Background: rho', rho'' vs finite differences (the stored field and its derivative agree); the
//     Klein-Gordon equation holds along the stored solution
//   * Oscillating phase: the energy redshifts as a^{-6n/(n+1)} (w -> (n-1)/(n+1)) for two n = 3 models
//     (for larger n the frequency drops with the amplitude and the average is reached only slowly)
//   * Perturbations: the Einstein constraints that are not evolved (Poisson, momentum) hold with the field's
//     density and momentum, around and after z_c
//   * Small f_EDE approaches LCDM
// (python/compare_class_ede.py compares with CLASS_EDE of Hill et al. 2020.)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/EarlyDarkEnergy.h>

#include <cmath>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    CosmologyParameters ede(double f, double log10zc, double theta, double n) {
        CosmologyParameters p;
        p.model = "ede";
        p.model_params = {{"ede_fEDE", f}, {"ede_log10zc", log10zc}, {"ede_theta_i", theta}, {"ede_n", n}};
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

    const EarlyDarkEnergySpecies & field(const BackgroundCosmology & b) {
        for (const auto & s : b.species())
            if (auto e = dynamic_cast<const EarlyDarkEnergySpecies *>(s.get()))
                return *e;
        throw std::runtime_error("no early dark energy");
    }
} // namespace

int main() {
    std::cout << std::unitbuf;
    std::cout << "Background (f_EDE = 0.1, log10 z_c = 3.5, theta_i = 2.8, n = 3)\n";
    BackgroundCosmology b(ede(0.1, 3.5, 2.8, 3.0));
    b.solve();
    const auto & e = field(b);
    std::cout << "  " << e.info() << "\n";
    {
        double fm, xp;
        e.peak(fm, xp);
        check_rel("max rho_EDE/rho_tot = f_EDE", fm, 0.1, 1e-9);
        check_close("at z_c", std::log10(std::exp(-xp) - 1.0), 3.5, 1e-8);
        check_close("E(0) = 1", b.E2_of_x(0.0), 1.0, 1e-12);
        double e1 = 0.0, e2 = 0.0, ekg = 0.0;
        for (double x : {-10.0, -8.3, -8.06, -7.9, -7.5, -6.0, -3.0}) {
            const double h = 1e-4;
            e1 = std::max(e1, std::fabs((e.rho(x + h) - e.rho(x - h)) / (2 * h) - e.drhodx(x)) / e.rho(x));
            e2 = std::max(e2, std::fabs((e.drhodx(x + h) - e.drhodx(x - h)) / (2 * h) - e.ddrhoddx(x)) / e.rho(x));
            // the stored phi_x is the derivative of the stored phi, and phi_xx of the Klein-Gordon equation that of
            // phi_x
            ekg = std::max(ekg, std::fabs((e.phi(x + h) - e.phi(x - h)) / (2 * h) - e.dphidx(x)) / e.f());
            ekg = std::max(ekg,
                           std::fabs((e.dphidx(x + h) - e.dphidx(x - h)) / (2 * h) - e.ddphidx(x)) /
                               (e.f() * e.mass_over_H0() / std::sqrt(b.E2_of_x(x))));
        }
        check_close("rho', rho'' vs finite differences", std::max(e1, e2), 0.0, 1e-4);
        check_close("stored solution satisfies the Klein-Gordon equation", ekg, 0.0, 1e-5);
        // energy redshifting in the oscillating phase: slope of ln rho over many oscillations
        const double slope = (std::log(e.rho(-4.0)) - std::log(e.rho(-6.0))) / 2.0;
        std::cout << "  dln rho/dln a over x = -6..-4: " << slope << " (oscillation average -6n/(n+1) = -4.5)\n";
        check_close("w -> (n-1)/(n+1) = 1/2 (n = 3)", slope / -4.5, 1.0, 0.03);
        // slow oscillations (theta_i = 2, z_c = 1e4: a few per e-fold), averaged over x = -8..-3
        BackgroundCosmology b2(ede(0.05, 4.0, 2.0, 3.0));
        const auto & e2s = field(b2);
        const double slope2 = (std::log(e2s.rho(-3.0)) - std::log(e2s.rho(-8.0))) / 5.0;
        std::cout << "  theta_i = 2, z_c = 1e4: " << slope2 << "\n";
        check_close("w -> 1/2 (theta_i = 2, z_c = 1e4)", slope2 / -4.5, 1.0, 0.03);
        check("n <= 2 rejected", [] {
            try {
                BackgroundCosmology bb(ede(0.05, 3.5, 2.0, 2.0));
            } catch (const std::invalid_argument &) {
                return true;
            }
            return false;
        }());
    }

    PerturbationOptions o;
    o.n_k = 4;
    o.k_min = 0.002 / Mpc;
    o.k_max = 0.2 / Mpc;
    std::cout << "Perturbations: Einstein constraints with the field\n";
    Run r(ede(0.1, 3.5, 2.8, 3.0), o);
    {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        const auto & f = field(cs);
        double eP = 0.0, eM = 0.0, frac = 0.0;
        for (double k : pt.get_k_array())
            for (double x : {-8.5, -8.06, -7.5, -6.5}) {
                const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
                const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
                const double dphi = pt.get("delta_phi", x, k), dfield = H0a2 * f.rho(x) * pt.get("delta_de", x, k);
                const double ffield = ckH * Hp * Hp * f.dphidx(x) * dphi / 3.0;
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
                // Poisson: k^2 Phi = (3/2) D (comoving density); momentum: Phi' - Psi = -(3/2) flux/(ck Hp)
                eP = std::max(eP, std::fabs(c * c * k * k * pt.get_Phi(x, k) - 1.5 * D) / std::fabs(1.5 * drho));
                eM = std::max(
                    eM,
                    std::fabs((pt.get_dPhidx(x, k) - pt.get_Psi(x, k)) / (-1.5 * flux / (ckH * Hp * Hp)) - 1.0));
            }
        std::cout << "  field share of the comoving density perturbation: up to " << frac << "\n";
        check_close("Poisson constraint (relative to delta rho)", eP, 0.0, 5e-6);
        check_close("momentum constraint", eM, 0.0, 1e-5);
        check("the field's perturbations are tested (> 1%)", frac > 0.01);
    }

    std::cout << "Small f_EDE\n";
    {
        PerturbationOptions o2 = o;
        o2.n_k = 3;
        Run s(ede(1e-5, 3.5, 2.8, 3.0), o2), l(CosmologyParameters(), o2);
        double e = 0.0;
        for (double k : s.pert.get_k_array())
            e = std::max(e, std::fabs(s.pert.get_delta_cdm(0.0, k) / l.pert.get_delta_cdm(0.0, k) - 1.0));
        check_close("f_EDE = 1e-5: delta_cdm today = LCDM (to 1e-4)", e, 0.0, 1e-4);
    }

    return Test::summary();
}
