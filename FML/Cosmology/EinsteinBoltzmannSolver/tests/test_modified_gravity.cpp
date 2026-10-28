// =============================================================================
// Tests of the modified-gravity models (mu-Sigma, DGP). Run with `make test`.
//
//   * GR limit: the model with GR parameter values reproduces LCDM
//   * Constraints: Phi and Psi agree with the algebraic mu-Sigma relations evaluated with the comoving
//     density of all species (checks the evolution equation for Phi_GR)
//   * Growth: the change of delta_cdm on sub-horizon scales agrees with the growth equation
//       D'' + (2 + dln H/dx) D' = (3/2) mu Omega_m(x) D,
//     solved independently here
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/DGP.h>
#include <FML/Cosmology/Models/HuSawicky.h>
#include <FML/Cosmology/Models/MuSigma.h>

#include <cmath>
#include <functional>
#include <memory>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    bool throws(const std::function<void()> & f) {
        try {
            f();
        } catch (const std::exception &) {
            return true;
        }
        return false;
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

    PerturbationOptions options() {
        PerturbationOptions o;
        o.n_k = 5;
        o.k_min = 0.01 / Mpc;
        o.k_max = 0.2 / Mpc;
        return o;
    }

    CosmologyParameters model(const std::string & name, std::map<std::string, double> mp) {
        CosmologyParameters p;
        p.model = name;
        p.model_params = mp;
        return p;
    }

    // Comoving density H0^2 a^2 rho Delta and shear of all species (photons, baryons, CDM, massless neutrinos)
    void comoving_density(const Run & r, double x, double k, double & D, double & S) {
        const auto & cs = r.cosmo;
        const auto & pt = r.pert;
        const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), ckH = c * k / cs.Hp_of_x(x);
        const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x);
        const double drho = rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) + 4.0 * rg * pt.get_Theta(0, x, k) +
                            4.0 * rn * pt.get_Nu(0, x, k);
        const double flux = -rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) + 4.0 * rg * pt.get_Theta(1, x, k) +
                            4.0 * rn * pt.get_Nu(1, x, k);
        D = H0a2 * (drho + 3.0 * flux / ckH);
        S = H0a2 * (4.0 * rg * pt.get_Theta(2, x, k) + 4.0 * rn * pt.get_Nu(2, x, k));
    }

    // Linear growth factor at x = 0 from the growth equation with mu(x), started in matter domination
    double growth(const BackgroundCosmology & cs, const std::function<double(double)> & mu) {
        const double x0 = -6.0, x1 = 0.0;
        const int n = 60000;
        const double h = (x1 - x0) / n;
        double D = std::exp(x0), dD = D; // D ~ a
        auto acc = [&](double x, double D, double dD) {
            const double E2 = cs.E2_of_x(x), dlnH = 0.5 * cs.dE2dx_of_x(x) / E2;
            const double Om = (cs.rhoCDM(x) + cs.rhoB(x)) / E2;
            return -(2.0 + dlnH) * dD + 1.5 * mu(x) * Om * D;
        };
        for (int i = 0; i < n; i++) { // RK4
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
    const PerturbationOptions o = options();
    Run lcdm(CosmologyParameters(), o);
    const auto & ks = lcdm.pert.get_k_array();

    std::cout << "mu-Sigma: GR limit\n";
    {
        Run ms(model("musigma", {}), o);
        double e = 0.0, eS = 0.0;
        for (double k : ks)
            for (double x : {-12.0, -7.0, -3.0, 0.0}) {
                e = std::max(e, std::fabs(ms.pert.get_Phi(x, k) / lcdm.pert.get_Phi(x, k) - 1.0));
                e = std::max(e, std::fabs(ms.pert.get_Psi(x, k) / lcdm.pert.get_Psi(x, k) - 1.0));
                e = std::max(e, std::fabs(ms.pert.get_delta_cdm(x, k) / lcdm.pert.get_delta_cdm(x, k) - 1.0));
                eS = std::max(eS,
                              std::fabs(ms.pert.get_Source_T(x, k) - lcdm.pert.get_Source_T(x, k)) /
                                  std::fabs(lcdm.pert.get_Theta(0, x, k) + lcdm.pert.get_Psi(x, k)));
            }
        check_close("mu0 = Sigma0 = 0: Phi, Psi, delta_cdm = LCDM", e, 0.0, 1e-9);
        check_close("mu0 = Sigma0 = 0: S_T = LCDM", eS, 0.0, 1e-7);
    }

    std::cout << "mu-Sigma: modified potentials and growth\n";
    {
        const double mu0 = 0.4, Sigma0 = 0.2;
        Run ms(model("musigma", {{"mu0", mu0}, {"Sigma0", Sigma0}}), o);
        MuSigmaGravity g(mu0, Sigma0);
        double ePhi = 0.0, ePsi = 0.0;
        for (double k : ks)
            for (double x : {-2.0, -1.0, 0.0}) {
                double D, S;
                comoving_density(ms, x, k, D, S);
                const auto m = g.mu_sigma(x, 0.0, ms.cosmo);
                const double PhiGR = 1.5 * D / (c * c * k * k);
                ePhi = std::max(ePhi, std::fabs(ms.pert.get_Phi(x, k) / ((2.0 * m.Sigma - m.mu) * PhiGR) - 1.0));
                ePsi = std::max(ePsi,
                                std::fabs(ms.pert.get_Psi(x, k) / (-m.mu * PhiGR - 3.0 * S / (c * c * k * k)) - 1.0));
            }
        check_close("Phi = (2 Sigma - mu) Phi_GR(rho Delta)", ePhi, 0.0, 1e-4);
        check_close("Psi = -mu Phi_GR(rho Delta) - shear", ePsi, 0.0, 1e-4);
        check_rel("mu(x=0) = 1 + mu0", g.mu_sigma(0.0, 0.0, ms.cosmo).mu, 1.0 + mu0, 1e-12);

        const double k = ks.back(); // 0.2/Mpc: sub-horizon, quasi-static
        const double ratio = ms.pert.get_delta_cdm(0.0, k) / lcdm.pert.get_delta_cdm(0.0, k);
        const double ref = growth(ms.cosmo, [&](double x) { return g.mu_sigma(x, 0.0, ms.cosmo).mu; }) /
                           growth(lcdm.cosmo, [](double) { return 1.0; });
        std::cout << "  delta_cdm(mu0 = " << mu0 << ") / LCDM = " << ratio << ", growth equation: " << ref << "\n";
        check_close("growth: delta_cdm ratio = growth equation", ratio / ref, 1.0, 3e-4);

        // Sigma only: no change of the growth, Weyl potential scaled by Sigma(0) = 1 + Sigma0
        Run s(model("musigma", {{"Sigma0", Sigma0}}), o);
        check_close("Sigma0 only: delta_cdm unchanged",
                    s.pert.get_delta_cdm(0.0, k) / lcdm.pert.get_delta_cdm(0.0, k),
                    1.0,
                    1e-5);
        const double weyl =
            (s.pert.get_Psi(0.0, k) - s.pert.get_Phi(0.0, k)) / (lcdm.pert.get_Psi(0.0, k) - lcdm.pert.get_Phi(0.0, k));
        check_close("Sigma0 only: (Psi - Phi) / LCDM = 1 + Sigma0 today", weyl, 1.0 + Sigma0, 1e-5);
    }

    std::cout << "mu-Sigma: errors\n";
    {
        check("1 + 2 Sigma0 - mu0 < 0 is an error",
              throws([] { BackgroundCosmology b(model("musigma", {{"mu0", 1.5}, {"Sigma0", 0.1}})); }));
        CosmologyParameters pk = model("musigma", {{"mu0", 0.1}});
        pk.OmegaK = 0.01;
        check("curvature is an error", throws([&] { BackgroundCosmology b(pk); }));
    }

    std::cout << "DGP: background\n";
    {
        // sDGP without dark energy: H^2 - H c/r_c = 8 pi G rho/3, i.e. E^2 - 2 sqrt(Omega_rc) E = rho
        CosmologyParameters p = model("dgp", {{"Omega_rc", 0.0}, {"dgp_branch", 1.0}});
        BackgroundCosmology b(p);
        b.solve();
        const double sq = 0.5 * (1.0 - (b.rho_species(0.0) - b.rhoDE(0.0))); // 1 - 2 sqrt(Omega_rc) = rho(0)
        double res = 0.0, eH = 0.0, eHH = 0.0;
        for (double x = -12.0; x <= 1.0; x += 0.5) {
            const double E = std::sqrt(b.E2_of_x(x));
            res = std::max(res, std::fabs(E * E - 2.0 * sq * E - b.rho_species(x)) / (E * E));
            const double h = 1e-4;
            eH = std::max(eH, std::fabs((b.Hp_of_x(x + h) - b.Hp_of_x(x - h)) / (2 * h) / b.dHpdx_of_x(x) - 1.0));
            eHH = std::max(eHH,
                           std::fabs((b.dHpdx_of_x(x + h) - b.dHpdx_of_x(x - h)) / (2 * h) - b.ddHpddx_of_x(x)) /
                               std::fabs(b.Hp_of_x(x)));
        }
        check_close("sDGP: no dark energy", b.rhoDE(0.0), 0.0, 1e-14);
        check_close("sDGP: E^2 - 2 sqrt(Omega_rc) E = rho", res, 0.0, 1e-13);
        check_close("sDGP: dHp/dx vs finite differences", eH, 0.0, 1e-7);
        check_close("sDGP: d2Hp/dx2 vs finite differences", eHH, 0.0, 1e-7);
        // eta0 by quadrature of the DGP expansion rate
        const double eta0 = Utils::integrate(
            [&](double u) {
                const double a = u * u, x = std::log(a);
                const double E = sq + std::sqrt(sq * sq + b.rho_species(x));
                return 2.0 * u * c / (a * a * b.get_H0() * E);
            },
            0.0,
            1.0,
            0.0,
            1e-12);
        check_rel("sDGP: eta0 vs quadrature", b.eta0(), eta0, 1e-7);
        check("sDGP accelerates today", b.dHpdx_of_x(0.0) > 0.0);
    }

    std::cout << "DGP: perturbations\n";
    for (const auto & [label, mp] : std::vector<std::pair<std::string, std::map<std::string, double>>>{
             {"nDGP (H0 r_c = 1)", {{"Omega_rc", 0.25}}},
             {"nDGP, LCDM background", {{"Omega_rc", 0.25}, {"dgp_lcdm_background", 1.0}}},
             {"sDGP", {{"Omega_rc", 0.0}, {"dgp_branch", 1.0}}}}) {
        Run r(model("dgp", mp), o);
        const auto & g = dynamic_cast<const DGPGravity &>(r.cosmo.gravity());
        // beta from the definition, with dlnH/dx by finite differences
        const double Orc = mp.at("Omega_rc") > 0.0 ? mp.at("Omega_rc") : -1.0;
        double emu = 0.0;
        for (double x : {-3.0, -1.0, 0.0}) {
            const double h = 1e-5, E = std::sqrt(r.cosmo.E2_of_x(x));
            const double dlnE = (std::log(r.cosmo.E2_of_x(x + h)) - std::log(r.cosmo.E2_of_x(x - h))) / (4.0 * h);
            const int eps = mp.count("dgp_branch") ? int(mp.at("dgp_branch")) : -1;
            const double O =
                Orc > 0.0 ? Orc : 0.25 * std::pow(1.0 - (r.cosmo.rho_species(0.0) - r.cosmo.rhoDE(0.0)), 2);
            const double beta = 1.0 - eps * E * (1.0 + dlnE / 3.0) / std::sqrt(O);
            emu = std::max(emu, std::fabs(g.mu_sigma(x, 0.0, r.cosmo).mu - (1.0 + 1.0 / (3.0 * beta))));
        }
        check_close(label + ": mu = 1 + 1/(3 beta)", emu, 0.0, 1e-8);

        double eW = 0.0;
        for (double k : ks)
            for (double x : {-1.0, 0.0}) {
                double D, S;
                comoving_density(r, x, k, D, S);
                const double PhiGR = 1.5 * D / (c * c * k * k);
                eW = std::max(eW,
                              std::fabs((r.pert.get_Psi(x, k) - r.pert.get_Phi(x, k)) /
                                            (-2.0 * PhiGR - 3.0 * S / (c * c * k * k)) -
                                        1.0));
            }
        check_close(label + ": Sigma = 1 (Psi - Phi from rho Delta as in GR)", eW, 0.0, 1e-4);

        const double k = ks.back();
        const double ratio = r.pert.get_delta_cdm(0.0, k) / lcdm.pert.get_delta_cdm(0.0, k);
        const double ref = growth(r.cosmo, [&](double x) { return g.mu_sigma(x, 0.0, r.cosmo).mu; }) /
                           growth(lcdm.cosmo, [](double) { return 1.0; });
        std::cout << "  " << label << ": mu(0) = " << g.mu_sigma(0.0, 0.0, r.cosmo).mu
                  << ", delta_cdm / LCDM = " << ratio << ", growth equation: " << ref << "\n";
        check_close(label + ": growth = growth equation", ratio / ref, 1.0, 3e-4);
    }

    std::cout << "Hu-Sawicky f(R)\n";
    {
        Run r0(model("fr", {{"fR0", 0.0}}), o);
        double e = 0.0;
        for (double k : ks)
            for (double x : {-7.0, -2.0, 0.0})
                e = std::max(e, std::fabs(r0.pert.get_Phi(x, k) / lcdm.pert.get_Phi(x, k) - 1.0));
        check_close("|f_R0| = 0: Phi = LCDM", e, 0.0, 1e-9);

        const double fR0 = 1e-4;
        PerturbationOptions of = o;
        of.k_max = 0.5 / Mpc;
        Run r(model("fr", {{"fR0", fR0}}), of), l(CosmologyParameters(), of);
        const auto & g = dynamic_cast<const HuSawickyGravity &>(r.cosmo.gravity());
        const auto & cs = r.cosmo;
        // f_R(0) and the Compton wavelength today: lambda_C^2 = 3 (n+1) |f_R0| / (R0 (1 - |f_R0|)), R0 = 3 H0^2 (rho -
        // 3p)(0)
        const double R0 = 3.0 * std::pow(cs.get_H0() / c, 2) * (cs.rhoB(0.0) + cs.rhoCDM(0.0) + 4.0 * cs.rhoDE(0.0));
        check_rel("f_R(0) = -|f_R0|", g.fR(0.0, cs), -fR0, 1e-14);
        check_rel("Compton wavelength today",
                  g.compton_wavelength(0.0, cs),
                  std::sqrt(6.0 * fR0 / (R0 * (1.0 - fR0))),
                  1e-10);
        check_rel("f_R ~ a^{-3(n+1)} in the matter era", g.fR(-6.0, cs) / g.fR(-5.0, cs), std::exp(-6.0), 1e-5);
        check_rel("mu -> 4/3 (1 + f_R)^-1 for k >> 1/lambda_C",
                  g.mu_sigma(0.0, 1e3 / Mpc, cs).mu,
                  4.0 / 3.0 / (1.0 - fR0),
                  1e-5);
        check_rel("mu -> (1 + f_R)^-1 for k << 1/lambda_C",
                  g.mu_sigma(0.0, 1e-5 / Mpc, cs).mu,
                  1.0 / (1.0 - fR0),
                  1e-6);

        double ePhi = 0.0, ePsi = 0.0;
        for (double k : r.pert.get_k_array())
            for (double x : {-1.0, 0.0}) {
                double D, S;
                comoving_density(r, x, k, D, S);
                const auto m = g.mu_sigma(x, k, cs);
                const double PhiGR = 1.5 * D / (c * c * k * k);
                ePhi = std::max(ePhi, std::fabs(r.pert.get_Phi(x, k) / ((2.0 * m.Sigma - m.mu) * PhiGR) - 1.0));
                ePsi =
                    std::max(ePsi, std::fabs(r.pert.get_Psi(x, k) / (-m.mu * PhiGR - 3.0 * S / (c * c * k * k)) - 1.0));
            }
        check_close("f(R): Phi = (2 Sigma - mu) Phi_GR(rho Delta) for every k", ePhi, 0.0, 1e-4);
        check_close("f(R): Psi = -mu Phi_GR(rho Delta) - shear for every k", ePsi, 0.0, 1e-4);

        // Scale-dependent growth: each k (grid nodes, no interpolation in k) against the growth equation with mu(k, a)
        for (size_t ik = 2; ik < r.pert.get_k_array().size(); ik++) {
            const double k = r.pert.get_k_array()[ik], kM = k * Mpc;
            const double ratio = r.pert.get_delta_cdm(0.0, k) / l.pert.get_delta_cdm(0.0, k);
            const double ref = growth(cs, [&](double x) { return g.mu_sigma(x, k, cs).mu; }) /
                               growth(l.cosmo, [](double) { return 1.0; });
            std::cout << "  |f_R0| = 1e-4, k = " << kM << "/Mpc: delta_cdm / LCDM = " << ratio
                      << ", growth equation: " << ref << "\n";
            check_close("f(R) growth at k = " + std::to_string(kM).substr(0, 5) + "/Mpc = growth equation",
                        ratio / ref,
                        1.0,
                        3e-4);
        }
    }

    return Test::summary();
}
