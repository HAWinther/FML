// =============================================================================
// Tests of the isocurvature modes (Isocurvature.h, Perturbations::isocurvature_values). Run with `make test`.
//
//   * Decomposition into independent components: sum_j u_j u_j^T reproduces P_RR, P_II, P_RI = c sqrt(P_RR P_II)
//     and P_IJ = 0
//   * Initial conditions: entropy S_c = delta_c - 3/4 delta_gamma = -1 for CDI (CLASS's entropy_ini = -1 relative
//     to our adiabatic normalization), neutrino density contrast for NID; superhorizon Phi << 1
//   * The modes do not depend on the initial time (x_start = -18 vs -16): the initial conditions are on the
//     regular attractor
//   * C_ell against CLASS v3.4 (ic = ad,cdi and ad,niv; f = 1, n = 1.1, c_ad = 0.3; Planck 2018, unlensed), CLASS
//     run with l_max_scalars = 1500 and fine k/ell sampling (with its defaults the NIV mode is off by ~1-3%, and C_ell
//     within ~100 of l_max_scalars are cut by its k_max). Pure modes (ic_adiabatic = 0):
//     CDI and NID agree to < 5e-4 for 10 <= ell <= 600, NIV to ~1e-3
//   * Isocurvature modes are refused for modified gravity and inconsistent correlations
//   * The CDI mode in the models with their own dark-matter block (dark_scattering, ide, coupled_quintessence)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Isocurvature.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
} // namespace

int main() {
    Utils::verbose = false;

    std::cout << "Decomposition into independent components\n";
    {
        RunSettings s = ParameterFile::read(
            "",
            {"preset=planck2018", "f_cdi=0.5", "n_cdi=1.2", "c_ad_cdi=0.4", "f_nid=0.3", "n_nid=0.9", "c_ad_nid=-0.6"});
        const auto comps = Isocurvature::components(s.cosmo, s.pert);
        check("three components (adiabatic + two modes)", comps.size() == 3);
        const double k = 0.01 / Mpc, kp = s.cosmo.k_pivot, ns = s.cosmo.n_s;
        double PRR = 0, PCC = 0, PNN = 0, PRC = 0, PRN = 0, PCN = 0;
        for (const auto & o : comps) {
            const double uR = o.ic_adiabatic;
            const double uC = o.iso_amplitude[0] * std::pow(k * Mpc / kp, o.iso_tilt[0]);
            const double uN = o.iso_amplitude[2] * std::pow(k * Mpc / kp, o.iso_tilt[2]);
            PRR += uR * uR;
            PCC += uC * uC;
            PNN += uN * uN;
            PRC += uR * uC;
            PRN += uR * uN;
            PCN += uC * uN;
        }
        // relative to P_RR(k): P_II/P_RR = f^2 (k/kp)^(n - n_s)
        const double rC = 0.25 * std::pow(k * Mpc / kp, 1.2 - ns), rN = 0.09 * std::pow(k * Mpc / kp, 0.9 - ns);
        check_close("P_RR", PRR, 1.0, 1e-14);
        check_rel("P_cdi", PCC, rC, 1e-13);
        check_rel("P_nid", PNN, rN, 1e-13);
        check_rel("P_R,cdi = c sqrt(P_RR P_cdi)", PRC, 0.4 * std::sqrt(rC), 1e-13);
        check_rel("P_R,nid = c sqrt(P_RR P_nid)", PRN, -0.6 * std::sqrt(rN), 1e-13);
        check_close("P_cdi,nid = 0", PCN, 0.0, 1e-14);
        check("sum c^2 >= 1 is an error",
              throws([] { ParameterFile::read("", {"f_cdi=1", "c_ad_cdi=0.8", "f_bi=1", "c_ad_bi=0.7"}); }));
        check("isocurvature with modified gravity is an error", throws([] {
                  RunSettings m = ParameterFile::read("", {"model=jbd", "f_cdi=1"});
                  BackgroundCosmology bg(m.cosmo);
                  bg.solve();
                  RecombinationHistory rec(bg);
                  Perturbations p(bg, rec, Isocurvature::components(m.cosmo, m.pert)[1]);
              }));
    }

    std::cout << "CDM isocurvature in the models with their own dark-matter block\n";
    {
        // dark_scattering and ide with zero coupling are the same wCDM (w = -0.95, c_s^2 = 1), and
        // coupled_quintessence with beta = 0 and a flat potential is close to LCDM: the CDI mode must be there and
        // agree (it was silently replaced by the adiabatic form, i.e. lost)
        auto cdi = [](const std::vector<std::string> & args) {
            std::vector<std::string> a = {"precision=fast"};
            a.insert(a.end(), args.begin(), args.end());
            RunSettings s = ParameterFile::read("", a);
            auto bg = std::make_unique<BackgroundCosmology>(s.cosmo);
            bg->solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
            auto rec = std::make_unique<RecombinationHistory>(*bg, s.rec);
            rec->solve();
            PerturbationOptions o = s.pert;
            o.ic_adiabatic = 0.0;
            o.iso_amplitude[0] = 1.0;
            o.k_max = 0.1 / Mpc;
            o.n_k = 20;
            o.dk_max = 0.0;
            Perturbations p(*bg, *rec, o);
            p.solve();
            std::vector<double> d;
            for (double k : {0.001 / Mpc, 0.01 / Mpc, 0.05 / Mpc})
                for (double x : {-7.0, 0.0})
                    d.push_back(p.get_delta_cdm(x, k));
            return d;
        };
        const auto lcdm = cdi({});
        const auto ds = cdi({"model=dark_scattering", "ds_xi=0", "ds_w=-0.95"});
        const auto ide = cdi({"model=ide", "ide_xi=0", "ide_w=-0.95"});
        const auto cq = cdi({"model=coupled_quintessence", "cq_beta=0", "cq_lambda=1e-3"});
        double e_ds_ide = 0.0, e_ds_lcdm = 0.0, e_cq_lcdm = 0.0;
        for (size_t i = 0; i < lcdm.size(); i++) {
            e_ds_ide = std::max(e_ds_ide, std::fabs(ds[i] / ide[i] - 1.0));
            e_ds_lcdm = std::max(e_ds_lcdm, std::fabs(ds[i] / lcdm[i] - 1.0));
            e_cq_lcdm = std::max(e_cq_lcdm, std::fabs(cq[i] / lcdm[i] - 1.0));
        }
        check_close("CDI: dark_scattering (xi = 0) = ide (xi = 0)", e_ds_ide, 0.0, 1e-4);
        check_close("CDI: dark_scattering (xi = 0, w = -0.95) ~ LCDM", e_ds_lcdm, 0.0, 0.1);
        check_close("CDI: coupled_quintessence (beta = 0, lambda = 1e-3) = LCDM", e_cq_lcdm, 0.0, 1e-4);
    }

    std::cout << "Initial conditions and attractor\n";
    {
        RunSettings s = ParameterFile::read("", {"preset=planck2018", "precision=fast"});
        BackgroundCosmology bg(s.cosmo);
        bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
        RecombinationHistory rec(bg, s.rec);
        rec.solve();
        auto solve = [&](int m, double x_start) {
            PerturbationOptions o = s.pert;
            o.ic_adiabatic = 0.0;
            o.iso_amplitude[m] = 1.0;
            o.x_start = x_start;
            o.k_max = 0.1 / Mpc;
            o.n_k = 30;
            o.dk_max = 0.0;
            auto p = std::make_unique<Perturbations>(bg, rec, o);
            p->solve();
            return p;
        };
        for (int m : {0, 2, 3}) {
            const std::string n = Isocurvature::mode_names[m];
            const auto p1 = solve(m, -18.0), p2 = solve(m, -16.0);
            const double k = p1->get_k_array()[20], x0 = -18.0;
            if (m == 0) {
                const double S = p1->get_delta_cdm(x0, k) - 3.0 * p1->get_Theta(0, x0, k); // delta_gamma = 4 Theta_0
                check_close(n + ": entropy delta_c - 3/4 delta_gamma = -1 initially", S, -1.0, 1e-4);
                check("... Phi << 1 initially", std::fabs(p1->get_Phi(x0, k)) < 1e-4);
            }
            // delta_cdm and Phi at recombination, z ~ 20 and today, Theta_0 at recombination, relative to max(|value|,
            // 0.1) (Phi of the NID mode crosses zero near recombination). The neutrino modes keep a small x_start
            // dependence (3e-4 for NID, 5e-4 for NIV between -18 and -16; it does not show in the C_ell at the 1e-4
            // level)
            double worst = 0.0;
            auto rel = [&](double u, double v) { return std::fabs(u - v) / std::max(std::fabs(u), 0.1); };
            for (double x : {-7.0, -3.0, 0.0}) {
                worst = std::max(worst, rel(p1->get_delta_cdm(x, k), p2->get_delta_cdm(x, k)));
                worst = std::max(worst, rel(p1->get_Phi(x, k), p2->get_Phi(x, k)));
            }
            worst = std::max(worst, rel(p1->get_Theta(0, -7.0, k), p2->get_Theta(0, -7.0, k)));
            const bool nu = m >= 2;
            check(n + ": independent of the initial time (x_start = -18 vs -16, < " + (nu ? "1e-3" : "1e-6") + ")",
                  worst < (nu ? 1e-3 : 1e-6));
        }
    }

    std::cout << "C_ell against CLASS (f = 1, n = 1.1, c_ad = 0.3; Planck 2018, unlensed, precision = fast)\n";
    {
        struct Ref {
            const char * mode;
            double tt10, tt100, tt500, ee100, ee500, te100;
        };
        for (const Ref & r : {Ref{"cdi", 1293.16, 2986.48, 2533.26, 0.716171, 8.12231, -18.8993},
                              Ref{"niv", 1032.84, 2791.04, 2436.66, 0.727323, 9.34012, -22.4623}}) {
            const std::string m = r.mode;
            RunSettings s = ParameterFile::read("",
                                                {"preset=planck2018",
                                                 "precision=fast",
                                                 "f_" + m + "=1",
                                                 "n_" + m + "=1.1",
                                                 "c_ad_" + m + "=0.3",
                                                 "ell_max=600"});
            BackgroundCosmology bg(s.cosmo);
            bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
            RecombinationHistory rec(bg, s.rec);
            rec.solve();
            const auto opts = Isocurvature::components(s.cosmo, s.pert);
            std::vector<std::unique_ptr<Perturbations>> ps;
            for (const auto & o : opts) {
                ps.push_back(std::make_unique<Perturbations>(bg, rec, o));
                ps.back()->solve();
            }
            PowerSpectrum P(bg, rec, *ps[0], s.ps);
            for (size_t i = 1; i < ps.size(); i++)
                P.add_component(*ps[i]);
            P.solve();
            check_rel(m + ": TT ell = 10", P.get_Dell_TT(10), r.tt10, 5e-3);
            check_rel(m + ": TT ell = 100", P.get_Dell_TT(100), r.tt100, 3e-3);
            check_rel(m + ": TT ell = 500", P.get_Dell_TT(500), r.tt500, 3e-3);
            check_rel(m + ": EE ell = 100", P.get_Dell_EE(100), r.ee100, 5e-3);
            check_rel(m + ": EE ell = 500", P.get_Dell_EE(500), r.ee500, 5e-3);
            check_rel(m + ": TE ell = 100", P.get_Dell_TE(100), r.te100, 5e-3);
        }
    }

    return Test::summary();
}
