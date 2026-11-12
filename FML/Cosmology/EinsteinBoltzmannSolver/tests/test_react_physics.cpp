// =============================================================================
// Tests of the solver's nonlinear = react (Cosmology/Spectra/PowerSpectrumReACT.cpp, the native halo-model reaction)
// with non-trivial reactions. Several minutes on 2 cores (14 Boltzmann solves, precision = fast).
//
//   * Composition: the solver's P_NL for f(R) = R x P_NL,pseudo rebuilt independently here (HaloModelReaction and
//     MGOneLoop directly at z = 0 and 1, not through the solver's table; HMcode-2020 of the matched pseudo cosmology
//     evaluated directly in h/Mpc). Catches unit, ordering and interpolation errors that the massless-GR identity of
//     test_nonlinear (R = 1) cannot.
//   * f(R): the enhancement P_f(R)/P_LCDM (same initial conditions) against the fitting function of Winther et al.
//     2019 (Emulator/FofrFittingFunction.h, calibrated on the ELEPHANT N-body simulations), for |fR0| = 1e-6, 1e-5
//     in their cosmology (measured: 2-3% for k <= 1 h/Mpc, up to 7% at k = 3, z = 0 for 1e-5)
//   * Limits: R -> 1 for |fR0| -> 0 and Omega_rc -> 0 (HaloModelReaction), and on large scales
//   * w0wa and LCDM with massive neutrinos: ReACT x matched pseudo against HMcode-2020 run directly (both include
//     these physics; they agree to the accuracy of the two halo models)
//   * A_s = 3.2e-9 (outside the range the ReACT code accepted) works
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Emulator/FofrFittingFunction.h>
#include <FML/HaloModel/HMcode2020.h>
#include <FML/HaloModel/HaloModelReaction.h>
#include <FML/PerturbationTheory/MGOneLoop.h>
#include <FML/SphericalCollapse/ScreenedSphericalCollapse.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;
using namespace Constants;
using FML::TESTS::check;
using Nonlinear = PowerSpectrumOptions::Nonlinear;

namespace {
    // One linear cosmology (background, recombination, perturbations) and its power spectrum
    struct Run {
        RunSettings s;
        std::unique_ptr<BackgroundCosmology> bg;
        std::unique_ptr<RecombinationHistory> rec;
        std::unique_ptr<Perturbations> pert;
        std::unique_ptr<PowerSpectrum> ps;
        double V{}; // (Mpc/h)^3 in SI

        Run(std::vector<std::string> args, Nonlinear nl, int react_n_z = 8, int react_n_k = 128) {
            args.insert(args.begin(), {"precision=fast", "pert_k_max=1", "compute_cells=false"});
            s = ParameterFile::read("", args);
            bg = std::make_unique<BackgroundCosmology>(s.cosmo);
            bg->solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
            rec = std::make_unique<RecombinationHistory>(*bg, s.rec);
            rec->solve();
            pert = std::make_unique<Perturbations>(*bg, *rec, s.pert);
            pert->solve();
            auto o = s.ps;
            o.nonlinear = nl;
            o.react_n_z = react_n_z;
            o.react_n_k = react_n_k;
            ps = std::make_unique<PowerSpectrum>(*bg, *rec, *pert, o);
            ps->solve_matter();
            V = std::pow(Mpc / s.cosmo.h, 3);
        }
        double kSI(double kh) const { return kh * s.cosmo.h / Mpc; }
        // Nonlinear and linear P in (Mpc/h)^3 for k in h/Mpc
        double pnl(double z, double kh) const { return ps->get_matter_power_spectrum_nonlinear(-std::log1p(z), kSI(kh)) / V; }
        double plin(double z, double kh, bool cold = false) const {
            const double x = -std::log1p(z);
            return (cold ? ps->get_matter_power_spectrum_cb_extrapolated(x, kSI(kh))
                         : ps->get_matter_power_spectrum_extrapolated(x, kSI(kh))) / V;
        }
    };

    std::string fmt(double v, int p = 4) {
        std::ostringstream s;
        s << std::setprecision(p) << v;
        return s.str();
    }
} // namespace

int main() {
    Utils::verbose = false;
    namespace PT = FML::PERTURBATION;
    namespace SC = SPHERICALCOLLAPSE;
    using HALOMODEL::HaloModelReaction;

    // The ELEPHANT cosmology (WMAP9) of the f(R) fitting function, massless neutrinos
    const std::vector<std::string> elephant = {"h=0.6974", "OmegaB=0.046", "OmegaCDM=0.235", "n_s=0.971",
                                               "A_s=2.43e-9", "m_nu=0"};
    auto with = [](std::vector<std::string> a, const std::vector<std::string> & extra) {
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    };

    const Run lcdm(elephant, Nonlinear::HMcode2020);
    const Run fr5(with(elephant, {"model=fr", "fR0=1e-5", "fR_n=1"}), Nonlinear::ReACT);

    //=========================================================================
    std::cout << "Composition: solver P_NL(f(R)) = R (library) x P_NL,pseudo (HMcode-2020 of the pseudo cosmology)\n";
    //=========================================================================
    {
        // The reaction directly at z = 0 and 1 (the solver tabulates it at 8 redshifts and 128 k)
        const double Om = fr5.bg->rhoB(0) + fr5.bg->rhoCDM(0);
        auto spectra = [&](double z) {
            HaloModelReaction::LinearSpectra S;
            S.P = [&fr5, z](double kh) { return fr5.plin(z, kh); };
            S.Pcb = [&fr5, z](double kh) { return fr5.plin(z, kh, true); };
            S.Pcb_lcdm = [&lcdm, z](double kh) { return lcdm.plin(z, kh, true); };
            return S;
        };
        const std::vector<double> az = {0.5, 1.0};
        const auto Sz1 = spectra(1.0), Sz0 = spectra(0.0);
        auto Pi = [&](double q, size_t i, bool cb) { return (i == 0 ? (cb ? Sz1.Pcb : Sz1.P) : (cb ? Sz0.Pcb : Sz0.P))(q); };
        const auto lr = PT::mg_one_loop(PT::mg_hu_sawicki(1e-5, Om, Om), 0.06, az, [&](double q, size_t i) { return Pi(q, i, true); });
        const auto lp = PT::mg_one_loop(PT::mg_lcdm(Om, Om), 0.06, az, [&](double q, size_t i) { return Pi(q, i, false); });
        HaloModelReaction::Cosmology hc;
        hc.OmegaM = Om;
        auto F = [Om](double R) { return SC::fifth_force_hu_sawicki(1e-5, 1.0, Om, R); };
        HaloModelReaction hm1(hc, 0.5, Sz1, F), hm0(hc, 1.0, Sz0, F);
        hm1.set_one_loop(lr[0], lp[0]);
        hm0.set_one_loop(lr[1], lp[1]);

        // The pseudo cosmology: LCDM background, GR, the target's linear P
        auto pp = fr5.s.cosmo;
        pp.model = "lcdm";
        pp.model_params.clear();
        BackgroundCosmology pseudo_bg(pp);
        auto target = [&](double x, double k) { return fr5.ps->get_matter_power_spectrum_extrapolated(x, k); };

        double err = 0, rmax = 0; // max |R - 1|
        for (double z : {0.0, 1.0}) {
            const std::vector<double> ks = {0.05, 0.2, 0.5, 1.0, 3.0};
            const auto pseudo = HALOMODEL::HMcode2020::spectrum(pseudo_bg, target, target, -std::log1p(z), ks);
            for (size_t i = 0; i < ks.size(); i++) {
                const double R = (z == 0.0 ? hm0 : hm1).reaction(ks[i]);
                rmax = std::max(rmax, std::abs(R - 1));
                const double mine = fr5.pnl(z, ks[i]), indep = R * pseudo.nonlinear[i];
                err = std::max(err, std::abs(mine / indep - 1));
                std::cout << "  z = " << z << " k = " << ks[i] << " h/Mpc: R = " << fmt(R) << ", solver / (R x pseudo) - 1 = "
                          << fmt(mine / indep - 1, 2) << "\n";
            }
        }
        check(rmax > 0.05, "f(R) reaction is non-trivial (max |R - 1| = " + fmt(rmax) + ")");
        check(err < 3e-3, "solver P_NL = R x P_NL,pseudo built independently (max rel diff " + fmt(err, 2) + ")");
    }

    //=========================================================================
    std::cout << "f(R) enhancement against the Winther et al. (2019) N-body fitting function\n";
    //=========================================================================
    {
        const Run fr6(with(elephant, {"model=fr", "fR0=1e-6", "fR_n=1"}), Nonlinear::ReACT);
        for (const auto & [run, fR0] : {std::pair<const Run *, double>{&fr6, 1e-6}, {&fr5, 1e-5}}) {
            double e1 = 0, e3 = 0;
            for (double z : {0.0, 0.5, 1.0}) {
                std::cout << "  |fR0| = " << fR0 << ", z = " << z << ": B_ReACT / B_fit - 1 at k =";
                for (double k : {0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0}) {
                    const double B = run->pnl(z, k) / lcdm.pnl(z, k);
                    const double fit = FML::EMULATOR::FOFRFITTINGFUNCTION::nonlinear_boost(z, fR0, k);
                    const double d = B / fit - 1;
                    std::cout << " " << k << ": " << fmt(d, 2);
                    (k <= 1.0 ? e1 : e3) = std::max(k <= 1.0 ? e1 : e3, std::abs(d));
                }
                std::cout << "\n";
            }
            // Measured (Nov 2026): 2.3% (1e-6) and 2.9% (1e-5) for k <= 1; 1.2% and 6.7% (z = 0, k = 3) for k <= 3.
            // The fit itself is good to 1-3%. ReACT x HMcode-2020 dips 1-2% below 1 at k = 0.05-0.1 for 1e-6, where
            // the N-body enhancement is >= 1: HMcode-2020's response to the non-LCDM shape of the pseudo spectrum
            // (upstream's Halofit pseudo is worse, -6% at k = 0.1). These bounds guard against regressions.
            check(e1 < 0.035, "|fR0| = " + fmt(fR0) + ": enhancement = N-body fit to 3.5% for k <= 1 h/Mpc, z <= 1 (" +
                                  fmt(e1, 2) + ")");
            check(e3 < 0.08, "|fR0| = " + fmt(fR0) + ": enhancement = N-body fit to 8% for 1 < k <= 3 h/Mpc, z <= 1 (" +
                                 fmt(e3, 2) + ")");
        }
    }

    //=========================================================================
    std::cout << "Limits: R -> 1 for |fR0|, Omega_rc -> 0 and on large scales\n";
    //=========================================================================
    {
        // GR targets (the LCDM spectra) with a tiny fifth force: R = R_GR = 1
        const double Om = lcdm.bg->rhoB(0) + lcdm.bg->rhoCDM(0);
        HaloModelReaction::Cosmology hc;
        hc.OmegaM = Om;
        SC::SphericalCollapseModel bg;
        bg.OmegaM = Om;
        bg.set_model_lcdm();
        for (int model : {0, 1}) {
            const std::vector<double> az = {0.5, 1.0};
            auto P = [&](double q, size_t i) { return lcdm.plin(i == 0 ? 1.0 : 0.0, q); };
            const auto kern = model == 0 ? PT::mg_hu_sawicki(1e-10, Om, Om) : PT::mg_ndgp(1e-6, Om, Om);
            const auto lr = PT::mg_one_loop(kern, 0.06, az, P), lp = PT::mg_one_loop(PT::mg_lcdm(Om, Om), 0.06, az, P);
            HaloModelReaction::FifthForceOfRadius F;
            if (model == 0)
                F = [Om](double R) { return SC::fifth_force_hu_sawicki(1e-10, 1.0, Om, R); };
            else
                F = [bg](double) { return SC::fifth_force_ndgp(1e-6, bg); };
            double d = 0;
            for (size_t i = 0; i < az.size(); i++) {
                const double z = i == 0 ? 1.0 : 0.0;
                HaloModelReaction::LinearSpectra S;
                S.P = S.Pcb = S.Pcb_lcdm = [&lcdm, z](double kh) { return lcdm.plin(z, kh); };
                HaloModelReaction hm(hc, az[i], S, F);
                hm.set_one_loop(lr[i], lp[i]);
                for (double k : {0.05, 0.3, 1.0, 3.0})
                    d = std::max(d, std::abs(hm.reaction(k) - 1));
            }
            check(d < 2e-3, std::string(model == 0 ? "|fR0| = 1e-10" : "Omega_rc = 1e-6") + ": R = 1 to " + fmt(d, 2) +
                                " for k <= 3 h/Mpc");
        }
        double large = 0;
        for (double z : {0.0, 1.0})
            large = std::max(large, std::abs(fr5.pnl(z, 0.005) / (fr5.plin(z, 0.005)) - lcdm.pnl(z, 0.005) / lcdm.plin(z, 0.005)));
        check(large < 1e-3, "f(R): P_NL/P_L = LCDM's on large scales (k = 0.005 h/Mpc)");
    }

    //=========================================================================
    std::cout << "w0wa and massive neutrinos: ReACT x pseudo against HMcode-2020 directly\n";
    //=========================================================================
    {
        const std::vector<std::string> base = {"h=0.67", "OmegaB=0.049", "OmegaCDM=0.266", "n_s=0.965", "A_s=2.1e-9"};
        struct Case {
            std::string name;
            std::vector<std::string> args;
        };
        for (const auto & c : {Case{"w0 = -0.9, wa = 0.1", with(base, {"model=w0wa", "w0=-0.9", "wa=0.1", "m_nu=0"})},
                               Case{"LCDM, sum m_nu = 0.3 eV", with(base, {"m_nu=0.3"})}}) {
            const Run react(c.args, Nonlinear::ReACT), hm(c.args, Nonlinear::HMcode2020);
            double e = 0;
            std::cout << "  " << c.name << ": P_ReACT / P_HMcode - 1 at";
            for (double z : {0.0, 1.0})
                for (double k : {0.1, 0.3, 1.0, 3.0}) {
                    const double d = react.pnl(z, k) / hm.pnl(z, k) - 1;
                    e = std::max(e, std::abs(d));
                    std::cout << " (z=" << z << ", k=" << k << ") " << fmt(d, 2);
                }
            std::cout << "\n";
            check(e < 0.03, c.name + ": ReACT x pseudo = HMcode-2020 to 3% for k <= 3 h/Mpc, z <= 1 (" + fmt(e, 2) + ")");
        }
    }

    //=========================================================================
    std::cout << "A_s outside 1.5e-9 - 2.5e-9\n";
    //=========================================================================
    {
        auto args = elephant;
        args[4] = "A_s=3.2e-9";
        const Run hi(with(args, {"model=fr", "fR0=1e-5", "fR_n=1"}), Nonlinear::ReACT);
        const Run hi_lcdm(args, Nonlinear::HMcode2020);
        const double B = hi.pnl(0.0, 1.0) / hi_lcdm.pnl(0.0, 1.0), B0 = fr5.pnl(0.0, 1.0) / lcdm.pnl(0.0, 1.0);
        check(std::isfinite(B) && B > 1.0, "A_s = 3.2e-9 accepted; f(R) enhancement at k = 1 h/Mpc: " + fmt(B) +
                                               " (A_s = 2.43e-9: " + fmt(B0) + ")");
        // A higher amplitude means more screening: a smaller enhancement
        check(B < B0, "higher A_s gives a smaller f(R) enhancement (more screening)");
    }

    return FML::TESTS::summary("ReACT solver physics");
}
