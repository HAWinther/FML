// =============================================================================
// Tests of nonlinear = react for coupled quintessence / the Fading Dark Sector (Cosmology/Spectra/
// PowerSpectrumReACT.cpp: unscreened fifth force G(1 + 2 beta^2 f_c^2) and the drag f_c (-beta dphi/dx) of the CDM in
// the spherical collapses and one-loop kernels). A few minutes on 2 cores (4 Boltzmann solves, precision = fast).
//
//   * c = c' = 0 (a cosmological constant): equal to LCDM with nonlinear = hmcode2020
//   * The one-fluid collapse model (target E(x) without radiation, Omega_cb(x), mu_eff, friction) has the linear
//     growth of the Boltzmann solver's P_cb, for c' = 0.2 (where the growth differs from LCDM by 15% at z = 19)
//   * c' = 0.053 (the best fit of the FDS paper): P_NL -> P_lin on large scales, and the nonlinear enhancement over
//     LCDM at k = 1 h/Mpc exceeds the linear one
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>
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
    const std::vector<std::string> base = {"m_nu=0"};
    auto fds = [&](const std::string & c, const std::string & cp) {
        auto a = base;
        a.insert(a.end(), {"model=fading_dark_sector", "fds_c=" + c, "fds_cprime=" + cp});
        return a;
    };
    const Run lcdm(base, Nonlinear::HMcode2020);
    const Run f00(fds("0", "0"), Nonlinear::ReACT);
    const Run fb(fds("-0.85", "0.053"), Nonlinear::ReACT);
    const Run f2(fds("-0.85", "0.2"), Nonlinear::ReACT);
    const std::vector<double> ks = {0.01, 0.1, 0.3, 1.0, 3.0};

    //=========================================================================
    std::cout << "c = c' = 0: equal to LCDM (HMcode-2020)\n";
    //=========================================================================
    {
        double err = 0;
        for (double z : {0.0, 1.0})
            for (double k : ks)
                err = std::max(err, std::abs(f00.pnl(z, k) / lcdm.pnl(z, k) - 1));
        check(err < 2e-3, "max |P_NL(FDS c=c'=0) / P_NL(LCDM) - 1| = " + fmt(err, 2) + " < 2e-3");
    }

    //=========================================================================
    std::cout << "Collapse-model linear growth against the Boltzmann P_cb (c' = 0.2)\n";
    //=========================================================================
    {
        const auto & bg = *f2.bg;
        const double beta = 0.2;
        auto E2 = [&](double x) { return bg.rhoCDM(x) + bg.rhoB(x) + bg.rhoDE(x); };
        SPHERICALCOLLAPSE::Func1 E = [&](double x) { return std::sqrt(E2(x)); };
        SPHERICALCOLLAPSE::Func1 dlogE = [&](double x) {
            return 0.5 * (bg.drhoCDMdx(x) + bg.drhoBdx(x) + bg.drhoDEdx(x)) / E2(x);
        };
        auto fc = [&](double x) { return bg.rhoCDM(x) / (bg.rhoCDM(x) + bg.rhoB(x)); };
        SPHERICALCOLLAPSE::Func1 Ocb = [&](double x) { return (bg.rhoCDM(x) + bg.rhoB(x)) / E2(x); };
        SPHERICALCOLLAPSE::Func1 mu = [&](double x) { return 1 + 2 * beta * beta * fc(x) * fc(x); };
        SPHERICALCOLLAPSE::Func1 w = [](double) { return -1.0; };
        SPHERICALCOLLAPSE::SphericalCollapseModel m(E, Ocb, dlogE, mu, w);
        m.frictionofx = [&](double x) { return fc(x) * (3 + bg.drhoCDMdx(x) / bg.rhoCDM(x)); };
        FML::INTERPOLATION::SPLINE::Spline D, f, Dl, fl;
        SPHERICALCOLLAPSE::compute_growthfactor(m, D, f, std::log(1e-5), 0.0, 2000);
        double err = 0, dev = 0;
        for (double z : {1.0, 3.0, 7.0, 19.0}) {
            const double x = -std::log1p(z);
            const double ebs = std::sqrt(f2.plin(z, 0.02, true) / f2.plin(0, 0.02, true));
            const double ebs_lcdm = std::sqrt(lcdm.plin(z, 0.02, true) / lcdm.plin(0, 0.02, true));
            err = std::max(err, std::abs(D(x) / D(0.0) / ebs - 1));
            dev = std::max(dev, std::abs(ebs / ebs_lcdm - 1));
        }
        check(dev > 0.1, "the growth differs from LCDM (max " + fmt(dev, 3) + ")");
        check(err < 4e-3, "max |D_collapse / D_Boltzmann - 1| (k = 0.02 h/Mpc, z = 1-19) = " + fmt(err, 2) + " < 4e-3");
    }

    //=========================================================================
    std::cout << "c' = 0.053: large-scale limit and the nonlinear enhancement\n";
    //=========================================================================
    {
        double err = 0;
        for (double z : {0.0, 1.0})
            err = std::max(err, std::abs(fb.pnl(z, 0.01) / fb.plin(z, 0.01) - lcdm.pnl(z, 0.01) / lcdm.plin(z, 0.01)));
        check(err < 2e-3, "P_NL/P_lin at k = 0.01 h/Mpc as in LCDM (" + fmt(err, 2) + ")");
        for (double z : {0.0, 1.0}) {
            const double lin = fb.plin(z, 1.0) / lcdm.plin(z, 1.0), nl = fb.pnl(z, 1.0) / lcdm.pnl(z, 1.0);
            check(nl > lin && nl < 1.5 * lin, "z = " + fmt(z) + ", k = 1 h/Mpc: P_NL ratio " + fmt(nl) +
                                                  " > linear ratio " + fmt(lin));
        }
        bool ok = true;
        for (double z : {0.0, 1.0})
            for (double k : ks) {
                const double b = f2.pnl(z, k) / f2.plin(z, k);
                ok = ok && std::isfinite(b) && b > 0.5;
            }
        check(ok, "c' = 0.2 (a strong fifth force): P_NL finite and positive");
    }
    return FML::TESTS::summary("ReACT coupled-quintessence");
}
