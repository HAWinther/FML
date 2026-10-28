// =============================================================================
// Tests of the theory and the likelihoods (Theory.h, Likelihoods.h, SampledParameterMap.h); the sampler is tested
// in MCMC/tests. Run with `make test` in CosmologyFitting.
//
//   * Theory: D_M, D_H against independent integrals of 1/H; the sound horizon at the drag epoch against
//     CLASS; the Planck 2018 distance priors at the Planck best fit (R, l_A within 1 sigma); the fitting formula
//     for r_d against the recombination value
//   * Likelihoods: the analytic marginalization over the supernova amplitude; the BAO model vector; plik_lite
//     against planck-lite-py (4 variants, 1e-8); the mock CMB likelihood
//   * Parameter aliases: theta_MC_100 -> h and tau_reio -> z_reion round trips; the Planck 2018 best fit with
//     precision = fast: 100 theta_*, plik_lite chi^2, sigma_8
// =============================================================================

#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/CosmologyFitting/Likelihoods.h>
#include <FML/MCMC/Sampler.h>
#include <FML/CosmologyFitting/SampledParameterMap.h>
#include <FML/CosmologyFitting/Theory.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace FML::COSMOLOGY;
using namespace FML::MCMC;
using namespace FML::COSMOLOGY::FITTING;

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
    std::cout << std::unitbuf;
    Utils::verbose = false;

    std::cout << "Theory: distances and sound horizons\n";
    {
        RunSettings s = ParameterFile::read("", {"preset=planck2018"});
        Theory t(s);
        const auto & b = t.background();
        double e = 0.0;
        for (double z : {0.3, 1.0, 2.33}) {
            const double dm =
                Utils::integrate([&](double zz) { return c / b.H_of_x(-std::log1p(zz)); }, 0.0, z, 0.0, 1e-12) / Mpc;
            e = std::max(e, std::fabs(t.D_M(z) / dm - 1.0));
            e = std::max(e, std::fabs(t.D_H(z) * b.H_of_x(-std::log1p(z)) * Mpc / c - 1.0));
        }
        check_close("D_M, D_H = integrals of c/H", e, 0.0, 1e-6);
        check_close("D_V = (z D_M^2 D_H)^(1/3)",
                    t.D_V(0.5) / std::cbrt(0.5 * t.D_M(0.5) * t.D_M(0.5) * t.D_H(0.5)),
                    1.0,
                    1e-14);
        // CLASS 3.3 (RECFAST, same parameters): z_d = 1059.963, r_s(z_d) = 147.2245 Mpc
        check_close("z_drag = CLASS", t.z_drag(), 1059.963, 1e-4 * 1059.963);
        check_rel("r_drag = CLASS (147.2245 Mpc)", t.r_drag(), 147.2245, 2e-4);
        TheoryOptions fo;
        fo.sound_horizon = TheoryOptions::SoundHorizon::Fit;
        Theory tf(s, fo);
        std::cout << std::setprecision(7) << "  r_d: recombination " << t.r_drag() << ", Aubourg et al. fit "
                  << tf.r_drag() << " Mpc\n";
        check_rel("r_drag fitting formula within 0.1%", tf.r_drag(), t.r_drag(), 1e-3);
        // Planck 2018 TT,TE,EE+lowE mean (omega_b = 0.02236, omega_c = 0.1202, h = 0.6727, m_nu = 0.06)
        RunSettings pl = ParameterFile::read("", {"preset=planck2018", "m_nu=0.06", "h=0.6727"});
        pl.cosmo.OmegaB = 0.02236 / (0.6727 * 0.6727);
        pl.cosmo.OmegaCDM = 0.1202 / (0.6727 * 0.6727);
        Theory tp(pl);
        std::cout << std::setprecision(7) << "  Planck mean: z_* = " << tp.z_star() << ", R = " << tp.R_shift()
                  << ", l_A = " << tp.l_A() << "\n";
        check_close("R at the Planck mean (prior 1.7502 +- 0.0046)", (tp.R_shift() - 1.7502) / 0.0046, 0.0, 1.0);
        check_close("l_A at the Planck mean (prior 301.471 +- 0.090)", (tp.l_A() - 301.471) / 0.090, 0.0, 1.0);
        check_close("z_* (tau = 1) = Planck's 1089.95 +- 0.27", (tp.z_star() - 1089.95) / 0.27, 0.0, 1.0);
        PlanckDistancePriors dp;
        check("distance-prior chi^2 < 2 at the Planck mean", -2.0 * dp.loglike(tp) < 2.0);
    }

    std::cout << "Likelihoods\n";
    {
        // free supernova amplitude: the analytic marginalization equals the numerical integral over A
        {
            RunSettings s0 = ParameterFile::read("", {"h=0.72"});
            Theory t0(s0);
            std::vector<double> z, d, e;
            std::ifstream f("external/supernovadata.txt");
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream is(line);
                double a1, a2, a3;
                if (is >> a1 >> a2 >> a3) {
                    z.push_back(a1);
                    d.push_back(a2);
                    e.push_back(a3);
                }
            }
            auto chi2 = [&](double A) {
                double c2 = 0.0;
                for (size_t i = 0; i < z.size(); i++)
                    c2 += std::pow((d[i] - A * t0.d_L(z[i]) / 1000.0) / e[i], 2);
                return c2;
            };
            const double ref = -0.5 * chi2(1.0); // offset for the numerical integral
            const double I =
                Utils::integrate([&](double A) { return std::exp(-0.5 * chi2(A) - ref); }, 0.5, 1.5, 0.0, 1e-12);
            const double analytic =
                SupernovaLikelihood("external/supernovadata.txt", true).loglike(t0) + 0.5 * std::log(2.0 * M_PI);
            check_close("free amplitude: analytic = numerical marginalization over A",
                        analytic,
                        std::log(I) + ref,
                        1e-8);
            check_close("fixed amplitude: ln L = -chi^2(A = 1)/2", SupernovaLikelihood().loglike(t0), ref, 1e-10);
        }
        RunSettings s = ParameterFile::read("", {"preset=planck2018", "m_nu=0.06"});
        Theory t(s);
        BAOLikelihood bao("desi_dr2", "external/bao/desi_dr2_mean.txt", "external/bao/desi_dr2_cov.txt");
        const auto m = bao.model(t);
        check("DESI DR2: 13 data points", bao.n_data() == 13 && m.size() == 13);
        check_rel("first point = D_V(0.295)/r_d", m[0], t.D_V(0.295) / t.r_drag(), 1e-14);
        const double chi2 = -2.0 * bao.loglike(t);
        std::cout << std::setprecision(4) << "  DESI DR2 chi^2 for the Planck 2018 best fit = " << chi2
                  << " (13 points)\n";
        check("DESI DR2 chi^2 for Planck LCDM is reasonable (< 40)", chi2 < 40.0);
    }

    std::cout << "Planck 2018 plik_lite against planck-lite-py\n";
    {
        // D_ell from CLASS (planck-lite-py's test spectrum) and its ln L from planck-lite-py (= Cobaya's plik_lite)
        std::vector<double> tt(2509, 0.0), te(2509, 0.0), ee(2509, 0.0);
        {
            std::ifstream f("tests/reference/Dl_planck2015fit.dat");
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream is(line);
                double l, a, b, c;
                if (is >> l >> a >> b >> c && l < 2509) {
                    tt[int(l)] = a;
                    te[int(l)] = b;
                    ee[int(l)] = c;
                }
            }
        }
        struct Case {
            const char * name;
            bool tt, lo;
            double expected;
        };
        for (const Case & c : {Case{"TTTEEE", false, false, -291.33481235418026},
                               Case{"TTTEEE + low-ell TT bins", false, true, -293.95586501795134},
                               Case{"TT", true, false, -101.58123068722583},
                               Case{"TT + low-ell TT bins", true, true, -104.20228335099686}}) {
            PlanckLiteLikelihood L("external/planck2018_lite", c.tt, c.lo);
            check_close(std::string("ln L, ") + c.name, L.loglike(tt, te, ee), c.expected, 1e-8);
        }
        PlanckLiteLikelihood L("external/planck2018_lite", false, false);
        check("613 bins", L.n_data() == 613);
        // calibration: C -> C/A^2; A = 1.0025 shifts ln L
        check("A_planck changes ln L", std::fabs(L.loglike(tt, te, ee, 1.0025) - L.loglike(tt, te, ee)) > 0.1);
    }

    std::cout << "Parameter aliases (SampledParameterMap)\n";
    {
        const std::vector<std::string> fixed = {"preset=planck2018", "m_nu=0.06", "precision=fast"};
        SampledParameterMap pm(fixed,
                               {"omega_b", "omega_cdm", "theta_MC_100", "logA", "tau_reio", "A_planck"},
                               {"A_planck"});
        const RunSettings s = pm.settings({0.02237, 0.1200, 1.03985, 3.044, 0.0544, 1.001});
        const double h2 = s.cosmo.h * s.cosmo.h;
        Theory t(s);
        check_close("theta_MC_100 -> h: theta_MC reproduced", t.theta_MC(), 1.03985, 1e-7);
        check_close("... at fixed omega_b", s.cosmo.OmegaB * h2, 0.02237, 1e-12);
        check_close("... at fixed omega_cdm", s.cosmo.OmegaCDM * h2, 0.1200, 1e-12);
        std::cout << std::setprecision(6) << "  h = " << s.cosmo.h << ", z_reion = " << s.cosmo.z_reion << "\n";
        check_close("h near the Planck value 0.6736", s.cosmo.h, 0.6736, 0.002);
        check_rel("logA -> A_s", s.cosmo.A_s, 1e-10 * std::exp(3.044), 1e-14);
        BackgroundCosmology bg(s.cosmo);
        bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
        check_close("tau_reio -> z_reion: model tau reproduced",
                    RecombinationHistory(bg, s.rec).tau_reionization_model(s.cosmo.z_reion),
                    0.0544,
                    1e-7);
        check_close("... full tau of the run (with residual ionization) within 3e-4", t.tau_reio(), 0.0544, 3e-4);
        check_close("100 theta_* (tau = 1) = Planck 1.04110 +- 0.00031", t.theta_star(), 1.04110, 0.00031);
        check("theta_MC_100 together with H0 is an error",
              throws([&] { SampledParameterMap(fixed, {"theta_MC_100", "H0"}); }));
        check("logA together with A_s is an error", throws([&] { SampledParameterMap(fixed, {"logA", "A_s"}); }));
        check("tau_reio outside the z_reion range is rejected",
              throws([&] { SampledParameterMap(fixed, {"tau_reio"}).settings({2.0}); }));

        // Full CMB at the Planck 2018 best fit (precision = fast): plik_lite chi^2 and sigma_8
        t.set_nuisance("A_planck", 1.0);
        const auto pl = Likelihoods::create("planck_lite_TTTEEE_lowl");
        const double chi2 = -2.0 * pl->loglike(t);
        const double s8 = t.sigma8();
        std::cout << std::setprecision(5) << "  plik_lite TTTEEE + low-ell TT bins: chi^2 = " << chi2
                  << " (615 bins), sigma_8 = " << s8 << "\n";
        check("plik_lite chi^2 at the Planck best fit < 650 (615 bins)", chi2 < 650.0);
        check_close("sigma_8 = Planck 0.8111 +- 0.0060 (1 sigma)", s8, 0.8111, 0.0060);
    }

    std::cout << "Mock CMB likelihood\n";
    {
        const std::string dir = "test_output_mcmc_mock";
        std::filesystem::create_directories(dir);
        std::vector<double> tt(301, 0.0), te(301, 0.0), ee(301, 0.0);
        {
            std::ofstream f(dir + "/cells.txt");
            f << std::setprecision(17);
            for (int l = 2; l <= 300; l++) {
                tt[l] = 1000.0 + 5000.0 * std::exp(-std::pow((l - 220.0) / 80.0, 2));
                ee[l] = 0.1 + 20.0 * std::exp(-std::pow((l - 150.0) / 100.0, 2));
                te[l] = 0.5 * std::sqrt(tt[l] * ee[l]) * std::cos(l / 40.0);
                f << l << " " << tt[l] << " " << ee[l] << " " << te[l] << "\n";
            }
        }
        // round trip through the file (17 digits) as the likelihood reads it
        for (int l = 2; l <= 300; l++) {
            std::ostringstream o;
            o << std::setprecision(17) << tt[l] << " " << ee[l] << " " << te[l];
            std::istringstream i(o.str());
            i >> tt[l] >> ee[l] >> te[l];
        }
        CMBMockLikelihood M(dir + "/cells.txt", 300, 0.6, 33.0, 7.0);
        check_close("ln L = 0 at the fiducial", M.loglike(tt, te, ee), 0.0, 1e-9);
        auto tt2 = tt;
        for (auto & v : tt2)
            v *= 1.01;
        const double l1 = M.loglike(tt2, te, ee);
        // small changes: -2 ln L ~ sum (2l+1) f_sky/2 (dC/C)^2 for TT alone (noise negligible at low ell)
        check("1% higher TT: ln L < 0", l1 < 0.0);
        check("unknown likelihood option is an error",
              throws([] { Likelihoods::create("cmb_mock", {{"cmb_mock_lmx", "10"}}); }));
        { // with BB (5th column)
            std::ofstream f(dir + "/cells_bb.txt");
            f << std::setprecision(17);
            for (int l = 2; l <= 300; l++)
                f << l << " " << tt[l] << " " << ee[l] << " " << te[l] << " " << 0.01 * l / 100.0 << "\n";
        }
        std::vector<double> bb(301, 0.0);
        for (int l = 2; l <= 300; l++) {
            std::ostringstream o;
            o << std::setprecision(17) << 0.01 * l / 100.0;
            bb[l] = std::stod(o.str());
        }
        CMBMockLikelihood MB(dir + "/cells_bb.txt", 300, 0.6, 33.0, 7.0, true);
        check_close("with BB: ln L = 0 at the fiducial", MB.loglike(tt, te, ee, bb), 0.0, 1e-9);
        auto bb2 = bb;
        for (auto & v : bb2)
            v *= 1.5;
        check("... and < 0 for a different BB", MB.loglike(tt, te, ee, bb2) < 0.0);
        check("cmb_mock_bb must be true or false",
              throws([] { Likelihoods::create("cmb_mock", {{"cmb_mock_bb", "yes"}}); }));
        std::filesystem::remove_all(dir);
    }

    std::cout << "Planck 2018 lensing likelihood\n";
    {
        // synthetic spectra; reference: Cobaya planck_2018_lensing.native / .CMBMarged (log_likelihood on the same
        // D_ell)
        std::vector<double> tt(2501, 0.0), ee(2501, 0.0), te(2501, 0.0), pp(2501, 0.0);
        for (int l = 2; l <= 2500; l++) {
            tt[l] = 5000.0 * std::exp(-l / 800.0) * (1.0 + 0.3 * std::sin(l / 50.0));
            ee[l] = 30.0 * std::exp(-l / 1500.0) * (1.0 + 0.5 * std::cos(l / 45.0));
            te[l] = 100.0 * std::sin(l / 60.0) * std::exp(-l / 1200.0);
            pp[l] = 1.5e-7 / (1.0 + std::pow(l / 120.0, 1.5));
        }
        PlanckLensingLikelihood L("external/planck2018_lensing", false), M("external/planck2018_lensing", true);
        check("9 bandpowers", L.n_data() == 9);
        check_rel("ln L = Cobaya (native)", L.loglike(tt, te, ee, pp, 1.0), -3719.091695463751, 1e-11);
        check_rel("... with A_planck = 1.01", L.loglike(tt, te, ee, pp, 1.01), -3718.384910680866, 1e-11);
        check_rel("ln L = Cobaya (CMB-marginalized)", M.loglike(tt, te, ee, pp), -23.153578141104, 1e-11);
        check("A_planck is its nuisance parameter (not of the marginalized one)",
              L.nuisance_parameters().size() == 1 && M.nuisance_parameters().empty());
    }

    return Test::summary();
}
