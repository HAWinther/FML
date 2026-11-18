// =============================================================================
// Tests of the theory and the likelihoods (Theory.h, Likelihoods.h, SampledParameterMap.h); the sampler is tested
// in MCMC/tests. Run with `make test` in CosmologyFitting.
//
//   * Theory: D_M, D_H against independent integrals of 1/H; the sound horizon at the drag epoch against
//     CLASS; the Planck 2018 distance priors at the Planck best fit (R, l_A within 1 sigma); the fitting formula
//     for r_d against the recombination value
//   * Likelihoods: the analytic marginalization over the supernova amplitude; the BAO model vector; plik_lite
//     against planck-lite-py (4 variants, 1e-8); the mock CMB likelihood
//   * BAO + RSD: the BOSS DR12 / eBOSS DR16 model vectors (r_d,fid scaling, H r_d); sigma_8 and f sigma_8 (CAMB's
//     definition, sigma_8,vd^2/sigma_8 of the CDM + baryons) against CAMB 2.0.5 (AccuracyBoost 2) for the planck2018
//     preset with m_nu = 0.06 (precision = fast: 3e-4 for f sigma_8; 4e-5 at the default precision); the same from
//     the run with the CMB; Brans-Dicke with omega = 1e7 = LCDM
//   * DESI DR1 ShapeFit + BAO: the theory vector (q_iso, q_ap, df, dm; q_par, q_per) against DESI's Cobaya likelihood
//     desi_shapefit_bao_all evaluated with cosmoprimo 1.0.0 (CAMB 2.0.5) for two cosmologies away from the fiducial
//     (precision = fast); chi^2 at the fiducial (theory = 1, 0)
//   * KiDS-1000 COSEBIs: chi^2 at the maximum posterior of the KiDS chain (CosmoSIS/kcap with CAMB HMcode-2016)
//     and the delta_z prior against the chain's uncorrelated parameters
//   * Theory::lensing_Sigma against the Weyl potential of the full perturbations (LCDM, Brans-Dicke omega = 50); the
//     log10_bd_omega and inv_bd_omega aliases
//   * Parameter aliases: theta_MC_100 -> h and tau_reio -> z_reion round trips; the Planck 2018 best fit with
//     precision = fast: 100 theta_*, plik_lite chi^2, sigma_8
// =============================================================================

#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>
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

using FML::TESTS::throws;

using namespace FML::COSMOLOGY;
using namespace FML::MCMC;
using namespace FML::COSMOLOGY::FITTING;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;


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
        check_close(e, 0.0, 1e-6, "D_M, D_H = integrals of c/H");
        check_close(t.D_V(0.5) / std::cbrt(0.5 * t.D_M(0.5) * t.D_M(0.5) * t.D_H(0.5)),
                    1.0,
                    1e-14,
                    "D_V = (z D_M^2 D_H)^(1/3)");
        // CLASS 3.3 (RECFAST, same parameters): z_d = 1059.963, r_s(z_d) = 147.2245 Mpc
        check_close(t.z_drag(), 1059.963, 1e-4 * 1059.963, "z_drag = CLASS");
        check_rel(t.r_drag(), 147.2245, 2e-4, "r_drag = CLASS (147.2245 Mpc)");
        TheoryOptions fo;
        fo.sound_horizon = TheoryOptions::SoundHorizon::Fit;
        Theory tf(s, fo);
        std::cout << std::setprecision(7) << "  r_d: recombination " << t.r_drag() << ", Aubourg et al. fit "
                  << tf.r_drag() << " Mpc\n";
        check_rel(tf.r_drag(), t.r_drag(), 1e-3, "r_drag fitting formula within 0.1%");
        // Planck 2018 TT,TE,EE+lowE mean (omega_b = 0.02236, omega_c = 0.1202, h = 0.6727, m_nu = 0.06)
        RunSettings pl = ParameterFile::read("", {"preset=planck2018", "m_nu=0.06", "h=0.6727"});
        pl.cosmo.OmegaB = 0.02236 / (0.6727 * 0.6727);
        pl.cosmo.OmegaCDM = 0.1202 / (0.6727 * 0.6727);
        Theory tp(pl);
        std::cout << std::setprecision(7) << "  Planck mean: z_* = " << tp.z_star() << ", R = " << tp.R_shift()
                  << ", l_A = " << tp.l_A() << "\n";
        check_close((tp.R_shift() - 1.7502) / 0.0046, 0.0, 1.0, "R at the Planck mean (prior 1.7502 +- 0.0046)");
        check_close((tp.l_A() - 301.471) / 0.090, 0.0, 1.0, "l_A at the Planck mean (prior 301.471 +- 0.090)");
        check_close((tp.z_star() - 1089.95) / 0.27, 0.0, 1.0, "z_* (tau = 1) = Planck's 1089.95 +- 0.27");
        PlanckDistancePriors dp;
        check(-2.0 * dp.loglike(tp) < 2.0, "distance-prior chi^2 < 2 at the Planck mean");
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
            check_close(analytic,
                        std::log(I) + ref,
                        1e-8,
                        "free amplitude: analytic = numerical marginalization over A");
            check_close(SupernovaLikelihood().loglike(t0), ref, 1e-10, "fixed amplitude: ln L = -chi^2(A = 1)/2");
        }
        RunSettings s = ParameterFile::read("", {"preset=planck2018", "m_nu=0.06"});
        Theory t(s);
        BAOLikelihood bao("desi_dr2", "external/bao/desi_dr2_mean.txt", "external/bao/desi_dr2_cov.txt");
        const auto m = bao.model(t);
        check(bao.n_data() == 13 && m.size() == 13, "DESI DR2: 13 data points");
        check_rel(m[0], t.D_V(0.295) / t.r_drag(), 1e-14, "first point = D_V(0.295)/r_d");
        const double chi2 = -2.0 * bao.loglike(t);
        std::cout << std::setprecision(4) << "  DESI DR2 chi^2 for the Planck 2018 best fit = " << chi2
                  << " (13 points)\n";
        check(chi2 < 40.0, "DESI DR2 chi^2 for Planck LCDM is reasonable (< 40)");
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
            check_close(L.loglike(tt, te, ee), c.expected, 1e-8, std::string("ln L, ") + c.name);
        }
        PlanckLiteLikelihood L("external/planck2018_lite", false, false);
        check(L.n_data() == 613, "613 bins");
        // calibration: C -> C/A^2; A = 1.0025 shifts ln L
        check(std::fabs(L.loglike(tt, te, ee, 1.0025) - L.loglike(tt, te, ee)) > 0.1, "A_planck changes ln L");
    }

    std::cout << "Parameter aliases (SampledParameterMap)\n";
    {
        const std::vector<std::string> fixed = {"preset=planck2018", "m_nu=0.06", "precision=fast"};
        SampledParameterMap pm(fixed,
                               {"omega_b", "omega_cdm", "theta_MC_100", "logA", "tau_reio", "A_planck"},
                               {"A_planck"});
        const RunSettings s = pm.settings({0.02237, 0.1200, 1.04092, 3.044, 0.0544, 1.001});
        const double h2 = s.cosmo.h * s.cosmo.h;
        Theory t(s);
        check_close(t.theta_MC(), 1.04092, 1e-7, "theta_MC_100 -> h: theta_MC reproduced");
        check_close(s.cosmo.OmegaB * h2, 0.02237, 1e-12, "... at fixed omega_b");
        check_close(s.cosmo.OmegaCDM * h2, 0.1200, 1e-12, "... at fixed omega_cdm");
        std::cout << std::setprecision(6) << "  h = " << s.cosmo.h << ", z_reion = " << s.cosmo.z_reion << "\n";
        // CosmoMC theta_MC: Planck 2018 TT,TE,EE+lowE+lensing 100 theta_MC = 1.04092 with H0 = 67.36
        check_close(s.cosmo.h, 0.6736, 0.0005, "h near the Planck value 0.6736");
        check_rel(s.cosmo.A_s, 1e-10 * std::exp(3.044), 1e-14, "logA -> A_s");
        BackgroundCosmology bg(s.cosmo);
        bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
        check_close(RecombinationHistory(bg, s.rec).tau_reionization_model(s.cosmo.z_reion),
                    0.0544,
                    1e-7,
                    "tau_reio -> z_reion: model tau reproduced");
        check_close(t.tau_reio(), 0.0544, 3e-5, "... full tau of the run (with residual ionization) within 3e-5");
        check_close(t.theta_star(), 1.04110, 0.00031, "100 theta_* (tau = 1) = Planck 1.04110 +- 0.00031");
        check(throws([&] { SampledParameterMap(fixed, {"theta_MC_100", "H0"}); }),
              "theta_MC_100 together with H0 is an error");
        check(throws([&] { SampledParameterMap(fixed, {"logA", "A_s"}); }), "logA together with A_s is an error");
        check(throws([&] { SampledParameterMap(fixed, {"tau_reio"}).settings({2.0}); }),
              "tau_reio outside the z_reion range is rejected");

        // Full CMB at the Planck 2018 best fit (precision = fast): plik_lite chi^2 and sigma_8
        t.set_nuisance("A_planck", 1.0);
        const auto pl = Likelihoods::create("planck_lite_TTTEEE_lowl");
        const double chi2 = -2.0 * pl->loglike(t);
        const double s8 = t.sigma8();
        std::cout << std::setprecision(5) << "  plik_lite TTTEEE + low-ell TT bins: chi^2 = " << chi2
                  << " (615 bins), sigma_8 = " << s8 << "\n";
        check(chi2 < 650.0, "plik_lite chi^2 at the Planck best fit < 650 (615 bins)");
        check_close(s8, 0.8111, 0.0060, "sigma_8 = Planck 0.8111 +- 0.0060 (1 sigma)");
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
        check_close(M.loglike(tt, te, ee), 0.0, 1e-9, "ln L = 0 at the fiducial");
        auto tt2 = tt;
        for (auto & v : tt2)
            v *= 1.01;
        const double l1 = M.loglike(tt2, te, ee);
        // small changes: -2 ln L ~ sum (2l+1) f_sky/2 (dC/C)^2 for TT alone (noise negligible at low ell)
        check(l1 < 0.0, "1% higher TT: ln L < 0");
        check(throws([] { Likelihoods::create("cmb_mock", {{"cmb_mock_lmx", "10"}}); }),
              "unknown likelihood option is an error");
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
        check_close(MB.loglike(tt, te, ee, bb), 0.0, 1e-9, "with BB: ln L = 0 at the fiducial");
        auto bb2 = bb;
        for (auto & v : bb2)
            v *= 1.5;
        check(MB.loglike(tt, te, ee, bb2) < 0.0, "... and < 0 for a different BB");
        check(throws([] { Likelihoods::create("cmb_mock", {{"cmb_mock_bb", "yes"}}); }),
              "cmb_mock_bb must be true or false");
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
        check(L.n_data() == 9, "9 bandpowers");
        check_rel(L.loglike(tt, te, ee, pp, 1.0), -3719.091695463751, 1e-11, "ln L = Cobaya (native)");
        check_rel(L.loglike(tt, te, ee, pp, 1.01), -3718.384910680866, 1e-11, "... with A_planck = 1.01");
        check_rel(M.loglike(tt, te, ee, pp), -23.153578141104, 1e-11, "ln L = Cobaya (CMB-marginalized)");
        check(L.nuisance_parameters().size() == 1 && M.nuisance_parameters().empty(),
              "A_planck is its nuisance parameter (not of the marginalized one)");
    }

    std::cout << "BAO + RSD (BOSS DR12, eBOSS DR16) and f sigma_8\n";
    {
        RunSettings s = ParameterFile::read("", {"preset=planck2018", "m_nu=0.06", "precision=fast"});
        TheoryOptions o;
        o.cmb = false;
        o.growth_z = {0.0, 0.38, 0.51, 0.61, 0.698, 1.48};
        Theory t(s, o);
        // CAMB 2.0.5 (AccuracyBoost = 2) for the same parameters (h = 0.6766, omega_b = 0.02242, omega_c = 0.11933,
        // one massive neutrino of 0.06 eV, N_eff = 3.046, Y_p = 0.2454, A_s = 2.105209e-9, n_s = 0.9665):
        // get_fsigma8() and get_sigmaR(8, delta_nonu)
        const double camb_fs8[] = {0.424699, 0.474930, 0.473577, 0.468638, 0.462262, 0.377847};
        const double camb_s8cb[] = {0.813674, 0.666596, 0.623849, 0.593622, 0.568843, 0.408141};
        double emax_f = 0.0, emax_s = 0.0;
        for (size_t i = 0; i < o.growth_z.size(); i++) {
            emax_f = std::max(emax_f, std::fabs(t.fsigma8(o.growth_z[i]) / camb_fs8[i] - 1.0));
            emax_s = std::max(emax_s, std::fabs(t.sigma8_cb(o.growth_z[i]) / camb_s8cb[i] - 1.0));
        }
        std::cout << std::setprecision(3) << "  f sigma_8, sigma_8,cb against CAMB at z = 0-1.48: max " << emax_f << ", "
                  << emax_s << "\n";
        check(emax_f < 5e-4, "f sigma_8 = CAMB's to 5e-4 (z = 0-1.48; 4e-5 at the default precision)");
        check(emax_s < 2e-4, "sigma_8 of the CDM + baryons = CAMB's to 2e-4");
        check(throws([&] { t.fsigma8(0.3); }), "f sigma_8 at a redshift not in growth_z is an error");
        check(throws([&] { t.cmb(); }), "cmb() with TheoryOptions::cmb = false is an error");

        auto boss = Likelihoods::create("boss_dr12");
        auto * b = dynamic_cast<BAOLikelihood *>(boss.get());
        const auto m = b->model(t);
        check(boss->n_data() == 9 && boss->growth_redshifts() == std::vector<double>{0.38, 0.51, 0.61},
              "boss_dr12: 9 values, f sigma_8 at z = 0.38, 0.51, 0.61");
        check_rel(m[0], t.D_M(0.38) * 147.78 / t.r_drag(), 1e-14, "D_M (r_d,fid/r_d) at z = 0.38");
        check_rel(m[1], c / 1e3 / t.D_H(0.38) * t.r_drag() / 147.78, 1e-14, "H (r_d/r_d,fid) at z = 0.38");
        check_rel(m[2], t.fsigma8(0.38), 1e-14, "f sigma_8 at z = 0.38");
        const double chi2 = -2.0 * boss->loglike(t);
        auto lrg = Likelihoods::create("eboss_dr16_lrg"), qso = Likelihoods::create("eboss_dr16_qso");
        auto * bl = dynamic_cast<BAOLikelihood *>(lrg.get());
        check_rel(bl->model(t)[7], t.D_H(0.698) / t.r_drag(), 1e-14, "eboss_dr16_lrg: D_H/r_d at z = 0.698");
        std::cout << "  chi^2 for this Planck cosmology: boss_dr12 " << chi2 << " (9), boss_dr12_bao "
                  << -2.0 * Likelihoods::create("boss_dr12_bao")->loglike(t) << " (6), eboss_dr16_lrg "
                  << -2.0 * lrg->loglike(t) << " (9), eboss_dr16_qso " << -2.0 * qso->loglike(t) << " (3)\n";
        check(chi2 < 15.0, "boss_dr12 chi^2 for Planck LCDM is reasonable (< 15 for 9 values)");
        check(throws([] { BAOLikelihood("x", "external/bao/sdss_DR12Consensus_final.dat",
                                        "external/bao/final_consensus_covtot_dM_Hz_fsig.txt"); }),
              "bao_Hz_rs without rs_fid is an error");

        // The same from the run with the CMB (cmb() computes the growth before freeing the perturbations)
        TheoryOptions oc;
        oc.growth_z = {0.38};
        Theory tc(s, oc);
        check_rel(tc.fsigma8(0.38), t.fsigma8(0.38), 1e-3, "f sigma_8 from the CMB run = from P(k) alone (1e-3)");

        // Brans-Dicke with omega -> infinity is LCDM
        RunSettings sj =
            ParameterFile::read("", {"preset=planck2018", "m_nu=0.06", "precision=fast", "model=jbd", "bd_omega=1e7"});
        Theory tj(sj, o);
        check_rel(tj.fsigma8(0.51), t.fsigma8(0.51), 1e-5, "Brans-Dicke omega = 1e7: f sigma_8 = LCDM (1e-5)");
    }

    std::cout << "DESI DR1 ShapeFit + BAO against DESI's likelihood (cosmoprimo)\n";
    {
        DESIShapeFitLikelihood L("external/bao/desi_dr1_shapefit_mean.txt", "external/bao/desi_dr1_shapefit_cov.txt");
        check(L.n_data() == 26 && L.growth_redshifts().size() == 6, "26 values, ShapeFit at 6 redshifts");
        const RunSettings base = ParameterFile::read("", {"precision=fast"});
        struct Case {
            const char * name;
            double h, ob, oc, As, mnu, chi2;
            std::vector<double> ref; // desi_shapefit_bao_all with cosmoprimo 1.0.0, engine camb (CAMB 2.0.5)
        };
        const std::vector<Case> cases = {
            {"h = 0.70, omega_cdm = 0.115",
             0.70, 0.02237, 0.115, 2.083e-9, 0.06, 36.64674444441464,
             {0.9628578, 1.0073951, 0.933496,  -0.035978,  0.9687711, 1.0119119, 0.9474341, -0.0359796, 0.9733283,
              1.0150733, 0.9575932, -0.03598,  0.9773687,  1.0175139, 0.9661098, -0.0359796, 0.9828858, 1.0200361,
              0.9768097, -0.0359777, 0.9846575, 1.0205758, 0.9799597, -0.0359765, 1.0039151, 0.9832435}},
            {"omega_b = 0.0222, m_nu = 0.12 eV, A_s = 2.2e-9",
             0.6736, 0.0222, 0.12, 2.2e-9, 0.12, 30.662663895449267,
             {0.9985567, 0.9997747, 1.0167898, -0.0046061, 0.9983727, 0.9996422, 1.0164252, -0.0043927, 0.9982336,
              0.9995519, 1.0161847, -0.0042049, 0.9981123, 0.9994838, 1.0160079, -0.0040114, 0.9979491, 0.9994159,
              1.015842,  -0.0036808, 0.9978974, 0.9994021, 1.0158148, -0.0035493, 0.9973372, 0.9979418}}};
        for (const auto & c : cases) {
            RunSettings s = DESIShapeFitLikelihood::fiducial_settings(base);
            s.cosmo.h = c.h;
            s.cosmo.OmegaB = c.ob / (c.h * c.h);
            s.cosmo.OmegaCDM = c.oc / (c.h * c.h);
            s.cosmo.A_s = c.As;
            s.cosmo.m_nu = c.mnu;
            TheoryOptions o;
            o.cmb = false;
            o.growth_z = L.growth_redshifts();
            o.pk_z = L.pk_redshifts();
            Theory t(s, o);
            const auto m = L.model(t);
            double eq = 0.0, edf = 0.0, edm = 0.0;
            for (size_t i = 0; i < m.size(); i++) {
                const double d = std::fabs(m[i] - c.ref[i]);
                if (i < 24 && i % 4 == 2)
                    edf = std::max(edf, d);
                else if (i < 24 && i % 4 == 3)
                    edm = std::max(edm, d);
                else
                    eq = std::max(eq, d);
            }
            const double chi2 = -2.0 * L.loglike(t);
            std::cout << std::setprecision(3) << "  " << c.name << ": max |difference| q " << eq << ", df " << edf
                      << ", dm " << edm << "; chi^2 " << std::setprecision(6) << chi2 << " (DESI " << c.chi2 << ")\n";
            check(eq < 1e-5 && edf < 5e-4 && edm < 1e-3,
                  std::string(c.name) + ": q to 1e-5, df to 5e-4, dm to 1e-3 of DESI's likelihood");
            check(std::fabs(chi2 - c.chi2) < 0.1, std::string(c.name) + ": chi^2 to 0.1 of DESI's likelihood");
        }
        RunSettings sf = DESIShapeFitLikelihood::fiducial_settings(base);
        TheoryOptions o;
        o.cmb = false;
        o.growth_z = L.growth_redshifts();
        o.pk_z = L.pk_redshifts();
        Theory tf(sf, o);
        check_close(-2.0 * L.loglike(tf), 32.80514750176213, 1e-6, "fiducial: chi^2 of the data alone (theory 1, 0)");
    }

    std::cout << "KiDS-1000 COSEBIs against the KiDS chain\n";
    {
        // The maximum posterior of the release's COSEBIs chain (maxpost_multinest_start_C.txt, last row): omch2, ombh2,
        // h, n_s, A_s, HMcode A, A_IA, the shifts bias_1..5, and the uncorrelated parameters u_1..5 of the shifts
        // (gaussian priors with means mu_u and unit variance); like = -chi^2/2 = -41.12300 (data only)
        const double omch2 = 0.11813, ombh2 = 0.02594, h = 0.76666, ns = 0.90077, As = 2.42206e-9, Ab = 2.85930,
                     AIA = 0.26447;
        const double bias[5] = {1.74311e-03, 8.93466e-03, -1.41311e-02, -1.55053e-02, 7.09853e-03};
        const double u[5] = {0.16470, 0.77302, -1.52139, -1.77832, 1.58341},
                     mu_u[5] = {0.000, -0.181, -1.110, -1.395, 1.265};
        KiDS1000COSEBIsLikelihood L(Utils::fml_file("CosmologyFitting/external/kids1000"));
        std::ostringstream os;
        os << std::setprecision(10);
        auto str = [&](double x) {
            os.str("");
            os << x;
            return os.str();
        };
        RunSettings s = ParameterFile::read("",
                                            {"precision=fast",
                                             "m_nu=0.06",
                                             "h=" + str(h),
                                             "OmegaB=" + str(ombh2 / (h * h)),
                                             "OmegaCDM=" + str(omch2 / (h * h)),
                                             "n_s=" + str(ns),
                                             "A_s=" + str(As),
                                             "nonlinear=hmcode2016",
                                             "hmcode2016_A=" + str(Ab),
                                             "hmcode2016_eta0_from_A=true"});
        TheoryOptions o;
        o.cmb = false;
        o.pk_nl_z = L.pk_nl_redshifts();
        Theory t(s, o);
        t.set_nuisance("kids_A_IA", AIA);
        for (int i = 0; i < 5; i++)
            t.set_nuisance("kids_dz" + std::to_string(i + 1), bias[i]);
        const double chi2 = L.chi2_data(t);
        std::cout << "  chi^2 " << chi2 << " (KiDS 82.24600), sigma_8 " << t.sigma8() << " (KiDS 0.83797)\n";
        check(std::fabs(chi2 - 82.24600) < 0.15, "maximum posterior: chi^2 of the data to 0.15 of the KiDS chain");
        check_rel(t.sigma8(), 0.83797, 3e-4, "maximum posterior: sigma_8 of the KiDS chain");
        double pu = 0;
        for (int i = 0; i < 5; i++)
            pu += (u[i] - mu_u[i]) * (u[i] - mu_u[i]);
        check_close(L.chi2_dz_prior(t), pu, 2e-4, "delta_z prior: chi^2 = that of the chain's uncorrelated shifts");
        check_close(-2.0 * L.loglike(t), chi2 + L.chi2_dz_prior(t), 1e-8, "ln L = -(chi^2 + delta_z prior)/2");
    }

    std::cout << "Modified lensing Sigma(z) and the Brans-Dicke aliases\n";
    {
        // Sigma = -k^2 (Psi - Phi) / (8 pi G_* a^2 rho_m Delta_M) of the full perturbations at sub-horizon k against
        // Theory::lensing_Sigma (the quasi-static g/phi of Brans-Dicke); 1 for LCDM
        for (const std::string m : {"lcdm", "jbd"}) {
            std::vector<std::string> kv = {"precision=fast", "compute_cells=false", "pert_k_max=1"};
            if (m == "jbd") {
                kv.push_back("model=jbd");
                kv.push_back("bd_omega=50");
            }
            const RunSettings s = ParameterFile::read("", kv);
            EinsteinBoltzmannSolver e(s);
            e.set_verbose(false);
            e.solve();
            TheoryOptions o;
            o.cmb = false;
            Theory t(s, o);
            const double Om = s.cosmo.OmegaB + s.cosmo.OmegaCDM, H0 = e.background().get_H0();
            double err = 0, dev = 0;
            for (double z : {0.0, 1.0, 3.0})
                for (double kMpc : {0.1, 0.5}) {
                    const double x = -std::log1p(z), k = kMpc / Mpc;
                    const double W = e.perturbations().get_Psi(x, k) - e.perturbations().get_Phi(x, k);
                    const double S = -(c * k) * (c * k) * W /
                                     (3.0 * H0 * H0 * Om * std::exp(-x) * e.power_spectrum().get_Delta_M(x, k));
                    err = std::max(err, std::fabs(S / t.lensing_Sigma(z) - 1.0));
                    dev = std::max(dev, std::fabs(t.lensing_Sigma(z) - 1.0));
                }
            std::cout << "  " << m << ": max |Sigma - 1| = " << dev << ", perturbations vs quasi-static " << err << "\n";
            check(err < 1e-4 && (m == "lcdm" ? dev == 0.0 : dev > 0.01),
                  m + ": Sigma(z) of the lensing kernel = that of the full perturbations (1e-4)");
        }
        SampledParameterMap pm({"model=jbd"}, {"log10_bd_omega", "bd_GeffG_today"});
        check_rel(pm.settings({3.5, 0.97}).cosmo.model_param("bd_omega"), std::pow(10.0, 3.5), 1e-14,
                  "log10_bd_omega sets bd_omega");
        SampledParameterMap pi({"model=jbd"}, {"inv_bd_omega"});
        check_rel(pi.settings({0.002}).cosmo.model_param("bd_omega"), 500.0, 1e-14, "inv_bd_omega sets bd_omega = 1/x");
        check(pi.settings({0.0}).cosmo.model_param("bd_omega") == 1e12 && throws([&] { pi.settings({-0.1}); }),
              "inv_bd_omega = 0 is GR (bd_omega = 1e12), < 0 rejected");
        check(throws([&] { SampledParameterMap({"model=jbd"}, {"inv_bd_omega", "bd_omega"}); }),
              "bd_omega sampled once");
    }

    return FML::TESTS::summary("CosmologyFitting fitting");
}
