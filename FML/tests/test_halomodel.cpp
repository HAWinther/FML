//==========================================================
// Tests for FML::COSMOLOGY::HALOMODEL (HaloModel/Halomodel.h, HaloModelTools.h):
//   - HMCode-2016 P(k, z = 0) against the HMCode output in HaloModel/example (pofk_nl_hmcode.txt) for 0.01 < k < 100
//     (CAMB mead2016 with neutrinos and w0wa: Cosmology/EinsteinBoltzmannSolver/tests/test_hmcode2016)
//   - the Fourier transform of a profile (numerical, a uniform sphere: 3 j1(kR)/(kR)) and the analytic NFW transform
//     against a direct integration
//   - the mass function dn/dlnM at z = 1 against a direct computation (sigma(R) from P(k), comoving R(M))
//   - calling compute_at_redshift again gives the same result (also with cofM_multiplier), a copy works
//   - z = 20 with HMCode works; errors: z beyond the spherical collapse splines, P(k) output outside the computed
//     range, P(k) = 0 in the input file
//   - HMcode-2016 does not change the options; a custom (numerically transformed) NFW profile = the analytic one;
//     the tools: NFW transform series, Sheth-Tormen normalisation, LinearVariance
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/HaloModel/Halomodel.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;

using namespace FML::COSMOLOGY::HALOMODEL;
using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;


const std::string folder = "../HaloModel/example/";
const double OmegaM = 0.295;

SphericalCollapseModel lcdm() {
    Func1 E = [](double x) { return std::sqrt(OmegaM * std::exp(-3 * x) + 1 - OmegaM); };
    Func1 Om = [E](double x) { return OmegaM * std::exp(-3 * x) / (E(x) * E(x)); };
    Func1 dlogE = [Om](double x) { return -1.5 * Om(x); };
    Func1 mu = [](double) { return 1.0; };
    Func1 w = [](double) { return -1.0; };
    return SphericalCollapseModel(E, Om, dlogE, mu, w);
}

// P_NL(k) from the halo model
double pnl(const HaloModel & hm, double k) { return hm.pofk(k); }

HaloModelOptions hmcode2016() {
    HaloModelOptions o;
    o.prescription = Prescription::HMcode2016;
    return o;
}

int main() {
    const auto scm = lcdm();
    const auto cosmology = cosmology_from_pofk_file(scm, folder + "pofk_lin.txt");

    // HMCode at z = 0 against the reference
    {
        HaloModel hm(cosmology, hmcode2016());
        hm.compute_at_redshift(0.0);
        auto ref = FML::FILEUTILS::loadtxt(folder + "pofk_nl_hmcode.txt");
        double maxerr = 0.0;
        for (auto & row : ref) {
            if (row[0] < 0.01 or row[0] > 100.0)
                continue;
            maxerr = std::max(maxerr, std::fabs(pnl(hm, row[0]) / row[1] - 1.0));
        }
        check_local(maxerr < 0.004,
                    "HMCode P(k, z=0) = reference HMCode output for 0.01 < k < 100 h/Mpc (max error " +
                        num(100 * maxerr) + "%, was 1.6%)");

        // A copy
        HaloModel copy = hm;
        copy.compute_at_redshift(0.5);
        hm.compute_at_redshift(0.5);
        check_local(std::fabs(pnl(copy, 1.0) / pnl(hm, 1.0) - 1) < 1e-12,
                    "a copy of a HaloModel gives the same result");

        // Very high redshift with HMCode and P(k) given up to k = 50 h/Mpc only (nu = 1 is then not in the range
        // where neff is computed)
        bool ok = true;
        const std::string fn = "test_halomodel_pofk_kmax50.txt";
        if (FML::ThisTask == 0) {
            std::ofstream fp(fn);
            fp.precision(12);
            for (auto & row : FML::FILEUTILS::loadtxt(folder + "pofk_lin.txt"))
                if (row[0] <= 50.0)
                    fp << row[0] << " " << row[1] << "\n";
        }
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        try {
            HaloModel hm50(cosmology_from_pofk_file(scm, fn), hmcode2016());
            hm50.compute_at_redshift(20.0);
            ok = std::isfinite(pnl(hm50, 1.0)) and pnl(hm50, 1.0) > 0.0;
        } catch (...) {
            ok = false;
        }
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        if (FML::ThisTask == 0)
            std::remove(fn.c_str());
        check_local(ok, "HMCode at z = 20 with P(k) up to k = 50 h/Mpc (crashed before)");

        // Output outside the computed k-range
        bool thrown = false;
        try {
            hm.output_pofk("test_halomodel_pofk.txt", 1e-4, 200.0, 10);
        } catch (std::exception &) {
            thrown = true;
        }
        std::remove("test_halomodel_pofk.txt");
        check_local(thrown, "P(k) output outside [kmin_pofk, kmax_pofk] throws (P(k) is wrong there)");
    }

    // The mass function at z = 1 against a direct computation
    {
        HaloModel hm(cosmology, hmcode2016());
        hm.compute_at_redshift(1.0);
        const double D = hm.growth_factor(-std::log(2.0)) / hm.growth_factor(0.0);
        auto pk = FML::FILEUTILS::loadtxt(folder + "pofk_lin.txt");
        auto sigma = [&](double R) {
            double s2 = 0.0;
            for (size_t i = 1; i < pk.size(); i++) {
                auto f = [&](size_t j) {
                    const double kR = pk[j][0] * R;
                    const double W = kR < 1e-4 ? 1.0 : 3 * (std::sin(kR) - kR * std::cos(kR)) / (kR * kR * kR);
                    return pk[j][1] * std::pow(pk[j][0], 3) / (2 * M_PI * M_PI) * W * W;
                };
                s2 += 0.5 * (f(i) + f(i - 1)) * std::log(pk[i][0] / pk[i - 1][0]);
            }
            return D * std::sqrt(s2);
        };
        const double rho_mean = FML::UTILS::SI::rho_crit_over_h2_msun_mpc3 * OmegaM; // comoving, h^2 Msun / Mpc^3
        double maxerr = 0.0;
        for (double M : {1e11, 1e13, 1e14, 1e15}) {
            auto nu_of_lnM = [&](double lnM) {
                const double R = std::cbrt(std::exp(lnM) / (4.0 * M_PI / 3.0 * rho_mean));
                return hm.state().deltac / sigma(R);
            };
            const double nu = nu_of_lnM(std::log(M));
            const double dnudlnM = (nu_of_lnM(std::log(M) + 0.01) - nu_of_lnM(std::log(M) - 0.01)) / 0.02;
            const auto & st = hm.get_options().sheth_tormen;
            const double f = st.A * (1.0 + std::pow(st.a * nu * nu, -st.p)) * std::exp(-st.a * nu * nu / 2.0);
            const double dndlnM = rho_mean / M * f * dnudlnM;
            maxerr = std::max(maxerr, std::fabs(hm.dndlogM(M) / dndlnM - 1.0));
        }
        check_local(maxerr < 0.01,
                    "dn/dlnM at z = 1 = direct computation for M = 1e11-1e15 Msun/h (max error " + num(100 * maxerr) +
                        "%, M(R) was (1+z)^3 too large)");
    }

    // Fourier transforms: uniform sphere numerically, NFW analytically
    {
        const double Rvir = 1.3, M = 1e13;
        Func1 sphere = [Rvir](double r) { return r > Rvir ? 0.0 : 3.0 * std::pow(r / Rvir, 3); };
        DVector k_array{1e-6, 0.3, 1.0, 3.0, 10.0, 30.0};
        auto y = fourier_transform_profile(sphere, k_array, Rvir);
        double maxerr = 0.0;
        for (size_t i = 1; i < k_array.size(); i++) {
            const double x = k_array[i] * Rvir;
            const double exact = 3 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
            maxerr = std::max(maxerr, std::fabs(y[i] - exact));
        }
        (void)M;
        check_local(maxerr < 1e-3,
                    "numerical Fourier transform of a truncated profile (uniform sphere) (max error " + num(maxerr) +
                        ")");

        HaloModel hm(cosmology, hmcode2016());
        hm.compute_at_redshift(0.0);
        maxerr = 0.0;
        double maxerr_spline = 0.0;
        for (double Mh : {1e10, 1e13, 1e15}) {
            const double Rv = hm.Rvir_of_M(Mh), c = hm.concentration(Mh);
            for (double kR : {0.1, 1.0, 10.0, 100.0}) {
                const double k = kR / Rv;
                // Direct integration of 4 pi r^2 rho j0(kr) / M (in log r, Simpson)
                const int n = 200000;
                const double l0 = std::log(Rv * 1e-8), l1 = std::log(Rv), h = (l1 - l0) / n;
                double s = 0.0;
                for (int i = 0; i <= n; i++) {
                    const double r = std::exp(l0 + i * h) * (i == n ? 1 - 1e-12 : 1.0);
                    const double w = (i == 0 or i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
                    s += w * nfw_profile_r3(r, Rv, c) * std::sin(k * r) / (k * r);
                }
                s *= h / 3.0;
                maxerr = std::max(maxerr, std::fabs(nfw_fourier(k * Rv, c) - s));
                if (kR <= 10.0)
                    maxerr_spline = std::max(maxerr_spline, std::fabs(hm.profile_fourier(k, Mh) - s));
            }
        }
        check_local(maxerr < 1e-6,
                    "analytic NFW Fourier transform = direct integration (max error " + num(maxerr) + ")");
        check_local(maxerr_spline < 2e-3,
                    "the halo model's u(k|M) spline = direct integration for k Rvir <= 10 (max error " +
                        num(maxerr_spline) + ")");

        // The small-k series of the NFW transform joins the closed form
        double jump = 0.0;
        for (double c : {1.0, 5.0, 30.0})
            jump = std::max(jump, std::fabs(nfw_fourier(0.01 * (1 - 1e-9), c) - nfw_fourier(0.01 * (1 + 1e-9), c)));
        check_local(jump < 1e-13, "NFW transform: series and closed form agree at k Rvir = 0.01 (" + num(jump) + ")");
        check_local(std::fabs(ShethTormen::normalisation(0.707, 0.3) / 0.21616 - 1) < 1e-4,
                    "Sheth-Tormen normalisation (0.21616 in HMcode)");

        // LinearVariance with P = A e^(-k^2): sigma^2(R -> 0) = int Delta^2 dln k = A sqrt(pi) / (8 pi^2) and
        // sigma_v^2 = int Delta^2 / k^2 dln k / 3 = A sqrt(pi) / (12 pi^2)
        const double A = 3.0;
        LinearVariance var([&](double k) { return A * k * k * k * std::exp(-k * k) / (2 * M_PI * M_PI); }, 1e-6, 1e3,
                           4000);
        check_local(std::fabs(var.sigma2(1e-8) / (A * std::sqrt(M_PI) / (8 * M_PI * M_PI)) - 1) < 1e-8,
                    "LinearVariance: sigma^2(R -> 0) = int Delta^2 dln k");
        check_local(std::fabs(var.sigma_v2() / (A * std::sqrt(M_PI) / (12 * M_PI * M_PI)) - 1) < 2e-6, // (k > 1e-6)
                    "LinearVariance: sigma_v^2 = int Delta^2 / k^2 dln k / 3");
        const double R = 2.0, e = 1e-4;
        const double dnum = (std::log(var.sigma(R * std::exp(e))) - std::log(var.sigma(R * std::exp(-e)))) / (2 * e);
        check_local(std::fabs(var.dlnsigma_dlnR(R) / dnum - 1) < 1e-6, "LinearVariance: dln sigma / dln R");
    }

    // HMcode-2016 sets its parameters in the state, not in the options; a custom profile
    {
        auto options = hmcode2016();
        options.kstar = 5.0;
        options.alpha = 1.3;
        HaloModel hm(cosmology, options);
        hm.compute_at_redshift(0.0);
        check_local(hm.get_options().kstar == 5.0 and hm.get_options().alpha == 1.3 and
                        std::fabs(hm.state().kstar - 0.584 / hm.state().sigmav) < 1e-12,
                    "HMcode-2016 does not overwrite the options (kstar, alpha were overwritten before)");

        // The NFW profile as a custom real-space profile (numerical transform) gives the same P(k)
        auto custom = options;
        custom.density_profile = [](double r, double M, const HaloModel & h) {
            return nfw_profile_r3(r, h.Rvir_of_M(M), h.concentration(M));
        };
        HaloModel hmc(cosmology, custom);
        hmc.compute_at_redshift(0.0);
        double maxerr = 0.0;
        for (double k : {0.1, 1.0, 10.0, 30.0})
            maxerr = std::max(maxerr, std::fabs(hmc.pofk(k) / hm.pofk(k) - 1));
        check_local(maxerr < 2e-3,
                    "custom (numerical) NFW profile = analytic NFW for k <= 30 h/Mpc (max error " + num(maxerr) + ")");
    }

    // Without HMCode (deltac, DeltaVir from spherical collapse)
    {
        HaloModelOptions options;
        options.concentration_multiplier = 1.5;
        HaloModel hm(cosmology, options);

        // Again (same object) with a concentration multiplier: the same result
        hm.compute_at_redshift(0.0);
        const double p1 = pnl(hm, 10.0);
        hm.compute_at_redshift(0.0);
        hm.compute_at_redshift(0.0);
        check_local(
            std::fabs(pnl(hm, 10.0) / p1 - 1) < 1e-12,
            "compute_at_redshift again gives the same P(k) (concentration multiplier 1.5 is not applied again; was x1.5 "
            "for k = 10 on the third call)");

        bool thrown = false;
        try {
            hm.compute_at_redshift(11.0);
        } catch (std::exception & e) {
            thrown = std::string(e.what()).find("zmax_spherical_collapse") != std::string::npos;
        }
        check_local(thrown, "z beyond the spherical collapse splines throws (was silently using z = 10)");

        const std::string fn = "test_halomodel_zero.txt";
        if (FML::ThisTask == 0) {
            std::ofstream fp(fn);
            fp << "0.01 100.0\n0.1 0.0\n1.0 10.0\n";
        }
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        thrown = false;
        try {
            Spline s;
            read_pofk_file(fn, s);
        } catch (std::exception & e) {
            thrown = std::string(e.what()).find("P(k) <= 0") != std::string::npos;
        }
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        if (FML::ThisTask == 0)
            std::remove(fn.c_str());
        check_local(thrown, "P(k) = 0 in the input file throws a clear error (aborted in GSL before)");
    }

    return FML::TESTS::summary("halo model");
}
