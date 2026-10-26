//==========================================================
// Tests for FML::COSMOLOGY::HALOMODEL (HaloModel/Halomodel.h):
//   - HMCode P(k, z = 0) against the HMCode output in HaloModel/example (pofk_nl_hmcode.txt) for 0.01 < k < 100
//   - the Fourier transform of a profile (numerical, a uniform sphere: 3 j1(kR)/(kR)) and the analytic NFW transform
//     against a direct integration
//   - the mass function dn/dlnM at z = 1 against a direct computation (sigma(R) from P(k), comoving R(M))
//   - calling compute_at_redshift again gives the same result (also with cofM_multiplier), a copy works
//   - z = 20 with HMCode works; errors: z beyond the spherical collapse splines, P(k) output outside the computed
//     range, P(k) = 0 in the input file
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/HaloModel/Halomodel.h>
#include <FML/FileUtils/FileUtils.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace FML::COSMOLOGY::HALOMODEL;
using SphericalCollapseModel = FML::COSMOLOGY::SPHERICALCOLLAPSE::SphericalCollapseModel;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(6);
    os << x;
    return os.str();
}

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
double pnl(const HaloModel & hm, double k) {
    return 2 * M_PI * M_PI / (k * k * k) * std::exp(hm.logDeltaHM_full_of_logk_spline(std::log(k)));
}

int main() {
    Spline logDelta;
    read_pofk_file(folder + "pofk_lin.txt", logDelta);
    const auto scm = lcdm();

    // HMCode at z = 0 against the reference
    {
        HaloModel hm(scm, logDelta, 0.0, true, false);
        hm.compute_at_redshift(0.0);
        auto ref = FML::FILEUTILS::loadtxt(folder + "pofk_nl_hmcode.txt");
        double maxerr = 0.0;
        for (auto & row : ref) {
            if (row[0] < 0.01 or row[0] > 100.0)
                continue;
            maxerr = std::max(maxerr, std::fabs(pnl(hm, row[0]) / row[1] - 1.0));
        }
        check(maxerr < 0.004, "HMCode P(k, z=0) = reference HMCode output for 0.01 < k < 100 h/Mpc (max error " +
                                  num(100 * maxerr) + "%, was 1.6%)");

        // A copy
        HaloModel copy = hm;
        copy.compute_at_redshift(0.5);
        hm.compute_at_redshift(0.5);
        check(std::fabs(pnl(copy, 1.0) / pnl(hm, 1.0) - 1) < 1e-12, "a copy of a HaloModel gives the same result");

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
            Spline logDelta50;
            read_pofk_file(fn, logDelta50);
            HaloModel hm50(scm, logDelta50, 0.0, true, false);
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
        check(ok, "HMCode at z = 20 with P(k) up to k = 50 h/Mpc (crashed before)");

        // Output outside the computed k-range
        bool thrown = false;
        try {
            hm.output_pofk("test_halomodel_pofk.txt", 1e-4, 200.0, 10);
        } catch (std::exception &) {
            thrown = true;
        }
        std::remove("test_halomodel_pofk.txt");
        check(thrown, "P(k) output outside [kmin_pofk, kmax_pofk] throws (P(k) is wrong there)");
    }

    // The mass function at z = 1 against a direct computation
    {
        HaloModel hm(scm, logDelta, 0.0, true, false);
        hm.compute_at_redshift(1.0);
        const double D = hm.growthfactor_of_x_spline(-std::log(2.0)) / hm.growthfactor_of_x_spline(0.0);
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
        const double rho_mean = 2.77536627e11 * OmegaM; // comoving, h^2 Msun / Mpc^3
        double maxerr = 0.0;
        for (double M : {1e11, 1e13, 1e14, 1e15}) {
            auto nu_of_lnM = [&](double lnM) {
                const double R = std::cbrt(std::exp(lnM) / (4.0 * M_PI / 3.0 * rho_mean));
                return hm.deltac / sigma(R);
            };
            const double nu = nu_of_lnM(std::log(M));
            const double dnudlnM = (nu_of_lnM(std::log(M) + 0.01) - nu_of_lnM(std::log(M) - 0.01)) / 0.02;
            const double f = hm.A_ST * (1.0 + std::pow(hm.a_ST * nu * nu, -hm.p_ST)) * std::exp(-hm.a_ST * nu * nu / 2.0);
            const double dndlnM = rho_mean / M * f * dnudlnM;
            maxerr = std::max(maxerr, std::fabs(hm.dndlogM_of_logM_spline(std::log(M)) / dndlnM - 1.0));
        }
        check(maxerr < 0.01, "dn/dlnM at z = 1 = direct computation for M = 1e11-1e15 Msun/h (max error " +
                                 num(100 * maxerr) + "%, M(R) was (1+z)^3 too large)");
    }

    // Fourier transforms: uniform sphere numerically, NFW analytically
    {
        const double Rvir = 1.3, M = 1e13;
        Func2 sphere = [Rvir](double r, double) { return r > Rvir ? 0.0 : 3.0 * std::pow(r / Rvir, 3); };
        DVector k_array{1e-6, 0.3, 1.0, 3.0, 10.0, 30.0};
        auto y = compute_rho_fourier_single(sphere, k_array, M, Rvir);
        double maxerr = 0.0;
        for (size_t i = 1; i < k_array.size(); i++) {
            const double x = k_array[i] * Rvir;
            const double exact = 3 * (std::sin(x) - x * std::cos(x)) / (x * x * x);
            maxerr = std::max(maxerr, std::fabs(y[i] - exact));
        }
        check(maxerr < 1e-3, "numerical Fourier transform of a truncated profile (uniform sphere) (max error " +
                                 num(maxerr) + ")");

        HaloModel hm(scm, logDelta, 0.0, true, false);
        hm.compute_at_redshift(0.0);
        NFWHaloDensityProfile nfw;
        maxerr = 0.0;
        for (double Mh : {1e10, 1e13, 1e15}) {
            const double Rv = RvirofM(Mh, hm.DeltaVir, hm.OmegaM, hm.xcollapse);
            for (double kR : {0.1, 1.0, 10.0, 100.0}) {
                const double k = kR / Rv;
                // Direct integration of 4 pi r^2 rho j0(kr) / M (in log r, Simpson)
                const int n = 200000;
                const double l0 = std::log(Rv * 1e-8), l1 = std::log(Rv), h = (l1 - l0) / n;
                double s = 0.0;
                for (int i = 0; i <= n; i++) {
                    const double r = std::exp(l0 + i * h) * (i == n ? 1 - 1e-12 : 1.0);
                    const double w = (i == 0 or i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
                    s += w * nfw(r, Mh, &hm) * std::sin(k * r) / (k * r);
                }
                s *= h / 3.0;
                maxerr = std::max(maxerr, std::fabs(nfw.fourier(k, Mh, &hm) - s));
            }
        }
        check(maxerr < 1e-6, "analytic NFW Fourier transform = direct integration (max error " + num(maxerr) + ")");
    }

    // Without HMCode (deltac, DeltaVir from spherical collapse)
    {
        HaloModel hm(scm, logDelta, 0.0, false, false);

        // Again (same object) with a cofM_multiplier: the same result as a new object
        hm.cofM_multiplier = 1.5;
        hm.compute_at_redshift(0.0);
        const double p1 = pnl(hm, 10.0);
        hm.compute_at_redshift(0.0);
        hm.compute_at_redshift(0.0);
        check(std::fabs(pnl(hm, 10.0) / p1 - 1) < 1e-12,
              "compute_at_redshift again gives the same P(k) (cofM_multiplier = 1.5 is not applied again; was x1.5 "
              "for k = 10 on the third call)");

        bool thrown = false;
        try {
            hm.compute_at_redshift(11.0);
        } catch (std::exception & e) {
            thrown = std::string(e.what()).find("zmax_spherical_collapse") != std::string::npos;
        }
        check(thrown, "z beyond the spherical collapse splines throws (was silently using z = 10)");

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
        check(thrown, "P(k) = 0 in the input file throws a clear error (aborted in GSL before)");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All halo model tests passed\n" : std::to_string(fails) + " halo model test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
