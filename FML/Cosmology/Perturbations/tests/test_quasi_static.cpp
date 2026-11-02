// Tests of the quasi-static theories (QuasiStaticGravity*.h) on an LCDM background: mu limits and the screening
//   - Hu-Sawicky: mu -> 1/(1 + f_R) and 4/3 of it for k << m, k >> m; Phi_crit = 3/2 |f_R(a)|; min(1, Phi_crit/|Phi|)
//   - DGP: mu = 1 + 1/(3 beta); the Vainshtein factor -> 1 for small e, ~ 2/sqrt(e) for large e
//   - symmetron: no fifth force and no screening factor before symmetry breaking; mu(k >> m) = 1 + 2 beta^2 phi^2
//   - (m(a), beta(a)): phi(a)/M_pl = 9 Omega_m beta0 a^p / (m0^2 p) (power laws), Phi_crit = phi / (2 beta)
//   - mu-Sigma and Brans-Dicke: no screening
#include <FML/Cosmology/Background/BackgroundModel_LCDM.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_DGP.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MBeta.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Symmetron.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

using namespace FML::COSMOLOGY;
using namespace FML::COSMOLOGY::PERTURBATIONS;
namespace {
    int nfail = 0, ncheck = 0;
    void check(double err, double tol, const std::string & what) {
        ncheck++;
        const bool ok = std::isfinite(err) and std::fabs(err) <= tol;
        if (not ok)
            nfail++;
        std::printf("[%s] %-72s err = %.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", what.c_str(), err, tol);
    }
    double rel(double a, double b) { return std::fabs(a / b - 1.0); }
} // namespace

int main() {
    BACKGROUND::BackgroundParameters p;
    p.h = 0.67;
    p.Omega_b = 0.049;
    p.Omega_cdm = 0.26;
    auto bg = BACKGROUND::make_lcdm(p);

    {
        QuasiStaticGravity_HuSawicky hs(*bg, 1e-5, 1.0);
        const double x = std::log(0.5), m = std::sqrt(hs.mass2(x)) * 0.5, fR = hs.fR(x);
        check(rel(hs.mu(x, 1e-3 * m), 1.0 / (1.0 + fR)), 1e-6, "f(R): mu(k << a m) = 1/(1 + f_R)");
        check(rel(hs.mu(x, 1e3 * m), 4.0 / 3.0 / (1.0 + fR)), 1e-6, "f(R): mu(k >> a m) = 4/3 / (1 + f_R)");
        const auto sc = hs.screening();
        check(sc.type == ScreeningType::Potential ? 0.0 : 1.0, 0.0, "f(R): screening by the potential");
        check(rel(sc.threshold(x), 1.5 * std::fabs(fR)), 1e-14, "f(R): Phi_crit = 3/2 |f_R(a)|");
        const double pc = sc.threshold(x);
        check(std::fabs(sc.factor(x, 0.1 * pc, 1.0) - 1.0) + rel(sc.factor(x, -10.0 * pc, 1.0), 0.1) +
                  rel(sc.factor(x, 10.0 * pc, 2.0), 0.2),
              1e-14,
              "f(R): factor = min(1, efficiency Phi_crit / |Phi_N|)");
    }
    {
        QuasiStaticGravity_DGP dgp(*bg, 1.0, -1);
        const double x = std::log(0.7), b = dgp.beta(x);
        check(rel(dgp.mu(x, 0.0), 1.0 + 1.0 / (3.0 * b)), 1e-15, "DGP: mu = 1 + 1/(3 beta)");
        const auto sc = dgp.screening();
        check(sc.type == ScreeningType::Density ? 0.0 : 1.0, 0.0, "DGP: screening by the density");
        const double c = 8.0 * bg->Omega_m(0.0) / (9.0 * b * b);
        check(std::fabs(sc.factor(x, 1e-8, 1.0) - 1.0), 1e-14, "DGP: factor -> 1 for small densities");
        const double rho = 1e4, e = c * rho;
        check(rel(sc.factor(x, rho, 1.0), 2.0 * (std::sqrt(1.0 + e) - 1.0) / e), 1e-14,
              "DGP: factor = 2 (sqrt(1 + e) - 1)/e, e = 8 Omega_m (r_c H0)^2 rho / (9 beta^2)");
    }
    {
        QuasiStaticGravity_Symmetron sy(*bg, 0.5, 1.0, 1.0);
        const auto sc = sy.screening();
        const double xb = std::log(0.4), xa = std::log(0.8);
        check(std::fabs(sy.mu(xb, 1e3) - 1.0) + sc.factor(xb, 1e-6, 1.0), 1e-15,
              "symmetron: no fifth force and no factor before symmetry breaking");
        const double ph = sy.phi(xa);
        check(rel(sy.mu(xa, 1e8), 1.0 + 2.0 * ph * ph), 1e-8, "symmetron: mu(k >> a m) = 1 + 2 beta^2 (phi/phi0)^2");
        const double L = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
        check(rel(sc.threshold(xa), 3.0 * bg->Omega_m(0.0) * L * L / 0.125), 1e-14,
              "symmetron: Phi_crit = 3 Omega_m (H0 L)^2 / a_ssb^3");
    }
    {
        const double beta0 = 1.0, nb = 0.0, m0 = 1.0, nm = -2.0; // p = 0 - 3 + 4 = 1
        QuasiStaticGravity_MBeta mb(*bg, beta0, nb, m0, nm);
        double e = 0.0;
        for (double a : {0.01, 0.1, 0.5, 1.0})
            e = std::max(e, rel(mb.phi_over_Mpl(a), 9.0 * bg->Omega_m(0.0) * beta0 * a / (m0 * m0)));
        check(e, 1e-3, "(m, beta): phi(a)/M_pl = 9 Omega_m beta0 a^p / (m0^2 p)");
        const auto sc = mb.screening();
        check(rel(sc.threshold(0.0), mb.phi_over_Mpl(1.0) / 2.0), 1e-14, "(m, beta): Phi_crit = phi / (2 beta)");
        QuasiStaticGravity_MBeta bad(*bg, 1.0, 0.0, 1.0, 0.0); // p = -3
        bool threw = false;
        try {
            bad.screening();
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw ? 0.0 : 1.0, 0.0, "(m, beta): screening throws when the phi integral diverges");
    }
    {
        QuasiStaticGravity_MuSigma ms(*bg, 0.1, 0.1);
        check(ms.screening() ? 1.0 : 0.0, 0.0, "mu-Sigma: no screening");
    }
    if (nfail == 0)
        std::cout << "All quasi-static gravity tests passed (" << ncheck << " checks)\n";
    else
        std::cout << nfail << " of " << ncheck << " quasi-static gravity tests FAILED\n";
    return nfail == 0 ? 0 : 1;
}
