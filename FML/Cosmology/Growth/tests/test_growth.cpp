// Tests of Cosmology/Growth/LPTGrowthFactors.h:
//   - Einstein-de Sitter: D1 = a/a_norm, D2/D1^2 = -3/7, D3a/D1^3 = -1/3, D3b/D1^3 = 5/21, f1 = 1, f2 = 2
//   - matter + radiation (no Lambda): D1 = Meszaros growing mode 1 + 3y/2, y = a/a_eq (start moved into the radiation era)
//   - LCDM background: history() = D1, D2; a QuasiStaticGravity with mu = 1 gives GR
//   - Hu-Sawicky f(R) (scale dependent): GR on large scales, 4/3 G on small scales (more growth)
//   - DGP: the second-order source N < 1 makes |D2| smaller
//   - EdS with a constant velocity friction Gamma: power-law growing modes D1 = a^p, D2 = c2 D1^2, D3 = c3 D1^3
#include <FML/Cosmology/Background/BackgroundModel_LCDM.h>
#include <FML/Cosmology/Growth/LPTGrowthFactors.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_DGP.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

using namespace FML::COSMOLOGY;
namespace {
    int nfail = 0, ncheck = 0;
    void check(double err, double tol, const std::string & what) {
        ncheck++;
        const bool ok = std::isfinite(err) and std::fabs(err) <= tol;
        if (not ok)
            nfail++;
        std::printf("[%s] %-70s err = %.3e (tol %.1e)\n", ok ? "PASS" : "FAIL", what.c_str(), err, tol);
    }
} // namespace

int main() {
    // EdS
    {
        GROWTH::LPTGrowthEquations eq;
        eq.E = [](double x) { return std::exp(-1.5 * x); };
        eq.dlogEdx = [](double) { return -1.5; };
        eq.matter_density = [](double x) { return std::exp(-3.0 * x); };
        GROWTH::LPTGrowthSettings s;
        s.a_min = 1e-3;
        s.a_max = 2.0;
        s.a_normalize = 0.02;
        GROWTH::LPTGrowthFactors g(eq, s);
        double e1 = 0, e2 = 0, e3a = 0, e3b = 0, ef = 0;
        for (double a : {0.01, 0.1, 0.5, 1.0, 1.5}) {
            const double x = std::log(a), D1 = g.D1(x);
            e1 = std::max(e1, std::fabs(D1 / (a / 0.02) - 1));
            e2 = std::max(e2, std::fabs(g.D2(x) / (D1 * D1) / (-3.0 / 7.0) - 1));
            e3a = std::max(e3a, std::fabs(g.D3a(x) / (D1 * D1 * D1) / (-1.0 / 3.0) - 1));
            e3b = std::max(e3b, std::fabs(g.D3b(x) / (D1 * D1 * D1) / (5.0 / 21.0) - 1));
            ef = std::max({ef, std::fabs(g.f1(x) - 1), std::fabs(g.f2(x) - 2), std::fabs(g.f3a(x) - 3)});
        }
        check(e1, 1e-6, "EdS: D1 = a / a_normalize");
        check(e2, 1e-6, "EdS: D2 / D1^2 = -3/7");
        check(e3a, 1e-6, "EdS: D3a / D1^3 = -1/3");
        check(e3b, 1e-6, "EdS: D3b / D1^3 = 5/21");
        check(ef, 1e-5, "EdS: f1 = 1, f2 = 2, f3a = 3");
        auto h = g.history();
        check(std::fabs(h.linear(0.5) / g.D1(std::log(0.5)) - 1) + std::fabs(h.second_order_rate(0.5) - 2), 1e-5,
              "EdS: history() = D1, f2");
    }
    // Matter + radiation: the Meszaros growing mode
    {
        const double aeq = 1e-3;
        GROWTH::LPTGrowthEquations eq;
        eq.E = [aeq](double x) {
            const double a = std::exp(x);
            return std::sqrt((1.0 / (a * a * a) + aeq / (a * a * a * a)) / (1.0 + aeq));
        };
        eq.dlogEdx = [aeq](double x) {
            const double y = std::exp(x) / aeq;
            return -0.5 * (3.0 * y + 4.0) / (y + 1.0);
        };
        eq.matter_density = [aeq](double x) { return std::exp(-3.0 * x) / (1.0 + aeq); };
        GROWTH::LPTGrowthSettings s;
        s.a_min = 1e-3; // y = 1: moved to y = 0.1
        s.a_max = 1.0;
        s.a_normalize = 1.0;
        GROWTH::LPTGrowthFactors g(eq, s);
        check(std::fabs(g.a_min() / aeq - 0.1), 1e-10, "radiation: a_min moved to a_eq / 10");
        auto mes = [aeq](double a) { return 1.0 + 1.5 * a / aeq; };
        double e = 0;
        for (double a : {1e-4, 2e-4, 1e-3, 1e-2, 0.3})
            e = std::max(e, std::fabs(g.D1(std::log(a)) / (mes(a) / mes(1.0)) - 1));
        check(e, 1e-6, "radiation: D1 = 1 + 3y/2 (Meszaros)");
        // D2 tends to -3/7 D1^2 in the matter era
        const double D1 = g.D1(0.0);
        check(std::fabs(g.D2(0.0) / (D1 * D1) / (-3.0 / 7.0) - 1), 2e-3, "radiation: D2 -> -3/7 D1^2 for a >> a_eq");
    }
    // LCDM, GR and mu = 1
    BACKGROUND::BackgroundParameters p;
    p.h = 0.67;
    p.Omega_b = 0.049;
    p.Omega_cdm = 0.26;
    auto bg = BACKGROUND::make_lcdm(p);
    GROWTH::LPTGrowthSettings s;
    s.a_normalize = 0.02;
    GROWTH::LPTGrowthFactors gr(*bg, s);
    {
        GROWTH::LPTGrowthFactors mu1(*bg, s, std::make_shared<PERTURBATIONS::QuasiStaticGravity_MuSigma>(*bg, 0.0));
        check(std::fabs(mu1.D2(0.0) / gr.D2(0.0) - 1), 1e-14, "LCDM: QuasiStaticGravity with mu = 1 = GR");
        const double D1 = gr.D1(0.0);
        check(std::fabs(gr.D2(0.0) / (D1 * D1) / (-3.0 / 7.0) - 1) < 0.02 ? 0.0 : 1.0, 0.0,
              "LCDM: D2/D1^2 within 2% of -3/7 today");
        check(std::fabs(gr.history().second_order(0.5) / gr.D2(std::log(0.5)) - 1), 1e-14, "LCDM: history()");
    }
    // f(R)
    {
        auto fs = s;
        fs.scale_dependent = true;
        fs.n_k = 40;
        GROWTH::LPTGrowthFactors fr(*bg, fs, std::make_shared<PERTURBATIONS::QuasiStaticGravity_HuSawicky>(*bg, 1e-5, 1.0));
        const double klow = fs.koverH0_min, khigh = fs.koverH0_max;
        check(std::fabs(fr.D1(0.0, 0.0) / gr.D1(0.0) - 1), 3e-5, "f(R): D1 at k = 1e-4 h/Mpc = GR (+ O(|f_R0|))");
        check(fr.D1(0.0, 0.0) == fr.D1(0.0, klow) ? 0.0 : 1.0, 0.0, "f(R): k below the grid = its first point");
        const double boost = fr.D1(0.0, khigh) / gr.D1(0.0);
        std::printf("       f(R) D1(k = 100 h/Mpc) / D1_GR today = %.4f\n", boost);
        // mu = 4/3 on these scales since a ~ 0.02: D ~ a^1.137 instead of a in the matter era
        check(boost > 1.3 and boost < 1.7 ? 0.0 : 1.0, 0.0, "f(R): more growth on small scales (30-70% at 100 h/Mpc)");
    }
    // DGP: the second-order source
    {
        auto qs = std::make_shared<PERTURBATIONS::QuasiStaticGravity_DGP>(*bg, 1.0, -1);
        GROWTH::LPTGrowthFactors full(*bg, s, qs);
        auto eq = GROWTH::LPTGrowthEquations::from_background(*bg);
        eq.mu = [qs](double x, double k) { return qs->mu(x, k); };
        GROWTH::LPTGrowthFactors nosource(eq, s);
        check(std::fabs(full.D1(0.0) / nosource.D1(0.0) - 1), 1e-10, "DGP: D1 does not depend on the 2LPT source");
        const double r = full.D2(0.0) / nosource.D2(0.0);
        std::printf("       DGP (rcH0 = 1) D2 with / without the second-order source today = %.5f\n", r);
        check(r < 1.0 and r > 0.95 ? 0.0 : 1.0, 0.0, "DGP: the second-order source makes |D2| smaller");
    }
    // EdS with a constant velocity friction Gamma (dv/dlna = -Gamma v + force): the growing mode is D1 = a^p with
    // p^2 + (1/2 + Gamma) p = 3/2, and with Q_n = (n p)^2 + (1/2 + Gamma) n p:
    // D2 = c2 D1^2, c2 (Q_2 - 3/2) = -3/2; D3a = c3a D1^3, c3a (Q_3 - 3/2) = -3; D3b = c3b D1^3, c3b (Q_3 - 3/2) =
    // 3/2 (1 - c2). (Gamma = 0: p = 1, -3/7, -1/3, 5/21.) The EdS initial conditions are not those of these modes: the
    // start is early enough for what they excite (in D2, D3 the homogeneous a^p, relative to a^2p, a^3p) to have died
    // out by a = 0.01
    for (double Gamma : {0.0, 0.7, -0.3}) {
        GROWTH::LPTGrowthEquations eq;
        eq.E = [](double x) { return std::exp(-1.5 * x); };
        eq.dlogEdx = [](double) { return -1.5; };
        eq.matter_density = [](double x) { return std::exp(-3.0 * x); };
        if (Gamma != 0.0)
            eq.friction = [Gamma](double) { return Gamma; };
        GROWTH::LPTGrowthSettings s;
        s.a_min = 1e-12;
        s.a_max = 2.0;
        s.n_a = 4000;
        s.a_normalize = 1.0;
        GROWTH::LPTGrowthFactors g(eq, s);
        const double b = 0.5 + Gamma;
        const double p = 0.5 * (-b + std::sqrt(b * b + 6.0));
        const double Q2 = 4 * p * p + 2 * b * p, Q3 = 9 * p * p + 3 * b * p;
        const double c2 = -1.5 / (Q2 - 1.5), c3a = -3.0 / (Q3 - 1.5), c3b = 1.5 * (1 - c2) / (Q3 - 1.5);
        double e1 = 0, e2 = 0, e3 = 0, ef = 0;
        for (double a : {0.01, 0.1, 0.5, 1.0, 1.5}) {
            const double x = std::log(a), D1 = g.D1(x);
            e1 = std::max(e1, std::fabs(D1 / std::pow(a, p) - 1));
            e2 = std::max(e2, std::fabs(g.D2(x) / (D1 * D1) / c2 - 1));
            e3 = std::max({e3,
                           std::fabs(g.D3a(x) / (D1 * D1 * D1) / c3a - 1),
                           std::fabs(g.D3b(x) / (D1 * D1 * D1) / c3b - 1)});
            ef = std::max({ef, std::fabs(g.f1(x) / p - 1), std::fabs(g.f2(x) / (2 * p) - 1)});
        }
        char G[64];
        std::snprintf(G, sizeof G, "EdS + friction %+.1f: ", Gamma);
        check(e1, 1e-6, std::string(G) + "D1 = a^p");
        check(e2, 1e-6, std::string(G) + "D2 / D1^2 = c2");
        check(e3, 1e-6, std::string(G) + "D3a, D3b / D1^3 = c3a, c3b");
        check(ef, 1e-5, std::string(G) + "f1 = p, f2 = 2p");
    }

    if (nfail == 0)
        std::cout << "All growth tests passed (" << ncheck << " checks)\n";
    else
        std::cout << nfail << " of " << ncheck << " growth tests FAILED\n";
    return nfail == 0 ? 0 : 1;
}
