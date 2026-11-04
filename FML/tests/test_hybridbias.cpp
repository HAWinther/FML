//==========================================================
// Tests of the one-loop LPT predictions of the hybrid bias spectra (ComputePowerSpectra/HybridBiasOneLoop*.h):
//   - the modified gravity loop functions with GR in Einstein-de Sitter (exact EdS kernels from the growth ODEs)
//     reproduce the EdS loop functions (Matsubara's Q_n, R_n as combined in velocileptors)
//   - the scale-dependent second and third order LPT kernels c2(k,p), c2(k,-p), c3(k,p,-p) agree with an
//     independent computation from the Eulerian fluid equations (the same quasi-static equations, the Eulerian
//     F2, F3 solved as ODEs and mapped to LPT kinematically; computed with scipy, values below) for f(R) with and
//     without screening, DGP and a mu(k,a) model
//   - separate filters for the operators and the displacement: the loop functions with an operator leg factor
//     (Gaussian smoothing) agree between the MG (GR in EdS) and the EdS code, and W_operators = c W_displacement
//     scales P_ab by c^(n_a + n_b) (n = 0 for 1, 1 for delta and nabla^2 delta, 2 for delta^2 and s^2)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/ComputePowerSpectra/HybridBiasOneLoopMG.h>
#include <FML/Cosmology/Background/FunctionBackground.h>
#include <FML/Global/Global.h>

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;

using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;


// A smooth LCDM-like linear power spectrum (BBKS, sigma8 ~ 0.8 at z = 0)
double pofk_bbks(double k) {
    const double Gamma = 0.21, q = k / Gamma;
    const double T =
        std::log(1.0 + 2.34 * q) / (2.34 * q) *
        std::pow(1.0 + 3.89 * q + std::pow(16.1 * q, 2) + std::pow(5.46 * q, 3) + std::pow(6.71 * q, 4), -0.25);
    return 2.0e6 * k * T * T;
}

// The LPT kernels at one node (k, p, mu) integrated with fine RK4 steps (same sources as the library)
std::array<double, 3> kernels_at_node(const MGModel & model, double k, double p, double mu) {
    using namespace detail;
    const double c = c_over_H0_mpch;
    const Vec3 kv{0, 0, k * c}, pv{p * c * std::sqrt(1 - mu * mu), 0, p * c * mu}, mv{-pv[0], 0, -pv[2]};
    const double sp = std::sqrt(dot3(add3(kv, pv), add3(kv, pv))), sm = std::sqrt(dot3(add3(kv, mv), add3(kv, mv)));
    // Dk, Dk', Dp, Dp', c2(k,p), ', c2(k,-p), ', c3, ', w(k,p), ', w(-p,k), '
    auto rhs = [&](double x, const double * Y, double * dY) {
        MGTime T;
        T.set(model, std::exp(x));
        const double damp = 2 + T.dlogHdx, iH2 = 1 / T.H2;
        auto lin = [&](int i, double K, double S) {
            dY[i] = Y[i + 1];
            dY[i + 1] = -damp * Y[i + 1] + (T.A(K) * Y[i] + S) * iH2;
        };
        lin(0, kv[2], 0);
        lin(2, p * c, 0);
        lin(4, sp, source_c2(T, kv, pv, Y[0], Y[2]));
        lin(6, sm, source_c2(T, kv, mv, Y[0], Y[2]));
        const Vec3 legs[3] = {kv, pv, mv};
        const double D[3] = {Y[0], Y[2], Y[2]}, c2p[3] = {0, Y[6], Y[4]}, w2[3] = {0, Y[12], Y[10]};
        lin(8, kv[2], source_c3(T, legs, D, c2p, w2));
        dY[10] = Y[11];
        dY[11] = -damp * Y[11] + source_w2(T, kv, pv, Y[0], Y[2]) * iH2;
        dY[12] = Y[13];
        dY[13] = -damp * Y[13] + source_w2(T, mv, kv, Y[2], Y[0]) * iH2;
    };
    const double ai = 1e-3;
    double y[14] = {0};
    {
        MGTime T;
        T.set(model, ai);
        y[0] = y[1] = y[2] = y[3] = ai;
        y[4] = source_c2(T, kv, pv, ai, ai) / (3.5 * T.H2);
        y[5] = 2 * y[4];
        y[6] = source_c2(T, kv, mv, ai, ai) / (3.5 * T.H2);
        y[7] = 2 * y[6];
        const Vec3 legs[3] = {kv, pv, mv};
        const double D[3] = {ai, ai, ai}, c2p[3] = {0, y[6], y[4]}, w2[3] = {0, 0, 0};
        y[8] = source_c3(T, legs, D, c2p, w2) / (9 * T.H2);
        y[9] = 3 * y[8];
    }
    const int n = 4000;
    double x = std::log(ai), h = -x / n, k1[14], k2[14], k3[14], k4[14], yt[14];
    for (int s = 0; s < n; s++) {
        rhs(x, y, k1);
        for (int i = 0; i < 14; i++)
            yt[i] = y[i] + 0.5 * h * k1[i];
        rhs(x + 0.5 * h, yt, k2);
        for (int i = 0; i < 14; i++)
            yt[i] = y[i] + 0.5 * h * k2[i];
        rhs(x + 0.5 * h, yt, k3);
        for (int i = 0; i < 14; i++)
            yt[i] = y[i] + h * k3[i];
        rhs(x + h, yt, k4);
        for (int i = 0; i < 14; i++)
            y[i] += h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        x += h;
    }
    return {y[4], y[6], y[8]};
}

int main() {
    if (FML::ThisTask == 0) {
        // 1. GR in EdS: the general loop functions vs the EdS ones
        const double kmin = 1e-4, kcut = 0.4;
        std::function<double(double)> P = [&](double k) { return (k < kmin or k > kcut) ? 0.0 : pofk_bbks(k); };
        std::function<double(double)> F = [](double k) { return -k * k; };
        double maxdev = 0.0, i3dev = 0.0;
        for (double k : {0.02, 0.1, 0.25}) {
            const auto L = detail::loop_spectra(k, P, kmin, kcut, F);
            const auto M = detail::mg_loop_functions_at_k(mgmodel_gr(1.0), k, {1.0}, {P}, kmin, kcut, {})[0];
            const double Pk = P(k);
            auto dev = [&](double mg, double gr) { maxdev = std::max(maxdev, std::abs(mg / gr - 1.0)); };
            dev(M.G22, 9.0 / 98 * L.Q1);
            dev(M.U20, 3.0 / 7 * L.Q8);
            dev(M.Us2, 1.0 / 7 * L.Qs2);
            dev(-2 * k * Pk * M.g, -6.0 / 7 * (L.R1 + L.R2));
            const double t = Pk * (M.A10R1t - M.A10R2t) - M.A10Qt, kk = Pk * (M.A10R1kk - M.A10R2t) - M.A10Qkk;
            const double a = 0.5 * (t - kk), b = 0.5 * (3 * kk - t);
            dev(-0.5 * (a + b / 3), (4 * L.R2 + 2 * L.Q5) / 14);
            dev(-b / 6, (3 * L.R1 + 4 * L.R2 + 2 * L.Q5) / 14);
            const double ca = 3 * (2 * M.WQa - 4 * Pk * M.WRa), cb = 3 * (2 * M.WQb - 4 * Pk * M.WRb);
            const double al = 0.5 * (cb - ca), be = ca - 3 * al;
            dev(-be / 3, -3.0 / 7 * (2 * L.R1 + 4 * L.R2 + L.Q1 + 2 * L.Q2));
            dev(al + be / 5, 3.0 / 35 * (-4 * L.R1 + 12 * L.R2 - 2 * L.Q1 + 6 * L.Q2));
            i3dev = std::max(i3dev, std::abs(0.5 * Pk * M.I3 / (5.0 / 21 * L.R1) - 1.0));
        }
        check_local(maxdev < 1e-3,
                    "GR (EdS): second order loop functions = EdS ones (max deviation " + num(maxdev, 3) + ")");
        check_local(i3dev < 1e-3, "GR (EdS): third order (A13) = 5/21 R1 (max deviation " + num(i3dev, 3) + ")");

        // 1b. The same with an operator leg factor (Gaussian smoothing of the operators, R = 2 Mpc/h): the terms
        // with one operator leg (A10, U11 for delta and nabla^2 delta) and two (U20, Us2)
        {
            std::function<double(double)> Wl = [](double k) { return std::exp(-0.5 * k * k * 4.0); };
            double dev_w = 0.0;
            for (double k : {0.05, 0.15, 0.3}) {
                const auto L = detail::loop_spectra(k, P, kmin, kcut, F, Wl);
                const auto M = detail::mg_loop_functions_at_k(mgmodel_gr(1.0), k, {1.0}, {P}, kmin, kcut, {}, Wl)[0];
                const double Pk = P(k), wk = Wl(k), Fk = -k * k;
                auto dev = [&](double mg, double gr) { dev_w = std::max(dev_w, std::abs(mg / gr - 1.0)); };
                for (int nabla = 0; nabla < 2; nabla++) {
                    const double Ak = nabla ? Fk * wk : wk;
                    const double Q5A = nabla ? L.Q5F : L.Q5, R1A = nabla ? L.R1F : L.R1W, R2A = nabla ? L.R2F : L.R2W;
                    const double k2a = 3.0 / 7 * Ak * L.R1;
                    const double k2b = -3.0 / 7 * Ak * (L.R1 + 2 * L.R2) - 6.0 / 7 * Q5A - 6.0 / 7 * (R1A + R2A);
                    const double R2t = nabla ? M.A10R2tF : M.A10R2t;
                    const double t = Ak * Pk * M.A10R1t - Pk * R2t - (nabla ? M.A10QtF : M.A10Qt);
                    const double kk = Ak * Pk * M.A10R1kk - Pk * R2t - (nabla ? M.A10QkkF : M.A10Qkk);
                    const double a = 0.5 * (t - kk), b = 0.5 * (3 * kk - t);
                    dev(-0.5 * (a + b / 3), -0.5 * (k2a + k2b / 3));
                    dev(-b / 6, -k2b / 6);
                }
                dev(-2 * k * wk * Pk * M.g, -6.0 / 7 * wk * (L.R1W + L.R2W));
                dev(-k * wk * Pk * (M.gF + Fk * M.g), -3.0 / 7 * wk * ((L.R1F + L.R2F) + Fk * (L.R1W + L.R2W)));
                dev(M.U20, 3.0 / 7 * L.Q8);
                dev(M.Us2, 1.0 / 7 * L.Qs2);
            }
            check_local(dev_w < 1e-3,
                        "GR (EdS), smoothed operators: loop functions with operator legs = EdS ones (max deviation " +
                            num(dev_w, 3) + ")");
        }

        // 1c. W_operators = c W_displacement: P_ab -> c^(n_a + n_b) P_ab
        {
            const double c = 0.7;
            std::function<double(double)> Wd = [&](double k) { return k < kcut ? 1.0 : 0.0; };
            std::function<double(double)> Wo = [&](double k) { return k < kcut ? c : 0.0; };
            ZeldovichOptions opt;
            opt.kmin_integration = kmin;
            opt.kmax_integration = kcut;
            const std::vector<double> kv{0.02, 0.1, 0.2};
            std::vector<std::array<int, 2>> pairs;
            const auto P1 = oneloop_hybrid_bias_spectra(pofk_bbks, Wd, Wd, kv, opt, nullptr, &pairs);
            const auto Pc = oneloop_hybrid_bias_spectra(pofk_bbks, Wo, Wd, kv, opt);
            const int order[5] = {0, 1, 2, 2, 1};
            double dev_c = 0.0;
            for (size_t p = 0; p < pairs.size(); p++)
                for (size_t i = 0; i < kv.size(); i++)
                    dev_c = std::max(
                        dev_c,
                        std::abs(Pc[p][i] / (std::pow(c, order[pairs[p][0]] + order[pairs[p][1]]) * P1[p][i]) - 1.0));
            check_local(dev_c < 1e-8,
                        "W_operators = 0.7 W_displacement: P_ab scales as 0.7^(n_a + n_b) (max deviation " +
                            num(dev_c, 3) + ")");
        }

        // 2. The kernels vs the Eulerian fluid equations (Omega_M = 0.281)
        struct Ref {
            MGModel model;
            double k, p, mu, c3;
            std::string name;
        };
        const double Om = 0.281;
        std::vector<Ref> refs{
            {mgmodel_gr(Om), 0.1, 0.05, 0.3, 1.541633880954e-01, "LCDM"},
            {mgmodel_fofr(Om, 1e-5, 1.0, false), 0.1, 0.05, 0.3, 1.784169867467e-01, "f(R) no screening"},
            {mgmodel_fofr(Om, 1e-5, 1.0, true), 0.1, 0.05, 0.3, 1.613678783263e-01, "f(R) screening"},
            {mgmodel_fofr(Om, 1e-5, 1.0, true), 0.3, 0.2, -0.7, 9.606520027441e-02, "f(R) screening"},
            {mgmodel_fofr(Om, 1e-5, 1.0, false), 0.2, 1.0, 0.9, 1.729352539561e-03, "f(R) no screening"},
            {mgmodel_dgp(Om, 1.0, true), 0.3, 0.2, -0.7, 8.149455358511e-02, "DGP screening"},
            {mgmodel_mu(Om, [](double k, double a) { return 1.0 + 0.3 * k * k / (k * k + 0.01) * a * a * a; }),
             0.1,
             0.001,
             0.5,
             1.306400906674e-01,
             "mu(k,a)"}};
        // The same through mgmodel_from_theory: the shared theories on a flat LCDM background without radiation
        namespace BG = FML::COSMOLOGY::BACKGROUND;
        namespace PT = FML::COSMOLOGY::PERTURBATIONS;
        BG::BackgroundParameters bp;
        bp.h = 0.7;
        bp.Omega_b = 0.05;
        bp.Omega_cdm = Om - 0.05;
        bp.T_CMB = 0.0;
        bp.N_eff = 0.0;
        static BG::FunctionBackground lcdm(
            "lcdm",
            bp,
            [Om](double x) { return std::sqrt(Om * std::exp(-3.0 * x) + 1.0 - Om); },
            [Om](double x) {
                const double m = Om * std::exp(-3.0 * x);
                return -1.5 * m / (m + 1.0 - Om);
            },
            [&bp, Om](BG::Component c, double x) {
                const double m = std::exp(-3.0 * x) / (Om * std::exp(-3.0 * x) + 1.0 - Om);
                if (c == BG::Component::CDM)
                    return bp.Omega_cdm * m;
                if (c == BG::Component::Baryons)
                    return bp.Omega_b * m;
                if (c == BG::Component::DarkEnergy)
                    return (1.0 - Om) / (Om * std::exp(-3.0 * x) + 1.0 - Om);
                return 0.0;
            });
        refs.push_back({mgmodel_from_theory(lcdm, std::make_shared<PT::QuasiStaticGravity_HuSawicky>(lcdm, 1e-5, 1.0)),
                        0.3,
                        0.2,
                        -0.7,
                        9.606520027441e-02,
                        "f(R) screening, from the theory"});
        refs.push_back({mgmodel_from_theory(lcdm, std::make_shared<PT::QuasiStaticGravity_DGP>(lcdm, 1.0, -1)),
                        0.3,
                        0.2,
                        -0.7,
                        8.149455358511e-02,
                        "DGP screening, from the theory"});
        for (auto & r : refs) {
            const auto c = kernels_at_node(r.model, r.k, r.p, r.mu);
            const double dev = std::abs(c[2] / r.c3 - 1.0);
            check_local(dev < 3e-5,
                        "third order LPT kernel = Eulerian, " + r.name + " (k, p, mu) = (" + num(r.k, 3) + ", " +
                            num(r.p, 3) + ", " + num(r.mu, 3) + "): relative deviation " + num(dev, 3));
        }
    }
    return FML::TESTS::summary("hybrid bias");
}
