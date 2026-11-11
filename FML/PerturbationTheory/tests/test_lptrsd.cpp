//==========================================================
// Tests for LPTRedshiftSpace.h (one-loop LPT in redshift space, a port of velocileptors' LPT_RSD) and
// LagrangianQFunctions.h, FFTLog/SphericalBesselTransform.h:
//   - velocileptors with its settings (jn = nmax = 5, N = 256): the P(k, mu) components and the multipole columns
//     with AP, also with an IR split and use_Pzel (reference: python/generate_lptrsd_reference.py)
//   - the redshift-space Zel'dovich P(k, mu) (one_loop = false) against a direct integration over q (independent
//     of the hypergeometric/FFTLog angular series)
//   - f = 0 against the real-space CLEFT of ComputePowerSpectra/HybridBiasOneLoop.h (direct integration)
//   - the Kaiser limit, convergence in jn/nmax and ngauss, the bias combination, isotropic AP, invalid options
// Run from this folder (reads reference/)
//==========================================================
#include <FML/ComputePowerSpectra/HybridBiasOneLoop.h>
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/LPTRedshiftSpace.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::check_rel;
using namespace FML::PERTURBATION;
using DVector = std::vector<double>;

namespace {
    void read_pk(DVector & k, DVector & p) {
        std::ifstream in("reference/lptrsd_linear_pk.txt");
        if (!in)
            throw std::runtime_error("Missing reference/lptrsd_linear_pk.txt (run from PerturbationTheory/tests)");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            double a, b;
            s >> a >> b;
            k.push_back(a);
            p.push_back(b);
        }
    }
    double maxrel(double a, double b, double scale) { return std::abs(a - b) / scale; }
} // namespace

int main() {
    DVector k, p;
    read_pk(k, p);

    //==========================================================
    // velocileptors
    //==========================================================
    {
        std::ifstream in("reference/lptrsd_velocileptors.txt");
        check(bool(in), "velocileptors reference present");
        std::map<std::string, std::vector<std::vector<double>>> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            std::vector<double> r;
            double v;
            while (s >> v)
                r.push_back(v);
            rows[tag].push_back(r);
        }
        for (int variant = 0; variant < 2; variant++) {
            LPTRedshiftSpace::Options o;
            o.jn = 5;
            o.nmax = 5;
            o.N = 256;
            o.kIR = variant == 0 ? 0.0 : 0.2;
            o.use_Pzel = variant == 1;
            LPTRedshiftSpace lpt(k, p, o);
            DVector kv;
            std::vector<std::vector<double>> ref;
            for (const auto & r : rows["C"])
                if (int(r[0] * 10 + 0.5) == int(o.kIR * 10 + 0.5)) {
                    kv.push_back(r[2]);
                    ref.push_back(r);
                }
            check(kv.size() == 20, "velocileptors reference rows found");
            const auto comp = lpt.pk_components(0.7, 0.6, kv);
            double err = 0;
            for (int c = 0; c < LPTRedshiftSpace::ncomponents; c++) {
                double scale = 0, d = 0;
                for (size_t i = 0; i < kv.size(); i++) {
                    scale = std::max(scale, std::abs(ref[i][3 + c]));
                    d = std::max(d, std::abs(comp[i][c] - ref[i][3 + c]));
                }
                err = std::max(err, d / scale);
            }
            std::cout << "velocileptors components (variant " << variant << "): " << err << "\n";
            check(err < 1e-7, "P(k, mu) components = velocileptors (max diff / max of each component)");

            const auto mt = lpt.multipole_components(0.7, kv, 1.02, 0.98, 3);
            double errm = 0;
            for (int ell = 0; ell < 3; ell++) {
                std::vector<std::vector<double>> r;
                for (const auto & row : rows["P" + std::to_string(2 * ell)])
                    if (int(row[0] * 10 + 0.5) == int(o.kIR * 10 + 0.5))
                        r.push_back(row);
                const auto & mine = ell == 0 ? mt.p0 : ell == 1 ? mt.p2 : mt.p4;
                check(r.size() == kv.size(), "velocileptors multipole rows found");
                // the 12 bias and the 4 counterterm columns (the stochastic ones are analytic and partly zero)
                for (int c = 0; c < 16; c++) {
                    double scale = 0, d = 0;
                    for (size_t i = 0; i < kv.size(); i++) {
                        scale = std::max(scale, std::abs(r[i][3 + c]));
                        d = std::max(d, std::abs(mine[i][c] - r[i][3 + c]));
                    }
                    errm = std::max(errm, d / scale);
                }
            }
            std::cout << "velocileptors multipoles (variant " << variant << "): " << errm << "\n";
            check(errm < 1e-7, "multipole columns with AP = velocileptors");
        }
    }

    //==========================================================
    // The Zel'dovich P(k, mu) by direct integration
    //==========================================================
    {
        LPTRedshiftSpace::Options o;
        o.one_loop = false;
        LPTRedshiftSpace lpt(k, p, o);
        const detail::LogInterp Pk(k, p);
        auto P = [&](double kk) { return kk > 10 ? 0.0 : Pk(kk) * std::exp(-kk * kk / 100.); };
        // X(q), Y(q) by quadrature in ln k
        const int nk = 20000;
        DVector lk(nk), w(nk), Pv(nk);
        for (int i = 0; i < nk; i++) {
            lk[i] = std::exp(std::log(1e-5) + i * std::log(1e6) / (nk - 1));
            w[i] = lk[i] * std::log(1e6) / (nk - 1) * (i == 0 or i == nk - 1 ? 0.5 : 1);
            Pv[i] = P(lk[i]);
        }
        const double dq = 0.5, qmax = 1000;
        const int nq = int(qmax / dq);
        DVector X(nq + 1), Y(nq + 1);
        double Xinf = 0;
        for (int i = 0; i < nk; i++)
            Xinf += w[i] * Pv[i] * 2.0 / 3.0 / (2 * M_PI * M_PI);
        for (int iq = 1; iq <= nq; iq++) {
            const double q = iq * dq;
            double sx = 0, sy = 0;
            for (int i = 0; i < nk; i++) {
                const double x = lk[i] * q;
                double j0, j1x;
                if (x < 1e-3) {
                    j0 = 1 - x * x / 6;
                    j1x = 1. / 3 - x * x / 30;
                } else {
                    j0 = std::sin(x) / x;
                    j1x = (std::sin(x) / x - std::cos(x)) / (x * x);
                }
                sx += w[i] * Pv[i] * (2. / 3 - 2 * j1x);
                sy += w[i] * Pv[i] * (-2 * j0 + 6 * j1x);
            }
            X[iq] = sx / (2 * M_PI * M_PI);
            Y[iq] = sy / (2 * M_PI * M_PI);
        }
        double worst = 0;
        const double f = 0.8;
        for (double nu : {0.3, 0.8})
            for (double kk : {0.05, 0.2}) {
                const double Kfac = std::sqrt(1 + f * (2 + f) * nu * nu), s = f * nu * std::sqrt(1 - nu * nu) / Kfac;
                const double c = std::sqrt(1 - s * s), K2 = kk * kk * Kfac * Kfac, Einf = std::exp(-0.5 * K2 * Xinf);
                double sum = 0;
                for (int iq = 1; iq <= nq; iq++) {
                    const double q = iq * dq;
                    DVector mx, mw;
                    LPTRedshiftSpace::gauss_legendre(64 + int(2 * kk * q), mx, mw);
                    double ms = 0;
                    for (size_t im = 0; im < mx.size(); im++) {
                        const double m = mx[im], eps = -0.5 * K2 * (X[iq] - Xinf + Y[iq] * m * m);
                        ms += mw[im] * (Einf * std::exp(eps) - Einf * (1 + eps)) * std::cos(kk * q * c * m) *
                              std::cyl_bessel_j(0.0, kk * q * s * std::sqrt(1 - m * m));
                    }
                    sum += (iq == nq ? 1 : (iq % 2 ? 4 : 2)) * q * q * ms;
                }
                const double pza = 2 * M_PI * dq / 3 * sum + Einf * std::pow(1 + f * nu * nu, 2) * P(kk);
                const double mine = lpt.pk_components(f, nu, {kk})[0][0];
                std::cout << "Zel'dovich P(k=" << kk << ", mu=" << nu << "): direct " << pza << " rel " << mine / pza - 1
                          << "\n";
                worst = std::max(worst, std::abs(mine / pza - 1));
            }
        check(worst < 1.5e-3, "redshift-space Zel'dovich P(k, mu) = direct integration over q (f = 0.8)");
    }

    //==========================================================
    // f = 0: the real-space CLEFT by direct integration (cutoff 1 h/Mpc to keep the direct integral cheap)
    //==========================================================
    {
        namespace HB = FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
        LPTRedshiftSpace::Options o;
        o.cutoff = 1.0;
        LPTRedshiftSpace lpt(k, p, o);
        const detail::LogInterp P(lpt.k_internal(), lpt.p_internal());
        const DVector kv{0.05, 0.1, 0.2};
        const auto c = lpt.pk_components(0.0, 0.5, kv);
        HB::ZeldovichOptions zo;
        zo.kmin_integration = 1e-4;
        zo.kmax_integration = 4.0;
        zo.noperators = 4;
        std::vector<std::array<int, 2>> pairs;
        const auto hb = HB::oneloop_hybrid_bias_spectra([&](double kk) { return P(kk); },
                                                        [](double) { return 1.0; },
                                                        kv,
                                                        zo,
                                                        nullptr,
                                                        &pairs);
        // velocileptors components and the basis spectra: [1] = P_11, [b1] = 2 P_1d, [b1^2] = P_dd, [b2] = P_1d2,
        // [b1 b2] = P_dd2, [b2^2] = P_d2d2 / 4, [bs] = 2 P_1s2, [b1 bs] = 2 P_ds2, [b2 bs] = P_d2s2, [bs^2] = P_s2s2
        struct Map {
            int a, b, comp;
            double fac;
        };
        const std::vector<Map> map{{0, 0, 0, 1},   {0, 1, 1, 0.5}, {1, 1, 2, 1},   {0, 2, 3, 1},   {1, 2, 4, 1},
                                   {2, 2, 5, 4},   {0, 3, 6, 0.5}, {1, 3, 7, 0.5}, {2, 3, 8, 1},   {3, 3, 9, 1}};
        double worst = 0;
        for (const auto & m : map) {
            size_t ip = 0;
            for (size_t i = 0; i < pairs.size(); i++)
                if (pairs[i][0] == m.a and pairs[i][1] == m.b)
                    ip = i;
            double scale = 0, d = 0;
            for (size_t i = 0; i < kv.size(); i++) {
                scale = std::max(scale, std::abs(hb[ip][i]));
                d = std::max(d, std::abs(c[i][m.comp] * m.fac - hb[ip][i]));
            }
            worst = std::max(worst, d / scale);
        }
        std::cout << "f = 0 against the real-space CLEFT (direct): " << worst << "\n";
        check(worst < 3e-3, "f = 0 limit = real-space CLEFT by direct integration (10 bias spectra)");
    }

    //==========================================================
    // Kaiser limit, convergence, combination, AP, options
    //==========================================================
    {
        LPTRedshiftSpace::Options o;
        o.log10_kmin = -5;
        LPTRedshiftSpace lpt(k, p, o);
        const detail::LogInterp P(lpt.k_internal(), lpt.p_internal());
        LPTRedshiftSpace::Bias b;
        b.b1 = 0.7;
        double worst = 0;
        for (double nu : {0.2, 0.9}) {
            const double kk = 0.001, f = 0.8;
            const auto c = lpt.pk_components(f, nu, {kk});
            worst = std::max(worst, std::abs(LPTRedshiftSpace::combine(c[0], kk, nu, b) /
                                                 (std::pow(1 + b.b1 + f * nu * nu, 2) * P(kk)) -
                                             1));
        }
        std::cout << "Kaiser limit at k = 0.001: " << worst << "\n";
        check(worst < 1e-3, "P(k, mu) -> (1 + b1 + f mu^2)^2 P_lin at k = 0.001 h/Mpc (log10_kmin = -5)");
    }
    {
        LPTRedshiftSpace lpt(k, p);
        LPTRedshiftSpace::Options o15;
        o15.jn = 15;
        o15.nmax = 15;
        LPTRedshiftSpace lpt15(k, p, o15);
        LPTRedshiftSpace::Bias b;
        b.b1 = 0.7;
        b.b2 = 0.3;
        b.bs = -0.2;
        b.b3 = 0.1;
        b.alpha0 = 10;
        b.alpha2 = -20;
        b.sn = 1000;
        b.sn2 = -500;
        const DVector kv{0.05, 0.1, 0.2, 0.3};
        const auto m = lpt.multipoles(0.8, kv, b);
        const auto m15 = lpt15.multipoles(0.8, kv, b);
        const auto m8 = lpt.multipoles(0.8, kv, b, 1.0, 1.0, 8);
        double conv = 0, convmu = 0;
        for (size_t i = 0; i < kv.size(); i++) {
            const double s0 = std::abs(m15.p0[i]);
            conv = std::max({conv, maxrel(m.p0[i], m15.p0[i], s0), maxrel(m.p2[i], m15.p2[i], s0),
                             maxrel(m.p4[i], m15.p4[i], s0)});
            convmu = std::max({convmu, maxrel(m.p0[i], m8.p0[i], s0), maxrel(m.p2[i], m8.p2[i], s0),
                               maxrel(m.p4[i], m8.p4[i], s0)});
        }
        std::cout << "convergence in jn = nmax (10 vs 15): " << conv << ", ngauss (4 vs 8): " << convmu << "\n";
        check(conv < 1e-4, "default jn = nmax = 10 converged (relative to P_0) for k <= 0.3 h/Mpc");
        check(convmu < 2e-4, "default ngauss = 4 converged (relative to P_0)");

        // The multipoles from the combined P(k, mu) by Gauss-Legendre
        DVector mus, ws;
        LPTRedshiftSpace::gauss_legendre(16, mus, ws);
        double comb = 0;
        for (size_t i = 0; i < kv.size(); i++) {
            double p0 = 0, p2 = 0;
            for (size_t j = 0; j < mus.size(); j++) {
                const double pk = lpt.pkmu(0.8, mus[j], {kv[i]}, b)[0];
                p0 += 0.5 * ws[j] * pk;
                p2 += 2.5 * ws[j] * 0.5 * (3 * mus[j] * mus[j] - 1) * pk;
            }
            const auto m16 = lpt.multipoles(0.8, {kv[i]}, b, 1.0, 1.0, 8);
            comb = std::max({comb, maxrel(p0, m16.p0[0], std::abs(p0)), maxrel(p2, m16.p2[0], std::abs(p0))});
        }
        check(comb < 1e-10, "multipole tables combined = Legendre integrals of pkmu");

        // Isotropic AP: P_obs(k) = P(k / a) / a^3
        const double a = 1.03;
        const auto mAP = lpt.multipoles(0.8, kv, b, a, a);
        DVector kt;
        for (double kk : kv)
            kt.push_back(kk / a);
        const auto mt = lpt.multipoles(0.8, kt, b);
        double ap = 0;
        for (size_t i = 0; i < kv.size(); i++) {
            // (the stochastic and counterterms are in the true coordinates too)
            ap = std::max({ap, maxrel(mAP.p0[i], mt.p0[i] / (a * a * a), std::abs(mAP.p0[i])),
                           maxrel(mAP.p2[i], mt.p2[i] / (a * a * a), std::abs(mAP.p0[i]))});
        }
        check(ap < 1e-12, "isotropic AP: P_obs(k) = P(k/a) / a^3");
    }
    {
        LPTRedshiftSpace::Options o;
        o.N = 255;
        check(FML::TESTS::throws([&] { LPTRedshiftSpace(k, p, o); }), "odd N rejected (wrong by up to 10%)");
        o.N = 256;
        o.log10_kmax = 2;
        check(FML::TESTS::throws([&] { LPTRedshiftSpace(k, p, o); }), "log10_kmax > 1 rejected");
    }
    return FML::TESTS::summary("LPT redshift space");
}
