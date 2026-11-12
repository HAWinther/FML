//==========================================================
// Tests for TNS.h (one-loop SPT TNS model with the McDonald-Roy nonlinear bias):
//   - FAST-PT (reference: python/generate_fastpt_reference.py): the one-loop P_dd, the nonlinear bias terms
//     (Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl) and the TNS A + B terms by powers of mu
//   - P_dd and P_tt against the direct one-loop SPT of OneLoop.h (EdS F and G kernels)
//   - P13 of dd, dt, tt against their analytic kernels (1D integrals)
//   - the Kaiser limit, the multipoles, the FoG, invalid input
// Run from this folder (reads reference/)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/OneLoop.h>
#include <FML/PerturbationTheory/TNS.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using FML::TESTS::check;
using namespace FML::PERTURBATION;
using DVector = std::vector<double>;

namespace {
    std::vector<DVector> read_table(const std::string & file, std::string * header = nullptr) {
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing " + file + " (run from PerturbationTheory/tests)");
        std::vector<DVector> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty())
                continue;
            if (line[0] == '#') {
                if (header and header->empty())
                    *header = line;
                continue;
            }
            std::istringstream s(line);
            DVector r;
            double v;
            while (s >> v)
                r.push_back(v);
            rows.push_back(r);
        }
        return rows;
    }
    double header_value(const std::string & header, const std::string & key) {
        const auto pos = header.find(key + " = ");
        if (pos == std::string::npos)
            throw std::runtime_error("No " + key + " in the header");
        return std::stod(header.substr(pos + key.size() + 3));
    }
    // The analytic P13 kernels (e.g. Carlson, White & Padmanabhan 2009): P13 = k^3 P(k) / (4 pi^2 N) int dr P(kr) K(r)
    double L(double r) { return std::log(std::abs((1 + r) / (1 - r))); }
    double Kdd(double r) {
        return 12 / (r * r) - 158 + 100 * r * r - 42 * std::pow(r, 4) + 3 / std::pow(r, 3) * std::pow(r * r - 1, 3) * (7 * r * r + 2) * L(r);
    }
    double Kdt(double r) {
        return 24 / (r * r) - 202 + 56 * r * r - 30 * std::pow(r, 4) + 3 / std::pow(r, 3) * std::pow(r * r - 1, 3) * (5 * r * r + 4) * L(r);
    }
    double Ktt(double r) {
        return 12 / (r * r) - 82 + 4 * r * r - 6 * std::pow(r, 4) + 3 / std::pow(r, 3) * std::pow(r * r - 1, 3) * (r * r + 2) * L(r);
    }
} // namespace

int main() {
    std::string header;
    const auto lin = read_table("reference/classpt_linear_pk.txt", &header);
    const double f = header_value(header, "f");
    DVector k, P;
    for (const auto & r : lin) {
        k.push_back(r[0]);
        P.push_back(r[1]);
    }
    const auto ref = read_table("reference/tns_fastpt.txt");
    DVector kout;
    for (const auto & r : ref)
        kout.push_back(r[0]);
    const TNS tns(k, P, kout);
    const auto & T = tns.tables();
    const LOOPS::Spectrum S(k, P);

    //==========================================================
    // FAST-PT
    //==========================================================
    {
        double e_loop = 0, e_bias = 0, e_s3 = 0, e_ab = 0, e_mu0 = 0;
        for (size_t i = 0; i < kout.size(); i++) {
            const auto & R = ref[i];
            const double s4 = R[7];
            // relative to |P22| + |P13| (they nearly cancel around k = 0.1)
            e_loop = std::max(e_loop, std::abs(T.P22_dd[i] + T.P13_dd[i] - R[1]) / (std::abs(T.P22_dd[i]) + std::abs(T.P13_dd[i])));
            const double mine[5] = {2 * T.Pb2d[i],
                                    4 * T.Pb22[i] + 2 * s4,
                                    2 * T.Pbs2d[i],
                                    2 * (2 * T.Pb2s2[i] + 2 * s4 / 3),
                                    2 * (2 * T.Pbs22[i] + 4 * s4 / 9)};
            for (int c = 0; c < 5; c++)
                e_bias = std::max(e_bias, std::abs(mine[c] / R[2 + c] - 1));
            e_s3 = std::max(e_s3, std::abs(64.0 / 315.0 * T.sigma3sq[i] * T.Plin[i] / R[8] - 1));
            // A + B (b1 = 1) per mu^(2m), m = 1..4, relative to the largest of them
            double ab[4] = {0, 0, 0, 0}, scale = 0;
            for (int m = 1; m < 5; m++) {
                for (int n = 1; n <= 4; n++)
                    ab[m - 1] += std::pow(f, n) * ((n <= 3 ? T.A[n][m][i] : 0) + (n >= 2 ? T.B[n][m][i] : 0));
                scale = std::max(scale, std::abs(R[8 + m]));
            }
            for (int m = 0; m < 4; m++)
                e_ab = std::max(e_ab, std::abs(ab[m] - R[9 + m]) / scale);
            // A_n, B_n have no mu^0 term
            e_mu0 = std::max({e_mu0, std::abs(T.A[1][0][i] / T.A[1][1][i]), std::abs(T.B[2][0][i] / T.A[1][1][i])});
        }
        std::cout << "  vs FAST-PT: P22 + P13 " << e_loop << ", bias terms " << e_bias << ", sig3nl " << e_s3
                  << ", A + B " << e_ab << "\n";
        check(e_mu0 < 1e-10, "A and B have no mu^0 term");
        check(e_loop < 2e-4, "one-loop P_dd = FAST-PT (k = 0.01 - 0.3 h/Mpc; relative to |P22| + |P13|)");
        check(e_bias < 5e-4, "Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2 = FAST-PT");
        check(e_s3 < 5e-4, "sigma3^2 P_lin = FAST-PT's sig3nl (x 315/64)");
        check(e_ab < 3e-3, "A + B by powers of mu = FAST-PT's RSD_ABsum_components");
    }

    //==========================================================
    // OneLoop.h: P_dd and P_tt (custom G kernels)
    //==========================================================
    {
        auto power = [&](double q) { return S(q); };
        DirectOptions dopt;
        dopt.order = 16;
        dopt.log_step = 0.1;
        const auto dd = direct_one_loop(kout, power, k.front(), k.back(), dopt);
        Kernels G;
        G.name = "G (velocity)";
        G.f2 = [](double q, double p, double mu) { return 3.0 / 7.0 + 0.5 * mu * (q / p + p / q) + 4.0 / 7.0 * mu * mu; };
        G.f3_collapsed = [](double kk, double q, double mu) {
            const SPT::Vec3 a{0, 0, kk}, b{q * std::sqrt(1 - mu * mu), 0, q * mu};
            double F, Gv;
            const SPT::Vec3 v[3] = {a, b, -b};
            SPT::FG(v, 3, F, Gv);
            return Gv;
        };
        const auto tt = direct_one_loop(kout, power, k.front(), k.back(), dopt, G);
        double edd = 0, ett = 0;
        for (size_t i = 0; i < kout.size(); i++) {
            edd = std::max(edd, std::abs(T.P22_dd[i] / dd.p22[i] - 1) + std::abs(T.P13_dd[i] / dd.p13[i] - 1));
            ett = std::max(ett, std::abs(T.P22_tt[i] / tt.p22[i] - 1) + std::abs(T.P13_tt[i] / tt.p13[i] - 1));
        }
        std::cout << "  vs OneLoop.h direct: P_dd loop " << edd << ", P_tt (P22, P13) " << ett << "\n";
        check(edd < 1e-4, "P22_dd and P13_dd = OneLoop.h direct_one_loop");
        check(ett < 1e-4, "P22_tt and P13_tt = OneLoop.h with the G2, G3 kernels");
    }

    //==========================================================
    // P13 against the analytic 1D kernels
    //==========================================================
    {
        double e = 0;
        for (size_t i = 0; i < kout.size(); i++) {
            const double kk = kout[i];
            const auto gl = LOOPS::detail::gauss_legendre(24);
            DVector r, w;
            LOOPS::QuadratureOptions o;
            o.log_step = 0.05;
            LOOPS::detail::log_nodes({k.front() / kk, 1.0, k.back() / kk}, o, gl, r, w);
            double a = 0, b = 0, c = 0;
            for (size_t j = 0; j < r.size(); j++) {
                const double p = S(kk * r[j]);
                a += w[j] * p * Kdd(r[j]);
                b += w[j] * p * Kdt(r[j]);
                c += w[j] * p * Ktt(r[j]);
            }
            const double pre = kk * kk * kk * S(kk) / (4 * M_PI * M_PI);
            e = std::max({e, std::abs(T.P13_dd[i] / (pre * a / 252) - 1), std::abs(T.P13_dt[i] / (pre * b / 252) - 1),
                          std::abs(T.P13_tt[i] / (pre * c / 84) - 1)});
        }
        std::cout << "  P13 (dd, dt, tt) vs the analytic kernels: " << e << "\n";
        check(e < 2e-4, "P13_dd, P13_dt, P13_tt = the analytic 1D kernels");
    }

    //==========================================================
    // Kaiser limit, multipoles, FoG
    //==========================================================
    {
        const TNS low(k, P, {1e-3});
        TNS::Parameters p;
        p.f = f;
        p.b1 = 1.7;
        p.sigma_v = 4.0;
        double e = 0;
        for (double mu : {0.0, 0.4, 1.0}) {
            const double kaiser = std::pow(p.b1 + f * mu * mu, 2) * low.tables().Plin[0];
            e = std::max(e, std::abs(low.pkmu(mu, p)[0] / kaiser - 1));
        }
        std::cout << "  Kaiser limit (k = 1e-3): " << e << "\n";
        check(e < 1e-3, "P_s -> (b1 + f mu^2)^2 P_lin at low k");

        // Without FoG the multipoles of the mu^(2m) polynomial are exact
        p = TNS::Parameters{};
        p.f = f;
        p.b1 = 2.0;
        p.b2 = 0.5;
        p.bs2 = -4.0 / 7.0;
        p.b3nl = 32.0 / 315.0;
        p.fog = TNS::FoG::None;
        p.N = 100.0;
        const auto mp = tns.multipoles(p);
        const auto real = tns.real_space_spectra(p);
        double em = 0;
        for (size_t i = 0; i < kout.size(); i++) {
            // Legendre moments of mu^(2m): [mu^0]_0 = 1, [mu^2]_0 = 1/3, [mu^4]_0 = 1/5, [mu^6]_0 = 1/7, [mu^8]_0 = 1/9
            double ab0 = 0;
            for (int m = 0; m < 5; m++)
                for (int n = 1; n <= 4; n++)
                    ab0 += std::pow(f, n) / (2 * m + 1) *
                           ((n <= 3 ? std::pow(p.b1, 3 - n) * T.A[n][m][i] : 0) + (n >= 2 ? std::pow(p.b1, 4 - n) * T.B[n][m][i] : 0));
            const double P0 = real[0][i] + 2 * f / 3 * real[1][i] + f * f / 5 * real[2][i] + ab0 + p.N;
            const double P4 = 8.0 / 35.0 * f * f * real[2][i];
            const double ab4 = [&] {
                double s = 0; // [mu^4]_4 = 8/35, [mu^6]_4 = 24/77, [mu^8]_4 = 48/143
                const double c4[5] = {0, 0, 8.0 / 35.0, 24.0 / 77.0, 48.0 / 143.0};
                for (int m = 0; m < 5; m++)
                    for (int n = 1; n <= 4; n++)
                        s += c4[m] * std::pow(f, n) *
                             ((n <= 3 ? std::pow(p.b1, 3 - n) * T.A[n][m][i] : 0) + (n >= 2 ? std::pow(p.b1, 4 - n) * T.B[n][m][i] : 0));
                return s;
            }();
            em = std::max({em, std::abs(mp[0][i] / P0 - 1), std::abs(mp[2][i] / (P4 + ab4) - 1)});
        }
        std::cout << "  multipoles (no FoG) vs the exact Legendre moments: " << em << "\n";
        check(em < 1e-10, "P_0, P_4 without FoG = the Legendre moments of the mu^(2m) terms");

        p.sigma_v = 5.0;
        p.fog = TNS::FoG::Lorentzian;
        const auto l0 = tns.pkmu(0.0, p);
        p.fog = TNS::FoG::None;
        const auto n0 = tns.pkmu(0.0, p), n1 = tns.pkmu(1.0, p);
        p.fog = TNS::FoG::Gaussian;
        const auto g1 = tns.pkmu(1.0, p);
        p.fog = TNS::FoG::Lorentzian;
        const auto l1 = tns.pkmu(1.0, p);
        bool ok0 = true, ok1 = true;
        for (size_t i = 0; i < kout.size(); i++) {
            const double x = kout[i] * f * p.sigma_v;
            ok0 = ok0 and std::abs(l0[i] - n0[i]) < 1e-10 * std::abs(n0[i]);
            ok1 = ok1 and std::abs((g1[i] - p.N) - std::exp(-x * x) * (n1[i] - p.N)) < 1e-10 * std::abs(n1[i]) and
                  std::abs((l1[i] - p.N) - (n1[i] - p.N) / (1 + x * x)) < 1e-10 * std::abs(n1[i]);
        }
        check(ok0, "no FoG damping at mu = 0");
        check(ok1, "Gaussian and Lorentzian FoG factors at mu = 1 (shot noise undamped)");
    }

    //==========================================================
    // Invalid input
    //==========================================================
    check(FML::TESTS::throws([&] { TNS(k, P, {0.1, -0.1}); }), "non-positive output k rejected");

    return FML::TESTS::summary("TNS");
}
