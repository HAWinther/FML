//==========================================================
// Tests for EFTofLSS.h (CLASS-PT style one-loop EFTofLSS multipoles of biased tracers in redshift space):
//   - CLASS-PT (reference: python/generate_classpt_reference.py), with and without IR resummation: the multipoles
//     l = 0, 2, 4 of each of the 12 bias monomials, the counterterm templates and the b4 term
//   - the tree-level (Kaiser) limit, the multipoles for given bias = the table combination plus the stochastic terms,
//     a given no-wiggle spectrum, convergence of the quadrature, invalid options
// Run from this folder (reads reference/)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/EFTofLSS.h>
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
} // namespace

int main() {
    std::string header;
    const auto lin = read_table("reference/classpt_linear_pk.txt", &header);
    const double f = header_value(header, "f"), h = header_value(header, "h"), rd = header_value(header, "r_d");
    DVector k, P;
    for (const auto & r : lin) {
        k.push_back(r[0]);
        P.push_back(r[1]);
    }

    //==========================================================
    // CLASS-PT
    //==========================================================
    for (bool ir : {false, true}) {
        const auto ref = read_table(std::string("reference/classpt_eft_ir") + (ir ? "Yes" : "No") + ".txt");
        DVector kout;
        for (const auto & r : ref)
            kout.push_back(r[0]);
        EFTofLSS::Options o;
        o.ir_resummation = ir;
        o.h = h;
        o.r_bao = h * rd;
        const EFTofLSS eft(k, P, kout, f, o);
        const auto & T = eft.multipole_table();
        const std::string tag = ir ? " (IR resummed)" : " (no IR)";
        std::cout << "  CLASS-PT" << tag << ": max |diff| / max |CLASS-PT| per column (l = 0, 2, 4)\n";
        // The error of each column relative to its largest value over k, with a floor of 1e-6 of the b1^2 monopole
        // (CLASS-PT's terms that vanish identically, e.g. b2^2 in l = 2, 4, are ~1e-13 of round-off)
        double floor = 0;
        for (const auto & r : ref)
            floor = std::max(floor, 1e-6 * std::abs(r[1 + EFTofLSS::B1B1]));
        double worst = 0, worst_ct = 0, worst_b4 = 0;
        for (int l = 0; l < 3; l++) {
            for (int c = 0; c < EFTofLSS::NMONOMIAL + 2; c++) {
                const int col = 1 + 14 * l + c;
                double scale = 0, err = 0;
                // b2^2 monopole: CLASS-PT's differs by a constant (its k -> 0 subtraction depends on its k grid)
                const bool shift = (l == 0 and c == EFTofLSS::B2B2);
                double off = 0;
                if (shift)
                    off = ref[0][col] - T[l][c][0];
                for (size_t i = 0; i < kout.size(); i++) {
                    double mine;
                    if (c < EFTofLSS::NMONOMIAL)
                        mine = T[l][c][i] + off;
                    else if (c == EFTofLSS::NMONOMIAL)
                        mine = T[l][EFTofLSS::CT0 + l][i];
                    else {
                        // b4 response (b1 = 0), CLASS-PT's b4 = h^2 x ours
                        const double C[3] = {f * f / 9, 4 * 70 * f * f / 693., 8 * 210 * f * f / 5005.};
                        mine = -35. / 8 * f * f * kout[i] * kout[i] * C[l] * T[2][EFTofLSS::CT4][i] * h * h;
                    }
                    scale = std::max(scale, std::abs(ref[i][col]));
                    err = std::max(err, std::abs(mine - ref[i][col]));
                }
                const double e = err / std::max(scale, floor);
                if (c < EFTofLSS::NMONOMIAL) {
                    if (scale > floor)
                        std::cout << "    l = " << 2 * l << " " << EFTofLSS::monomial_name(c) << ": " << e << "\n";
                    worst = std::max(worst, e);
                } else if (c == EFTofLSS::NMONOMIAL)
                    worst_ct = std::max(worst_ct, e);
                else
                    worst_b4 = std::max(worst_b4, e);
            }
        }
        std::cout << "    T_l: " << worst_ct << ", b4: " << worst_b4 << "\n";
        check(worst < 3e-3, "all 12 bias monomials, l = 0, 2, 4 = CLASS-PT" + tag);
        check(worst_ct < 5e-4, "the counterterm templates = CLASS-PT" + tag);
        check(worst_b4 < 5e-4, "the b4 term = CLASS-PT (b4_CLASS-PT = h^2 b4)" + tag);
        if (ir) {
            // CLASS-PT's Sigma^2 (h = 0.6736, Planck-like at z = 0.5) is not printed; check against the DST split
            check(eft.Sigma2() > 0 and eft.deltaSigma2() > 0, "Sigma^2, delta Sigma^2 positive");

            // A given no-wiggle spectrum (the same split) reproduces the ClassPT method
            EFTofLSS::Options og = o;
            og.nowiggle = EFTofLSS::Options::NoWiggleMethod::Given;
            og.Pnw = nowiggle_classpt(k, P, h, k);
            const EFTofLSS given(k, P, kout, f, og);
            double eg = 0;
            for (int l = 0; l < 3; l++)
                for (int c = 0; c < EFTofLSS::NCOLUMN; c++)
                    for (size_t i = 0; i < kout.size(); i++)
                        eg = std::max(eg, std::abs(given.multipole_table()[l][c][i] - T[l][c][i]) /
                                              (std::abs(T[0][EFTofLSS::B1B1][i])));
            check(eg < 1e-12, "NoWiggleMethod::Given with the ClassPT split = ClassPT");
        }
    }

    //==========================================================
    // Tree level, the bias combination, convergence
    //==========================================================
    {
        EFTofLSS::Options o;
        o.ir_resummation = false;
        const EFTofLSS low(k, P, {1e-3}, f, o);
        EFTofLSS::Bias b;
        b.b1 = 1.9;
        const auto m = low.multipoles(b);
        const LOOPS::Spectrum S(k, P);
        const double Pl = S(1e-3), b1 = b.b1;
        const double K[3] = {b1 * b1 + 2 * f * b1 / 3 + f * f / 5, 4 * f * b1 / 3 + 4 * f * f / 7, 8 * f * f / 35};
        double e = 0;
        for (int l = 0; l < 3; l++)
            e = std::max(e, std::abs(m[l][0] / (K[l] * Pl) - 1));
        std::cout << "  Kaiser limit (k = 1e-3): " << e << "\n";
        check(e < 2e-3, "P_l -> Kaiser multipoles at low k");

        const DVector kout{0.02, 0.1, 0.2};
        o.ir_resummation = true;
        o.h = h;
        o.r_bao = h * rd;
        const EFTofLSS eft(k, P, kout, f, o);
        EFTofLSS::Bias bb{2.0, -0.5, 0.3, 0.2, 5.0, 10.0, -3.0, 50.0, 400.0, 200.0, -100.0};
        const auto mp = eft.multipoles(bb);
        const auto & T = eft.multipole_table();
        double ec = 0;
        for (size_t i = 0; i < kout.size(); i++) {
            const double coef[12] = {1, bb.b1, bb.b1 * bb.b1, bb.b2, bb.b1 * bb.b2, bb.b2 * bb.b2, bb.bG2, bb.b1 * bb.bG2,
                                     bb.b2 * bb.bG2, bb.bG2 * bb.bG2, bb.bGamma3, bb.b1 * bb.bGamma3};
            const double x = kout[i] * kout[i] / (0.45 * 0.45), k2 = kout[i] * kout[i];
            const double C[3] = {f * f / 9 + 2 * f * bb.b1 / 7 + bb.b1 * bb.b1 / 5,
                                 4 * (70 * f * f + 165 * f * bb.b1 + 99 * bb.b1 * bb.b1) / 693,
                                 8 * (210 * f * f + 390 * f * bb.b1 + 143 * bb.b1 * bb.b1) / 5005};
            const double cl[3] = {bb.c0, bb.c2, bb.c4};
            for (int l = 0; l < 3; l++) {
                double v = 0;
                for (int c = 0; c < 12; c++)
                    v += coef[c] * T[l][c][i];
                v += -2 * cl[l] * T[l][EFTofLSS::CT0 + l][i] - 35. / 8 * bb.b4 * f * f * k2 * C[l] * T[2][EFTofLSS::CT4][i];
                if (l == 0)
                    v += bb.Pshot + (bb.a0 + bb.a2 / 3) * x;
                if (l == 1)
                    v += 2. / 3 * bb.a2 * x;
                ec = std::max(ec, std::abs(mp[l][i] - v) / std::abs(mp[0][i]));
            }
        }
        check(ec < 1e-12, "multipoles(bias) = the monomial tables + counterterms + stochastic terms");

        EFTofLSS::Options fine = o;
        fine.quadrature.order = 12;
        fine.quadrature.log_step = 0.15;
        fine.quadrature.nphi = 10;
        fine.nmu = 60;
        const EFTofLSS ref(k, P, kout, f, fine);
        const auto mr = ref.multipoles(bb);
        double ef = 0;
        for (size_t i = 0; i < kout.size(); i++)
            for (int l = 0; l < 3; l++)
                ef = std::max(ef, std::abs(mp[l][i] - mr[l][i]) / std::abs(mr[0][i]));
        std::cout << "  default vs refined quadrature: " << ef << " of P_0\n";
        check(ef < 2e-4, "the default quadrature is converged (vs a refined one, k <= 0.2 h/Mpc)");
    }

    //==========================================================
    // Invalid options
    //==========================================================
    {
        EFTofLSS::Options o;
        o.h = h;
        check(FML::TESTS::throws([&] { EFTofLSS(k, P, {0.1}, f, o); }), "IR resummation without r_bao rejected");
        o.r_bao = h * rd;
        o.h = 0;
        check(FML::TESTS::throws([&] { EFTofLSS(k, P, {0.1}, f, o); }), "ClassPT split without h rejected");
        o.h = h;
        o.nowiggle = EFTofLSS::Options::NoWiggleMethod::Given;
        o.Pnw = DVector(5, 1.0);
        check(FML::TESTS::throws([&] { EFTofLSS(k, P, {0.1}, f, o); }), "given P_nw of the wrong size rejected");
        check(FML::TESTS::throws([&] { EFTofLSS(k, P, {0.0}, f, EFTofLSS::Options{}); }), "non-positive k rejected");
    }

    return FML::TESTS::summary("EFTofLSS");
}
