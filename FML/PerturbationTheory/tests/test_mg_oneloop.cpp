//==========================================================
// Tests for MGOneLoop.h (one-loop SPT with numerically evolved modified-gravity kernels):
//   - EdS (Omega_m = 1, GR): the kernels are the EdS F2, F3 and P equals the direct one-loop SPT of OneLoop.h
//   - limits: f(R) with |fR0| = 1e-8 and nDGP with Omega_rc = 1e-8 -> GR (LCDM); LCDM vs EdS kernels (< 1% of the loop)
//   - f(R) third-order source: for equal Pi (k -> 0 in all Pi, homogeneous) the average of the three gamma3 terms and
//     the gamma2 iteration reproduce the exact expansion of the field perturbation df(drho)
//   - against ReACT (ACTio-ReACTio a36dda02, the one-loop P(k0 = 0.06) of the real and pseudo spectra printed by
//     tests/reference/react/tools/halomodel_oracle.cpp, reference ../../tests/reference/react/halomodel_{fr,dgp}.txt) for the linear
//     spectra of the ReACT module's fixtures, z = 0, 0.5, 1, 2; for f(R) with ReACT's gamma3 (1/3 of ours)
// Run from this folder (reads ../../tests/reference/react)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/MGOneLoop.h>
#include <FML/PerturbationTheory/OneLoop.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using FML::TESTS::check;
using namespace FML::PERTURBATION;
using DVector = std::vector<double>;

namespace {
    const std::string refdir = "../../tests/reference/react/";
    std::vector<DVector> load(const std::string & file) {
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing " + file + " (run from PerturbationTheory/tests)");
        std::vector<DVector> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            DVector r;
            double v;
            while (s >> v)
                r.push_back(v);
            rows.push_back(r);
        }
        return rows;
    }
    std::string fmt(double v) {
        std::ostringstream s;
        s.precision(2);
        s << v;
        return s.str();
    }
} // namespace

int main() {
    const auto gr = load(refdir + "gr_linear.txt");
    DVector k, P;
    for (const auto & r : gr) {
        k.push_back(r[0]);
        P.push_back(r[1]);
    }
    const LOOPS::Spectrum S(k, P);
    auto Pz0 = [&](double q, size_t) { return S(q); };

    std::cout << "EdS = OneLoop.h\n";
    {
        DirectOptions d;
        d.order = 16;
        d.log_step = 0.1;
        double e = 0;
        for (double kk : {0.06, 0.2}) {
            const auto ref = direct_one_loop({kk}, [&](double q) { return S(q); }, 1e-4, 30.0, d);
            const auto r = mg_one_loop(mg_lcdm(1.0, 1.0), kk, {1.0}, Pz0);
            e = std::max(e, std::abs((r[0] - S(kk)) / ref.loop[0] - 1));
        }
        std::cout << "  loop vs direct EdS SPT: " << e << "\n";
        check(e < 1e-3, "Omega_m = 1, GR: P22 + P13 = OneLoop.h (k = 0.06, 0.2 h/Mpc)");
    }

    std::cout << "Limits\n";
    {
        const double Om = 0.317, kk = 0.06;
        const auto lcdm = mg_one_loop(mg_lcdm(Om, Om), kk, {0.5, 1.0}, Pz0);
        // |fR0| = 1e-10: Compton wavelength ~ 0.03 Mpc/h (beyond the loop's q <= 30 h/Mpc)
        const auto fr = mg_one_loop(mg_hu_sawicki(1e-10, Om, Om), kk, {0.5, 1.0}, Pz0);
        const auto dgp = mg_one_loop(mg_ndgp(1e-8, Om, Om), kk, {0.5, 1.0}, Pz0);
        double ef = 0, ed = 0;
        for (size_t i = 0; i < 2; i++) {
            const double loop = lcdm[i] - S(kk);
            ef = std::max(ef, std::abs((fr[i] - lcdm[i]) / loop));
            ed = std::max(ed, std::abs((dgp[i] - lcdm[i]) / loop));
        }
        check(ef < 1e-3 and ed < 1e-3, "|fR0| = 1e-10 and Omega_rc = 1e-8 give the GR loop (" + fmt(ef) + ", " + fmt(ed) + ")");
        // At k = 0.2 h/Mpc (P22 and P13 do not cancel as at k0) the LCDM kernels change the loop by ~1%
        const auto l2 = mg_one_loop(mg_lcdm(Om, Om), 0.2, {1.0}, Pz0), e2 = mg_one_loop(mg_lcdm(1.0, 1.0), 0.2, {1.0}, Pz0);
        const double dl = std::abs((l2[0] - S(0.2)) / (e2[0] - S(0.2)) - 1);
        check(dl > 1e-4 and dl < 5e-2, "LCDM kernels differ from EdS by " + fmt(dl) + " of the loop at k = 0.2 h/Mpc");
    }

    std::cout << "f(R) third order: the homogeneous limit\n";
    {
        // df solves (dR(df) - 8 pi G drho)/3 = -(k/a)^2 df; for k -> 0 in every Pi (Pi = M1/3) the series of df in
        // s = 8 pi G drho is the inversion of dR = M1 df + M2 df^2/2 + M3 df^3/6: df = s/M1 - M2 s^2/(2 M1^3)
        // + (M2^2/(2 M1^5) - M3/(6 M1^4)) s^3. gamma_n = (k/a)^2 / (2 H^2) x [coefficient of delta^n in df] with
        // s = 3 H0^2 Om a^-3 delta: check gamma2 and the average of the gamma3 terms at small, equal |k_i|
        const double fR0 = 1e-5, Om = 0.3, a = 0.8;
        const auto m = mg_hu_sawicki(fR0, Om, Om);
        const detail::HuSawicki hs{fR0, Om};
        const double M1 = hs.M1(a), M2 = hs.M2(a), M3 = hs.M3(a);
        const double H2 = std::pow(H0_hMpc * m.E(a), 2), s = 3.0 * H0_hMpc * H0_hMpc * Om / (a * a * a);
        // tiny wave vectors (k/a)^2 << M1/3 with |k1 + k2 + k3| = K
        const double eps = 1e-7;
        const SPT::Vec3 k1{eps, 0, 0}, k2{0, eps, 0}, k3{0, 0, eps};
        const double K2 = (k1 + k2 + k3).norm2(), K2p = (k1 + k2).norm2();
        const double g2 = m.gamma2(a, k1, k2);
        const double g3 = (m.gamma3(a, k1, k2, k3) + m.gamma3(a, k2, k3, k1) + m.gamma3(a, k3, k1, k2)) / 3.0;
        // -(k/a)^2 Psi^(n) = (k/a)^2 df^(n) / 2; gamma_n = that / H^2 per delta^n (s = 8 pi G rho per delta)
        const double exact2 = K2p / (a * a) / (2.0 * H2) * (-M2 / (2.0 * std::pow(M1, 3))) * std::pow(s, 2);
        const double exact3 = K2 / (a * a) / (2.0 * H2) * (M2 * M2 / (2.0 * std::pow(M1, 5)) - M3 / (6.0 * std::pow(M1, 4))) *
                              std::pow(s, 3);
        std::cout << "  gamma2 / exact " << g2 / exact2 << ", <gamma3> / exact " << g3 / exact3 << "\n";
        check(std::abs(g2 / exact2 - 1) < 1e-6, "f(R) gamma2 = the homogeneous expansion");
        check(std::abs(g3 / exact3 - 1) < 1e-6, "f(R) gamma3 (averaged over the splits) = the homogeneous expansion");
    }

    std::cout << "Against ReACT's one-loop P(k0)\n";
    for (const std::string name : {"fr", "dgp"}) {
        const auto L = load(refdir + name + "_linear.txt");
        DVector kk;
        std::vector<DVector> Pt(4), Pc(4);
        for (const auto & r : L) {
            kk.push_back(r[0]);
            for (int i = 0; i < 4; i++) {
                Pt[i].push_back(r[1 + 3 * i]);
                Pc[i].push_back(r[2 + 3 * i]);
            }
        }
        std::vector<LOOPS::Spectrum> St, Sc;
        for (int i = 0; i < 4; i++) {
            St.emplace_back(kk, Pt[i]);
            Sc.emplace_back(kk, Pc[i]);
        }
        const DVector a = {1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0};
        const int col[4] = {3, 2, 1, 0};
        const double Om = 0.317;
        // f(R) with ReACT's gamma3 (1/3 of the correct one) to compare with ReACT
        const auto real = name == "fr" ? mg_hu_sawicki(1e-5, Om, Om, true) : mg_ndgp(0.1, Om, Om);
        const auto rr = mg_one_loop(real, 0.06, a, [&](double q, size_t i) { return Sc[col[i]](q); });
        const auto rp = mg_one_loop(mg_lcdm(Om, Om), 0.06, a, [&](double q, size_t i) { return St[col[i]](q); });
        // ReACT's values (Z lines: a, ploopr, ploopp)
        std::map<int, std::pair<double, double>> ref;
        std::ifstream in(refdir + "halomodel_" + name + ".txt");
        std::string line;
        while (std::getline(in, line))
            if (line.rfind("Z ", 0) == 0) {
                std::istringstream s(line);
                std::string t;
                double z, aa, r1, p1;
                s >> t >> z >> t >> aa >> t >> r1 >> t >> p1;
                ref[int(std::lround(aa * 1000))] = {r1, p1};
            }
        double eP = 0, eD = 0;
        for (size_t i = 0; i < 4; i++) {
            const auto & R = ref.at(int(std::lround(a[i] * 1000)));
            eP = std::max({eP, std::abs(rr[i] / R.first - 1), std::abs(rp[i] / R.second - 1)});
            eD = std::max(eD, std::abs((rr[i] - rp[i]) / (R.first - R.second) - 1));
        }
        std::cout << "  " << name << ": P(k0) " << eP << ", real - pseudo " << eD << "\n";
        // ReACT integrates P22 with 32 Gauss-Legendre points in the cosine (missing ~4% of the loop at k0, P22 near
        // |k - q| -> 0) and over q with a 50-point trapezoid: the totals agree to that; the real - pseudo
        // difference (which sets k*) to 0.7% (nDGP) and 3% (f(R), screening at large q)
        check(eP < 6e-3, name + ": real and pseudo one-loop P(k0) = ReACT to " + fmt(eP));
        check(eD < 0.05, name + ": real - pseudo one-loop P(k0) = ReACT to " + fmt(eD));
    }

    check(FML::TESTS::throws([&] { mg_one_loop(mg_lcdm(0.3, 0.3), 0.06, {1.0, 0.5}, Pz0); }),
          "decreasing a_out rejected");
    check(FML::TESTS::throws([&] { mg_hu_sawicki(0.0, 0.3, 0.3); }), "fR0 = 0 rejected");
    return FML::TESTS::summary("MG one-loop");
}
