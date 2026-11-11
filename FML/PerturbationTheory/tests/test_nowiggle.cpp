//==========================================================
// Tests for NoWiggle.h (wiggle/no-wiggle splits of the linear P(k) and the Eisenstein & Hu fits):
//   - the discrete sine transform method = velocileptors' pnw_dst, the Savitzky-Golay one = scipy's savgol_filter,
//     the Eisenstein & Hu transfer functions = colossus (reference: python/generate_nowiggle_reference.py)
//   - a smooth input (no wiggles) is returned unchanged (DST, Gaussian with a different reference shape; the
//     Savitzky-Golay bias is bounded)
//   - the wiggles of a CAMB P(k) are removed (residual about a smooth fit in ln k); DST and Gaussian-EH agree
//   - the DST bump detection does not depend on the sampling of P; manual indices; invalid input
// Run from this folder (reads reference/)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/NoWiggle.h>
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
    void read_two(const std::string & file, DVector & k, DVector & p) {
        std::ifstream in(file);
        if (!in)
            throw std::runtime_error("Missing " + file + " (run from PerturbationTheory/tests)");
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

    // The largest |ln y - fit| on [a, b] for the least-squares polynomial of degree d in ln k
    double polyfit_residual(const DVector & k, const DVector & y, double a, double b, int d) {
        DVector x, v;
        for (size_t i = 0; i < k.size(); i++)
            if (k[i] >= a and k[i] <= b) {
                x.push_back(std::log(k[i] / std::sqrt(a * b)) / std::log(std::sqrt(b / a)));
                v.push_back(std::log(y[i]));
            }
        const int n = d + 1;
        std::vector<DVector> M(n, DVector(n + 1, 0.0));
        for (size_t i = 0; i < x.size(); i++) {
            DVector pw(n, 1.0);
            for (int j = 1; j < n; j++)
                pw[j] = pw[j - 1] * x[i];
            for (int r = 0; r < n; r++) {
                for (int c = 0; c < n; c++)
                    M[r][c] += pw[r] * pw[c];
                M[r][n] += pw[r] * v[i];
            }
        }
        for (int r = 0; r < n; r++)
            for (int r2 = r + 1; r2 < n; r2++) {
                const double f = M[r2][r] / M[r][r];
                for (int c = r; c <= n; c++)
                    M[r2][c] -= f * M[r][c];
            }
        DVector c(n);
        for (int r = n - 1; r >= 0; r--) {
            double s = M[r][n];
            for (int j = r + 1; j < n; j++)
                s -= M[r][j] * c[j];
            c[r] = s / M[r][r];
        }
        double worst = 0;
        for (size_t i = 0; i < x.size(); i++) {
            double f = 0, pw = 1;
            for (int j = 0; j < n; j++, pw *= x[i])
                f += c[j] * pw;
            worst = std::max(worst, std::abs(v[i] - f));
        }
        return worst;
    }
} // namespace

int main() {
    DVector k, p, kc, pc;
    read_two("reference/lptrsd_linear_pk.txt", k, p);
    read_two("reference/nowiggle_camb_pk.txt", kc, pc);
    EisensteinHuParameters eh; // the colossus reference values
    EisensteinHuParameters eh_camb;
    eh_camb.Omega_m = (0.02242 + 0.11933) / (0.6766 * 0.6766);
    eh_camb.Omega_b = 0.02242 / (0.6766 * 0.6766);

    //==========================================================
    // References
    //==========================================================
    {
        std::ifstream in("reference/nowiggle_reference.txt");
        check(bool(in), "reference present");
        std::map<std::string, std::vector<DVector>> rows;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            DVector r;
            double v;
            while (s >> v)
                r.push_back(v);
            rows[tag].push_back(r);
        }
        const detail::LogInterp D(k, nowiggle_dst(k, p)), G(k, nowiggle_savgol(k, p));
        double ed = 0, eg = 0, et = 0, en = 0;
        for (const auto & r : rows["D"])
            ed = std::max(ed, std::abs(D(r[0]) / r[1] - 1));
        for (const auto & r : rows["G"])
            eg = std::max(eg, std::abs(G(r[0]) / r[1] - 1));
        for (const auto & r : rows["E"]) {
            et = std::max(et, std::abs(eisenstein_hu_transfer(r[0], eh) / r[1] - 1));
            en = std::max(en, std::abs(eisenstein_hu_transfer_nowiggle(r[0], eh) / r[2] - 1));
        }
        std::cout << "  DST vs velocileptors " << ed << " (" << rows["D"].size() << " k), Savitzky-Golay vs scipy " << eg
                  << ", EH " << et << ", EH no-wiggle " << en << "\n";
        check(rows["D"].size() > 50 and rows["G"].size() > 50 and rows["E"].size() == 61, "reference rows found");
        check(ed < 1e-10, "nowiggle_dst = velocileptors' pnw_dst");
        check(eg < 2e-6, "nowiggle_savgol = scipy savgol_filter pipeline (scipy's coefficients are good to ~1e-6)");
        check(et < 1e-12 and en < 1e-12, "Eisenstein & Hu transfer functions (with and without wiggles) = colossus");
    }

    //==========================================================
    // A smooth input is returned unchanged
    //==========================================================
    {
        const DVector ks = detail::logspace(-5, 2, 2000);
        DVector ps(ks.size());
        for (size_t i = 0; i < ks.size(); i++)
            ps[i] = 2e4 * std::pow(ks[i], eh.n_s) * std::pow(eisenstein_hu_transfer_nowiggle(ks[i], eh), 2);
        check(FML::TESTS::throws([&] { nowiggle_dst(ks, ps); }), "DST: no BAO bump found in a smooth P -> error");
        NoWiggleDSTOptions o;
        o.ii_l = 115;
        o.ii_r = 190;
        const DVector d = nowiggle_dst(ks, ps, o), g = nowiggle_savgol(ks, ps);
        EisensteinHuParameters other = eh;
        other.Omega_m = 0.25;
        other.n_s = 1.0;
        const DVector ga = nowiggle_gaussian(ks, ps, other);
        double ed = 0, eg = 0, ega = 0;
        for (size_t i = 0; i < ks.size(); i++) {
            if (ks[i] < 0.005 or ks[i] > 5)
                continue;
            ed = std::max(ed, std::abs(d[i] / ps[i] - 1));
            eg = std::max(eg, std::abs(g[i] / ps[i] - 1));
            ega = std::max(ega, std::abs(ga[i] / ps[i] - 1));
        }
        std::cout << "  smooth P, k = 0.005 - 5: DST " << ed << ", Savitzky-Golay " << eg
                  << ", Gaussian (different EH shape) " << ega << "\n";
        check(ed < 3e-4, "DST leaves a smooth P unchanged (3e-4)");
        check(ega < 1e-2, "Gaussian smoothing with a different reference shape (Omega_m 0.25, n_s = 1) biases a smooth P < 1%");
        check(eg < 8e-3, "Savitzky-Golay bias on a smooth P bounded (< 0.8%)");
    }

    //==========================================================
    // The wiggles of a CAMB P(k) are removed
    //==========================================================
    {
        const DVector d = nowiggle_dst(kc, pc), g = nowiggle_savgol(kc, pc), ga = nowiggle_gaussian(kc, pc, eh_camb);
        const double a = 0.05, b = 0.4;
        const double rp = polyfit_residual(kc, pc, a, b, 10), rd = polyfit_residual(kc, d, a, b, 10);
        const double rg = polyfit_residual(kc, g, a, b, 10), rga = polyfit_residual(kc, ga, a, b, 10);
        std::cout << "  residual about a degree-10 fit in ln k (k = 0.05 - 0.4): P " << rp << ", DST " << rd
                  << ", Savitzky-Golay " << rg << ", Gaussian-EH " << rga << "\n";
        check(rd < 0.2 * rp and rga < 0.2 * rp, "DST and Gaussian-EH remove > 80% of the wiggles (k = 0.05 - 0.4)");
        check(rg < 0.5 * rp, "Savitzky-Golay removes > 50% of the wiggles (k = 0.05 - 0.4)");
        double dg = 0, dlow = 0;
        for (size_t i = 0; i < kc.size(); i++) {
            if (kc[i] >= 0.15 and kc[i] <= 2)
                dg = std::max(dg, std::abs(d[i] / ga[i] - 1));
            if (kc[i] >= 0.005 and kc[i] < 0.15)
                dlow = std::max(dlow, std::abs(d[i] / ga[i] - 1));
        }
        std::cout << "  DST vs Gaussian-EH: k = 0.15 - 2: " << dg << ", k = 0.005 - 0.15: " << dlow << "\n";
        check(dg < 1e-2, "DST and Gaussian-EH no-wiggle spectra agree to 1% for k = 0.15 - 2 h/Mpc");
    }

    //==========================================================
    // The DST bump does not depend on the sampling of P (velocileptors takes a spurious dip for its edge with
    // the resampled spectrum: ~3% differences)
    //==========================================================
    {
        const DVector kg = detail::logspace(-5, 3, 6000);
        const detail::LogInterp Pk(k, p);
        DVector pg(kg.size());
        for (size_t i = 0; i < kg.size(); i++)
            pg[i] = Pk(kg[i]);
        const detail::LogInterp A(k, nowiggle_dst(k, p)), B(kg, nowiggle_dst(kg, pg));
        double e = 0;
        for (double kk : detail::logspace(-2.3, 0, 200))
            e = std::max(e, std::abs(A(kk) / B(kk) - 1));
        std::cout << "  DST on the input grid vs a resampled one: " << e << "\n";
        check(e < 1e-2, "DST no-wiggle spectrum insensitive to the sampling of P (< 1%)");
    }

    //==========================================================
    // Invalid input
    //==========================================================
    {
        NoWiggleDSTOptions o;
        o.kmax = 1e3;
        check(FML::TESTS::throws([&] { nowiggle_dst(k, p, o); }), "DST range outside the input rejected");
        check(FML::TESTS::throws([&] { nowiggle_dst(k, DVector(3, 1.0)); }), "size mismatch rejected");
        check(FML::TESTS::throws([&] { nowiggle_gaussian(k, p, eh, -1.0); }), "negative width rejected");
    }

    return FML::TESTS::summary("No-wiggle spectra");
}
