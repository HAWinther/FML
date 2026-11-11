//==========================================================
// Tests for EPTRedshiftSpace.h (IR-resummed one-loop EPT in redshift space, a port of velocileptors' REPT):
//   - velocileptors (reference: python/generate_rept_reference.py): the Savitzky-Golay no-wiggle spectrum, Sigma^2,
//     the EPT velocity-moment tables, the 19 columns of P(k, mu) with and without AP, P(k, mu) with bFoG and the
//     multipole columns
//   - the real-space matter P(k) (b1 = 1, f = 0, no wiggles) against the direct one-loop SPT of OneLoop.h
//   - the Kaiser limit, no-wiggle = P (no resummation), the not-a-knot spline, convergence of the internal grid,
//     the bias combination, the default (internal) no-wiggle spectrum, invalid options
// Run from this folder (reads reference/)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/EPTRedshiftSpace.h>
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
using Bias = EPTRedshiftSpace::Bias;

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
    using Rows = std::map<std::string, std::vector<DVector>>;
    Rows read_rows(const std::string & file) {
        std::ifstream in(file);
        Rows rows;
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
        return rows;
    }
    double maxabs(const DVector & v) {
        double m = 0;
        for (double x : v)
            m = std::max(m, std::abs(x));
        return m;
    }
} // namespace

int main() {
    DVector k, p, knw, pnw;
    read_two("reference/lptrsd_linear_pk.txt", k, p);
    read_two("reference/rept_nowiggle_pk.txt", knw, pnw);
    check(knw.size() == k.size(), "no-wiggle reference on the same grid");
    const double f = 0.75;
    const Bias bias{1.8, 0.4, -0.3, 0.2, 10.0, -5.0, 2.0, 1.0, 800.0, -300.0, 100.0};

    EPTRedshiftSpace::Options opt;
    opt.kmin = 1e-2;
    opt.kmax = 0.5;
    opt.nk = 60;
    // velocileptors' internal grid (the defaults are finer, see below)
    EPTRedshiftSpace::Options opt_vel = opt;
    opt_vel.internal.log10_kmin = -4;
    opt_vel.internal.N = 4000;
    EPTRedshiftSpace rept(k, p, pnw, opt_vel);
    const auto & kv = rept.k();

    //==========================================================
    // velocileptors
    //==========================================================
    {
        const auto rows = read_rows("reference/rept_velocileptors.txt");
        check(rows.count("T") and rows.at("T").size() == 60, "velocileptors reference present");

        // The Savitzky-Golay filter on the internal grid
        {
            const auto & io = opt_vel.internal;
            const DVector kint = detail::logspace(io.log10_kmin, io.log10_kmax, io.N);
            const detail::LogInterp Pk(k, p);
            DVector praw(io.N);
            for (int i = 0; i < io.N; i++)
                praw[i] = Pk(kint[i]);
            const int window = int(std::ceil(std::log(7.0) / std::log(kint[io.N - 1] / kint[io.N - 2]))) / 2 * 2 + 1;
            const DVector sg = detail::savgol_filter(praw, window, 4);
            double err = 0;
            for (const auto & r : rows.at("S")) {
                const int i = int(r[0]);
                err = std::max(err, std::abs(sg[i] / r[2] - 1));
            }
            std::cout << "  savgol (window " << window << ") max rel diff " << err << "\n";
            // scipy's own coefficients (an unscaled least-squares fit) are good to ~6e-8 for this window
            check(err < 3e-7, "Savitzky-Golay no-wiggle = scipy savgol_filter (edges and interior)");
        }

        const double s2 = rows.at("SIGMA")[0][0];
        std::cout << "  Sigma^2_BAO " << rept.sigma2_BAO() << " (velocileptors " << s2 << ")\n";
        check(std::abs(rept.sigma2_BAO() / s2 - 1) < 1e-8, "Sigma^2_BAO = velocileptors");

        // The EPT tables (each column relative to its maximum)
        {
            const auto & e = rept.moments();
            std::vector<DVector> mine(31, DVector(60)), ref(31, DVector(60));
            for (int j = 0; j < 60; j++) {
                const auto & r = rows.at("T")[j];
                if (std::abs(r[0] / kv[j] - 1) > 1e-12)
                    throw std::runtime_error("different output k");
                int c = 0;
                for (int i = 0; i < 7; i++, c++)
                    mine[c][j] = e.pk[i][j];
                for (int i = 0; i < 7; i++, c++)
                    mine[c][j] = e.vk[i][j];
                for (int i = 0; i < 5; i++, c++)
                    mine[c][j] = e.s0k[i][j];
                for (int i = 0; i < 5; i++, c++)
                    mine[c][j] = e.s2k[i][j];
                for (int i = 0; i < 2; i++, c++)
                    mine[c][j] = e.g1k[i][j];
                for (int i = 0; i < 2; i++, c++)
                    mine[c][j] = e.g3k[i][j];
                mine[c++][j] = e.k0[j];
                mine[c++][j] = e.k2[j];
                mine[c++][j] = e.k4[j];
                for (int i = 0; i < 31; i++)
                    ref[i][j] = r[1 + i];
            }
            double err = 0;
            for (int i = 0; i < 31; i++) {
                const double scale = maxabs(ref[i]);
                double e = 0;
                for (int j = 0; j < 60; j++)
                    e = std::max(e, std::abs(mine[i][j] - ref[i][j]) / scale);
                if (e > 1e-7)
                    std::cout << "    column " << i << ": " << e << " scale " << scale << " mine " << mine[i][30] << " ref " << ref[i][30] << "\n";
                err = std::max(err, e);
            }
            std::cout << "  EPT tables max diff / max " << err << "\n";
            check(err < 1e-7, "EPT velocity-moment tables (pk, vk, s0, s2, g1, g3, k0, k2, k4) = velocileptors");
        }

        // The columns of P(k, mu) and P(k, mu) with bFoG, with and without AP
        {
            double errm = 0, errf = 0;
            int nm = 0, nf = 0;
            for (const auto & r : rows.at("M")) {
                const double apar = r[0], aperp = r[1], mu = r[2];
                const auto t = rept.table_at_mu(f, mu, apar, aperp);
                int j = 0;
                while (j < 60 and std::abs(kv[j] / r[3] - 1) > 1e-10)
                    j++;
                for (int c = 0; c < 19; c++) {
                    // scale: the largest column value at this k (the columns are summed with O(1) coefficients)
                    double scale = 0;
                    for (int cc = 0; cc < 12; cc++)
                        scale = std::max(scale, std::abs(r[4 + cc]));
                    if (c >= 12)
                        scale = std::max(std::abs(r[4 + c]), 1e-300);
                    errm = std::max(errm, std::abs(t[j][c] - r[4 + c]) / scale);
                }
                nm++;
            }
            for (const auto & r : rows.at("F")) {
                const auto pk = rept.pkmu(f, r[2], bias, r[0], r[1], 0.5);
                int j = 0;
                while (j < 60 and std::abs(kv[j] / r[3] - 1) > 1e-10)
                    j++;
                errf = std::max(errf, std::abs(pk[j] / r[4] - 1));
                nf++;
            }
            std::cout << "  columns at mu: max diff " << errm << " (" << nm << " rows), P(k,mu): " << errf << " (" << nf
                      << " rows)\n";
            check(nm == 240 and nf == 240, "velocileptors P(k, mu) rows found");
            check(errm < 1e-7, "the 19 columns of P(k, mu) (mu = 0.15, 0.7; with and without AP) = velocileptors");
            check(errf < 1e-7, "P(k, mu) with bFoG = velocileptors");
        }

        // The multipole columns
        {
            double err = 0;
            int n = 0;
            for (double apar : {1.0, 1.03}) {
                const double aperp = apar == 1.0 ? 1.0 : 0.97;
                const auto t = rept.multipole_components(f, apar, aperp, 4);
                for (int ell : {0, 2, 4}) {
                    const auto & tab = ell == 0 ? t.p0 : (ell == 2 ? t.p2 : t.p4);
                    for (const auto & r : rows.at("P" + std::to_string(ell))) {
                        if (std::abs(r[0] - apar) > 1e-12)
                            continue;
                        int j = 0;
                        while (j < 60 and std::abs(kv[j] / r[2] - 1) > 1e-10)
                            j++;
                        double scale = 0;
                        for (int cc = 0; cc < 12; cc++)
                            scale = std::max(scale, std::abs(r[3 + cc]));
                        for (int c = 0; c < 19; c++) {
                            // counterterms/stochastic: relative to the column in P_0 (some are zero in P_2, P_4)
                            const double sc = c < 12 ? scale : std::abs(t.p0[j][c]);
                            const double e = std::abs(tab[j][c] - r[3 + c]) / sc;
                            if (e > 1e-6)
                                std::cout << "    P" << ell << " apar " << apar << " k " << kv[j] << " col " << c << ": " << e << " " << tab[j][c] << " " << r[3 + c] << "\n";
                            err = std::max(err, e);
                        }
                        n++;
                    }
                }
            }
            std::cout << "  multipole columns max diff " << err << " (" << n << " rows)\n";
            check(n == 360, "velocileptors multipole rows found");
            check(err < 1e-7, "multipole columns (P0, P2, P4; with and without AP) = velocileptors");
        }
    }

    //==========================================================
    // Real-space matter P(k) (no wiggles: Pnw = P) = P_lin + P_22 + P_13 of the direct one-loop SPT
    //==========================================================
    {
        EPTRedshiftSpace::Options o = opt;
        o.nk = 30;
        o.kmin = 0.02;
        o.kmax = 0.5;
        EPTRedshiftSpace e(k, p, p, o);
        const Bias matter{1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const auto pk = e.pkmu(0.0, 0.5, matter);
        const detail::LogInterp Pk(k, p);
        const double cutoff = o.internal.cutoff;
        auto power = [&](double kk) { return Pk(kk) * std::exp(-std::pow(kk / cutoff, 2)); };
        DirectOptions dopt;
        dopt.order = 24;
        dopt.log_step = 0.05;
        const auto direct = direct_one_loop(e.k(), power, std::pow(10.0, o.internal.log10_kmin),
                                            std::pow(10.0, o.internal.log10_kmax), dopt);
        double err = 0;
        for (size_t j = 0; j < e.k().size(); j++)
            err = std::max(err, std::abs(pk[j] / direct.total[j] - 1));
        std::cout << "  real-space matter vs direct SPT: max rel diff " << err << "\n";
        check(err < 1e-4, "f = 0, b1 = 1: P = P_lin + P_22 + P_13 (direct one-loop SPT, k = 0.02 - 0.5 h/Mpc)");
        // velocileptors' grid: the k-expanded terms lose ~0.2% (k = 0.3) - 0.5% (k = 0.5)
        o.internal.log10_kmin = -4;
        o.internal.N = 4000;
        EPTRedshiftSpace ev(k, p, p, o);
        const auto pkv = ev.pkmu(0.0, 0.5, matter);
        const auto directv = direct_one_loop(e.k(), power, 1e-4, 1e3, dopt);
        double errv = 0;
        for (size_t j = 0; j < e.k().size(); j++)
            errv = std::max(errv, std::abs(pkv[j] / directv.total[j] - 1));
        std::cout << "  (velocileptors' internal grid: " << errv << ")\n";
    }

    //==========================================================
    // Kaiser limit (low k)
    //==========================================================
    {
        EPTRedshiftSpace::Options o = opt;
        o.kmin = 1e-3;
        o.kmax = 3e-3;
        o.nk = 8;
        EPTRedshiftSpace e(k, p, pnw, o);
        Bias b{};
        b.b1 = 1.7;
        double worst = 0;
        for (double mu : {0.1, 0.5, 0.9}) {
            const auto pk = e.pkmu(f, mu, b);
            for (size_t j = 0; j < e.k().size(); j++)
                worst = std::max(worst, std::abs(pk[j] / (std::pow(b.b1 + f * mu * mu, 2) * e.plinear()[j]) - 1));
        }
        std::cout << "  Kaiser limit at k = 0.001 - 0.003: max rel diff " << worst << "\n";
        check(worst < 1e-3, "P(k, mu) -> (b1 + f mu^2)^2 P_lin at low k");
    }

    //==========================================================
    // No wiggles (Pnw = P): the resummation does nothing, the multipoles are the unresummed EPT ones; also
    // sigma2_bao = 0 with wiggles
    //==========================================================
    {
        EPTRedshiftSpace::Options o = opt;
        o.nk = 20;
        EPTRedshiftSpace e1(k, p, p, o);
        o.sigma2_bao = 0.0;
        EPTRedshiftSpace e2(k, p, pnw, o);
        const auto a = e1.pkmu(f, 0.6, bias), b = e2.pkmu(f, 0.6, bias);
        double err = 0;
        for (size_t j = 0; j < a.size(); j++)
            err = std::max(err, std::abs(a[j] / b[j] - 1));
        std::cout << "  no wiggles vs Sigma^2 = 0: max rel diff " << err << "\n";
        check(err < 1e-9, "no resummation two ways (Pnw = P, Sigma^2 = 0) agree");
    }

    //==========================================================
    // The not-a-knot spline is exact for cubics (also when extrapolating)
    //==========================================================
    {
        DVector x, y, xe;
        for (int i = 0; i < 9; i++) {
            x.push_back(0.3 + i * i * 0.1);
            y.push_back(1 - 2 * x.back() + 0.5 * std::pow(x.back(), 3));
        }
        for (double v : {0.1, 0.5, 2.0, 6.0, 8.0})
            xe.push_back(v);
        const auto ye = detail::NotAKnotSpline(x)(y, xe);
        double err = 0;
        for (size_t i = 0; i < xe.size(); i++)
            err = std::max(err, std::abs(ye[i] - (1 - 2 * xe[i] + 0.5 * std::pow(xe[i], 3))));
        check(err < 1e-10, "not-a-knot cubic spline exact for a cubic (non-uniform knots, extrapolation)");
    }

    //==========================================================
    // Convergence of the internal grid (N = 12000 over 9 decades) and of ngauss; combine = pkmu
    //==========================================================
    {
        EPTRedshiftSpace::Options o = opt;
        o.nk = 30;
        o.kmax = 0.5;
        EPTRedshiftSpace a(k, p, pnw, o);
        o.internal.N = 12000;
        o.internal.log10_kmin = -6;
        EPTRedshiftSpace b(k, p, pnw, o);
        const auto ma = a.multipoles(f, bias), mb = b.multipoles(f, bias);
        const auto mc = a.multipoles(f, bias, 1.0, 1.0, 8);
        double conv = 0, convg = 0;
        for (size_t j = 0; j < ma.k.size(); j++) {
            const double s = std::abs(ma.p0[j]);
            conv = std::max({conv, std::abs(ma.p0[j] - mb.p0[j]) / s, std::abs(ma.p2[j] - mb.p2[j]) / s,
                             std::abs(ma.p4[j] - mb.p4[j]) / s});
            convg = std::max({convg, std::abs(ma.p0[j] - mc.p0[j]) / s, std::abs(ma.p2[j] - mc.p2[j]) / s,
                              std::abs(ma.p4[j] - mc.p4[j]) / s});
        }
        std::cout << "  internal grid: max diff / P_0 " << conv << ", ngauss 4 vs 8: " << convg << "\n";
        check(conv < 1e-4, "default internal grid converged (vs N = 12000 from 1e-6, relative to P_0, k <= 0.5)");
        check(convg < 1e-3, "ngauss = 4 converged (relative to P_0)");

        // combine(table) = direct sum at mu, and multipoles from pkmu
        DVector nus, ws;
        LPTRedshiftSpace::gauss_legendre(8, nus, ws);
        DVector p0(a.k().size(), 0.0);
        for (int i = 0; i < 8; i++) {
            const auto pk = a.pkmu(f, nus[i], bias);
            for (size_t j = 0; j < p0.size(); j++)
                p0[j] += 0.5 * ws[i] * pk[j];
        }
        double err = 0;
        for (size_t j = 0; j < p0.size(); j++)
            err = std::max(err, std::abs(p0[j] / ma.p0[j] - 1));
        check(err < 1e-12, "multipoles = Legendre integral of pkmu");
    }

    //==========================================================
    // The default no-wiggle spectrum (internal Savitzky-Golay filter) is close to the reference one
    //==========================================================
    {
        EPTRedshiftSpace::Options o = opt;
        o.nk = 30;
        EPTRedshiftSpace d(k, p, {}, o);
        double err = 0;
        const detail::LogInterp Pnw(knw, pnw);
        for (size_t j = 0; j < d.k().size(); j++) {
            const double ref = Pnw(d.k()[j]);
            err = std::max(err, std::abs(d.plinear_nowiggle()[j] / ref - 1));
        }
        std::cout << "  default no-wiggle vs reference: max rel diff " << err << "; Sigma^2 " << d.sigma2_BAO() << "\n";
        // (the reference is the filter on velocileptors' grid, interpolated through the input k-grid)
        check(err < 1e-3, "default no-wiggle spectrum ~ the Savitzky-Golay filter of the reference");
        check(std::abs(d.sigma2_BAO() / rept.sigma2_BAO() - 1) < 1e-3, "default Sigma^2 ~ reference (different internal grid)");
    }

    //==========================================================
    // Invalid options
    //==========================================================
    {
        EPTRedshiftSpace::Options o;
        o.kmax = 2e3;
        check(FML::TESTS::throws([&] { EPTRedshiftSpace(k, p, {}, o); }), "output k outside the internal grid rejected");
        check(FML::TESTS::throws([&] { EPTRedshiftSpace(k, p, DVector(5, 1.0)); }), "Pnw of the wrong size rejected");
    }

    return FML::TESTS::summary("EPT redshift space (REPT)");
}
