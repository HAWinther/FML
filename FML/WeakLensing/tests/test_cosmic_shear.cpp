// Tests of CosmicShear.h: n(z) shifts, Limber shear + NLA spectra and xi_+- against CCL, Hankel transforms against
// analytic results, log-COSEBIs against Schneider, Eifler & Krause (2010) in 60 digits, and the E_n/B_n identities.
#include <FML/Global/Global.h>
#include <FML/WeakLensing/CosmicShear.h>
#include <FML/tests/TestHelpers.h>

#include "ReferenceCosmicShear.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

using FML::TESTS::check_local;
using FML::TESTS::num;
using namespace FML::WEAKLENSING;

static bool rel(double x, double ref, double tol, const std::string & label) {
    const double e = std::abs(x / ref - 1);
    return check_local(std::isfinite(x) && e < tol, label + " (rel. err " + num(e, 2) + ")");
}

static void test_shift() {
    RedshiftDistribution d{{0.0, 0.5, 1.0}, {0.0, 2.0, 0.0}};
    auto s = shifted_distribution(d, 0.1, 0.01);
    check_local(std::abs(s.z.front() - 0.1) < 1e-14 && std::abs(s.z.back() - 1.1) < 1e-14 && s.z.size() == 101,
                "shift +0.1 moves and refines the nodes");
    auto c = shifted_distribution(d, -0.25, 1.0);
    // cut at z = 0: n(0) = n_old(0.25) = 1
    check_local(c.z.front() == 0.0 && std::abs(c.n.front() - 1.0) < 1e-14 && std::abs(c.z.back() - 0.75) < 1e-14,
                "shift -0.25 is cut at z = 0");
    check_local(FML::TESTS::throws([&] { shifted_distribution(d, -2.0, 0.1); }), "shift below z = 0 rejected");
}

static void test_hankel() {
    // C = exp(-l^2 s^2/2): xi_+ = exp(-theta^2/(2 s^2))/(2 pi s^2)
    const double s = 1e-3;
    Function C0 = [&](double l) { return std::exp(-0.5 * l * l * s * s); };
    for (double th : {2e-4, 1e-3, 3e-3}) {
        const double ref = std::exp(-th * th / (2 * s * s)) / (2 * M_PI * s * s);
        rel(hankel(C0, 1e-3, 2e4, th, 0), ref, 1e-9, "Hankel J0 of a Gaussian at theta = " + num(th));
    }
    // C = l^4 exp(-a l^2): int dl l^5 e^{-a l^2} J4(b l) = b^4/(2a)^5 exp(-b^2/(4a))
    const double a = 1e-6;
    Function C4 = [&](double l) { return std::pow(l, 4) * std::exp(-a * l * l); };
    for (double th : {5e-4, 2e-3, 4e-3}) {
        const double ref = std::pow(th, 4) / std::pow(2 * a, 5) * std::exp(-th * th / (4 * a)) / (2 * M_PI);
        rel(hankel(C4, 1e-3, 1e4, th, 4), ref, 1e-8, "Hankel J4 at theta = " + num(th));
    }
}

static void test_spectra() {
    const int nt = sizeof(ref_z) / sizeof(double);
    // chi(z), z(chi), D(z) as cubic splines of CCL's tables
    std::vector<double> zt(ref_z, ref_z + nt), chit(ref_chi, ref_chi + nt), Dt(ref_D, ref_D + nt);
    detail::CubicSpline chi_of_z(zt, chit), z_of_chi(chit, zt), D(zt, Dt);
    ShearModel model;
    model.geometry.omega_m = ref_omega_m;
    model.geometry.z_of_chi = [&](double c) { return std::max(0.0, z_of_chi(c)); };
    model.chi_of_z = [&](double z) { return chi_of_z(z); };
    model.growth = [&](double z) { return D(z); };
    model.power.value = [](double k, double z) {
        const double g = std::pow(1 + z, -0.8), x = k / 0.02;
        return g * g * 2e4 * std::pow(x, 0.96) / std::pow(1 + x * x, 1.9) * (1 + g * g * std::pow(k / 0.5, 1.5));
    };
    model.power.k_min = 1e-5;
    model.power.k_max = 2e3 / ref_h;
    model.power.z_max = 7.0;

    std::vector<RedshiftDistribution> nz;
    for (int b = 0; b < 3; b++) {
        RedshiftDistribution d;
        for (int i = 0; i <= 2400; i++) {
            const double z = 0.0025 * i;
            d.z.push_back(z);
            d.n.push_back(z * z * std::exp(-std::pow(z / ref_z0[b], 1.5)));
        }
        nz.push_back(d);
    }
    ShearSystematics sys;
    sys.delta_z.assign(ref_dz, ref_dz + 3);
    ShearSettings st;
    st.spin_factor = true; // as CCL
    std::vector<double> ells(ref_ells, ref_ells + 7);

    auto t0 = std::chrono::steady_clock::now();
    const auto gg = shear_spectra(nz, model, sys, ells, st);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  shear_spectra: 3 bins, 7 ell: " << dt << " s\n";
    sys.ia.A = ref_ia_A;
    sys.ia.eta = ref_ia_eta;
    sys.ia.z_pivot = ref_ia_zpiv;
    sys.ia.C1_rho_crit = ref_C1rho;
    const auto ia = shear_spectra(nz, model, sys, ells, st);

    double egg = 0, eia = 0, egi = 0;
    for (int p = 0; p < 6; p++)
        for (int l = 0; l < 7; l++) {
            egg = std::max(egg, std::abs(gg.total[p][l] / ref_cl_gg[p][l] - 1));
            eia = std::max(eia, std::abs(ia.total[p][l] / ref_cl_tot[p][l] - 1));
            // IA part alone relative to the shear signal
            egi = std::max(egi, std::abs((ia.GI[p][l] + ia.II[p][l]) - (ref_cl_tot[p][l] - ref_cl_gg[p][l])) /
                                    ref_cl_gg[p][l]);
        }
    std::cout << "  max rel. diff. vs CCL: GG " << egg << ", GG + NLA " << eia << ", NLA part " << egi << "\n";
    check_local(egg < 5e-4, "shear C_ell (3 bins, photo-z shifts) = CCL to 5e-4, ell = 2 - 2e4");
    check_local(eia < 5e-4, "shear + NLA C_ell = CCL to 5e-4");
    // independent QUADPACK integrals with the exact n(z)
    for (const auto & r : ref_cl_quad) {
        const auto one = shear_spectra(nz, model, ShearSystematics{sys.delta_z, {}, {}}, {r[0]}, st);
        rel(one.total[one.pair(int(r[1]), int(r[2]))][0], r[3], 1e-5,
            "C_ell(" + num(r[0]) + ", " + num(r[1]) + num(r[2]) + ") = QUADPACK with the exact n(z)");
    }
    check_local(egi < 1e-4, "NLA GI + II = CCL to 1e-4 of the signal");
    // the shear part of the IA run is the GG run
    check_local(std::abs(ia.GG[3][4] / gg.GG[3][4] - 1) < 1e-14, "GG independent of the IA parameters");

    // multiplicative bias
    sys.m = {0.01, -0.02, 0.03};
    const auto mm = shear_spectra(nz, model, sys, ells, st);
    check_local(std::abs(mm.total[mm.pair(1, 2)][3] / ia.total[ia.pair(1, 2)][3] - 0.98 * 1.03) < 1e-13,
                "multiplicative bias (1 + m_i)(1 + m_j)");
    sys.m.clear();

    // xi_+- vs CCL (Bessel method) for pairs (0,0) and (1,2)
    std::vector<double> ld;
    for (int i = 0; i < 600; i++)
        ld.push_back(2.0 * std::pow(5e4, i / 599.0));
    t0 = std::chrono::steady_clock::now();
    const auto dense = shear_spectra(nz, model, sys, ld, st);
    std::vector<double> th;
    for (double t : ref_theta_arcmin)
        th.push_back(t / 60 * M_PI / 180);
    const auto xi = shear_correlation(dense, th);
    std::cout << "  600 ell + xi_+- at 7 theta, 6 pairs: "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    // the transform alone: CCL's C_ell of the pairs (0,0), (1,2) -> xi_+-
    ShearSpectra given;
    given.nbins = 1;
    given.ell.assign(ref_ell_dense, ref_ell_dense + 600);
    double ep = 0, em = 0, epm = 0;
    for (int q = 0; q < 2; q++) {
        given.total = {std::vector<double>(ref_cl_dense[q], ref_cl_dense[q] + 600)};
        const auto x = shear_correlation(given, th);
        const std::size_t pq = q == 0 ? dense.pair(0, 0) : dense.pair(1, 2);
        for (int t = 0; t < 7; t++) {
            ep = std::max(ep, std::abs(x.plus[0][t] / ref_xi[q][0][t] - 1));
            em = std::max(em, std::abs(x.minus[0][t] / ref_xi[q][1][t] - 1));
            // and with our own C_ell
            epm = std::max(epm, std::abs(xi.plus[pq][t] / ref_xi[q][0][t] - 1));
            if (ref_theta_arcmin[t] >= 1.0)
                epm = std::max(epm, std::abs(xi.minus[pq][t] / ref_xi[q][1][t] - 1));
        }
    }
    std::cout << "  xi_+- of CCL's C_ell vs QUADPACK: " << ep << ", " << em << "; of our C_ell: " << epm << "\n";
    check_local(ep < 1e-6 && em < 1e-6, "Hankel xi_+-(theta) = QUADPACK to 1e-6 (0.5' - 300')");
    check_local(epm < 6e-4, "xi_+- of our C_ell within 6e-4 (as C_ell vs CCL)");

    // binned xi in a narrow bin -> the point value
    const auto xb = shear_correlation_binned(dense, {th[3] * 0.999, th[3] * 1.001});
    check_local(std::abs(xb.plus[0][0] / xi.plus[0][3] - 1) < 1e-5, "bin-averaged xi_+ of a narrow bin");

    // COSEBIs of the spectra: ell space (W_n table, ell = 2 - 1e5) vs real space (T_+-, xi_+-)
    const double tmin = 0.5 / 60 * M_PI / 180, tmax = 300.0 / 60 * M_PI / 180;
    t0 = std::chrono::steady_clock::now();
    LogCOSEBIs cos(tmin, tmax, 5, 2.0, 1e5);
    std::cout << "  W_n table (n = 1..5, ell = 2 - 1e5): "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
    t0 = std::chrono::steady_clock::now();
    const auto En = cos.E(dense);
    std::cout << "  E_n (6 pairs): " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s; E_n(0,0) =";
    for (double e : En[0])
        std::cout << " " << e;
    std::cout << "\n";
    const std::size_t p = dense.pair(1, 2);
    detail::EllSpline cl(dense.ell, dense.total[p]);
    Function xp = [&](double t) { return hankel([&](double l) { return cl(l); }, 2.0, 1e5, t, 0); };
    Function xm = [&](double t) { return hankel([&](double l) { return cl(l); }, 2.0, 1e5, t, 4); };
    std::vector<double> E, B;
    cos.EB(xp, xm, E, B);
    double eE = 0, eB = 0;
    for (int n = 0; n < 5; n++) {
        eE = std::max(eE, std::abs(E[n] / En[p][n] - 1));
        eB = std::max(eB, std::abs(B[n] / En[p][n]));
    }
    std::cout << "  real space 1/2 int theta [T+ xi+ +- T- xi-] vs ell space: E " << eE << ", |B_n/E_n| " << eB
              << "\n";
    check_local(eE < 3e-5, "E_n in real space (T_+-, xi_+-) = E_n in ell space (W_n) to 3e-5");
    check_local(eB < 2e-5, "B_n = 0 for a pure E mode (2e-5 of E_n)");
    // W_n of the table (asymptotic Bessel + Filon) = brute force
    double eW = 0;
    for (double L : {3.0, 40.0, 300.0, 2e3, 1.3e4, 7e4}) {
        std::vector<double> W;
        cos.W_all(L, W);
        double scale = 0;
        for (int n = 1; n <= 5; n++)
            scale = std::max(scale, std::abs(cos.W(n, L)));
        for (int n = 1; n <= 5; n++)
            eW = std::max(eW, std::abs(W[n - 1] - cos.W(n, L)) / scale);
    }
    check_local(eW < 1e-6, "W_n(ell) (asymptotic J_0 + Filon) = brute force (" + num(eW, 2) + ")");
}

static void test_cosebis() {
    const double tmin = 0.5 / 60 * M_PI / 180, tmax = 300.0 / 60 * M_PI / 180;
    LogCOSEBIs c(tmin, tmax, 5, 1.0, 0.0);
    for (int n = 1; n <= 5; n++) {
        rel(c.normalization(n), ref_cos_norm[n - 1], 1e-10, "COSEBI N_" + std::to_string(n) + " (SEK10, 60 digits)");
        const auto r = c.roots(n);
        double e = r.size() == std::size_t(n + 1) ? 0.0 : 1.0;
        for (std::size_t i = 0; i < r.size() && i <= std::size_t(n); i++)
            e = std::max(e, std::abs(r[i] - ref_cos_roots[n - 1][i]));
        check_local(e < 1e-10, "COSEBI roots r_" + std::to_string(n) + "i (err " + num(e, 2) + ")");
        double et = 0;
        for (int t = 0; t < 5; t++) {
            const double th = ref_cos_theta_arcmin[t] / 60 * M_PI / 180;
            et = std::max(et, std::abs(c.T_plus(n, th) - ref_cos_T[n - 1][t][0]));
            et = std::max(et, std::abs(c.T_minus(n, th) - ref_cos_T[n - 1][t][1]));
        }
        check_local(et < 1e-10, "T_+" + std::to_string(n) + ", T_-" + std::to_string(n) + " (err " + num(et, 2) + ")");
    }
    // conditions with an independent quadrature, and T_- = 0 beyond theta_max
    std::vector<double> z, w;
    detail::panel_rule(0.0, std::log(tmax / tmin), 40, 10, z, w);
    double emax = 0;
    for (int m = 1; m <= 5; m++) {
        double a = 0, b = 0;
        for (int n = 1; n <= 5; n++) {
            double o = 0;
            for (std::size_t k = 0; k < z.size(); k++) {
                const double t = tmin * std::exp(z[k]);
                o += w[k] * t * c.t_plus_of_z(m, z[k]) * c.t_plus_of_z(n, z[k]);
                if (n == 1) {
                    a += w[k] * t * t * c.t_plus_of_z(m, z[k]);
                    b += w[k] * std::pow(t, 4) * c.t_plus_of_z(m, z[k]);
                }
            }
            emax = std::max(emax, std::abs(o / (tmax - tmin) - (m == n ? 1.0 : 0.0)));
        }
        emax = std::max(emax, std::abs(a) / (tmax * tmax) + std::abs(b) / std::pow(tmax, 4));
        emax = std::max(emax, std::abs(c.T_minus(m, 1.5 * tmax)));
    }
    check_local(emax < 1e-12, "T_+ orthonormal, int theta T_+ = int theta^3 T_+ = 0, T_- = 0 beyond theta_max (" +
                                  num(emax, 2) + ")");
    // higher n stay well conditioned
    LogCOSEBIs c20(tmin, tmax, 20, 1.0, 0.0);
    double e20 = 0;
    for (int m = 15; m <= 20; m++)
        for (int n = m; n <= 20; n++) {
            double o = 0;
            for (std::size_t k = 0; k < z.size(); k++)
                o += w[k] * tmin * std::exp(z[k]) * c20.t_plus_of_z(m, z[k]) * c20.t_plus_of_z(n, z[k]);
            e20 = std::max(e20, std::abs(o / (tmax - tmin) - (m == n ? 1.0 : 0.0)));
        }
    check_local(e20 < 1e-10 && std::abs(c20.normalization(3) / ref_cos_norm[2] - 1) < 1e-10,
                "n_max = 20: orthonormal (" + num(e20, 2) + "), lower n unchanged");
}

int main() {
    test_shift();
    test_hankel();
    test_cosebis();
    test_spectra();
    return FML::TESTS::summary("CosmicShear");
}
