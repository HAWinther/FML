// =============================================================================
// How good are EdS loop kernels for the Fading Dark Sector (desi_dr1_fs uses them; Obied, Vafa & Wu use exact-time
// two-fluid kernels)? The one-loop real-space P_cb and P_thetatheta with exact-time two-fluid kernels (quasi-static:
// CDM with the fifth force G(1 + 2c'^2) from the CDM and the drag -c' phi_x, baryons with neither; the kernels
// integrated in ln a for every loop node, from a = 1e-3 with delta = theta = a for both fluids, on the model's
// background) against EdS kernels, for the same linear P_cb of the run (P0 = P_L/F1^2 for each kernel set).
//
//   make check_fds_kernels && ./check_fds_kernels c c'      (from CosmologyFitting/, e.g. -0.85 0.053; 0 0 = LCDM)
//
// Output per z (0.295, 0.51, 0.92, 1.49) and k [h/Mpc]: P_L, the delta-delta loop with two-fluid and EdS kernels,
// the theta-theta loop with both (EdS: times f^2), f. Results (README, "Fading Dark Sector").
// =============================================================================
#include <FML/Cosmology/Common.h>
#include <FML/CosmologyFitting/Theory.h>
#include <FML/PerturbationTheory/LoopIntegrals.h>
#include <FML/PerturbationTheory/SPTKernels.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace FML::COSMOLOGY;
using namespace FML::COSMOLOGY::FITTING;
using Vec3 = FML::PERTURBATION::SPT::Vec3;
namespace LOOPS = FML::PERTURBATION::LOOPS;

struct BG { // tabulated in x = ln a
    std::vector<double> x, Oc, Ob, h, drag;
    double get(const std::vector<double> & v, double xx) const {
        if (xx <= x.front()) return v.front();
        if (xx >= x.back()) return v.back();
        const double t = (xx - x.front()) / (x[1] - x[0]);
        const size_t i = size_t(t);
        const double f = t - i;
        return v[i] * (1 - f) + v[i + 1] * f;
    }
};

struct KV { double F1k, F2, F3, G1k, G2, G3, F1q, F1kq; };

// mode: 0 = two fluids with the model background; 1 = EdS (Omega = 1, one fluid)
std::vector<KV> kernels(const BG & bg, double mucc, const Vec3 & k, const Vec3 & q, const std::vector<double> & aout,
                        int mode, double a_ini = 1e-3, int steps = 40) {
    const Vec3 mq = q * -1.0, kmq = k - q, kpq = k + q;
    auto alpha = [](const Vec3 & v1, const Vec3 & v2) { const double n = v1.norm2(); return n > 0 ? 1.0 + v1.dot(v2) / n : 0.0; };
    auto beta = [](const Vec3 & v1, const Vec3 & v2) {
        const double n = v1.norm2() * v2.norm2(); return n > 0 ? (v1 + v2).norm2() * v1.dot(v2) / (2.0 * n) : 0.0; };
    const double a_q_kmq = alpha(q, kmq), a_kmq_q = alpha(kmq, q), b_q_kmq = beta(q, kmq);
    const double a_k_q = alpha(k, q), a_q_k = alpha(q, k), b_k_q = beta(k, q);
    const double a_k_mq = alpha(k, mq), a_mq_k = alpha(mq, k), b_k_mq = beta(k, mq);
    const double a3_1 = alpha(mq, kpq), a3_2 = alpha(kpq, mq), a3_3 = alpha(q, kmq), a3_4 = alpha(kmq, q);
    const double b3_1 = beta(mq, kpq), b3_2 = beta(q, kmq);
    constexpr int NF = 14, N = 28; // per fluid: 0..5 linear (k, q, k-q), 6..11 F2 (q,k-q),(k,q),(k,-q), 12,13 F3
    auto rhs = [&](double x, const double * y, double * dy) {
        double Oc, Ob, h, drag;
        if (mode == 1) { Oc = 1.0; Ob = 0.0; h = -1.5; drag = 0.0; }
        else { Oc = bg.get(bg.Oc, x); Ob = bg.get(bg.Ob, x); h = bg.get(bg.h, x); drag = bg.get(bg.drag, x); }
        for (int s = 0; s < 2; s++) {
            const double * Y = y + NF * s;
            double * D = dy + NF * s;
            const double fr = 2.0 + h + (s == 0 ? drag : 0.0);
            // the gravitational source at each order from both fluids: 1.5 (Oc mu_sc d_c + Ob d_b)
            auto src = [&](int i) {
                const double mu = (s == 0 ? mucc : 1.0);
                return 1.5 * (Oc * mu * y[i] + Ob * y[NF + i]);
            };
            D[0] = Y[1]; D[1] = -fr * Y[1] + src(0);
            D[2] = Y[3]; D[3] = -fr * Y[3] + src(2);
            D[4] = Y[5]; D[5] = -fr * Y[5] + src(4);
            D[6] = Y[7] + 0.5 * (a_q_kmq * Y[3] * Y[4] + a_kmq_q * Y[5] * Y[2]);
            D[7] = -fr * Y[7] + src(6) + b_q_kmq * Y[3] * Y[5];
            D[8] = Y[9] + 0.5 * (a_k_q * Y[1] * Y[2] + a_q_k * Y[3] * Y[0]);
            D[9] = -fr * Y[9] + src(8) + b_k_q * Y[1] * Y[3];
            D[10] = Y[11] + 0.5 * (a_k_mq * Y[1] * Y[2] + a_mq_k * Y[3] * Y[0]);
            D[11] = -fr * Y[11] + src(10) + b_k_mq * Y[1] * Y[3];
            D[12] = Y[13] + (a3_1 * Y[3] * Y[8] + a3_2 * Y[9] * Y[2] + a3_3 * Y[3] * Y[10] + a3_4 * Y[11] * Y[2]) / 3.0;
            D[13] = -fr * Y[13] + src(12) + (2.0 * b3_1 * Y[3] * Y[9] + 2.0 * b3_2 * Y[3] * Y[11]) / 3.0;
        }
    };
    double y[N] = {0};
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 6; i++)
            y[NF * s + i] = a_ini;
    std::vector<KV> out;
    double x = std::log(a_ini), k1[N], k2[N], k3[N], k4[N], t[N];
    for (double ao : aout) {
        const double xe = std::log(ao);
        const int n = std::max(1, int(std::ceil((xe - x) * steps)));
        const double dx = (xe - x) / n;
        for (int st = 0; st < n; st++) {
            rhs(x, y, k1);
            for (int i = 0; i < N; i++) t[i] = y[i] + 0.5 * dx * k1[i];
            rhs(x + 0.5 * dx, t, k2);
            for (int i = 0; i < N; i++) t[i] = y[i] + 0.5 * dx * k2[i];
            rhs(x + 0.5 * dx, t, k3);
            for (int i = 0; i < N; i++) t[i] = y[i] + dx * k3[i];
            rhs(x + dx, t, k4);
            for (int i = 0; i < N; i++) y[i] += dx / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
            x += dx;
        }
        double wc = 1.0, wb = 0.0;
        if (mode == 0) { const double Oc = bg.get(bg.Oc, x), Ob = bg.get(bg.Ob, x); wc = Oc / (Oc + Ob); wb = 1 - wc; }
        auto cb = [&](int i) { return wc * y[i] + wb * y[NF + i]; };
        out.push_back({cb(0), cb(6), cb(12), cb(1), cb(7), cb(13), cb(2), cb(4)});
    }
    return out;
}

int main(int argc, char ** argv) {
    Utils::verbose = false;
    const double c = std::stod(argv[1]), cp = std::stod(argv[2]);
    std::vector<std::string> args{"preset=planck2018", "m_nu=0.06"};
    if (cp != 0.0 || c != 0.0) {
        args.push_back("model=fading_dark_sector");
        args.push_back("fds_c=" + std::to_string(c));
        args.push_back("fds_cprime=" + std::to_string(cp));
    }
    RunSettings s = ParameterFile::read("", args);
    const std::vector<double> zs{0.295, 0.51, 0.92, 1.49};
    TheoryOptions o; o.cmb = false; o.pk_z = zs;
    Theory t(s, o);
    const auto & b = t.background();
    BG bg;
    const int nx = 4000;
    for (int i = 0; i < nx; i++) {
        const double x = std::log(1e-4) + (0.0 - std::log(1e-4)) * i / (nx - 1.0), d = 1e-4;
        bg.x.push_back(x);
        bg.Oc.push_back(b.OmegaCDM_of_x(x));
        bg.Ob.push_back(b.OmegaB_of_x(x));
        bg.h.push_back(0.5 * b.dE2dx_of_x(x) / b.E2_of_x(x));
        auto lnrc = [&](double xx) { return std::log(b.OmegaCDM_of_x(xx) * b.E2_of_x(xx)); };
        // drag = -c' phi_x = 3 + dln rho_c/dx
        bg.drag.push_back(3.0 + (lnrc(x + d) - lnrc(x - d)) / (2 * d));
    }
    const double mucc = 1.0 + 2.0 * cp * cp;
    std::vector<double> aout;
    for (double z : zs) aout.push_back(1.0 / (1.0 + z));
    // P_L(k) [h/Mpc, (Mpc/h)^3] by log-log interpolation of pk_cb
    const auto & kg = Theory::pk_k_grid();
    std::vector<std::vector<double>> PT;
    for (double z : zs) PT.push_back(t.pk_cb(z)); // solve before the threads
    auto PL = [&](double k, size_t iz) {
        const auto & P = PT[iz];
        const double lk = std::log10(k), l0 = std::log10(kg.front()), dl = (std::log10(kg.back()) - l0) / (kg.size() - 1);
        double u = (lk - l0) / dl; size_t i = std::min(kg.size() - 2, size_t(std::max(0.0, u))); double f = u - i;
        return std::exp(std::log(P[i]) * (1 - f) + std::log(P[i + 1]) * f);
    };
    const double qmin = 1e-4, qmax = 10.0;
    const auto gl = LOOPS::detail::gauss_legendre(10);
    const auto glmu = LOOPS::detail::gauss_legendre(16);
    LOOPS::QuadratureOptions qo; qo.log_step = 0.3;
    printf("# c = %g, c' = %g; z, k, P_L, loop_dd two-fluid, loop_dd EdS, loop_tt two-fluid, loop_tt EdS, f\n", c, cp);
    for (double k : {0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3}) {
        const Vec3 kv{0, 0, k};
        struct Node { double q, x, w; bool p22; };
        std::vector<Node> nodes;
        std::vector<double> qs, wq;
        LOOPS::detail::log_nodes({qmin, k, qmax}, qo, gl, qs, wq);
        for (size_t iq = 0; iq < qs.size(); iq++) {
            const double q = qs[iq];
            for (size_t im = 0; im < glmu.first.size(); im++)
                nodes.push_back({q, glmu.first[im], wq[iq] * glmu.second[im] * q * q / (4 * M_PI * M_PI), false});
            const double pmin = std::max(std::abs(k - q), qmin), pmax = std::min(k + q, qmax);
            if (!(pmax > pmin)) continue;
            std::vector<double> ps, wp;
            LOOPS::detail::log_nodes({pmin, pmax}, qo, gl, ps, wp);
            for (size_t ip = 0; ip < ps.size(); ip++) {
                const double x = std::clamp((k * k + q * q - ps[ip] * ps[ip]) / (2 * k * q), -1.0, 1.0);
                nodes.push_back({q, x, wq[iq] * wp[ip] * q * ps[ip] / (4 * M_PI * M_PI * k), true});
            }
        }
        const size_t nz = zs.size();
        std::vector<double> L[2][2]; // [mode][dd, tt]
        for (int m = 0; m < 2; m++) for (int v = 0; v < 2; v++) L[m][v].assign(nz, 0.0);
        std::vector<double> fz(nz);
#pragma omp parallel for schedule(dynamic, 32)
        for (size_t n = 0; n < nodes.size(); n++) {
            const auto & nd = nodes[n];
            const Vec3 qv{nd.q * std::sqrt(std::max(0.0, 1 - nd.x * nd.x)), 0, nd.q * nd.x};
            const double kmq = std::sqrt((kv - qv).norm2());
            for (int m = 0; m < 2; m++) {
                const auto K = kernels(bg, mucc, kv, qv, aout, m);
                for (size_t i = 0; i < nz; i++) {
                    // the same linear P_cb for both: P0 = P_L / F1^2 of each kernel set
                    const double P0q = PL(nd.q, i) / (K[i].F1q * K[i].F1q);
                    double vdd, vtt;
                    if (nd.p22) {
                        const double P0p = PL(kmq, i) / (K[i].F1kq * K[i].F1kq);
                        vdd = nd.w * 2 * K[i].F2 * K[i].F2 * P0q * P0p;
                        vtt = nd.w * 2 * K[i].G2 * K[i].G2 * P0q * P0p;
                    } else {
                        const double P0k = PL(k, i) / (K[i].F1k * K[i].F1k);
                        vdd = nd.w * 6 * P0k * K[i].F1k * K[i].F3 * P0q;
                        vtt = nd.w * 6 * P0k * K[i].G1k * K[i].G3 * P0q;
                    }
#pragma omp atomic
                    L[m][0][i] += vdd;
#pragma omp atomic
                    L[m][1][i] += vtt;
                }
                if (n == 0 && m == 0)
                    for (size_t i = 0; i < nz; i++) fz[i] = K[i].G1k / K[i].F1k;
            }
        }
        for (size_t i = 0; i < nz; i++) {
            // the EdS theta loop is in units of f_EdS = 1: scale by f^2 (EdS kernels with the true linear velocity)
            printf("%.3f %.3f %.8e %.8e %.8e %.8e %.8e %.6f\n", zs[i], k, PL(k, i), L[0][0][i], L[1][0][i], L[0][1][i],
                   L[1][1][i] * fz[i] * fz[i], fz[i]);
        }
        fflush(stdout);
    }
}
