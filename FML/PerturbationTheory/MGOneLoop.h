#ifndef FML_PERTURBATION_MGONELOOP_H
#define FML_PERTURBATION_MGONELOOP_H

#include <FML/PerturbationTheory/LoopIntegrals.h>
#include <FML/PerturbationTheory/SPTKernels.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

//=====================================================================================================================
///
/// One-loop SPT for modified gravity and dark energy with the perturbation-theory kernels solved numerically in time
/// (Taruya et al. 2014, arXiv:1408.4232; Bose & Koyama 2016, arXiv:1606.02520): the matter (or cold matter) obeys
///
///   delta' - theta = alpha(k1, k2) theta_1 delta_2
///   theta' + (2 + dlog E/dlog a) theta - 3/2 Omega_s(a) mu(k, a) delta = beta(k1, k2) theta_1 theta_2
///                                  + gamma2(k1, k2) delta_1 delta_2 + gamma3(k1, k2, k3) delta_1 delta_2 delta_3
///
/// (' = d/dlog a, theta = -div v / (a H), convolutions implied; Omega_s(a) = Omega_s a^-3 / E^2 the clustering
/// source) with the linear modification mu(k, a) = G_eff/G and the nonlinear (screening) sources gamma2, gamma3 of the
/// scalar field. gamma3 is given as one cyclic term (k1 alone, k2 and k3 the pair); the source is the average over
/// the three.
///
/// The kernels F1(k), F1(q), F1(|k - q|), F2(q, k - q), F2(k, q), F2(k, -q) and F3(k, q, -q) are integrated in log a
/// (4th-order Runge-Kutta from a_ini with the growing mode delta = theta = a) for each loop node (P22 over q and
/// p = |k - q|, which resolves p -> 0; P13 over q and the cosine), so
///
///   P(k) = P_L(k) + 2 int d^3q/(2pi)^3 F2^2 P0(q) P0(|k - q|) + 6 P0(k) F1(k) int d^3q/(2pi)^3 F3(k, q, -q) P0(q)
///
/// with P0 = P_L / F1^2 (the linear spectra are given at each output time, as ReACT's input).
///
/// Models: mg_lcdm (GR with any background), mg_hu_sawicki (Hu-Sawicki f(R) n = 1 in the large-curvature limit:
/// mu, gamma2 and gamma3 from the expansion of the scalar-field equation, Koyama, Taruya & Hiramatsu 2009,
/// arXiv:0902.0618, eq. 3.6; Bose & Koyama 2016, arXiv:1606.02520, eqs. 2.32-2.35) and mg_ndgp (normal-branch DGP,
/// Vainshtein nonlinearities, Bose & Koyama 2016 eqs. 2.37-2.39). The third-order f(R) field perturbation reduces to
/// the exact expansion of R(f_R) for a homogeneous perturbation (tests/test_mg_oneloop.cpp). ReACT's f(R) gamma3
/// (BeyondLCDM.cpp, a36dda02) is 1/3 of this (an extra 1/3 on top of the average over the three splits). ReACT uses this one-loop P at
/// k0 = 0.06 h/Mpc for k* (HaloModel/HaloModelReaction.h): the real P_cb with the model's kernels and the pseudo P with
/// the GR kernels of the LCDM background.
///
/// Units: k [h/Mpc], P [(Mpc/h)^3], H0 = 1 / 2997.92458 h/Mpc (c = 1).
///
//=====================================================================================================================

namespace FML::PERTURBATION {

    /// The model: background and the modifications of the Poisson equation (functions of a and wave vectors in h/Mpc)
    struct MGKernelModel {
        using Vec3 = SPT::Vec3;
        std::function<double(double a)> E;            ///< H / H0
        std::function<double(double a)> dlogE_dloga;  ///< dlog H / dlog a
        double Omega_source{0.3};                     ///< The clustering matter today in the Poisson source
        std::function<double(double a, double k)> mu; ///< G_eff / G (empty: 1)
        /// The second-order source: coefficient of delta(k1) delta(k2) in the theta equation (empty: 0)
        std::function<double(double a, const Vec3 & k1, const Vec3 & k2)> gamma2;
        /// One cyclic term of the third-order source (k1 alone; k2, k3 the pair) (empty: 0)
        std::function<double(double a, const Vec3 & k1, const Vec3 & k2, const Vec3 & k3)> gamma3;
    };

    /// H0 in h/Mpc
    constexpr double H0_hMpc = 1.0 / 2997.92458;

    /// GR on a flat LCDM background with matter density OmegaM (in H) and Omega_source clustering
    inline MGKernelModel mg_lcdm(double OmegaM, double Omega_source) {
        MGKernelModel m;
        m.E = [OmegaM](double a) { return std::sqrt(OmegaM / (a * a * a) + 1.0 - OmegaM); };
        m.dlogE_dloga = [OmegaM](double a) {
            const double x = OmegaM / (a * a * a);
            return -1.5 * x / (x + 1.0 - OmegaM);
        };
        m.Omega_source = Omega_source;
        return m;
    }

    /// Hu-Sawicki f(R) with n = 1 (|fR0|) on a flat LCDM background, OmegaM in the curvature and the background
    namespace detail {
        struct HuSawicki {
            double fR0, Om;
            // (OmegaM a^-3 + 4 OmegaL) and the derivatives d^n R / d f_R^n of R(f_R) in units of H0^2 (k in h/Mpc)
            double rho(double a) const { return Om / (a * a * a) + 4.0 * (1.0 - Om); }
            double M1(double a) const {
                return 1.5 * H0_hMpc * H0_hMpc * std::pow(rho(a), 3) / (std::pow(4.0 - 3.0 * Om, 2) * fR0);
            }
            double M2(double a) const {
                return 2.25 * H0_hMpc * H0_hMpc * std::pow(rho(a), 5) / (std::pow(4.0 - 3.0 * Om, 4) * fR0 * fR0);
            }
            double M3(double a) const {
                return 45.0 / 8.0 * H0_hMpc * H0_hMpc * std::pow(rho(a), 7) /
                       (std::pow(4.0 - 3.0 * Om, 6) * fR0 * fR0 * fR0);
            }
            // Pi(k) = (k/a)^2 + M1/3
            double Pi(double a, double k2) const { return k2 / (a * a) + M1(a) / 3.0; }
        };
    } // namespace detail

    inline MGKernelModel mg_hu_sawicki(double fR0, double OmegaM, double Omega_source) {
        if (not(fR0 > 0.0))
            throw std::invalid_argument("mg_hu_sawicki: need |fR0| > 0");
        MGKernelModel m = mg_lcdm(OmegaM, Omega_source);
        const detail::HuSawicki hs{fR0, OmegaM};
        // The field equation -(k/a)^2 df = (dR - 8 pi G drho) / 3, dR = M1 df + M2 df^2 / 2 + M3 df^3 / 6, and
        // -(k/a)^2 Psi = 4 pi G drho + (k/a)^2 df / 2, 8 pi G rho / 3 = H0^2 OmegaM a^-3
        m.mu = [hs](double a, double k) { return 1.0 + k * k / (a * a) / (3.0 * hs.Pi(a, k * k)); };
        const auto E = m.E;
        m.gamma2 = [hs, E](double a, const SPT::Vec3 & k1, const SPT::Vec3 & k2) {
            const double k2tot = (k1 + k2).norm2();
            const double s = H0_hMpc * H0_hMpc * hs.Om / (a * a * a);
            const double H2 = std::pow(H0_hMpc * E(a), 2);
            return -k2tot / (a * a) * hs.M2(a) * s * s /
                   (12.0 * H2 * hs.Pi(a, k2tot) * hs.Pi(a, k1.norm2()) * hs.Pi(a, k2.norm2()));
        };
        m.gamma3 = [hs, E](double a, const SPT::Vec3 & k1, const SPT::Vec3 & k2, const SPT::Vec3 & k3) {
            const double k2tot = (k1 + k2 + k3).norm2();
            const double s = H0_hMpc * H0_hMpc * hs.Om / (a * a * a);
            const double H2 = std::pow(H0_hMpc * E(a), 2);
            const double M2 = hs.M2(a);
            return k2tot / (a * a) * s * s * s *
                   (M2 * M2 / hs.Pi(a, (k2 + k3).norm2()) - hs.M3(a)) /
                   (36.0 * H2 * hs.Pi(a, k2tot) * hs.Pi(a, k1.norm2()) * hs.Pi(a, k2.norm2()) *
                    hs.Pi(a, k3.norm2()));
        };
        return m;
    }

    /// Normal-branch DGP (Omega_rc = 1 / (4 r_c^2 H0^2)) on a flat LCDM background
    inline MGKernelModel mg_ndgp(double Omega_rc, double OmegaM, double Omega_source) {
        if (not(Omega_rc > 0.0))
            throw std::invalid_argument("mg_ndgp: need Omega_rc > 0");
        MGKernelModel m = mg_lcdm(OmegaM, Omega_source);
        const auto E = m.E, dlogE = m.dlogE_dloga;
        auto beta = [=](double a) { return 1.0 + E(a) / std::sqrt(Omega_rc) * (1.0 + dlogE(a) / 3.0); };
        auto ker = [](const SPT::Vec3 & p, const SPT::Vec3 & q) {
            const double pq = p.dot(q), p2q2 = p.norm2() * q.norm2();
            return p2q2 > 0.0 ? 1.0 - pq * pq / p2q2 : 0.0;
        };
        // 3 beta lap(phi) + r_c^2 / a^2 [(lap phi)^2 - (grad grad phi)^2] = 8 pi G a^2 drho, Psi = Psi_N + phi / 2
        m.mu = [beta](double a, double) { return 1.0 + 1.0 / (3.0 * beta(a)); };
        m.gamma2 = [=](double a, const SPT::Vec3 & k1, const SPT::Vec3 & k2) {
            const double b = beta(a), s = OmegaM / (a * a * a);
            return -s * s * ker(k1, k2) / (24.0 * b * b * b * Omega_rc * E(a) * E(a));
        };
        m.gamma3 = [=](double a, const SPT::Vec3 & k1, const SPT::Vec3 & k2, const SPT::Vec3 & k3) {
            const double b = beta(a), s = OmegaM / (a * a * a);
            return s * s * s * ker(k1, k2 + k3) * ker(k2, k3) /
                   (144.0 * std::pow(b, 5) * Omega_rc * Omega_rc * E(a) * E(a));
        };
        return m;
    }

    struct MGOneLoopOptions {
        double qmin{1e-4}, qmax{30.0}; ///< The loop range in q [h/Mpc] (ReACT: 1e-4 - 30)
        int order{8};                  ///< Gauss-Legendre points per segment in log q and log p
        double log_step{0.5};          ///< Segment length in log q and log p
        int nmu{16};                   ///< Gauss-Legendre points in the angle of the P13 integral
        double a_ini{1e-5};            ///< Initial time (growing mode delta = theta = a)
        int steps_per_efold{20};       ///< RK4 steps per e-fold in a
        // The defaults give the loop to ~2e-4 at k = 0.06 h/Mpc and 6e-4 at 0.2 h/Mpc (EdS against OneLoop.h; f(R)
        // against a refined quadrature), ~1 s for a spectrum at four redshifts on 4 threads
    };

    namespace detail {
        /// The kernels at the output times for one loop node (k, q at cosine x)
        struct KernelValues {
            double F1k, F1q, F1kq, F2_22, F3;
        };

        inline std::vector<KernelValues> mg_kernels(const MGKernelModel & m,
                                                    const SPT::Vec3 & k,
                                                    const SPT::Vec3 & q,
                                                    const std::vector<double> & a_out,
                                                    const MGOneLoopOptions & opt) {
            using Vec3 = SPT::Vec3;
            const Vec3 mq = q * -1.0, kmq = k - q, kpq = k + q;
            auto mu = [&](double a, const Vec3 & v) { return m.mu ? m.mu(a, std::sqrt(v.norm2())) : 1.0; };
            auto g2 = [&](double a, const Vec3 & v1, const Vec3 & v2) { return m.gamma2 ? m.gamma2(a, v1, v2) : 0.0; };
            auto alpha = [](const Vec3 & v1, const Vec3 & v2) {
                const double n = v1.norm2();
                return n > 0.0 ? 1.0 + v1.dot(v2) / n : 0.0;
            };
            auto beta = [](const Vec3 & v1, const Vec3 & v2) {
                const double n = v1.norm2() * v2.norm2();
                return n > 0.0 ? (v1 + v2).norm2() * v1.dot(v2) / (2.0 * n) : 0.0;
            };
            // Constant geometric factors
            const double a_q_kmq = alpha(q, kmq), a_kmq_q = alpha(kmq, q), b_q_kmq = beta(q, kmq);
            const double a_k_q = alpha(k, q), a_q_k = alpha(q, k), b_k_q = beta(k, q);
            const double a_k_mq = alpha(k, mq), a_mq_k = alpha(mq, k), b_k_mq = beta(k, mq);
            const double a3_1 = alpha(mq, kpq), a3_2 = alpha(kpq, mq), a3_3 = alpha(q, kmq), a3_4 = alpha(kmq, q);
            const double b3_1 = beta(mq, kpq), b3_2 = beta(q, kmq);
            // y: 0,1 F1/T1(k); 2,3 (q); 4,5 (k - q); 6,7 F2/T2(q, k - q); 8,9 (k, q); 10,11 (k, -q); 12,13 F3/T3
            auto rhs = [&](double a, const double * y, double * dy) {
                const double h = m.dlogE_dloga(a);
                const double S = 1.5 * m.Omega_source / (a * a * a) / std::pow(m.E(a), 2);
                const double fr = 2.0 + h;
                dy[0] = y[1];
                dy[1] = -fr * y[1] + S * mu(a, k) * y[0];
                dy[2] = y[3];
                dy[3] = -fr * y[3] + S * mu(a, q) * y[2];
                dy[4] = y[5];
                dy[5] = -fr * y[5] + S * mu(a, kmq) * y[4];
                // F2(q, k - q)
                dy[6] = y[7] + 0.5 * (a_q_kmq * y[3] * y[4] + a_kmq_q * y[5] * y[2]);
                dy[7] = -fr * y[7] + S * mu(a, k) * y[6] + g2(a, q, kmq) * y[2] * y[4] + b_q_kmq * y[3] * y[5];
                // F2(k, q)
                dy[8] = y[9] + 0.5 * (a_k_q * y[1] * y[2] + a_q_k * y[3] * y[0]);
                dy[9] = -fr * y[9] + S * mu(a, kpq) * y[8] + g2(a, k, q) * y[0] * y[2] + b_k_q * y[1] * y[3];
                // F2(k, -q)
                dy[10] = y[11] + 0.5 * (a_k_mq * y[1] * y[2] + a_mq_k * y[3] * y[0]);
                dy[11] = -fr * y[11] + S * mu(a, kmq) * y[10] + g2(a, k, mq) * y[0] * y[2] + b_k_mq * y[1] * y[3];
                // F3(k, q, -q): the splits (-q | k + q) and (q | k - q); F2(q, -q) = 0
                dy[12] = y[13] + (a3_1 * y[3] * y[8] + a3_2 * y[9] * y[2] + a3_3 * y[3] * y[10] + a3_4 * y[11] * y[2]) /
                                     3.0;
                double g3 = 0.0;
                if (m.gamma3)
                    g3 = (m.gamma3(a, k, q, mq) + m.gamma3(a, q, mq, k) + m.gamma3(a, mq, k, q)) / 3.0;
                dy[13] = -fr * y[13] + S * mu(a, k) * y[12] +
                         (2.0 * b3_1 * y[3] * y[9] + 2.0 * b3_2 * y[3] * y[11] + 2.0 * g2(a, mq, kpq) * y[2] * y[8] +
                          2.0 * g2(a, q, kmq) * y[2] * y[10]) /
                             3.0 +
                         g3 * y[2] * y[2] * y[0];
            };
            constexpr int N = 14;
            double y[N] = {0};
            const double ai = opt.a_ini;
            for (int i = 0; i < 6; i++)
                y[i] = ai;
            std::vector<KernelValues> out;
            double x = std::log(ai);
            double k1[N], k2[N], k3[N], k4[N], t[N];
            for (double aout : a_out) {
                const double xend = std::log(aout);
                const int n = std::max(1, int(std::ceil((xend - x) * opt.steps_per_efold)));
                const double dx = (xend - x) / n;
                for (int s = 0; s < n; s++) {
                    auto f = [&](double xx, const double * yy, double * dd) {
                        const double a = std::exp(xx);
                        rhs(a, yy, dd);
                    };
                    f(x, y, k1);
                    for (int i = 0; i < N; i++)
                        t[i] = y[i] + 0.5 * dx * k1[i];
                    f(x + 0.5 * dx, t, k2);
                    for (int i = 0; i < N; i++)
                        t[i] = y[i] + 0.5 * dx * k2[i];
                    f(x + 0.5 * dx, t, k3);
                    for (int i = 0; i < N; i++)
                        t[i] = y[i] + dx * k3[i];
                    f(x + dx, t, k4);
                    for (int i = 0; i < N; i++)
                        y[i] += dx / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                    x += dx;
                }
                out.push_back({y[0], y[2], y[4], y[6], y[12]});
            }
            return out;
        }
    } // namespace detail

    /// The one-loop P(k) (linear + one loop) at the times a_out (increasing). P(k, i) is the linear spectrum at a_out[i]
    inline std::vector<double> mg_one_loop(const MGKernelModel & model,
                                           double k,
                                           const std::vector<double> & a_out,
                                           const std::function<double(double k, size_t i)> & P,
                                           const MGOneLoopOptions & opt = {}) {
        if (not(k > 0.0) or a_out.empty() or not(opt.qmax > opt.qmin and opt.qmin > 0.0))
            throw std::invalid_argument("mg_one_loop: need k > 0, a_out and 0 < qmin < qmax");
        for (size_t i = 0; i < a_out.size(); i++)
            if (not(a_out[i] > opt.a_ini) or (i > 0 and not(a_out[i] > a_out[i - 1])))
                throw std::invalid_argument("mg_one_loop: a_out must be increasing and > a_ini");
        if (not model.E or not model.dlogE_dloga)
            throw std::invalid_argument("mg_one_loop: the model needs E(a) and dlogE/dloga");
        const auto gl = LOOPS::detail::gauss_legendre(opt.order);
        LOOPS::QuadratureOptions qo;
        qo.log_step = opt.log_step;
        const auto glmu = LOOPS::detail::gauss_legendre(opt.nmu);
        const size_t nz = a_out.size();
        const SPT::Vec3 kv{0.0, 0.0, k};
        // The loop nodes: P22 over (q, p = |k - q|) (log Gauss-Legendre in both, resolving p -> 0), P13 over (q, x)
        struct Node {
            double q, x, w;
            bool p22;
        };
        std::vector<Node> nodes;
        {
            std::vector<double> qs, wq;
            LOOPS::detail::log_nodes({opt.qmin, k, opt.qmax}, qo, gl, qs, wq);
            for (size_t iq = 0; iq < qs.size(); iq++) {
                const double q = qs[iq];
                // P13: d^3q / (2 pi)^3 = q^2 dq dx / (4 pi^2)
                for (size_t im = 0; im < glmu.first.size(); im++)
                    nodes.push_back({q, glmu.first[im], wq[iq] * glmu.second[im] * q * q / (4.0 * M_PI * M_PI), false});
                // P22: d^3q / (2 pi)^3 = q p dq dp / (4 pi^2 k)
                const double pmin = std::max(std::abs(k - q), opt.qmin), pmax = std::min(k + q, opt.qmax);
                if (not(pmax > pmin))
                    continue;
                std::vector<double> ps, wp;
                LOOPS::detail::log_nodes({pmin, pmax}, qo, gl, ps, wp);
                for (size_t ip = 0; ip < ps.size(); ip++) {
                    const double x = std::clamp((k * k + q * q - ps[ip] * ps[ip]) / (2.0 * k * q), -1.0, 1.0);
                    nodes.push_back({q, x, wq[iq] * wp[ip] * q * ps[ip] / (4.0 * M_PI * M_PI * k), true});
                }
            }
        }
        std::vector<double> loop(nz, 0.0);
        std::vector<double> Pk(nz);
        for (size_t i = 0; i < nz; i++)
            Pk[i] = P(k, i);
#ifdef USE_OMP
#pragma omp parallel
#endif
        {
            std::vector<double> local(nz, 0.0);
#ifdef USE_OMP
#pragma omp for schedule(dynamic, 16)
#endif
            for (size_t n = 0; n < nodes.size(); n++) {
                const auto & nd = nodes[n];
                const SPT::Vec3 qv{nd.q * std::sqrt(std::max(0.0, 1.0 - nd.x * nd.x)), 0.0, nd.q * nd.x};
                const double kmq = std::sqrt((kv - qv).norm2());
                const auto K = detail::mg_kernels(model, kv, qv, a_out, opt);
                for (size_t i = 0; i < nz; i++) {
                    const double P0q = P(nd.q, i) / (K[i].F1q * K[i].F1q);
                    if (nd.p22) {
                        const double P0p = P(kmq, i) / (K[i].F1kq * K[i].F1kq);
                        local[i] += nd.w * 2.0 * K[i].F2_22 * K[i].F2_22 * P0q * P0p;
                    } else {
                        const double P0k = Pk[i] / (K[i].F1k * K[i].F1k);
                        local[i] += nd.w * 6.0 * P0k * K[i].F1k * K[i].F3 * P0q;
                    }
                }
            }
#ifdef USE_OMP
#pragma omp critical
#endif
            for (size_t i = 0; i < nz; i++)
                loop[i] += local[i];
        }
        std::vector<double> out(nz);
        for (size_t i = 0; i < nz; i++)
            out[i] = P(k, i) + loop[i];
        return out;
    }

} // namespace FML::PERTURBATION

#endif
