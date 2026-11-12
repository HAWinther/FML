#ifndef FML_PERTURBATION_TNS_H
#define FML_PERTURBATION_TNS_H

#include <FML/PerturbationTheory/LoopIntegrals.h>
#include <FML/PerturbationTheory/SPTKernels.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

//=====================================================================================================================
///
/// The TNS model of the redshift-space power spectrum (Taruya, Nishimichi & Saito 2010, arXiv:1006.0699) with
/// one-loop standard perturbation theory:
///
///   P_s(k, mu) = D_FoG(k mu f sigma_v) [P_g,dd + 2 f mu^2 P_g,dt + f^2 mu^4 P_tt + A(k, mu) + B(k, mu)] + N
///
///  - P_dd, P_dt, P_tt: the one-loop SPT density, density-velocity and velocity spectra (P_lin + P22 + P13 with the
///    EdS F, G kernels; theta = -div v / (a H f), so all three are P_lin at linear order).
///  - A and B: the TNS correction terms (TNS eq. 19-20) from the tree-level bispectrum
///    <theta {delta_g + f mu^2 theta} {delta_g + f mu^2 theta}> and the product of the linear velocity-density
///    spectra. With a linear bias b1 they are A = sum_n f^n b1^(3-n) A_n(k, mu) and B = sum_n f^n b1^(4-n) B_n(k, mu)
///    (A_n, B_n polynomials in mu^2; the tables hold their mu^(2m) coefficients, so f and b1 can be varied freely).
///  - The galaxy spectra with the nonlinear bias of McDonald & Roy (2009) as used in BOSS TNS fits (Beutler et al.
///    2014, arXiv:1312.4611, eq. 19-27; b2, the tidal bias bs2 and the third order nonlocal b3nl):
///      P_g,dd = b1^2 P_dd + 2 b1 b2 P_b2,d + 2 b1 bs2 P_bs2,d + 2 b1 b3nl sigma3^2 P_lin + b2^2 P_b22
///               + 2 b2 bs2 P_b2s2 + bs2^2 P_bs22
///      P_g,dt = b1 P_dt + b2 P_b2,t + bs2 P_bs2,t + b3nl sigma3^2 P_lin
///    (local Lagrangian bias: bs2 = -4/7 (b1 - 1), b3nl = 32/315 (b1 - 1)). P_b22, P_b2s2, P_bs22 have their
///    k -> 0 constants subtracted (they are absorbed by the shot noise N).
///  - D_FoG: none, Gaussian exp(-x^2) or Lorentzian 1 / (1 + x^2) with x = k mu f sigma_v (sigma_v in Mpc/h).
///
/// All loop integrals are direct quadratures of the kernels (LoopIntegrals.h), with P_lin zero outside its table:
/// supply P_lin over a wide range (e.g. 1e-5 - 50 h/Mpc). Units: k [h/Mpc], P [(Mpc/h)^3].
///
/// Validation (tests/test_tns.cpp): P_dd against OneLoop.h; P13 of dt and tt against their analytic kernels; A + B
/// (b1 = 1) to 2e-3 and the nonlinear bias terms to 3e-4 against FAST-PT (McEwen et al. 2016).
///
//=====================================================================================================================

namespace FML::PERTURBATION {

    struct TNSOptions {
        LOOPS::QuadratureOptions quadrature{};
        /// The integration range of the loops (0: the range of the P_lin table)
        double qmin = 0.0, qmax = 0.0;
        /// Gauss-Legendre points in mu for the multipoles
        int nmu_multipoles = 24;
    };

    class TNS {
      public:
        using DVector = std::vector<double>;

        using Options = TNSOptions;

        /// The k-dependent tables on the output k: P_lin, the one-loop spectra (the loop part, without P_lin), the
        /// nonlinear bias terms and A_n, B_n as coefficients of mu^(2m), m = 0..4: A[n][m] multiplies f^n b1^(3-n)
        /// mu^(2m) (n = 1, 2, 3), B[n][m] multiplies f^n b1^(4-n) mu^(2m) (n = 2, 3, 4)
        struct Tables {
            DVector k, Plin;
            DVector P22_dd, P13_dd, P22_dt, P13_dt, P22_tt, P13_tt;
            DVector Pb2d, Pb2t, Pbs2d, Pbs2t, Pb22, Pb2s2, Pbs22, sigma3sq;
            std::array<std::array<DVector, 5>, 5> A, B;
        };

        enum class FoG { None, Gaussian, Lorentzian };

        struct Parameters {
            double f = 0.0;
            double b1 = 1.0, b2 = 0.0, bs2 = 0.0, b3nl = 0.0;
            double sigma_v = 0.0; // Mpc/h
            double N = 0.0;       // constant shot noise (Mpc/h)^3
            FoG fog = FoG::Gaussian;
            bool AB = true; // include the A and B terms
        };

        TNS(const DVector & k_lin, const DVector & P_lin, const DVector & k_out, const Options & options = {})
            : opt(options), P(k_lin, P_lin) {
            for (double k : k_out)
                if (not(k > 0))
                    throw std::invalid_argument("TNS: output k must be positive");
            const double qmin = opt.qmin > 0 ? opt.qmin : P.kmin();
            const double qmax = opt.qmax > 0 ? opt.qmax : P.kmax();
            compute(k_out, qmin, qmax);
        }

        const Tables & tables() const { return T; }
        const DVector & k() const { return T.k; }

        /// P_g,dd, P_g,dt and P_tt (one loop, with the bias)
        std::array<DVector, 3> real_space_spectra(const Parameters & p) const {
            std::array<DVector, 3> out;
            for (size_t i = 0; i < T.k.size(); i++) {
                const double Pdd = T.Plin[i] + T.P22_dd[i] + T.P13_dd[i];
                const double Pdt = T.Plin[i] + T.P22_dt[i] + T.P13_dt[i];
                const double Ptt = T.Plin[i] + T.P22_tt[i] + T.P13_tt[i];
                const double s3 = T.sigma3sq[i] * T.Plin[i];
                out[0].push_back(p.b1 * p.b1 * Pdd + 2 * p.b1 * p.b2 * T.Pb2d[i] + 2 * p.b1 * p.bs2 * T.Pbs2d[i] +
                                 2 * p.b1 * p.b3nl * s3 + p.b2 * p.b2 * T.Pb22[i] + 2 * p.b2 * p.bs2 * T.Pb2s2[i] +
                                 p.bs2 * p.bs2 * T.Pbs22[i]);
                out[1].push_back(p.b1 * Pdt + p.b2 * T.Pb2t[i] + p.bs2 * T.Pbs2t[i] + p.b3nl * s3);
                out[2].push_back(Ptt);
            }
            return out;
        }

        /// A(k, mu) and B(k, mu) for growth rate f and linear bias b1
        std::array<DVector, 2> AB(double mu, double f, double b1) const {
            std::array<DVector, 2> out{DVector(T.k.size(), 0.0), DVector(T.k.size(), 0.0)};
            const double mu2 = mu * mu;
            for (size_t i = 0; i < T.k.size(); i++) {
                double mupow = 1.0;
                for (int m = 0; m < 5; m++, mupow *= mu2)
                    for (int n = 1; n <= 4; n++) {
                        if (n <= 3)
                            out[0][i] += std::pow(f, n) * std::pow(b1, 3 - n) * mupow * T.A[n][m][i];
                        if (n >= 2)
                            out[1][i] += std::pow(f, n) * std::pow(b1, 4 - n) * mupow * T.B[n][m][i];
                    }
            }
            return out;
        }

        /// P_s(k, mu) on the output k
        DVector pkmu(double mu, const Parameters & p) const {
            const auto real = real_space_spectra(p);
            std::array<DVector, 2> ab;
            if (p.AB)
                ab = AB(mu, p.f, p.b1);
            DVector out(T.k.size());
            for (size_t i = 0; i < T.k.size(); i++) {
                const double x = T.k[i] * mu * p.f * p.sigma_v;
                double D = 1.0;
                if (p.fog == FoG::Gaussian)
                    D = std::exp(-x * x);
                else if (p.fog == FoG::Lorentzian)
                    D = 1.0 / (1.0 + x * x);
                const double mu2 = mu * mu;
                double v = real[0][i] + 2 * p.f * mu2 * real[1][i] + p.f * p.f * mu2 * mu2 * real[2][i];
                if (p.AB)
                    v += ab[0][i] + ab[1][i];
                out[i] = D * v + p.N;
            }
            return out;
        }

        /// The multipoles P_0, P_2, P_4 (Gauss-Legendre in mu)
        std::array<DVector, 3> multipoles(const Parameters & p) const {
            const auto gl = LOOPS::detail::gauss_legendre(opt.nmu_multipoles);
            std::array<DVector, 3> out{DVector(T.k.size(), 0.0), DVector(T.k.size(), 0.0), DVector(T.k.size(), 0.0)};
            for (size_t j = 0; j < gl.first.size(); j++) {
                const double mu = gl.first[j], w = gl.second[j], mu2 = mu * mu;
                const double L[3] = {1.0, 0.5 * (3 * mu2 - 1), (35 * mu2 * mu2 - 30 * mu2 + 3) / 8.0};
                const auto pk = pkmu(mu, p);
                for (int l = 0; l < 3; l++)
                    for (size_t i = 0; i < T.k.size(); i++)
                        out[l][i] += (4 * l + 1) / 2.0 * w * L[l] * pk[i];
            }
            return out;
        }

      private:
        Options opt;
        LOOPS::Spectrum P;
        Tables T;

        /// A polynomial in f (b1 = 1; the b1 powers follow from homogeneity)
        using Poly = std::array<double, 5>;

        /// The tree-level B_sigma(k1, k2, k3) = <theta(k1) X(k2) X(k3)>', X = delta + f mu^2 theta, in powers of f
        static Poly bsigma(const SPT::Vec3 & k1,
                           const SPT::Vec3 & k2,
                           const SPT::Vec3 & k3,
                           double P1,
                           double P2,
                           double P3,
                           const SPT::Vec3 & los) {
            auto mu2 = [&](const SPT::Vec3 & v) {
                const double z = v.dot(los);
                return z * z / v.norm2();
            };
            const double m2 = mu2(k2), m3 = mu2(k3);
            Poly b{};
            // 2 G2(k2, k3) L(k2) L(k3) P2 P3, L = 1 + f mu^2
            const double g23 = 2.0 * SPT::G2(k2, k3) * P2 * P3;
            b[0] += g23;
            b[1] += g23 * (m2 + m3);
            b[2] += g23 * m2 * m3;
            // 2 [F2(k1, k3) + f mu2^2 G2(k1, k3)] L(k3) P1 P3
            const double f13 = 2.0 * SPT::F2(k1, k3) * P1 * P3, g13 = 2.0 * SPT::G2(k1, k3) * P1 * P3;
            b[0] += f13;
            b[1] += f13 * m3 + g13 * m2;
            b[2] += g13 * m2 * m3;
            // 2 [F2(k1, k2) + f mu3^2 G2(k1, k2)] L(k2) P1 P2
            const double f12 = 2.0 * SPT::F2(k1, k2) * P1 * P2, g12 = 2.0 * SPT::G2(k1, k2) * P1 * P2;
            b[0] += f12;
            b[1] += f12 * m2 + g12 * m3;
            b[2] += g12 * m2 * m3;
            return b;
        }

        void compute(const DVector & kout, double qmin, double qmax) {
            T = Tables{};
            T.k = kout;
            const size_t nk = kout.size();
            for (double k : kout)
                T.Plin.push_back(P(k));
            const std::vector<const LOOPS::Spectrum *> spectra{&P};
            auto S2 = [](const SPT::Vec3 & a, const SPT::Vec3 & b) {
                const double ab = a.dot(b);
                return ab * ab / (a.norm2() * b.norm2()) - 1.0 / 3.0;
            };

            // Isotropic P22-type terms: 2 F2^2, 2 F2 G2, 2 G2^2 and the bias terms
            enum { DD, DT, TT, B2D, B2T, BS2D, BS2T, B22, B2S2, BS22, NISO };
            const auto iso = LOOPS::integrate_22(
                kout, {0.0}, spectra, qmin, qmax, NISO, opt.quadrature, [&](const LOOPS::Point & pt, double * v) {
                    const double PP = pt.Pq[0] * pt.Pp[0];
                    if (PP == 0.0)
                        return;
                    const double F = SPT::F2(pt.q, pt.p), G = SPT::G2(pt.q, pt.p), S = S2(pt.q, pt.p);
                    v[DD] = 2 * F * F * PP;
                    v[DT] = 2 * F * G * PP;
                    v[TT] = 2 * G * G * PP;
                    v[B2D] = F * PP;
                    v[B2T] = G * PP;
                    v[BS2D] = F * S * PP;
                    v[BS2T] = G * S * PP;
                    v[B22] = 0.5 * PP;
                    v[B2S2] = 0.5 * S * PP;
                    v[BS22] = 0.5 * S * S * PP;
                });

            // The k -> 0 constants int d^3q/(2pi)^3 P(q)^2 of the subtracted bias terms
            double sigma4 = 0.0;
            {
                const auto gl = LOOPS::detail::gauss_legendre(opt.quadrature.order);
                DVector q, w;
                LOOPS::detail::log_nodes({qmin, qmax}, opt.quadrature, gl, q, w);
                for (size_t i = 0; i < q.size(); i++)
                    sigma4 += w[i] * q[i] * q[i] * P(q[i]) * P(q[i]) / (2 * LOOPS::pi * LOOPS::pi);
            }

            // P13-type: 6 F3, 3 (F3 + G3), 6 G3 and the sigma3^2 kernel (symmetrized in q -> -q)
            const double kmin_out = *std::min_element(kout.begin(), kout.end());
            const double kmax_out = *std::max_element(kout.begin(), kout.end());
            const LOOPS::Angular13 ang(
                std::min(0.5, qmin / kmax_out),
                std::max(2.0, qmax / kmin_out),
                {0.0},
                4,
                opt.quadrature,
                [&](const SPT::Vec3 & k, const SPT::Vec3 & q, const SPT::Vec3 &, double * v) {
                    const SPT::Vec3 a[3] = {k, q, -q};
                    double F, G;
                    SPT::FG(a, 3, F, G);
                    v[0] = 6 * F;
                    v[1] = 3 * (F + G);
                    v[2] = 6 * G;
                    // sigma3^2: 105/16 [D2(-q, k) S2(q, k - q) + 8/63], D2 = 2/7 (S2 - 2/3)
                    auto term = [&](const SPT::Vec3 & qq) {
                        const double D2 = 2.0 / 7.0 * (S2(-qq, k) - 2.0 / 3.0);
                        return 105.0 / 16.0 * (D2 * S2(qq, k - qq) + 8.0 / 63.0);
                    };
                    v[3] = 0.5 * (term(q) + term(-q));
                });

            // A and B at mu nodes, then as polynomials in mu^2
            const LOOPS::MuPolynomial mupoly(5);
            constexpr int NAB = 3 + 3; // A: f^1..f^3, B: f^2..f^4
            const auto ab = LOOPS::integrate_22(
                kout, mupoly.mu, spectra, qmin, qmax, NAB, opt.quadrature, [&](const LOOPS::Point & pt, double * v) {
                    const SPT::Vec3 & n = pt.los;
                    const double kz = pt.k.dot(n);
                    const double Pq = pt.Pq[0], Pp = pt.Pp[0], Pk = pt.Pk[0];
                    // A = k_z f int (q_z/q^2) [B_sigma(q, p, -k) + B_sigma(-q, k, -p)]
                    const Poly b1 = bsigma(pt.q, pt.p, -pt.k, Pq, Pp, Pk, n);
                    const Poly b2 = bsigma(-pt.q, pt.k, -pt.p, Pq, Pk, Pp, n);
                    const double a = kz * pt.q.dot(n) / (pt.qmag * pt.qmag);
                    for (int j = 0; j < 3; j++)
                        v[j] = a * (b1[j] + b2[j]);
                    // B = k_z^2 f^2 int (mu_q/q)(mu_p/p)(1 + f mu_q^2)(1 + f mu_p^2) P(q) P(p)
                    const double qz = pt.q.dot(n), pz = pt.p.dot(n);
                    const double mq2 = qz * qz / (pt.qmag * pt.qmag), mp2 = pz * pz / (pt.pmag * pt.pmag);
                    const double base = kz * kz * qz / (pt.qmag * pt.qmag) * pz / (pt.pmag * pt.pmag) * Pq * Pp;
                    v[3] = base;
                    v[4] = base * (mq2 + mp2);
                    v[5] = base * mq2 * mp2;
                });

            for (auto & a : T.A)
                for (auto & c : a)
                    c.assign(nk, 0.0);
            for (auto & b : T.B)
                for (auto & c : b)
                    c.assign(nk, 0.0);
            for (size_t i = 0; i < nk; i++) {
                const double k = kout[i], Pk = T.Plin[i];
                T.P22_dd.push_back(iso[i][0][DD]);
                T.P22_dt.push_back(iso[i][0][DT]);
                T.P22_tt.push_back(iso[i][0][TT]);
                T.Pb2d.push_back(iso[i][0][B2D]);
                T.Pb2t.push_back(iso[i][0][B2T]);
                T.Pbs2d.push_back(iso[i][0][BS2D]);
                T.Pbs2t.push_back(iso[i][0][BS2T]);
                T.Pb22.push_back(iso[i][0][B22] - 0.5 * sigma4);
                T.Pb2s2.push_back(iso[i][0][B2S2] - 1.0 / 3.0 * sigma4);
                T.Pbs22.push_back(iso[i][0][BS22] - 2.0 / 9.0 * sigma4);
                const auto j13 = ang.integrate(k, P);
                T.P13_dd.push_back(j13[0][0] * Pk);
                T.P13_dt.push_back(j13[0][1] * Pk);
                T.P13_tt.push_back(j13[0][2] * Pk);
                T.sigma3sq.push_back(j13[0][3]);
                for (int c = 0; c < NAB; c++) {
                    DVector vals;
                    for (size_t m = 0; m < mupoly.mu.size(); m++)
                        vals.push_back(ab[i][m][c]);
                    const auto coef = mupoly.coefficients(vals);
                    for (int m = 0; m < 5; m++) {
                        if (c < 3)
                            T.A[c + 1][m][i] = coef[m];
                        else
                            T.B[c - 1][m][i] = coef[m];
                    }
                }
            }
        }
    };

} // namespace FML::PERTURBATION
#endif
