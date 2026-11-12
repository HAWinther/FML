#ifndef FML_PERTURBATION_EFTOFLSS_H
#define FML_PERTURBATION_EFTOFLSS_H

#include <FML/PerturbationTheory/LoopIntegrals.h>
#include <FML/PerturbationTheory/NoWiggle.h>
#include <FML/PerturbationTheory/SPTKernels.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

//=====================================================================================================================
///
/// The one-loop EFTofLSS power spectrum multipoles of biased tracers in redshift space in the form of CLASS-PT
/// (Ivanov, Simonovic & Zaldarriaga 2020, arXiv:1909.05277; Chudaykin, Ivanov, Philcox & Simonovic 2020,
/// arXiv:2004.10607; github.com/Michalychforever/CLASS-PT), the same parameters and conventions:
///
///   P(k, mu) = Z1^2 P_tree,IR + P_22 + P_13 (one-loop SPT with the redshift-space kernels Z1, Z2, Z3 and the bias
///   delta_g = b1 delta + b2/2 delta^2 + bG2 G2 + bGamma3 Gamma3), IR resummed, and per multipole
///     P_0 += -2 c0 [k^2 P_IR]_0 + Pshot + (a0 + a2/3) (k/0.45)^2
///     P_2 += -2 c2 [f mu^2 k^2 P_IR]_2 + 2/3 a2 (k/0.45)^2
///     P_4 += -2 c4 [f^2 mu^4 k^2 P_IR]_4
///   and the next-to-leading FoG counterterm b4 (CLASS-PT's ctilde): P_l += -(35/8) b4 f^2 k^2 C_l(f, b1) [f^2 mu^4 k^2
///   P_IR]_4 with C_0 = f^2/9 + 2 f b1/7 + b1^2/5, C_2 = 4 (70 f^2 + 165 f b1 + 99 b1^2)/693, C_4 = 8 (210 f^2 + 390 f b1
///   + 143 b1^2)/5005 ([x]_l: the Legendre multipole of x; c_l in (Mpc/h)^2, b4 in (Mpc/h)^4, k in h/Mpc).
///
/// IR resummation (Senatore & Zaldarriaga 2015, Ivanov & Sibiryakov 2018) as in CLASS-PT: P = P_nw + P_w split,
/// Sigma^2 and delta Sigma^2 from P_nw below k_S = 0.2 h/Mpc with the BAO scale r_bao, Sigma_tot^2(mu) = (1 + f mu^2
/// (2 + f)) Sigma^2 + f^2 mu^2 (mu^2 - 1) delta Sigma^2, e(mu) = exp(-k^2 Sigma_tot^2), P_bin = P_nw + exp(-k^2
/// Sigma^2) P_w:
///  - tree: Z1^2 [P_nw + (1 + k^2 Sigma_tot^2) e P_w]
///  - matter loops (the b1^0, b1, b1^2 terms): L[P_nw] + e L_w, L_w the part linear in P_w; the P13 outer P_lin is
///    P_nw + e P_w, and the UV part of the wiggle P13 (-k^2 sigma_v^2[P_w] P) is not damped
///  - bias loops (b2, bG2, bGamma3): P22 with P_bin; P13 with the inner P_bin and the outer P_nw + e P_w, kept (as in
///    CLASS-PT) only in the multipoles of its undamped mu dependence (b1 x: l = 0; f mu^2 x: l = 0, 2)
///  - counterterms with P_nw + e P_w
/// The k -> 0 constant of the b2^2 term is subtracted (a shot noise; CLASS-PT's b2^2 monopole differs from this by
/// a constant that depends on the k grid given to its python wrapper).
///
/// The loops are direct quadratures of the kernels of SPTKernels.h (Z3 is derived from the redshift-space mapping,
/// not typed in) with LoopIntegrals.h; P13 drops the b2 and b1 b2 terms (proportional to P_lin: they renormalize b1,
/// as in CLASS-PT). k in the counterterm and stochastic terms is in h/Mpc (CLASS-PT's python wrapper uses the k given
/// to initialize_output, usually 1/Mpc, for b4, a0, a2: its b4 is h^2 times this one). The bias independent part is computed once (for fixed f); multipoles for any bias and
/// counterterms are a cheap linear combination of the 12 bias monomial tables (multipole_table).
///
/// No-wiggle split: CLASS-PT's DST method (nowiggle_classpt; needs h), the DST method of NoWiggle.h (velocileptors),
/// or a given P_nw. r_bao: CLASS-PT uses the sound horizon at the drag epoch r_d (r_bao = h r_d in Mpc/h).
///
/// Units: k [h/Mpc], P [(Mpc/h)^3]. P_lin must cover a wide range (e.g. 1e-5 - 50 h/Mpc): the loops are integrals over
/// its table. Validation (tests/test_eftoflss.cpp): CLASS-PT's pk_gg_l0/2/4 for each bias monomial and counterterm.
///
//=====================================================================================================================

namespace FML::PERTURBATION {

    struct EFTofLSSOptions {
        LOOPS::QuadratureOptions quadrature{};
        double qmin = 0.0, qmax = 0.0; // the loop integration range (0: the range of the P_lin table)
        int nmu = 40;                  // Gauss-Legendre points in mu for the multipoles (CLASS-PT: 40)
        bool ir_resummation = true;
        enum class NoWiggleMethod { ClassPT, DST, Given } nowiggle = NoWiggleMethod::ClassPT;
        double h = 0.0;          // needed by the ClassPT split (it works in 1/Mpc)
        std::vector<double> Pnw; // NoWiggleMethod::Given: P_nw on the P_lin k
        double r_bao = 0.0;      // the BAO scale in Mpc/h (h r_d); must be set when ir_resummation
        double k_S = 0.2;        // the IR scale of Sigma^2 (h/Mpc)
    };

    class EFTofLSS {
      public:
        using DVector = std::vector<double>;
        using Options = EFTofLSSOptions;

        /// The bias monomials (columns of the tables)
        enum Monomial { ONE, B1, B1B1, B2, B1B2, B2B2, BG2, B1BG2, B2BG2, BG2BG2, BGAMMA3, B1BGAMMA3, NMONOMIAL };
        static const char * monomial_name(int m) {
            static const char * names[NMONOMIAL] = {
                "1", "b1", "b1^2", "b2", "b1 b2", "b2^2", "bG2", "b1 bG2", "b2 bG2", "bG2^2", "bGamma3", "b1 bGamma3"};
            return names[m];
        }
        /// The counterterm templates T_l (the columns after the monomials): [k^2 P_IR]_0, [f mu^2 k^2 P_IR]_2,
        /// [f^2 mu^4 k^2 P_IR]_4 (each only in its own multipole)
        enum Counterterm { CT0 = NMONOMIAL, CT2, CT4, NCOLUMN };

        struct Bias {
            double b1 = 1.0, b2 = 0.0, bG2 = 0.0, bGamma3 = 0.0;
            double c0 = 0.0, c2 = 0.0, c4 = 0.0; // (Mpc/h)^2
            double b4 = 0.0;                     // (Mpc/h)^4
            double Pshot = 0.0, a0 = 0.0, a2 = 0.0; // (Mpc/h)^3
        };

        EFTofLSS(const DVector & k_lin, const DVector & P_lin, const DVector & k_out, double f_, const Options & options = {})
            : opt(options), f(f_), kout(k_out) {
            for (double k : kout)
                if (not(k > 0))
                    throw std::invalid_argument("EFTofLSS: output k must be positive");
            const LOOPS::Spectrum P(k_lin, P_lin);
            const double qmin = opt.qmin > 0 ? opt.qmin : P.kmin();
            const double qmax = opt.qmax > 0 ? opt.qmax : P.kmax();

            // The wiggle/no-wiggle split on the P_lin grid
            DVector Pnw_lin;
            if (not opt.ir_resummation)
                Pnw_lin = P_lin;
            else if (opt.nowiggle == Options::NoWiggleMethod::Given) {
                if (opt.Pnw.size() != P_lin.size())
                    throw std::invalid_argument("EFTofLSS: the given P_nw must be on the P_lin k");
                Pnw_lin = opt.Pnw;
            } else if (opt.nowiggle == Options::NoWiggleMethod::ClassPT) {
                if (not(opt.h > 0))
                    throw std::invalid_argument("EFTofLSS: the ClassPT no-wiggle split needs h");
                Pnw_lin = nowiggle_classpt(k_lin, P_lin, opt.h, k_lin);
            } else {
                NoWiggleDSTOptions o;
                o.kmin = std::max(1e-3, k_lin.front());
                o.kmax = std::min(10.0, k_lin.back());
                Pnw_lin = nowiggle_dst(k_lin, P_lin, o);
            }
            DVector Pw_lin(P_lin.size());
            for (size_t i = 0; i < P_lin.size(); i++)
                Pw_lin[i] = P_lin[i] - Pnw_lin[i];
            const LOOPS::Spectrum Pnw(k_lin, Pnw_lin);
            const LOOPS::Spectrum Pw = LOOPS::Spectrum::signed_spectrum(k_lin, Pw_lin);

            if (opt.ir_resummation) {
                if (not(opt.r_bao > 0))
                    throw std::invalid_argument("EFTofLSS: set the BAO scale r_bao (Mpc/h) for the IR resummation");
                compute_sigma(Pnw);
            }
            // P_bin = P_nw + exp(-k^2 Sigma^2) P_w: the input of the bias loops (CLASS-PT)
            DVector Pbin_lin(P_lin.size());
            for (size_t i = 0; i < P_lin.size(); i++)
                Pbin_lin[i] = Pnw_lin[i] + std::exp(-k_lin[i] * k_lin[i] * sigma2) * Pw_lin[i];
            const LOOPS::Spectrum Pbin(k_lin, Pbin_lin);
            compute_loops(Pnw, Pw, Pbin, qmin, qmax);
            compute_multipoles();
        }

        const DVector & k() const { return kout; }
        double growth_rate() const { return f; }
        double Sigma2() const { return sigma2; }
        double deltaSigma2() const { return dsigma2; }
        const DVector & Pnw() const { return pnw; }
        const DVector & Pw() const { return pw; }

        /// table[l/2][column][ik]: the multipole l = 0, 2, 4 of each bias monomial (columns 0..11) and the
        /// counterterm templates (CT0, CT2, CT4)
        const std::array<std::vector<DVector>, 3> & multipole_table() const { return table; }

        /// P_0, P_2, P_4 for the bias, counterterm and stochastic parameters
        std::array<DVector, 3> multipoles(const Bias & b) const {
            const double coef[NMONOMIAL] = {1.0,
                                            b.b1,
                                            b.b1 * b.b1,
                                            b.b2,
                                            b.b1 * b.b2,
                                            b.b2 * b.b2,
                                            b.bG2,
                                            b.b1 * b.bG2,
                                            b.b2 * b.bG2,
                                            b.bG2 * b.bG2,
                                            b.bGamma3,
                                            b.b1 * b.bGamma3};
            const double C[3] = {f * f / 9.0 + 2.0 * f * b.b1 / 7.0 + b.b1 * b.b1 / 5.0,
                                 4.0 * (70.0 * f * f + 165.0 * f * b.b1 + 99.0 * b.b1 * b.b1) / 693.0,
                                 8.0 * (210.0 * f * f + 390.0 * f * b.b1 + 143.0 * b.b1 * b.b1) / 5005.0};
            std::array<DVector, 3> out;
            for (int l = 0; l < 3; l++) {
                out[l].assign(kout.size(), 0.0);
                for (size_t i = 0; i < kout.size(); i++) {
                    double v = 0.0;
                    for (int m = 0; m < NMONOMIAL; m++)
                        v += coef[m] * table[l][m][i];
                    const double k2 = kout[i] * kout[i], x = k2 / (0.45 * 0.45);
                    const double c = l == 0 ? b.c0 : (l == 1 ? b.c2 : b.c4);
                    v += -2.0 * c * table[l][CT0 + l][i];
                    v += -35.0 / 8.0 * b.b4 * f * f * k2 * C[l] * table[2][CT4][i];
                    if (l == 0)
                        v += b.Pshot + (b.a0 + b.a2 / 3.0) * x;
                    if (l == 1)
                        v += 2.0 / 3.0 * b.a2 * x;
                    out[l][i] = v;
                }
            }
            return out;
        }

      private:
        Options opt;
        double f;
        DVector kout, pnw, pw;
        double sigma2{0.0}, dsigma2{0.0};
        // loop tables: [ik][monomial][mu^(2n), n = 0..4] for the no-wiggle and the wiggle parts
        std::vector<std::array<std::array<double, 5>, NMONOMIAL>> L_nw, L_w13, L_w22, L13_nw, L_bin, L13_bin;
        std::array<std::vector<DVector>, 3> table;

        void compute_sigma(const LOOPS::Spectrum & Pnw) {
            // Sigma^2 = 1/(6 pi^2) int_0^kS dq P_nw (1 - j0(q r) + 2 j2(q r)), delta Sigma^2 = 1/(2 pi^2) int P_nw j2
            const auto gl = LOOPS::detail::gauss_legendre(16);
            DVector q, w;
            LOOPS::QuadratureOptions o;
            o.log_step = 0.1;
            LOOPS::detail::log_nodes({std::max(Pnw.kmin(), 1e-6), opt.k_S}, o, gl, q, w);
            sigma2 = dsigma2 = 0.0;
            for (size_t i = 0; i < q.size(); i++) {
                const double x = q[i] * opt.r_bao, s = std::sin(x), c = std::cos(x);
                const double j0 = s / x;
                const double j2 = (3.0 / (x * x) - 1.0) * s / x - 3.0 * c / (x * x);
                const double Pq = Pnw(q[i]);
                sigma2 += w[i] * Pq * (1.0 - j0 + 2.0 * j2) / (6.0 * LOOPS::pi * LOOPS::pi);
                dsigma2 += w[i] * Pq * j2 / (2.0 * LOOPS::pi * LOOPS::pi);
            }
        }

        static int pair_monomial(int i, int j) {
            // Z2 components {1, b1, b2, bG2} (SPT::BiasIndex 0..3); the monomial of z_i z_j
            static const int map[4][4] = {
                {ONE, B1, B2, BG2}, {B1, B1B1, B1B2, B1BG2}, {B2, B1B2, B2B2, B2BG2}, {BG2, B1BG2, B2BG2, BG2BG2}};
            return map[i][j];
        }

        void compute_loops(const LOOPS::Spectrum & Pnw,
                           const LOOPS::Spectrum & Pw,
                           const LOOPS::Spectrum & Pbin,
                           double qmin,
                           double qmax) {
            const size_t nk = kout.size();
            for (double k : kout) {
                pnw.push_back(Pnw(k));
                pw.push_back(Pw(k));
            }
            const LOOPS::MuPolynomial mupoly(5);
            const size_t nmu = mupoly.mu.size();
            const double ff = f;

            // P22 = 2 int Z2^2 P(q) P(p): no-wiggle (P_nw P_nw) and wiggle (P_nw P_w + P_w P_nw) parts
            const std::vector<const LOOPS::Spectrum *> spectra{&Pnw, &Pw, &Pbin};
            const auto r22 = LOOPS::integrate_22(
                kout,
                mupoly.mu,
                spectra,
                qmin,
                qmax,
                3 * NMONOMIAL,
                opt.quadrature,
                [ff](const LOOPS::Point & pt, double * v) {
                    const double PPnw = pt.Pq[0] * pt.Pp[0];
                    const double PPw = pt.Pq[0] * pt.Pp[1] + pt.Pq[1] * pt.Pp[0];
                    const double PPbin = pt.Pq[2] * pt.Pp[2];
                    if (PPnw == 0.0 and PPw == 0.0)
                        return;
                    const auto Z = SPT::Z2(pt.q, pt.p, ff, pt.los);
                    for (int i = 0; i < 4; i++)
                        for (int j = i; j < 4; j++) {
                            const double zz = (i == j ? 2.0 : 4.0) * Z[i] * Z[j];
                            const int m = pair_monomial(i, j);
                            v[m] += zz * PPnw;
                            v[NMONOMIAL + m] += zz * PPw;
                            v[2 * NMONOMIAL + m] += zz * PPbin;
                        }
                });
            // The k -> 0 constant of the b2^2 term: 2 (1/2)^2 int P(q)^2 (no-wiggle and wiggle parts)
            double b2b2_nw = 0.0, b2b2_w = 0.0, b2b2_bin = 0.0;
            {
                const auto gl = LOOPS::detail::gauss_legendre(opt.quadrature.order);
                DVector q, w;
                LOOPS::detail::log_nodes({qmin, qmax}, opt.quadrature, gl, q, w);
                for (size_t i = 0; i < q.size(); i++) {
                    const double a = Pnw(q[i]), b = Pw(q[i]), wq = w[i] * q[i] * q[i] / (2.0 * LOOPS::pi * LOOPS::pi);
                    b2b2_nw += 0.5 * wq * a * a;
                    b2b2_w += 0.5 * wq * 2.0 * a * b;
                    b2b2_bin += 0.5 * wq * Pbin(q[i]) * Pbin(q[i]);
                }
            }

            // P13 = 6 Z1(k) P(k) int Z3(k, q, -q) P(q) with Z1 = f mu^2 + b1 (the b2 and b1 b2 parts are dropped)
            const double kmin_out = *std::min_element(kout.begin(), kout.end());
            const double kmax_out = *std::max_element(kout.begin(), kout.end());
            auto kernel13 = [ff](const SPT::Vec3 & k, const SPT::Vec3 & q, const SPT::Vec3 & los, double * v) {
                const SPT::Vec3 a[3] = {k, q, -q};
                const auto Z3 = SPT::Zn(a, 3, ff, los);
                const double mu = los.z, z1 = ff * mu * mu;
                // Z1 = z1 * 1 + 1 * b1
                v[ONE] += 6 * z1 * Z3[SPT::B_ONE];
                v[B1] += 6 * (z1 * Z3[SPT::B_B1] + Z3[SPT::B_ONE]);
                v[B1B1] += 6 * Z3[SPT::B_B1];
                v[BG2] += 6 * z1 * Z3[SPT::B_BG2];
                v[B1BG2] += 6 * Z3[SPT::B_BG2];
                v[BGAMMA3] += 6 * z1 * Z3[SPT::B_BGAMMA3];
                v[B1BGAMMA3] += 6 * Z3[SPT::B_BGAMMA3];
            };
            const LOOPS::Angular13 ang(std::min(0.5, qmin / kmax_out),
                                       std::max(2.0, qmax / kmin_out),
                                       mupoly.mu,
                                       NMONOMIAL,
                                       opt.quadrature,
                                       kernel13);

            // The UV limit of the matter P13 kernels, J -> A / r^2 (P13 -> -k^2 sigma_v^2 P(k) x mu-polynomial), and
            // int dq P_w: CLASS-PT damps only the rest of the wiggle P13 (its UV term uses sigma_v of the full P_lin)
            std::vector<DVector> A_uv(nmu, DVector(NMONOMIAL, 0.0));
            double int_Pw = 0.0;
            if (opt.ir_resummation) {
                const double rbig = 1e4;
                const auto J = LOOPS::Angular13::angular(
                    rbig, mupoly.mu, NMONOMIAL, opt.quadrature, LOOPS::detail::gauss_legendre(opt.quadrature.order),
                    kernel13);
                for (size_t im = 0; im < nmu; im++)
                    for (int m = 0; m < NMONOMIAL; m++)
                        A_uv[im][m] = rbig * rbig * J[im][m];
                const auto gl = LOOPS::detail::gauss_legendre(opt.quadrature.order);
                DVector q, w;
                LOOPS::detail::log_nodes({qmin, qmax}, opt.quadrature, gl, q, w);
                for (size_t i = 0; i < q.size(); i++)
                    int_Pw += w[i] * Pw(q[i]);
            }

            L_nw.assign(nk, {});
            L_w22.assign(nk, {});
            L_w13.assign(nk, {});
            L13_nw.assign(nk, {});
            L_bin.assign(nk, {});
            L13_bin.assign(nk, {});
            for (size_t i = 0; i < nk; i++) {
                const auto j_nw = ang.integrate(kout[i], Pnw);
                const auto j_w = ang.integrate(kout[i], Pw);
                const auto j_bin = ang.integrate(kout[i], Pbin);
                for (int m = 0; m < NMONOMIAL; m++) {
                    DVector v22nw(nmu), v22w(nmu), v13nw(nmu), v13w(nmu), vbin(nmu), v13bin(nmu);
                    for (size_t im = 0; im < nmu; im++) {
                        vbin[im] = r22[i][im][2 * NMONOMIAL + m];
                        v13bin[im] = j_bin[im][m];
                        v22nw[im] = r22[i][im][m];
                        v22w[im] = r22[i][im][NMONOMIAL + m];
                        v13nw[im] = j_nw[im][m] * pnw[i];
                        v13w[im] = j_w[im][m] * pnw[i];
                        if (m == ONE or m == B1 or m == B1B1) {
                            // the UV (k^2 sigma_v^2) part of the wiggle P13 is not damped (as in CLASS-PT)
                            const double uv =
                                pnw[i] * kout[i] * kout[i] / std::pow(2.0 * LOOPS::pi, 3) * A_uv[im][m] * int_Pw;
                            v13w[im] -= uv;
                            v13nw[im] += uv;
                        }
                    }
                    if (m == B2B2) {
                        for (size_t im = 0; im < nmu; im++) {
                            v22nw[im] -= b2b2_nw;
                            v22w[im] -= b2b2_w;
                            vbin[im] -= b2b2_bin;
                        }
                    }
                    const auto c22nw = mupoly.coefficients(v22nw), c22w = mupoly.coefficients(v22w);
                    const auto c13nw = mupoly.coefficients(v13nw), c13w = mupoly.coefficients(v13w);
                    const auto cbin = mupoly.coefficients(vbin), c13bin = mupoly.coefficients(v13bin);
                    for (int n = 0; n < 5; n++) {
                        L_nw[i][m][n] = c22nw[n];
                        L13_nw[i][m][n] = c13nw[n];
                        L_w22[i][m][n] = c22w[n];
                        L_w13[i][m][n] = c13w[n];
                        L_bin[i][m][n] = cbin[n];
                        L13_bin[i][m][n] = c13bin[n];
                    }
                }
            }
        }

        void compute_multipoles() {
            const size_t nk = kout.size();
            const auto gl = LOOPS::detail::gauss_legendre(opt.nmu);
            for (auto & t : table)
                t.assign(NCOLUMN, DVector(nk, 0.0));
            for (size_t j = 0; j < gl.first.size(); j++) {
                const double mu = gl.first[j], w = gl.second[j], mu2 = mu * mu;
                const double L[3] = {1.0, 0.5 * (3 * mu2 - 1), (35 * mu2 * mu2 - 30 * mu2 + 3) / 8.0};
                const double sigtot = sigma2 * (1.0 + f * mu2 * (2.0 + f)) + f * f * mu2 * (mu2 - 1.0) * dsigma2;
                for (size_t i = 0; i < nk; i++) {
                    const double k2 = kout[i] * kout[i];
                    const double e = std::exp(-k2 * sigtot);
                    const double Ptree = pnw[i] + (1.0 + k2 * sigtot) * e * pw[i];
                    const double Pbin = pnw[i] + e * pw[i];
                    double val[NCOLUMN] = {};
                    // tree: (f mu^2 + b1)^2 P_tree
                    val[ONE] += f * f * mu2 * mu2 * Ptree;
                    val[B1] += 2.0 * f * mu2 * Ptree;
                    val[B1B1] += Ptree;
                    const double ratio13 = pnw[i] != 0.0 ? 1.0 + e * pw[i] / pnw[i] : 1.0;
                    double val13[NMONOMIAL] = {};
                    for (int m = 0; m < NMONOMIAL; m++) {
                        double mupow = 1.0, s = 0.0, s13 = 0.0;
                        if (m == ONE or m == B1 or m == B1B1) {
                            for (int n = 0; n < 5; n++, mupow *= mu2)
                                s += mupow * (L_nw[i][m][n] + L13_nw[i][m][n] * ratio13 +
                                              e * (L_w22[i][m][n] + L_w13[i][m][n]));
                        } else {
                            // P22 with P_bin; P13 with the inner P_bin and the outer P_nw + e P_w
                            for (int n = 0; n < 5; n++, mupow *= mu2) {
                                s += mupow * L_bin[i][m][n];
                                s13 += mupow * L13_bin[i][m][n] * Pbin;
                            }
                        }
                        val[m] += s;
                        val13[m] = s13;
                    }
                    val[CT0] = k2 * Pbin;
                    val[CT2] = f * mu2 * k2 * Pbin;
                    val[CT4] = f * f * mu2 * mu2 * k2 * Pbin;
                    for (int l = 0; l < 3; l++) {
                        const double wl = (4 * l + 1) / 2.0 * w * L[l];
                        for (int c = 0; c < NMONOMIAL; c++)
                            table[l][c][i] += wl * val[c];
                        // The bias P13 terms are b1 x isotropic and f mu^2 x isotropic: CLASS-PT keeps them only in
                        // the multipoles they have without the mu-dependent IR damping (l = 0, and l = 0, 2)
                        for (int c : {B1BG2, B1BGAMMA3})
                            if (l == 0)
                                table[l][c][i] += wl * val13[c];
                        for (int c : {BG2, BGAMMA3})
                            if (l <= 1)
                                table[l][c][i] += wl * val13[c];
                        // each counterterm only in its own multipole
                        table[l][CT0 + l][i] += wl * val[CT0 + l];
                    }
                }
            }
        }
    };

} // namespace FML::PERTURBATION
#endif
