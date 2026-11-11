#ifndef FML_PERTURBATIONTHEORY_LAGRANGIANQFUNCTIONS_H
#define FML_PERTURBATIONTHEORY_LAGRANGIANQFUNCTIONS_H

#include <FML/FFTLog/SphericalBesselTransform.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace FML::PERTURBATION {

    //=================================================================================================================
    // Small numerical helpers (shared with LPTRedshiftSpace.h)
    //=================================================================================================================
    namespace detail {
        using DVector = std::vector<double>;

        /// Linear interpolation with the end values outside the table (numpy.interp)
        inline double interp_clamped(double x, const DVector & xs, const DVector & ys) {
            if (x <= xs.front())
                return ys.front();
            if (x >= xs.back())
                return ys.back();
            const size_t i = size_t(std::upper_bound(xs.begin(), xs.end(), x) - xs.begin());
            const double t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + t * (ys[i] - ys[i - 1]);
        }
        inline DVector interp_clamped(const DVector & x, const DVector & xs, const DVector & ys) {
            DVector out(x.size());
            for (size_t i = 0; i < x.size(); i++)
                out[i] = interp_clamped(x[i], xs, ys);
            return out;
        }

        /// A natural cubic spline y(x) (second derivative zero at the ends)
        class NaturalCubicSpline {
          public:
            NaturalCubicSpline() = default;
            NaturalCubicSpline(DVector x_, DVector y_) : x(std::move(x_)), y(std::move(y_)) {
                const size_t n = x.size();
                if (n < 3 or y.size() != n)
                    throw std::invalid_argument("[NaturalCubicSpline] need >= 3 points");
                M.assign(n, 0.0);
                DVector c(n, 0.0), d(n, 0.0);
                for (size_t i = 1; i + 1 < n; i++) {
                    const double h0 = x[i] - x[i - 1], h1 = x[i + 1] - x[i];
                    const double a = h0 / 6.0, b = (h0 + h1) / 3.0, cc = h1 / 6.0;
                    const double r = (y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0;
                    const double m = b - a * c[i - 1];
                    c[i] = cc / m;
                    d[i] = (r - a * d[i - 1]) / m;
                }
                for (size_t i = n - 2; i >= 1; i--)
                    M[i] = d[i] - c[i] * M[i + 1];
            }
            double operator()(double xx) const {
                const size_t i = segment(xx);
                const double h = x[i + 1] - x[i], A = (x[i + 1] - xx) / h, B = (xx - x[i]) / h;
                return A * y[i] + B * y[i + 1] + ((A * A * A - A) * M[i] + (B * B * B - B) * M[i + 1]) * h * h / 6.0;
            }
            double derivative(double xx) const {
                const size_t i = segment(xx);
                const double h = x[i + 1] - x[i], A = (x[i + 1] - xx) / h, B = (xx - x[i]) / h;
                return (y[i + 1] - y[i]) / h + (-(3 * A * A - 1) * M[i] + (3 * B * B - 1) * M[i + 1]) * h / 6.0;
            }

          private:
            DVector x, y, M;
            size_t segment(double xx) const {
                size_t i = size_t(std::upper_bound(x.begin(), x.end(), xx) - x.begin());
                return std::clamp<size_t>(i, 1, x.size() - 1) - 1;
            }
        };

        /// Extrapolation of y(x) as power laws beyond the first and last two points of the same sign, a natural cubic
        /// spline in ln x between (velocileptors Utils/loginterp.py, from Chirag Modi's CLEFT code)
        class LogInterp {
          public:
            LogInterp() = default;
            LogInterp(const DVector & x, const DVector & y) {
                const size_t n = x.size();
                if (n < 4 or y.size() != n)
                    throw std::invalid_argument("[LogInterp] need >= 4 points");
                auto sgn = [](double v) { return (v > 0) - (v < 0); };
                size_t lp = 0;
                while (sgn(y[lp]) != sgn(y[lp + 1])) {
                    if (lp < n - 2)
                        lp++;
                    else
                        throw std::runtime_error("[LogInterp] the input points never stop changing sign");
                }
                size_t rp = n - 1;
                while (sgn(y[rp]) != sgn(y[rp - 1])) {
                    if (rp > 2)
                        rp--;
                    else
                        throw std::runtime_error("[LogInterp] the input points never stop changing sign");
                }
                DVector lx, yy;
                for (size_t i = lp; i <= rp; i++) {
                    lx.push_back(std::log(x[i]));
                    yy.push_back(y[i]);
                }
                spline = NaturalCubicSpline(lx, yy);
                xl = x[lp];
                xr = x[rp];
                yl = y[lp];
                yr = y[rp];
                nl = spline.derivative(std::log(xl)) / yl;
                nr = spline.derivative(std::log(xr)) / yr;
            }
            double operator()(double xx) const {
                if (xx <= xl)
                    return yl * std::pow(xx / xl, nl);
                if (xx >= xr)
                    return yr * std::pow(xx / xr, nr);
                return spline(std::log(xx));
            }

          private:
            NaturalCubicSpline spline;
            double xl{}, xr{}, yl{}, yr{}, nl{}, nr{};
        };

        inline DVector logspace(double log10_min, double log10_max, int n) {
            DVector v(n);
            for (int i = 0; i < n; i++)
                v[i] = std::pow(10.0, log10_min + (log10_max - log10_min) * i / double(n - 1));
            return v;
        }
    } // namespace detail

    //=================================================================================================================
    ///
    /// The functions of the Lagrangian separation q of one-loop Lagrangian perturbation theory (CLEFT; Carlson, Reid &
    /// White 2013, Vlah, Castorina & White 2016, Chen, Vlah & White 2020): the linear X(q), Y(q), U(q), xi(q), the
    /// shear terms, the one-loop A_ij, W_ijk, A10, U3, U11, U20, Us2 and the third-order bias terms (theta, Ub3),
    /// computed with FFTLog from the loop spectra Q_n(k), R_n(k) (Matsubara 2008). A port of velocileptors'
    /// Utils/qfuncfft.py (Chen, Vlah & White, https://github.com/sfschen/velocileptors, MIT license), reproduced to
    /// rounding. Input: the linear P(k) on the logarithmic grid k (cut off at high k, e.g. exp(-(k/10)^2)), output
    /// on the logarithmic grid qv. kIR > 0 splits X and Y into the parts from k < kIR (_lt, a Gaussian split) and the
    /// rest (_gt); kIR = 0: no split (_gt = 0). Units: those of the inputs (k in h/Mpc, P in (Mpc/h)^3, q in Mpc/h).
    ///
    //=================================================================================================================
    class LagrangianQFunctions {
      public:
        using DVector = std::vector<double>;

        DVector k, p, qv;
        // Linear
        DVector Xlin, Ylin, Xlin_lt, Ylin_lt, Xlin_gt, Ylin_gt, Ulin, corlin;
        // Shear
        DVector V, Xs2, Ys2, zeta, chi;
        // Loop spectra (on k)
        DVector Q1, Q2, Q3, Q5, Q8, Qs2, R1, R2;
        // One loop
        DVector Xloop13, Yloop13, Xloop22, Yloop22, Tloop112, V1loop112, V3loop112, X10loop12, Y10loop12;
        double zerolag_10_loop12{0.0};
        DVector U3, U11, U20, Us2;
        // Third order bias
        DVector Rb3, theta, Ub3;

        LagrangianQFunctions() = default;
        LagrangianQFunctions(const DVector & k_, const DVector & p_, const DVector & qv_, double kIR = 0.0, bool oneloop = true)
            : k(k_), p(p_), qv(qv_) {
            if (k.size() != p.size() or k.size() < 4 or qv.size() < 4)
                throw std::invalid_argument("[LagrangianQFunctions] k, p must have the same size (>= 4)");
            const size_t n = k.size();
            ir_less.assign(n, 1.0);
            ir_greater.assign(n, 0.0);
            if (kIR > 0.0)
                for (size_t i = 0; i < n; i++) {
                    ir_less[i] = std::exp(-std::pow(k[i] / kIR, 2));
                    ir_greater[i] = -std::expm1(-std::pow(k[i] / kIR, 2));
                }
            sph = SOLVERS::FFTLog::SphericalBesselTransform(k, 5, true, true, false);
            sphr = SOLVERS::FFTLog::SphericalBesselTransform(qv, 5, false, true, false);
            setup_xiln();
            setup_2pts();
            setup_shear();
            if (oneloop) {
                setup_QR();
                setup_oneloop_2pts();
            } else {
                const DVector zq(qv.size(), 0.0);
                Xloop13 = Yloop13 = Xloop22 = Yloop22 = Tloop112 = V1loop112 = V3loop112 = X10loop12 = Y10loop12 = zq;
                U3 = U11 = U20 = Us2 = zq;
            }
            setup_third_order();
        }

        /// The generalized correlation function xi_l^n(q) = int dk k^(2+n) G(k) j_l(kq) / (2 pi^2) on qv (G = P if
        /// G is empty); ir: 0 all, 1 the k < kIR part, 2 the rest
        DVector xi_l_n(int l, int n, const DVector & G = {}, int ir = 0) {
            const DVector & base = G.empty() ? p : G;
            DVector integrand(k.size());
            for (size_t i = 0; i < k.size(); i++) {
                integrand[i] = base[i] * std::pow(k[i], n);
                if (ir == 1)
                    integrand[i] *= ir_less[i];
                if (ir == 2)
                    integrand[i] *= ir_greater[i];
            }
            const auto x = sph.transform(l, integrand);
            return detail::interp_clamped(qv, sph.y(l), x);
        }

      private:
        DVector ir_less, ir_greater;
        SOLVERS::FFTLog::SphericalBesselTransform sph, sphr;
        DVector xi00, xi1m1, xi0m2, xi2m2, xi0m2_lt, xi2m2_lt, xi20, xi40, xi11, xi31, xi3m1, xi02, xi22;

        // The transform (q to k) of a function of q, interpolated onto k
        DVector template_QR(int l, const DVector & integrand) {
            const auto x = sphr.transform(l, integrand);
            return detail::interp_clamped(k, sphr.y(l), x);
        }
        template <class F>
        DVector qmap(F && f) const {
            DVector v(qv.size());
            for (size_t i = 0; i < qv.size(); i++)
                v[i] = f(i);
            return v;
        }
        template <class F>
        DVector kmap(F && f) const {
            DVector v(k.size());
            for (size_t i = 0; i < k.size(); i++)
                v[i] = f(i);
            return v;
        }

        void setup_xiln() {
            xi00 = xi_l_n(0, 0);
            xi1m1 = xi_l_n(1, -1);
            xi0m2 = xi_l_n(0, -2);
            xi2m2 = xi_l_n(2, -2);
            xi0m2_lt = xi_l_n(0, -2, {}, 1);
            xi2m2_lt = xi_l_n(2, -2, {}, 1);
            xi20 = xi_l_n(2, 0);
            xi40 = xi_l_n(4, 0);
            xi11 = xi_l_n(1, 1);
            xi31 = xi_l_n(3, 1);
            xi3m1 = xi_l_n(3, -1);
            xi02 = xi_l_n(0, 2);
            xi22 = xi_l_n(2, 2);
        }

        void setup_2pts() {
            Xlin = qmap([&](size_t i) { return 2.0 / 3.0 * (xi0m2[0] - xi0m2[i] - xi2m2[i]); });
            Ylin = qmap([&](size_t i) { return 2.0 * xi2m2[i]; });
            Xlin_lt = qmap([&](size_t i) { return 2.0 / 3.0 * (xi0m2_lt[0] - xi0m2_lt[i] - xi2m2_lt[i]); });
            Ylin_lt = qmap([&](size_t i) { return 2.0 * xi2m2_lt[i]; });
            Xlin_gt = qmap([&](size_t i) { return Xlin[i] - Xlin_lt[i]; });
            Ylin_gt = qmap([&](size_t i) { return Ylin[i] - Ylin_lt[i]; });
            Ulin = qmap([&](size_t i) { return -xi1m1[i]; });
            corlin = xi00;
        }

        void setup_shear() {
            V.resize(qv.size());
            Xs2.resize(qv.size());
            Ys2.resize(qv.size());
            zeta.resize(qv.size());
            chi.resize(qv.size());
            for (size_t i = 0; i < qv.size(); i++) {
                const double J2 = 2. * xi1m1[i] / 15 - 0.2 * xi3m1[i];
                const double J3 = -0.2 * xi1m1[i] - 0.2 * xi3m1[i];
                const double J4 = xi3m1[i];
                V[i] = 4 * J2 * xi20[i];
                Xs2[i] = 4 * J3 * J3;
                Ys2[i] = 6 * J2 * J2 + 8 * J2 * J3 + 4 * J2 * J4 + 4 * J3 * J3 + 8 * J3 * J4 + 2 * J4 * J4;
                zeta[i] = 2 * (4 * xi00[i] * xi00[i] / 45. + 8 * xi20[i] * xi20[i] / 63. + 8 * xi40[i] * xi40[i] / 35);
                chi[i] = 4 * xi20[i] * xi20[i] / 3.;
            }
        }

        void setup_QR() {
            const double Qfac = 4 * M_PI;
            auto sq = [](double v) { return v * v; };
            Q1 = template_QR(0, qmap([&](size_t i) {
                                 return Qfac * (8. / 15 * sq(xi00[i]) - 16. / 21 * sq(xi20[i]) + 8. / 35 * sq(xi40[i]));
                             }));
            Q2 = template_QR(0, qmap([&](size_t i) {
                                 return Qfac * (4. / 5 * sq(xi00[i]) - 4. / 7 * sq(xi20[i]) - 8. / 35 * sq(xi40[i]) -
                                                4. / 5 * xi11[i] * xi1m1[i] + 4. / 5 * xi31[i] * xi3m1[i]);
                             }));
            Q3 = template_QR(0, qmap([&](size_t i) {
                                 return Qfac * (38. / 15 * sq(xi00[i]) + 2. / 3 * xi02[i] * xi0m2[i] -
                                                32. / 5 * xi1m1[i] * xi11[i] + 68. / 21 * sq(xi20[i]) +
                                                4. / 3 * xi22[i] * xi2m2[i] - 8. / 5 * xi31[i] * xi3m1[i] +
                                                8. / 35 * sq(xi40[i]));
                             }));
            Q5 = template_QR(0, qmap([&](size_t i) {
                                 return Qfac * (2. / 3 * sq(xi00[i]) - 2. / 3 * sq(xi20[i]) - 2. / 5 * xi11[i] * xi1m1[i] +
                                                2. / 5 * xi31[i] * xi3m1[i]);
                             }));
            Q8 = template_QR(0, qmap([&](size_t i) { return Qfac * (2. / 3 * sq(xi00[i]) - 2. / 3 * sq(xi20[i])); }));
            Qs2 = template_QR(0, qmap([&](size_t i) {
                                  return Qfac * (-4. / 15 * sq(xi00[i]) + 20. / 21 * sq(xi20[i]) - 24. / 35 * sq(xi40[i]));
                              }));
            const auto R1_0 = template_QR(0, qmap([&](size_t i) { return xi00[i] / qv[i]; }));
            const auto R1_2 = template_QR(2, qmap([&](size_t i) { return xi20[i] / qv[i]; }));
            const auto R1_4 = template_QR(4, qmap([&](size_t i) { return xi40[i] / qv[i]; }));
            const auto R2_1 = template_QR(1, qmap([&](size_t i) { return xi1m1[i] / qv[i]; }));
            const auto R2_3 = template_QR(3, qmap([&](size_t i) { return xi3m1[i] / qv[i]; }));
            R1 = kmap([&](size_t i) {
                return k[i] * k[i] * p[i] * (8. / 15 * R1_0[i] - 16. / 21 * R1_2[i] + 8. / 35 * R1_4[i]);
            });
            R2 = kmap([&](size_t i) {
                return k[i] * k[i] * p[i] *
                       (-2. / 15 * R1_0[i] - 2. / 21 * R1_2[i] + 8. / 35 * R1_4[i] + k[i] * 2. / 5 * R2_1[i] -
                        k[i] * 2. / 5 * R2_3[i]);
            });
        }

        void setup_oneloop_2pts() {
            const auto G13 = kmap([&](size_t i) { return 5. / 21 * R1[i]; });
            const auto xi0m2loop13 = xi_l_n(0, -2, G13);
            const auto xi2m2loop13 = xi_l_n(2, -2, G13);
            Xloop13 = qmap([&](size_t i) { return 2. / 3 * (xi0m2loop13[0] - xi0m2loop13[i] - xi2m2loop13[i]); });
            Yloop13 = qmap([&](size_t i) { return 2 * xi2m2loop13[i]; });

            const auto G22 = kmap([&](size_t i) { return 9. / 98 * Q1[i]; });
            const auto xi0m2loop22 = xi_l_n(0, -2, G22);
            const auto xi2m2loop22 = xi_l_n(2, -2, G22);
            Xloop22 = qmap([&](size_t i) { return 2. / 3 * (xi0m2loop22[0] - xi0m2loop22[i] - xi2m2loop22[i]); });
            Yloop22 = qmap([&](size_t i) { return 2 * xi2m2loop22[i]; });

            Tloop112 = xi_l_n(3, -3, kmap([&](size_t i) { return -3. / 7 * (2 * R1[i] + 4 * R2[i] + Q1[i] + 2 * Q2[i]); }));
            const auto v1 =
                xi_l_n(1, -3, kmap([&](size_t i) { return 3. / 35 * (-3 * R1[i] + 4 * R2[i] + Q1[i] + 2 * Q2[i]); }));
            const auto v3 =
                xi_l_n(1, -3, kmap([&](size_t i) { return 3. / 35 * (2 * R1[i] + 4 * R2[i] - 4 * Q1[i] + 2 * Q2[i]); }));
            V1loop112 = qmap([&](size_t i) { return v1[i] - 0.2 * Tloop112[i]; });
            V3loop112 = qmap([&](size_t i) { return v3[i] - 0.2 * Tloop112[i]; });

            // A10 (trapezoid in k)
            zerolag_10_loop12 = 0.0;
            for (size_t i = 0; i + 1 < k.size(); i++)
                zerolag_10_loop12 +=
                    0.5 * ((R1[i] - R2[i]) / 7. + (R1[i + 1] - R2[i + 1]) / 7.) * (k[i + 1] - k[i]);
            zerolag_10_loop12 /= 2 * M_PI * M_PI;
            const auto x0 = xi_l_n(0, -2, kmap([&](size_t i) { return 4 * R2[i] + 2 * Q5[i]; }));
            const auto x2 = xi_l_n(2, -2, kmap([&](size_t i) { return 3 * R1[i] + 4 * R2[i] + 2 * Q5[i]; }));
            X10loop12 = qmap([&](size_t i) { return zerolag_10_loop12 - x0[i] / 14. - x2[i] / 14.; });
            Y10loop12 = qmap([&](size_t i) { return 3 * x2[i] / 14.; });

            U3 = xi_l_n(1, -1, kmap([&](size_t i) { return -5. / 21 * R1[i]; }));
            U11 = xi_l_n(1, -1, kmap([&](size_t i) { return -6. / 7 * (R1[i] + R2[i]); }));
            U20 = xi_l_n(1, -1, kmap([&](size_t i) { return -3. / 7 * Q8[i]; }));
            Us2 = xi_l_n(1, -1, kmap([&](size_t i) { return -1. / 7 * Qs2[i]; }));
        }

        void setup_third_order() {
            const auto t0 = template_QR(0, qmap([&](size_t i) { return 24. / 5 * xi00[i] / qv[i]; }));
            const auto t1 = template_QR(1, qmap([&](size_t i) { return -16. / 5 * xi11[i] / qv[i]; }));
            const auto t2a = template_QR(2, qmap([&](size_t i) { return -20. / 7 * xi20[i] / qv[i]; }));
            const auto t2b = template_QR(2, qmap([&](size_t i) { return 4. * xi22[i] / qv[i]; }));
            const auto t3 = template_QR(3, qmap([&](size_t i) { return -24. / 5 * xi31[i] / qv[i]; }));
            const auto t4 = template_QR(4, qmap([&](size_t i) { return 72. / 35 * xi40[i] / qv[i]; }));
            Rb3 = kmap([&](size_t i) {
                const double kk = k[i], k2 = kk * kk;
                return 2 * 2. / 63 * (k2 * t0[i] + kk * t1[i] + (k2 * t2a[i] + t2b[i]) + kk * t3[i] + k2 * t4[i]);
            });
            const auto pRb3 = kmap([&](size_t i) { return p[i] * Rb3[i]; });
            theta = xi_l_n(0, 0, pRb3);
            const auto ub3 = xi_l_n(1, -1, pRb3);
            Ub3 = qmap([&](size_t i) { return -ub3[i]; });
        }
    };

} // namespace FML::PERTURBATION

#endif
