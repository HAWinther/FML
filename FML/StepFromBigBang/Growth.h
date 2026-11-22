#ifndef FML_STEPFROMBIGBANG_GROWTH_H
#define FML_STEPFROMBIGBANG_GROWTH_H
// =====================================================================================================================
// The growth history that StepFromBigBang needs: the LPT growth factors to third order, their derivatives, and the
// linear response of the force. Functions of the scale factor a and of k = koverH0 = c k / H0 (k ignored when the
// growth does not depend on scale).
//
// Conventions (see README.md):
//   - Positions in box units, velocities v = a^2 dx/dt / (H0 B) = a^2 E dx/dlna (the FML NBody convention).
//   - The growing mode is x = q + D1 Psi1 + r2 Psi2 + r3a Psi3a + r3b Psi3b + r3c Psi3c with the fields Psi normalised
//     as in Einstein-de Sitter: there r2 = D1^2 and r3a = r3b = r3c = D1^3. In terms of the usual growth factors
//     (LPTGrowthFactors, SimulationCode): r2 = -7/3 D2, r3a = -3 D3a, r3b = 21/5 D3b.
//   - kappa(a, k): the force on a linear (longitudinal) displacement psi is dv/dlna = kappa psi. For GR-like models
//     kappa = a^2 E (3/2) Omega_m(a) mu(a, k) s_1(a, k), and the force of any configuration is kappa(a) times a
//     time-independent functional of the displacement field (with the second-order part times N(a)).
//
// With ' = d/dlna and beta = 2 + dlnE/dlna the growth factors obey (GR-like models)
//   (a^2 E D1')'  = kappa D1
//   (a^2 E r2')'  = kappa (r2 + 7/3 N D1^2)
//   (a^2 E r3a')' = kappa (r3a + 6 D1^3)
//   (a^2 E r3b')' = kappa (r3b + 9/5 D1 r2 + 21/5 D1^3)
//   (a^2 E r3c')' = kappa 7 D1^3                     (the transverse part: no linear term)
// because the third-order force of the configuration splits as Y = F3[Psi1^3] = 6 Psi3a + 21/5 Psi3b + 7 Psi3c and
// X = 2 F2[Psi1, Psi2] = 9/5 Psi3b (measured numerically with the exact particle force; StepFromBigBang.h).
// =====================================================================================================================

#include <FML/Cosmology/Growth/LPTGrowthFactors.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace FML::STEPFROMBIGBANG {

    struct GrowthModel {
        using F1 = std::function<double(double)>;
        using F2 = std::function<double(double, double)>;
        F1 E;                                          ///< H/H0 of a
        F2 kappa;                                      ///< linear response dv/dlna = kappa psi, of (a, koverH0)
        F2 D1, dD1;                                    ///< D1 and dD1/dlna
        F2 r2, dr2, r3a, dr3a, r3b, dr3b, r3c, dr3c;   ///< the EdS-normalised higher orders and d/dlna
        F2 N;                                          ///< second-order source (1 in GR)
        bool scale_dependent{false};                   ///< D1 depends on k
        bool gr_third_order{true};                     ///< the third-order force is that of GR (lpt3 needs it)
        double a_min{1e-10}, a_max{10.0};              ///< where the functions are valid

        /// v = vnorm(a) dx/dlna
        double vnorm(double a) const { return a * a * E(a); }

        /// The scale factor where D1(a, k) = D (bisection in ln a over [a_min, a_max])
        double a_of_D1(double D, double k) const {
            double lo = std::log(a_min), hi = std::log(a_max);
            if (D <= D1(a_min, k))
                return a_min;
            if (D >= D1(a_max, k))
                throw std::runtime_error("[StepFromBigBang] a_of_D1: D1 beyond a_max");
            for (int i = 0; i < 200; i++) {
                const double mid = 0.5 * (lo + hi);
                if (D1(std::exp(mid), k) < D)
                    lo = mid;
                else
                    hi = mid;
                if (hi - lo < 1e-14)
                    break;
            }
            return std::exp(0.5 * (lo + hi));
        }
    };

    /// Einstein-de Sitter: E = a^-3/2, D1 = a, r2 = a^2, r3 = a^3, kappa = 3/2 a^1/2
    inline GrowthModel growth_EdS() {
        GrowthModel g;
        g.E = [](double a) { return std::pow(a, -1.5); };
        g.kappa = [](double a, double) { return 1.5 * std::sqrt(a); };
        g.D1 = [](double a, double) { return a; };
        g.dD1 = [](double a, double) { return a; };
        g.r2 = [](double a, double) { return a * a; };
        g.dr2 = [](double a, double) { return 2 * a * a; };
        auto cube = [](double a, double) { return a * a * a; };
        auto dcube = [](double a, double) { return 3 * a * a * a; };
        g.r3a = g.r3b = g.r3c = cube;
        g.dr3a = g.dr3b = g.dr3c = dcube;
        g.N = [](double, double) { return 1.0; };
        g.a_min = 1e-12;
        g.a_max = 1e4;
        return g;
    }

    /// From the library's growth solver. The equations (E, dlnE/dx, rho_m, mu, N, source_1) must stay valid as long
    /// as the model is used (they are copied, but may refer to a background). r3c is not computed by LPTGrowthFactors:
    /// it is integrated here from its equation, at k = 0 (it only enters at third order).
    inline GrowthModel growth_from_LPTGrowthFactors(std::shared_ptr<const FML::COSMOLOGY::GROWTH::LPTGrowthFactors> lpt,
                                                    const FML::COSMOLOGY::GROWTH::LPTGrowthEquations & eq) {
        using FML::INTERPOLATION::SPLINE::Spline;
        GrowthModel g;
        g.scale_dependent = lpt->scale_dependent();
        g.a_min = lpt->a_min();
        g.a_max = lpt->a_max();
        auto E = eq.E;
        auto rhom = eq.matter_density;
        auto mu = eq.mu ? eq.mu : [](double, double) { return 1.0; };
        auto s1 = eq.source_1 ? eq.source_1 : [](double, double) { return 1.0; };
        auto N = eq.second_order_source ? eq.second_order_source : [](double, double) { return 1.0; };
        g.E = [E](double a) { return E(std::log(a)); };
        g.kappa = [E, rhom, mu, s1](double a, double k) {
            const double x = std::log(a), e = E(x);
            return a * a * e * 1.5 * rhom(x) / (e * e) * mu(x, k) * s1(x, k);
        };
        g.N = [N](double a, double k) { return N(std::log(a), k); };
        g.D1 = [lpt](double a, double k) { return lpt->D1(std::log(a), k); };
        g.dD1 = [lpt](double a, double k) { return lpt->f1(std::log(a), k) * lpt->D1(std::log(a), k); };
        g.r2 = [lpt](double a, double k) { return -7.0 / 3.0 * lpt->D2(std::log(a), k); };
        g.dr2 = [lpt](double a, double k) { return -7.0 / 3.0 * lpt->f2(std::log(a), k) * lpt->D2(std::log(a), k); };
        g.r3a = [lpt](double a, double k) { return -3.0 * lpt->D3a(std::log(a), k); };
        g.dr3a = [lpt](double a, double k) { return -3.0 * lpt->f3a(std::log(a), k) * lpt->D3a(std::log(a), k); };
        g.r3b = [lpt](double a, double k) { return 21.0 / 5.0 * lpt->D3b(std::log(a), k); };
        g.dr3b = [lpt](double a, double k) { return 21.0 / 5.0 * lpt->f3b(std::log(a), k) * lpt->D3b(std::log(a), k); };

        // r3c: (a^2 E r3c')' = 7 kappa D1^3 at k = 0, starting at a_min with r3c = r3b (equal in the EdS growing mode)
        // and the same rate. RK4 in x = ln a on a fine grid.
        const int n = 8000;
        const double x0 = std::log(g.a_min), x1 = std::log(g.a_max), h = (x1 - x0) / (n - 1);
        auto kap = g.kappa;
        auto D1k0 = [lpt](double x) { return lpt->D1_k0(x); };
        auto vn = [E](double x) { return std::exp(2 * x) * E(x); };
        // y = (r3c, p = a^2 E r3c')
        auto rhs = [&](double x, const std::array<double, 2> & y) {
            const double D = D1k0(x);
            return std::array<double, 2>{y[1] / vn(x), 7.0 * kap(std::exp(x), 0.0) * D * D * D};
        };
        std::vector<double> xs(n), r(n), p(n);
        std::array<double, 2> y{21.0 / 5.0 * lpt->D3b_k0(x0),
                                vn(x0) * 21.0 / 5.0 * lpt->D3b_k0(x0) * lpt->f3b(x0, 0.0)};
        for (int i = 0; i < n; i++) {
            const double x = x0 + i * h;
            xs[i] = x;
            r[i] = y[0];
            p[i] = y[1];
            if (i == n - 1)
                break;
            auto k1 = rhs(x, y);
            auto k2 = rhs(x + h / 2, {y[0] + h / 2 * k1[0], y[1] + h / 2 * k1[1]});
            auto k3 = rhs(x + h / 2, {y[0] + h / 2 * k2[0], y[1] + h / 2 * k2[1]});
            auto k4 = rhs(x + h, {y[0] + h * k3[0], y[1] + h * k3[1]});
            for (int j = 0; j < 2; j++)
                y[j] += h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        }
        std::vector<double> dr(n);
        for (int i = 0; i < n; i++)
            dr[i] = p[i] / vn(xs[i]);
        auto r3c = std::make_shared<Spline>(xs, r, "r3c");
        auto dr3c = std::make_shared<Spline>(xs, dr, "dr3c");
        g.r3c = [r3c](double a, double) { return (*r3c)(std::log(a)); };
        g.dr3c = [dr3c](double a, double) { return (*dr3c)(std::log(a)); };
        return g;
    }

} // namespace FML::STEPFROMBIGBANG
#endif
