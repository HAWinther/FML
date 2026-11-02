#ifndef FML_COSMOLOGY_GROWTH_LPTGROWTHFACTORS_H
#define FML_COSMOLOGY_GROWTH_LPTGROWTHFACTORS_H

// =====================================================================================================================
// LPTGrowthFactors: the growth factors of Lagrangian perturbation theory to third order, D1, D2, D3a, D3b, as functions
// of x = ln a (and of k when the growth is scale dependent), on a background and for a theory of gravity. The one
// growth solver of the library: SimulationCode's GravityModel and LPT's compute_LPT_growth_factors_* use it.
//
// With ' = d/dx, S = (3/2) rho_m(x) mu(x, k) / E^2 (rho_m in units of rho_c0, so S = (3/2) Omega_m(a) mu) and the
// per-order source factors s_i(x, k) (1 by default; e.g. 1 - f_nu for the CDM+baryon fluid with massive neutrinos):
//   D1''  + (2 + dlnE/dx) D1'  = S s_1  D1
//   D2''  + (2 + dlnE/dx) D2'  = S s_2  (D2 - N D1^2)          N = second_order_source(x, k) (1 in GR; DGP < 1)
//   D3a'' + (2 + dlnE/dx) D3a' = S s_3a (D3a - 2 D1^3)
//   D3b'' + (2 + dlnE/dx) D3b' = S s_3b (D3b + D1^3 - D2 D1)
// normalized to D1 = 1 at a_normalize (D2 by D1^2, D3 by D1^3 at the same time), so in EdS D2 = -3/7 D1^2,
// D3a = -1/3 D1^3, D3b = 5/21 D1^3.
//
// Initial conditions at a_min: the growing mode of a matter + radiation universe, a power series in y = a/a_eq
// (radiation_matter_growing_mode: D1 = 1 + 3y/2 and the higher orders that vanish for y -> 0), with
// y = r/(1 - r), r = rho_m/E^2 at a_min; the EdS values for y >= 0.5 (or with eds_initial_conditions). With
// start_in_radiation_era, a_min is moved to a_eq/10 when 0.1 < a_min/a_eq < 1e4 (radiation matters but a_min is
// not yet deep in its era). initial_rate_1(x_min, k), if given, replaces dlnD1/dx at a_min for k > 0 (e.g. the
// CDM+baryon growth rate of transfer functions with massive neutrinos).
//
// The model (LPTGrowthEquations) is a set of functions: E, dlnE/dx, rho_m (defaults from a Background:
// Omega_m(x) E^2, all matter including massive neutrinos), mu and N (from a QuasiStaticGravity, or the user's), the
// source factors. They are only called while solving (in parallel over k with OpenMP: they must be thread safe).
//
// Results: D1(x, k), ..., the rates f_i = dlnD_i/dx; k = koverH0 = c k / H0 (k [h/Mpc] * 2997.92458). Without scale
// dependence k is ignored; with it k is clamped to [koverH0_min, koverH0_max] (also k = 0: the large-scale value is
// the solution at koverH0_min, D1_k0 etc. give the k = 0 solution). The splines cover [a_min(), a_max]. history()
// gives the scale-independent (k = 0) D1, D2 as a GrowthHistory (LPT/GrowthHistory.h, for the N-body time stepping).
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/LPT/GrowthHistory.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::GROWTH {

    /// The functions that define the growth equations (see the header)
    struct LPTGrowthEquations {
        using Function = std::function<double(double)>;
        using FunctionK = std::function<double(double, double)>;
        Function E;              ///< H/H0 (required)
        Function dlogEdx;        ///< dlnH/dlna (required)
        Function matter_density; ///< rho_m / rho_c0 of the clustering matter (required)
        FunctionK mu;            ///< G_eff/G (x, koverH0); default 1
        FunctionK second_order_source; ///< N(x, koverH0) of the 2LPT source; default 1
        FunctionK source_1, source_2, source_3a, source_3b; ///< per-order source factors; default 1
        FunctionK initial_rate_1; ///< dlnD1/dx at (x_min, koverH0) for koverH0 > 0; default the growing mode

        /// E, dlnE/dx and rho_m = Omega_m(x) E^2 from a background (it must outlive the equations)
        static LPTGrowthEquations from_background(const BACKGROUND::Background & bg) {
            LPTGrowthEquations eq;
            eq.E = [&bg](double x) { return bg.E(x); };
            eq.dlogEdx = [&bg](double x) { return bg.dlogEdx(x); };
            eq.matter_density = [&bg](double x) {
                const double E = bg.E(x);
                return bg.Omega_m(x) * E * E;
            };
            return eq;
        }
        /// ... and mu and the second-order source of a quasi-static theory (kept alive by the equations)
        static LPTGrowthEquations from_background(const BACKGROUND::Background & bg,
                                                  PERTURBATIONS::QuasiStaticGravityPtr qs) {
            auto eq = from_background(bg);
            if (qs) {
                eq.mu = [qs](double x, double k) { return qs->mu(x, k); };
                eq.second_order_source = [qs](double x, double k) { return qs->second_order_source(x, k); };
            }
            return eq;
        }
    };

    struct LPTGrowthSettings {
        double a_min = 1e-4;
        double a_max = 10.0;
        int n_a = 1000;
        double a_normalize = 1.0;              ///< D1 = 1 here
        bool start_in_radiation_era = true;    ///< move a_min to a_eq/10 when radiation matters there
        bool eds_initial_conditions = false;   ///< EdS initial conditions whatever a_min/a_eq
        bool scale_dependent = false;          ///< solve on a grid in k (else only k = 0)
        double koverH0_min = 1e-4 * FML::UTILS::SI::c_over_H0_mpch;  ///< 1e-4 h/Mpc
        double koverH0_max = 100.0 * FML::UTILS::SI::c_over_H0_mpch; ///< 100 h/Mpc
        int n_k = 200;
    };

    class LPTGrowthFactors {
      public:
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        LPTGrowthFactors(const LPTGrowthEquations & eq, const LPTGrowthSettings & settings) : s(settings) {
            if (not eq.E or not eq.dlogEdx or not eq.matter_density)
                throw std::invalid_argument("[LPTGrowthFactors] E, dlogEdx and matter_density are required");
            if (not(s.a_min > 0.0 and s.a_min < s.a_max and s.n_a >= 4))
                throw std::invalid_argument("[LPTGrowthFactors] Need 0 < a_min < a_max and n_a >= 4");
            if (s.scale_dependent and not(s.koverH0_min > 0.0 and s.koverH0_min < s.koverH0_max and s.n_k >= 4))
                throw std::invalid_argument("[LPTGrowthFactors] Need 0 < koverH0_min < koverH0_max and n_k >= 4");
            solve(eq);
        }
        /// From a background (and optionally a quasi-static theory), with GR / that theory's mu and source
        LPTGrowthFactors(const BACKGROUND::Background & bg,
                         const LPTGrowthSettings & settings,
                         PERTURBATIONS::QuasiStaticGravityPtr qs = nullptr)
            : LPTGrowthFactors(LPTGrowthEquations::from_background(bg, std::move(qs)), settings) {}

        /// The growth factors at x = ln a and k = koverH0 (see the header)
        double D1(double x, double k = 0.0) const { return value(0, x, k); }
        double D2(double x, double k = 0.0) const { return value(1, x, k); }
        double D3a(double x, double k = 0.0) const { return value(2, x, k); }
        double D3b(double x, double k = 0.0) const { return value(3, x, k); }
        /// The growth rates dlnD/dx
        double f1(double x, double k = 0.0) const { return rate(0, x, k); }
        double f2(double x, double k = 0.0) const { return rate(1, x, k); }
        double f3a(double x, double k = 0.0) const { return rate(2, x, k); }
        double f3b(double x, double k = 0.0) const { return rate(3, x, k); }
        /// The k = 0 solution (also when the growth is scale dependent)
        double D1_k0(double x) const { return D0[0](x); }
        double D2_k0(double x) const { return D0[1](x); }
        double D3a_k0(double x) const { return D0[2](x); }
        double D3b_k0(double x) const { return D0[3](x); }

        bool scale_dependent() const { return s.scale_dependent; }
        /// The start of the integration (a_min of the settings, or earlier with start_in_radiation_era)
        double a_min() const { return std::exp(xgrid.front()); }
        double a_max() const { return std::exp(xgrid.back()); }
        const DVector & x_grid() const { return xgrid; }
        /// ln koverH0 of the k grid (empty without scale dependence)
        const DVector & logk_grid() const { return logkgrid; }
        const LPTGrowthSettings & settings() const { return s; }
        /// The splines of the k = 0 solution D1, D2, D3a, D3b (of x)
        const std::array<Spline, 4> & splines_k0() const { return D0; }

        /// The scale-independent (k = 0 of the getters above) D1, D2 and their rates as functions of a
        FML::COSMOLOGY::GrowthHistory history() const {
            auto self = std::make_shared<const std::array<Spline, 4>>(D0);
            return {[self](double a) { return (*self)[0](std::log(a)); },
                    [self](double a) { return (*self)[0].deriv_x(std::log(a)) / (*self)[0](std::log(a)); },
                    [self](double a) { return (*self)[1](std::log(a)); },
                    [self](double a) { return (*self)[1].deriv_x(std::log(a)) / (*self)[1](std::log(a)); }};
        }

        //========================================================================
        /// Growing-mode initial conditions of the LPT growth equations in a universe with matter and (smooth)
        /// radiation, as a power series in y = a/a_eq: D1 = 1 + 3y/2 is the Meszaros growing mode, and D2, D3a, D3b
        /// are the solutions that vanish as y -> 0 (no growth during radiation domination). Returns
        /// {D1, dD1/dlna, D2, dD2/dlna, D3a, dD3a/dlna, D3b, dD3b/dlna}. For y -> infinity this tends to the EdS
        /// values D2 = -3/7 D1^2, D3a = -1/3 D1^3, D3b = 5/21 D1^3. Converges for y < 1.
        //========================================================================
        static std::array<double, 8> radiation_matter_growing_mode(double y, int nterms = 200) {
            const int N = nterms;
            auto mult = [N](const std::vector<double> & u, const std::vector<double> & v) {
                std::vector<double> w(N, 0.0);
                for (int i = 0; i < N; i++)
                    for (int j = 0; i + j < N; j++)
                        w[i + j] += u[i] * v[j];
                return w;
            };
            // (1+y) D'' + y/2 D' - 3/2 y D + 3/2 y S = 0 with ' = d/dlny, D = sum c_n y^n, c_0 = 0
            auto solve = [N](const std::vector<double> & S) {
                std::vector<double> c(N, 0.0);
                for (int n = 1; n < N; n++) {
                    const double m = n - 1;
                    c[n] = -((m * m + 0.5 * m - 1.5) * c[n - 1] + 1.5 * S[n - 1]) / double(n * n);
                }
                return c;
            };
            std::vector<double> D1(N, 0.0);
            D1[0] = 1.0;
            D1[1] = 1.5;
            auto D1sq = mult(D1, D1);
            auto D1cube = mult(D1sq, D1);
            auto D2 = solve(D1sq);
            std::vector<double> S3a(N), S3b(N);
            auto D2D1 = mult(D2, D1);
            for (int n = 0; n < N; n++) {
                S3a[n] = 2.0 * D1cube[n];
                S3b[n] = D2D1[n] - D1cube[n];
            }
            auto D3a = solve(S3a);
            auto D3b = solve(S3b);

            std::array<double, 8> res{};
            const std::vector<double> * series[4] = {&D1, &D2, &D3a, &D3b};
            for (int k = 0; k < 4; k++) {
                double val = 0.0, der = 0.0, yn = 1.0;
                for (int n = 0; n < N; n++) {
                    val += (*series[k])[n] * yn;
                    der += n * (*series[k])[n] * yn;
                    yn *= y;
                }
                res[2 * k] = val;
                res[2 * k + 1] = der;
            }
            return res;
        }

        /// a/a_eq from r = rho_m/E^2 (= y/(1 + y) in a matter + radiation universe); 1e30 when radiation is negligible
        static double a_over_aeq(const LPTGrowthEquations & eq, double x) {
            const double E = eq.E(x);
            const double r = eq.matter_density(x) / (E * E);
            if (r >= 1.0 - 1e-12)
                return 1e30;
            return r / (1.0 - r);
        }

      private:
        LPTGrowthSettings s;
        DVector xgrid, logkgrid;
        std::array<Spline, 4> D0;
        std::array<Spline2D, 4> Dk;

        double value(int i, double x, double k) const {
            if (not s.scale_dependent)
                return D0[i](x);
            return Dk[i](std::log(std::max(k, s.koverH0_min)), x);
        }
        double rate(int i, double x, double k) const {
            if (not s.scale_dependent)
                return D0[i].deriv_x(x) / D0[i](x);
            const double lk = std::log(std::max(k, s.koverH0_min));
            return Dk[i].deriv_y(lk, x) / Dk[i](lk, x);
        }

        static double call(const LPTGrowthEquations::FunctionK & f, double x, double k) { return f ? f(x, k) : 1.0; }

        void solve(const LPTGrowthEquations & eq) {
            double amin = s.a_min;
            if (s.start_in_radiation_era and not s.eds_initial_conditions) {
                const double y = a_over_aeq(eq, std::log(amin));
                if (y > 0.1 and y < 1e4)
                    amin *= 0.1 / y;
            }
            const double y_low = a_over_aeq(eq, std::log(amin));

            xgrid.resize(s.n_a);
            for (int i = 0; i < s.n_a; i++)
                xgrid[i] = std::log(amin) + std::log(s.a_max / amin) * i / double(s.n_a - 1);

            auto D = solve_k(eq, 0.0, y_low);
            const char * names[4] = {"D1LPT(log(a))", "D2LPT(log(a))", "D3LPTa(log(a))", "D3LPTb(log(a))"};
            for (int i = 0; i < 4; i++)
                D0[i].create(xgrid, D[i], names[i]);
            if (not s.scale_dependent)
                return;

            logkgrid.resize(s.n_k);
            for (int i = 0; i < s.n_k; i++)
                logkgrid[i] =
                    std::log(s.koverH0_min) + std::log(s.koverH0_max / s.koverH0_min) * i / double(s.n_k - 1);
            std::array<DVector2D, 4> Dg;
            for (auto & d : Dg)
                d.assign(s.n_k, DVector(s.n_a));
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int ik = 0; ik < s.n_k; ik++) {
                auto d = solve_k(eq, std::exp(logkgrid[ik]), y_low);
                for (int i = 0; i < 4; i++)
                    Dg[i][ik] = std::move(d[i]);
            }
            const char * names2[4] = {"D1LPT(log(k/H0),log(a))",
                                      "D2LPT(log(k/H0),log(a))",
                                      "D3LPTa(log(k/H0),log(a))",
                                      "D3LPTb(log(k/H0),log(a))"};
            for (int i = 0; i < 4; i++)
                Dk[i].create(logkgrid, xgrid, Dg[i], names2[i]);
        }

        std::array<DVector, 4> solve_k(const LPTGrowthEquations & eq, double koverH0, double y_low) const {
            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                const double H = eq.E(x);
                const double dlogHdx = eq.dlogEdx(x);
                const double rhs = 1.5 * eq.matter_density(x) * call(eq.mu, x, koverH0) / (H * H);
                const double rhs_1LPT = rhs * call(eq.source_1, x, koverH0);
                const double rhs_2LPT = rhs * call(eq.source_2, x, koverH0);
                const double rhs_3LPTa = rhs * call(eq.source_3a, x, koverH0);
                const double rhs_3LPTb = rhs * call(eq.source_3b, x, koverH0);
                const double N2 = call(eq.second_order_source, x, koverH0);
                const double D1 = y[0], dD1dx = y[1], D2 = y[2], dD2dx = y[3];
                const double D3a = y[4], dD3adx = y[5], D3b = y[6], dD3bdx = y[7];
                dydx[0] = dD1dx;
                dydx[1] = rhs_1LPT * D1 - (2.0 + dlogHdx) * dD1dx;
                dydx[2] = dD2dx;
                dydx[3] = rhs_2LPT * (D2 - N2 * D1 * D1) - (2.0 + dlogHdx) * dD2dx;
                dydx[4] = dD3adx;
                dydx[5] = rhs_3LPTa * (D3a - 2.0 * D1 * D1 * D1) - (2.0 + dlogHdx) * dD3adx;
                dydx[6] = dD3bdx;
                dydx[7] = rhs_3LPTb * (D3b + D1 * D1 * D1 - D2 * D1) - (2.0 + dlogHdx) * dD3bdx;
                return GSL_SUCCESS;
            };

            std::array<double, 8> ini;
            if (not s.eds_initial_conditions and y_low < 0.5)
                ini = radiation_matter_growing_mode(y_low);
            else
                ini = {1.0, 1.0, -3.0 / 7.0, -6.0 / 7.0, -1.0 / 3.0, -1.0, 5.0 / 21.0, 5.0 / 7.0};
            if (eq.initial_rate_1 and koverH0 > 0.0)
                ini[1] = ini[0] * eq.initial_rate_1(xgrid.front(), koverH0);

            std::vector<double> yini(ini.begin(), ini.end());
            FML::SOLVERS::ODESOLVER::ODESolver ode;
            ode.solve(deriv, xgrid, yini);
            std::array<DVector, 4> D{ode.get_data_by_component(0),
                                     ode.get_data_by_component(2),
                                     ode.get_data_by_component(4),
                                     ode.get_data_by_component(6)};

            // Normalize to D1 = 1 at a_normalize
            Spline tmp;
            tmp.create(xgrid, D[0], "Temporary D1(loga) Spline");
            const double Dn = tmp(std::log(s.a_normalize));
            for (size_t i = 0; i < xgrid.size(); i++) {
                D[0][i] /= Dn;
                D[1][i] /= Dn * Dn;
                D[2][i] /= Dn * Dn * Dn;
                D[3][i] /= Dn * Dn * Dn;
            }
            return D;
        }
    };

} // namespace FML::COSMOLOGY::GROWTH

#endif
