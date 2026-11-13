#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_TABULATED_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_TABULATED_H

// =====================================================================================================================
// Tabulated dark energy: the standard species + a smooth dark energy given by a table, general relativity.
//
//   make_tabulated_E(p, z, E, r)   a table of H(z)/H0 (z = 0 included, E(0) = 1). The dark energy is what the table
//                                  needs beyond the standard species and the curvature:
//                                    rho_DE = E^2 - sum_i rho_i - Omega_k e^{-2x}
//                                  so inside the table E is the tabulated one. log E^2 is splined in x = ln a (and
//                                  continued past the end of the table by 6 points of the 4-point Lagrange polynomial
//                                  through the last 4, which gives a good spline derivative at z = 0); outside the
//                                  table rho_DE is the value at its end.
//   make_tabulated_w(p, a, w, r)   a table of w(a) (constant outside the table); rho_DE = Omega_DE exp(-3 int (1 + w)
//                                  dx) normalized by E(0) = 1 (trapezoidal integral on 25001 points in [ln 1e-10,
//                                  ln 10]; outside that w is constant).
// SimulationCode's Tabulated cosmology (formats H, CLASS and w) uses these. The tables need increasing x (decreasing z)
// for E and increasing a for w; duplicate points are an error.
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Cosmology/Background/ComponentBackground.h>
#include <FML/Cosmology/Background/FieldEquation_GeneralRelativity.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    class TabulatedDarkEnergySpecies : public Species {
        using Spline = FML::INTERPOLATION::SPLINE::GSLSpline;
        using SpeciesList = std::vector<std::shared_ptr<const Species>>;

      public:
        /// From a table of E = H/H0 at x = ln a (increasing); others: all the other species; Omega_k: the curvature
        static std::shared_ptr<TabulatedDarkEnergySpecies>
        from_E(const std::vector<double> & x, const std::vector<double> & E, SpeciesList others, double Omega_k) {
            const size_t n = x.size();
            if (n < 4 or E.size() != n)
                throw std::invalid_argument("[TabulatedDarkEnergy] Need 4 or more points of x and E");
            std::shared_ptr<TabulatedDarkEnergySpecies> s(new TabulatedDarkEnergySpecies());
            s->from_w_table = false;
            s->others = std::move(others);
            s->Omega_k = Omega_k;
            std::vector<double> logE2(n);
            for (size_t i = 0; i < n; i++) {
                if (i > 0 and not(x[i] > x[i - 1]))
                    throw std::invalid_argument("[TabulatedDarkEnergy] x = ln a must be strictly increasing");
                if (not(E[i] > 0.0))
                    throw std::invalid_argument("[TabulatedDarkEnergy] H must be > 0");
                logE2[i] = 2.0 * std::log(E[i]);
            }
            s->x_lo = x.front();
            s->x_hi = x.back();
            // Continue log E^2 past the end with the Lagrange polynomial through the last 4 points
            std::vector<double> xe(x), le(logE2);
            const double hx = x[n - 1] - x[n - 2];
            for (int j = 1; j <= 6; j++) {
                const double xj = x[n - 1] + j * hx;
                double value = 0.0;
                for (size_t i = n - 4; i < n; i++) {
                    double weight = 1.0;
                    for (size_t m = n - 4; m < n; m++)
                        if (m != i)
                            weight *= (xj - x[m]) / (x[i] - x[m]);
                    value += weight * logE2[i];
                }
                xe.push_back(xj);
                le.push_back(value);
            }
            s->logE2.create(xe, le, "TabulatedDarkEnergy log E^2(x)");
            s->rho_lo = std::exp(logE2.front()) - s->rho_others(s->x_lo);
            s->rho_hi = std::exp(logE2.back()) - s->rho_others(s->x_hi);
            return s;
        }

        /// From a table of w at a (increasing); Omega0: the density today
        static std::shared_ptr<TabulatedDarkEnergySpecies>
        from_w(const std::vector<double> & a, const std::vector<double> & w, double Omega0) {
            const size_t n = a.size();
            if (n < 4 or w.size() != n)
                throw std::invalid_argument("[TabulatedDarkEnergy] Need 4 or more points of a and w");
            std::shared_ptr<TabulatedDarkEnergySpecies> s(new TabulatedDarkEnergySpecies());
            s->from_w_table = true;
            std::vector<double> xw(n);
            for (size_t i = 0; i < n; i++) {
                if (not(a[i] > 0.0) or (i > 0 and not(a[i] > a[i - 1])))
                    throw std::invalid_argument("[TabulatedDarkEnergy] a must be positive and strictly increasing");
                xw[i] = std::log(a[i]);
            }
            s->xw_lo = xw.front();
            s->xw_hi = xw.back();
            s->w_table.create(xw, w, "TabulatedDarkEnergy w(x)");
            // log rho = log Omega0 + 3 (I(0) - I(x)), I(x) = int_x0^x (1 + w) dx' (trapezoidal)
            const int m = 25001;
            s->x_lo = std::log(1e-10);
            s->x_hi = std::log(10.0);
            std::vector<double> xg(m), I(m, 0.0), logrho(m);
            for (int i = 0; i < m; i++)
                xg[i] = s->x_lo + (s->x_hi - s->x_lo) * i / double(m - 1);
            for (int i = 1; i < m; i++)
                I[i] = I[i - 1] + 0.5 * (xg[i] - xg[i - 1]) * (2.0 + s->w_of_x(xg[i]) + s->w_of_x(xg[i - 1]));
            Spline Ispline(xg, I, "TabulatedDarkEnergy int (1 + w) dx");
            const double I0 = Ispline(0.0);
            for (int i = 0; i < m; i++)
                logrho[i] = std::log(Omega0) + 3.0 * (I0 - I[i]);
            s->logrho.create(xg, logrho, "TabulatedDarkEnergy log rho(x)");
            return s;
        }

        std::string name() const override { return from_w_table ? "dark_energy (w table)" : "dark_energy (H table)"; }
        Component component() const override { return Component::DarkEnergy; }

        double rho(double x) const override {
            if (from_w_table) {
                if (x < x_lo)
                    return std::exp(logrho(x_lo) - 3.0 * (1.0 + w_of_x(x)) * (x - x_lo));
                if (x > x_hi)
                    return std::exp(logrho(x_hi) - 3.0 * (1.0 + w_of_x(x)) * (x - x_hi));
                return std::exp(logrho(x));
            }
            if (x < x_lo)
                return rho_lo;
            if (x > x_hi)
                return rho_hi;
            return std::exp(logE2(x)) - rho_others(x);
        }
        double drhodx(double x) const override {
            if (from_w_table) {
                if (x < x_lo or x > x_hi)
                    return -3.0 * (1.0 + w_of_x(x)) * rho(x);
                return rho(x) * logrho.deriv_x(x);
            }
            if (x < x_lo or x > x_hi)
                return 0.0;
            return std::exp(logE2(x)) * logE2.deriv_x(x) - drho_others(x);
        }
        /// Analytic from the splines (a finite difference of drhodx with a fixed step mixes spline intervals)
        double d2rhodx2(double x) const override {
            if (from_w_table) {
                if (x < x_lo or x > x_hi)
                    return Species::d2rhodx2(x);
                const double d = logrho.deriv_x(x);
                return rho(x) * (logrho.deriv_xx(x) + d * d);
            }
            if (x < x_lo or x > x_hi)
                return 0.0;
            const double d = logE2.deriv_x(x);
            return std::exp(logE2(x)) * (logE2.deriv_xx(x) + d * d) - d2rho_others(x);
        }
        double p(double x) const override { return -rho(x) - drhodx(x) / 3.0; }
        /// The equation of state p/rho
        double w(double x) const { return p(x) / rho(x); }

      private:
        TabulatedDarkEnergySpecies() = default;
        bool from_w_table{false};
        double x_lo{0.0}, x_hi{0.0};
        // E table
        SpeciesList others;
        double Omega_k{0.0}, rho_lo{0.0}, rho_hi{0.0};
        Spline logE2;
        // w table
        double xw_lo{0.0}, xw_hi{0.0};
        Spline w_table, logrho;

        double w_of_x(double x) const { return w_table(std::min(std::max(x, xw_lo), xw_hi)); }
        double rho_others(double x) const {
            double r = Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->rho(x);
            return r;
        }
        double drho_others(double x) const {
            double r = -2.0 * Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->drhodx(x);
            return r;
        }
        double d2rho_others(double x) const {
            double r = 4.0 * Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->d2rhodx2(x);
            return r;
        }
    };

    /// Standard species + the dark energy of a table of H(z)/H0 (z increasing or decreasing, z = 0 included), GR
    inline ComponentBackgroundPtr make_tabulated_E(const BackgroundParameters & p,
                                                   const std::vector<double> & z,
                                                   const std::vector<double> & E,
                                                   BackgroundRange r = {}) {
        if (z.size() != E.size())
            throw std::invalid_argument("[make_tabulated_E] z and E differ in length");
        std::vector<std::pair<double, double>> t;
        for (size_t i = 0; i < z.size(); i++)
            t.push_back({-std::log1p(z[i]), E[i]});
        std::sort(t.begin(), t.end());
        std::vector<double> x, e;
        for (auto & v : t) {
            x.push_back(v.first);
            e.push_back(v.second);
        }
        auto s = standard_species(p);
        auto de = TabulatedDarkEnergySpecies::from_E(x, e, s, p.Omega_k);
        s.push_back(de);
        return std::make_shared<const ComponentBackground>("tabulated", p, s, std::make_shared<GeneralRelativity>(), r);
    }

    /// Standard species + a dark energy with w(a) from a table (Omega_DE from E(0) = 1), GR
    inline ComponentBackgroundPtr make_tabulated_w(const BackgroundParameters & p,
                                                   const std::vector<double> & a,
                                                   const std::vector<double> & w,
                                                   BackgroundRange r = {}) {
        auto s = standard_species(p);
        const double OmegaDE = 1.0 - p.Omega_k - density_today(s);
        s.push_back(TabulatedDarkEnergySpecies::from_w(a, w, OmegaDE));
        return std::make_shared<const ComponentBackground>("tabulated", p, s, std::make_shared<GeneralRelativity>(), r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
