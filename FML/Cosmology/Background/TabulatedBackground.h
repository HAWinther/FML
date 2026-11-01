#ifndef FML_COSMOLOGY_BACKGROUND_TABULATEDBACKGROUND_H
#define FML_COSMOLOGY_BACKGROUND_TABULATEDBACKGROUND_H

// =====================================================================================================================
// TabulatedBackground: a Background from tables of E(x) and Omega_i(x) (splines in x = ln a; dlogE/dx from the
// derivative of the spline of log E unless it is tabulated). Components without a table have Omega = 0.
//
// TabulatedBackground::from_class(filename) reads the background file of a CLASS run (write_background = yes,
// <root>background.dat): H, the densities (.)rho_g, _b, _cdm, _ncdm[i], _ur, _lambda, _fld and (.)rho_crit. The
// parameters are inferred from it: h from H(z = 0), Omega_i today, T_CMB from Omega_gamma h^2, N_eff from the
// neutrino-to-photon ratio at the earliest time, and the neutrino masses from Omega_ncdm[i] today (assuming CLASS's
// default T_ncdm).
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>
#include <FML/Cosmology/Background/Species.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    class TabulatedBackground : public Background {
      public:
        struct Tables {
            std::vector<double> x;       ///< ln a, increasing
            std::vector<double> E;       ///< H / H0
            std::vector<double> dlogEdx; ///< optional
            std::map<Component, std::vector<double>> Omega;
        };

        /// range: the default (x_min = x_max = 0) is the range of the tables
        TabulatedBackground(std::string name,
                            BackgroundParameters params,
                            const Tables & t,
                            BackgroundRange r = {0.0, 0.0, 200})
            : label(std::move(name)), params(std::move(params)) {
            const std::string msg = "[" + label + "] ";
            this->params.validate();
            const size_t n = t.x.size();
            if (n < 4 or t.E.size() != n)
                throw std::invalid_argument(msg + "Need 4 or more points of x and E");
            for (size_t i = 1; i < n; i++)
                if (not(t.x[i] > t.x[i - 1]))
                    throw std::invalid_argument(msg + "x must be increasing");
            std::vector<double> logE(n);
            for (size_t i = 0; i < n; i++) {
                if (not(t.E[i] > 0.0))
                    throw std::invalid_argument(msg + "E must be positive");
                logE[i] = std::log(t.E[i]);
            }
            logE_spline.create(t.x, logE, label + " log E(x)");
            if (not t.dlogEdx.empty()) {
                if (t.dlogEdx.size() != n)
                    throw std::invalid_argument(msg + "dlogEdx needs one value per x");
                dlogE_spline.create(t.x, t.dlogEdx, label + " dlogE/dx(x)");
                have_dlogE = true;
            }
            for (const auto & [c, values] : t.Omega) {
                if (values.size() != n)
                    throw std::invalid_argument(msg + "Omega(" + component_name(c) + ") needs one value per x");
                Omega_splines[c].create(t.x, values, label + " Omega_" + component_name(c));
            }
            if (r.x_min == 0.0 and r.x_max == 0.0) {
                r.x_min = t.x.front();
                r.x_max = t.x.back();
            }
            if (r.x_min < t.x.front() - 1e-12 or r.x_max > t.x.back() + 1e-12)
                throw std::invalid_argument(msg + "The range is larger than the tables");
            range = r;
            if (not(std::abs(E(0.0) - 1.0) < 1e-6))
                throw std::invalid_argument(msg + "E(0) = " + std::to_string(E(0.0)) + ", not 1");
            finalize();
        }

        std::string name() const override { return label; }
        const BackgroundParameters & parameters() const override { return params; }
        double E(double x) const override { return std::exp(logE_spline(x)); }
        double dlogEdx(double x) const override { return have_dlogE ? dlogE_spline(x) : logE_spline.deriv_x(x); }
        double Omega(Component c, double x) const override {
            auto it = Omega_splines.find(c);
            return it == Omega_splines.end() ? 0.0 : it->second(x);
        }

        /// From the background file of a CLASS run
        static std::shared_ptr<const TabulatedBackground> from_class(const std::string & filename,
                                                                     int points_per_unit_x = 200) {
            const std::string msg = "[TabulatedBackground::from_class] ";
            std::ifstream in(filename);
            if (not in)
                throw std::runtime_error(msg + "Cannot open [" + filename + "]");
            std::string line, header;
            std::vector<std::vector<double>> rows;
            while (std::getline(in, line)) {
                if (line.empty())
                    continue;
                if (line[0] == '#') {
                    header = line;
                    continue;
                }
                std::istringstream s(line);
                std::vector<double> row;
                double v;
                while (s >> v)
                    row.push_back(v);
                if (not row.empty())
                    rows.push_back(row);
            }
            // The last header line: "1:z   2:proper time [Gyr]   3:conf. time [Mpc]   4:H [1/Mpc] ..."
            std::vector<std::string> labels;
            const std::regex number(R"((^|\s)(\d+):)");
            std::vector<std::pair<size_t, size_t>> pos; // start of the label, start of the match
            for (auto it = std::sregex_iterator(header.begin(), header.end(), number); it != std::sregex_iterator();
                 ++it)
                pos.push_back({size_t(it->position(0) + it->length(0)), size_t(it->position(0))});
            for (size_t i = 0; i < pos.size(); i++) {
                const size_t end = i + 1 < pos.size() ? pos[i + 1].second : header.size();
                std::string l = header.substr(pos[i].first, end - pos[i].first);
                l.erase(l.find_last_not_of(" \t") + 1);
                labels.push_back(l);
            }
            if (labels.empty() or rows.empty())
                throw std::runtime_error(msg + "No columns in [" + filename + "]");
            for (auto & r : rows)
                if (r.size() != labels.size())
                    throw std::runtime_error(msg + "Rows and header of [" + filename + "] differ in length");
            const auto find = [&](const std::string & key) -> int {
                for (size_t i = 0; i < labels.size(); i++)
                    if (labels[i] == key or labels[i].rfind(key + " ", 0) == 0)
                        return int(i);
                return -1;
            };
            const auto column = [&](const std::string & key) {
                const int i = find(key);
                if (i < 0)
                    throw std::runtime_error(msg + "No column [" + key + "] in [" + filename + "]");
                std::vector<double> c(rows.size());
                for (size_t j = 0; j < rows.size(); j++)
                    c[j] = rows[j][i];
                return c;
            };
            const auto optional = [&](const std::string & key) {
                return find(key) >= 0 ? column(key) : std::vector<double>(rows.size(), 0.0);
            };

            // The file runs from high z to z = 0
            const auto z = column("z"), H = column("H"), crit = column("(.)rho_crit");
            const size_t n = z.size(), last = n - 1;
            if (std::abs(z[last]) > 1e-12)
                throw std::runtime_error(msg + "The last row of [" + filename + "] is not z = 0");
            int nncdm = 0;
            while (find("(.)rho_ncdm[" + std::to_string(nncdm) + "]") >= 0)
                nncdm++;
            std::vector<std::vector<double>> ncdm;
            for (int i = 0; i < nncdm; i++)
                ncdm.push_back(column("(.)rho_ncdm[" + std::to_string(i) + "]"));
            const auto g = column("(.)rho_g"), b = column("(.)rho_b"), cdm = optional("(.)rho_cdm"),
                       ur = optional("(.)rho_ur"), lambda = optional("(.)rho_lambda"), fld = optional("(.)rho_fld"),
                       tot = column("(.)rho_tot");

            Tables t;
            for (auto c : {Component::Baryons,
                           Component::Photons,
                           Component::CDM,
                           Component::MasslessNeutrinos,
                           Component::MassiveNeutrinos,
                           Component::DarkEnergy,
                           Component::Curvature})
                t.Omega[c].resize(n);
            t.x.resize(n);
            t.E.resize(n);
            for (size_t j = 0; j < n; j++) {
                const size_t i = j; // the file has increasing x
                t.x[j] = -std::log1p(z[i]);
                t.E[j] = H[i] / H[last];
                double nu = 0.0;
                for (auto & c : ncdm)
                    nu += c[i];
                t.Omega[Component::Baryons][j] = b[i] / crit[i];
                t.Omega[Component::Photons][j] = g[i] / crit[i];
                t.Omega[Component::CDM][j] = cdm[i] / crit[i];
                t.Omega[Component::MasslessNeutrinos][j] = ur[i] / crit[i];
                t.Omega[Component::MassiveNeutrinos][j] = nu / crit[i];
                t.Omega[Component::DarkEnergy][j] = (lambda[i] + fld[i]) / crit[i];
                t.Omega[Component::Curvature][j] = 1.0 - tot[i] / crit[i];
            }

            BackgroundParameters p;
            p.h = H[last] * FML::UTILS::SI::c_over_H0_mpch;
            p.Omega_b = b[last] / crit[last];
            p.Omega_cdm = cdm[last] / crit[last];
            p.Omega_k = 1.0 - tot[last] / crit[last];
            if (std::abs(p.Omega_k) < 1e-10)
                p.Omega_k = 0.0;
            {
                BackgroundParameters ref;
                ref.h = p.h;
                ref.T_CMB = 2.7255;
                p.T_CMB = ref.T_CMB * std::pow(g[last] / crit[last] / ref.Omega_gamma(), 0.25);
            }
            double nu0 = ur[0];
            for (auto & c : ncdm)
                nu0 += c[0];
            p.N_eff = nu0 / g[0] * 8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0);
            for (int i = 0; i < nncdm; i++) {
                // The mass with Omega_ncdm[i] today (bisection in log m)
                const double target = ncdm[i][last] / crit[last];
                double lo = std::log(1e-6), hi = std::log(1e3);
                for (int it = 0; it < 200; it++) {
                    const double mid = 0.5 * (lo + hi);
                    (MassiveNeutrinoSpecies(std::exp(mid), p).rho(0.0) > target ? hi : lo) = mid;
                }
                p.m_nu_eV.push_back(std::exp(0.5 * (lo + hi)));
            }
            BackgroundRange r{t.x.front(), t.x.back(), points_per_unit_x};
            return std::make_shared<const TabulatedBackground>("class:" + filename, p, t, r);
        }

      private:
        using Spline = FML::INTERPOLATION::SPLINE::GSLSpline;
        std::string label;
        BackgroundParameters params;
        Spline logE_spline, dlogE_spline;
        bool have_dlogE{false};
        std::map<Component, Spline> Omega_splines;
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
