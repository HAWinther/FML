#ifndef FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_HORNDESKITABLE_H
#define FML_COSMOLOGY_BACKGROUND_BACKGROUNDMODEL_HORNDESKITABLE_H

// =====================================================================================================================
// A reduced Horndeski theory (K(phi, X), G3(phi, X), G4(phi)) given by the table of
// SimulationCode/scripts/horndeski_frontend.py (the model horndeski_table; BackgroundParameters::model_file): the
// background from its H(a), as a tabulated dark energy on the standard species (BackgroundModel_Tabulated.h:
// rho_DE = E^2 - the other species, GR sum; outside the table rho_DE is the value at its end, so the table should
// start where the scalar is negligible, deep in the radiation era: the frontend's default a = 1e-8). The perturbations
// (PerturbationModel_HorndeskiTable.h) use its alphas.
//
// The table: comment lines start with '#'; the second line holds "Omega_m = ..., Omega_r = ..., Omega_Lambda = ...";
// rows "a E dlnE/dlna GG4 mu Sigma b5 c2 phi dphi/dlna M*^2 M*^2-1 alpha_K alpha_B alpha_M dalpha_K/dlna
// dalpha_B/dlna Rx" with increasing a. The frontend works without massive neutrinos, flat, with dust, radiation
// (photons and massless neutrinos) and Omega_Lambda: Omega_m and Omega_r of the table must be those of the parameters
// (relative difference 1e-6; Omega_r: compute it with the same h, T_CMB and N_eff).
// =====================================================================================================================

#include <FML/Cosmology/Background/BackgroundModel_Tabulated.h>
#include <FML/Cosmology/Background/BackgroundParameters.h>

#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    struct HorndeskiTable {
        static constexpr int ncols = 18;
        std::string filename;
        std::map<std::string, double> header; // the "name = value" of the comment lines (Omega_m, Omega_r, ...)
        std::vector<double> col[ncols];       // the columns (see the header of this file)
        const std::vector<double> & a() const { return col[0]; }
        const std::vector<double> & E() const { return col[1]; }
        bool has_alphas() const { return not col[ncols - 1].empty(); }
    };

    /// Read the table of horndeski_frontend.py (the first 8 columns are needed, the alphas are optional)
    inline HorndeskiTable read_horndeski_table(const std::string & filename) {
        HorndeskiTable t;
        t.filename = filename;
        std::ifstream fp(filename);
        if (not fp.is_open())
            throw std::runtime_error("[HorndeskiTable] Cannot open [" + filename + "]");
        std::string line;
        const std::regex kv(R"(([A-Za-z_][A-Za-z0-9_]*) = ([-+0-9.eE]+))");
        int ncols = -1;
        while (std::getline(fp, line)) {
            if (line.empty())
                continue;
            if (line[0] == '#') {
                for (std::sregex_iterator it(line.begin(), line.end(), kv), end; it != end; ++it) {
                    try {
                        size_t used = 0;
                        const std::string v = (*it)[2];
                        const double d = std::stod(v, &used);
                        if (used == v.size())
                            t.header[(*it)[1]] = d;
                    } catch (const std::exception &) {
                        // not a number (the expressions K = ..., G3 = ...)
                    }
                }
                continue;
            }
            std::istringstream is(line);
            std::vector<double> v;
            double x;
            while (is >> x)
                v.push_back(x);
            if (ncols < 0)
                ncols = int(v.size());
            if (int(v.size()) != ncols or (ncols != 10 and ncols != HorndeskiTable::ncols))
                throw std::runtime_error("[HorndeskiTable] Bad line in [" + filename + "] (need 10 or " +
                                         std::to_string(HorndeskiTable::ncols) + " columns): " + line);
            for (int i = 0; i < ncols; i++)
                t.col[i].push_back(v[i]);
        }
        const auto & a = t.a();
        if (a.size() < 4)
            throw std::runtime_error("[HorndeskiTable] [" + filename + "] has fewer than 4 rows");
        for (size_t i = 1; i < a.size(); i++)
            if (not(a[i] > a[i - 1]) or not(a[i - 1] > 0.0))
                throw std::runtime_error("[HorndeskiTable] [" + filename + "] needs increasing a > 0");
        return t;
    }

    /// The background of the table on the standard species of p (checks that the table was made for p)
    inline ComponentBackgroundPtr make_horndeski_table(const BackgroundParameters & p, BackgroundRange r = {}) {
        const std::string msg = "[horndeski_table] ";
        if (p.model_file.empty())
            throw std::invalid_argument(msg + "needs model_file: the _gravity.txt table of horndeski_frontend.py");
        const auto t = read_horndeski_table(p.model_file);
        if (not p.m_nu_eV.empty() or p.Omega_k != 0.0)
            throw std::invalid_argument(msg + "only without massive neutrinos and flat (as the frontend)");
        auto require = [&](const std::string & name, double value) {
            const auto it = t.header.find(name);
            if (it == t.header.end())
                throw std::invalid_argument(msg + "[" + p.model_file + "] has no " + name + " in its header");
            if (std::fabs(it->second / value - 1.0) > 1e-6)
                throw std::invalid_argument(msg + "[" + p.model_file + "] was made for " + name + " = " +
                                            std::to_string(it->second) + ", the cosmology has " +
                                            std::to_string(value) +
                                            " (run horndeski_frontend.py with the Omega_m, h, T_CMB, Neff of the run)");
        };
        require("Omega_m", p.Omega_b + p.Omega_cdm);
        require("Omega_r", p.Omega_gamma() + p.Omega_nu_massless());
        if (t.a().front() > 1e-6)
            throw std::invalid_argument(msg + "[" + p.model_file + "] starts at a = " + std::to_string(t.a().front()) +
                                        " (it should start deep in the radiation era: a_min = 1e-8 by default)");
        std::vector<double> z, E;
        for (size_t i = 0; i < t.a().size(); i++) {
            z.push_back(1.0 / t.a()[i] - 1.0);
            E.push_back(t.E()[i]);
        }
        return make_tabulated_E(p, z, E, r);
    }

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
