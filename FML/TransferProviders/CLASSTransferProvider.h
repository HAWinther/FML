#ifndef FML_CLASS_TRANSFER_PROVIDER_H
#define FML_CLASS_TRANSFER_PROVIDER_H

#include <FML/InitialConditions/TransferProvider.h>
#include <FML/TransferProviders/GaugeView.h>
#include <FML/TransferProviders/TabulatedTransferProvider.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

//=====================================================================================================================
// Transfer functions from native CLASS output (format = class, output = mTk[, vTk], one file per redshift in z_pk),
// read by column name. CLASS normalizes to unit initial curvature; this provider returns the FML/CAMB sign convention
// (delta_provider = -d_i, velocity_provider = -t_i / calH), the same as CAMBFileTransferProvider.
//
// The gauge of the files (CLASS's gauge = synchronous, the default, or gauge = newtonian) must be declared; it is
// checked against the columns (a synchronous-gauge run has no t_cdm column since CDM is at rest, a Newtonian run has
// one). Newtonian-gauge densities can be returned in the CDM-synchronous gauge (output_gauge): they are converted at
// the redshifts of the files with the Newtonian CDM velocity (see GaugeView.h for the formula), so that the
// interpolation in a and the growth rates act on the converted (smooth) densities. Velocities stay in the gauge of
// the files.
//
// TotalMatter is the rho-weighted CDM + baryon + massive neutrino density in the gauge of the provider. CLASS's own
// d_m column is not used: it is the gauge-invariant (comoving) matter density, not the density in the run's gauge.
// Interpolation, growth rates and extrapolation: TabulatedTransferProvider.h.
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    namespace detail {

        /// A CLASS output table: the column names (from the "#  1:k (h/Mpc)  2:d_g ..." header line), the header
        /// lines and the data rows
        struct ClassTable {
            std::vector<std::string> names;
            std::vector<std::string> header;
            std::vector<std::vector<double>> rows;

            /// The column with this name (exact, or the name followed by a space or '[', e.g. "k" for "k (h/Mpc)").
            /// -1 if not there
            int column(const std::string & name) const {
                for (size_t i = 0; i < names.size(); i++)
                    if (names[i] == name)
                        return int(i);
                for (size_t i = 0; i < names.size(); i++)
                    if (names[i].size() > name.size() and names[i].compare(0, name.size(), name) == 0 and
                        (names[i][name.size()] == ' ' or names[i][name.size()] == '['))
                        return int(i);
                return -1;
            }
            int require_column(const std::string & name, const std::string & filename) const {
                const int col = column(name);
                if (col < 0)
                    throw std::runtime_error("[CLASSTransferProvider] Column [" + name + "] not found in [" + filename +
                                             "]");
                return col;
            }
            std::vector<double> get(int col) const {
                std::vector<double> values(rows.size());
                for (size_t i = 0; i < rows.size(); i++)
                    values[i] = rows[i][col];
                return values;
            }
        };

        inline ClassTable read_class_table(const std::string & filename) {
            std::ifstream fp(filename);
            if (not fp)
                throw std::runtime_error("[CLASSTransferProvider] Cannot open [" + filename + "]");
            ClassTable table;
            const std::regex first_column(R"(^#\s*1:)");
            const std::regex name_regex(R"((\d+):\s*(.*?)\s*(?=\s\d+:|$))");
            std::string line;
            while (std::getline(fp, line)) {
                if (line.find_first_not_of(" \t\r") == std::string::npos)
                    continue;
                if (line[0] == '#') {
                    table.header.push_back(line);
                    if (std::regex_search(line, first_column)) {
                        table.names.clear();
                        const std::string text = line.substr(1);
                        for (auto it = std::sregex_iterator(text.begin(), text.end(), name_regex);
                             it != std::sregex_iterator();
                             ++it) {
                            const int number = std::stoi((*it)[1]);
                            if (number != int(table.names.size()) + 1)
                                throw std::runtime_error("[CLASSTransferProvider] Cannot read the column names in [" +
                                                         filename + "]");
                            table.names.push_back((*it)[2]);
                        }
                    }
                    continue;
                }
                if (table.names.empty())
                    throw std::runtime_error("[CLASSTransferProvider] No column names (#  1:...) before the data in [" +
                                             filename + "]");
                std::istringstream ss(line);
                std::vector<double> row;
                row.reserve(table.names.size());
                std::string token;
                while (ss >> token)
                    row.push_back(std::stod(token));
                if (row.size() != table.names.size())
                    throw std::runtime_error("[CLASSTransferProvider] A line in [" + filename + "] has " +
                                             std::to_string(row.size()) + " columns, the header names " +
                                             std::to_string(table.names.size()));
                table.rows.push_back(std::move(row));
            }
            if (table.rows.empty())
                throw std::runtime_error("[CLASSTransferProvider] No data in [" + filename + "]");
            return table;
        }

        /// The redshift of a CLASS transfer file, from "... at redshift z=..." in its header
        inline double class_table_redshift(const ClassTable & table, const std::string & filename) {
            const std::regex z_regex(R"(at redshift z=\s*([-+0-9.eE]+))");
            std::smatch match;
            for (auto & line : table.header)
                if (std::regex_search(line, match, z_regex))
                    return std::stod(match[1]);
            throw std::runtime_error("[CLASSTransferProvider] No 'at redshift z=' in the header of [" + filename + "]");
        }

        /// The number of massive neutrino species (the columns name[0], name[1], ...)
        inline int count_indexed_columns(const ClassTable & table, const std::string & name) {
            int n = 0;
            while (table.column(name + "[" + std::to_string(n) + "]") >= 0)
                n++;
            return n;
        }
    } // namespace detail

    //=================================================================================================================
    /// The background of a CLASS run (the file written with write_background = yes): H(a) and the densities and
    /// pressures needed for the gauge transformation and the species weights
    //=================================================================================================================
    class CLASSBackground {
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        Spline H_of_loga;          // H in 1/Mpc
        Spline rho_cdm_of_loga;    // (8 pi G / 3) rho in 1/Mpc^2
        Spline rho_b_of_loga;
        std::vector<Spline> rho_ncdm_of_loga;
        std::vector<Spline> p_ncdm_of_loga;
        double H0_mpc{0.0};

      public:
        CLASSBackground() = default;
        explicit CLASSBackground(const std::string & filename) {
            const auto table = detail::read_class_table(filename);
            const auto z = table.get(table.require_column("z", filename));
            const int nncdm = detail::count_indexed_columns(table, "(.)rho_ncdm");
            // The file runs from large z to z = 0: log(a) increasing
            std::vector<double> loga(z.size());
            for (size_t i = 0; i < z.size(); i++)
                loga[i] = -std::log1p(z[i]);
            for (size_t i = 1; i < loga.size(); i++)
                if (not(loga[i] > loga[i - 1]))
                    throw std::runtime_error("[CLASSBackground] The redshifts in [" + filename + "] are not decreasing");
            auto spline = [&](const std::string & name) {
                return Spline(loga, table.get(table.require_column(name, filename)), name + "(log a)");
            };
            H_of_loga = spline("H");
            rho_cdm_of_loga = spline("(.)rho_cdm");
            rho_b_of_loga = spline("(.)rho_b");
            for (int i = 0; i < nncdm; i++) {
                rho_ncdm_of_loga.push_back(spline("(.)rho_ncdm[" + std::to_string(i) + "]"));
                p_ncdm_of_loga.push_back(spline("(.)p_ncdm[" + std::to_string(i) + "]"));
            }
            H0_mpc = H_of_loga(0.0);
        }

        /// The Hubble parameter today in units of 100 km/s/Mpc (CLASS: H0 in 1/Mpc times c)
        double h() const { return H0_mpc * 2997.92458; }
        /// H(a)/H0
        double E(double a) const { return H_of_loga(std::log(a)) / H0_mpc; }
        int n_ncdm() const { return int(rho_ncdm_of_loga.size()); }
        double rho_cdm(double a) const { return rho_cdm_of_loga(std::log(a)); }
        double rho_b(double a) const { return rho_b_of_loga(std::log(a)); }
        double rho_ncdm(int i, double a) const { return rho_ncdm_of_loga.at(i)(std::log(a)); }
        double p_ncdm(int i, double a) const { return p_ncdm_of_loga.at(i)(std::log(a)); }

        /// (rho + p) / rho of a species (1 for CDM and baryons; the sum over the massive neutrinos; the total of
        /// CDM, baryons and massive neutrinos for TotalMatter)
        double one_plus_w(Species species, double a) const {
            if (species == Species::CDM or species == Species::Baryon or species == Species::CDMBaryon)
                return 1.0;
            double rho = 0.0, rho_plus_p = 0.0;
            for (int i = 0; i < n_ncdm(); i++) {
                rho += rho_ncdm(i, a);
                rho_plus_p += rho_ncdm(i, a) + p_ncdm(i, a);
            }
            if (species == Species::TotalMatter) {
                rho += rho_cdm(a) + rho_b(a);
                rho_plus_p += rho_cdm(a) + rho_b(a);
            }
            if (not(rho > 0.0))
                throw std::invalid_argument("[CLASSBackground] No massive neutrinos in the background");
            return rho_plus_p / rho;
        }

        /// The background functions needed by a GaugeView
        GaugeBackground gauge_background() const {
            auto self = std::make_shared<CLASSBackground>(*this);
            GaugeBackground bg;
            bg.E_of_a = [self](double a) { return self->E(a); };
            bg.one_plus_w = [self](Species s, double a) { return self->one_plus_w(s, a); };
            return bg;
        }
    };

    //=================================================================================================================
    /// Transfer functions from native CLASS output files (see the top of this file)
    //=================================================================================================================
    class CLASSTransferProvider final : public TabulatedTransferProvider {
      public:
        struct Settings {
            /// The transfer files, one per redshift, any order (CLASS: <root>z1_tk.dat, <root>z2_tk.dat, ...)
            std::vector<std::string> transfer_files;
            /// The background file of the same run (<root>background.dat, needs write_background = yes)
            std::string background_file;
            /// The gauge CLASS was run in (gauge = synchronous or newtonian): checked against the columns
            Gauge gauge{Gauge::SynchronousCDM};
            /// The density gauge returned: the same as gauge, or SynchronousCDM for a Newtonian run (converted)
            Gauge output_gauge{Gauge::SynchronousCDM};
            /// The primordial spectrum of the run: P_R(k) = A_s (k / k_pivot)^(n_s - 1)
            double A_s{0.0};
            double n_s{1.0};
            double k_pivot_mpc{0.05};
            RangePolicy range_policy{RangePolicy::Reject};
        };

      private:
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;
        CLASSBackground bg;

      public:
        explicit CLASSTransferProvider(const Settings & settings) {
            if (settings.gauge != Gauge::SynchronousCDM and settings.gauge != Gauge::Newtonian)
                throw std::invalid_argument("[CLASSTransferProvider] The gauge must be SynchronousCDM or Newtonian");
            if (settings.output_gauge != settings.gauge and
                not(settings.gauge == Gauge::Newtonian and settings.output_gauge == Gauge::SynchronousCDM))
                throw std::invalid_argument("[CLASSTransferProvider] Only Newtonian -> CDM-synchronous conversion is "
                                            "supported: run CLASS with gauge = newtonian");
            if (settings.transfer_files.empty())
                throw std::invalid_argument("[CLASSTransferProvider] No transfer files");
            bg = CLASSBackground(settings.background_file);

            // Read the files and sort them by scale factor
            struct Entry {
                double a;
                std::string filename;
                detail::ClassTable table;
            };
            std::vector<Entry> entries;
            for (auto & filename : settings.transfer_files) {
                auto table = detail::read_class_table(filename);
                const double z = detail::class_table_redshift(table, filename);
                entries.push_back({1.0 / (1.0 + z), filename, std::move(table)});
            }
            std::sort(entries.begin(), entries.end(), [](const Entry & x, const Entry & y) { return x.a < y.a; });
            for (size_t i = 1; i < entries.size(); i++)
                if (not(entries[i].a > entries[i - 1].a))
                    throw std::invalid_argument("[CLASSTransferProvider] Two transfer files have the same redshift");

            // The k-grid (the same in all files) and the columns
            const auto & first = entries.front();
            const DVector k = first.table.get(first.table.require_column("k", first.filename));
            const int nncdm = detail::count_indexed_columns(first.table, "d_ncdm");
            if (nncdm != bg.n_ncdm())
                throw std::invalid_argument("[CLASSTransferProvider] The transfer and background files have different "
                                            "numbers of massive neutrino species");
            const bool has_t_cdm = first.table.column("t_cdm") >= 0;
            if (settings.gauge == Gauge::SynchronousCDM and has_t_cdm)
                throw std::invalid_argument("[CLASSTransferProvider] Declared synchronous gauge, but the files have a "
                                            "t_cdm column (CLASS writes it only in the Newtonian gauge)");
            // The gauge is checked as far as the columns allow: a Newtonian run with velocities (vTk) has t_cdm,
            // a synchronous run never has it; with extra_metric_transfer_functions a synchronous run has h and eta
            const bool has_sync_metric = first.table.column("h") >= 0 or first.table.column("eta") >= 0;
            if (settings.gauge == Gauge::Newtonian and has_sync_metric)
                throw std::invalid_argument("[CLASSTransferProvider] Declared Newtonian gauge, but the files have the "
                                            "synchronous-gauge metric h, eta (run CLASS with gauge = newtonian)");
            if (settings.gauge == Gauge::Newtonian and first.table.column("t_b") >= 0 and not has_t_cdm)
                throw std::invalid_argument("[CLASSTransferProvider] Declared Newtonian gauge, but the files have "
                                            "velocities without t_cdm: this is a synchronous-gauge run");
            if (settings.output_gauge != settings.gauge and not has_t_cdm)
                throw std::invalid_argument("[CLASSTransferProvider] Converting Newtonian-gauge densities needs the "
                                            "CDM velocity t_cdm: run CLASS with output = mTk, vTk");
            // In the synchronous gauge the CDM velocity is 0 (CDM at rest), known when the run has velocities (t_b)
            const bool have_v_cdm = has_t_cdm or (settings.gauge == Gauge::SynchronousCDM and first.table.column("t_b") >= 0);
            const bool have_v_b = first.table.column("t_b") >= 0;
            const bool have_v_ncdm = nncdm > 0 and first.table.column("t_ncdm[0]") >= 0;

            const size_t na = entries.size(), nk = k.size();
            TransferTables t;
            t.name = "CLASSTransferProvider";
            t.h = bg.h();
            t.k_hmpc = k;
            t.a.resize(na);
            t.density_gauge = settings.output_gauge;
            t.velocity_gauge = settings.gauge;
            t.range_policy = settings.range_policy;
            t.A_s = settings.A_s;
            t.n_s = settings.n_s;
            t.k_pivot_mpc = settings.k_pivot_mpc;
            const auto table = [&](bool have) { return have ? DVector2D(na, DVector(nk, 0.0)) : DVector2D{}; };
            const bool have_v_cb = have_v_cdm and have_v_b, have_v_m = have_v_cb and (nncdm == 0 or have_v_ncdm);
            DVector2D dc = table(true), db = table(true), dn = table(nncdm > 0), dcb = table(true), dm = table(true);
            DVector2D vc = table(have_v_cdm), vb = table(have_v_b), vn = table(have_v_ncdm), vcb = table(have_v_cb),
                      vm = table(have_v_m);
            for (size_t ia = 0; ia < na; ia++) {
                const auto & e = entries[ia];
                const double a = e.a;
                t.a[ia] = a;
                if (e.table.rows.size() != nk)
                    throw std::invalid_argument("[CLASSTransferProvider] The files have different k-grids");
                const auto kk = e.table.get(e.table.require_column("k", e.filename));
                for (size_t ik = 0; ik < nk; ik++)
                    if (std::abs(kk[ik] / k[ik] - 1.0) > 1e-6)
                        throw std::invalid_argument("[CLASSTransferProvider] The files have different k-grids");
                // calH = a H in 1/Mpc (CLASS: conformal time in Mpc, so theta is in 1/Mpc)
                const double calH = a * bg.E(a) * t.h / 2997.92458;
                const auto get = [&](const std::string & name) { return e.table.get(e.table.require_column(name, e.filename)); };
                const auto cdm = get("d_cdm"), b = get("d_b");
                for (size_t ik = 0; ik < nk; ik++) {
                    dc[ia][ik] = -cdm[ik];
                    db[ia][ik] = -b[ik];
                }
                // Massive neutrinos: density weighted by rho_i, velocity by rho_i + p_i (as d_m and t_tot in CLASS)
                double rho_nu = 0.0, rho_plus_p_nu = 0.0;
                for (int i = 0; i < nncdm; i++) {
                    rho_nu += bg.rho_ncdm(i, a);
                    rho_plus_p_nu += bg.rho_ncdm(i, a) + bg.p_ncdm(i, a);
                }
                for (int i = 0; i < nncdm; i++) {
                    const auto dd = get("d_ncdm[" + std::to_string(i) + "]");
                    const double w = bg.rho_ncdm(i, a) / rho_nu;
                    for (size_t ik = 0; ik < nk; ik++)
                        dn[ia][ik] -= w * dd[ik];
                    if (have_v_ncdm) {
                        const auto tt = get("t_ncdm[" + std::to_string(i) + "]");
                        const double wv = (bg.rho_ncdm(i, a) + bg.p_ncdm(i, a)) / rho_plus_p_nu;
                        for (size_t ik = 0; ik < nk; ik++)
                            vn[ia][ik] -= wv * tt[ik] / calH;
                    }
                }
                if (has_t_cdm) {
                    const auto tt = get("t_cdm");
                    for (size_t ik = 0; ik < nk; ik++)
                        vc[ia][ik] = -tt[ik] / calH;
                }
                if (have_v_b) {
                    const auto tt = get("t_b");
                    for (size_t ik = 0; ik < nk; ik++)
                        vb[ia][ik] = -tt[ik] / calH;
                }

                // Newtonian -> CDM-synchronous at this redshift: delta_s += 3 (1 + w_s) (calH / c k)^2 V_cdm
                if (settings.output_gauge != settings.gauge) {
                    const double one_plus_w_nu = nncdm > 0 ? bg.one_plus_w(Species::MassiveNeutrino, a) : 1.0;
                    for (size_t ik = 0; ik < nk; ik++) {
                        const double x = a * bg.E(a) / (2997.92458 * k[ik]);
                        const double shift = 3.0 * x * x * vc[ia][ik];
                        dc[ia][ik] += shift;
                        db[ia][ik] += shift;
                        if (nncdm > 0)
                            dn[ia][ik] += one_plus_w_nu * shift;
                    }
                }

                // CDM + baryons and total matter (CDM + baryons + massive neutrinos): densities weighted by rho,
                // velocities by rho + p, in the gauge of the provider. CLASS's own d_m column is not used: it is the
                // gauge-invariant (comoving) matter density
                const double rc = bg.rho_cdm(a), rb = bg.rho_b(a);
                for (size_t ik = 0; ik < nk; ik++) {
                    const double dnu = nncdm > 0 ? dn[ia][ik] : 0.0;
                    dcb[ia][ik] = (rc * dc[ia][ik] + rb * db[ia][ik]) / (rc + rb);
                    dm[ia][ik] = (rc * dc[ia][ik] + rb * db[ia][ik] + rho_nu * dnu) / (rc + rb + rho_nu);
                    if (have_v_cb)
                        vcb[ia][ik] = (rc * vc[ia][ik] + rb * vb[ia][ik]) / (rc + rb);
                    if (have_v_m) {
                        const double vnu = nncdm > 0 ? vn[ia][ik] : 0.0;
                        vm[ia][ik] = (rc * vc[ia][ik] + rb * vb[ia][ik] + rho_plus_p_nu * vnu) / (rc + rb + rho_plus_p_nu);
                    }
                }
            }
            const auto set = [&](Species s, DVector2D & dens, DVector2D & vel) {
                t.density[TransferTables::index(s)] = std::move(dens);
                t.velocity[TransferTables::index(s)] = std::move(vel);
            };
            set(Species::CDM, dc, vc);
            set(Species::Baryon, db, vb);
            set(Species::MassiveNeutrino, dn, vn);
            set(Species::CDMBaryon, dcb, vcb);
            set(Species::TotalMatter, dm, vm);
            set_tables(t);
        }

        const CLASSBackground & background() const { return bg; }
    };

    /// The CLASS output files of a run with this root (CLASS's "root" parameter): the transfer files <root>z*_tk.dat
    /// (sorted by name) and <root>background.dat
    inline std::pair<std::vector<std::string>, std::string> find_class_output_files(const std::string & root) {
        namespace fs = std::filesystem;
        const fs::path prefix(root);
        const fs::path folder = prefix.has_parent_path() ? prefix.parent_path() : fs::path(".");
        const std::string base = prefix.filename().string();
        const std::regex tk_regex("^" + std::regex_replace(base, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)") +
                                  R"(z\d+_tk\.dat$)");
        std::vector<std::string> files;
        for (auto & entry : fs::directory_iterator(folder))
            if (entry.is_regular_file() and std::regex_match(entry.path().filename().string(), tk_regex))
                files.push_back(entry.path().string());
        std::sort(files.begin(), files.end());
        if (files.empty())
            throw std::runtime_error("[find_class_output_files] No files " + root + "z*_tk.dat");
        return {files, root + "background.dat"};
    }

} // namespace FML::INITIALCONDITIONS
#endif
