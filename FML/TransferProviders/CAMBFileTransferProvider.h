#ifndef FML_CAMB_FILE_TRANSFER_PROVIDER_H
#define FML_CAMB_FILE_TRANSFER_PROVIDER_H

#include <FML/TransferProviders/TabulatedTransferProvider.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

//=====================================================================================================================
// Transfer functions from CAMB transfer files (one per redshift), listed in a transfer info file:
//
//   <folder> <number of files>
//   <file 1> <redshift 1>
//   ...
//
// (the files are <folder>/<file i>; if not found there, they are looked for next to the info file). Any order of the
// redshifts. The columns are read by name when the file starts with a header line of CAMB's column names
// ("k/h CDM baryon photon nu mass_nu total no_nu total_de Weyl v_CDM v_b v_b-v_c", with or without a leading #), and
// otherwise by position from the number of columns: 13 (CAMB), 9 (AxionCAMB: no Weyl and velocities) or 7 (old CAMB,
// or CLASS run with format = camb: k CDM baryon photon nu mass_nu total).
//
// CAMB's columns are T = delta / k^2 in Mpc^2 (unit initial curvature, the FML sign convention): density = T k^2 with
// k in 1/Mpc. The densities are CDM-synchronous. v_CDM and v_b are the Newtonian-gauge velocities -theta / (aH k^2):
// velocity = -T_v k^2. CDMBaryon is weighted by Omegab and OmegaCDM (needed), TotalMatter is CAMB's total column (CDM +
// baryons + massive neutrinos weighted by their densities; without it the weights Omegab, OmegaCDM, OmegaMNu). The
// massive neutrinos are available when the mass_nu column is there and not zero; their velocity is not in the files.
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    class CAMBFileTransferProvider final : public TabulatedTransferProvider {
      public:
        struct Settings {
            /// The transfer info file (see above)
            std::string infofile;
            /// h (CAMB's k is in h/Mpc, T in Mpc^2) and the density parameters today for the weights of CDMBaryon (and
            /// of TotalMatter if the files have no total column)
            double h{0.0};
            double Omegab{0.0};
            double OmegaCDM{0.0};
            double OmegaMNu{0.0};
            /// The primordial spectrum of the run: P_R(k) = A_s (k / k_pivot)^(n_s - 1)
            double A_s{0.0};
            double n_s{1.0};
            double k_pivot_mpc{0.05};
            RangePolicy range_policy{RangePolicy::SourceExtrapolation};
        };

      private:
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        /// The columns of a file: name -> values
        static std::map<std::string, DVector> read_camb_file(const std::string & filename) {
            std::ifstream fp(filename);
            if (not fp)
                throw std::runtime_error("[CAMBFileTransferProvider] Cannot open [" + filename + "]");
            const auto is_number = [](const std::string & s) {
                char * end = nullptr;
                std::strtod(s.c_str(), &end);
                return end != s.c_str() and *end == '\0';
            };
            std::vector<std::string> names;
            std::vector<DVector> rows;
            std::string line;
            bool first = true;
            while (std::getline(fp, line)) {
                if (line.find_first_not_of(" \t\r") == std::string::npos)
                    continue;
                const bool comment = line.find_first_not_of(" \t") != std::string::npos and
                                     line[line.find_first_not_of(" \t")] == '#';
                std::istringstream ss(comment ? line.substr(line.find('#') + 1) : line);
                std::vector<std::string> tokens;
                for (std::string token; ss >> token;)
                    tokens.push_back(token);
                if (tokens.empty())
                    continue;
                const bool header = comment or not is_number(tokens[0]);
                if (header) {
                    // The column names: the first header line that has k/h first
                    if (first and names.empty() and tokens[0] == "k/h")
                        names = tokens;
                    if (not rows.empty())
                        throw std::runtime_error("[CAMBFileTransferProvider] A header line after the data in [" +
                                                 filename + "]");
                    continue;
                }
                first = false;
                DVector row;
                for (auto & token : tokens) {
                    if (not is_number(token))
                        throw std::runtime_error("[CAMBFileTransferProvider] Cannot read [" + token + "] in [" +
                                                 filename + "]");
                    row.push_back(std::stod(token));
                }
                if (not rows.empty() and row.size() != rows.front().size())
                    throw std::runtime_error("[CAMBFileTransferProvider] The lines in [" + filename +
                                             "] have different numbers of columns");
                rows.push_back(std::move(row));
            }
            if (rows.empty())
                throw std::runtime_error("[CAMBFileTransferProvider] No data in [" + filename + "]");
            const size_t ncol = rows.front().size();
            if (names.empty()) {
                if (ncol == 13)
                    names = {"k/h", "CDM", "baryon", "photon", "nu", "mass_nu", "total", "no_nu", "total_de", "Weyl",
                             "v_CDM", "v_b", "v_b-v_c"};
                else if (ncol == 9)
                    names = {"k/h", "CDM", "baryon", "photon", "nu", "mass_nu", "total", "no_nu", "total_de"};
                else if (ncol == 7)
                    names = {"k/h", "CDM", "baryon", "photon", "nu", "mass_nu", "total"};
                else
                    throw std::runtime_error("[CAMBFileTransferProvider] [" + filename + "] has " +
                                             std::to_string(ncol) +
                                             " columns and no header with the column names: expected 13, 9 or 7");
            }
            if (names.size() != ncol)
                throw std::runtime_error("[CAMBFileTransferProvider] The header of [" + filename + "] names " +
                                         std::to_string(names.size()) + " columns, the data have " +
                                         std::to_string(ncol));
            std::map<std::string, DVector> columns;
            for (size_t c = 0; c < ncol; c++) {
                DVector values(rows.size());
                for (size_t i = 0; i < rows.size(); i++)
                    values[i] = rows[i][c];
                columns[names[c]] = std::move(values);
            }
            return columns;
        }

      public:
        explicit CAMBFileTransferProvider(const Settings & settings) {
            namespace fs = std::filesystem;
            if (not(settings.h > 0.0) or not(settings.Omegab > 0.0) or not(settings.OmegaCDM > 0.0) or
                not(settings.OmegaMNu >= 0.0))
                throw std::invalid_argument("[CAMBFileTransferProvider] Set h, Omegab and OmegaCDM (> 0) and OmegaMNu "
                                            "(>= 0)");

            // The info file: the folder, the number of files, then the files and their redshifts
            std::ifstream fp(settings.infofile);
            if (not fp)
                throw std::runtime_error("[CAMBFileTransferProvider] Cannot open [" + settings.infofile + "]");
            std::string folder;
            int nfiles = 0;
            if (not(fp >> folder >> nfiles) or nfiles < 2)
                throw std::runtime_error("[CAMBFileTransferProvider] [" + settings.infofile +
                                         "] must start with the folder and the number of files (2 or more)");
            struct Entry {
                double a;
                std::string filename;
                std::map<std::string, DVector> columns;
            };
            std::vector<Entry> entries;
            const fs::path info_folder = fs::path(settings.infofile).parent_path();
            for (int i = 0; i < nfiles; i++) {
                std::string name;
                double z;
                if (not(fp >> name >> z))
                    throw std::runtime_error("[CAMBFileTransferProvider] [" + settings.infofile + "] lists fewer than " +
                                             std::to_string(nfiles) + " files");
                if (not(z > -1.0))
                    throw std::runtime_error("[CAMBFileTransferProvider] Invalid redshift in [" + settings.infofile + "]");
                fs::path path = fs::path(folder) / name;
                if (not fs::exists(path) and fs::exists(info_folder / name))
                    path = info_folder / name;
                entries.push_back({1.0 / (1.0 + z), path.string(), read_camb_file(path.string())});
            }
            std::sort(entries.begin(), entries.end(), [](const Entry & x, const Entry & y) { return x.a < y.a; });
            for (size_t i = 1; i < entries.size(); i++)
                if (not(entries[i].a > entries[i - 1].a))
                    throw std::invalid_argument("[CAMBFileTransferProvider] Two transfer files have the same redshift");

            const auto & first = entries.front();
            const auto require = [](const Entry & e, const std::string & name) -> const DVector & {
                auto it = e.columns.find(name);
                if (it == e.columns.end())
                    throw std::runtime_error("[CAMBFileTransferProvider] No column [" + name + "] in [" + e.filename +
                                             "]");
                return it->second;
            };
            const auto has = [&](const std::string & name) { return first.columns.count(name) > 0; };
            const DVector & k = require(first, "k/h");
            const bool have_total = has("total"), have_vc = has("v_CDM"), have_vb = has("v_b");
            bool have_nu = has("mass_nu");
            if (have_nu) {
                bool nonzero = false;
                for (auto & e : entries)
                    for (double value : require(e, "mass_nu"))
                        nonzero = nonzero or value != 0.0;
                have_nu = nonzero;
            }
            if (not have_total and settings.OmegaMNu > 0.0 and not have_nu)
                throw std::invalid_argument("[CAMBFileTransferProvider] OmegaMNu > 0 but the files have neither a total "
                                            "nor a massive neutrino column");

            const size_t na = entries.size(), nk = k.size();
            TransferTables t;
            t.name = "CAMBFileTransferProvider";
            t.h = settings.h;
            t.k_hmpc = k;
            t.a.resize(na);
            t.density_gauge = Gauge::SynchronousCDM;
            t.velocity_gauge = Gauge::Newtonian;
            t.range_policy = settings.range_policy;
            t.A_s = settings.A_s;
            t.n_s = settings.n_s;
            t.k_pivot_mpc = settings.k_pivot_mpc;
            const auto table = [&](bool have) { return have ? DVector2D(na, DVector(nk, 0.0)) : DVector2D{}; };
            DVector2D dc = table(true), db = table(true), dn = table(have_nu), dcb = table(true), dm = table(true);
            DVector2D vc = table(have_vc), vb = table(have_vb), vcb = table(have_vc and have_vb);
            const double wc = settings.OmegaCDM, wb = settings.Omegab, wn = have_nu ? settings.OmegaMNu : 0.0;
            for (size_t ia = 0; ia < na; ia++) {
                const auto & e = entries[ia];
                t.a[ia] = e.a;
                const DVector & kk = require(e, "k/h");
                if (kk.size() != nk)
                    throw std::invalid_argument("[CAMBFileTransferProvider] The files have different k-grids");
                for (size_t ik = 0; ik < nk; ik++)
                    if (std::abs(kk[ik] / k[ik] - 1.0) > 1e-6)
                        throw std::invalid_argument("[CAMBFileTransferProvider] The files have different k-grids");
                const DVector &cdm = require(e, "CDM"), &b = require(e, "baryon");
                for (size_t ik = 0; ik < nk; ik++) {
                    const double k2 = std::pow(k[ik] * settings.h, 2);
                    dc[ia][ik] = cdm[ik] * k2;
                    db[ia][ik] = b[ik] * k2;
                    dcb[ia][ik] = (wc * dc[ia][ik] + wb * db[ia][ik]) / (wc + wb);
                    if (have_nu)
                        dn[ia][ik] = require(e, "mass_nu")[ik] * k2;
                    dm[ia][ik] = have_total ? require(e, "total")[ik] * k2 :
                                              (wc * dc[ia][ik] + wb * db[ia][ik] + wn * (have_nu ? dn[ia][ik] : 0.0)) /
                                                  (wc + wb + wn);
                    if (have_vc)
                        vc[ia][ik] = -require(e, "v_CDM")[ik] * k2;
                    if (have_vb)
                        vb[ia][ik] = -require(e, "v_b")[ik] * k2;
                    if (have_vc and have_vb)
                        vcb[ia][ik] = (wc * vc[ia][ik] + wb * vb[ia][ik]) / (wc + wb);
                }
            }
            const auto set = [&](Species s, DVector2D & dens, DVector2D vel) {
                t.density[TransferTables::index(s)] = std::move(dens);
                t.velocity[TransferTables::index(s)] = std::move(vel);
            };
            set(Species::CDM, dc, std::move(vc));
            set(Species::Baryon, db, std::move(vb));
            set(Species::CDMBaryon, dcb, std::move(vcb));
            set(Species::MassiveNeutrino, dn, {});
            set(Species::TotalMatter, dm, {});
            set_tables(t);
        }
    };

} // namespace FML::INITIALCONDITIONS
#endif
