#ifndef FML_TRANSFER_TABLES_FILE_H
#define FML_TRANSFER_TABLES_FILE_H

#include <FML/TransferProviders/TabulatedTransferProvider.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=====================================================================================================================
// TransferTables in one self-describing text file (all redshifts, all species, optionally the N-body gauge source).
//
//   # FML transfer tables
//   # format_version = 1
//   # <key> = <value>                 the tables (h, A_s, n_s, k_pivot_mpc, density_gauge, velocity_gauge,
//   # ...                             range_policy, na, nk) and any other metadata of the writer (e.g. the cosmology)
//   # columns = a k_hmpc d_cdm ...    the names of the columns
//   a k d_cdm ...                     na x nk rows: all k (increasing) for the first a, then the next a, ...
//
// The columns after a and k_hmpc are d_<species> = delta/R (CAMB sign, R the primordial curvature) and
// t_<species> = theta/(aH R) with species cdm, b, cb, ncdm, m (CDM, Baryon, CDMBaryon, MassiveNeutrino, TotalMatter),
// and nbody_source = delta_extra/R (see TransferProvider). Species the tables do not have are left out.
// The values are written with 10 significant digits (the header values with 17), far below the accuracy of any
// Boltzmann code. Metadata keys may not contain '=' and values not newlines.
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    struct TransferTablesFile {
        TransferTables tables;
        /// The metadata of the writer (in the order of the file), without the keys of the tables
        std::vector<std::pair<std::string, std::string>> metadata;

        /// The value of a metadata key (empty if missing)
        std::string get(const std::string & key) const {
            for (const auto & kv : metadata)
                if (kv.first == key)
                    return kv.second;
            return "";
        }
        bool contains(const std::string & key) const {
            for (const auto & kv : metadata)
                if (kv.first == key)
                    return true;
            return false;
        }
    };

    namespace TRANSFERTABLESFILE_DETAIL {
        inline const std::vector<std::string> & species_names() {
            static const std::vector<std::string> names{"cdm", "b", "cb", "ncdm", "m"};
            return names;
        }
        inline std::string gauge_name(Gauge g) {
            switch (g) {
                case Gauge::Newtonian:
                    return "newtonian";
                case Gauge::SynchronousCDM:
                    return "synchronous_cdm";
                case Gauge::NBody:
                    return "nbody";
            }
            return "";
        }
        inline Gauge gauge_of(const std::string & name, const std::string & msg) {
            for (auto g : {Gauge::Newtonian, Gauge::SynchronousCDM, Gauge::NBody})
                if (gauge_name(g) == name)
                    return g;
            throw std::runtime_error(msg + "Unknown gauge [" + name + "] (newtonian, synchronous_cdm or nbody)");
        }
        inline std::string trim(const std::string & s) {
            const auto b = s.find_first_not_of(" \t\r");
            if (b == std::string::npos)
                return "";
            return s.substr(b, s.find_last_not_of(" \t\r") - b + 1);
        }
        inline std::string exact(double x) {
            std::ostringstream o;
            o << std::setprecision(17) << x;
            return o.str();
        }
        inline double to_double(const std::string & s, const std::string & what, const std::string & msg) {
            try {
                size_t pos = 0;
                const double x = std::stod(s, &pos);
                if (trim(s.substr(pos)).empty())
                    return x;
            } catch (...) {
            }
            throw std::runtime_error(msg + "Cannot read " + what + " = [" + s + "]");
        }
    } // namespace TRANSFERTABLESFILE_DETAIL

    /// Write the tables (and the metadata, in the given order) to filename
    inline void write_transfer_tables(const std::string & filename,
                                      const TransferTables & t,
                                      const std::vector<std::pair<std::string, std::string>> & metadata = {}) {
        using namespace TRANSFERTABLESFILE_DETAIL;
        const std::string msg = "[write_transfer_tables] ";
        const size_t na = t.a.size(), nk = t.k_hmpc.size();
        std::vector<std::pair<std::string, const TransferTables::DVector2D *>> columns;
        for (int s = 0; s < TransferTables::nspecies; s++)
            if (not t.density[s].empty())
                columns.push_back({"d_" + species_names()[s], &t.density[s]});
        for (int s = 0; s < TransferTables::nspecies; s++)
            if (not t.velocity[s].empty())
                columns.push_back({"t_" + species_names()[s], &t.velocity[s]});
        if (not t.nbody_gauge_source.empty())
            columns.push_back({"nbody_source", &t.nbody_gauge_source});
        for (const auto & c : columns) {
            if (c.second->size() != na)
                throw std::invalid_argument(msg + c.first + ": one table per redshift needed");
            for (const auto & row : *c.second)
                if (row.size() != nk)
                    throw std::invalid_argument(msg + c.first + ": one value per k needed");
        }

        std::ofstream f(filename);
        if (not f)
            throw std::runtime_error(msg + "Cannot open [" + filename + "] for writing");
        f << "# FML transfer tables\n";
        f << "# format_version = 1\n";
        for (const auto & kv : metadata) {
            if (kv.first.find('=') != std::string::npos or kv.second.find('\n') != std::string::npos)
                throw std::invalid_argument(msg + "Invalid metadata key [" + kv.first + "]");
            f << "# " << kv.first << " = " << kv.second << "\n";
        }
        f << "# h = " << exact(t.h) << "\n";
        f << "# A_s = " << exact(t.A_s) << "\n";
        f << "# n_s = " << exact(t.n_s) << "\n";
        f << "# k_pivot_mpc = " << exact(t.k_pivot_mpc) << "\n";
        f << "# density_gauge = " << gauge_name(t.density_gauge) << "\n";
        f << "# velocity_gauge = " << gauge_name(t.velocity_gauge) << "\n";
        f << "# range_policy = "
          << (t.range_policy == RangePolicy::SourceExtrapolation ? "source_extrapolation" : "reject") << "\n";
        f << "# na = " << na << "\n";
        f << "# nk = " << nk << "\n";
        f << "# columns = a k_hmpc";
        for (const auto & c : columns)
            f << " " << c.first;
        f << "\n";
        char buf[32];
        for (size_t ia = 0; ia < na; ia++) {
            for (size_t ik = 0; ik < nk; ik++) {
                std::string line = exact(t.a[ia]) + " " + exact(t.k_hmpc[ik]);
                for (const auto & c : columns) {
                    std::snprintf(buf, sizeof(buf), " %.9e", (*c.second)[ia][ik]);
                    line += buf;
                }
                f << line << "\n";
            }
        }
        f.close();
        if (not f)
            throw std::runtime_error(msg + "Failed writing [" + filename + "]");
    }

    /// Read a file written by write_transfer_tables
    inline TransferTablesFile read_transfer_tables(const std::string & filename) {
        using namespace TRANSFERTABLESFILE_DETAIL;
        const std::string msg = "[read_transfer_tables] [" + filename + "] ";
        std::ifstream f(filename);
        if (not f)
            throw std::runtime_error(msg + "Cannot open the file");

        TransferTablesFile out;
        auto & t = out.tables;
        t.name = "TransferTables(" + filename + ")";
        std::map<std::string, std::string> keys;
        std::vector<std::string> columns;
        std::string line;
        bool first = true;
        long lineno = 0;
        std::vector<std::vector<double>> rows;
        while (std::getline(f, line)) {
            lineno++;
            if (first) {
                if (trim(line) != "# FML transfer tables")
                    throw std::runtime_error(msg + "Not an FML transfer tables file (the first line must be "
                                                   "'# FML transfer tables')");
                first = false;
                continue;
            }
            if (trim(line).empty())
                continue;
            if (line[0] == '#') {
                const auto eq = line.find('=');
                if (eq == std::string::npos)
                    continue;
                const auto key = trim(line.substr(1, eq - 1));
                const auto value = trim(line.substr(eq + 1));
                if (key == "columns") {
                    std::istringstream is(value);
                    std::string c;
                    while (is >> c)
                        columns.push_back(c);
                } else if (keys.count(key) == 0) {
                    keys[key] = value;
                    out.metadata.push_back({key, value});
                }
                continue;
            }
            if (columns.empty())
                throw std::runtime_error(msg + "Data before the '# columns = ...' line");
            std::vector<double> row;
            row.reserve(columns.size());
            const char * p = line.c_str();
            char * end = nullptr;
            for (;;) {
                const double x = std::strtod(p, &end);
                if (end == p)
                    break;
                row.push_back(x);
                p = end;
            }
            if (row.size() != columns.size() or not trim(p).empty())
                throw std::runtime_error(msg + "Line " + std::to_string(lineno) + " has " + std::to_string(row.size()) +
                                         " numbers, the header " + std::to_string(columns.size()) + " columns");
            rows.push_back(std::move(row));
        }
        if (first)
            throw std::runtime_error(msg + "Empty file");

        const auto need = [&](const std::string & key) {
            const auto it = keys.find(key);
            if (it == keys.end())
                throw std::runtime_error(msg + "Missing '# " + key + " = ...' in the header");
            return it->second;
        };
        if (need("format_version") != "1")
            throw std::runtime_error(msg + "Unknown format_version " + need("format_version") + " (this code reads 1)");
        t.h = to_double(need("h"), "h", msg);
        t.A_s = to_double(need("A_s"), "A_s", msg);
        t.n_s = to_double(need("n_s"), "n_s", msg);
        t.k_pivot_mpc = to_double(need("k_pivot_mpc"), "k_pivot_mpc", msg);
        t.density_gauge = gauge_of(need("density_gauge"), msg);
        t.velocity_gauge = gauge_of(need("velocity_gauge"), msg);
        const auto policy = need("range_policy");
        if (policy != "source_extrapolation" and policy != "reject")
            throw std::runtime_error(msg + "Unknown range_policy [" + policy + "]");
        t.range_policy = policy == "reject" ? RangePolicy::Reject : RangePolicy::SourceExtrapolation;
        const long na = long(to_double(need("na"), "na", msg)), nk = long(to_double(need("nk"), "nk", msg));
        if (columns.size() < 3 or columns[0] != "a" or columns[1] != "k_hmpc")
            throw std::runtime_error(msg + "The columns must start with 'a k_hmpc'");
        if (na < 1 or nk < 1 or long(rows.size()) != na * nk)
            throw std::runtime_error(msg + "The header says na x nk = " + std::to_string(na) + " x " +
                                     std::to_string(nk) + " rows, the file has " + std::to_string(rows.size()));
        for (const auto & kv : {"format_version",
                                "h",
                                "A_s",
                                "n_s",
                                "k_pivot_mpc",
                                "density_gauge",
                                "velocity_gauge",
                                "range_policy",
                                "na",
                                "nk"})
            for (auto it = out.metadata.begin(); it != out.metadata.end(); ++it)
                if (it->first == kv) {
                    out.metadata.erase(it);
                    break;
                }

        // The grids (the same k for every a)
        t.a.resize(na);
        t.k_hmpc.resize(nk);
        for (long ia = 0; ia < na; ia++)
            t.a[ia] = rows[ia * nk][0];
        for (long ik = 0; ik < nk; ik++)
            t.k_hmpc[ik] = rows[ik][1];
        for (long ia = 0; ia < na; ia++)
            for (long ik = 0; ik < nk; ik++) {
                const auto & r = rows[ia * nk + ik];
                if (r[0] != t.a[ia] or r[1] != t.k_hmpc[ik])
                    throw std::runtime_error(msg + "Row " + std::to_string(ia * nk + ik + 1) +
                                             ": the rows must be all k (the same for every a) for each a in turn");
            }

        // The columns
        std::vector<bool> seen(columns.size(), false);
        const auto fill = [&](const std::string & name, TransferTables::DVector2D & table) {
            for (size_t c = 2; c < columns.size(); c++) {
                if (columns[c] != name)
                    continue;
                if (seen[c])
                    throw std::runtime_error(msg + "The column " + name + " appears twice");
                seen[c] = true;
                table.assign(na, TransferTables::DVector(nk));
                for (long ia = 0; ia < na; ia++)
                    for (long ik = 0; ik < nk; ik++)
                        table[ia][ik] = rows[ia * nk + ik][c];
            }
        };
        for (int s = 0; s < TransferTables::nspecies; s++) {
            fill("d_" + species_names()[s], t.density[s]);
            fill("t_" + species_names()[s], t.velocity[s]);
        }
        fill("nbody_source", t.nbody_gauge_source);
        for (size_t c = 2; c < columns.size(); c++)
            if (not seen[c])
                throw std::runtime_error(msg + "Unknown column [" + columns[c] +
                                         "] (d_<s> and t_<s> with s = cdm, b, cb, ncdm, m, and nbody_source)");
        return out;
    }

    /// A TabulatedTransferProvider from a file of write_transfer_tables
    class TransferTablesFileProvider final : public TabulatedTransferProvider {
      public:
        explicit TransferTablesFileProvider(const std::string & filename) {
            set_tables(read_transfer_tables(filename).tables);
        }
    };

} // namespace FML::INITIALCONDITIONS

#endif
