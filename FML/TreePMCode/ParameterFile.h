#ifndef FML_TREEPMCODE_PARAMETERFILE_HEADER
#define FML_TREEPMCODE_PARAMETERFILE_HEADER

#include <FML/Global/Global.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace FML::NBODY::TREEPM {

    //==========================================================================================
    /// The parameter file of the simple N-body codes: "key = value" lines, # starts a comment, "include = file"
    //==========================================================================================
    class ParameterFile {
        std::map<std::string, std::string> values;
        mutable std::set<std::string> used;

      public:
        /// "include = file" reads the keys of another file there (relative to this file's folder); a key set later
        /// (after the include, or in the including file) replaces an earlier value
        explicit ParameterFile(const std::string & filename, int depth = 0) { read(filename, depth); }
        void read(const std::string & filename, int depth) {
            assert_mpi(depth < 10, "ParameterFile: too many nested includes");
            std::ifstream fp(filename);
            assert_mpi(fp.good(), ("Cannot open parameter file " + filename).c_str());
            const auto slash = filename.find_last_of('/');
            const std::string folder = slash == std::string::npos ? "" : filename.substr(0, slash + 1);
            std::string line;
            while (std::getline(fp, line)) {
                line = line.substr(0, line.find('#'));
                auto eq = line.find('=');
                if (eq == std::string::npos)
                    continue;
                auto trim = [](std::string s) {
                    s.erase(0, s.find_first_not_of(" \t\r"));
                    s.erase(s.find_last_not_of(" \t\r") + 1);
                    return s;
                };
                const auto key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
                if (key == "include") {
                    read(value.size() and value[0] == '/' ? value : folder + value, depth + 1);
                    continue;
                }
                values[key] = value;
            }
        }
        /// Keys in the file that were never read (typos): a warning on task 0
        void warn_unused() const {
            for (auto & kv : values)
                if (not used.count(kv.first) and FML::ThisTask == 0)
                    std::cout << "Warning: [" << kv.first << "] in the parameter file is not a parameter (typo?)\n";
        }
        std::string get(const std::string & key, const std::string & def) const {
            used.insert(key);
            auto it = values.find(key);
            return it == values.end() ? def : it->second;
        }
        double get(const std::string & key, double def) const {
            used.insert(key);
            auto it = values.find(key);
            return it == values.end() ? def : std::stod(it->second);
        }
        int get(const std::string & key, int def) const { return int(get(key, double(def))); }
        bool get(const std::string & key, bool def) const {
            used.insert(key);
            auto it = values.find(key);
            return it == values.end() ? def : (it->second == "true" or it->second == "1" or it->second == "yes");
        }
        std::vector<double> get_list(const std::string & key) const {
            std::vector<double> v;
            std::stringstream ss(get(key, std::string("")));
            std::string item;
            while (std::getline(ss, item, ','))
                if (item.find_first_not_of(" \t") != std::string::npos)
                    v.push_back(std::stod(item));
            return v;
        }
    };

} // namespace FML::NBODY::TREEPM
#endif
