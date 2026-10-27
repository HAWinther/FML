#include "FileUtils.h"
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace FML {
    namespace FILEUTILS {

        namespace {

            // Parse a line of numbers into row (cleared first). Everything after a '#' is a comment. Returns false if
            // the line has no numbers (blank or a comment line, or a line starting with '!' or '/'). Throws on a
            // token that is not a number (nan, inf and values below the smallest normal double are accepted).
            bool parse_line(const std::string & line,
                            std::vector<double> & row,
                            const std::string & filename,
                            size_t linenumber,
                            const std::string & caller) {
                row.clear();
                const size_t first = line.find_first_not_of(" \t\r");
                if (first == std::string::npos or line[first] == '#' or line[first] == '!' or line[first] == '/')
                    return false;
                std::istringstream ss(line.substr(0, line.find('#')));
                std::string token;
                while (ss >> token) {
                    const char * begin = token.c_str();
                    char * end = nullptr;
                    errno = 0;
                    const double value = std::strtod(begin, &end);
                    // ERANGE: overflow gives +-inf (kept), underflow gives a denormal or 0 (kept)
                    if (end == begin or *end != '\0')
                        throw std::runtime_error("[" + caller + "] Could not read [" + token + "] as a number on line " +
                                                 std::to_string(linenumber) + " of [" + filename + "]");
                    row.push_back(value);
                }
                return not row.empty();
            }

            // The common reader for read_regular_ascii(_subsampled)
            DVector2D read_regular_ascii_impl(const std::string & filename,
                                              int ncols,
                                              const std::vector<int> & cols_to_keep,
                                              int nskip,
                                              size_t nestimated_lines,
                                              double fraction_to_read,
                                              unsigned int randomSeed) {
                const std::string caller = "read_regular_ascii";
                if (cols_to_keep.empty() or nskip < 0 or ncols <= 0)
                    throw std::runtime_error("[" + caller + "] Need ncols > 0, nskip >= 0 and columns to keep");
                for (auto & i : cols_to_keep)
                    if (i < 0 or i >= ncols)
                        throw std::runtime_error("[" + caller + "] Column " + std::to_string(i) + " to keep is not in [0, " +
                                                 std::to_string(ncols) + ")");

                std::ifstream fp(filename.c_str());
                if (!fp)
                    throw std::runtime_error("[" + caller + "] Failed to open [" + filename + "]\n");

                std::mt19937 generator(randomSeed);
                auto udist = std::uniform_real_distribution<double>(0.0, 1.0);

                DVector2D result;
                result.reserve(nestimated_lines);
                DVector row;
                row.reserve(ncols);
                DVector newline(cols_to_keep.size());

                std::string line;
                size_t linenumber = 0;
                for (int i = 0; i < nskip; i++) {
                    std::getline(fp, line);
                    linenumber++;
#ifdef DEBUG_READASCII
                    std::cout << "Skipping headerline: " << line << "\n";
#endif
                }

                while (std::getline(fp, line)) {
                    linenumber++;
                    if (not parse_line(line, row, filename, linenumber, caller))
                        continue;
                    if (int(row.size()) != ncols)
                        throw std::runtime_error("[" + caller + "] Found " + std::to_string(row.size()) +
                                                 " columns on line " + std::to_string(linenumber) + " of [" + filename +
                                                 "], expected " + std::to_string(ncols));
                    if (fraction_to_read < 1.0 and udist(generator) >= fraction_to_read)
                        continue;
                    for (size_t i = 0; i < cols_to_keep.size(); i++)
                        newline[i] = row[cols_to_keep[i]];
                    result.push_back(newline);
                }
                return result;
            }
        } // namespace

        DVector2D read_regular_ascii(std::string filename,
                                     int ncols,
                                     std::vector<int> cols_to_keep,
                                     int nskip,
                                     size_t nestimated_lines) {
            return read_regular_ascii_impl(filename, ncols, cols_to_keep, nskip, nestimated_lines, 1.0, 0);
        }

        DVector2D read_regular_ascii_subsampled(std::string filename,
                                                int ncols,
                                                std::vector<int> cols_to_keep,
                                                int nskip,
                                                size_t nestimated_lines,
                                                double fraction_to_read,
                                                unsigned int randomSeed) {
            return read_regular_ascii_impl(
                filename, ncols, cols_to_keep, nskip, nestimated_lines, fraction_to_read, randomSeed);
        }

        std::pair<DVector, DVector> read_file_and_extract_two_columns(std::string filename, int col1, int col2) {
            auto data = loadtxt(filename);
            DVector x, y;
            x.reserve(data.size());
            y.reserve(data.size());
            for (size_t i = 0; i < data.size(); i++) {
                const int ncol = int(data[i].size());
                if (col1 < 0 or col2 < 0 or col1 >= ncol or col2 >= ncol)
                    throw std::runtime_error("[read_file_and_extract_two_columns] Columns " + std::to_string(col1) +
                                             ", " + std::to_string(col2) + " not found in data row " +
                                             std::to_string(i) + " of [" + filename + "] (" + std::to_string(ncol) +
                                             " columns)");
                x.push_back(data[i][col1]);
                y.push_back(data[i][col2]);
            }
            return {x, y};
        }

        DVector2D loadtxt(std::string filename, int nreserve_rows, int nreserve_cols) {
            std::ifstream fp(filename.c_str());
            if (!fp)
                throw std::runtime_error("[loadtxt] Failed to open [" + filename + "]\n");

            DVector2D result;
            result.reserve(std::max(nreserve_rows, 1));
            DVector row;
            row.reserve(std::max(nreserve_cols, 1));

            std::string line;
            size_t linenumber = 0;
            while (std::getline(fp, line)) {
                linenumber++;
                if (parse_line(line, row, filename, linenumber, "loadtxt"))
                    result.push_back(row);
            }

            // Check if the result is regular
            if (not result.empty()) {
                size_t ncolmin = result[0].size();
                size_t ncolmax = result[0].size();
                for (auto & r : result) {
                    ncolmin = std::min(ncolmin, r.size());
                    ncolmax = std::max(ncolmax, r.size());
                }
                if (ncolmin != ncolmax) {
                    std::cout << "[loadtxt] Warning the file " + filename + " is not regular\n";
                    std::cout << "The minimum number of columns is " << ncolmin << "\n";
                    std::cout << "The maximum number of columns is " << ncolmax << "\n";
                }
            }
            return result;
        }

    } // namespace FILEUTILS
} // namespace FML
