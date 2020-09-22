#include "FileUtils.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace FML {
    namespace FILEUTILS {

        // Read a regular ascii files with nskip header lines and containing ncol collums
        // nestimated_lines is the amount we allocate for originally. Reallocated if file is larger
        // Not perfect for realy large files due to all the allocations we have to do
        DVector2D read_regular_ascii(std::string filename,
                                     int ncols,
                                     std::vector<int> cols_to_keep,
                                     int nskip,
                                     size_t nestimated_lines) {

            // Sanity check
            assert(cols_to_keep.size() > 0 and nskip >= 0 and ncols > 0);
            for (auto & i : cols_to_keep)
                assert(i < ncols and i >= 0);

            // Open file
            std::ifstream fp(filename.c_str());
            if (!fp) {
                throw std::runtime_error("[read_regular_ascii] Failed to open [" + filename + "]\n");
            }

            // Allocate memory for reading
            int ntokeep = int(cols_to_keep.size());
            DVector2D result;
            result.reserve(nestimated_lines);

            // For reading the file
            DVector newline(ntokeep);
            DVector temp(ncols);

            // Read and skip header lines
            for (int i = 0; i < nskip; i++) {
                std::string line;
                std::getline(fp, line);
#ifdef DEBUG_READASCII
                std::cout << "Skipping headerline: " << line << "\n";
#endif
            }

            // Read first entry and check that it has the right number of columns
            std::string line;
            std::getline(fp, line);
            std::stringstream ss;
            ss << line;
            int count = 0;
            while (ss >> line) {
                temp[count] = std::stod(line);
                ++count;
            }
            for (int i = 0; i < ntokeep; i++)
                newline[i] = temp[cols_to_keep[i]];
            result.push_back(newline);
            if (count != ncols) {
                throw std::runtime_error("Found ncols " + std::to_string(count) + " which differs from specified " +
                                         std::to_string(ncols) + "\n");
            }

            // Read the rest of the file
            while (1) {
                fp >> temp[0];
                if (fp.eof())
                    break;
                for (int i = 1; i < ncols; i++)
                    fp >> temp[i];
                for (int i = 0; i < ntokeep; i++)
                    newline[i] = temp[cols_to_keep[i]];
#ifdef DEBUG_READASCII
                if (result.size() < 10) {
                    std::cout << "Line " << result.size() - 1 << " : ";
                    for (auto e : result[result.size() - 1])
                        std::cout << e << " ";
                    std::cout << std::endl;
                }
#endif
                result.push_back(newline);
            }
            return result;
        }

        // Read a regular ascii files with nskip header lines and containing ncol collums
        // nestimated_lines is the amount we allocate for originally. Reallocated if file is larger
        // Not perfect for realy large files due to all the allocations we have to do
        DVector2D read_regular_ascii_subsampled(std::string filename,
                                                int ncols,
                                                std::vector<int> cols_to_keep,
                                                int nskip,
                                                size_t nestimated_lines,
                                                double fraction_to_read,
                                                unsigned int randomSeed) {
            std::mt19937 generator(randomSeed);
            auto udist = std::uniform_real_distribution<double>(0.0, 1.0);

            // Sanity check
            assert(cols_to_keep.size() > 0 and nskip >= 0 and ncols > 0);
            for (auto & i : cols_to_keep)
                assert(i < ncols and i >= 0);

            // Open file
            std::ifstream fp(filename.c_str());
            if (!fp) {
                throw std::runtime_error("[read_regular_ascii] Failed to open [" + filename + "]\n");
            }

            // Allocate memory for reading
            int ntokeep = int(cols_to_keep.size());
            DVector2D result;
            result.reserve(nestimated_lines);

            // For reading the file
            DVector newline(ntokeep);
            DVector temp(ncols);

            // Read and skip header lines
            for (int i = 0; i < nskip; i++) {
                std::string line;
                std::getline(fp, line);
#ifdef DEBUG_READASCII
                std::cout << "Skipping headerline: " << line << "\n";
#endif
            }

            // Read first entry and check that it has the right number of columns
            std::string line;
            std::getline(fp, line);
            std::stringstream ss;
            ss << line;
            int count = 0;
            while (ss >> line) {
                temp[count] = std::stod(line);
                ++count;
            }
            for (int i = 0; i < ntokeep; i++)
                newline[i] = temp[cols_to_keep[i]];
            result.push_back(newline);
            if (count != ncols) {
                throw std::runtime_error("Found ncols " + std::to_string(count) + " which differs from specified " +
                                         std::to_string(ncols) + "\n");
            }

            // Read the rest of the file
            while (1) {
                fp >> temp[0];
                if (fp.eof())
                    break;
                for (int i = 1; i < ncols; i++)
                    fp >> temp[i];
                for (int i = 0; i < ntokeep; i++)
                    newline[i] = temp[cols_to_keep[i]];
#ifdef DEBUG_READASCII
                if (result.size() < 10) {
                    std::cout << "Line " << result.size() - 1 << " : ";
                    for (auto e : result[result.size() - 1])
                        std::cout << e << " ";
                    std::cout << std::endl;
                }
#endif
                if (fraction_to_read >= 1.0)
                    result.push_back(newline);
                else if (udist(generator) < fraction_to_read)
                    result.push_back(newline);
            }
            return result;
        }
    } // namespace FILEUTILS
} // namespace FML
