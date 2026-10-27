#ifndef FILEUTILS_HEADER
#define FILEUTILS_HEADER
#include <FML/FileUtils/DataPath.h>
#include <string>
#include <utility>
#include <vector>

namespace FML {

    //================================================================================
    /// This namespace deals with reading and writing files to disc.
    ///
    //================================================================================

    namespace FILEUTILS {

        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        /// File format for all the readers: numbers separated by whitespace. Blank lines, lines starting with #, ! or
        /// / and everything after a # on a line are ignored. nan, inf and very small numbers (denormals) are read.
        /// A token that is not a number gives an error (std::runtime_error) with the line number.

        /// Read a regular ascii file with nskip header lines and ncols columns on every (data) line, keeping the
        /// columns cols_to_keep (numbering from 0). nestimated_lines is the number of rows reserved initially.
        /// A line with another number of columns gives an error.
        DVector2D read_regular_ascii(std::string filename,
                                     int ncols,
                                     std::vector<int> cols_to_keep,
                                     int nskip,
                                     size_t nestimated_lines = 10000);

        /// As above, but keep every line with probability fraction_to_read
        DVector2D read_regular_ascii_subsampled(std::string filename,
                                                int ncols,
                                                std::vector<int> cols_to_keep,
                                                int nskip,
                                                size_t nestimated_lines = 10000,
                                                double fraction_to_read = 1.0,
                                                unsigned int randomSeed = 1234);

        /// Similar to numpy's loadtxt: all the numbers on each line (rows may differ in length; a warning is printed)
        DVector2D loadtxt(std::string filename, int nreserve_rows = 1, int nreserve_cols = 1);

        /// Read a file with loadtxt and extract two columns (numbering starting with 0)
        std::pair<DVector, DVector> read_file_and_extract_two_columns(std::string filename, int col1, int col2);
    } // namespace FILEUTILS
} // namespace FML
#endif
