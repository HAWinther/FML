#ifndef FILEUTILS_HEADER
#define FILEUTILS_HEADER
#include <string>
#include <vector>

namespace FML {

    //================================================================================
    /// This namespace deals with reading and writing files to disc.
    ///
    //================================================================================

    namespace FILEUTILS {

        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;

        /// Read a regular ascii files with nskip header lines and containing ncol collums
        /// nestimated_lines is the amount we allocate for originally. Reallocated if file is larger
        /// Not perfect for realy large files due to all the allocations we have to do
        DVector2D read_regular_ascii(std::string filename,
                                     int ncols,
                                     std::vector<int> cols_to_keep,
                                     int nskip,
                                     size_t nestimated_lines = 10000);

        /// As above, but include every line read with probabillity fraction_to_read
        DVector2D read_regular_ascii_subsampled(std::string filename,
                                                int ncols,
                                                std::vector<int> cols_to_keep,
                                                int nskip,
                                                size_t nestimated_lines = 10000,
                                                double fraction_to_read = 1.0,
                                                unsigned int randomSeed = 1234);

        /// Similar to pythons loadtxt
        DVector2D loadtxt(std::string filename, int nreserve_rows = 1, int nreserve_cols = 1);

        /// Read a regular file and extract two columns (numbering starting with 0)
        std::pair<DVector, DVector> read_file_and_extract_two_columns(int col1, int col2);
    } // namespace FILEUTILS
} // namespace FML
#endif
