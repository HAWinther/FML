#ifndef FILEUTILS_HEADER
#define FILEUTILS_HEADER
#include <vector>
#include <string>

namespace FML {
  namespace FILEUTILS {

    using DVector   = std::vector<double>;
    using DVector2D = std::vector<DVector>;

    // Read a regular ascii files with nskip header lines and containing ncol collums
    // nestimated_lines is the amount we allocate for originally. Reallocated if file is larger
    // Not perfect for realy large files due to all the allocations we have to do
    DVector2D read_regular_ascii(
        std::string filename, 
        int ncols, 
        std::vector<int> cols_to_keep, 
        int nskip, 
        size_t nestimated_lines = 10000);
        
    // As above, but include every line read with probabillity fraction_to_read
    DVector2D read_regular_ascii_subsampled(
        std::string filename, 
        int ncols, 
        std::vector<int> cols_to_keep, 
        int nskip, 
        size_t nestimated_lines = 10000,
        double fraction_to_read = 1.0,
        unsigned int randomSeed = 1234);
  }
}
#endif
