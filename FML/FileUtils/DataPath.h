#ifndef FML_FILEUTILS_DATA_PATH_H
#define FML_FILEUTILS_DATA_PATH_H
#include <cstdlib>
#include <string>

namespace FML { namespace FILEUTILS {
    /// Resolve an FML data path: FML_DIR environment variable, compiled FML_ROOT_DIR, then relative path.
    inline std::string fml_file(const std::string & relative_path) {
        if (const char * env = std::getenv("FML_DIR"))
            if (*env) return std::string(env) + "/" + relative_path;
#ifdef FML_ROOT_DIR
        return std::string(FML_ROOT_DIR) + "/" + relative_path;
#else
        return relative_path;
#endif
    }
} }
#endif
