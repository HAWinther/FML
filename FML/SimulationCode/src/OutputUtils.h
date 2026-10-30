#ifndef SIMCODE_OUTPUT_UTILS_H
#define SIMCODE_OUTPUT_UTILS_H
#include <iomanip>
#include <sstream>
#include <string>
/// The redshift formatted as used in output filenames.
inline std::string redshift_to_string(double redshift) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << redshift;
    return stream.str();
}
#endif
