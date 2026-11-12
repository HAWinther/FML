#ifndef FML_FOUR_POINT_COMMON_H
#define FML_FOUR_POINT_COMMON_H
#include <FML/Global/Global.h>
#include <array>
#include <climits>
namespace FML::FOURPOINT {
    // These checks are collective: all ranks fail before subsequent FFTs/reductions.
    inline void require(bool valid, const char * message) {
        int ok = valid;
        FML::MinOverTasks(&ok);
        if (!ok)
            throw std::invalid_argument(message);
    }
    inline void agree(const std::vector<double> & values) {
        size_t lo = values.size(), hi = lo;
        FML::MinOverTasks(&lo);
        FML::MaxOverTasks(&hi);
        require(lo == hi && hi <= size_t(INT_MAX), "[FourPoint] Incompatible option lengths");
        bool finite = true;
        for (auto v : values)
            finite = finite && std::isfinite(v);
        require(finite, "[FourPoint] Nonfinite options");
#ifdef USE_MPI
        std::vector<double> lower(values), upper(values);
        FML::MinArrayOverTasks(lower.data(), lower.size());
        FML::MaxArrayOverTasks(upper.data(), upper.size());
        require(lower == upper, "[FourPoint] MPI options differ");
#endif
    }
    namespace detail {
        // Semidefinite Cholesky, with a roundoff tolerance scaled to the input.
        inline bool valid_covariance(const std::array<double, 16> & c) {
            long double scale = 0, l[4][4]{};
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    if (!std::isfinite(c[4 * i + j]) || c[4 * i + j] != c[4 * j + i])
                        return false;
                    if (i == j && c[4 * i + j] < 0)
                        return false;
                    scale = std::max(scale, std::abs((long double)c[4 * i + j]));
                }
            const long double tol = 64 * std::numeric_limits<double>::epsilon() * scale;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j <= i; ++j) {
                    long double v = c[4 * i + j];
                    for (int k = 0; k < j; ++k)
                        v -= l[i][k] * l[j][k];
                    if (i == j) {
                        if (v < -tol)
                            return false;
                        l[i][j] = std::sqrt(std::max(0.L, v));
                    } else if (l[j][j] > 0)
                        l[i][j] = v / l[j][j];
                    else if (std::abs(v) > tol)
                        return false;
                }
            return true;
        }
    } // namespace detail
} // namespace FML::FOURPOINT
#endif
