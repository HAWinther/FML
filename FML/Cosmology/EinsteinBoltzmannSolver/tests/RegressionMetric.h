#ifndef FML_TEST_REGRESSION_METRIC_H
#define FML_TEST_REGRESSION_METRIC_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
namespace Test {
    /// TE is signed and crosses zero. Measure its change against the natural
    /// covariance scale sqrt(TT*EE), equivalent to an absolute correlation change.
    /// All other observables keep their ordinary relative-error metric.
    inline double
    regression_error(const std::string & key, double value, const std::map<std::string, double> & reference) {
        const double expected = reference.at(key);
        if (!std::isfinite(value) || !std::isfinite(expected))
            return std::numeric_limits<double>::infinity();
        double scale = std::abs(expected);
        const auto te = key.find(".TE(");
        if (te != std::string::npos) {
            auto tt = key, ee = key;
            tt.replace(te, 4, ".TT(");
            ee.replace(te, 4, ".EE(");
            const double t = reference.at(tt), e = reference.at(ee);
            if (!std::isfinite(t) || !std::isfinite(e) || t < 0 || e < 0)
                return std::numeric_limits<double>::infinity();
            scale = std::max(scale, std::sqrt(t) * std::sqrt(e));
        }
        return std::abs(value - expected) / std::max(scale, 1e-300);
    }
} // namespace Test
#endif
